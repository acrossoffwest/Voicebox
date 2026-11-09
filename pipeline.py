"""Windowing + DeepFilterNet + RVC + crossfade.

A window is 256 ms by default; the hop is `window - crossfade` (192 ms by
default). Each call to `process()` consumes one window and returns one hop
worth of fresh, crossfaded audio at 48 kHz mono float32."""

from __future__ import annotations

import time
from collections import deque
from math import gcd
from typing import Callable, Optional

import numpy as np


def _resample(x: np.ndarray, sr_in: int, sr_out: int) -> np.ndarray:
    if sr_in == sr_out:
        return x.astype(np.float32, copy=False)
    from scipy.signal import resample_poly

    g = gcd(sr_in, sr_out)
    up = sr_out // g
    down = sr_in // g
    return resample_poly(x, up, down).astype(np.float32, copy=False)


class Pipeline:
    def __init__(
        self,
        denoiser,
        rvc,
        sr: int = 48000,
        window_ms: int = 256,
        crossfade_ms: int = 64,
        context_ms: int | None = None,
        denoise: bool = True,
        bypass: bool = False,
        sola: bool | None = None,
    ):
        self.denoiser = denoiser
        self.rvc = rvc
        self.sr = sr
        self.window_ms = window_ms
        self.crossfade_ms = crossfade_ms
        self.context_ms = crossfade_ms if context_ms is None else context_ms
        self.window_samples = sr * window_ms // 1000
        self.crossfade_samples = sr * crossfade_ms // 1000
        self.context_samples = sr * self.context_ms // 1000
        if self.context_samples < self.crossfade_samples:
            raise ValueError("context_ms must be >= crossfade_ms")
        self.hop_samples = self.window_samples - self.context_samples
        if self.hop_samples <= 0:
            raise ValueError("window_ms must exceed context_ms")
        if self.hop_samples < self.crossfade_samples:
            raise ValueError("hop (window_ms - context_ms) must be >= crossfade_ms")
        # RVC output is head-aligned and comes up short at the tail (whole
        # hubert frames only), and the vocoder re-synthesizes each window
        # with a different phase. SOLA searches +-search around the nominal
        # splice point for the best waveform match; guard keeps every slice
        # clear of the zero-padded tail (shortfall ~24 ms) and inside the
        # model output.
        self.sola = (rvc is not None and not bypass) if sola is None else sola
        self.guard_samples = sr * 32 // 1000 if self.sola else 0
        self.search_samples = sr * 8 // 1000 if self.sola else 0
        if self.sola and self.context_samples < (
            self.crossfade_samples + self.guard_samples + self.search_samples
        ):
            raise ValueError("context_ms too small for SOLA guard + search")
        self.denoise = denoise
        self.bypass = bypass
        self._prev_tail: np.ndarray | None = None
        self.last_sola_offset: int = 0

        self.t_denoise: deque[float] = deque(maxlen=512)
        self.t_rvc: deque[float] = deque(maxlen=512)
        self.t_total: deque[float] = deque(maxlen=512)

        self._chain: Optional[Callable[[np.ndarray], tuple[np.ndarray, int]]] = None

    def process(self, window: np.ndarray) -> np.ndarray:
        if window.shape != (self.window_samples,):
            raise ValueError(
                f"window must have shape ({self.window_samples},), got {window.shape}"
            )

        t0 = time.perf_counter()

        if self.bypass:
            out = window
        elif self._chain is not None:
            out, _ = self._chain(window)
        else:
            out = self._run_chain(window)

        cf = self.crossfade_samples
        ctx = self.context_samples
        hop = self.hop_samples
        # Emission lags capture by cf (+ SOLA guard when active): the tail is
        # kept back as the next iteration's crossfade reference, so
        # consecutive emissions blend over audio both windows processed.
        base = ctx - self.guard_samples
        k = 0
        corr = 0.0
        tail_energy = (
            float(np.dot(self._prev_tail, self._prev_tail))
            if self._prev_tail is not None
            else 0.0
        )
        # A silent reference carries no alignment evidence; argmax over
        # all-zero correlation would pin k to -search.
        if self.sola and cf > 0 and tail_energy > 1e-6:
            s_n = self.search_samples
            region = out[base - cf - s_n : base + s_n]
            num = np.correlate(region, self._prev_tail, mode="valid")
            energy = np.convolve(
                region * region, np.ones(cf, dtype=np.float32), mode="valid"
            )
            scores = num / np.sqrt(energy + 1e-9)
            j = int(np.argmax(scores))
            k = j - s_n
            corr = float(
                np.clip(scores[j] / np.sqrt(tail_energy + 1e-9), 0.0, 1.0)
            )
        self.last_sola_offset = k
        start = base - cf + k
        if self._prev_tail is None or cf == 0:
            fresh = out[start : start + hop].copy()
        else:
            # Correlated segments want a linear (sum-to-one) fade — it is
            # transparent, while equal-power would bump +3 dB mid-fade.
            # Uncorrelated segments (unvoiced sounds, weak SOLA match) want
            # equal-power (sqrt cosine), or the fade dips -6 dB mid-way.
            # Blend the two by the measured correlation at the splice.
            t = np.linspace(0.0, 1.0, cf, endpoint=False, dtype=np.float32)
            ep_out = np.cos(t * 0.5 * np.pi).astype(np.float32)
            ep_in = np.sin(t * 0.5 * np.pi).astype(np.float32)
            if self.sola:
                fade_in = corr * t + (1.0 - corr) * ep_in
                fade_out = corr * (1.0 - t) + (1.0 - corr) * ep_out
            else:
                fade_out = ep_out
                fade_in = ep_in
            mixed = fade_out * self._prev_tail + fade_in * out[start : start + cf]
            fresh = np.empty(hop, dtype=np.float32)
            fresh[:cf] = mixed
            fresh[cf:] = out[start + cf : start + hop]

        self._prev_tail = (
            out[start + hop : start + hop + cf].copy() if cf > 0 else None
        )

        self.t_total.append(time.perf_counter() - t0)
        return fresh

    def _run_chain(self, window: np.ndarray) -> np.ndarray:
        x = window
        if self.denoise and self.denoiser is not None:
            t = time.perf_counter()
            x = self.denoiser.enhance(x)
            self.t_denoise.append(time.perf_counter() - t)

        if self.rvc is None:
            return x

        x16 = _resample(x, self.sr, 16000)
        t = time.perf_counter()
        y, sr_out = self.rvc.infer(x16)
        self.t_rvc.append(time.perf_counter() - t)
        y48 = _resample(y, sr_out, self.sr)

        # RVC consumes whole hubert frames and loses the remainder at the
        # tail; its output is head-aligned with the input window. The SOLA
        # guard keeps emission clear of this zero tail.
        if y48.size >= self.window_samples:
            return y48[: self.window_samples]
        pad = np.zeros(self.window_samples - y48.size, dtype=np.float32)
        return np.concatenate([y48, pad])

    def last_timings_ms(self) -> dict[str, float]:
        def _last_ms(buf):
            return (buf[-1] * 1000.0) if buf else 0.0

        return {
            "denoise_ms": _last_ms(self.t_denoise),
            "rvc_ms": _last_ms(self.t_rvc),
            "total_ms": _last_ms(self.t_total),
        }
