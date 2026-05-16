"""Windowing + DeepFilterNet + RVC + crossfade.

A window is 256 ms by default; the hop is `window - crossfade` (192 ms by
default). Each call to `process()` consumes one window and returns one hop
worth of fresh, crossfaded audio at 48 kHz mono float32."""

from __future__ import annotations

import time
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
        denoise: bool = True,
        bypass: bool = False,
    ):
        self.denoiser = denoiser
        self.rvc = rvc
        self.sr = sr
        self.window_ms = window_ms
        self.crossfade_ms = crossfade_ms
        self.window_samples = sr * window_ms // 1000
        self.crossfade_samples = sr * crossfade_ms // 1000
        self.hop_samples = self.window_samples - self.crossfade_samples
        self.denoise = denoise
        self.bypass = bypass
        self._prev_tail: np.ndarray | None = None

        self.t_denoise: list[float] = []
        self.t_rvc: list[float] = []
        self.t_total: list[float] = []

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
        if self._prev_tail is None or cf == 0:
            fresh = out[: self.hop_samples].copy()
        else:
            # Equal-power crossfade (sqrt cosine) keeps energy constant
            # through the overlap. Linear blending creates an amplitude dip
            # mid-crossfade which sounds "robotic" on continuous speech.
            t = np.linspace(0.0, 1.0, cf, endpoint=False, dtype=np.float32)
            fade_out = np.cos(t * 0.5 * np.pi).astype(np.float32)
            fade_in = np.sin(t * 0.5 * np.pi).astype(np.float32)
            mixed = fade_out * self._prev_tail + fade_in * out[:cf]
            fresh = np.empty(self.hop_samples, dtype=np.float32)
            fresh[:cf] = mixed
            fresh[cf:] = out[cf : self.hop_samples]

        self._prev_tail = out[-cf:].copy() if cf > 0 else None

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
