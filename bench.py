from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from audio_io import RingBuffer
from engine import Engine, EngineConfig
from pipeline import Pipeline

BASE = 1_000_000.0


@dataclass
class ScenarioResult:
    p50_ms: float
    p95_ms: float
    max_ms: float
    final_ms: float
    underrun_samples: int
    measured_samples: int


def run_scenario(
    window_ms: int,
    crossfade_ms: int,
    infer_ms: float,
    duration_s: float,
    spike: tuple[float, float] | None = None,
    sr: int = 48000,
    blocksize: int = 480,
    warmup_s: float = 2.0,
) -> ScenarioResult:
    engine = Engine(EngineConfig(sample_rate=sr, blocksize=blocksize,
                                 window_ms=window_ms, crossfade_ms=crossfade_ms))
    pipeline = Pipeline(denoiser=None, rvc=None, sr=sr, window_ms=window_ms,
                        crossfade_ms=crossfade_ms, denoise=False, bypass=False)
    in_ring = RingBuffer(48000)
    out_ring = RingBuffer(96000)
    out_ring.write(np.zeros(pipeline.window_samples, dtype=np.float32))
    engine._pipeline = pipeline
    engine._audio_io = SimpleNamespace(input_ring=in_ring, output_ring=out_ring)

    t0 = time.perf_counter()
    spike_pending = [spike is not None]

    def fake_chain(x):
        delay_ms = infer_ms
        if spike_pending[0] and time.perf_counter() - t0 >= spike[0]:
            spike_pending[0] = False
            delay_ms = spike[1]
        time.sleep(delay_ms / 1000)
        return x, sr

    pipeline._chain = fake_chain

    stop = engine._stop_event
    samples: list[tuple[float, float]] = []
    underruns = 0
    measured = 0

    def feeder():
        k = 0
        while not stop.is_set():
            time.sleep(max(0.0, t0 + k * blocksize / sr - time.perf_counter()))
            start = BASE + k * blocksize
            in_ring.write(np.arange(start, start + blocksize, dtype=np.float32))
            k += 1

    def consumer():
        nonlocal underruns, measured
        k = 0
        while not stop.is_set():
            time.sleep(max(0.0, t0 + k * blocksize / sr - time.perf_counter()))
            k += 1
            now = time.perf_counter()
            chunk = out_ring.read(blocksize)
            rel = now - t0
            if chunk is None:
                if rel >= warmup_s:
                    underruns += blocksize
                continue
            if rel < warmup_s:
                continue
            measured += blocksize
            if chunk[0] >= BASE and chunk[-1] - chunk[0] == blocksize - 1:
                lat_ms = (now - (t0 + (float(chunk[0]) - BASE) / sr)) * 1000
                samples.append((rel, lat_ms))

    threads = [
        threading.Thread(target=engine._processor_loop),
        threading.Thread(target=feeder),
        threading.Thread(target=consumer),
    ]
    for t in threads:
        t.start()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join()

    lats = np.array([lat for _, lat in samples]) if samples else np.zeros(1)
    last = np.array([lat for rel, lat in samples if rel >= duration_s - 1.0])
    final = float(np.median(last)) if last.size else float(lats[-1])
    return ScenarioResult(
        p50_ms=float(np.percentile(lats, 50)),
        p95_ms=float(np.percentile(lats, 95)),
        max_ms=float(lats.max()),
        final_ms=final,
        underrun_samples=underruns,
        measured_samples=measured,
    )
