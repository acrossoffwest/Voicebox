from __future__ import annotations

import argparse
import json
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path
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
    context_ms: int | None = None,
    spike: tuple[float, float] | None = None,
    sr: int = 48000,
    blocksize: int = 480,
    warmup_s: float = 2.0,
) -> ScenarioResult:
    engine = Engine(EngineConfig(sample_rate=sr, blocksize=blocksize,
                                 window_ms=window_ms, crossfade_ms=crossfade_ms,
                                 context_ms=context_ms))
    pipeline = Pipeline(denoiser=None, rvc=None, sr=sr, window_ms=window_ms,
                        crossfade_ms=crossfade_ms, context_ms=context_ms,
                        denoise=False, bypass=False)
    in_ring = RingBuffer(48000)
    out_ring = RingBuffer(96000)
    out_ring.write(np.zeros(pipeline.hop_samples + 2 * blocksize, dtype=np.float32))
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


def _build_matrix() -> list[dict]:
    matrix = []
    for window_ms, crossfade_ms, context_ms in (
        (384, 128, None),
        (256, 64, None),
        (256, 24, 192),
        (240, 16, 192),
    ):
        hop_ms = window_ms - (context_ms if context_ms is not None else crossfade_ms)
        label = (f"{window_ms}/{crossfade_ms}" if context_ms is None
                 else f"ctx {window_ms}/{context_ms}/{crossfade_ms}")
        for kind, factor, spike in (
            ("steady_fast", 0.6, None),
            ("steady_slow", 0.9, None),
            ("spike", 0.6, (4.0, 800.0)),
        ):
            matrix.append({
                "name": f"{label} {kind}",
                "window_ms": window_ms,
                "crossfade_ms": crossfade_ms,
                "context_ms": context_ms,
                "infer_ms": hop_ms * factor,
                "spike": spike,
                "duration_s": 12.0,
                "warmup_s": 2.0,
            })
    return matrix


DEFAULT_MATRIX = _build_matrix()


def results_to_dict(results: dict[str, ScenarioResult]) -> dict:
    try:
        git_rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True,
            cwd=Path(__file__).resolve().parent,
        ).stdout.strip() or "unknown"
    except (OSError, subprocess.CalledProcessError):
        git_rev = "unknown"
    return {
        "git_rev": git_rev,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "scenarios": {name: asdict(r) for name, r in results.items()},
    }


def load_results(path) -> dict:
    return json.loads(Path(path).read_text())


def format_table(results: dict[str, ScenarioResult]) -> str:
    width = max(len(n) for n in results)
    lines = [f"{'scenario':<{width}}  {'p50':>8} {'p95':>8} {'max':>8} {'final':>8} {'underruns':>10}"]
    for name, r in results.items():
        lines.append(
            f"{name:<{width}}  {r.p50_ms:8.1f} {r.p95_ms:8.1f} {r.max_ms:8.1f} "
            f"{r.final_ms:8.1f} {r.underrun_samples:10d}"
        )
    return "\n".join(lines)


def format_delta(results: dict[str, ScenarioResult], baseline: dict) -> str:
    base = baseline["scenarios"]
    width = max(len(n) for n in results)
    lines = [
        f"delta vs {baseline.get('git_rev', 'unknown')} (+ worse / - better, ms)",
        f"{'scenario':<{width}}  {'p50':>8} {'p95':>8} {'final':>8}",
    ]
    for name, r in results.items():
        b = base.get(name)
        if b is None:
            lines.append(f"{name:<{width}}  {'n/a':>8} {'n/a':>8} {'n/a':>8}")
            continue
        lines.append(
            f"{name:<{width}}  {r.p50_ms - b['p50_ms']:+8.1f} "
            f"{r.p95_ms - b['p95_ms']:+8.1f} {r.final_ms - b['final_ms']:+8.1f}"
        )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--save")
    parser.add_argument("--baseline")
    args = parser.parse_args()

    results = {}
    for sc in DEFAULT_MATRIX:
        params = {k: v for k, v in sc.items() if k != "name"}
        if args.quick:
            params["duration_s"] = 5.0
            params["warmup_s"] = 1.0
        results[sc["name"]] = run_scenario(**params)

    print(format_table(results))
    if args.baseline:
        print()
        print(format_delta(results, load_results(args.baseline)))
    if args.save:
        Path(args.save).write_text(json.dumps(results_to_dict(results), indent=2))


if __name__ == "__main__":
    main()
