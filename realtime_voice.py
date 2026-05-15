"""Real-time mic -> DeepFilterNet -> RVC -> BlackHole pipeline.

This is the entry point. The processor thread polls the input ring buffer
for full windows, calls `Pipeline.process`, and pushes hops into the output
ring buffer. Audio callbacks never block on inference."""

from __future__ import annotations

import argparse
import platform
import signal
import sys
import threading
import time
from pathlib import Path

from audio_io import AudioIO
from pipeline import Pipeline


def _select_device(arg: str) -> str:
    if arg == "cpu":
        return "cpu"
    if arg == "mps":
        return "mps"
    try:
        import torch

        if platform.machine() == "arm64" and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def _resolve_output_device(sd_module, requested: int | None) -> int | None:
    if requested is not None:
        return requested
    for idx, info in enumerate(sd_module.query_devices()):
        if "BlackHole" in info["name"] and info["max_output_channels"] > 0:
            print(f"[info] auto-selected BlackHole at device index {idx}: {info['name']}")
            return idx
    print(
        "[warn] BlackHole 2ch not found in CoreAudio devices. "
        "Specify --output-device or install via `brew install blackhole-2ch`.",
        file=sys.stderr,
    )
    return None


def _format_stats(io_stats, pipe_timings, runtime_s) -> str:
    return (
        f"t={runtime_s:5.1f}s "
        f"in={io_stats.in_fill * 100:5.1f}% "
        f"out={io_stats.out_fill * 100:5.1f}% "
        f"dfn={pipe_timings['denoise_ms']:5.1f}ms "
        f"rvc={pipe_timings['rvc_ms']:5.1f}ms "
        f"total={pipe_timings['total_ms']:5.1f}ms "
        f"under={io_stats.underruns} over={io_stats.overruns}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Real-time voice pipeline (mic -> DFN -> RVC -> BlackHole)."
    )
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--rvc-model", type=str, default=None, help="Subdirectory under ./models/rvc/")
    parser.add_argument("--pitch-shift", type=int, default=0, help="Semitones; positive raises pitch.")
    parser.add_argument("--denoise", choices=["on", "off"], default="on")
    parser.add_argument("--bypass", action="store_true", help="Skip denoise + RVC entirely.")
    parser.add_argument("--list-devices", action="store_true")
    parser.add_argument("--device", choices=["auto", "mps", "cpu"], default="auto")
    parser.add_argument("--window-ms", type=int, default=256)
    parser.add_argument("--crossfade-ms", type=int, default=64)
    parser.add_argument("--blocksize", type=int, default=480)
    args = parser.parse_args(argv)

    import sounddevice as sd

    if args.list_devices:
        print(sd.query_devices())
        return 0

    device = _select_device(args.device)
    print(f"[info] torch device: {device}")

    denoiser = None
    rvc = None

    if not args.bypass:
        if args.denoise == "on":
            from denoise import Denoiser

            print("[info] loading DeepFilterNet...")
            denoiser = Denoiser()

        if args.rvc_model:
            from rvc import RVC

            model_dir = Path("models/rvc") / args.rvc_model
            print(f"[info] loading RVC model from {model_dir}")
            rvc = RVC(
                model_dir=model_dir,
                base_dir=Path("models/base"),
                device=device,
                pitch_shift_semitones=args.pitch_shift,
            )
        else:
            print("[warn] --rvc-model not set; running denoise-only (no voice change).")

    pipeline = Pipeline(
        denoiser=denoiser,
        rvc=rvc,
        sr=48000,
        window_ms=args.window_ms,
        crossfade_ms=args.crossfade_ms,
        denoise=(args.denoise == "on"),
        bypass=args.bypass,
    )

    output_idx = _resolve_output_device(sd, args.output_device)
    audio_io = AudioIO(
        input_device=args.input_device,
        output_device=output_idx,
        sample_rate=48000,
        blocksize=args.blocksize,
    )

    stop_event = threading.Event()

    def _processor():
        while not stop_event.is_set():
            if audio_io.input_ring.available() < pipeline.window_samples:
                time.sleep(0.001)
                continue
            window = audio_io.input_ring.read(pipeline.window_samples)
            if window is None:
                continue
            hop = pipeline.process(window)
            audio_io.output_ring.write(hop)

    def _shutdown(_signum, _frame):
        stop_event.set()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    print("[info] starting streams. Ctrl+C to stop.")
    audio_io.start()
    worker = threading.Thread(target=_processor, name="processor", daemon=False)
    worker.start()

    started_at = time.perf_counter()
    next_report = started_at + 5.0
    try:
        while not stop_event.is_set():
            time.sleep(0.1)
            now = time.perf_counter()
            if now >= next_report:
                print(_format_stats(audio_io.stats(), pipeline.last_timings_ms(), now - started_at))
                next_report = now + 5.0
    finally:
        stop_event.set()
        worker.join(timeout=1.0)
        audio_io.stop()
        print(_format_stats(audio_io.stats(), pipeline.last_timings_ms(), time.perf_counter() - started_at))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
