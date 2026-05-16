"""Headless real-time voice engine. Both the CLI and the PyQt6 UI use this.

Wraps AudioIO + Denoiser + RVC + Pipeline + a processor thread behind a
thin lifecycle: `prepare()`, `start()`, `stop()`, `stats()`. Loads heavy
models lazily in `prepare()` so the UI can show a loading state."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from audio_io import AudioIO
from pipeline import Pipeline

log = logging.getLogger("engine")


@dataclass
class EngineConfig:
    input_device: int | None = None
    output_device: int | None = None
    sample_rate: int = 48000
    blocksize: int = 480
    device: str = "cpu"
    denoise: bool = True
    bypass: bool = False
    rvc_model_dir: Optional[Path] = None
    rvc_base_dir: Path = field(default_factory=lambda: Path("models/base"))
    pitch_shift: int = 0
    window_ms: int = 256
    crossfade_ms: int = 64


class Engine:
    def __init__(self, config: EngineConfig):
        self.config = config
        self._denoiser = None
        self._rvc = None
        self._pipeline: Pipeline | None = None
        self._audio_io: AudioIO | None = None
        self._worker: threading.Thread | None = None
        self._stop_event = threading.Event()
        self._prepared = False
        self._lock = threading.Lock()

    def prepare(self) -> None:
        """Load DeepFilterNet and RVC according to config. Idempotent."""
        if self._prepared:
            return
        cfg = self.config

        if not cfg.bypass:
            if cfg.denoise:
                from denoise import Denoiser

                log.info("loading DeepFilterNet")
                self._denoiser = Denoiser()

            if cfg.rvc_model_dir is not None:
                from rvc import RVC

                log.info("loading RVC model from %s", cfg.rvc_model_dir)
                self._rvc = RVC(
                    model_dir=cfg.rvc_model_dir,
                    base_dir=cfg.rvc_base_dir,
                    device=cfg.device,
                    pitch_shift_semitones=cfg.pitch_shift,
                )

        self._pipeline = Pipeline(
            denoiser=self._denoiser,
            rvc=self._rvc,
            sr=cfg.sample_rate,
            window_ms=cfg.window_ms,
            crossfade_ms=cfg.crossfade_ms,
            denoise=cfg.denoise,
            bypass=cfg.bypass,
        )
        self._prepared = True

    def is_running(self) -> bool:
        return self._worker is not None and self._worker.is_alive()

    def start(self) -> None:
        with self._lock:
            if self.is_running():
                raise RuntimeError("Engine is already running")
            if not self._prepared:
                self.prepare()
            assert self._pipeline is not None
            cfg = self.config
            self._audio_io = AudioIO(
                input_device=cfg.input_device,
                output_device=cfg.output_device,
                sample_rate=cfg.sample_rate,
                blocksize=cfg.blocksize,
            )
            self._stop_event.clear()
            self._audio_io.start()
            self._worker = threading.Thread(
                target=self._processor_loop,
                name="engine-processor",
                daemon=False,
            )
            self._worker.start()

    def stop(self) -> None:
        with self._lock:
            self._stop_event.set()
            if self._worker is not None:
                self._worker.join(timeout=1.0)
                self._worker = None
            if self._audio_io is not None:
                self._audio_io.stop()
                self._audio_io = None

    def stats(self) -> dict:
        timings = (
            self._pipeline.last_timings_ms()
            if self._pipeline is not None
            else {"denoise_ms": 0.0, "rvc_ms": 0.0, "total_ms": 0.0}
        )
        if self._audio_io is None:
            return {
                "in_fill": 0.0,
                "out_fill": 0.0,
                "underruns": 0,
                "overruns": 0,
                "denoise_ms": timings["denoise_ms"],
                "rvc_ms": timings["rvc_ms"],
                "total_ms": timings["total_ms"],
                "running": self.is_running(),
            }
        io_stats = self._audio_io.stats()
        return {
            "in_fill": io_stats.in_fill,
            "out_fill": io_stats.out_fill,
            "underruns": io_stats.underruns,
            "overruns": io_stats.overruns,
            "denoise_ms": timings["denoise_ms"],
            "rvc_ms": timings["rvc_ms"],
            "total_ms": timings["total_ms"],
            "running": self.is_running(),
        }

    def _processor_loop(self) -> None:
        assert self._audio_io is not None and self._pipeline is not None
        ring = self._audio_io.input_ring
        out_ring = self._audio_io.output_ring
        window_n = self._pipeline.window_samples
        process = self._pipeline.process
        stop_event = self._stop_event

        while not stop_event.is_set():
            if ring.available() < window_n:
                time.sleep(0.001)
                continue
            window = ring.read(window_n)
            if window is None:
                continue
            try:
                hop = process(window)
            except Exception:
                log.exception("processor crashed on window")
                stop_event.set()
                break
            out_ring.write(hop)
