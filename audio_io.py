"""Lock-protected single-producer / single-consumer ring buffer plus the
sounddevice I/O wrapper. The audio callbacks must never block on the
processing thread, so all coordination happens through these buffers."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np


class RingBuffer:
    """Single-producer / single-consumer ring buffer of float32 samples."""

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = int(capacity)
        self._buf = np.zeros(self.capacity, dtype=np.float32)
        self._head = 0   # next write index
        self._tail = 0   # next read index
        self._size = 0   # samples currently stored
        self._lock = threading.Lock()
        self.overruns = 0
        self.underruns = 0

    def available(self) -> int:
        with self._lock:
            return self._size

    def fill_ratio(self) -> float:
        with self._lock:
            return self._size / self.capacity

    def write(self, samples: np.ndarray) -> int:
        if samples.dtype != np.float32:
            samples = samples.astype(np.float32, copy=False)
        n = int(samples.size)
        with self._lock:
            free = self.capacity - self._size
            to_write = min(n, free)
            if to_write < n:
                self.overruns += n - to_write
            if to_write == 0:
                return 0
            end = self._head + to_write
            if end <= self.capacity:
                self._buf[self._head:end] = samples[:to_write]
            else:
                first = self.capacity - self._head
                self._buf[self._head:] = samples[:first]
                self._buf[:to_write - first] = samples[first:to_write]
            self._head = (self._head + to_write) % self.capacity
            self._size += to_write
            return to_write

    def read(self, n: int) -> np.ndarray | None:
        with self._lock:
            if self._size < n:
                return None
            out = np.empty(n, dtype=np.float32)
            end = self._tail + n
            if end <= self.capacity:
                out[:] = self._buf[self._tail:end]
            else:
                first = self.capacity - self._tail
                out[:first] = self._buf[self._tail:]
                out[first:] = self._buf[:n - first]
            self._tail = (self._tail + n) % self.capacity
            self._size -= n
            return out


@dataclass
class IOStats:
    in_fill: float = 0.0
    out_fill: float = 0.0
    overruns: int = 0
    underruns: int = 0


class AudioIO:
    """Owns the sounddevice InputStream + OutputStream and their ring buffers."""

    def __init__(
        self,
        input_device: int | None,
        output_device: int | None,
        sample_rate: int = 48000,
        blocksize: int = 480,
        in_capacity: int = 48000,
        out_capacity: int = 96000,
    ):
        self.sample_rate = sample_rate
        self.blocksize = blocksize
        self.input_ring = RingBuffer(in_capacity)
        self.output_ring = RingBuffer(out_capacity)
        self._input_device = input_device
        self._output_device = output_device
        self._in_stream = None
        self._out_stream = None

    def _on_input(self, indata, frames, time_info, status):  # noqa: ARG002
        mono = indata[:, 0] if indata.ndim == 2 else indata
        self.input_ring.write(mono)

    def _on_output(self, outdata, frames, time_info, status):  # noqa: ARG002
        chunk = self.output_ring.read(frames)
        if chunk is None:
            self.output_ring.underruns += frames
            outdata.fill(0.0)
            return
        if outdata.ndim == 2:
            outdata[:, 0] = chunk
            if outdata.shape[1] > 1:
                outdata[:, 1] = chunk
        else:
            outdata[:] = chunk

    def start(self) -> None:
        import sounddevice as sd

        self._in_stream = sd.InputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            device=self._input_device,
            channels=1,
            dtype="float32",
            callback=self._on_input,
        )
        self._out_stream = sd.OutputStream(
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
            device=self._output_device,
            channels=2,
            dtype="float32",
            callback=self._on_output,
        )
        self._in_stream.start()
        self._out_stream.start()

    def stop(self) -> None:
        for s in (self._in_stream, self._out_stream):
            if s is not None:
                try:
                    s.stop()
                    s.close()
                except Exception:
                    pass
        self._in_stream = None
        self._out_stream = None

    def stats(self) -> IOStats:
        return IOStats(
            in_fill=self.input_ring.fill_ratio(),
            out_fill=self.output_ring.fill_ratio(),
            overruns=self.input_ring.overruns,
            underruns=self.output_ring.underruns,
        )

    @staticmethod
    def list_devices() -> str:
        import sounddevice as sd

        return str(sd.query_devices())
