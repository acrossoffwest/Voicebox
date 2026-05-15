import numpy as np
import pytest

from audio_io import RingBuffer


def test_empty_buffer_has_zero_available():
    rb = RingBuffer(capacity=10)
    assert rb.available() == 0
    assert rb.fill_ratio() == 0.0


def test_write_then_read_round_trips():
    rb = RingBuffer(capacity=10)
    written = rb.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert written == 3
    assert rb.available() == 3
    out = rb.read(3)
    assert out is not None
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0], dtype=np.float32))
    assert rb.available() == 0


def test_read_returns_none_when_not_enough_samples():
    rb = RingBuffer(capacity=10)
    rb.write(np.array([1.0, 2.0], dtype=np.float32))
    assert rb.read(3) is None


def test_write_overflow_is_dropped_and_counted():
    rb = RingBuffer(capacity=4)
    rb.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    written = rb.write(np.array([4.0, 5.0, 6.0], dtype=np.float32))
    assert written == 1
    assert rb.overruns == 2
    out = rb.read(4)
    np.testing.assert_array_equal(out, np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32))


def test_wraparound():
    rb = RingBuffer(capacity=5)
    rb.write(np.array([1.0, 2.0, 3.0], dtype=np.float32))
    rb.read(3)
    rb.write(np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))
    out = rb.read(4)
    np.testing.assert_array_equal(out, np.array([4.0, 5.0, 6.0, 7.0], dtype=np.float32))


def test_fill_ratio():
    rb = RingBuffer(capacity=10)
    rb.write(np.zeros(5, dtype=np.float32))
    assert rb.fill_ratio() == pytest.approx(0.5)
