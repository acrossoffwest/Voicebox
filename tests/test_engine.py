from pathlib import Path

import numpy as np

from audio_io import RingBuffer
from engine import Engine, EngineConfig, drop_stale_input


def test_engine_config_defaults():
    cfg = EngineConfig()
    assert cfg.sample_rate == 48000
    assert cfg.blocksize == 480
    assert cfg.bypass is False
    assert cfg.rvc_model_dir is None
    assert cfg.rvc_base_dir.name == "base"
    assert cfg.rvc_base_dir.parent.name == "models"
    assert cfg.window_ms == 1184
    assert cfg.crossfade_ms == 24
    assert cfg.context_ms == 1024
    assert cfg.pitch_method == "pm"
    assert cfg.index_rate == 0.0


def test_engine_prepare_bypass_does_not_load_models():
    eng = Engine(EngineConfig(bypass=True))
    eng.prepare()
    assert not eng.is_running()
    s = eng.stats()
    assert s["running"] is False
    assert s["in_fill"] == 0.0
    assert s["out_fill"] == 0.0
    assert s["underruns"] == 0
    assert s["overruns"] == 0


def test_engine_prepare_idempotent():
    eng = Engine(EngineConfig(bypass=True))
    eng.prepare()
    eng.prepare()  # second call must not raise


def test_engine_double_stop_is_safe():
    eng = Engine(EngineConfig(bypass=True))
    eng.prepare()
    eng.stop()
    eng.stop()


def test_engine_stats_before_prepare():
    eng = Engine(EngineConfig(bypass=True))
    s = eng.stats()
    assert s["running"] is False
    assert s["total_ms"] == 0.0


def test_drop_stale_input_discards_oldest_keeps_freshest():
    ring = RingBuffer(capacity=48000)
    ring.write(np.arange(1000, dtype=np.float32))
    dropped = drop_stale_input(ring, keep=300)
    assert dropped == 700
    assert ring.available() == 300
    out = ring.read(300)
    np.testing.assert_array_equal(out, np.arange(700, 1000, dtype=np.float32))


def test_drop_stale_input_noop_when_backlog_small():
    ring = RingBuffer(capacity=48000)
    ring.write(np.arange(200, dtype=np.float32))
    assert drop_stale_input(ring, keep=300) == 0
    assert ring.available() == 200


def test_clamp_backlog_triggers_past_hop_plus_margin():
    from engine import clamp_backlog

    ring = RingBuffer(capacity=48000)
    ring.write(np.zeros(500, dtype=np.float32))
    assert clamp_backlog(ring, hop_n=400, margin=100) == 0
    assert ring.available() == 500
    ring.write(np.zeros(1, dtype=np.float32))
    assert clamp_backlog(ring, hop_n=400, margin=100) == 101
    assert ring.available() == 400
