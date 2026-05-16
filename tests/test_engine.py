from pathlib import Path

from engine import Engine, EngineConfig


def test_engine_config_defaults():
    cfg = EngineConfig()
    assert cfg.sample_rate == 48000
    assert cfg.blocksize == 480
    assert cfg.bypass is False
    assert cfg.rvc_model_dir is None
    assert cfg.rvc_base_dir.name == "base"
    assert cfg.rvc_base_dir.parent.name == "models"
    assert cfg.window_ms == 384
    assert cfg.crossfade_ms == 128


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
