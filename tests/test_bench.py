from bench import run_scenario


def test_steady_scenario_reports_sane_latency():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                     duration_s=3.0, warmup_s=1.0)
    assert r.measured_samples > 48000
    assert r.underrun_samples == 0
    assert 48 <= r.p50_ms <= 400
    assert r.p50_ms <= r.p95_ms <= r.max_ms


def test_overloaded_scenario_counts_underruns_without_crash():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=120,
                     duration_s=3.0, warmup_s=1.0)
    assert r.underrun_samples > 0
