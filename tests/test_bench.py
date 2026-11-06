import json

from bench import run_scenario


def test_steady_scenario_reports_sane_latency():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                     duration_s=3.0, warmup_s=1.0)
    assert r.measured_samples > 48000
    assert r.underrun_samples <= 960
    assert 48 <= r.p50_ms <= 400
    assert r.p50_ms <= r.p95_ms <= r.max_ms


def test_overloaded_scenario_counts_underruns_without_crash():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=120,
                     duration_s=3.0, warmup_s=1.0)
    assert r.underrun_samples > 0


def test_results_round_trip_and_delta(tmp_path):
    from bench import format_delta, load_results, results_to_dict
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                     duration_s=2.0, warmup_s=0.5)
    d = results_to_dict({"steady": r})
    p = tmp_path / "base.json"
    p.write_text(json.dumps(d))
    base = load_results(p)
    out = format_delta({"steady": r}, base)
    assert "steady" in out
    assert "+0.0" in out


def test_latency_recovers_after_stall():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                     duration_s=6.0, warmup_s=1.0,
                     spike=(2.0, 600.0))
    steady = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                          duration_s=3.0, warmup_s=1.0)
    assert r.final_ms <= steady.p95_ms + 150


def test_latency_never_exceeds_clamp_ceiling():
    r = run_scenario(window_ms=64, crossfade_ms=16, infer_ms=20,
                     duration_s=6.0, warmup_s=1.0,
                     spike=(2.0, 600.0))
    assert r.max_ms <= 1200
