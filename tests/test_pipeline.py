import numpy as np

from pipeline import Pipeline


def _make_bypass_pipeline(window_ms=256, crossfade_ms=64):
    return Pipeline(
        denoiser=None,
        rvc=None,
        sr=48000,
        window_ms=window_ms,
        crossfade_ms=crossfade_ms,
        denoise=False,
        bypass=True,
    )


def test_bypass_emits_hop_samples():
    p = _make_bypass_pipeline()
    window = np.arange(p.window_samples, dtype=np.float32)
    out = p.process(window)
    assert out.shape == (p.hop_samples,)
    np.testing.assert_array_equal(out, window[: p.hop_samples])


def test_window_and_hop_math():
    p = _make_bypass_pipeline(window_ms=256, crossfade_ms=64)
    assert p.window_samples == 48000 * 256 // 1000
    assert p.crossfade_samples == 48000 * 64 // 1000
    assert p.hop_samples == p.window_samples - p.crossfade_samples


def test_timing_history_is_bounded():
    p = _make_bypass_pipeline()
    window = np.zeros(p.window_samples, dtype=np.float32)
    for _ in range(2000):
        p.process(window)
    assert len(p.t_total) <= 512
    assert p.last_timings_ms()["total_ms"] >= 0.0


def test_context_defaults_to_crossfade():
    p = _make_bypass_pipeline(window_ms=256, crossfade_ms=64)
    assert p.context_samples == p.crossfade_samples
    assert p.hop_samples == p.window_samples - p.context_samples


def test_context_validation():
    import pytest

    with pytest.raises(ValueError):
        Pipeline(denoiser=None, rvc=None, sr=48000, window_ms=16,
                 crossfade_ms=4, context_ms=2, denoise=False, bypass=False)
    with pytest.raises(ValueError):
        Pipeline(denoiser=None, rvc=None, sr=48000, window_ms=16,
                 crossfade_ms=2, context_ms=16, denoise=False, bypass=False)


def test_context_window_emits_continuous_stream():
    p = Pipeline(denoiser=None, rvc=None, sr=48000, window_ms=16,
                 crossfade_ms=2, context_ms=8, denoise=False, bypass=False)
    p._chain = lambda x: (x, 48000)  # type: ignore[attr-defined]
    ctx, cf, hop = p.context_samples, p.crossfade_samples, p.hop_samples
    assert (ctx, cf, hop) == (384, 96, 384)

    stream = np.arange(1, 5 * hop + 1, dtype=np.float32)
    carry = np.zeros(ctx, dtype=np.float32)
    emissions = []
    for k in range(4):
        new_hop = stream[k * hop:(k + 1) * hop]
        window = np.concatenate([carry, new_hop])
        out = p.process(window)
        assert out.shape == (hop,)
        emissions.append(out)
        carry = window[-ctx:]

    for k in (1, 2, 3):
        em = emissions[k]
        blend_src = stream[k * hop - cf:k * hop]
        body_src = stream[k * hop:(k + 1) * hop - cf]
        np.testing.assert_allclose(em[cf:], body_src, rtol=1e-6)
        ratio = em[:cf] / blend_src
        assert ratio.min() >= 0.99
        assert ratio.max() <= 1.42


def test_crossfade_is_applied_when_not_bypass():
    p = Pipeline(
        denoiser=None,
        rvc=None,
        sr=48000,
        window_ms=256,
        crossfade_ms=64,
        denoise=False,
        bypass=False,
    )
    p._chain = lambda x: (x, 48000)  # type: ignore[attr-defined]

    first = np.ones(p.window_samples, dtype=np.float32)
    out1 = p.process(first)
    assert out1.shape == (p.hop_samples,)
    np.testing.assert_allclose(out1[: p.crossfade_samples], first[: p.crossfade_samples])

    second = np.full(p.window_samples, 2.0, dtype=np.float32)
    out2 = p.process(second)
    cf = p.crossfade_samples
    t = np.linspace(0.0, 1.0, cf, endpoint=False, dtype=np.float32)
    fade_out = np.cos(t * 0.5 * np.pi)
    fade_in = np.sin(t * 0.5 * np.pi)
    expected_cf = fade_out * 1.0 + fade_in * 2.0
    np.testing.assert_allclose(out2[:cf], expected_cf, atol=1e-5)
    np.testing.assert_allclose(out2[cf:], second[cf : p.hop_samples])
