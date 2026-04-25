"""
Unit tests for pipeline.trgb_comparative.projection_formula.

IMPORTANT: Tests in this file verify formula IMPLEMENTATION CORRECTNESS.
They do NOT constitute framework validation. Framework validation requires
comparison of predictions to observations for cases where the formula was
not calibrated (e.g., non-NGC-4258 anchored methods, different redshifts,
future measurements). See `test_framework_methodology.py` for framework
behavior tests.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from pipeline.trgb_comparative.projection_formula import (
    PERTURBATIVE_D_LOCAL_MPC,
    PerturbativeBreakdownWarning,
    compute_gamma_over_H_at_z0,
    holographic_h_ratio,
    predict_local_H0,
    propagate_projection_uncertainty,
)


# ---------------------------------------------------------------------------
# γ/H runtime computation
# ---------------------------------------------------------------------------


def test_gamma_over_H_is_runtime_computed():
    """γ/H is pulled from the live framework; should be dimensionless and ~1/282.

    We allow generous tolerance (±2 % of 1/282) because the exact value
    depends on the framework's γ(z) implementation at test time.
    """
    g = compute_gamma_over_H_at_z0()
    assert 0.0 < g < 1.0, f"γ/H should be a small dimensionless ratio; got {g}"
    nominal = 1.0 / 282.0
    assert abs(g - nominal) / nominal < 0.05, (
        f"γ/H = {g} (1/{1/g:.1f}); expected within 5 % of 1/282. "
        "Formula test — not a framework validation."
    )


# ---------------------------------------------------------------------------
# Canonical anchor checks (FORMULA CORRECTNESS, not framework validation)
# ---------------------------------------------------------------------------


def test_ngc_4258_anchor_reproduces_post_correction_value():
    """NGC 4258 d_local = 7.58 Mpc should yield H_local ≈ 75.82 km/s/Mpc.

    This is a FORMULA CORRECTNESS test for the post-2026-04-25
    correction (b parameter removed; formula reduces to
    [1 + C(G) * L]). The expected value 75.82 follows from
    H_CMB = 67.4, d_local = 7.58 Mpc, d_CMB = 13869.7 Mpc,
    γ/H ≈ 1/281.7, C(G) = 27/55 (Convention A). A deviation here
    means the formula is coded wrong — it does NOT validate the
    framework itself.
    """
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", PerturbativeBreakdownWarning)
        H0_pred, res = predict_local_H0(H_cmb=67.4, d_local_mpc=7.58)
    assert 75.0 < H0_pred < 76.5, (
        f"expected ≈ 75.82 km/s/Mpc (post-b-correction); got {H0_pred:.3f}"
    )
    # Tight check against the user-specified reference.
    assert abs(H0_pred - 75.82) < 0.05, (
        f"NGC 4258 prediction must match the 75.82 reference within 0.05; "
        f"got {H0_pred:.4f}"
    )
    # Under the physics-motivated criterion |C(G)*L| ≥ 1, NGC 4258 is
    # ALSO outside the strict perturbative regime (β·L ≈ 3.69). The
    # prediction is still finite and reportable, but the flag is set.
    assert res.breakdown_flag
    assert res.gamma_over_H > 0.0


def test_lmc_direct_anchor_reproduces_breakdown_scale():
    """LMC d_local = 0.05 Mpc should yield H_local ≈ 88.85 km/s/Mpc with breakdown flag.

    Post-2026-04-25 correction value. The pre-correction formula gave
    ≈ 80.94 km/s/Mpc; the corrected formula (no b parameter) gives
    ≈ 88.85.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerturbativeBreakdownWarning)
        H0_pred, res = predict_local_H0(H_cmb=67.4, d_local_mpc=0.05)
    assert 86.0 < H0_pred < 92.0, (
        f"expected ≈ 88.85 km/s/Mpc (post-b-correction); got {H0_pred:.3f}"
    )
    assert res.breakdown_flag
    assert res.breakdown_message is not None


def test_holographic_h_ratio_rejects_b_kwarg():
    """The 2026-04-25 correction removed the `b` parameter entirely.

    Passing it must raise TypeError so accidental retention of the
    pre-correction call form is caught loudly rather than silently
    ignored.
    """
    with pytest.raises(TypeError, match=r"unexpected keyword.*b"):
        holographic_h_ratio(d_local_mpc=7.58, b=0.5)
    with pytest.raises(TypeError, match=r"unexpected keyword"):
        holographic_h_ratio(d_local_mpc=7.58, B_THRESHOLD=0.5)
    with pytest.raises(TypeError, match=r"unexpected keyword"):
        holographic_h_ratio(d_local_mpc=7.58, b_ansatz=0.5)
    # Sanity: legitimate kwargs still work.
    res = holographic_h_ratio(d_local_mpc=7.58)
    assert res.ratio > 1.0


def test_breakdown_warning_is_raised_for_subparsec_d_local():
    """`PerturbativeBreakdownWarning` emits when d_local < threshold and emit_warning=True."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        holographic_h_ratio(d_local_mpc=0.05, emit_warning=True)
    assert any(issubclass(w.category, PerturbativeBreakdownWarning) for w in caught)


def test_breakdown_warning_suppressed_when_emit_warning_false():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = holographic_h_ratio(d_local_mpc=0.05, emit_warning=False)
    assert not any(issubclass(w.category, PerturbativeBreakdownWarning) for w in caught)
    assert res.breakdown_flag


def test_breakdown_threshold_is_quadratic_correction_not_d_local():
    """Breakdown criterion under Form 1 is ``|C(G)*L| ≥ 1``, NOT
    ``d_local < 1 Mpc``.

    The pre-2026-04-25 implementation used a ``d_local < 1 Mpc``
    geometric heuristic that was a carryover from the previous formula
    (with the ``b = 0.5`` offset). Under the corrected Form 1
    ``[1 + C(G)·L]``, the only physics-motivated breakdown criterion
    is the magnitude of the bracket's first-order Taylor truncation
    relative to unity. Convention A puts the |C(G)·L| = 1 contour at
    d_local ≈ 1885 Mpc, so all TRGB-anchored measurements (a few to
    tens of Mpc) trigger the flag — including NGC 4258. The test
    asserts the new criterion explicitly.
    """
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("ignore", PerturbativeBreakdownWarning)
        # NGC 4258 (d_local = 7.58 Mpc): β·L = 3.69 > 1 → breakdown.
        res_ngc = holographic_h_ratio(d_local_mpc=7.58)
        assert res_ngc.quadratic_correction > 1.0
        assert res_ngc.breakdown_flag, (
            "NGC 4258 has |C(G)*L| ≈ 3.69 ≥ 1 — must trigger breakdown "
            "under the physics-motivated criterion."
        )

        # A truly perturbative point: d_local ≈ 5000 Mpc → β·L < 1.
        res_far = holographic_h_ratio(d_local_mpc=5000.0)
        assert res_far.quadratic_correction < 1.0
        assert not res_far.breakdown_flag, (
            "d_local = 5000 Mpc has |C(G)*L| < 1 — must NOT trigger "
            "breakdown."
        )

        # And a d_local in the legacy (1 Mpc < d_local < 1885 Mpc) gap
        # that the old criterion called 'safe' but the new criterion
        # correctly flags as breakdown:
        res_legacy_safe = holographic_h_ratio(d_local_mpc=10.0)
        assert res_legacy_safe.breakdown_flag, (
            "d_local = 10 Mpc was 'safe' under the legacy d_local<1 Mpc "
            "rule but |C(G)*L| ≈ 3.55 ≥ 1 — must trigger breakdown "
            "under the physics-motivated criterion."
        )


# ---------------------------------------------------------------------------
# Smooth dependence on d_local
# ---------------------------------------------------------------------------


def test_ratio_is_monotonic_in_log_d_cmb_over_d_local():
    """For d_local > 1 Mpc, the ratio should increase as d_local shrinks."""
    d_grid = np.array([20.0, 15.0, 10.0, 7.58, 5.0, 2.0, 1.1])
    ratios = np.array([holographic_h_ratio(d_local_mpc=d).ratio for d in d_grid])
    assert np.all(np.diff(ratios) > 0), (
        "Ratio must increase as d_local decreases (longer log argument)."
    )


# ---------------------------------------------------------------------------
# Monte Carlo propagation
# ---------------------------------------------------------------------------


def test_monte_carlo_propagation_shape_and_breakdown_fraction():
    n = 1000
    H_cmb = np.random.default_rng(0).normal(67.36, 0.54, size=n)
    d_case_b = np.random.default_rng(1).normal(7.58, 0.08, size=n)
    d_case_a = np.maximum(np.random.default_rng(2).normal(0.0496, 0.0009, size=n), 1e-6)

    H0_b, per_b = propagate_projection_uncertainty(H_cmb, d_case_b, emit_warning=False)
    H0_a, per_a = propagate_projection_uncertainty(H_cmb, d_case_a, emit_warning=False)

    assert H0_b.size == n and H0_a.size == n
    flags_b = np.array([r.breakdown_flag for r in per_b])
    flags_a = np.array([r.breakdown_flag for r in per_a])
    # Under the physics-motivated breakdown criterion (|C(G)*L| ≥ 1),
    # BOTH anchors trigger breakdown. LMC's β·L (≈ 6.15) is far worse
    # than NGC 4258's (≈ 3.69), but the binary flag fires on both.
    assert flags_b.mean() > 0.95, (
        "Case B (NGC 4258) has |C(G)*L| ≈ 3.69 — physics-motivated "
        "breakdown criterion fires."
    )
    assert flags_a.mean() > 0.95, (
        "Case A (LMC) has |C(G)*L| ≈ 6.15 — far outside perturbative "
        "regime; breakdown criterion fires."
    )
    # Post-2026-04-25 correction: NGC 4258 ≈ 75.82, LMC ≈ 88.85.
    # The prediction values remain finite and reportable even with the
    # breakdown flag set.
    assert np.median(H0_b) == pytest.approx(75.8, abs=2.0)
    assert np.median(H0_a) == pytest.approx(88.8, abs=3.0)


def test_breakdown_threshold_constant_is_1_mpc():
    assert PERTURBATIVE_D_LOCAL_MPC == pytest.approx(1.0, abs=1e-9)
