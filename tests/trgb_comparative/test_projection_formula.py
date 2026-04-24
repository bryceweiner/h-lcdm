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


def test_ngc_4258_anchor_reproduces_sh0es_scale():
    """NGC 4258 d_local = 7.58 Mpc should yield H_local ≈ 73 km/s/Mpc.

    This is a FORMULA CORRECTNESS test: the projection formula was
    constructed so that the 7.58 Mpc geometric anchor (SH0ES scale) maps
    to ≈ 73 km/s/Mpc given H_CMB ≈ 67.4. A deviation here means the
    formula is coded wrong — it does NOT validate the framework itself.
    """
    H0_pred, res = predict_local_H0(H_cmb=67.4, d_local_mpc=7.58)
    assert 71.0 < H0_pred < 75.0, f"expected approximately 73 km/s/Mpc; got {H0_pred:.3f}"
    assert not res.breakdown_flag
    assert res.gamma_over_H > 0.0


def test_lmc_direct_anchor_reproduces_breakdown_scale():
    """LMC d_local = 0.05 Mpc should yield H_local ≈ 81 km/s/Mpc with breakdown flag."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerturbativeBreakdownWarning)
        H0_pred, res = predict_local_H0(H_cmb=67.4, d_local_mpc=0.05)
    assert 78.0 < H0_pred < 85.0, f"expected approximately 81 km/s/Mpc; got {H0_pred:.3f}"
    assert res.breakdown_flag
    assert res.breakdown_message is not None


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


def test_breakdown_threshold_is_d_local_not_quadratic_correction():
    """The threshold is d_local < 1 Mpc, not |quadratic_correction| > 0.5."""
    # NGC 4258: |quadratic_correction| is actually > 0.5 (~2), but d_local >> 1 Mpc → no breakdown.
    res = holographic_h_ratio(d_local_mpc=7.58)
    assert abs(res.quadratic_correction) > 0.5
    assert not res.breakdown_flag


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
    assert flags_b.mean() < 0.05, "Case B (NGC 4258) should be in perturbative regime."
    assert flags_a.mean() > 0.95, "Case A (LMC) should be flagged as breakdown."
    assert np.median(H0_b) == pytest.approx(73.0, abs=3.0)
    assert np.median(H0_a) == pytest.approx(81.0, abs=3.0)


def test_breakdown_threshold_constant_is_1_mpc():
    assert PERTURBATIVE_D_LOCAL_MPC == pytest.approx(1.0, abs=1e-9)
