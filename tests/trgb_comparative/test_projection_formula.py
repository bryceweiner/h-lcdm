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
    PERTURBATIVE_LINEAR_THRESHOLD,
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

    We allow generous tolerance (±5 % of 1/282) because the exact value
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
    """NGC 4258 d_local = 7.58 Mpc should yield H_local ≈ 69.20 km/s/Mpc.

    This is a FORMULA CORRECTNESS test for the post-2026-04-25 linear-form
    correction (b parameter removed AND C(G) term removed; formula reduces
    to ``1 + (γ/H) · L``). The expected value 69.20 follows from
    H_CMB = 67.4, d_local = 7.58 Mpc, d_CMB = 13869.7 Mpc, γ/H ≈ 1/281.7.
    A deviation here means the formula is coded wrong — it does NOT
    validate the framework itself.
    """
    H0_pred, res = predict_local_H0(H_cmb=67.4, d_local_mpc=7.58)
    assert 68.5 < H0_pred < 70.0, (
        f"expected ≈ 69.20 km/s/Mpc (linear-form post-correction); "
        f"got {H0_pred:.3f}"
    )
    # Tight check against the user-specified reference.
    assert abs(H0_pred - 69.20) < 0.05, (
        f"NGC 4258 prediction must match the 69.20 reference within 0.05; "
        f"got {H0_pred:.4f}"
    )
    # Under the linear-form |γ/H · L| ≥ 1 criterion, NGC 4258 is well
    # inside the perturbative regime (γ/H · L ≈ 0.027 << 1) — no breakdown.
    assert not res.breakdown_flag
    assert res.gamma_over_H > 0.0


def test_lmc_direct_anchor_reproduces_linear_form_value():
    """LMC d_local = 0.05 Mpc should yield H_local ≈ 70.40 km/s/Mpc.

    Post-correction value under the linear form. The pre-2026-04-25
    formula gave ≈ 80.94 (with b = 0.5); the intermediate Form-1
    correction gave ≈ 88.85; the linear-form correction collapses to
    ≈ 70.40 because there is no longer a quadratic amplification term.
    LMC at γ/H · L ≈ 0.045 sits within the perturbative regime under
    the linear-form criterion.
    """
    H0_pred, res = predict_local_H0(H_cmb=67.4, d_local_mpc=0.05)
    assert 69.5 < H0_pred < 71.5, (
        f"expected ≈ 70.40 km/s/Mpc (linear-form post-correction); "
        f"got {H0_pred:.3f}"
    )
    assert abs(H0_pred - 70.40) < 0.05, (
        f"LMC prediction must match the 70.40 reference within 0.05; "
        f"got {H0_pred:.4f}"
    )
    # γ/H · L ≈ 0.045 — well below the |γ/H · L| ≥ 1 breakdown threshold.
    assert not res.breakdown_flag
    assert res.linear_term < 0.5


def test_holographic_h_ratio_rejects_deprecated_kwargs():
    """The 2026-04-25 corrections removed the `a` amplitude prefactor,
    the `b` parameter, and the `C(G)` term entirely.

    Passing any of `a=…`, `b=…`, `clustering_coefficient=…`, `C_graph=…`,
    `second_order=…`, or any other unrecognised kwarg must raise
    TypeError so accidental retention of the pre-correction call form
    is caught loudly rather than silently ignored.
    """
    deprecated = [
        ("a", 1.0),
        ("a_amplitude", 1.0),
        ("A_PREFACTOR", 1.0),
        ("b", 0.5),
        ("B_THRESHOLD", 0.5),
        ("b_ansatz", 0.5),
        ("clustering_coefficient", 0.5),
        ("C_graph", 0.4909),
        ("C_GRAPH", 0.4909),
        ("second_order", True),
    ]
    for name, value in deprecated:
        with pytest.raises(TypeError, match=r"unexpected keyword"):
            holographic_h_ratio(d_local_mpc=7.58, **{name: value})

    # Sanity: legitimate kwargs still work.
    res = holographic_h_ratio(d_local_mpc=7.58)
    assert res.ratio > 1.0


def test_breakdown_warning_does_not_fire_for_realistic_anchors():
    """Under the linear form, |γ/H · L| ≥ 1 never triggers for
    realistic distance-ladder anchors.

    The pre-2026-04-25 implementation flagged LMC (d_local = 0.05 Mpc)
    as breakdown via the d_local < 1 Mpc heuristic; the intermediate
    Form-1 correction flagged BOTH LMC and NGC 4258 via |C(G)·L| ≥ 1.
    Under the linear-form correction the small parameter is γ/H · L
    itself, which is ≪ 1 for any TRGB-anchored measurement (LMC ≈
    0.045; NGC 4258 ≈ 0.027). The breakdown infrastructure remains as
    defense in depth but does not fire for any realistic anchor.
    """
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res_lmc = holographic_h_ratio(d_local_mpc=0.05, emit_warning=True)
        res_ngc = holographic_h_ratio(d_local_mpc=7.58, emit_warning=True)
    assert not res_lmc.breakdown_flag
    assert not res_ngc.breakdown_flag
    assert not any(
        issubclass(w.category, PerturbativeBreakdownWarning) for w in caught
    )


def test_breakdown_fires_only_at_extreme_d_local():
    """The linear-form breakdown criterion |γ/H · L| ≥ 1 only fires for
    astronomically small d_local.

    With γ/H ≈ 1/282 and d_CMB ≈ 13870 Mpc, the |γ/H · L| = 1 contour
    sits at L = 282, i.e. d_local = d_CMB / e^282 — far below any
    physical anchor. We trigger it with an unphysically tiny d_local
    just to confirm the criterion is wired correctly.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PerturbativeBreakdownWarning)
        # γ/H · L = 1 ⟹ L = ~282. Pick d_local such that L > 282.
        # ln(13870 / d_local) > 282 ⟹ d_local < 13870 · e^{-282} ≈ 10^{-118}.
        # Use 1e-120 Mpc — ridiculous but inside the regime.
        res = holographic_h_ratio(d_local_mpc=1e-120, emit_warning=False)
    assert res.breakdown_flag
    assert res.breakdown_message is not None
    assert abs(res.linear_term) >= PERTURBATIVE_LINEAR_THRESHOLD


def test_breakdown_warning_suppressed_when_emit_warning_false():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        # Use the extreme-d_local trigger from the test above so the flag
        # actually fires; verify emit_warning=False suppresses stderr.
        res = holographic_h_ratio(d_local_mpc=1e-120, emit_warning=False)
    assert not any(
        issubclass(w.category, PerturbativeBreakdownWarning) for w in caught
    )
    assert res.breakdown_flag


def test_result_struct_has_no_clustering_or_quadratic_fields():
    """The HolographicRatioResult struct dropped clustering_coefficient
    and quadratic_correction fields after the 2026-04-25 C(G)-removal.

    Tests that nothing in the struct preserves the deprecated quantities
    even by accident.
    """
    res = holographic_h_ratio(d_local_mpc=7.58)
    fields = set(res.__dataclass_fields__.keys())
    forbidden = {"clustering_coefficient", "quadratic_correction", "second_order"}
    assert fields.isdisjoint(forbidden), (
        f"HolographicRatioResult must not carry deprecated fields; "
        f"found {fields & forbidden}"
    )
    # Required fields under the linear form.
    expected = {
        "ratio", "gamma_over_H", "L", "linear_term",
        "breakdown_flag", "breakdown_message",
        "d_local_mpc", "d_cmb_mpc",
    }
    assert expected.issubset(fields)


# ---------------------------------------------------------------------------
# Smooth dependence on d_local
# ---------------------------------------------------------------------------


def test_ratio_is_monotonic_in_log_d_cmb_over_d_local():
    """The ratio should increase monotonically as d_local shrinks
    (because L = ln(d_CMB/d_local) grows)."""
    d_grid = np.array([100.0, 50.0, 20.0, 10.0, 7.58, 5.0, 2.0, 1.0, 0.1, 0.05])
    ratios = np.array([holographic_h_ratio(d_local_mpc=d).ratio for d in d_grid])
    assert np.all(np.diff(ratios) > 0), (
        "Ratio must increase as d_local decreases."
    )


# ---------------------------------------------------------------------------
# Monte Carlo propagation
# ---------------------------------------------------------------------------


def test_monte_carlo_propagation_shape_and_breakdown_fraction():
    n = 1000
    H_cmb = np.random.default_rng(0).normal(67.36, 0.54, size=n)
    d_case_b = np.random.default_rng(1).normal(7.58, 0.08, size=n)
    d_case_a = np.maximum(
        np.random.default_rng(2).normal(0.0496, 0.0009, size=n), 1e-6
    )

    H0_b, per_b = propagate_projection_uncertainty(
        H_cmb, d_case_b, emit_warning=False
    )
    H0_a, per_a = propagate_projection_uncertainty(
        H_cmb, d_case_a, emit_warning=False
    )

    assert H0_b.size == n and H0_a.size == n
    flags_b = np.array([r.breakdown_flag for r in per_b])
    flags_a = np.array([r.breakdown_flag for r in per_a])
    # Under the linear form, neither anchor triggers the breakdown
    # criterion — γ/H · L ≪ 1 at both LMC and NGC 4258.
    assert flags_b.mean() < 0.05
    assert flags_a.mean() < 0.05
    # Linear-form predictions: NGC 4258 ≈ 69.20, LMC ≈ 70.40.
    assert np.median(H0_b) == pytest.approx(69.2, abs=1.0)
    assert np.median(H0_a) == pytest.approx(70.4, abs=1.0)


def test_breakdown_threshold_constant_is_1():
    assert PERTURBATIVE_LINEAR_THRESHOLD == pytest.approx(1.0, abs=1e-9)
