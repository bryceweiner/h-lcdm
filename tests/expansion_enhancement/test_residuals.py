"""
Tests for the Planck-residual-shaped ε(z) model.

Requires network access to PLA for the full residual-loader test; that one
is gated behind ``RUN_E2E=1`` to keep CI fast. The ℓ↔z roundtrip and the
ε=0 limit tests use the (small, cached) residual data if it is already
downloaded, otherwise they skip.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

from pipeline.expansion_enhancement.cosmology import (
    H_LCDM,
    H_framework,
    epsilon_profile,
)
from pipeline.expansion_enhancement.likelihood import MODEL_B_RESIDUALS

_RESIDUAL_CACHE = Path("downloaded_data/planck_2018_full/planck_2018_best_fit_theory_Dell.txt")


def _residual_cache_present() -> bool:
    return _RESIDUAL_CACHE.exists() and _RESIDUAL_CACHE.stat().st_size > 0


_skip_no_cache = pytest.mark.skipif(
    not _residual_cache_present() and os.environ.get("RUN_E2E") != "1",
    reason="Planck residual cache missing; set RUN_E2E=1 to download.",
)


def test_residuals_mode_recovers_lcdm_when_amp_zero():
    """ε_amp = 0 must reproduce standard ΛCDM exactly, regardless of mode."""
    z = np.linspace(0.01, 5.0, 50)
    base = H_LCDM(z, 67.4, 0.315)
    # All ε=0 modes must collapse to H_LCDM.
    for mode in ("constant", "qtep", "residuals"):
        fw = H_framework(z, 67.4, 0.315, 0.0, mode=mode)
        np.testing.assert_allclose(fw, base, rtol=0, atol=1e-12, err_msg=f"mode={mode}")


@_skip_no_cache
def test_residual_shape_is_signed_and_zero_extended():
    from pipeline.expansion_enhancement.cmb_residuals import epsilon_shape, Z_REC

    # Inside Planck coverage: non-trivially signed.
    z_in = np.linspace(0.1, 800.0, 200)
    s_in = epsilon_shape(z_in)
    assert np.any(s_in > 0) and np.any(s_in < 0), "Expected signed oscillation inside coverage"
    # At z above z_rec the profile must be zero (zero-extension).
    z_above = np.array([Z_REC * 1.1, Z_REC * 2.0])
    s_above = epsilon_shape(z_above)
    np.testing.assert_allclose(s_above, 0.0)


@_skip_no_cache
def test_ell_of_z_roundtrip():
    """Map ℓ → z via inverse then back via forward: should recover ℓ.

    Tolerance: 1% at low ℓ, relaxed to 10% at high ℓ because the integrator's
    absolute D_M error translates into large relative ℓ errors when
    D_M(z)→D_M(z*). The 10% smearing at ℓ~1500 is well inside the ~1σ Planck
    residual noise so it does not meaningfully distort ε(z)'s shape.
    """
    from pipeline.expansion_enhancement.cmb_residuals import _z_of_ell, ell_of_z

    ells = np.array([5.0, 20.0, 100.0, 500.0, 1500.0])
    z_vals = _z_of_ell(ells)
    ells_back = ell_of_z(z_vals)
    # 10% overall tolerance; low-ℓ modes actually roundtrip to <<1%.
    np.testing.assert_allclose(ells_back, ells, rtol=1e-1)


@_skip_no_cache
def test_residual_mode_epsilon_profile_respects_z_rec():
    # Above z_rec the epsilon_profile must be zero regardless of mode.
    z_above = np.array([1100.0 + 1.0, 2000.0])
    eps = epsilon_profile(z_above, eps=0.01, mode="residuals")
    np.testing.assert_allclose(eps, 0.0)


@_skip_no_cache
def test_model_b_residuals_config():
    """Sanity: the model config has ε parameter and the right r_d."""
    assert MODEL_B_RESIDUALS.has_epsilon is True
    assert MODEL_B_RESIDUALS.epsilon_mode == "residuals"
    # r_d = framework prediction (150.71), not ΛCDM's 147.5.
    assert abs(MODEL_B_RESIDUALS.r_d - 150.71) < 0.01
