"""
End-to-end: short MCMC against synthetic bundle → best-fit near truth.

Uses the synthetic ``fake_bundle`` (no network) so this test runs in CI.
The full pipeline's ``run()`` against real Pantheon+ data is covered by
``test_data_loaders.py::test_pantheon_plus_loader`` (gated by RUN_E2E).
"""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.expansion_enhancement.likelihood import MODEL_A, MODEL_B_CONST
from pipeline.expansion_enhancement.mcmc_runner import MCMCSettings, run_model


def _quick_settings() -> MCMCSettings:
    # Tiny chain for speed — convergence is NOT checked here, only that the
    # best-fit recovers the injected truth within reasonable scatter.
    return MCMCSettings(n_walkers=16, n_steps=600, n_burnin=200, progress=False, seed=7)


def test_model_a_recovers_truth(fake_bundle, fiducial_params):
    """Short chain against synthetic data must locate the fiducial cosmology."""
    result = run_model(MODEL_A, fake_bundle, _quick_settings())
    # H0 within 2 km/s/Mpc of truth; Ω_m within 0.03.
    assert abs(result.best_fit["H0"] - fiducial_params["H0"]) < 2.0, result.best_fit
    assert abs(result.best_fit["Om"] - fiducial_params["Om"]) < 0.03, result.best_fit


def test_model_b_converges_on_synthetic(fake_bundle):
    """Model B fits run to completion, produce finite posteriors, and respect
    the ε∈[0,0.1] prior.

    NOTE: the synthetic bundle uses r_d=147.5 (ΛCDM), so Model B's r_d=150.71
    forces ε≠0 to compensate for the r_d mismatch — we do *not* expect ε=0
    here. This test just verifies the fit is numerically sensible.
    """
    result = run_model(MODEL_B_CONST, fake_bundle, _quick_settings())
    eps_lo, eps_med, eps_hi = result.credible_intervals["eps"]
    assert 0.0 <= eps_lo <= eps_med <= eps_hi <= 0.1, (eps_lo, eps_med, eps_hi)
    # Fit should populate best-fit entries and finite χ²
    assert np.isfinite(result.best_fit_chi2["total"])
    assert all(np.isfinite(v) for v in result.r_hat.values())
