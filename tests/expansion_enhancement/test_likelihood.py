"""
Likelihood tests: χ² zero at the truth, covariances positive-definite, priors enforced.
"""

import numpy as np
import pytest

from pipeline.expansion_enhancement.likelihood import (
    MODEL_A,
    MODEL_B_CONST,
    MODEL_B_QTEP,
    chi2_bao,
    chi2_cmb,
    chi2_sn,
    chi2_total,
    log_posterior,
)


def test_bao_chi2_zero_at_truth(fake_bundle, fiducial_params):
    """Synthetic data generated from the fiducial cosmology + Model-A r_d
    must give χ²(BAO) = 0 when evaluated at the same cosmology."""
    theta = np.array([fiducial_params["H0"], fiducial_params["Om"]])
    chi2 = chi2_bao(theta, MODEL_A, fake_bundle.bao)
    assert chi2 == pytest.approx(0.0, abs=1e-8)


def test_sn_chi2_zero_at_truth_after_M_marginalization(fake_bundle, fiducial_params):
    """SN block with profiled M should give 0 when data = model."""
    theta = np.array([fiducial_params["H0"], fiducial_params["Om"]])
    chi2 = chi2_sn(theta, MODEL_A, fake_bundle.sn)
    assert chi2 == pytest.approx(0.0, abs=1e-8)


def test_cmb_chi2_zero_at_truth(fake_bundle, fiducial_params):
    """CMB θ* χ² is 0 when θ*_model == θ*_obs."""
    theta = np.array([fiducial_params["H0"], fiducial_params["Om"]])
    chi2 = chi2_cmb(theta, MODEL_A, fake_bundle.cmb)
    assert chi2 == pytest.approx(0.0, abs=1e-6)


def test_total_chi2_zero_at_truth(fake_bundle, fiducial_params):
    theta = np.array([fiducial_params["H0"], fiducial_params["Om"]])
    out = chi2_total(theta, MODEL_A, fake_bundle)
    assert out["total"] == pytest.approx(0.0, abs=1e-6)
    assert out["total"] == out["bao"] + out["sn"] + out["cmb"]


def test_chi2_positive_off_truth(fake_bundle, fiducial_params):
    """Offset parameters yield strictly positive χ²."""
    theta = np.array([fiducial_params["H0"] + 5.0, fiducial_params["Om"] - 0.05])
    out = chi2_total(theta, MODEL_A, fake_bundle)
    assert out["total"] > 0.0


def test_log_posterior_neg_inf_outside_prior(fake_bundle):
    """Walkers outside the box must score -inf, not NaN/error."""
    # H0 below lower prior
    theta = np.array([50.0, 0.315])
    lp = log_posterior(theta, MODEL_A, fake_bundle)
    assert lp == -np.inf
    # Om above upper prior
    theta = np.array([67.4, 0.5])
    lp = log_posterior(theta, MODEL_A, fake_bundle)
    assert lp == -np.inf


def test_log_posterior_model_b_enforces_eps_prior(fake_bundle):
    """Model B's ε prior is [0, 0.1]; negative ε → -inf."""
    theta = np.array([67.4, 0.315, -0.01])
    lp = log_posterior(theta, MODEL_B_CONST, fake_bundle)
    assert lp == -np.inf
    # Valid ε → finite.
    theta = np.array([67.4, 0.315, 0.02])
    lp = log_posterior(theta, MODEL_B_CONST, fake_bundle)
    assert np.isfinite(lp)


def test_model_b_qtep_has_eps(fake_bundle):
    """Model B_qtep has 3 params including eps, and uses qtep mode."""
    assert MODEL_B_QTEP.has_epsilon
    assert MODEL_B_QTEP.epsilon_mode == "qtep"
    theta = np.array([67.4, 0.315, 0.02])
    lp = log_posterior(theta, MODEL_B_QTEP, fake_bundle)
    assert np.isfinite(lp)


def test_covariance_psd(fake_bundle):
    """Our synthetic BAO+SN covariance matrices are positive definite."""
    w_bao = np.linalg.eigvalsh(fake_bundle.bao.cov)
    w_sn = np.linalg.eigvalsh(fake_bundle.sn.cov)
    assert np.all(w_bao > 0)
    assert np.all(w_sn > 0)
