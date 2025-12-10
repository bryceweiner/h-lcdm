"""
Tests for CMB Model Comparison Recommendation Pipeline
======================================================

Tests the new scientifically rigorous ΛCDM vs H-ΛCDM comparison pipeline.
"""

import numpy as np
import pytest

from pipeline.recommendation import RecommendationPipeline


def _mock_cmb_spectra():
    """Generate mock CMB TT power spectrum."""
    ell = np.arange(2, 2500, 10.0)
    # Simple power law with acoustic oscillations
    cl_base = 1e-10 * (ell / 1000.0) ** -1.0
    # Add acoustic oscillations
    cl_base *= (1.0 + 0.1 * np.sin(ell / 100.0))
    sigma = 0.05 * cl_base
    return ell, cl_base, sigma


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    """Create pipeline with mocked data."""
    pipe = RecommendationPipeline(output_dir=tmp_path)
    ell, cl, sigma = _mock_cmb_spectra()
    
    # Mock data loader
    monkeypatch.setattr(
        pipe.data_loader,
        "load_planck_2018",
        lambda: {"TT": (ell, cl, sigma)},
    )
    monkeypatch.setattr(
        pipe.data_loader,
        "load_act_dr6",
        lambda: {"TT": (ell, cl * 1.01, sigma)},  # Slight difference
    )
    monkeypatch.setattr(
        pipe.data_loader,
        "load_spt3g",
        lambda: {"TT": (ell, cl * 0.99, sigma)},  # Slight difference
    )
    
    # Mock ML anomaly context
    monkeypatch.setattr(
        pipe,
        "_load_ml_anomaly_context",
        lambda *_args, **_kwargs: {
            "z_eff": 2.2,
            "used_anomaly_indices": [4, 6, 9],
            "redshift_sources_found": 3,
            "fallback_used": False,
        },
    )
    
    return pipe


def test_pipeline_initialization(pipeline):
    """Test pipeline initializes correctly."""
    assert pipeline.lcdm_model is not None
    assert pipeline.hlcdm_model is not None
    assert pipeline.camb_interface is not None


def test_load_cmb_data(pipeline):
    """Test CMB data loading."""
    datasets = pipeline._load_cmb_data()
    assert isinstance(datasets, dict)
    assert len(datasets) > 0
    assert "planck_2018" in datasets or "act_dr6" in datasets or "spt3g" in datasets


def test_combine_datasets(pipeline):
    """Test dataset combination."""
    datasets = pipeline._load_cmb_data()
    combined = pipeline._combine_datasets(datasets)
    assert "ell" in combined
    assert "cl_obs" in combined
    assert "cl_err" in combined
    assert len(combined["ell"]) > 0


@pytest.mark.slow
def test_run_model_comparison(pipeline):
    """Test full model comparison analysis."""
    # Use shorter MCMC for testing
    results = pipeline.run({
        "mcmc_n_steps": 100,  # Short for testing
        "n_bootstrap": 10,  # Short for testing
        "n_null": 10,  # Short for testing
    })
    
    # Check structure
    assert "datasets" in results
    assert "lcdm_fit" in results
    assert "hlcdm_fit" in results
    assert "model_comparison" in results
    assert "residuals" in results
    
    # Check ΛCDM fit
    lcdm_fit = results["lcdm_fit"]
    assert "best_fit_params" in lcdm_fit
    assert "chi2" in lcdm_fit
    assert lcdm_fit["chi2"] > 0
    
    # Check H-ΛCDM fit
    hlcdm_fit = results["hlcdm_fit"]
    assert "best_fit_params" in hlcdm_fit
    assert "chi2" in hlcdm_fit
    assert hlcdm_fit["chi2"] > 0
    
    # Check model comparison
    comparison = results["model_comparison"]
    assert "delta_chi2_eff" in comparison
    assert "preferred_model" in comparison
    assert comparison["preferred_model"] in ["LCDM", "HLCDM", "INDETERMINATE"]


def test_validate_bootstrap_and_null(pipeline):
    """Test validation includes bootstrap and null hypothesis tests."""
    # Run analysis first
    pipeline.run({
        "mcmc_n_steps": 50,  # Very short for testing
        "n_bootstrap": 10,
        "n_null": 10,
    })
    
    # Run validation
    validation = pipeline.validate({
        "n_bootstrap": 10,
        "n_null": 10,
    })
    
    # Check structure
    assert "bootstrap" in validation
    assert "null_hypothesis" in validation
    
    # Check bootstrap
    bootstrap = validation["bootstrap"]
    assert "lcdm" in bootstrap
    assert "hlcdm" in bootstrap
    
    bootstrap_lcdm = bootstrap["lcdm"]
    assert bootstrap_lcdm["n_bootstrap"] >= 10
    assert "chi2_mean" in bootstrap_lcdm
    
    # Check null hypothesis
    null_hyp = validation["null_hypothesis"]
    assert "lcdm" in null_hyp
    assert "hlcdm" in null_hyp
    
    null_lcdm = null_hyp["lcdm"]
    assert null_lcdm["n_null"] >= 10
    assert "p_value_chi2" in null_lcdm


def test_model_comparison_statistics(pipeline):
    """Test model comparison computes all statistics."""
    results = pipeline.run({
        "mcmc_n_steps": 50,
        "n_bootstrap": 5,
        "n_null": 5,
    })
    
    comparison = results["model_comparison"]
    
    # Check all required statistics
    assert "chi2_lcdm" in comparison
    assert "chi2_hlcdm" in comparison
    assert "delta_chi2_eff" in comparison
    assert "aic_lcdm" in comparison
    assert "aic_hlcdm" in comparison
    assert "delta_aic" in comparison
    assert "bic_lcdm" in comparison
    assert "bic_hlcdm" in comparison
    assert "delta_bic" in comparison
    assert "preferred_model" in comparison
    assert "significance" in comparison


def test_residuals_computed(pipeline):
    """Test residuals are computed for both models."""
    results = pipeline.run({
        "mcmc_n_steps": 50,
        "n_bootstrap": 5,
        "n_null": 5,
    })
    
    residuals = results["residuals"]
    assert "lcdm" in residuals
    assert "hlcdm" in residuals
    
    lcdm_res = residuals["lcdm"]
    assert "ell" in lcdm_res
    assert "residual" in lcdm_res
    assert "residual_fraction" in lcdm_res
    
    hlcdm_res = residuals["hlcdm"]
    assert "ell" in hlcdm_res
    assert "residual" in hlcdm_res
    assert "residual_fraction" in hlcdm_res
    assert "gamma_modulation" in hlcdm_res
