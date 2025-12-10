"""
Tests for QTEP Ratio Verification - Recommendation 6
=====================================================

Tests the Bayesian verification of QTEP ratio in CMB polarization cross-spectra.
"""

import numpy as np
import pytest

from pipeline.recommendation.qtep_verification import QTEPVerificationTest
from hlcdm.parameters import HLCDM_PARAMS


@pytest.fixture
def qtep_verifier():
    """Create QTEP verification test instance."""
    return QTEPVerificationTest()


@pytest.fixture
def mock_residuals():
    """Create mock TE and EE residuals with QTEP-like coherence."""
    ell = np.arange(800, 1201, 10.0)
    n = len(ell)
    
    # Create correlated residuals (QTEP-like signal)
    # TE residuals
    res_te = 0.01 * np.sin(ell / 100.0) + 0.005 * np.random.randn(n)
    
    # EE residuals correlated with TE (QTEP ratio ≈ 2.257)
    # EE should have amplitude related to TE by QTEP ratio
    qtep_ratio = HLCDM_PARAMS.QTEP_RATIO
    res_ee = (res_te / qtep_ratio) + 0.003 * np.random.randn(n)
    
    return ell, res_te, res_ee


@pytest.fixture
def mock_residuals_null():
    """Create mock TE and EE residuals with no correlation (ΛCDM null)."""
    ell = np.arange(800, 1201, 10.0)
    n = len(ell)
    
    # Uncorrelated residuals
    res_te = 0.01 * np.random.randn(n)
    res_ee = 0.01 * np.random.randn(n)
    
    return ell, res_te, res_ee


def test_qtep_verifier_initialization(qtep_verifier):
    """Test QTEP verifier initializes correctly."""
    assert qtep_verifier is not None
    assert qtep_verifier.QTEP_RATIO_THEORY == pytest.approx(HLCDM_PARAMS.QTEP_RATIO, rel=1e-6)
    assert qtep_verifier.S_COH == pytest.approx(HLCDM_PARAMS.S_COH, rel=1e-6)
    assert qtep_verifier.S_DECOH == pytest.approx(HLCDM_PARAMS.S_DECOH, rel=1e-6)


def test_extract_coherence_amplitude(qtep_verifier, mock_residuals):
    """Test coherence amplitude extraction from TE-EE residuals."""
    ell, res_te, res_ee = mock_residuals
    
    coherence_data = qtep_verifier.extract_coherence_amplitude(
        res_te, res_ee, ell
    )
    
    assert 'ell_binned' in coherence_data
    assert 'coherence' in coherence_data
    assert 'coherence_err' in coherence_data
    assert 'correlation' in coherence_data
    assert 'n_points' in coherence_data
    
    assert len(coherence_data['ell_binned']) > 0
    assert len(coherence_data['coherence']) > 0
    assert np.all(np.isfinite(coherence_data['coherence']))
    assert np.all(coherence_data['coherence_err'] > 0)


def test_extract_coherence_amplitude_empty(qtep_verifier):
    """Test coherence extraction with empty/invalid data."""
    ell = np.array([])
    res_te = np.array([])
    res_ee = np.array([])
    
    coherence_data = qtep_verifier.extract_coherence_amplitude(
        res_te, res_ee, ell
    )
    
    assert len(coherence_data['ell_binned']) == 0
    assert len(coherence_data['coherence']) == 0


def test_extract_coherence_amplitude_out_of_range(qtep_verifier):
    """Test coherence extraction with data outside target range."""
    # Data at low multipoles (outside 800-1200 range)
    ell = np.arange(100, 500, 10.0)
    res_te = 0.01 * np.random.randn(len(ell))
    res_ee = 0.01 * np.random.randn(len(ell))
    
    coherence_data = qtep_verifier.extract_coherence_amplitude(
        res_te, res_ee, ell
    )
    
    # Should return empty result or handle gracefully
    assert isinstance(coherence_data, dict)


def test_fit_qtep_ratio_bayesian(qtep_verifier, mock_residuals):
    """Test Bayesian MCMC fit of QTEP ratio."""
    ell, res_te, res_ee = mock_residuals
    
    # Extract coherence
    coherence_data = qtep_verifier.extract_coherence_amplitude(
        res_te, res_ee, ell
    )
    
    if len(coherence_data['coherence']) == 0:
        pytest.skip("No coherence data extracted")
    
    # Fit QTEP ratio (use shorter MCMC for testing)
    qtep_fit = qtep_verifier.fit_qtep_ratio_bayesian(
        coherence_data,
        n_walkers=10,
        n_steps=100,
        n_burn=20
    )
    
    assert 'R_median' in qtep_fit
    assert 'R_mean' in qtep_fit
    assert 'R_std' in qtep_fit
    assert 'R_credible_68' in qtep_fit
    assert 'R_credible_95' in qtep_fit
    
    # Check reasonable values
    assert 0.5 <= qtep_fit['R_median'] <= 5.0
    assert qtep_fit['R_std'] > 0
    assert len(qtep_fit['R_credible_68']) == 2
    assert len(qtep_fit['R_credible_95']) == 2


def test_compute_bayes_factor(qtep_verifier, mock_residuals):
    """Test Bayes factor computation."""
    ell, res_te, res_ee = mock_residuals
    
    # Extract coherence
    coherence_data = qtep_verifier.extract_coherence_amplitude(
        res_te, res_ee, ell
    )
    
    if len(coherence_data['coherence']) == 0:
        pytest.skip("No coherence data extracted")
    
    # Fit QTEP ratio
    qtep_fit = qtep_verifier.fit_qtep_ratio_bayesian(
        coherence_data,
        n_walkers=10,
        n_steps=50,
        n_burn=10
    )
    
    # Compute Bayes factor
    bayes_factor = qtep_verifier.compute_bayes_factor(
        coherence_data, qtep_fit
    )
    
    assert 'bayes_factor' in bayes_factor
    assert 'log_bf' in bayes_factor
    assert 'interpretation' in bayes_factor
    assert 'evidence_strength' in bayes_factor
    
    assert np.isfinite(bayes_factor['bayes_factor']) or np.isnan(bayes_factor['bayes_factor'])
    assert bayes_factor['evidence_strength'] in ['none', 'weak', 'moderate', 'strong', 'very strong']


def test_check_hlcdm_consistency(qtep_verifier):
    """Test H-ΛCDM consistency check."""
    # Mock fit result
    qtep_fit = {
        'R_median': 2.2,
        'R_std': 0.1,
    }
    
    consistency = qtep_verifier.check_hlcdm_consistency(qtep_fit)
    
    assert 'R_predicted' in consistency
    assert 'R_fitted' in consistency
    assert 'within_1sigma' in consistency
    assert 'within_2sigma' in consistency
    assert 'tension_sigma' in consistency
    
    assert consistency['R_predicted'] == pytest.approx(HLCDM_PARAMS.QTEP_RATIO, rel=1e-6)
    assert consistency['R_fitted'] == 2.2
    assert isinstance(consistency['within_1sigma'], bool)
    assert isinstance(consistency['within_2sigma'], bool)


def test_check_hlcdm_consistency_nan(qtep_verifier):
    """Test consistency check with NaN values."""
    qtep_fit = {
        'R_median': np.nan,
        'R_std': np.nan,
    }
    
    consistency = qtep_verifier.check_hlcdm_consistency(qtep_fit)
    
    assert np.isnan(consistency['R_fitted'])
    assert consistency['within_1sigma'] == False
    assert consistency['within_2sigma'] == False


def test_run_verification(qtep_verifier):
    """Test complete QTEP verification workflow."""
    # Create mock residuals structure
    ell = np.arange(800, 1201, 10.0)
    n = len(ell)
    
    # Create correlated residuals
    res_te = 0.01 * np.sin(ell / 100.0) + 0.005 * np.random.randn(n)
    qtep_ratio = HLCDM_PARAMS.QTEP_RATIO
    res_ee = (res_te / qtep_ratio) + 0.003 * np.random.randn(n)
    
    residuals = {
        'planck_2018': {
            'TE': {
                'ell': ell,
                'residual': res_te,
                'residual_fraction': res_te / 1e-10,
                'cl_err': 0.001 * np.ones(n),
            },
            'EE': {
                'ell': ell,
                'residual': res_ee,
                'residual_fraction': res_ee / 1e-10,
                'cl_err': 0.001 * np.ones(n),
            },
        }
    }
    
    results = qtep_verifier.run_verification(residuals)
    
    assert 'test_name' in results
    assert 'ell_range' in results
    assert 'spectra' in results
    assert 'surveys' in results
    assert 'combined' in results
    
    assert results['test_name'] == 'QTEP Ratio Verification'
    assert results['ell_range'] == [800, 1200]
    assert 'TE' in results['spectra']
    assert 'EE' in results['spectra']
    
    # Check survey results
    if 'planck_2018' in results['surveys']:
        survey_results = results['surveys']['planck_2018']
        assert 'coherence_data' in survey_results
        assert 'qtep_fit' in survey_results
        assert 'bayes_factor' in survey_results
        assert 'hlcdm_consistency' in survey_results


def test_run_verification_empty(qtep_verifier):
    """Test verification with empty residuals."""
    residuals = {}
    
    results = qtep_verifier.run_verification(residuals)
    
    assert 'test_name' in results
    assert 'combined' in results
    assert len(results['surveys']) == 0


def test_run_verification_missing_spectra(qtep_verifier):
    """Test verification with missing TE or EE spectra."""
    ell = np.arange(800, 1201, 10.0)
    n = len(ell)
    
    residuals = {
        'planck_2018': {
            'TE': {
                'ell': ell,
                'residual': 0.01 * np.random.randn(n),
                'residual_fraction': np.random.randn(n),
                'cl_err': 0.001 * np.ones(n),
            },
            # Missing EE
        }
    }
    
    results = qtep_verifier.run_verification(residuals)
    
    # Should handle gracefully
    assert 'test_name' in results
    assert isinstance(results['surveys'], dict)


def test_qtep_ratio_theoretical_value(qtep_verifier):
    """Test that theoretical QTEP ratio matches HLCDM_PARAMS."""
    assert qtep_verifier.QTEP_RATIO_THEORY == pytest.approx(
        HLCDM_PARAMS.QTEP_RATIO, rel=1e-10
    )
    
    # Verify the calculation
    expected = np.log(2) / (1 - np.log(2))
    assert qtep_verifier.QTEP_RATIO_THEORY == pytest.approx(expected, rel=1e-10)


def test_entropy_components(qtep_verifier):
    """Test that entropy components match HLCDM_PARAMS."""
    assert qtep_verifier.S_COH == pytest.approx(HLCDM_PARAMS.S_COH, rel=1e-10)
    assert qtep_verifier.S_DECOH == pytest.approx(HLCDM_PARAMS.S_DECOH, rel=1e-10)
    
    # Verify calculations
    assert qtep_verifier.S_COH == pytest.approx(np.log(2), rel=1e-10)
    assert qtep_verifier.S_DECOH == pytest.approx(np.log(2) - 1, rel=1e-10)


@pytest.mark.slow
def test_bayesian_fit_convergence(qtep_verifier, mock_residuals):
    """Test that Bayesian fit converges with sufficient samples."""
    ell, res_te, res_ee = mock_residuals
    
    coherence_data = qtep_verifier.extract_coherence_amplitude(
        res_te, res_ee, ell
    )
    
    if len(coherence_data['coherence']) == 0:
        pytest.skip("No coherence data extracted")
    
    # Run with more samples
    qtep_fit = qtep_verifier.fit_qtep_ratio_bayesian(
        coherence_data,
        n_walkers=20,
        n_steps=500,
        n_burn=100
    )
    
    # Check that credible intervals are reasonable
    assert qtep_fit['R_credible_68'][0] < qtep_fit['R_median'] < qtep_fit['R_credible_68'][1]
    assert qtep_fit['R_credible_95'][0] < qtep_fit['R_median'] < qtep_fit['R_credible_95'][1]
    assert qtep_fit['R_credible_68'][1] - qtep_fit['R_credible_68'][0] < qtep_fit['R_credible_95'][1] - qtep_fit['R_credible_95'][0]

