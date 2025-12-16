"""
Unit tests for peak ratio statistics.

Verifies that beta_err, ndof, and chi2_reduced are returned and finite.
"""

import pytest
import numpy as np
from pipeline.cmb_gw.analysis.peak_analysis import fit_peak_ratios_to_data


def test_peak_statistics_present():
    """Test that all required statistics are present in results."""
    # Create mock peak data
    planck_peaks = {
        'R21': 0.46,
        'R21_err': 0.01,
        'R31': 0.40,
        'R31_err': 0.01
    }
    
    results = fit_peak_ratios_to_data(planck_peaks=planck_peaks)
    
    # Check all required keys are present
    required_keys = ['beta_fit', 'beta_err', 'chi2_min', 'chi2_lcdm', 
                     'delta_chi2', 'ndof', 'chi2_reduced', 'n_data_points']
    
    for key in required_keys:
        assert key in results, f"Results missing key: {key}"


def test_peak_statistics_finite():
    """Test that statistics are finite when fit succeeds."""
    planck_peaks = {
        'R21': 0.46,
        'R21_err': 0.01,
        'R31': 0.40,
        'R31_err': 0.01
    }
    
    results = fit_peak_ratios_to_data(planck_peaks=planck_peaks)
    
    if np.isfinite(results['beta_fit']):
        # If fit succeeded, all statistics should be finite
        assert np.isfinite(results['beta_err']), "beta_err should be finite"
        assert np.isfinite(results['chi2_min']), "chi2_min should be finite"
        assert np.isfinite(results['chi2_lcdm']), "chi2_lcdm should be finite"
        assert np.isfinite(results['delta_chi2']), "delta_chi2 should be finite"
        assert np.isfinite(results['ndof']), "ndof should be finite"
        assert np.isfinite(results['chi2_reduced']), "chi2_reduced should be finite"
        assert results['ndof'] > 0, "ndof should be positive"
        assert results['n_data_points'] > 0, "n_data_points should be positive"


def test_peak_ndof_calculation():
    """Test that ndof is calculated correctly."""
    planck_peaks = {
        'R21': 0.46,
        'R21_err': 0.01,
        'R31': 0.40,
        'R31_err': 0.01
    }
    
    results = fit_peak_ratios_to_data(planck_peaks=planck_peaks)
    
    # With 2 data points (R21, R31) and 1 parameter (β), ndof = 2 - 1 = 1
    expected_ndof = 1
    assert results['ndof'] == expected_ndof, \
        f"ndof should be {expected_ndof} for 2 data points, got {results['ndof']}"
    
    assert results['n_data_points'] == 2, \
        f"n_data_points should be 2, got {results['n_data_points']}"


def test_peak_beta_err_positive():
    """Test that beta_err is positive when finite."""
    planck_peaks = {
        'R21': 0.46,
        'R21_err': 0.01,
        'R31': 0.40,
        'R31_err': 0.01
    }
    
    results = fit_peak_ratios_to_data(planck_peaks=planck_peaks)
    
    if np.isfinite(results['beta_err']):
        assert results['beta_err'] > 0, \
            f"beta_err should be positive, got {results['beta_err']}"

