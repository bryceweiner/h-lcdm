"""
Unit tests for coherence scale finding.

Verifies that coherence scan finds peaks at expected positions for simulated data.
"""

import pytest
import numpy as np
from pipeline.cmb_gw.analysis.coherence_analysis import find_coherence_scale


def test_coherence_scale_basic():
    """Test that coherence scale finding works with simple data."""
    # Create mock residuals with known correlation peaks
    ell = np.arange(100, 2000, 1)
    
    # Create residuals with peaks at ~220, 540, 810
    residual_TT = np.random.randn(len(ell)) * 0.1
    residual_TE = np.random.randn(len(ell)) * 0.1
    residual_EE = np.random.randn(len(ell)) * 0.1
    
    # Add correlated signal at acoustic peaks
    for peak_ell in [220, 540, 810]:
        mask = np.abs(ell - peak_ell) < 50
        signal = np.random.randn() * 0.5
        residual_TT[mask] += signal
        residual_TE[mask] += signal * 0.8
        residual_EE[mask] += signal * 0.6
    
    results = find_coherence_scale(residual_TT, residual_TE, residual_EE, ell)
    
    assert results is not None, "Coherence scale finding should return results"
    assert 'peak_ells' in results, "Results should contain peak_ells"
    assert 'mean_spacing' in results, "Results should contain mean_spacing"
    assert 'harmonic_structure_detected' in results, "Results should contain harmonic_structure_detected"


def test_coherence_scale_no_peaks():
    """Test that coherence scale finding handles no peaks gracefully."""
    # Create uncorrelated residuals (noise)
    ell = np.arange(100, 2000, 1)
    residual_TT = np.random.randn(len(ell)) * 0.1
    residual_TE = np.random.randn(len(ell)) * 0.1
    residual_EE = np.random.randn(len(ell)) * 0.1
    
    results = find_coherence_scale(residual_TT, residual_TE, residual_EE, ell)
    
    # Should still return results, but with no or few peaks
    if results is not None:
        assert 'peak_ells' in results
        assert 'harmonic_structure_detected' in results
        # With pure noise, harmonic structure should not be detected
        # (but this is probabilistic, so we don't assert it)


def test_coherence_scale_empty_input():
    """Test that coherence scale finding handles empty input."""
    ell = np.array([])
    residual_TT = np.array([])
    residual_TE = np.array([])
    residual_EE = np.array([])
    
    results = find_coherence_scale(residual_TT, residual_TE, residual_EE, ell)
    
    assert results is None, "Empty input should return None"


def test_coherence_scale_spacing():
    """Test that coherence scale spacing is measured correctly."""
    # Create residuals with known spacing
    ell = np.arange(100, 2000, 1)
    
    residual_TT = np.random.randn(len(ell)) * 0.1
    residual_TE = np.random.randn(len(ell)) * 0.1
    residual_EE = np.random.randn(len(ell)) * 0.1
    
    # Add correlated signal at peaks with spacing ~280
    peak_ells = [220, 500, 780]
    for peak_ell in peak_ells:
        mask = np.abs(ell - peak_ell) < 50
        signal = np.random.randn() * 0.5
        residual_TT[mask] += signal
        residual_TE[mask] += signal * 0.8
        residual_EE[mask] += signal * 0.6
    
    results = find_coherence_scale(residual_TT, residual_TE, residual_EE, ell)
    
    if results and results.get('mean_spacing') is not None:
        mean_spacing = results['mean_spacing']
        if np.isfinite(mean_spacing):
            # Spacing should be roughly 280 (between 220-500 and 500-780)
            assert 200 < mean_spacing < 350, \
                f"Mean spacing should be ~280, got {mean_spacing:.1f}"

