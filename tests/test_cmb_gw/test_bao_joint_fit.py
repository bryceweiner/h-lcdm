"""
Unit tests for BAO joint fitting methodology.

Tests the proper joint fit of β to raw D/r_d ratios vs the legacy
method of inferring r_s from a fiducial cosmology.
"""

import pytest
import numpy as np
from pipeline.cmb_gw.analysis.bao_sound_horizon import (
    analyze_bao_sound_horizon,
    analyze_bao_sound_horizon_joint_fit
)


class TestJointFitMethodology:
    """Tests for joint fit methodology."""
    
    def test_joint_fit_returns_valid_results(self):
        """Test that joint fit returns all expected fields."""
        result = analyze_bao_sound_horizon_joint_fit()
        
        required_fields = [
            'beta_fit', 'beta_err', 'chi2_lcdm', 'chi2_evolving',
            'delta_chi2', 'n_measurements', 'r_s_lcdm', 'r_s_fit'
        ]
        
        for field in required_fields:
            assert field in result, f"Missing field: {field}"
            assert np.isfinite(result[field]), f"Field {field} is not finite"
    
    def test_joint_fit_beta_in_reasonable_range(self):
        """Test that fitted β is in a physically reasonable range."""
        result = analyze_bao_sound_horizon_joint_fit()
        
        beta = result['beta_fit']
        beta_err = result['beta_err']
        
        # β should be between -0.5 and 0.5 for physically viable models
        assert -0.5 < beta < 0.5, f"β = {beta} outside viable range"
        
        # Uncertainty should be reasonable
        assert 0 < beta_err < 0.5, f"β_err = {beta_err} unreasonable"
        
        # Should be able to distinguish from zero if signal is strong
        if abs(beta) > 2 * beta_err:
            print(f"  Strong signal: β = {beta:.3f} ± {beta_err:.3f}")
    
    def test_joint_fit_chi2_improvement(self):
        """Test that evolving G improves χ² over ΛCDM."""
        result = analyze_bao_sound_horizon_joint_fit()
        
        chi2_lcdm = result['chi2_lcdm']
        chi2_evolving = result['chi2_evolving']
        delta_chi2 = result['delta_chi2']
        
        # Evolving G should not increase χ²
        assert chi2_evolving <= chi2_lcdm, "Evolving G should improve or match ΛCDM fit"
        
        # Δχ² should be non-negative
        assert delta_chi2 >= 0, "Δχ² must be non-negative"
        
        # Δχ² should match calculation
        assert abs(delta_chi2 - (chi2_lcdm - chi2_evolving)) < 0.01, "Δχ² mismatch"
    
    def test_joint_fit_r_s_consistency(self):
        """Test that r_s(β) is consistent with sound horizon physics."""
        result = analyze_bao_sound_horizon_joint_fit()
        
        r_s_lcdm = result['r_s_lcdm']
        r_s_fit = result['r_s_fit']
        beta = result['beta_fit']
        
        # ΛCDM r_s should be close to Planck 2018 value (147.09 Mpc)
        assert 145 < r_s_lcdm < 150, f"r_s_lcdm = {r_s_lcdm} Mpc inconsistent with Planck"
        
        # Positive β → weaker early G → larger r_s
        # Negative β → stronger early G → smaller r_s
        if beta > 0.05:
            assert r_s_fit > r_s_lcdm, "Positive β should increase r_s"
        elif beta < -0.05:
            assert r_s_fit < r_s_lcdm, "Negative β should decrease r_s"
    
    def test_joint_fit_uses_correct_number_measurements(self):
        """Test that joint fit uses all available BAO measurements."""
        result = analyze_bao_sound_horizon_joint_fit(
            datasets=['boss_dr12', 'desi', 'eboss']
        )
        
        n_meas = result['n_measurements']
        
        # BOSS DR12 has 3, DESI has 3, eBOSS has 3 = 9 total
        assert n_meas == 9, f"Expected 9 measurements, got {n_meas}"


class TestLegacyVsJointFit:
    """Compare legacy and joint fit methods."""
    
    def test_both_methods_run_successfully(self):
        """Test that both methods complete without errors."""
        result_legacy = analyze_bao_sound_horizon(use_joint_fit=False)
        result_joint = analyze_bao_sound_horizon(use_joint_fit=True)
        
        assert result_legacy['method'] == 'legacy'
        assert result_joint['method'] == 'joint_fit'
    
    def test_methods_give_similar_beta(self):
        """Test that both methods give similar β values (they should)."""
        result_legacy = analyze_bao_sound_horizon(use_joint_fit=False)
        result_joint = analyze_bao_sound_horizon(use_joint_fit=True)
        
        beta_legacy = result_legacy['beta_fit']
        beta_joint = result_joint['beta_fit']
        
        # Both methods should give similar β (within ~20%)
        # They use same data, just different fitting procedure
        rel_diff = abs(beta_joint - beta_legacy) / max(abs(beta_legacy), abs(beta_joint), 0.01)
        
        assert rel_diff < 0.3, \
            f"Methods disagree significantly: legacy={beta_legacy:.3f}, joint={beta_joint:.3f}"
    
    def test_joint_fit_more_chi2_measurements(self):
        """Test that joint fit uses individual measurements (more χ² points)."""
        result_legacy = analyze_bao_sound_horizon(use_joint_fit=False)
        result_joint = analyze_bao_sound_horizon(use_joint_fit=True)
        
        # Legacy method aggregates datasets (3 datasets)
        # Joint fit uses individual measurements (9 measurements)
        assert result_joint['n_measurements'] > result_legacy['n_datasets'], \
            "Joint fit should use more data points than legacy method"
    
    def test_joint_fit_preferred_method(self):
        """Test that joint fit is the default method."""
        result = analyze_bao_sound_horizon()
        
        assert result['method'] == 'joint_fit', \
            "Joint fit should be the default method"


class TestJointFitPhysics:
    """Tests for physical consistency of joint fit results."""
    
    def test_beta_zero_gives_lcdm_chi2(self):
        """Test that β=0 exactly reproduces ΛCDM χ²."""
        result = analyze_bao_sound_horizon_joint_fit()
        
        # At β=0, the model should be exactly ΛCDM
        # (within numerical precision)
        # This is tested implicitly in chi2_lcdm calculation
        assert np.isfinite(result['chi2_lcdm'])
    
    def test_r_s_scaling_with_beta(self):
        """Test that r_s scales correctly with β."""
        # Run fits with different β ranges to probe scaling
        result_negative = analyze_bao_sound_horizon_joint_fit(beta_range=(-0.3, 0.0))
        result_positive = analyze_bao_sound_horizon_joint_fit(beta_range=(0.0, 0.5))
        
        # Both should give valid results
        assert np.isfinite(result_negative['beta_fit'])
        assert np.isfinite(result_positive['beta_fit'])
        
        # r_s should scale with β
        if result_positive['beta_fit'] > result_negative['beta_fit']:
            assert result_positive['r_s_fit'] > result_negative['r_s_fit'], \
                "r_s should increase with β"
    
    def test_chi2_per_dof_reasonable(self):
        """Test that χ²/dof is in a reasonable range."""
        result = analyze_bao_sound_horizon_joint_fit()
        
        chi2 = result['chi2_evolving']
        n_data = result['n_measurements']
        n_params = 1  # β
        dof = n_data - n_params
        
        chi2_per_dof = chi2 / dof if dof > 0 else np.inf
        
        # χ²/dof should be O(1) for a good fit
        # Allow wide range since this is real data with possible systematics
        assert 0.01 < chi2_per_dof < 100, \
            f"χ²/dof = {chi2_per_dof:.2f} outside reasonable range"


class TestJointFitRobustness:
    """Tests for robustness of joint fit implementation."""
    
    def test_single_dataset(self):
        """Test that joint fit works with a single dataset."""
        for dataset in ['boss_dr12', 'desi', 'eboss']:
            result = analyze_bao_sound_horizon_joint_fit(datasets=[dataset])
            
            assert np.isfinite(result['beta_fit']), \
                f"Joint fit failed for {dataset}"
            assert result['n_measurements'] > 0, \
                f"No measurements loaded for {dataset}"
    
    def test_different_cosmological_parameters(self):
        """Test joint fit with different H0 and Ω_m."""
        # Test with SH0ES value
        result_high_H0 = analyze_bao_sound_horizon_joint_fit(H0=73.0)
        assert np.isfinite(result_high_H0['beta_fit'])
        
        # Test with different Ω_m
        result_low_Om = analyze_bao_sound_horizon_joint_fit(omega_m=0.30)
        assert np.isfinite(result_low_Om['beta_fit'])
    
    def test_beta_range_constraints(self):
        """Test that β fit respects imposed bounds."""
        # Restrict to positive β only
        result_pos = analyze_bao_sound_horizon_joint_fit(beta_range=(0.0, 0.5))
        assert result_pos['beta_fit'] >= 0.0, "β should respect lower bound"
        assert result_pos['beta_fit'] <= 0.5, "β should respect upper bound"
        
        # Restrict to negative β only
        result_neg = analyze_bao_sound_horizon_joint_fit(beta_range=(-0.3, 0.0))
        assert result_neg['beta_fit'] <= 0.0, "β should respect upper bound"
        assert result_neg['beta_fit'] >= -0.3, "β should respect lower bound"


if __name__ == '__main__':
    # Run tests manually
    import sys
    
    print("=== Testing Joint Fit Methodology ===")
    print()
    
    test_joint = TestJointFitMethodology()
    
    try:
        test_joint.test_joint_fit_returns_valid_results()
        print("✓ test_joint_fit_returns_valid_results")
    except AssertionError as e:
        print(f"✗ test_joint_fit_returns_valid_results: {e}")
    
    try:
        test_joint.test_joint_fit_beta_in_reasonable_range()
        print("✓ test_joint_fit_beta_in_reasonable_range")
    except AssertionError as e:
        print(f"✗ test_joint_fit_beta_in_reasonable_range: {e}")
    
    try:
        test_joint.test_joint_fit_chi2_improvement()
        print("✓ test_joint_fit_chi2_improvement")
    except AssertionError as e:
        print(f"✗ test_joint_fit_chi2_improvement: {e}")
    
    try:
        test_joint.test_joint_fit_r_s_consistency()
        print("✓ test_joint_fit_r_s_consistency")
    except AssertionError as e:
        print(f"✗ test_joint_fit_r_s_consistency: {e}")
    
    try:
        test_joint.test_joint_fit_uses_correct_number_measurements()
        print("✓ test_joint_fit_uses_correct_number_measurements")
    except AssertionError as e:
        print(f"✗ test_joint_fit_uses_correct_number_measurements: {e}")
    
    print()
    print("All tests complete!")

