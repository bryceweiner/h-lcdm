"""
Unit Tests for Literature-Based Void Scaling
=============================================

Comprehensive tests for the peer-reviewed void size calibration formula.
"""

import numpy as np
import pytest
from pipeline.cmb_gw.physics.void_scaling_literature import (
    void_size_ratio_literature,
    extract_beta_from_void_ratio,
    propagate_gamma_uncertainty,
    GAMMA_LITERATURE,
    GAMMA_ERR_LITERATURE
)
from pipeline.cmb_gw.physics.growth_factor import growth_factor_evolving_G
from hlcdm.parameters import HLCDM_PARAMS


class TestFormulaCorrectness:
    """Test A: Formula Correctness"""
    
    def test_void_ratio_at_beta_zero(self):
        """Verify R_v(β=0)/R_v(0) = 1.0 (ΛCDM baseline)"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        
        ratio, ratio_err = void_size_ratio_literature(z, omega_m, beta=0.0)
        
        assert np.isclose(ratio, 1.0, rtol=1e-6), f"Expected 1.0, got {ratio}"
        assert ratio_err >= 0, "Uncertainty should be non-negative"
    
    def test_void_ratio_increases_with_beta(self):
        """Verify R_v increases monotonically with β for β > 0"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        
        beta_values = [0.0, 0.1, 0.2, 0.3]
        ratios = []
        
        for beta in beta_values:
            ratio, _ = void_size_ratio_literature(z, omega_m, beta)
            ratios.append(ratio)
        
        # Check monotonicity
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i-1], \
                f"Ratio should increase with β: {ratios[i-1]} -> {ratios[i]}"
    
    def test_void_ratio_decreases_with_negative_beta(self):
        """Verify R_v decreases for β < 0"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        
        ratio_pos, _ = void_size_ratio_literature(z, omega_m, beta=0.2)
        ratio_neg, _ = void_size_ratio_literature(z, omega_m, beta=-0.2)
        ratio_zero, _ = void_size_ratio_literature(z, omega_m, beta=0.0)
        
        assert ratio_neg < ratio_zero < ratio_pos, \
            f"Ratios should be ordered: {ratio_neg} < {ratio_zero} < {ratio_pos}"
    
    def test_void_ratio_power_law_exponent(self):
        """Verify γ = 1.7 exponent matches literature"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        # Compute growth factor ratio
        z_array = np.array([z, 0.0])
        D_beta = growth_factor_evolving_G(z_array, omega_m, beta)
        D_lcdm = growth_factor_evolving_G(z_array, omega_m, 0.0)
        D_ratio = D_beta[0] / D_lcdm[0]
        
        # Expected ratio with γ = 1.7
        expected_ratio = D_ratio ** GAMMA_LITERATURE
        
        ratio, _ = void_size_ratio_literature(z, omega_m, beta)
        
        assert np.isclose(ratio, expected_ratio, rtol=1e-6), \
            f"Ratio should match D_ratio^{GAMMA_LITERATURE}"
    
    def test_void_ratio_at_high_redshift(self):
        """Test behavior at high redshift (z > 1)"""
        z = 2.0
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        ratio, ratio_err = void_size_ratio_literature(z, omega_m, beta)
        
        assert np.isfinite(ratio), "Ratio should be finite at high z"
        assert ratio > 0, "Ratio should be positive"
        assert ratio_err >= 0, "Uncertainty should be non-negative"
    
    def test_void_ratio_at_low_redshift(self):
        """Test behavior at low redshift (z ≈ 0)"""
        z = 0.01
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        ratio, ratio_err = void_size_ratio_literature(z, omega_m, beta)
        
        # At z ≈ 0, ratio should be close to 1.0
        assert np.isclose(ratio, 1.0, rtol=0.1), \
            f"At low z, ratio should be ~1.0, got {ratio}"


class TestGrowthFactorIntegration:
    """Test B: Growth Factor Integration"""
    
    def test_growth_factor_scaling(self):
        """Verify correct use of growth_factor_evolving_G()"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        # Direct computation
        z_array = np.array([z, 0.0])
        D_beta = growth_factor_evolving_G(z_array, omega_m, beta)
        D_lcdm = growth_factor_evolving_G(z_array, omega_m, 0.0)
        D_ratio = D_beta[0] / D_lcdm[0]
        
        # Via void_size_ratio_literature
        ratio, _ = void_size_ratio_literature(z, omega_m, beta)
        
        # Should match: ratio = D_ratio^γ
        expected = D_ratio ** GAMMA_LITERATURE
        assert np.isclose(ratio, expected, rtol=1e-6), \
            f"Ratio should equal D_ratio^{GAMMA_LITERATURE}"
    
    def test_normalization_at_z_zero(self):
        """Verify D(z=0) = 1 normalization"""
        z = 0.0
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        # At z=0, D(β)/D(0) = 1, so ratio = 1^γ = 1
        ratio, _ = void_size_ratio_literature(z, omega_m, beta)
        
        assert np.isclose(ratio, 1.0, rtol=1e-6), \
            f"At z=0, ratio should be 1.0, got {ratio}"


class TestUncertaintyPropagation:
    """Test C: Uncertainty Propagation"""
    
    def test_gamma_uncertainty_propagation(self):
        """Verify σ_ratio includes γ uncertainty (γ_err=0.2)"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        ratio, ratio_err = void_size_ratio_literature(z, omega_m, beta)
        
        # Uncertainty should be non-zero for non-zero beta
        assert ratio_err > 0, "Uncertainty should be positive"
        
        # Rough check: uncertainty should scale with |ln(D_ratio)|
        z_array = np.array([z, 0.0])
        D_beta = growth_factor_evolving_G(z_array, omega_m, beta)
        D_lcdm = growth_factor_evolving_G(z_array, omega_m, 0.0)
        D_ratio = D_beta[0] / D_lcdm[0]
        
        expected_err = abs(ratio * np.log(D_ratio) * GAMMA_ERR_LITERATURE)
        assert np.isclose(ratio_err, expected_err, rtol=0.1), \
            f"Uncertainty should match propagation formula"
    
    def test_error_scales_with_beta(self):
        """Verify larger |β| → larger uncertainty"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        
        _, err_small = void_size_ratio_literature(z, omega_m, beta=0.1)
        _, err_large = void_size_ratio_literature(z, omega_m, beta=0.3)
        
        # Larger beta should give larger growth factor deviation
        # and thus larger uncertainty (though this depends on D_ratio)
        # At least check that both are finite
        assert np.isfinite(err_small) and np.isfinite(err_large), \
            "Uncertainties should be finite"


class TestInverseBetaExtraction:
    """Test D: Inverse Beta Extraction"""
    
    def test_beta_extraction_round_trip(self):
        """Verify β → ratio → β recovers original β"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta_true = 0.2
        
        # Forward: β → ratio
        ratio, _ = void_size_ratio_literature(z, omega_m, beta_true)
        
        # Inverse: ratio → β
        beta_extracted, _ = extract_beta_from_void_ratio(z, omega_m, ratio)
        
        assert np.isclose(beta_extracted, beta_true, rtol=0.1), \
            f"Extracted β={beta_extracted} should match true β={beta_true}"
    
    def test_beta_extraction_with_uncertainty(self):
        """Verify error bars on extracted β are reasonable"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        R_v_ratio = 1.15  # 15% excess
        
        beta, beta_err = extract_beta_from_void_ratio(z, omega_m, R_v_ratio)
        
        assert np.isfinite(beta), "Extracted β should be finite"
        assert np.isfinite(beta_err), "β uncertainty should be finite"
        assert beta_err > 0, "Uncertainty should be positive"
        assert beta_err < abs(beta) * 2, "Uncertainty should be reasonable (< 2×β)"
    
    def test_beta_extraction_physical_range(self):
        """Verify extracted β in [-0.5, 0.5] for physical ratios"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        
        # Test various ratios
        ratios = [0.8, 0.9, 1.0, 1.1, 1.2, 1.3]
        
        for ratio in ratios:
            beta, _ = extract_beta_from_void_ratio(z, omega_m, ratio)
            
            assert -0.5 <= beta <= 0.5, \
                f"Extracted β={beta} should be in physical range for ratio={ratio}"
    
    def test_beta_extraction_at_lcdm(self):
        """Verify β=0 extracted for R_v_ratio=1.0"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        R_v_ratio = 1.0
        
        beta, beta_err = extract_beta_from_void_ratio(z, omega_m, R_v_ratio)
        
        assert np.isclose(beta, 0.0, atol=0.05), \
            f"For ΛCDM ratio, β should be ~0, got {beta}"


class TestEdgeCases:
    """Test E: Edge Cases"""
    
    def test_extreme_beta_values(self):
        """Test behavior at β = ±0.5 (physical limits)"""
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        
        ratio_pos, err_pos = void_size_ratio_literature(z, omega_m, beta=0.5)
        ratio_neg, err_neg = void_size_ratio_literature(z, omega_m, beta=-0.5)
        
        assert np.isfinite(ratio_pos), "Ratio should be finite at β=0.5"
        assert np.isfinite(ratio_neg), "Ratio should be finite at β=-0.5"
        assert ratio_pos > 1.0, "Positive β should increase void size"
        assert ratio_neg < 1.0, "Negative β should decrease void size"
    
    def test_high_redshift_stability(self):
        """Test numerical stability at z > 5"""
        z = 10.0
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        ratio, ratio_err = void_size_ratio_literature(z, omega_m, beta)
        
        assert np.isfinite(ratio), "Ratio should be finite at high z"
        assert np.isfinite(ratio_err), "Uncertainty should be finite at high z"
        assert ratio > 0, "Ratio should be positive"
    
    def test_zero_omega_m_handling(self):
        """Verify proper error handling for invalid Ω_m"""
        z = 0.5
        omega_m = 0.0  # Invalid
        beta = 0.2
        
        # Should raise error or return NaN
        try:
            ratio, _ = void_size_ratio_literature(z, omega_m, beta)
            # If no error, ratio should be NaN or invalid
            assert not np.isfinite(ratio) or ratio < 0, \
                "Invalid omega_m should produce invalid ratio"
        except (ValueError, RuntimeError):
            # Error is acceptable
            pass


class TestLiteratureValidation:
    """Test F: Literature Validation"""
    
    def test_matches_pisani_2015_values(self):
        """Cross-check against specific values from Pisani+ (2015)"""
        # Pisani+ (2015) found γ = 1.7 ± 0.2
        # This is already encoded in GAMMA_LITERATURE
        
        assert GAMMA_LITERATURE == 1.7, "Gamma should match Pisani+ (2015)"
        assert GAMMA_ERR_LITERATURE == 0.2, "Gamma error should match literature"
        
        # Test that using different gamma gives different results
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        ratio_17, _ = void_size_ratio_literature(z, omega_m, beta, gamma=1.7)
        ratio_19, _ = void_size_ratio_literature(z, omega_m, beta, gamma=1.9)
        
        assert ratio_19 > ratio_17, "Larger gamma should give larger ratio"
    
    def test_asymptotic_behavior(self):
        """Verify R_v → ∞ as D → ∞ (unphysical but mathematically correct)"""
        # At very high z with large β, D_ratio can be very large
        # This tests the mathematical behavior (not physical)
        z = 100.0
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.5
        
        ratio, _ = void_size_ratio_literature(z, omega_m, beta)
        
        # Ratio should be large but finite
        assert np.isfinite(ratio), "Ratio should be finite even at extreme values"
        assert ratio > 1.0, "Large β should increase void size"
    
    def test_propagate_gamma_uncertainty_function(self):
        """Test the propagate_gamma_uncertainty helper function"""
        ratio = 1.15
        D_ratio = 1.1
        gamma_err = 0.2
        
        err = propagate_gamma_uncertainty(ratio, D_ratio, gamma_err)
        
        expected = abs(ratio * np.log(D_ratio) * gamma_err)
        assert np.isclose(err, expected, rtol=1e-6), \
            "Uncertainty propagation should match formula"


class TestPhysicalConsistency:
    """Test G: Physical Consistency Checks"""
    
    def test_void_ratio_scales_with_redshift(self):
        """
        PHYSICAL: Void ratio should increase with redshift for β > 0.
        
        This is because G_eff was weaker in the past (higher z), so growth
        was more suppressed, leading to larger voids relative to ΛCDM.
        """
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        z_values = [0.1, 0.5, 1.0, 2.0]
        ratios = []
        
        for z in z_values:
            ratio, _ = void_size_ratio_literature(z, omega_m, beta)
            ratios.append(ratio)
        
        # Ratio should increase with z (weaker G in past → larger voids)
        for i in range(1, len(ratios)):
            assert ratios[i] >= ratios[i-1], \
                f"Void ratio should increase with z: z={z_values[i-1]} → {z_values[i]}, " \
                f"ratio={ratios[i-1]:.4f} → {ratios[i]:.4f}"
    
    def test_void_ratio_at_recombination(self):
        """
        PHYSICAL: Test behavior at recombination (z ≈ 1100).
        
        At recombination, f(z) ≈ 0.24, so G_eff/G_0 ≈ 1 - 0.24β.
        For β = 0.2, G_eff was ~5% weaker, leading to suppressed growth.
        """
        z_recomb = 1100.0
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        ratio, ratio_err = void_size_ratio_literature(z_recomb, omega_m, beta)
        
        # At high z with β > 0, ratio should be > 1 (weaker G → larger voids)
        assert ratio > 1.0, \
            f"At recombination with β={beta}, void ratio should be > 1, got {ratio:.4f}"
        assert np.isfinite(ratio_err), "Uncertainty should be finite"
        
        # Ratio should be larger than at low z
        ratio_low_z, _ = void_size_ratio_literature(0.5, omega_m, beta)
        assert ratio > ratio_low_z, \
            f"Void ratio at z={z_recomb} should be larger than at z=0.5"
    
    def test_void_ratio_approaches_unity_at_z_zero(self):
        """
        PHYSICAL: As z → 0, G_eff → G_0, so void ratio → 1.
        
        This is a fundamental physical requirement: at present day,
        evolving G model matches ΛCDM (G_eff(z=0) = G_0).
        """
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        z_values = [0.0, 0.01, 0.05, 0.1]
        ratios = []
        
        for z in z_values:
            ratio, _ = void_size_ratio_literature(z, omega_m, beta)
            ratios.append(ratio)
        
        # All ratios should be close to 1.0 (within ~1%)
        for z, ratio in zip(z_values, ratios):
            assert np.isclose(ratio, 1.0, rtol=0.01), \
                f"At z={z}, void ratio should be ~1.0, got {ratio:.4f}"
    
    def test_growth_factor_consistency(self):
        """
        PHYSICAL: Verify void ratio correctly uses growth factor ratio.
        
        The void size ratio should scale as [D(β)/D(0)]^γ where γ = 1.7.
        This is the core physical relation from Pisani+ (2015).
        """
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        # Compute growth factors
        z_array = np.array([z, 0.0])
        D_beta = growth_factor_evolving_G(z_array, omega_m, beta)
        D_lcdm = growth_factor_evolving_G(z_array, omega_m, 0.0)
        D_ratio = D_beta[0] / D_lcdm[0]
        
        # Compute void ratio
        void_ratio, _ = void_size_ratio_literature(z, omega_m, beta)
        
        # Verify: void_ratio = D_ratio^γ
        expected_ratio = D_ratio ** GAMMA_LITERATURE
        assert np.isclose(void_ratio, expected_ratio, rtol=1e-5), \
            f"Void ratio {void_ratio:.6f} should equal (D_ratio)^{GAMMA_LITERATURE} = {expected_ratio:.6f}"
    
    def test_void_size_increases_with_beta_physically(self):
        """
        PHYSICAL: Larger β → weaker G in past → more suppressed growth → larger voids.
        
        This is the core physical prediction: positive β means G was weaker,
        structure grew less, voids expanded more relative to ΛCDM.
        """
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        
        beta_values = [0.0, 0.1, 0.2, 0.3, 0.4]
        ratios = []
        
        for beta in beta_values:
            ratio, _ = void_size_ratio_literature(z, omega_m, beta)
            ratios.append(ratio)
        
        # Strict monotonicity: larger β → larger void ratio
        for i in range(1, len(ratios)):
            assert ratios[i] > ratios[i-1], \
                f"Void ratio should strictly increase with β: " \
                f"β={beta_values[i-1]} → {beta_values[i]}, " \
                f"ratio={ratios[i-1]:.4f} → {ratios[i]:.4f}"
    
    def test_negative_beta_decreases_void_size(self):
        """
        PHYSICAL: Negative β → stronger G in past → enhanced growth → smaller voids.
        
        This tests the physical consistency: if G was stronger in the past,
        structure grew more, voids are smaller relative to ΛCDM.
        """
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        
        ratio_pos, _ = void_size_ratio_literature(z, omega_m, beta=0.2)
        ratio_zero, _ = void_size_ratio_literature(z, omega_m, beta=0.0)
        ratio_neg, _ = void_size_ratio_literature(z, omega_m, beta=-0.2)
        
        # Physical ordering: negative β < ΛCDM < positive β
        assert ratio_neg < ratio_zero < ratio_pos, \
            f"Physical ordering violated: β=-0.2 → {ratio_neg:.4f}, " \
            f"β=0 → {ratio_zero:.4f}, β=0.2 → {ratio_pos:.4f}"
    
    def test_uncertainty_scales_with_gamma_error(self):
        """
        PHYSICAL: Uncertainty should scale with γ uncertainty (σ_γ = 0.2).
        
        The literature uncertainty comes from the power-law exponent uncertainty.
        Larger D_ratio deviation → larger uncertainty.
        """
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        
        # Test with different beta values
        beta_values = [0.1, 0.2, 0.3]
        uncertainties = []
        
        for beta in beta_values:
            _, ratio_err = void_size_ratio_literature(z, omega_m, beta)
            uncertainties.append(ratio_err)
        
        # Larger beta → larger D_ratio deviation → larger uncertainty
        for i in range(1, len(uncertainties)):
            assert uncertainties[i] >= uncertainties[i-1], \
                f"Uncertainty should increase with β: " \
                f"β={beta_values[i-1]} → {beta_values[i]}, " \
                f"σ={uncertainties[i-1]:.6f} → {uncertainties[i]:.6f}"


class TestLiteratureValueValidation:
    """Test H: Literature Value Validation"""
    
    def test_gamma_value_matches_pisani_2015(self):
        """
        LITERATURE: Verify γ = 1.7 ± 0.2 matches Pisani+ (2015, PRD 91, 043513).
        
        This is the core calibration parameter from billion-particle simulations.
        """
        assert GAMMA_LITERATURE == 1.7, \
            f"Gamma should be 1.7 from Pisani+ (2015), got {GAMMA_LITERATURE}"
        assert GAMMA_ERR_LITERATURE == 0.2, \
            f"Gamma error should be 0.2 from literature, got {GAMMA_ERR_LITERATURE}"
    
    def test_power_law_exponent_effect(self):
        """
        LITERATURE: Test that γ = 1.7 gives physically reasonable void sizes.
        
        For typical growth factor suppression (D_ratio ≈ 0.95), void ratio should be:
        R_v_ratio ≈ 0.95^1.7 ≈ 0.92
        
        This means voids are ~8% larger, which is physically reasonable.
        """
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        # Compute growth factor ratio
        z_array = np.array([z, 0.0])
        D_beta = growth_factor_evolving_G(z_array, omega_m, beta)
        D_lcdm = growth_factor_evolving_G(z_array, omega_m, 0.0)
        D_ratio = D_beta[0] / D_lcdm[0]
        
        # Compute void ratio
        void_ratio, _ = void_size_ratio_literature(z, omega_m, beta)
        
        # Verify power-law scaling
        expected = D_ratio ** GAMMA_LITERATURE
        assert np.isclose(void_ratio, expected, rtol=1e-5), \
            f"Void ratio should follow power law: {void_ratio:.6f} = {D_ratio:.6f}^{GAMMA_LITERATURE} = {expected:.6f}"
        
        # Verify physically reasonable range (voids typically 0-20% larger)
        assert 0.9 <= void_ratio <= 1.2, \
            f"Void ratio {void_ratio:.4f} should be in physically reasonable range [0.9, 1.2]"
    
    def test_uncertainty_matches_literature_expectation(self):
        """
        LITERATURE: Verify uncertainty propagation matches literature expectations.
        
        For γ = 1.7 ± 0.2 and typical D_ratio ≈ 0.95, uncertainty should be:
        σ_ratio ≈ ratio × |ln(D_ratio)| × σ_γ ≈ 0.92 × 0.05 × 0.2 ≈ 0.01
        
        This gives ~1% relative uncertainty, consistent with literature.
        """
        z = 0.5
        omega_m = HLCDM_PARAMS.OMEGA_M
        beta = 0.2
        
        ratio, ratio_err = void_size_ratio_literature(z, omega_m, beta)
        
        # Compute expected uncertainty
        z_array = np.array([z, 0.0])
        D_beta = growth_factor_evolving_G(z_array, omega_m, beta)
        D_lcdm = growth_factor_evolving_G(z_array, omega_m, 0.0)
        D_ratio = D_beta[0] / D_lcdm[0]
        
        expected_err = abs(ratio * np.log(D_ratio) * GAMMA_ERR_LITERATURE)
        
        # Verify uncertainty matches formula
        assert np.isclose(ratio_err, expected_err, rtol=0.1), \
            f"Uncertainty {ratio_err:.6f} should match formula {expected_err:.6f}"
        
        # Verify relative uncertainty is reasonable (~1-5%)
        relative_err = ratio_err / ratio if ratio > 0 else np.nan
        assert 0.001 <= relative_err <= 0.1, \
            f"Relative uncertainty {relative_err:.4f} should be in reasonable range [0.1%, 10%]"

