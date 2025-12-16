"""
Tests for gravitational constant physics calculations.

The corrected formula is:
G = πc⁵/(ℏH²N_P)

where N_P = exp[2α⁻¹ + 2ln(4π²) + 1/π]

No correction factors (ln(3), f_quantum) are needed.
"""

import pytest
import numpy as np
from pipeline.gravitational_constant.physics import (
    calculate_np_from_alpha_inverse,
    calculate_g_from_holographic_bound,
    calculate_g_base,
    calculate_g_geometric,
    calculate_g_final,
    calculate_g,
)


class TestNPFromAlphaInverse:
    """Test N_P calculation from α⁻¹."""
    
    def test_np_calculation(self):
        """Test that N_P is calculated correctly from α⁻¹."""
        result = calculate_np_from_alpha_inverse(137.035999084)
        
        assert 'N_P' in result
        assert 'ln_N_P' in result
        assert result['N_P'] > 0
        assert result['ln_N_P'] > 0
    
    def test_ln_N_P_value(self):
        """Test that ln(N_P) ≈ 281.742."""
        result = calculate_np_from_alpha_inverse(137.035999084)
        
        ln_N_P = result['ln_N_P']
        expected = 281.742
        
        assert np.abs(ln_N_P - expected) < 1.0
    
    def test_N_P_order_of_magnitude(self):
        """Test that N_P ≈ 2.2 × 10^122 (corrected value)."""
        result = calculate_np_from_alpha_inverse(137.035999084)
        
        N_P = result['N_P']
        
        # Should be ~10^122, specifically around 2.2 × 10^122
        log10_N_P = np.log10(N_P)
        assert 122 < log10_N_P < 123
    
    def test_components(self):
        """Test that all components are included."""
        result = calculate_np_from_alpha_inverse(137.035999084)
        
        assert 'geometric_term' in result
        assert 'vacuum_term' in result
        assert 'constant_sum' in result


class TestGFromHolographicBound:
    """Test the main G calculation from holographic bound."""
    
    def test_g_calculation(self):
        """Test that G is calculated correctly from holographic bound."""
        result = calculate_g_from_holographic_bound()
        
        assert 'G' in result
        assert 'N_P' in result
        assert result['G'] > 0
    
    def test_g_value_order_of_magnitude(self):
        """Test that G ≈ 6.6 × 10⁻¹¹ (no corrections)."""
        result = calculate_g_from_holographic_bound()
        
        G = result['G']
        
        # Should be in the range 6-7 × 10^-11
        assert 6e-11 < G < 7e-11
    
    def test_g_agreement_with_codata(self):
        """Test that G agrees with CODATA to within ~2%."""
        result = calculate_g_from_holographic_bound()
        
        G_predicted = result['G']
        G_codata = 6.67430e-11
        
        relative_diff = abs(G_predicted - G_codata) / G_codata
        
        # Should agree within 2% (allowing for H0 uncertainty)
        assert relative_diff < 0.02
    
    def test_formula(self):
        """Test that formula is correct."""
        result = calculate_g_from_holographic_bound()
        
        assert result['formula'] == 'πc⁵/(ℏH²N_P)'


class TestGBase:
    """Test base gravitational constant calculation (backwards compatibility)."""
    
    def test_g_base_calculation(self):
        """Test that G_base is calculated correctly."""
        result = calculate_g_base()
        
        assert 'G_base' in result
        assert 'G' in result
        assert 'N_P' in result
        assert result['G_base'] > 0
    
    def test_g_base_equals_g(self):
        """Test that G_base = G (no corrections applied)."""
        result = calculate_g_base()
        
        # In the corrected formula, G_base should equal G
        assert np.abs(result['G_base'] - result['G']) < 1e-25
    
    def test_formula(self):
        """Test that formula is correct."""
        result = calculate_g_base()
        
        assert result['formula'] == 'πc⁵/(ℏH²N_P)'


class TestGGeometric:
    """Test geometric-corrected gravitational constant (deprecated)."""
    
    def test_no_correction_applied(self):
        """Test that ln(3) correction is NOT applied."""
        result = calculate_g_geometric()
        
        assert 'G_geom' in result
        assert 'G' in result
        assert 'geometric_correction_applied' in result
        
        # Correction should NOT be applied
        assert result['geometric_correction_applied'] == False
        
        # G_geom should equal G (no correction)
        assert np.abs(result['G_geom'] - result['G']) < 1e-25
    
    def test_note_explains_no_correction(self):
        """Test that note explains why correction is not needed."""
        result = calculate_g_geometric()
        
        assert 'note' in result
        assert 'holographic' in result['note'].lower()


class TestGFinal:
    """Test final gravitational constant calculation."""
    
    def test_g_final_no_correction_default(self):
        """Test that G_final uses f_quantum=1.0 by default."""
        result = calculate_g_final()
        
        assert 'G' in result
        assert 'f_quantum' in result
        assert result['f_quantum'] == 1.0
        assert result['quantum_correction_applied'] == False
    
    def test_g_final_equals_uncorrected(self):
        """Test that G_final equals uncorrected G."""
        result = calculate_g_final()
        
        assert np.abs(result['G'] - result['G_uncorrected']) < 1e-25
    
    def test_g_final_with_custom_f_quantum(self):
        """Test that custom f_quantum can still be applied if desired."""
        result = calculate_g_final(f_quantum=1.01)
        
        assert result['f_quantum'] == 1.01
        assert result['quantum_correction_applied'] == True
        
        # G should be smaller when f_quantum > 1
        assert result['G'] < result['G_uncorrected']


class TestGComplete:
    """Test complete gravitational constant calculation."""
    
    def test_g_calculation(self):
        """Test complete G calculation."""
        result = calculate_g()
        
        assert 'G' in result
        assert 'N_P' in result
        assert 'ln_N_P' in result
        assert 'formula' in result
    
    def test_g_value(self):
        """Test that G ≈ 6.6 × 10⁻¹¹ (no corrections)."""
        result = calculate_g()
        
        G = result['G']
        G_codata = 6.67430e-11
        
        # Should agree within 2%
        relative_diff = abs(G - G_codata) / G_codata
        assert relative_diff < 0.02
    
    def test_no_corrections_applied(self):
        """Test that no corrections are applied by default."""
        result = calculate_g()
        
        assert result['corrections_applied'] == False
        assert result['f_quantum'] == 1.0
    
    def test_components(self):
        """Test that all components are included."""
        result = calculate_g()
        
        assert 'components' in result
        assert 'np_calculation' in result['components']
        assert 'holographic_calculation' in result['components']
    
    def test_backwards_compatibility_keys(self):
        """Test that backwards-compatible keys are present."""
        result = calculate_g()
        
        # These should be present for compatibility
        assert 'G_base' in result
        assert 'G_geom' in result
        assert 'ln_3' in result
        
        # G_base and G_geom should equal G (no corrections)
        assert np.abs(result['G_base'] - result['G']) < 1e-25
        assert np.abs(result['G_geom'] - result['G']) < 1e-25


class TestNumericalPrecision:
    """Test numerical precision of calculations."""
    
    def test_high_precision(self):
        """Test that calculations maintain high precision."""
        np_result = calculate_np_from_alpha_inverse(137.035999084)
        g_result = calculate_g()
        
        # All should be finite and well-defined
        assert np.isfinite(np_result['N_P'])
        assert np.isfinite(np_result['ln_N_P'])
        assert np.isfinite(g_result['G'])
        
        # Values should be in expected ranges
        assert 1e120 < np_result['N_P'] < 1e125
        assert 280 < np_result['ln_N_P'] < 285
        assert 5e-11 < g_result['G'] < 8e-11
    
    def test_consistency_across_functions(self):
        """Test that different calculation paths give same result."""
        g1 = calculate_g()['G']
        g2 = calculate_g_from_holographic_bound()['G']
        g3 = calculate_g_base()['G']
        g4 = calculate_g_final()['G']
        
        # All should give the same result (within floating point precision)
        assert np.abs(g1 - g2) < 1e-25
        assert np.abs(g1 - g3) < 1e-25
        assert np.abs(g1 - g4) < 1e-25


class TestPhysicalConsistency:
    """Test physical consistency of the derivation."""
    
    def test_g_increases_with_lower_alpha(self):
        """Test that G increases when α⁻¹ decreases (N_P decreases)."""
        g_low = calculate_g(alpha_inverse=136.0)['G']
        g_high = calculate_g(alpha_inverse=138.0)['G']
        
        # G ∝ 1/N_P ∝ 1/exp(2α⁻¹), so G should decrease with higher α⁻¹
        assert g_low > g_high
    
    def test_g_increases_with_lower_H(self):
        """Test that G increases when H decreases (G ∝ H⁻²)."""
        H_low = 2.0e-18
        H_high = 2.4e-18
        
        g_low = calculate_g(H=H_low)['G']
        g_high = calculate_g(H=H_high)['G']
        
        # G ∝ H⁻², so G should be higher for lower H
        assert g_low > g_high
    
    def test_h_sensitivity(self):
        """Test that G has expected H⁻² scaling."""
        H1 = 2.0e-18
        H2 = 2.2e-18
        
        g1 = calculate_g(H=H1)['G']
        g2 = calculate_g(H=H2)['G']
        
        # G ∝ H⁻², so G1/G2 ≈ (H2/H1)²
        expected_ratio = (H2 / H1) ** 2
        actual_ratio = g1 / g2
        
        # Should agree within 1% (small deviation from pure H⁻² due to constant factors)
        assert np.abs(actual_ratio - expected_ratio) / expected_ratio < 0.01
