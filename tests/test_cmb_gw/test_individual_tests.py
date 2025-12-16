"""
Unit tests for individual test implementations.
"""

import numpy as np
import pytest
from pipeline.cmb_gw.physics.sound_horizon import sound_horizon_evolving_G, sound_horizon_lcdm
from pipeline.cmb_gw.physics.growth_factor import growth_factor_evolving_G, void_size_ratio
from pipeline.cmb_gw.physics.luminosity_distance import luminosity_distance_evolving_G, dL_residual
from pipeline.cmb_gw.physics.cmb_peaks import cmb_peak_ratios_evolving_G


class TestSoundHorizon:
    """Test sound horizon calculations."""
    
    def test_sound_horizon_increases_with_beta(self):
        """Verify r_s increases with β (weaker G → slower expansion → larger r_s)."""
        omega_b = 0.049
        r_s_beta0 = sound_horizon_evolving_G(omega_b, beta=0.0)
        r_s_beta02 = sound_horizon_evolving_G(omega_b, beta=0.2)
        
        assert r_s_beta02 > r_s_beta0
    
    def test_sound_horizon_lcdm_matches_evolving_at_beta_zero(self):
        """Verify sound_horizon_lcdm matches evolving_G at β=0."""
        omega_b = 0.049
        r_s_lcdm = sound_horizon_lcdm(omega_b)
        r_s_evolving = sound_horizon_evolving_G(omega_b, beta=0.0)
        
        assert abs(r_s_lcdm - r_s_evolving) < 1e-2  # Allow small numerical differences


class TestGrowthFactor:
    """Test growth factor calculations."""
    
    def test_growth_suppression_direction(self):
        """Verify weaker G → suppressed growth."""
        z_array = np.array([1.0, 0.5, 0.0])
        
        D_beta0 = growth_factor_evolving_G(z_array, beta=0.0)
        D_beta02 = growth_factor_evolving_G(z_array, beta=0.2)
        
        # Growth should be suppressed (smaller) at all redshifts for β>0
        assert all(D_beta02 <= D_beta0)
    
    def test_void_size_ratio(self):
        """Test void size ratio calculation."""
        z_form = 0.5
        ratio = void_size_ratio(z_form, beta=0.2)
        
        # Ratio should be > 1 for β>0 (voids larger with weaker G)
        assert ratio > 1.0
        
        # At β=0, ratio should be 1.0
        ratio_beta0 = void_size_ratio(z_form, beta=0.0)
        assert abs(ratio_beta0 - 1.0) < 1e-10


class TestLuminosityDistance:
    """Test luminosity distance calculations."""
    
    def test_dL_residual_at_beta_zero(self):
        """Verify dL_residual = 0 at β=0."""
        z_test = 1.0
        residual = dL_residual(z_test, beta=0.0)
        assert abs(residual) < 1e-10
    
    def test_dL_increases_with_redshift(self):
        """Verify d_L increases with redshift."""
        z1, z2 = 0.5, 1.0
        beta = 0.2
        
        dL1 = luminosity_distance_evolving_G(z1, beta=beta)
        dL2 = luminosity_distance_evolving_G(z2, beta=beta)
        
        assert dL2 > dL1


class TestCMBPeaks:
    """Test CMB peak ratio calculations."""
    
    def test_peak_ratios_at_beta_zero(self):
        """Verify peak ratios match ΛCDM at β=0."""
        omega_b = 0.049
        omega_c = 0.266
        
        ratios = cmb_peak_ratios_evolving_G(omega_b, omega_c, beta=0.0)
        
        assert abs(ratios['R21'] - ratios['R21_lcdm']) < 1e-10
        assert abs(ratios['R31'] - ratios['R31_lcdm']) < 1e-10
    
    def test_peak_ratios_modification_direction(self):
        """Verify peak ratios change in expected direction for β>0."""
        omega_b = 0.049
        omega_c = 0.266
        
        ratios_beta0 = cmb_peak_ratios_evolving_G(omega_b, omega_c, beta=0.0)
        ratios_beta02 = cmb_peak_ratios_evolving_G(omega_b, omega_c, beta=0.2)
        
        # R21 should increase (even peak enhanced)
        assert ratios_beta02['R21'] > ratios_beta0['R21']
        
        # R31 should decrease (odd peak suppressed)
        assert ratios_beta02['R31'] < ratios_beta0['R31']

