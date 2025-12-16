"""
Unit tests for sound horizon baseline validation.

Verifies that sound_horizon_lcdm() returns ~147 Mpc with Planck 2018 parameters
using CAMB (full Boltzmann solver) for gold-standard precision.
"""

import pytest
import numpy as np
from pipeline.cmb_gw.physics.sound_horizon import (
    sound_horizon_lcdm,
    sound_horizon_camb,
    sound_horizon_evolving_G,
    z_drag_eisenstein_hu,
    CAMB_AVAILABLE
)


class TestCAMBSoundHorizon:
    """Tests for CAMB-based sound horizon calculation."""
    
    @pytest.mark.skipif(not CAMB_AVAILABLE, reason="CAMB not installed")
    def test_camb_planck_2018(self):
        """Test that CAMB returns exact Planck 2018 value."""
        result = sound_horizon_camb(
            H0=67.36,
            omega_b_h2=0.02237,
            omega_c_h2=0.1200
        )
        
        # Planck 2018: r_d = 147.09 ± 0.26 Mpc
        expected = 147.09
        tolerance = 0.003  # 0.3% (much tighter than semi-analytic)
        
        assert abs(result['r_s'] - expected) / expected < tolerance, \
            f"CAMB r_s = {result['r_s']:.2f} Mpc, expected {expected} ± {tolerance*100:.1f}%"
        
        # Also check z_drag
        # Planck 2018: z_drag = 1059.94 ± 0.30
        assert abs(result['z_drag'] - 1059.94) < 1.0, \
            f"z_drag = {result['z_drag']:.1f}, expected ~1060"
    
    @pytest.mark.skipif(not CAMB_AVAILABLE, reason="CAMB not installed")
    def test_camb_z_star(self):
        """Test that CAMB returns correct recombination redshift."""
        result = sound_horizon_camb()
        
        # Planck 2018: z_* = 1089.92 ± 0.25
        assert abs(result['z_star'] - 1089.92) < 1.0, \
            f"z_* = {result['z_star']:.1f}, expected ~1090"


class TestSoundHorizonLCDM:
    """Tests for sound_horizon_lcdm wrapper function."""
    
    def test_default_parameters(self):
        """Test sound horizon with default parameters uses CAMB if available."""
        r_s = sound_horizon_lcdm()
        
        expected = 147.09
        # With CAMB: < 0.5% error; without: ~2% error
        tolerance = 0.005 if CAMB_AVAILABLE else 0.03
        
        assert np.isfinite(r_s), "r_s must be finite"
        assert r_s > 0, "r_s must be positive"
        assert abs(r_s - expected) / expected < tolerance, \
            f"r_s = {r_s:.2f} Mpc, expected ~{expected} Mpc"
    
    def test_explicit_parameters(self):
        """Test sound horizon with explicit Planck 2018 parameters."""
        h = 0.6736
        omega_b = 0.02237 / h**2
        omega_m = (0.02237 + 0.1200) / h**2
        H0 = 67.36
        
        r_s = sound_horizon_lcdm(omega_b=omega_b, omega_m=omega_m, H0=H0)
        
        expected = 147.09
        tolerance = 0.005 if CAMB_AVAILABLE else 0.03
        
        assert abs(r_s - expected) / expected < tolerance, \
            f"r_s = {r_s:.2f} Mpc, expected ~{expected} Mpc"
    
    def test_semi_analytic_fallback(self):
        """Test semi-analytic calculation gives reasonable result."""
        h = 0.6736
        omega_b = 0.02237 / h**2
        omega_m = (0.02237 + 0.1200) / h**2
        H0 = 67.36
        
        r_s = sound_horizon_lcdm(omega_b=omega_b, omega_m=omega_m, H0=H0, use_camb=False)
        
        # Semi-analytic should be within ~3% of CAMB
        expected = 147.09
        tolerance = 0.03  # 3%
        
        assert r_s > 140 and r_s < 155, f"r_s = {r_s:.2f} Mpc out of reasonable range"
        assert abs(r_s - expected) / expected < tolerance, \
            f"Semi-analytic r_s = {r_s:.2f} Mpc, expected ~{expected} Mpc within {tolerance*100:.0f}%"


class TestEisensteinHuZDrag:
    """Tests for Eisenstein & Hu fitting formula."""
    
    def test_planck_2018_z_drag(self):
        """Test z_drag formula with Planck 2018 parameters."""
        omega_b_h2 = 0.02237
        omega_m_h2 = 0.02237 + 0.1200
        
        z_d = z_drag_eisenstein_hu(omega_b_h2, omega_m_h2)
        
        # EH98 formula gives z_drag ~ 1020 for Planck parameters
        # This is different from CAMB's z_drag ~ 1060 due to different physics
        assert 1000 < z_d < 1100, f"z_drag = {z_d:.1f} outside reasonable range"
    
    def test_z_drag_vs_omega_b(self):
        """Test that z_drag increases with baryon density."""
        omega_m_h2 = 0.14
        
        z_d_low = z_drag_eisenstein_hu(0.02, omega_m_h2)
        z_d_high = z_drag_eisenstein_hu(0.03, omega_m_h2)
        
        # Higher baryon density → later drag epoch → lower z_drag
        # (This is counterintuitive but correct per EH98)
        assert z_d_high > z_d_low, \
            f"z_drag should increase with omega_b: {z_d_low:.1f} vs {z_d_high:.1f}"


class TestEvolvingGSoundHorizon:
    """Tests for sound horizon with evolving G."""
    
    def test_beta_zero_matches_lcdm(self):
        """Test that β=0 gives same result as ΛCDM."""
        h = 0.6736
        omega_b = 0.02237 / h**2
        omega_m = (0.02237 + 0.1200) / h**2
        H0 = 67.36
        
        r_s_evolving = sound_horizon_evolving_G(omega_b, omega_m, H0, beta=0.0)
        r_s_lcdm = sound_horizon_lcdm(omega_b, omega_m, H0, use_camb=False)
        
        assert abs(r_s_evolving - r_s_lcdm) / r_s_lcdm < 0.001, \
            f"β=0 should match ΛCDM: {r_s_evolving:.2f} vs {r_s_lcdm:.2f}"
    
    def test_positive_beta_increases_r_s(self):
        """Test that positive β increases sound horizon."""
        h = 0.6736
        omega_b = 0.02237 / h**2
        omega_m = 0.315
        H0 = 67.36
        
        r_s_0 = sound_horizon_evolving_G(omega_b, omega_m, H0, beta=0.0)
        r_s_02 = sound_horizon_evolving_G(omega_b, omega_m, H0, beta=0.2)
        
        # Positive β → weaker early G → slower expansion → larger sound horizon
        assert r_s_02 > r_s_0, \
            f"β > 0 should increase r_s: {r_s_02:.2f} vs {r_s_0:.2f}"
    
    def test_scaling_with_beta(self):
        """Test that r_s scaling with β is approximately linear for small β."""
        h = 0.6736
        omega_b = 0.02237 / h**2
        omega_m = 0.315
        H0 = 67.36
        
        r_s_0 = sound_horizon_evolving_G(omega_b, omega_m, H0, beta=0.0)
        r_s_01 = sound_horizon_evolving_G(omega_b, omega_m, H0, beta=0.1)
        r_s_02 = sound_horizon_evolving_G(omega_b, omega_m, H0, beta=0.2)
        
        # For small β, r_s ≈ r_s(0) × (1 + α×β) for some coefficient α
        delta_1 = (r_s_01 - r_s_0) / r_s_0
        delta_2 = (r_s_02 - r_s_0) / r_s_0
        
        # δr_s/r_s should approximately double when β doubles
        ratio = delta_2 / delta_1
        assert 1.8 < ratio < 2.2, \
            f"r_s scaling should be approximately linear: ratio = {ratio:.2f}"

