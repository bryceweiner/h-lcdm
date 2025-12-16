"""
Unit tests for evolving G physics modules.
"""

import numpy as np
import pytest
from hlcdm.parameters import HLCDM_PARAMS
from pipeline.cmb_gw.physics.evolving_g import G_ratio, H_evolving_G, c_s_baryon_photon


class TestEvolvingGPhysics:
    """Test evolving G physics calculations."""
    
    def test_G_ratio_at_z0(self):
        """Verify G_eff(0) = G_0 regardless of β."""
        for beta in [0.0, 0.1, 0.2, 0.3]:
            ratio = G_ratio(0.0, beta)
            assert abs(ratio - 1.0) < 1e-10, f"G_ratio(0, {beta}) should equal 1.0"
    
    def test_G_ratio_at_recombination(self):
        """Verify ~5% reduction at z=1100 for β=0.2."""
        z_rec = HLCDM_PARAMS.Z_RECOMB
        beta = 0.2
        ratio = G_ratio(z_rec, beta)
        
        # At z=1100, f(z) ≈ 0.24, so G_ratio ≈ 1 - 0.2×0.24 = 0.952
        expected = 1.0 - 0.2 * 0.24  # ≈ 0.952
        assert abs(ratio - expected) < 0.01, f"G_ratio({z_rec}, {beta}) should be ~{expected}"
    
    def test_G_ratio_limits(self):
        """Test limiting cases."""
        # At β=0, should always be 1.0
        z_test = 1100.0
        assert abs(G_ratio(z_test, 0.0) - 1.0) < 1e-10
        
        # At high z, f(z) → 1, so G_ratio → 1 - β
        z_high = 1e4
        beta = 0.2
        ratio_high = G_ratio(z_high, beta)
        assert abs(ratio_high - (1 - beta)) < 0.01
    
    def test_H_evolving_G_at_beta_zero(self):
        """Verify H_evolving_G(z, β=0) matches standard ΛCDM Hubble."""
        z_test = 1.0
        H_evolving = H_evolving_G(z_test, 0.0)
        H_lcdm = HLCDM_PARAMS.get_hubble_at_redshift(z_test)
        
        # Should match exactly (within numerical precision)
        assert abs(H_evolving - H_lcdm) < 1e-10
    
    def test_H_evolving_G_reduced_at_high_z(self):
        """Verify H is reduced at high z for β>0."""
        z_high = 1100.0
        H_beta0 = H_evolving_G(z_high, 0.0)
        H_beta02 = H_evolving_G(z_high, 0.2)
        
        # With weaker G, H should be smaller
        assert H_beta02 < H_beta0
    
    def test_c_s_baryon_photon(self):
        """Test sound speed calculation."""
        z_test = 1100.0
        c_s = c_s_baryon_photon(z_test)
        
        # Sound speed should be positive and less than c
        c_km_s = HLCDM_PARAMS.C / 1000.0
        assert c_s > 0
        assert c_s < c_km_s
        
        # At high z, R → 0, so c_s → c/√3
        c_s_high_z = c_s_baryon_photon(1e4)
        expected = c_km_s / np.sqrt(3.0)
        assert abs(c_s_high_z - expected) < 0.01 * c_km_s
    
    def test_array_inputs(self):
        """Test that functions handle array inputs."""
        z_array = np.array([0.0, 1.0, 10.0, 100.0])
        beta = 0.2
        
        ratios = G_ratio(z_array, beta)
        assert len(ratios) == len(z_array)
        assert all(np.isfinite(ratios))
        
        H_array = H_evolving_G(z_array, beta)
        assert len(H_array) == len(z_array)
        assert all(np.isfinite(H_array))
        
        c_s_array = c_s_baryon_photon(z_array)
        assert len(c_s_array) == len(z_array)
        assert all(np.isfinite(c_s_array))

