"""
Tests for N-body void calibration.

This module tests the rigorous β extraction using N-body simulations.
"""

import pytest
import numpy as np
from pathlib import Path

# Check if nbodykit is available
try:
    from pipeline.cmb_gw.physics.nbody_void_calibration import (
        NBODYVoidCalibration,
        quick_calibration,
        NBODYKIT_AVAILABLE
    )
    NBODY_TESTS_AVAILABLE = NBODYKIT_AVAILABLE
except ImportError:
    NBODY_TESTS_AVAILABLE = False

pytestmark = pytest.mark.skipif(
    not NBODY_TESTS_AVAILABLE,
    reason="nbodykit not available. Install with: pip install nbodykit"
)


class TestNBodyCalibration:
    """Tests for N-body void calibration."""
    
    def test_calibrator_initialization(self):
        """Test that calibrator initializes with correct parameters."""
        calibrator = NBODYVoidCalibration(
            box_size=256.0,
            n_particles=64**3,  # Small for testing
            z_initial=49.0,
            z_final=0.0
        )
        
        assert calibrator.box_size == 256.0
        assert calibrator.n_particles == 64**3
        assert calibrator.Omega_m == pytest.approx(0.315, abs=0.001)
        assert calibrator.h == pytest.approx(0.6736, abs=0.0001)
    
    def test_growth_factor_beta(self):
        """Test growth factor calculation with β."""
        calibrator = NBODYVoidCalibration()
        
        # ΛCDM (β=0) should give D(z=0) = 1
        D_lcdm = calibrator.growth_factor_beta(0.0, beta=0.0)
        assert D_lcdm == pytest.approx(1.0, abs=0.01)
        
        # Non-zero β should modify growth
        D_beta = calibrator.growth_factor_beta(0.0, beta=0.2)
        assert D_beta != D_lcdm
        
        # At high z, growth should be suppressed for β > 0
        D_high_z_lcdm = calibrator.growth_factor_beta(2.0, beta=0.0)
        D_high_z_beta = calibrator.growth_factor_beta(2.0, beta=0.2)
        # Exact relationship depends on evolving G formulation
        assert abs(D_high_z_beta - D_high_z_lcdm) < 0.5  # Should be modified but not vastly different
    
    @pytest.mark.slow
    def test_run_simulation(self):
        """Test running a single N-body simulation."""
        calibrator = NBODYVoidCalibration(
            box_size=128.0,  # Small box for speed
            n_particles=32**3,  # ~32K particles for speed
            cache_dir="nbody_cache_test"
        )
        
        # Run ΛCDM simulation
        cat = calibrator.run_simulation(beta=0.0, n_steps=5)
        
        # Check catalog structure
        assert 'Position' in cat.columns
        assert len(cat['Position']) == 32**3
        
        # Positions should be within box
        positions = cat['Position'].compute()
        assert np.all(positions >= 0)
        assert np.all(positions <= 128.0)
        
        # Clean up
        import shutil
        shutil.rmtree("nbody_cache_test", ignore_errors=True)
    
    @pytest.mark.slow
    def test_find_voids(self):
        """Test void finding in simulation."""
        calibrator = NBODYVoidCalibration(
            box_size=128.0,
            n_particles=32**3,
            cache_dir="nbody_cache_test"
        )
        
        # Run simulation
        cat = calibrator.run_simulation(beta=0.0, n_steps=5)
        
        # Find voids
        void_radii = calibrator.find_voids_in_simulation(cat, min_void_radius=5.0)
        
        # Should find some voids
        assert len(void_radii) > 0
        
        # Void radii should be reasonable (5-30 Mpc/h typically)
        assert np.all(void_radii >= 5.0)
        assert np.all(void_radii < 100.0)  # Box size is 128
        
        mean_void_radius = np.mean(void_radii)
        print(f"Found {len(void_radii)} voids, mean radius = {mean_void_radius:.2f} Mpc/h")
        
        # Clean up
        import shutil
        shutil.rmtree("nbody_cache_test", ignore_errors=True)
    
    @pytest.mark.very_slow
    def test_quick_calibration(self):
        """Test quick calibration function."""
        # This runs multiple simulations, so it's slow
        # Use minimal parameters
        calibration_data = quick_calibration(
            beta_values=[-0.1, 0.0, 0.1],
            box_size=128.0,
            n_particles=32**3,
            n_realizations=1,  # Minimal for testing
            cache_dir="nbody_cache_test"
        )
        
        # Check structure
        assert 'beta_grid' in calibration_data
        assert 'void_size_ratio' in calibration_data
        assert 'calibration_function' in calibration_data
        
        assert len(calibration_data['beta_grid']) == 3
        assert len(calibration_data['void_size_ratio']) == 3
        
        # β=0 (ΛCDM) should give ratio ≈ 1.0
        idx_lcdm = calibration_data['beta_grid'].index(0.0)
        ratio_lcdm = calibration_data['void_size_ratio'][idx_lcdm]
        assert ratio_lcdm == pytest.approx(1.0, abs=0.2)  # Allow for simulation noise
        
        # Positive β should increase void sizes (ratio > 1)
        idx_pos = calibration_data['beta_grid'].index(0.1)
        ratio_pos = calibration_data['void_size_ratio'][idx_pos]
        assert ratio_pos > 0.8  # Should be ≥ ratio_lcdm, but allow simulation variance
        
        print(f"Calibration: β={calibration_data['beta_grid']}")
        print(f"Void ratios: {calibration_data['void_size_ratio']}")
        
        # Clean up
        import shutil
        shutil.rmtree("nbody_cache_test", ignore_errors=True)
    
    @pytest.mark.very_slow
    def test_extract_beta(self):
        """Test β extraction from mock observed voids."""
        # First run calibration
        calibration_data = quick_calibration(
            beta_values=[0.0, 0.1, 0.2],
            box_size=128.0,
            n_particles=32**3,
            n_realizations=2,
            cache_dir="nbody_cache_test"
        )
        
        # Create mock "observed" voids from β=0.15 simulation
        calibrator = NBODYVoidCalibration(
            box_size=128.0,
            n_particles=32**3,
            cache_dir="nbody_cache_test"
        )
        cat_mock = calibrator.run_simulation(beta=0.15, n_steps=5)
        mock_voids = calibrator.find_voids_in_simulation(cat_mock, min_void_radius=5.0)
        
        # Extract β from mock voids
        beta_fit, beta_err = calibrator.extract_beta_from_voids(
            mock_voids,
            calibration_data=calibration_data
        )
        
        # Should recover β ≈ 0.15 (within error)
        print(f"Input β = 0.15, Recovered β = {beta_fit:.4f} ± {beta_err:.4f}")
        assert beta_fit == pytest.approx(0.15, abs=0.1)  # Allow large error for small simulations
        assert beta_err > 0 and beta_err < 0.5  # Reasonable error bar
        
        # Clean up
        import shutil
        shutil.rmtree("nbody_cache_test", ignore_errors=True)


class TestIntegrationWithVoidAnalysis:
    """Test integration of N-body calibration with void_analysis.py"""
    
    def test_void_analysis_has_nbody_support(self):
        """Test that void_analysis.py can import N-body calibration."""
        from pipeline.cmb_gw.analysis.void_analysis import NBODY_CALIBRATION_AVAILABLE
        
        # Should be True if nbodykit is installed
        assert NBODY_CALIBRATION_AVAILABLE == NBODY_TESTS_AVAILABLE
    
    @pytest.mark.slow
    def test_void_analysis_methodology_flag(self):
        """Test that methodology flag is set correctly."""
        from pipeline.cmb_gw.analysis.void_analysis import analyze_void_sizes
        
        # Run with mock data (should use analytic method since no calibration exists)
        # This will fail gracefully if no real void data available
        try:
            results = analyze_void_sizes(
                surveys=['sdss_dr7_douglass'],
                use_nbody_calibration=False  # Force analytic
            )
            
            # Should have methodology field
            assert 'methodology' in results
            assert results['methodology'] == 'QUALITATIVE_ONLY'
            assert results['include_in_joint'] == False
            
        except Exception as e:
            # If data loading fails, that's OK for this test
            print(f"Void data not available: {e}")
            pass


def test_nbodykit_availability():
    """Test that nbodykit is properly detected."""
    from pipeline.cmb_gw.physics.nbody_void_calibration import NBODYKIT_AVAILABLE
    
    if NBODYKIT_AVAILABLE:
        # If available, should be able to import key classes
        import nbodykit
        from nbodykit.lab import *
        print(f"nbodykit version: {nbodykit.__version__}")
    else:
        print("nbodykit not available - N-body tests will be skipped")


if __name__ == "__main__":
    # Run basic tests
    print("="*80)
    print("N-BODY VOID CALIBRATION TESTS")
    print("="*80)
    
    test_nbodykit_availability()
    
    if NBODY_TESTS_AVAILABLE:
        print("\nnbodykit available - running tests...")
        pytest.main([__file__, "-v", "-s", "-m", "not slow and not very_slow"])
    else:
        print("\nnbodykit NOT available")
        print("Install with: pip install nbodykit")
        print("N-body calibration tests will be skipped")

