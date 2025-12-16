"""
Tests for CMB Cold Spot QTEP Pipeline
=====================================

Unit tests for the CMB Cold Spot analysis pipeline.
"""

import unittest
import numpy as np
from pathlib import Path
import tempfile
import shutil

from pipeline.cmb_cold_spot import CMBColdSpotPipeline
from pipeline.cmb_cold_spot.extraction import extract_cold_spot_region
from pipeline.cmb_cold_spot.qtep_map import generate_qtep_efficiency_map
from pipeline.cmb_cold_spot.physics import (
    test_temperature_deficit,
    test_angular_power_spectrum,
    test_spatial_correlation
)
from pipeline.cmb_cold_spot.validation import ColdSpotValidator


class TestCMBColdSpotPipeline(unittest.TestCase):
    """Test CMB Cold Spot pipeline."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.pipeline = CMBColdSpotPipeline(output_dir=self.temp_dir)
    
    def tearDown(self):
        """Clean up test fixtures."""
        shutil.rmtree(self.temp_dir)
    
    def test_pipeline_initialization(self):
        """Test pipeline initialization."""
        self.assertEqual(self.pipeline.name, "cmb_cold_spot")
        self.assertIsNotNone(self.pipeline.data_loader)
        self.assertIsNotNone(self.pipeline.validator)
    
    def test_cold_spot_extraction(self):
        """Test Cold Spot region extraction."""
        # Create mock HEALPix map
        try:
            import healpy as hp
            nside = 64
            npix = hp.nside2npix(nside)
            cmb_map = np.random.normal(0, 1, npix)
            
            cold_spot_data = extract_cold_spot_region(
                cmb_map,
                nside=nside,
                center_galactic=(209.6, -57.0),
                radius_deg=10.0
            )
            
            self.assertIn('temperature_map', cold_spot_data)
            self.assertIn('mean_deficit', cold_spot_data)
            self.assertIn('metadata', cold_spot_data)
        except ImportError:
            self.skipTest("healpy not available")
    
    def test_qtep_map_generation(self):
        """Test QTEP efficiency map generation."""
        try:
            import healpy as hp
            nside = 64
            qtep_map_result = generate_qtep_efficiency_map(nside=nside)
            
            self.assertIn('efficiency_map', qtep_map_result)
            self.assertIn('mean_efficiency', qtep_map_result)
            self.assertIn('nside', qtep_map_result)
            self.assertEqual(qtep_map_result['nside'], nside)
        except ImportError:
            self.skipTest("healpy not available")
    
    def test_temperature_deficit_test(self):
        """Test temperature deficit analysis."""
        cold_spot_data = {
            'normalized_deficit': -2.6e-5,
            'std_deficit': 1.0e-6,
            'temperature_map': np.random.normal(-2.6e-5, 1.0e-6, 100)
        }
        
        results = test_temperature_deficit(cold_spot_data)
        
        self.assertIn('observed_deficit', results)
        self.assertIn('predicted_deficit', results)
        self.assertIn('consistency_sigma', results)
        self.assertIn('p_value', results)
        self.assertIn('result', results)
    
    def test_angular_power_spectrum_test(self):
        """Test angular power spectrum analysis."""
        try:
            import healpy as hp
            nside = 64
            npix = hp.nside2npix(nside)
            cold_spot_region = np.random.normal(0, 1, npix)
            
            results = test_angular_power_spectrum(
                cold_spot_region,
                nside=nside
            )
            
            self.assertIn('discrete_feature_score', results)
            self.assertIn('gaussian_p_value', results)
            self.assertIn('result', results)
        except ImportError:
            self.skipTest("healpy not available")
    
    def test_spatial_correlation_test(self):
        """Test spatial correlation analysis."""
        try:
            import healpy as hp
            nside = 64
            npix = hp.nside2npix(nside)
            cold_spot_map = np.random.normal(0, 1, npix)
            qtep_map = np.random.normal(2.257, 0.001, npix)
            
            results = test_spatial_correlation(
                cold_spot_map,
                qtep_map
            )
            
            self.assertIn('correlation_coefficient', results)
            self.assertIn('random_location_p_value', results)
            self.assertIn('result', results)
        except ImportError:
            self.skipTest("healpy not available")
    
    def test_validator_initialization(self):
        """Test validator initialization."""
        validator = ColdSpotValidator(n_bootstrap=50)
        self.assertEqual(validator.n_bootstrap, 50)
    
    def test_cross_survey_consistency(self):
        """Test cross-survey consistency check."""
        validator = ColdSpotValidator()
        
        survey_results = {
            'ACT_DR6': {
                'test_1_temperature_deficit': {
                    'observed_deficit': -2.58e-5,
                    'observed_deficit_uncertainty': 5.0e-6
                }
            },
            'Planck_2018': {
                'test_1_temperature_deficit': {
                    'observed_deficit': -2.62e-5,
                    'observed_deficit_uncertainty': 3.5e-6
                }
            }
        }
        
        consistency = validator.cross_survey_consistency(survey_results)
        
        self.assertIn('chi_squared_per_dof', consistency)
        self.assertIn('survey_consistency_p_value', consistency)
        self.assertIn('surveys_agree', consistency)


if __name__ == '__main__':
    unittest.main()

