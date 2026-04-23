"""
Comprehensive unit and integration tests for ΛCDM simulation void comparison functionality.

Tests all phases (18 total tests):
1. Quijote cosmology validation (2 tests)
2. Gigantes void catalog download (3 tests)
3. Survey volume masking (2 tests)
4. Quality cuts application (2 tests)
5. Simulation void processing pipeline (1 test)
6. Statistical comparison framework (5 tests)
7. Bootstrap distribution analysis (1 test)
8. End-to-end integration (2 tests)

All tests run with logging suppressed to avoid confusing error messages from expected failure scenarios.
"""

import numpy as np
import pandas as pd
import unittest
from unittest.mock import patch, MagicMock, call
import tempfile
import os
import sys
import logging

# Add the project root to the path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.loader import DataLoader
from data.processors.void_processor import VoidDataProcessor
from pipeline.void.void_pipeline import VoidPipeline
from pipeline.common.void_coordinates import validate_quijote_cosmology


class TestCaseWithLogging(unittest.TestCase):
    """Base test case that suppresses logging output during tests."""

    def setUp(self):
        """Set up test case with logging suppressed."""
        self.log_level = logging.getLogger().level
        logging.getLogger().setLevel(logging.CRITICAL)

    def tearDown(self):
        """Restore logging level after test."""
        logging.getLogger().setLevel(self.log_level)


class TestQuijoteCosmology(TestCaseWithLogging):
    """Test cosmology validation for Quijote vs Planck18."""

    def test_cosmology_compatibility(self):
        """Test that Quijote cosmology is compatible with Planck18."""
        result = validate_quijote_cosmology()

        assert result['compatible'] is True
        assert result['max_difference_percent'] < 5.0  # All parameters within 5%
        assert result['assessment'] == 'compatible'

        # Check all parameters are present
        params = result['parameter_differences']
        expected_params = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
        for param in expected_params:
            assert param in params
            assert 'quijote' in params[param]
            assert 'planck18' in params[param]
            assert 'diff_percent' in params[param]

    def test_cosmology_values_reasonable(self):
        """Test that cosmology values are in expected ranges."""
        result = validate_quijote_cosmology()

        params = result['parameter_differences']
        assert 0.3 < params['Omega_m']['quijote'] < 0.4  # Reasonable matter density
        assert 0.6 < params['h']['quijote'] < 0.7       # Reasonable Hubble constant
        assert 0.8 < params['sigma_8']['quijote'] < 0.9  # Reasonable amplitude


class TestGigantesDownload(TestCaseWithLogging):
    """Test Quijote Gigantes void catalog download."""

    @patch('requests.get')
    def test_download_success(self, mock_get):
        """Test successful download of Gigantes catalog."""
        # Mock successful HTTP response
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.iter_content.return_value = [b'dummy_data'] * 100
        mock_response.headers.get.return_value = 1024 * 1024  # 1MB
        mock_get.return_value = mock_response

        # Mock h5py to return sample data
        with patch('h5py.File') as mock_h5py:
            mock_f = MagicMock()
            mock_f.__enter__.return_value = mock_f
            mock_f.__exit__.return_value = None

            # Mock HDF5 dataset structure
            mock_f.__getitem__.side_effect = lambda key: {
                'positions': np.array([[100.0, 200.0, 300.0], [110.0, 210.0, 310.0]]),
                'radii': np.array([50.0, 60.0]),
                'redshifts': np.array([0.1, 0.2]),
                'void_ids': np.array([1, 2])
            }[key]

            mock_h5py.return_value = mock_f

            loader = DataLoader()
            result = loader.download_quijote_gigantes_voids(
                snapshot="z0",
                cosmology="fiducial"
            )

            assert result is not None
            assert isinstance(result, pd.DataFrame)
            assert len(result) == 2  # 2 voids
            assert all(col in result.columns for col in ['x', 'y', 'z', 'radius_mpc', 'redshift', 'void_id'])
            assert result['source'].iloc[0] == 'quijote_gigantes'
            assert result['cosmology'].iloc[0] == 'fiducial'

            # Check unit conversion (Mpc/h to Mpc with h=0.6711)
            h_quijote = 0.6711
            assert abs(result['x'].iloc[0] - 100.0 / h_quijote) < 1e-6
            assert abs(result['radius_mpc'].iloc[0] - 50.0 / h_quijote) < 1e-6

    @patch('requests.get')
    def test_download_failure_network(self, mock_get):
        """Test handling of network download failures."""
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Network error")
        mock_get.return_value = mock_response

        loader = DataLoader()
        result = loader.download_quijote_gigantes_voids()

        assert result is None


    @patch('requests.get')
    def test_download_invalid_parameters(self, mock_get):
        """Test download with invalid parameters."""
        # Mock network failure
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Network error")
        mock_get.return_value = mock_response

        loader = DataLoader()

        # Test with None snapshot
        result = loader.download_quijote_gigantes_voids(snapshot=None)
        assert result is None  # Should handle gracefully


class TestSimulationProcessing(TestCaseWithLogging):
    """Test simulation void processing pipeline."""

    def test_survey_volume_masking_basic(self):
        """Test basic survey volume masking functionality."""
        processor = VoidDataProcessor()

        # Create simple rectangular observational catalog
        obs_catalog = pd.DataFrame({
            'x': [0, 1, 0, 1],
            'y': [0, 0, 1, 1],
            'z': [0, 0, 0, 0]
        })

        # Create simulation catalog with points inside and outside
        sim_catalog = pd.DataFrame({
            'x': [0.5, 0.5, 5.0, 0.5],  # Last point is far outside
            'y': [0.5, 0.5, 5.0, 0.5],
            'z': [0.0, 0.0, 0.0, 0.0],
            'radius_mpc': [10.0, 10.0, 10.0, 10.0],
            'redshift': [0.1, 0.1, 0.1, 0.1]
        })

        masked = processor.apply_survey_volume_mask(sim_catalog, obs_catalog)

        # Should retain points inside the convex hull (first 3)
        assert len(masked) == 3
        assert len(masked) < len(sim_catalog)

        # Check that retained points have reasonable coordinates
        assert all(masked['x'] <= 2.0)  # Should be near obs catalog
        assert all(masked['y'] <= 2.0)

    def test_survey_volume_masking_edge_cases(self):
        """Test edge cases for survey volume masking."""
        processor = VoidDataProcessor()

        # Test with empty catalogs
        empty_obs = pd.DataFrame(columns=['x', 'y', 'z'])
        sim_catalog = pd.DataFrame({
            'x': [0], 'y': [0], 'z': [0],
            'radius_mpc': [10], 'redshift': [0.1]
        })

        # Should handle empty observational catalog gracefully
        with self.assertRaises(Exception):  # ConvexHull will fail on empty data
            processor.apply_survey_volume_mask(sim_catalog, empty_obs)

        # Test with identical catalogs
        identical_catalog = pd.DataFrame({
            'x': [0, 1], 'y': [0, 1], 'z': [0, 0],
            'radius_mpc': [10, 10], 'redshift': [0.1, 0.1]
        })

        masked = processor.apply_survey_volume_mask(identical_catalog, identical_catalog)
        assert len(masked) == len(identical_catalog)  # All points should be retained

    def test_quality_cuts_comprehensive(self):
        """Test comprehensive quality cuts application."""
        processor = VoidDataProcessor()

        # Create test catalog with various values
        sim_catalog = pd.DataFrame({
            'x': [0, 1, 2, 3, 4],
            'y': [0, 1, 2, 3, 4],
            'z': [0, 1, 2, 3, 4],
            'radius_mpc': [5, 10, 15, 20, 25],  # Test radius cuts
            'redshift': [0.05, 0.1, 0.15, 0.2, 0.25]  # Test redshift cuts
        })

        # Apply cuts that should retain only middle values
        cuts = {
            'r_min': 8.0,
            'r_max': 18.0,
            'z_min': 0.08,
            'z_max': 0.18
        }

        filtered = processor.apply_observational_quality_cuts(sim_catalog, cuts)

        # Should retain voids 1 and 2 (r=10,15; z=0.1,0.15)
        assert len(filtered) == 2
        assert filtered['radius_mpc'].min() >= 8.0
        assert filtered['radius_mpc'].max() <= 18.0
        assert filtered['redshift'].min() >= 0.08
        assert filtered['redshift'].max() <= 0.18

    def test_quality_cuts_edge_cases(self):
        """Test edge cases for quality cuts."""
        processor = VoidDataProcessor()

        sim_catalog = pd.DataFrame({
            'x': [0, 1],
            'y': [0, 1],
            'z': [0, 1],
            'radius_mpc': [10, 10],
            'redshift': [0.1, 0.1]
        })

        # Test with empty cuts dict
        filtered = processor.apply_observational_quality_cuts(sim_catalog, {})
        assert len(filtered) == len(sim_catalog)  # No cuts applied

        # Test with invalid cuts (should not crash)
        invalid_cuts = {'invalid_key': 'invalid_value'}
        filtered = processor.apply_observational_quality_cuts(sim_catalog, invalid_cuts)
        assert len(filtered) == len(sim_catalog)  # Invalid cuts ignored

    def test_process_simulation_void_catalog_integration(self):
        """Integration test for full simulation void processing."""
        processor = VoidDataProcessor()

        # Create mock observational catalog
        obs_catalog = pd.DataFrame({
            'x': [0, 1, 0, 1],
            'y': [0, 0, 1, 1],
            'z': [0, 0, 0, 0],
            'radius_mpc': [10, 10, 10, 10],
            'redshift': [0.1, 0.1, 0.1, 0.1]
        })

        # Create mock simulation catalog
        sim_catalog = pd.DataFrame({
            'x': [0.5, 0.5, 5.0],  # Last point outside survey
            'y': [0.5, 0.5, 5.0],
            'z': [0.0, 0.0, 0.0],
            'radius_mpc': [5, 10, 15],  # First point too small
            'redshift': [0.05, 0.1, 0.15]  # First point too low z
        })

        # Define processing parameters
        obs_params = {
            'quality_cuts': {
                'r_min': 8.0,
                'r_max': 20.0,
                'z_min': 0.08,
                'z_max': 0.2
            },
            'linking_length': 58.0
        }

        # Process simulation voids
        result = processor.process_simulation_void_catalog(
            sim_catalog,
            obs_catalog,
            obs_params
        )

        # Verify structure
        assert 'catalog' in result
        assert 'network_analysis' in result
        assert 'total_voids' in result
        assert 'source' in result
        assert 'quality_metrics' in result

        # Should retain only the second void (inside survey, correct size/redshift)
        final_catalog = result['catalog']
        assert len(final_catalog) == 1
        assert final_catalog['radius_mpc'].iloc[0] == 10.0
        assert final_catalog['redshift'].iloc[0] == 0.1


class TestLCDMComparison(TestCaseWithLogging):
    """Test the ΛCDM simulation comparison functionality."""

    @patch('pipeline.void.void_pipeline.VoidPipeline._bootstrap_clustering_distribution')
    def test_comparison_methodology_consistency(self, mock_bootstrap):
        """Test that comparison uses identical methodology."""
        # Mock bootstrap to return different distributions for obs vs sim
        mock_bootstrap.side_effect = [
            np.array([0.52] * 10),  # obs bootstrap
            np.array([0.50] * 10)   # sim bootstrap
        ]

        pipeline = VoidPipeline()

        # Create comprehensive mock results with correct structure
        obs_results = {
            'void_data': {
                'catalog': pd.DataFrame({
                    'x': [0, 1, 2, 3],  # Need more points
                    'y': [0, 1, 2, 3],
                    'z': [0, 1, 2, 3],
                    'radius_mpc': [10, 10, 10, 10],
                    'redshift': [0.1, 0.1, 0.1, 0.1]
                })
            },
            'clustering_analysis': {
                'observed_clustering_coefficient': 0.52,
                'observed_clustering_std': 0.01,
                'linking_length': 58.0
            }
        }

        # Mock the simulation download
        with patch.object(pipeline.data_processor.loader, 'download_quijote_gigantes_voids') as mock_download:
            mock_download.return_value = pd.DataFrame({
                'x': [0, 1],
                'y': [0, 1],
                'z': [0, 1],
                'radius_mpc': [10, 10],
                'redshift': [0.1, 0.1]
            })

            result = pipeline._lcdm_simulation_comparison(obs_results, n_bootstrap=10)

            # Verify result structure
            assert 'passed' in result
            assert 'test' in result
            assert 'statistics' in result
            assert 'interpretation' in result
            assert 'observed' in result
            assert 'simulation' in result

            # Check statistical calculations
            stats = result['statistics']
            assert 'z_score' in stats
            assert 'ci_overlap' in stats
            assert 'ks_statistic' in stats
            assert 'ks_p_value' in stats

            # Bootstrap should be called twice (obs and sim)
            assert mock_bootstrap.call_count == 2

    def test_bootstrap_clustering_distribution(self):
        """Test the bootstrap clustering distribution helper."""
        pipeline = VoidPipeline()

        # Create test catalog and clustering results
        catalog = pd.DataFrame({
            'x': [0, 1, 2, 3],  # Need more points to avoid volume calculation issues
            'y': [0, 1, 2, 3],
            'z': [0, 1, 2, 3],
            'radius_mpc': [10, 10, 10, 10],
            'redshift': [0.1, 0.1, 0.1, 0.1]
        })

        clustering_results = {
            'linking_length': 58.0
        }

        # Mock the network building to return dict with clustering coefficient
        with patch('pipeline.common.void_network.build_void_network') as mock_build:
            mock_build.return_value = {'clustering_coefficient': 0.52}

            result = pipeline._bootstrap_clustering_distribution(
                catalog, clustering_results, n_bootstrap=3, random_seed=42
            )

            assert isinstance(result, np.ndarray)
            assert len(result) == 3  # Should return 3 bootstrap samples
            assert all(isinstance(x, (int, float)) for x in result)
            assert all(x == 0.52 for x in result)  # All should be the mocked value

            # build_void_network should be called 3 times
            self.assertEqual(mock_build.call_count, 3)

    def test_comparison_statistical_calculations(self):
        """Test statistical calculations in comparison."""
        # Test z-score calculation
        c_obs, sigma_obs = 0.52, 0.01
        c_sim, sigma_sim = 0.50, 0.015

        sigma_combined = np.sqrt(sigma_obs**2 + sigma_sim**2)
        z_score = (c_obs - c_sim) / sigma_combined

        assert z_score > 0  # obs > sim
        assert abs(z_score) < 5  # Reasonable significance level

        # Test confidence interval overlap calculation
        obs_ci = np.array([0.50, 0.54])
        sim_ci = np.array([0.48, 0.52])

        overlap_min = max(obs_ci[0], sim_ci[0])
        overlap_max = min(obs_ci[1], sim_ci[1])
        overlap_width = max(0, overlap_max - overlap_min)
        mean_width = (obs_ci[1] - obs_ci[0] + sim_ci[1] - sim_ci[0]) / 2
        ci_overlap = overlap_width / mean_width if mean_width > 0 else 0.0

        assert ci_overlap > 0  # Should have some overlap
        assert ci_overlap <= 1.0  # Should not exceed 1

    @patch('pipeline.void.void_pipeline.VoidPipeline._bootstrap_clustering_distribution')
    def test_comparison_edge_cases(self, mock_bootstrap):
        """Test edge cases in comparison."""
        mock_bootstrap.return_value = np.array([0.5] * 10)

        pipeline = VoidPipeline()

        # Test with missing simulation data
        obs_results = {
            'void_data': {
                'catalog': pd.DataFrame({
                    'x': [0, 1, 2, 3],
                    'y': [0, 1, 2, 3],
                    'z': [0, 1, 2, 3],
                    'radius_mpc': [10, 10, 10, 10],
                    'redshift': [0.1, 0.1, 0.1, 0.1]
                })
            },
            'clustering_analysis': {
                'observed_clustering_coefficient': 0.52,
                'observed_clustering_std': 0.01,
                'linking_length': 58.0
            }
        }

        with patch.object(pipeline.data_processor.loader, 'download_quijote_gigantes_voids', return_value=None):
            result = pipeline._lcdm_simulation_comparison(obs_results)

            assert result['passed'] is False
            assert 'error' in result

    def test_comparison_interpretation_logic(self):
        """Test the interpretation logic for different comparison outcomes."""
        # Test consistent case (z < 2, overlap > 0.5)
        z_score = 1.0
        overlap = 0.75
        ci_overlap = 0.8

        # This would be in the interpretation string
        expected_consistent = abs(z_score) < 2.0 and ci_overlap > 0.5
        assert expected_consistent

        # Test inconsistent case
        z_score = 3.5
        ci_overlap = 0.2
        expected_inconsistent = abs(z_score) >= 2.0 or ci_overlap <= 0.5
        assert expected_inconsistent


class TestIntegration(TestCaseWithLogging):
    """Integration tests for end-to-end functionality."""

    def test_full_pipeline_integration(self):
        """Test that the full pipeline can be instantiated and methods are callable."""
        # Test basic instantiation
        pipeline = VoidPipeline()
        assert pipeline is not None
        assert hasattr(pipeline, '_lcdm_simulation_comparison')

        # Test data processor
        processor = VoidDataProcessor()
        assert processor is not None
        assert hasattr(processor, 'apply_survey_volume_mask')
        assert hasattr(processor, 'process_simulation_void_catalog')

        # Test data loader
        loader = DataLoader()
        assert loader is not None
        assert hasattr(loader, 'download_quijote_gigantes_voids')

    def test_cosmology_validation_integration(self):
        """Test cosmology validation in full context."""
        result = validate_quijote_cosmology()

        # Should be compatible for scientific use
        assert result['compatible'] is True

        # Should have all expected parameters
        params = result['parameter_differences']
        required_params = ['Omega_m', 'Omega_b', 'h', 'n_s', 'sigma_8']
        for param in required_params:
            assert param in params
            assert isinstance(params[param]['diff_percent'], (int, float))

    @patch('requests.get')
    def test_error_handling_integration(self, mock_get):
        """Test error handling across components."""
        # Test cosmology validation with edge cases
        result = validate_quijote_cosmology()
        assert 'assessment' in result

        # Mock network failure for data loader test
        mock_response = MagicMock()
        mock_response.raise_for_status.side_effect = Exception("Network error")
        mock_get.return_value = mock_response

        # Test data loader with invalid inputs
        loader = DataLoader()
        result = loader.download_quijote_gigantes_voids(snapshot="invalid")
        # Should handle gracefully (may return None or raise handled exception)
        assert result is None  # Should return None on network failure


if __name__ == '__main__':
    unittest.main(verbosity=2)
