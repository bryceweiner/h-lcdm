"""
Integration Tests for ML Pipeline
=================================

End-to-end integration tests for the complete ML pipeline.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, patch

from pipeline.ml.ml_pipeline import MLPipeline


class TestFullPipelineIntegration:
    """Integration tests for complete ML pipeline."""

    @pytest.fixture
    def mock_data_components(self):
        """Mock all data loading components."""
        # Mock data loader
        data_loader = Mock()
        data_loader.load_cmb_data.return_value = {'mock': 'cmb_data'}
        data_loader.load_bao_data.return_value = {'mock': 'bao_data'}
        data_loader.load_void_catalog.return_value = pd.DataFrame({
            'ra': np.random.uniform(0, 360, 100),
            'dec': np.random.uniform(-30, 30, 100),
            'redshift': np.random.uniform(0.01, 0.15, 100),
            'radius_mpc': np.random.lognormal(1, 0.5, 100)
        })
        data_loader.load_sdss_galaxy_catalog.return_value = pd.DataFrame({
            'ra': np.random.uniform(0, 360, 100),
            'dec': np.random.uniform(-30, 30, 100),
            'z': np.random.uniform(0.1, 1.0, 100),
            'r_mag': np.random.uniform(15, 21, 100)
        })

        # Mock mock generator
        mock_generator = Mock()
        mock_generator.generate_validation_dataset.return_value = np.random.normal(0, 1, (50, 10))

        return {
            'data_loader': data_loader,
            'mock_generator': mock_generator
        }

    @patch('pipeline.ml.ml_pipeline.DataLoader')
    @patch('pipeline.ml.ml_pipeline.MockDatasetGenerator')
    def test_pipeline_stages_execution(self, mock_generator_class, mock_dataloader_class, mock_data_components):
        """Test execution of individual pipeline stages."""
        mock_dataloader_class.return_value = mock_data_components['data_loader']
        mock_generator_class.return_value = mock_data_components['mock_generator']

        pipeline = MLPipeline()

        # Test SSL training stage
        ssl_results = pipeline.run_ssl_training()
        assert 'training_completed' in ssl_results
        assert pipeline.stage_completed['ssl_training']

        # Test pattern detection stage (should fail without prior stages)
        with pytest.raises(ValueError):
            pipeline.run_pattern_detection()

        # Reset stage completion for testing
        pipeline.stage_completed['ssl_training'] = True
        pipeline.stage_completed['domain_adaptation'] = True

        # Mock pattern detection components
        with patch.object(pipeline, '_extract_features_with_ssl', return_value=np.random.normal(0, 1, (100, 512))):
            detection_results = pipeline.run_pattern_detection()
            assert 'detection_completed' in detection_results
            assert pipeline.stage_completed['pattern_detection']

    @patch('pipeline.ml.ml_pipeline.DataLoader')
    @patch('pipeline.ml.ml_pipeline.MockDatasetGenerator')
    def test_full_pipeline_run(self, mock_generator_class, mock_dataloader_class, mock_data_components):
        """Test full pipeline execution (mocked)."""
        mock_dataloader_class.return_value = mock_data_components['data_loader']
        mock_generator_class.return_value = mock_data_components['mock_generator']

        pipeline = MLPipeline()

        # Mock all the heavy computation methods
        with patch.object(pipeline, '_load_all_cosmological_data', return_value={'cmb': {}, 'bao': {}}), \
             patch.object(pipeline, '_prepare_ssl_training_data', return_value=[{'cmb': torch.randn(4, 500).to(pipeline.device), 'bao': torch.randn(4, 10).to(pipeline.device)}]), \
             patch.object(pipeline, '_extract_features_with_ssl', return_value=np.random.normal(0, 1, (100, 512))), \
             patch.object(pipeline, 'ssl_learner', create=True) as mock_ssl, \
             patch.object(pipeline, 'domain_adapter', create=True) as mock_domain, \
             patch.object(pipeline, 'ensemble_detector', create=True) as mock_detector:

            # Mock SSL learner
            mock_ssl.train_step.return_value = {'loss': 0.5}
            mock_ssl.encode.return_value = {'cmb': torch.randn(4, 512).to(pipeline.device), 'bao': torch.randn(4, 512).to(pipeline.device)}

            # Mock domain adapter
            mock_domain.adapt_domains.return_value = {'total_adaptation': 0.1}

            # Mock ensemble detector
            mock_detector.fit.return_value = None
            mock_detector.predict.return_value = {
                'ensemble_scores': np.random.normal(0, 1, 100),
                'individual_scores': {'if': np.random.normal(0, 1, 100)}
            }

            # Run full pipeline
            results = pipeline.run(context={'stages': ['ssl', 'domain', 'detect']})

            assert 'pipeline_completed' in results
            assert 'stages_completed' in results
            assert results['stages_completed']['ssl_training']
            assert results['stages_completed']['domain_adaptation']
            assert results['stages_completed']['pattern_detection']

    def test_pipeline_error_handling(self):
        """Test pipeline error handling."""
        pipeline = MLPipeline()

        # Test running pattern detection without prerequisites
        with pytest.raises(ValueError, match="Domain adaptation must be completed"):
            pipeline.run_pattern_detection()

        # Test running interpretability without pattern detection
        with pytest.raises(ValueError, match="Pattern detection must be completed"):
            pipeline.run_interpretability()

    def test_data_loading_integration(self, mock_data_components):
        """Test data loading integration."""
        pipeline = MLPipeline()

        # Test cosmological data loading (mocked)
        with patch.object(pipeline.data_loader, 'load_cmb_data', return_value={'test': 'data'}), \
             patch.object(pipeline.data_loader, 'load_bao_data', return_value={'test': 'data'}), \
             patch.object(pipeline.data_loader, 'load_void_catalog', return_value=pd.DataFrame({'test': [1]})), \
             patch.object(pipeline.data_loader, 'load_sdss_galaxy_catalog', return_value=pd.DataFrame({'test': [1]})), \
             patch.object(pipeline.data_loader, 'load_frb_data', return_value=pd.DataFrame({'test': [1]})), \
             patch.object(pipeline.data_loader, 'load_lyman_alpha_data', return_value=pd.DataFrame({'test': [1]})), \
             patch.object(pipeline.data_loader, 'load_jwst_data', return_value=pd.DataFrame({'test': [1]})):
            
            data = pipeline._load_all_cosmological_data()
            assert 'cmb' in data
            assert 'bao' in data
            assert 'void' in data
            assert 'galaxy' in data

    def test_feature_extraction(self):
        """Test feature extraction pipeline."""
        pipeline = MLPipeline()

        # Test encoder dimension calculation
        mock_data = {'cmb': {}, 'bao': {}, 'galaxy': {}}
        dims = pipeline._get_encoder_dimensions(mock_data)

        assert isinstance(dims, dict)
        assert 'cmb' in dims
        assert dims['cmb'] > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
