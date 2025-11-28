"""
Unit tests for ML Pipeline
==========================

Basic unit tests for the ML pipeline components.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, patch

from pipeline.ml.ml_pipeline import MLPipeline
from pipeline.ml.anomaly_detectors import EnsembleDetector
from pipeline.ml.ensemble import EnsembleAggregator
from pipeline.ml.interpretability.lime_explainer import LIMEExplainer


class TestMLPipeline:
    """Test ML pipeline functionality."""

    @pytest.fixture
    def mock_data_loader(self):
        """Mock data loader for testing."""
        loader = Mock()
        # Mock cosmological data loading
        loader.load_cmb_data.return_value = {'mock': 'cmb_data'}
        loader.load_bao_data.return_value = {'mock': 'bao_data'}
        loader.load_void_catalog.return_value = {'mock': 'void_data'}
        loader.load_sdss_galaxy_catalog.return_value = pd.DataFrame({
            'ra': np.random.uniform(0, 360, 100),
            'dec': np.random.uniform(-30, 30, 100),
            'z': np.random.uniform(0.1, 1.0, 100),
            'r_mag': np.random.uniform(15, 21, 100)
        })
        return loader

    @patch('pipeline.ml.ml_pipeline.DataLoader')
    def test_pipeline_initialization(self, mock_dataloader_class, mock_data_loader):
        """Test pipeline initialization."""
        mock_dataloader_class.return_value = mock_data_loader

        pipeline = MLPipeline()

        assert pipeline.stage_completed['ssl_training'] == False
        assert pipeline.stage_completed['pattern_detection'] == False
        assert hasattr(pipeline, 'ssl_learner')
        assert hasattr(pipeline, 'data_loader')

    def test_encoder_dimensions(self):
        """Test encoder dimension calculation."""
        pipeline = MLPipeline()

        # Mock data with different modalities
        mock_data = {
            'cmb': {'mock': 'data'},
            'bao': {'mock': 'data'},
            'galaxy': {'mock': 'data'}
        }

        dims = pipeline._get_encoder_dimensions(mock_data)

        assert 'cmb' in dims
        assert 'bao' in dims
        assert 'galaxy' in dims
        assert dims['cmb'] == 500  # Default dimension

    def test_run_scientific_tests(self, mock_data_loader):
        """Test running scientific tests."""
        pipeline = MLPipeline()
        
        # Mock the specific test methods to avoid complex data dependency
        pipeline._run_e8_pattern_analysis = Mock(return_value={'e8_signature_detected': True})
        pipeline._run_network_analysis = Mock(return_value={'theoretical_comparison': {'consistent': True}})
        pipeline._run_chirality_analysis = Mock(return_value={'chirality_detected': False})
        pipeline._run_gamma_qtep_analysis = Mock(return_value={'pattern_detected': True})
        
        results = pipeline.run_scientific_tests()
        
        assert 'test_results' in results
        assert 'synthesis' in results
        assert results['synthesis']['strength_category'] == 'STRONG'
        assert len(results['test_results']) == 4


class TestEnsembleDetector:
    """Test ensemble anomaly detection."""

    def test_detector_initialization(self):
        """Test detector initialization."""
        detector = EnsembleDetector(input_dim=64)

        assert hasattr(detector, 'detectors')
        assert 'isolation_forest' in detector.detectors

    def test_detector_fit_predict(self):
        """Test basic fit and predict."""
        detector = EnsembleDetector(input_dim=64, methods=['isolation_forest'])

        # Generate test data
        np.random.seed(42)
        X = np.random.normal(0, 1, (100, 64))

        # Fit detector
        detector.fit(X)

        # Predict
        results = detector.predict(X)

        assert 'ensemble_scores' in results
        assert 'individual_scores' in results
        assert len(results['ensemble_scores']) == 100


class TestLIMEExplainer:
    """Test LIME interpretability."""

    def test_explainer_initialization(self):
        """Test explainer initialization."""
        def mock_predict(X):
            return np.mean(X, axis=1, keepdims=True)

        explainer = LIMEExplainer(mock_predict, n_samples=10)

        assert hasattr(explainer, 'predict_function')
        assert explainer.n_samples == 10

    def test_explanation_generation(self):
        """Test explanation generation."""
        def mock_predict(X):
            return np.sum(X, axis=1, keepdims=True)

        explainer = LIMEExplainer(mock_predict, n_samples=10)

        # Test instance
        instance = np.random.normal(0, 1, 20)

        explanation = explainer.explain_instance(instance, n_features=5)

        assert 'predicted_score' in explanation
        assert 'top_features' in explanation
        assert len(explanation['top_features']) <= 5


class TestEnsembleAggregator:
    """Test ensemble aggregation."""

    def test_aggregator_initialization(self):
        """Test aggregator initialization."""
        aggregator = EnsembleAggregator(['method1', 'method2'])

        assert len(aggregator.weights) == 2
        assert aggregator.methods == ['method1', 'method2']

    def test_score_aggregation(self):
        """Test score aggregation."""
        aggregator = EnsembleAggregator(['method1', 'method2'])

        # Mock individual scores
        individual_scores = {
            'method1': np.random.normal(0, 1, 50),
            'method2': np.random.normal(0.5, 1, 50)
        }

        results = aggregator.aggregate_scores(individual_scores)

        assert 'ensemble_scores' in results
        assert 'predictions' in results
        assert 'top_anomalies' in results
        assert len(results['ensemble_scores']) == 50


if __name__ == '__main__':
    pytest.main([__file__])
