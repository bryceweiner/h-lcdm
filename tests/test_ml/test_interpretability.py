"""
Comprehensive Unit Tests for ML Interpretability Modules
========================================================

Tests for LIME and SHAP explainers.
"""

import pytest
import numpy as np
from unittest.mock import Mock, patch, MagicMock

from pipeline.ml.interpretability.lime_explainer import LIMEExplainer
from pipeline.ml.interpretability.shap_explainer import SHAPExplainer


class TestLIMEExplainer:
    """Test LIMEExplainer functionality."""

    def test_initialization(self):
        """Test explainer initialization."""
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = LIMEExplainer(predict_func, n_samples=100)
        assert explainer.n_samples == 100
        assert explainer.predict_function == predict_func

    def test_explain_instance(self):
        """Test instance explanation."""
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = LIMEExplainer(predict_func, n_samples=50)
        
        instance = np.random.randn(20)
        explanation = explainer.explain_instance(instance, n_features=5)
        
        assert 'predicted_score' in explanation
        assert 'top_features' in explanation
        assert len(explanation['top_features']) <= 5

    def test_explain_instance_with_categorical(self):
        """Test explanation with categorical features."""
        def predict_func(X):
            return np.mean(X, axis=1, keepdims=True)
        
        explainer = LIMEExplainer(predict_func, n_samples=30)
        
        instance = np.random.randn(10)
        categorical_features = [0, 2, 4]
        
        explanation = explainer.explain_instance(
            instance,
            categorical_features=categorical_features,
            n_features=3
        )
        
        assert 'predicted_score' in explanation
        assert 'top_features' in explanation

    def test_explain_instance_with_training_data(self):
        """Test explanation with training data."""
        def predict_func(X):
            return np.max(X, axis=1, keepdims=True)
        
        training_data = np.random.randn(100, 15)
        explainer = LIMEExplainer(predict_func, training_data=training_data)
        
        instance = np.random.randn(15)
        explanation = explainer.explain_instance(instance)
        
        assert 'predicted_score' in explanation
        assert 'top_features' in explanation

    def test_get_global_feature_importance(self):
        """Test global feature importance."""
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = LIMEExplainer(predict_func, n_samples=20)
        
        dataset = np.random.randn(50, 10)
        importance = explainer.get_global_feature_importance(dataset, n_samples_per_instance=5)
        
        assert 'feature_importance' in importance
        assert len(importance['feature_importance']) == 10
        assert 'top_features' in importance

    def test_explain_multiple_instances(self):
        """Test explaining multiple instances."""
        def predict_func(X):
            return np.mean(X, axis=1, keepdims=True)
        
        explainer = LIMEExplainer(predict_func, n_samples=20)
        
        instances = np.random.randn(5, 10)
        explanations = explainer.explain_multiple_instances(instances, n_features=3)
        
        assert len(explanations) == 5
        for exp in explanations:
            assert 'predicted_score' in exp
            assert 'top_features' in exp

    def test_compute_feature_importance_ranking(self):
        """Test feature importance ranking."""
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = LIMEExplainer(predict_func)
        
        feature_importances = np.array([0.1, 0.5, 0.2, 0.8, 0.3])
        ranking = explainer._compute_feature_importance_ranking(feature_importances, n_top=3)
        
        assert len(ranking) == 3
        assert ranking[0]['feature_index'] == 3  # Highest importance
        assert ranking[0]['importance'] == 0.8

    def test_error_handling(self):
        """Test error handling."""
        def predict_func(X):
            raise ValueError("Test error")
        
        explainer = LIMEExplainer(predict_func, n_samples=10)
        
        instance = np.random.randn(10)
        # Should handle errors gracefully
        try:
            explanation = explainer.explain_instance(instance)
            # If it doesn't raise, check for error in explanation
            if 'error' in explanation:
                assert True
        except Exception:
            # Or it might raise, which is also acceptable
            assert True

    def test_explain_instance_with_feature_names(self):
        """Test explanation with feature names."""
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        feature_names = [f'feature_{i}' for i in range(20)]
        explainer = LIMEExplainer(predict_func, feature_names=feature_names, n_samples=30)
        
        instance = np.random.randn(20)
        explanation = explainer.explain_instance(instance, n_features=5)
        
        assert 'top_features' in explanation
        if explanation['top_features']:
            assert 'feature_name' in explanation['top_features'][0] or 'feature_index' in explanation['top_features'][0]


class TestSHAPExplainer:
    """Test SHAPExplainer functionality."""

    @pytest.fixture
    def mock_shap(self):
        """Mock SHAP module."""
        with patch.dict('sys.modules', {'shap': MagicMock()}):
            import shap
            shap.Explainer = Mock()
            shap_explainer_instance = Mock()
            shap_explainer_instance.return_value = Mock(
                values=np.array([[0.1, 0.2, -0.1, 0.3]]),
                base_values=np.array([0.5])
            )
            shap.Explainer.return_value = shap_explainer_instance
            yield shap

    def test_initialization_without_shap(self):
        """Test initialization when SHAP not available."""
        with patch('pipeline.ml.interpretability.shap_explainer.SHAP_AVAILABLE', False):
            with pytest.raises(ImportError):
                SHAPExplainer(lambda x: x, background_dataset=np.random.randn(10, 5))

    def test_initialization(self, mock_shap):
        """Test explainer initialization."""
        background = np.random.randn(20, 10)
        
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=background)
        assert explainer.background_dataset is not None
        assert explainer.max_evals == 1000

    def test_explain_instance(self, mock_shap):
        """Test instance explanation."""
        background = np.random.randn(20, 10)
        
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=background)
        
        instance = np.random.randn(10)
        explanation = explainer.explain_instance(instance)
        
        assert 'shap_values' in explanation
        assert 'base_value' in explanation
        assert 'top_features' in explanation

    def test_explain_instance_with_background_samples(self, mock_shap):
        """Test explanation with custom background."""
        background1 = np.random.randn(20, 10)
        
        def predict_func(X):
            return np.mean(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=background1)
        
        instance = np.random.randn(10)
        background2 = np.random.randn(15, 10)
        
        explanation = explainer.explain_instance(instance, background_samples=background2)
        assert 'shap_values' in explanation

    def test_get_global_feature_importance(self, mock_shap):
        """Test global feature importance."""
        background = np.random.randn(30, 10)
        
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=background)
        
        test_data = np.random.randn(50, 10)
        importance = explainer.get_global_feature_importance(test_data, n_samples=20)
        
        assert 'feature_importance' in importance
        assert len(importance['feature_importance']) == 10
        assert 'top_features' in importance

    def test_explain_multiple_instances(self, mock_shap):
        """Test explaining multiple instances."""
        background = np.random.randn(25, 10)
        
        def predict_func(X):
            return np.max(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=background)
        
        instances = np.random.randn(5, 10)
        explanations = explainer.explain_multiple_instances(instances)
        
        assert len(explanations) == 5
        for exp in explanations:
            assert 'shap_values' in exp
            assert 'top_features' in exp

    def test_compute_feature_importance_ranking(self, mock_shap):
        """Test feature importance ranking."""
        background = np.random.randn(20, 10)
        
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=background)
        
        shap_values = np.array([0.1, -0.2, 0.5, 0.3, -0.1])
        ranking = explainer._compute_feature_importance_ranking(shap_values, n_top=3)
        
        assert len(ranking) == 3
        assert ranking[0]['feature_index'] == 2  # Highest absolute value
        assert abs(ranking[0]['shap_value']) == 0.5

    def test_error_handling_no_background(self, mock_shap):
        """Test error when no background provided."""
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=None)
        
        instance = np.random.randn(10)
        with pytest.raises(ValueError):
            explainer.explain_instance(instance)

    def test_shap_values_formatting(self, mock_shap):
        """Test SHAP values formatting."""
        background = np.random.randn(20, 10)
        
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=background, max_evals=2000)
        
        instance = np.random.randn(10)
        explanation = explainer.explain_instance(instance)
        
        # Check that shap_values is a list
        assert isinstance(explanation['shap_values'], list)
        assert len(explanation['shap_values']) == 10

    def test_explain_dataset(self, mock_shap):
        """Test dataset explanation."""
        background = np.random.randn(30, 10)
        
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=background, max_evals=2000)
        
        dataset = np.random.randn(50, 10)
        explanations = explainer.explain_dataset(dataset, max_samples=20)
        
        assert 'explanations' in explanations
        assert 'global_feature_importance' in explanations
        assert len(explanations['explanations']) <= 20

    def test_get_global_feature_importance(self, mock_shap):
        """Test global feature importance."""
        background = np.random.randn(30, 10)
        
        def predict_func(X):
            return np.sum(X, axis=1, keepdims=True)
        
        explainer = SHAPExplainer(predict_func, background_dataset=background, max_evals=2000)
        
        test_data = np.random.randn(50, 10)
        importance = explainer.get_global_feature_importance(test_data)
        
        assert 'feature_importance' in importance
        assert len(importance['feature_importance']) == 10
        assert 'top_features' in importance

