"""
Unit tests for ML Pipeline Components
=====================================

Detailed unit tests for DomainAdaptationTrainer and EnsembleAggregator.
"""

import pytest
import numpy as np
import torch
import torch.nn as nn
from unittest.mock import Mock, MagicMock

from pipeline.ml.domain_adapter import DomainAdaptationTrainer
from pipeline.ml.ensemble import EnsembleAggregator

# --- DomainAdaptationTrainer Tests ---

class TestDomainAdaptationTrainer:
    """Test DomainAdaptationTrainer functionality."""

    @pytest.fixture
    def mock_base_model(self):
        """Mock base SSL model."""
        model = Mock()
        # Mock encoders structure
        model.encoders = {
            'cmb': Mock(),
            'bao': Mock()
        }
        
        # Mock encode output
        def mock_encode(batch):
            return {
                modality: torch.randn(len(next(iter(batch.values()))), 10) # 10 latent dims
                for modality in batch.keys()
            }
        model.encode.side_effect = mock_encode
        return model

    def test_initialization(self, mock_base_model):
        """Test initialization."""
        trainer = DomainAdaptationTrainer(
            base_model=mock_base_model,
            n_surveys=5,
            latent_dim=10,
            adaptation_method='both'
        )
        
        assert trainer.n_surveys == 5
        assert trainer.latent_dim == 10
        assert hasattr(trainer, 'domain_discriminators')
        assert 'cmb' in trainer.domain_discriminators
        assert 'bao' in trainer.domain_discriminators
        assert isinstance(trainer.survey_embeddings, nn.Embedding)

    def test_compute_mmd_loss(self, mock_base_model):
        """Test MMD loss computation."""
        trainer = DomainAdaptationTrainer(mock_base_model, latent_dim=10)
        
        source = torch.randn(32, 10)
        target = torch.randn(32, 10)
        
        loss = trainer.compute_mmd_loss(source, target)
        
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert not torch.isnan(loss)

    def test_adapt_domains(self, mock_base_model):
        """Test domain adaptation step."""
        trainer = DomainAdaptationTrainer(
            mock_base_model, 
            n_surveys=3, 
            latent_dim=10,
            adaptation_method='mmd'
        )
        
        batch = {
            'cmb': torch.randn(10, 5),
            'bao': torch.randn(10, 3)
        }
        # Create survey IDs ensuring at least 2 per survey for MMD
        survey_ids = torch.tensor([0, 0, 0, 1, 1, 1, 2, 2, 2, 0])
        
        losses = trainer.adapt_domains(batch, survey_ids)
        
        assert 'total_adaptation' in losses
        assert 'cmb_mmd' in losses
        assert 'bao_mmd' in losses
        assert len(trainer.adaptation_losses) == 1

    def test_adaptation_metrics(self, mock_base_model):
        """Test metrics retrieval."""
        trainer = DomainAdaptationTrainer(mock_base_model, latent_dim=10)
        
        # Mock some losses
        trainer.adaptation_losses = [
            {'total_adaptation': 0.5, 'cmb_mmd': 0.3},
            {'total_adaptation': 0.4, 'cmb_mmd': 0.2}
        ]
        
        metrics = trainer.get_adaptation_metrics()
        
        assert 'average_losses' in metrics
        assert metrics['average_losses']['avg_total_adaptation'] == 0.45
        assert 'recent_adaptation_loss' in metrics

    def test_compute_adversarial_loss(self, mock_base_model):
        """Test adversarial loss computation."""
        trainer = DomainAdaptationTrainer(
            mock_base_model,
            latent_dim=10,
            adaptation_method='adv'
        )
        
        features = torch.randn(20, 10)
        domain_labels = torch.randint(0, 2, (20,))
        
        loss = trainer.compute_adversarial_loss(features, domain_labels, 'cmb')
        
        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_save_load_adaptation_state(self, mock_base_model, tmp_path):
        """Test saving and loading adaptation state."""
        trainer = DomainAdaptationTrainer(mock_base_model, latent_dim=10)
        
        # Add some losses
        trainer.adaptation_losses = [{'total_adaptation': 0.5}]
        
        # Save
        save_path = tmp_path / "adaptation_state.pt"
        trainer.save_adaptation_state(str(save_path))
        assert save_path.exists()
        
        # Load
        trainer2 = DomainAdaptationTrainer(mock_base_model, latent_dim=10)
        trainer2.load_adaptation_state(str(save_path))
        
        assert len(trainer2.adaptation_losses) == 1

    def test_survey_invariant_validator(self):
        """Test SurveyInvariantValidator."""
        from pipeline.ml.domain_adapter import SurveyInvariantValidator
        
        validator = SurveyInvariantValidator(latent_dim=10)
        
        features_by_survey = {
            'survey1': torch.randn(50, 10),
            'survey2': torch.randn(50, 10),
            'survey3': torch.randn(50, 10)
        }
        
        metrics = validator.compute_survey_invariance_metrics(features_by_survey)
        
        assert 'average_distribution_distance' in metrics
        assert 'survey_invariance_score' in metrics
        
        # Test with insufficient surveys
        metrics2 = validator.compute_survey_invariance_metrics({'survey1': torch.randn(50, 10)})
        assert 'insufficient_surveys' in metrics2

# --- EnsembleAggregator Tests ---

class TestEnsembleAggregator:
    """Test EnsembleAggregator functionality."""

    def test_initialization(self):
        """Test initialization."""
        methods = ['method1', 'method2', 'method3']
        aggregator = EnsembleAggregator(methods)
        
        assert aggregator.methods == methods
        assert len(aggregator.weights) == 3
        assert np.allclose(aggregator.weights, 1.0/3.0)

    def test_weighted_average_aggregation(self):
        """Test weighted average aggregation."""
        aggregator = EnsembleAggregator(['m1', 'm2'], aggregation_method='weighted_average')
        aggregator.weights = np.array([0.8, 0.2])
        
        scores = {
            'm1': np.array([1.0, 0.0, 0.5]),
            'm2': np.array([0.0, 1.0, 0.5])
        }
        
        result = aggregator.aggregate_scores(scores)
        ensemble_scores = result['ensemble_scores']
        
        expected = 0.8 * scores['m1'] + 0.2 * scores['m2']
        assert np.allclose(ensemble_scores, expected)

    def test_rank_aggregation(self):
        """Test rank aggregation."""
        aggregator = EnsembleAggregator(['m1', 'm2'], aggregation_method='rank_aggregation')
        
        scores = {
            'm1': np.array([0.9, 0.1, 0.5]), # Ranks: 3, 1, 2 (highest score = highest rank)
            'm2': np.array([0.8, 0.2, 0.6])  # Ranks: 3, 1, 2
        }
        
        result = aggregator.aggregate_scores(scores)
        assert 'ensemble_scores' in result
        # Higher scores should result in higher ensemble scores
        assert result['ensemble_scores'][0] > result['ensemble_scores'][1]

    def test_consensus_aggregation(self):
        """Test consensus aggregation."""
        aggregator = EnsembleAggregator(['m1', 'm2', 'm3'], 
                                      aggregation_method='consensus',
                                      consensus_threshold=0.5)
        
        scores = {
            'm1': np.array([0.9, 0.1, 0.9]), # 1, 0, 1
            'm2': np.array([0.8, 0.2, 0.1]), # 1, 0, 0
            'm3': np.array([0.6, 0.3, 0.8])  # 1, 0, 1
        }
        
        result = aggregator.aggregate_scores(scores)
        consensus = result['ensemble_scores']
        
        # Sample 0: 3/3 agreement -> 1.0
        # Sample 1: 0/3 agreement -> 0.0
        # Sample 2: 2/3 agreement -> 0.666
        
        assert np.isclose(consensus[0], 1.0)
        assert np.isclose(consensus[1], 0.0)
        assert np.isclose(consensus[2], 2/3)

    def test_evaluate_ensemble_performance(self):
        """Test performance evaluation."""
        aggregator = EnsembleAggregator(['m1'])
        
        test_scores = {'m1': np.random.rand(100)}
        
        perf = aggregator.evaluate_ensemble_performance(test_scores)
        
        assert 'ensemble_statistics' in perf
        assert 'individual_performance' in perf
        assert 'm1' in perf['individual_performance']

    def test_calibrate_thresholds(self):
        """Test threshold calibration."""
        aggregator = EnsembleAggregator(['m1'])
        
        # 10 samples, 0 to 0.9
        scores = {'m1': np.arange(10) / 10.0}
        
        # Expected rate 0.2 -> top 2 samples -> threshold should be between 0.7 and 0.8
        # Scores: 0.0, ..., 0.7, 0.8, 0.9
        # Top 20% are 0.8, 0.9. Threshold should identify them.
        
        result = aggregator.calibrate_thresholds(scores, expected_anomaly_rate=0.2)
        
        assert result['expected_anomaly_rate'] == 0.2
        assert result['actual_anomaly_rate'] == 0.2
        assert aggregator.consensus_threshold == result['new_threshold']

    def test_compute_method_correlations(self):
        """Test method correlation computation."""
        aggregator = EnsembleAggregator(['m1', 'm2', 'm3'])
        
        score_matrix = np.array([
            np.random.rand(50),
            np.random.rand(50),
            np.random.rand(50)
        ])
        
        correlations = aggregator._compute_method_correlations(score_matrix)
        
        assert 'm1_m2' in correlations
        assert 'm1_m3' in correlations
        assert 'm2_m3' in correlations

    def test_compute_ensemble_statistics(self):
        """Test ensemble statistics computation."""
        aggregator = EnsembleAggregator(['m1'])
        
        scores = np.random.rand(100)
        stats = aggregator._compute_ensemble_statistics(scores)
        
        assert 'mean_score' in stats
        assert 'std_score' in stats
        assert 'median_score' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats

    def test_learn_optimal_weights(self):
        """Test weight learning."""
        aggregator = EnsembleAggregator(['m1', 'm2'])
        
        validation_scores = {
            'm1': np.random.rand(50),
            'm2': np.random.rand(50)
        }
        
        # With ground truth
        ground_truth = np.random.randint(0, 2, 50)
        result = aggregator.learn_optimal_weights(validation_scores, ground_truth)
        
        assert 'optimal_weights' in result
        assert 'validation_score' in result
        assert len(aggregator.weights) == 2
        
        # Without ground truth (uses ensemble agreement)
        result2 = aggregator.learn_optimal_weights(validation_scores, None)
        assert 'optimal_weights' in result2

    def test_learn_optimal_weights_edge_cases(self):
        """Test weight learning edge cases."""
        aggregator = EnsembleAggregator(['m1', 'm2'])
        
        # Test with all zeros ground truth
        validation_scores = {
            'm1': np.random.rand(50),
            'm2': np.random.rand(50)
        }
        ground_truth = np.zeros(50, dtype=int)
        
        result = aggregator.learn_optimal_weights(validation_scores, ground_truth)
        assert 'optimal_weights' in result

    def test_evaluate_ensemble_performance(self):
        """Test ensemble performance evaluation."""
        aggregator = EnsembleAggregator(['m1', 'm2'])
        
        test_scores = {
            'm1': np.random.rand(100),
            'm2': np.random.rand(100)
        }
        
        ground_truth = np.random.randint(0, 2, 100)
        
        performance = aggregator.evaluate_ensemble_performance(test_scores, ground_truth)
        
        assert 'ensemble_statistics' in performance
        assert 'individual_performance' in performance
        assert 'n_detected_anomalies' in performance

    def test_evaluate_ensemble_performance_no_ground_truth(self):
        """Test ensemble performance without ground truth."""
        aggregator = EnsembleAggregator(['m1', 'm2'])
        
        test_scores = {
            'm1': np.random.rand(100),
            'm2': np.random.rand(100)
        }
        
        performance = aggregator.evaluate_ensemble_performance(test_scores, None)
        
        assert 'ensemble_statistics' in performance
        assert 'individual_performance' in performance

    def test_get_ensemble_statistics(self):
        """Test ensemble statistics computation."""
        aggregator = EnsembleAggregator(['m1'])
        
        scores = np.random.rand(100)
        stats = aggregator._compute_ensemble_statistics(scores)
        
        assert 'mean_score' in stats
        assert 'std_score' in stats
        assert 'median_score' in stats
        assert 'skewness' in stats
        assert 'kurtosis' in stats

    def test_compute_method_correlations(self):
        """Test method correlation computation."""
        aggregator = EnsembleAggregator(['m1', 'm2', 'm3'])
        
        score_matrix = np.array([
            np.random.rand(50),
            np.random.rand(50),
            np.random.rand(50)
        ])
        
        correlations = aggregator._compute_method_correlations(score_matrix)
        
        assert 'm1_m2' in correlations
        assert 'm1_m3' in correlations
        assert 'm2_m3' in correlations

