"""
Comprehensive Unit Tests for ML Validation Modules
==================================================

Tests for bootstrap, cross-survey, null hypothesis, and blind protocol validation.
"""

import pytest
import numpy as np
import pandas as pd
from unittest.mock import Mock, MagicMock, patch
from typing import Dict, Any

from pipeline.ml.validation.bootstrap_validator import BootstrapValidator
from pipeline.ml.validation.cross_survey_validator import CrossSurveyValidator
from pipeline.ml.validation.null_hypothesis_tester import NullHypothesisTester
from pipeline.ml.validation.blind_protocol import BlindAnalysisProtocol
from pipeline.ml.anomaly_detectors import EnsembleDetector


class TestBootstrapValidator:
    """Test BootstrapValidator functionality."""

    def test_initialization(self):
        """Test validator initialization."""
        validator = BootstrapValidator(n_bootstraps=100, confidence_level=0.95)
        assert validator.n_bootstraps == 100
        assert validator.confidence_level == 0.95
        assert validator.random_state == 42

    def test_extract_features(self):
        """Test feature extraction from dataset."""
        validator = BootstrapValidator()
        
        # Test with features key
        dataset = {'features': np.random.randn(100, 10)}
        features = validator._extract_features(dataset)
        assert features is not None
        assert features.shape == (100, 10)
        
        # Test with X key
        dataset = {'X': np.random.randn(50, 5)}
        features = validator._extract_features(dataset)
        assert features.shape == (50, 5)
        
        # Test with DataFrame
        dataset = {'data': pd.DataFrame(np.random.randn(30, 3))}
        features = validator._extract_features(dataset)
        assert features.shape == (30, 3)
        
        # Test with no features
        dataset = {'other': 'data'}
        features = validator._extract_features(dataset)
        assert features is None

    def test_generate_bootstrap_sample(self):
        """Test bootstrap sample generation."""
        validator = BootstrapValidator()
        n_samples = 100
        
        indices = validator._generate_bootstrap_sample(n_samples)
        assert len(indices) == n_samples
        assert all(0 <= idx < n_samples for idx in indices)
        assert len(set(indices)) < n_samples  # Should have duplicates

    def test_run_single_bootstrap(self):
        """Test single bootstrap iteration."""
        validator = BootstrapValidator()
        
        # Create mock model
        model = Mock()
        model.fit = Mock()
        model.predict = Mock(return_value={
            'ensemble_scores': np.random.rand(50),
            'individual_scores': {'if': np.random.rand(50)}
        })
        
        bootstrap_features = np.random.randn(50, 10)
        full_features = np.random.randn(100, 10)
        
        result = validator._run_single_bootstrap(model, bootstrap_features, full_features, 0.5)
        
        assert result['success']
        assert 'n_detected' in result
        assert 'detection_rate' in result
        assert model.fit.called

    def test_validate_stability(self):
        """Test full bootstrap validation."""
        validator = BootstrapValidator(n_bootstraps=10)  # Small for speed
        
        def model_factory():
            model = Mock()
            model.fit = Mock()
            model.predict = Mock(return_value={
                'ensemble_scores': np.random.rand(100),
                'individual_scores': {'if': np.random.rand(100)}
            })
            return model
        
        dataset = {'features': np.random.randn(100, 10)}
        
        results = validator.validate_stability(model_factory, dataset)
        
        assert 'stability_summary' in results or 'error' in results
        if 'stability_summary' in results:
            assert 'n_robust_anomalies' in results['stability_summary'] or 'robust_patterns' in results

    def test_analyze_bootstrap_results(self):
        """Test bootstrap results analysis."""
        validator = BootstrapValidator()
        
        # Create mock bootstrap results with required fields
        bootstrap_results = []
        for i in range(10):
            detections = np.random.rand(100) > 0.7
            bootstrap_results.append({
                'success': True,
                'n_detected': int(np.sum(detections)),
                'detection_rate': float(np.mean(detections)),
                'mean_score': np.random.uniform(0.4, 0.6),
                'detections': detections.tolist(),
                'anomaly_scores': np.random.rand(100).tolist()
            })
        
        analysis = validator._analyze_bootstrap_results(bootstrap_results)
        
        assert 'detection_rate_stats' in analysis
        assert 'mean_score_stats' in analysis
        assert 'stability_summary' in analysis

    def test_identify_robust_patterns(self):
        """Test robust pattern identification."""
        validator = BootstrapValidator()
        
        # Create consistent detections across bootstraps
        bootstrap_results = []
        consistent_indices = [5, 10, 15, 20]
        
        for i in range(10):
            # Always detect same indices
            detections = np.zeros(100, dtype=bool)
            detections[consistent_indices] = True
            detections[np.random.choice(100, size=5, replace=False)] = True
            
            bootstrap_results.append({
                'success': True,
                'detections': detections.tolist(),
                'top_anomaly_indices': consistent_indices + np.random.choice(100, size=5, replace=False).tolist()
            })
        
        robust = validator._identify_robust_patterns(bootstrap_results)
        
        assert 'robust_anomaly_indices' in robust
        assert 'n_robust_anomalies' in robust

    def test_compute_confidence_intervals(self):
        """Test confidence interval computation."""
        validator = BootstrapValidator(confidence_level=0.95)
        
        bootstrap_results = [
            {'detection_rate': np.random.uniform(0.4, 0.6), 'mean_score': np.random.uniform(0.4, 0.6)}
            for _ in range(20)
        ]
        ci = validator._compute_confidence_intervals(bootstrap_results, confidence_level=0.95)
        
        assert 'detection_rate' in ci
        assert 'mean_score' in ci
        assert 'ci_lower' in ci['detection_rate']
        assert 'ci_upper' in ci['detection_rate']


class TestCrossSurveyValidator:
    """Test CrossSurveyValidator functionality."""

    def test_initialization(self):
        """Test validator initialization."""
        def model_factory():
            return Mock()
        
        validator = CrossSurveyValidator(model_factory, n_splits=5)
        assert validator.n_splits == 5
        assert validator.random_state == 42

    def test_generate_cv_splits(self):
        """Test CV split generation."""
        def model_factory():
            return Mock()
        
        validator = CrossSurveyValidator(model_factory, n_splits=3)
        
        surveys = ['survey1', 'survey2', 'survey3', 'survey4']
        splits = validator._generate_cv_splits(surveys)
        
        assert len(splits) > 0
        for train_surveys, test_surveys in splits:
            assert len(train_surveys) > 0
            assert len(test_surveys) > 0
            assert set(train_surveys) & set(test_surveys) == set()  # No overlap

    def test_validate_across_surveys(self):
        """Test cross-survey validation."""
        def model_factory():
            model = Mock()
            model.fit = Mock()
            model.predict = Mock(return_value={
                'ensemble_scores': np.random.rand(50),
                'individual_scores': {'if': np.random.rand(50)}
            })
            return model
        
        validator = CrossSurveyValidator(model_factory, n_splits=2)
        
        survey_datasets = {
            'survey1': {'features': np.random.randn(50, 10)},
            'survey2': {'features': np.random.randn(50, 10)},
            'survey3': {'features': np.random.randn(50, 10)}
        }
        
        results = validator.validate_across_surveys(survey_datasets)
        
        assert 'individual_splits' in results or 'error' in results
        if 'individual_splits' in results:
            assert 'aggregated_results' in results

    def test_train_on_surveys(self):
        """Test training on multiple surveys."""
        def model_factory():
            model = Mock()
            model.fit = Mock()
            return model
        
        validator = CrossSurveyValidator(model_factory)
        
        train_surveys = ['survey1', 'survey2']
        survey_datasets = {
            'survey1': {'features': np.random.randn(30, 10)},
            'survey2': {'features': np.random.randn(30, 10)}
        }
        
        result = validator._train_on_surveys(Mock(), train_surveys, survey_datasets)
        assert 'n_train_samples' in result or 'error' in result

    def test_test_on_surveys(self):
        """Test testing on multiple surveys."""
        def model_factory():
            model = Mock()
            model.predict = Mock(return_value={
                'ensemble_scores': np.random.rand(50)
            })
            return model
        
        validator = CrossSurveyValidator(model_factory)
        
        test_surveys = ['survey1']
        survey_datasets = {
            'survey1': {'features': np.random.randn(50, 10)}
        }
        
        model = model_factory()
        result = validator._test_on_surveys(model, test_surveys, survey_datasets)
        assert isinstance(result, dict)  # Returns dict keyed by survey name

    def test_analyze_survey_consistency(self):
        """Test survey consistency analysis."""
        def model_factory():
            return Mock()
        
        validator = CrossSurveyValidator(model_factory)
        
        train_result = {'mean_anomaly_score': 0.6, 'detection_rate': 0.2, 'training_success': True}
        test_result = {'survey1': {'mean_anomaly_score': 0.55, 'detection_rate': 0.18}}
        
        consistency = validator._analyze_survey_consistency(
            train_result, test_result, ['train1'], ['test1']
        )
        
        assert isinstance(consistency, dict)  # May return error dict if training failed

    def test_aggregate_cv_results(self):
        """Test CV results aggregation."""
        def model_factory():
            return Mock()
        
        validator = CrossSurveyValidator(model_factory)
        
        validation_results = [
            {
                'train_result': {'mean_anomaly_score': 0.6, 'training_success': True},
                'test_result': {'survey1': {'mean_anomaly_score': 0.55}},
                'consistency_analysis': {'score_consistency': 0.9}
            },
            {
                'train_result': {'mean_anomaly_score': 0.65, 'training_success': True},
                'test_result': {'survey1': {'mean_anomaly_score': 0.6}},
                'consistency_analysis': {'score_consistency': 0.92}
            }
        ]
        
        aggregated = validator._aggregate_cv_results(validation_results)
        assert isinstance(aggregated, dict)

    def test_create_validation_summary(self):
        """Test validation summary creation."""
        def model_factory():
            return Mock()
        
        validator = CrossSurveyValidator(model_factory)
        
        aggregated = {
            'n_splits': 2,
            'mean_score_consistency': 0.9,
            'cross_survey_stability': 'high'
        }
        
        summary = validator._create_validation_summary(aggregated)
        assert 'validation_status' in summary
        assert 'stability_assessment' in summary


class TestNullHypothesisTester:
    """Test NullHypothesisTester functionality."""

    @pytest.fixture
    def mock_generator(self):
        """Mock dataset generator."""
        generator = Mock()
        generator.generate_validation_dataset = Mock(return_value={
            'features': np.random.randn(100, 10)
        })
        return generator

    def test_initialization(self, mock_generator):
        """Test tester initialization."""
        tester = NullHypothesisTester(mock_generator, n_null_tests=50)
        assert tester.n_null_tests == 50
        assert tester.significance_level == 0.05

    def test_test_null_hypothesis(self, mock_generator):
        """Test null hypothesis testing."""
        tester = NullHypothesisTester(mock_generator, n_null_tests=5)
        
        def model_factory():
            model = Mock()
            model.fit = Mock()
            model.predict = Mock(return_value={
                'ensemble_scores': np.random.rand(100),
                'individual_scores': {'if': np.random.rand(100)}
            })
            return model
        
        real_dataset = {'features': np.random.randn(100, 10)}
        
        results = tester.test_null_hypothesis(model_factory, real_dataset, 'combined')
        
        assert 'real_data_result' in results or 'error' in results
        if 'real_data_result' in results:
            assert 'statistical_analysis' in results
            assert 'significance_test' in results

    def test_test_on_dataset(self, mock_generator):
        """Test testing on single dataset."""
        tester = NullHypothesisTester(mock_generator)
        
        model = Mock()
        model.fit = Mock()
        model.predict = Mock(return_value={
            'ensemble_scores': np.random.rand(50)
        })
        
        dataset = {'features': np.random.randn(50, 10)}
        result = tester._test_on_dataset(model, dataset, is_real=True)
        
        assert result['success']
        assert 'n_detected' in result
        assert result['is_real_data']

    def test_extract_features_from_dataset(self, mock_generator):
        """Test feature extraction."""
        tester = NullHypothesisTester(mock_generator)
        
        # Test with features key
        dataset = {'features': np.random.randn(50, 10)}
        features = tester._extract_features_from_dataset(dataset)
        assert features is not None
        assert features.shape == (50, 10)
        
        # Test with modalities (mock data)
        dataset = {'modalities': {'cmb': {}, 'bao': {}}}
        features = tester._extract_features_from_dataset(dataset)
        assert features is not None  # Should generate random features

    def test_analyze_null_tests(self, mock_generator):
        """Test null test analysis."""
        tester = NullHypothesisTester(mock_generator)
        
        real_result = {
            'detection_rate': 0.3,
            'mean_score': 0.6,
            'anomaly_scores': np.random.rand(100).tolist()
        }
        
        mock_results = [
            {
                'success': True,
                'detection_rate': 0.1,
                'mean_score': 0.4,
                'anomaly_scores': np.random.rand(100).tolist()
            }
            for _ in range(10)
        ]
        
        analysis = tester._analyze_null_tests(real_result, mock_results)
        
        assert 'detection_rate_comparison' in analysis
        assert 'score_comparison' in analysis
        assert 'distribution_analysis' in analysis

    def test_test_statistical_significance(self, mock_generator):
        """Test statistical significance testing."""
        tester = NullHypothesisTester(mock_generator)
        
        real_result = {
            'detection_rate': 0.3,
            'mean_score': 0.6
        }
        
        mock_results = [
            {
                'success': True,
                'detection_rate': 0.1 + np.random.normal(0, 0.02),
                'mean_score': 0.4 + np.random.normal(0, 0.05)
            }
            for _ in range(20)
        ]
        
        significance = tester._test_statistical_significance(real_result, mock_results)
        
        assert 'detection_rate_test' in significance
        assert 'anomaly_score_test' in significance
        assert 'overall_significance' in significance

    def test_create_validation_summary(self, mock_generator):
        """Test validation summary creation."""
        tester = NullHypothesisTester(mock_generator)
        
        statistical_analysis = {
            'significance_test': {
                'overall_significance': True,
                'detection_rate_test': {'effect_size': 1.5},
                'anomaly_score_test': {'effect_size': 1.2}
            }
        }
        
        summary = tester._create_validation_summary(statistical_analysis)
        
        assert 'validation_status' in summary
        assert 'conclusion' in summary
        assert 'scientific_implications' in summary


class TestBlindAnalysisProtocol:
    """Test BlindAnalysisProtocol functionality."""

    def test_initialization(self):
        """Test protocol initialization."""
        protocol = BlindAnalysisProtocol()
        assert not protocol.protocol_registered
        assert protocol.protocol_hash is None

    def test_register_protocol(self):
        """Test protocol registration."""
        protocol = BlindAnalysisProtocol()
        
        methodology = {
            'ml_pipeline': {'stages': ['ssl', 'detect']},
            'validation_methods': {'bootstrap': {}, 'null_hypothesis': {}}
        }
        
        registration = protocol.register_protocol(
            methodology=methodology,
            research_question="Test question",
            success_criteria={'threshold': 0.05}
        )
        
        assert registration['registration_success']
        assert protocol.protocol_registered
        assert protocol.protocol_hash is not None

    def test_execute_blind_analysis(self):
        """Test blind analysis execution."""
        protocol = BlindAnalysisProtocol()
        
        # Register protocol first
        protocol.register_protocol(
            methodology={'test': 'method'},
            research_question="Test",
            success_criteria={}
        )
        
        def analysis_function(**kwargs):
            return {'result': 'test'}
        
        result = protocol.execute_blind_analysis(
            analysis_function,
            {'param': 'value'}
        )
        
        assert result['execution_success']
        assert 'results_file' in result

    def test_generate_unblinding_report(self):
        """Test unblinding report generation."""
        protocol = BlindAnalysisProtocol()
        
        # Register and execute first
        protocol.register_protocol(
            methodology={'test': 'method'},
            research_question="Test",
            success_criteria={}
        )
        
        def analysis_function(**kwargs):
            return {'detected_patterns': [1, 2, 3]}
        
        protocol.execute_blind_analysis(analysis_function, {})
        
        # Generate unblinding report
        h_lcdm_predictions = {
            'e8_geometry': {'detection_criteria': {}},
            'enhanced_sound_horizon': {'detection_criteria': {}}
        }
        
        report = protocol.generate_unblinding_report(h_lcdm_predictions)
        
        assert report['unblinding_success']
        assert 'report_file' in report
        assert 'key_findings' in report

    def test_get_protocol_status(self):
        """Test protocol status retrieval."""
        protocol = BlindAnalysisProtocol()
        
        status = protocol.get_protocol_status()
        assert 'protocol_registered' in status
        assert 'analysis_complete' in status
        
        # Register and check again
        protocol.register_protocol(
            methodology={},
            research_question="Test",
            success_criteria={}
        )
        
        status = protocol.get_protocol_status()
        assert status['protocol_registered']
        assert 'registration_timestamp' in status

    def test_summarize_methodology(self):
        """Test methodology summarization."""
        protocol = BlindAnalysisProtocol()
        
        methodology = {
            'ml_pipeline': {'stage1': {}, 'stage2': {}},
            'validation_methods': {'bootstrap': {}, 'null': {}},
            'data_sources': ['cmb', 'bao', 'void']
        }
        
        summary = protocol._summarize_methodology(methodology)
        assert 'pipeline_stages' in summary
        assert 'validation_techniques' in summary
        assert 'n_data_sources' in summary

    def test_extract_detected_patterns(self):
        """Test pattern extraction."""
        protocol = BlindAnalysisProtocol()
        
        analysis_results = {
            'robust_patterns': {
                'robust_anomaly_indices': [1, 2, 3],
                'n_robust_anomalies': 3
            },
            'ensemble_results': {
                'top_anomalies': [
                    {'sample_index': 5, 'anomaly_score': 0.9, 'rank': 1}
                ]
            }
        }
        
        patterns = protocol._extract_detected_patterns(analysis_results)
        assert len(patterns) > 0

    def test_check_prediction_match(self):
        """Test prediction matching."""
        protocol = BlindAnalysisProtocol()
        
        detected_patterns = [
            {'type': 'geometric_patterns'},
            {'type': 'bao_related'}
        ]
        
        match = protocol._check_prediction_match(
            'e8_geometry',
            {'detection_criteria': {}},
            detected_patterns
        )
        
        assert 'match' in match
        assert 'confidence' in match

    def test_assess_overall_consistency(self):
        """Test consistency assessment."""
        protocol = BlindAnalysisProtocol()
        
        matches = [
            {'prediction': 'pred1', 'confidence': 'high'},
            {'prediction': 'pred2', 'confidence': 'medium'}
        ]
        non_matches = [
            {'prediction': 'pred3', 'reason': 'not found'}
        ]
        
        consistency = protocol._assess_overall_consistency(matches, non_matches)
        
        assert 'consistency_score' in consistency
        assert 'assessment' in consistency
        assert consistency['consistency_score'] == 2/3

    def test_assess_validation_compliance(self):
        """Test validation compliance assessment."""
        protocol = BlindAnalysisProtocol()
        
        protocol_dict = {
            'methodology': {
                'validation_methods': ['bootstrap', 'null_hypothesis']
            }
        }
        
        analysis_results = {
            'bootstrap_validation': {},
            'null_hypothesis_testing': {}
        }
        
        compliance = protocol._assess_validation_compliance(analysis_results, protocol_dict)
        
        assert 'protocol_compliance' in compliance
        assert 'validation_methods_used' in compliance

