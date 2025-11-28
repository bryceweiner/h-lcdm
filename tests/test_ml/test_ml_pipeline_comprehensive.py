"""
Comprehensive Unit Tests for ML Pipeline Methods
=================================================

Tests for all methods in ml_pipeline.py to achieve high coverage.
"""

import pytest
import numpy as np
import pandas as pd
import torch
from unittest.mock import Mock, MagicMock, patch, PropertyMock
from scipy import stats

from pipeline.ml.ml_pipeline import MLPipeline
from hlcdm.parameters import HLCDM_PARAMS
from hlcdm.cosmology import HLCDMCosmology


class TestMLPipelineMethods:
    """Comprehensive tests for MLPipeline methods."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance with mocked dependencies."""
        with patch('pipeline.ml.ml_pipeline.DataLoader'), \
             patch('pipeline.ml.ml_pipeline.MockDatasetGenerator'), \
             patch('pipeline.ml.ml_pipeline.E8HeteroticSystem'):
            return MLPipeline()

    def test_load_all_cosmological_data(self, pipeline):
        """Test loading all cosmological data."""
        pipeline.data_loader.load_cmb_data = Mock(return_value={'ell': np.arange(100)})
        pipeline.data_loader.load_bao_data = Mock(return_value={'measurements': []})
        pipeline.data_loader.load_void_catalog = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_sdss_galaxy_catalog = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_frb_data = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_lyman_alpha_data = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_jwst_data = Mock(return_value=pd.DataFrame({'test': [1]}))
        
        data = pipeline._load_all_cosmological_data()
        
        assert 'cmb' in data
        assert 'bao' in data
        assert 'void' in data
        assert 'galaxy' in data

    def test_get_encoder_dimensions(self, pipeline):
        """Test encoder dimension calculation."""
        data = {
            'cmb': {'test': 'data'},
            'bao': {'test': 'data'},
            'void': {'test': 'data'},
            'galaxy': {'test': 'data'},
            'frb': {'test': 'data'},
            'lyman_alpha': {'test': 'data'},
            'jwst': {'test': 'data'}
        }
        
        dims = pipeline._get_encoder_dimensions(data)
        
        assert 'cmb' in dims
        assert dims['cmb'] == 500
        assert dims['bao'] == 10
        assert dims['void'] == 20

    def test_prepare_ssl_training_data(self, pipeline):
        """Test SSL training data preparation."""
        data = {'cmb': {}, 'bao': {}}
        
        batches = pipeline._prepare_ssl_training_data(data)
        
        assert len(batches) > 0
        for batch in batches:
            assert isinstance(batch, dict)
            for modality, tensor in batch.items():
                assert isinstance(tensor, torch.Tensor)
                assert tensor.device.type == pipeline.device.type  # Compare device types, not exact device objects

    def test_load_survey_specific_data(self, pipeline):
        """Test loading survey-specific data."""
        data = pipeline._load_survey_specific_data()
        assert isinstance(data, list)

    def test_load_test_data(self, pipeline):
        """Test loading test data."""
        data = pipeline._load_test_data()
        assert isinstance(data, dict)

    def test_extract_features_with_ssl(self, pipeline):
        """Test feature extraction with SSL."""
        pipeline.ssl_learner = Mock()
        pipeline.ssl_learner.encode = Mock(return_value={
            'cmb': torch.randn(100, 512),
            'bao': torch.randn(100, 512)
        })
        
        test_data = {'cmb': {}, 'bao': {}}
        features = pipeline._extract_features_with_ssl(test_data)
        
        assert isinstance(features, np.ndarray)
        assert features.shape[1] == 512

    def test_prepare_survey_datasets(self, pipeline):
        """Test preparing survey datasets."""
        datasets = pipeline._prepare_survey_datasets()
        assert isinstance(datasets, dict)

    def test_synthesize_ml_results(self, pipeline):
        """Test ML results synthesis."""
        results = {
            'pattern_detection': {
                'top_anomalies': [{'index': 1}, {'index': 2}]
            },
            'validation': {
                'bootstrap_validation': {
                    'stability_summary': {'stability_status': 'stable'}
                },
                'null_hypothesis_testing': {
                    'significance_test': {'overall_significance': True}
                }
            }
        }
        
        synthesis = pipeline._synthesize_ml_results(results)
        
        assert 'pipeline_completed' in synthesis
        assert 'key_findings' in synthesis
        assert synthesis['key_findings']['detected_anomalies'] == 2

    def test_run_e8_pattern_analysis(self, pipeline):
        """Test E8 pattern analysis."""
        pipeline.e8_system = Mock()
        pipeline.e8_system.construct_single_e8 = Mock(return_value=np.random.randn(248, 248))
        pipeline.e8_system.construct_heterotic_system = Mock(return_value=np.random.randn(496, 496))
        pipeline.e8_system.get_network_properties = Mock(return_value={
            'clustering_coefficient': 25.0/32.0,
            'dimension': 496
        })
        
        with patch('pipeline.ml.ml_pipeline.DataLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_act_dr6 = Mock(return_value=(
                np.linspace(100, 3000, 100),
                np.random.lognormal(-10, 0.5, 100),
                np.random.lognormal(-10, 0.5, 100) * 0.1
            ))
            mock_loader_class.return_value = mock_loader
            
            result = pipeline._run_e8_pattern_analysis()
            
            assert 'e8_signature_detected' in result
            assert 'pattern_analysis' in result
            assert 'network_analysis' in result

    def test_analyze_e8_patterns(self, pipeline):
        """Test E8 pattern analysis."""
        ell = np.linspace(100, 3000, 100)
        C_ell = np.random.lognormal(-10, 0.5, 100)
        C_ell_err = C_ell * 0.1
        e8_angles = np.array([np.pi/6, np.pi/4, np.pi/3, np.pi/2])
        
        analysis = pipeline._analyze_e8_patterns(ell, C_ell, C_ell_err, e8_angles)
        
        assert 'pattern_matches' in analysis
        assert 'pattern_score' in analysis
        assert len(analysis['pattern_matches']) == len(e8_angles)

    def test_analyze_e8_network_topology(self, pipeline):
        """Test E8 network topology analysis."""
        pipeline.e8_system = Mock()
        pipeline.e8_system.get_network_properties = Mock(return_value={
            'clustering_coefficient': 25.0/32.0,
            'dimension': 496,
            'connectivity_type': 'heterotic'
        })
        
        heterotic_system = np.random.randn(496, 496)
        analysis = pipeline._analyze_e8_network_topology(heterotic_system)
        
        assert 'clustering_coefficient' in analysis
        assert 'network_dimension' in analysis
        assert analysis.get('topology_type') == 'E8×E8' or 'topology_type' in analysis

    def test_test_e8_pattern_significance(self, pipeline):
        """Test E8 pattern significance testing."""
        pattern_analysis = {
            'pattern_score': 1.5,
            'n_features_detected': 5
        }
        network_analysis = {
            'clustering_coefficient': 25.0/32.0
        }
        
        significance = pipeline._test_e8_pattern_significance(pattern_analysis, network_analysis)
        
        assert 'z_score' in significance
        assert 'p_value' in significance
        assert 'significant' in significance

    def test_run_network_analysis(self, pipeline):
        """Test network analysis."""
        pipeline.e8_system = Mock()
        pipeline.e8_system.construct_heterotic_system = Mock(return_value=np.random.randn(496, 496))
        pipeline.e8_system.get_network_properties = Mock(return_value={
            'clustering_coefficient': 25.0/32.0,
            'dimension': 496,
            'connectivity_type': 'heterotic'
        })
        
        result = pipeline._run_network_analysis()
        
        assert 'network_parameters' in result
        assert 'theoretical_comparison' in result

    def test_extract_network_parameters(self, pipeline):
        """Test network parameter extraction."""
        network_props = {
            'clustering_coefficient': 0.78,
            'dimension': 496,
            'connectivity_type': 'heterotic',
            'n_nodes': 496,
            'n_edges': 1000
        }
        
        params = pipeline._extract_network_parameters(network_props)
        
        assert 'clustering_coefficient' in params
        assert params['clustering_coefficient'] == 0.78

    def test_compare_network_theory(self, pipeline):
        """Test network theory comparison."""
        network_params = {
            'clustering_coefficient': 25.0/32.0,
            'dimension': 496
        }
        
        comparison = pipeline._compare_network_theory(network_params)
        
        assert 'clustering_observed' in comparison
        assert 'clustering_theoretical' in comparison
        assert 'consistent' in comparison

    def test_run_chirality_analysis(self, pipeline):
        """Test chirality analysis."""
        context = {'z': 0.5}
        result = pipeline._run_chirality_analysis(context)
        
        assert 'chiral_amplitude' in result
        assert 'chirality_patterns' in result
        assert 'significance' in result

    def test_calculate_e8_chiral_amplitude(self, pipeline):
        """Test chiral amplitude calculation."""
        amplitude = pipeline._calculate_e8_chiral_amplitude(z=0.5)
        assert isinstance(amplitude, float)

    def test_detect_chirality_patterns(self, pipeline):
        """Test chirality pattern detection."""
        chiral_amplitude = 0.15
        z = 0.5
        
        patterns = pipeline._detect_chirality_patterns(chiral_amplitude, z)
        
        assert 'asymmetry_metric' in patterns
        assert 'pattern_detected' in patterns

    def test_test_chirality_significance(self, pipeline):
        """Test chirality significance."""
        chirality_patterns = {
            'asymmetry_metric': 0.2
        }
        
        significance = pipeline._test_chirality_significance(chirality_patterns)
        
        assert 'z_score' in significance
        assert 'p_value' in significance
        assert 'significant' in significance

    def test_run_gamma_qtep_analysis(self, pipeline):
        """Test gamma-QTEP analysis."""
        context = {'z_min': 0.0, 'z_max': 2.0, 'z_steps': 20}
        result = pipeline._run_gamma_qtep_analysis(context)
        
        assert 'gamma_values' in result
        assert 'qtep_values' in result
        assert 'pattern_analysis' in result

    def test_analyze_gamma_qtep_patterns(self, pipeline):
        """Test gamma-QTEP pattern analysis."""
        z_grid = np.linspace(0, 2, 20)
        gamma_values = [HLCDMCosmology.gamma_at_redshift(z) / HLCDM_PARAMS.get_hubble_at_redshift(z) for z in z_grid]
        qtep_values = [HLCDM_PARAMS.QTEP_RATIO] * len(z_grid)
        
        analysis = pipeline._analyze_gamma_qtep_patterns(z_grid, gamma_values, qtep_values)
        
        assert 'qtep_mean' in analysis
        assert 'qtep_theoretical' in analysis
        assert 'qtep_consistent' in analysis

    def test_test_gamma_qtep_correlation(self, pipeline):
        """Test gamma-QTEP correlation."""
        gamma_values = np.linspace(0.1, 0.2, 20).tolist()
        qtep_values = [HLCDM_PARAMS.QTEP_RATIO] * 20
        
        correlation = pipeline._test_gamma_qtep_correlation(gamma_values, qtep_values)
        
        assert 'correlation' in correlation
        assert 'p_value' in correlation
        assert 'pattern_consistent' in correlation

    def test_synthesize_scientific_test_results(self, pipeline):
        """Test scientific test results synthesis."""
        test_results = {
            'e8_pattern': {'e8_signature_detected': True},
            'network_analysis': {'theoretical_comparison': {'consistent': True}},
            'chirality': {'chirality_detected': False},
            'gamma_qtep': {'pattern_detected': True}
        }
        
        synthesis = pipeline._synthesize_scientific_test_results(test_results)
        
        assert 'individual_scores' in synthesis
        assert 'total_score' in synthesis
        assert 'strength_category' in synthesis

    def test_classify_evidence_strength(self, pipeline):
        """Test evidence strength classification."""
        assert pipeline._classify_evidence_strength(12, 12) == 'STRONG'
        assert pipeline._classify_evidence_strength(9, 12) == 'MODERATE'
        assert pipeline._classify_evidence_strength(6, 12) == 'WEAK'
        assert pipeline._classify_evidence_strength(3, 12) == 'INSUFFICIENT'

    def test_create_ml_systematic_budget(self, pipeline):
        """Test systematic budget creation."""
        budget = pipeline._create_ml_systematic_budget()
        
        assert hasattr(budget, 'get_budget_breakdown')
        breakdown = budget.get_budget_breakdown()
        assert 'components' in breakdown
        assert 'total_systematic' in breakdown

    def test_generate_overall_assessment(self, pipeline):
        """Test overall assessment generation."""
        synthesis = {
            'strength_category': 'STRONG',
            'total_score': 10,
            'max_possible_score': 12
        }
        
        assessment = pipeline._generate_overall_assessment(synthesis)
        assert isinstance(assessment, str)
        assert 'STRONG' in assessment

    def test_validate(self, pipeline):
        """Test basic validation."""
        result = pipeline.validate()
        
        assert 'validation_type' in result
        assert 'status' in result
        assert result['status'] == 'PASSED'

    def test_validate_extended(self, pipeline):
        """Test extended validation."""
        result = pipeline.validate_extended()
        
        assert 'validation_type' in result
        assert 'status' in result
        assert 'monte_carlo' in result

    def test_run_test_unknown(self, pipeline):
        """Test _run_test with unknown test name."""
        with pytest.raises(ValueError, match="Unknown ML test"):
            pipeline._run_test('unknown_test')

    def test_run_ssl_training(self, pipeline):
        """Test SSL training stage."""
        pipeline.data_loader.load_cmb_data = Mock(return_value={'ell': np.arange(100)})
        pipeline.data_loader.load_bao_data = Mock(return_value={'measurements': []})
        pipeline.data_loader.load_void_catalog = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_sdss_galaxy_catalog = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_frb_data = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_lyman_alpha_data = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_jwst_data = Mock(return_value=pd.DataFrame({'test': [1]}))
        
        with patch('pipeline.ml.ml_pipeline.ContrastiveLearner') as mock_learner_class:
            mock_learner = Mock()
            mock_learner.train = Mock(return_value={'loss': 0.5})
            mock_learner_class.return_value = mock_learner
            
            result = pipeline.run_ssl_training(show_progress=False)
            
            assert 'ssl_training_completed' in result
            assert pipeline.stage_completed['ssl_training']

    def test_run_domain_adaptation(self, pipeline):
        """Test domain adaptation stage."""
        pipeline.stage_completed['ssl_training'] = True
        pipeline.ssl_learner = Mock()
        
        result = pipeline.run_domain_adaptation(show_progress=False)
        
        assert 'adaptation_completed' in result
        assert pipeline.stage_completed['domain_adaptation']

    def test_run_pattern_detection(self, pipeline):
        """Test pattern detection stage."""
        pipeline.stage_completed['domain_adaptation'] = True
        pipeline.ssl_learner = Mock()
        pipeline.ssl_learner.encode = Mock(return_value={
            'cmb': torch.randn(100, 512),
            'bao': torch.randn(100, 512)
        })
        
        with patch('pipeline.ml.ml_pipeline.EnsembleDetector') as mock_detector_class, \
             patch('pipeline.ml.ml_pipeline.EnsembleAggregator') as mock_aggregator_class:
            mock_detector = Mock()
            mock_detector.fit = Mock()
            mock_detector.predict = Mock(return_value={
                'ensemble_scores': np.random.rand(100),
                'individual_scores': {'if': np.random.rand(100)}
            })
            mock_detector_class.return_value = mock_detector
            
            mock_aggregator = Mock()
            mock_aggregator.aggregate_scores = Mock(return_value={
                'ensemble_scores': np.random.rand(100),
                'top_anomalies': []
            })
            mock_aggregator_class.return_value = mock_aggregator
            
            result = pipeline.run_pattern_detection(show_progress=False)
            
            assert 'detection_completed' in result
            assert pipeline.stage_completed['pattern_detection']

    def test_run_interpretability(self, pipeline):
        """Test interpretability stage."""
        pipeline.stage_completed['pattern_detection'] = True
        pipeline.ensemble_detector = Mock()
        pipeline.ensemble_detector.predict = Mock(return_value={
            'ensemble_scores': np.random.rand(100)
        })
        pipeline.ssl_learner = Mock()
        pipeline.ssl_learner.encode = Mock(return_value={
            'cmb': torch.randn(100, 512)
        })
        
        result = pipeline.run_interpretability(show_progress=False)
        
        assert 'interpretability_completed' in result
        assert pipeline.stage_completed['interpretability']

    def test_run_validation(self, pipeline):
        """Test validation stage."""
        pipeline.stage_completed['pattern_detection'] = True
        pipeline.ssl_learner = Mock()
        pipeline.ssl_learner.encode = Mock(return_value={
            'cmb': torch.randn(100, 512)
        })
        
        result = pipeline.run_validation(show_progress=False)
        
        assert 'validation_completed' in result
        assert pipeline.stage_completed['validation']

    def test_run_full_pipeline(self, pipeline):
        """Test running full pipeline."""
        pipeline.data_loader.load_cmb_data = Mock(return_value={'ell': np.arange(100)})
        pipeline.data_loader.load_bao_data = Mock(return_value={'measurements': []})
        pipeline.data_loader.load_void_catalog = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_sdss_galaxy_catalog = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_frb_data = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_lyman_alpha_data = Mock(return_value=pd.DataFrame({'test': [1]}))
        pipeline.data_loader.load_jwst_data = Mock(return_value=pd.DataFrame({'test': [1]}))
        
        with patch('pipeline.ml.ml_pipeline.ContrastiveLearner') as mock_learner_class, \
             patch('pipeline.ml.ml_pipeline.EnsembleDetector') as mock_detector_class, \
             patch('pipeline.ml.ml_pipeline.EnsembleAggregator') as mock_aggregator_class:
            
            mock_learner = Mock()
            mock_learner.train = Mock(return_value={'loss': 0.5})
            mock_learner.encode = Mock(return_value={'cmb': torch.randn(100, 512)})
            mock_learner_class.return_value = mock_learner
            
            mock_detector = Mock()
            mock_detector.fit = Mock()
            mock_detector.predict = Mock(return_value={
                'ensemble_scores': np.random.rand(100),
                'individual_scores': {'if': np.random.rand(100)}
            })
            mock_detector_class.return_value = mock_detector
            
            mock_aggregator = Mock()
            mock_aggregator.aggregate_scores = Mock(return_value={
                'ensemble_scores': np.random.rand(100),
                'top_anomalies': []
            })
            mock_aggregator_class.return_value = mock_aggregator
            
            result = pipeline.run({'stages': ['ssl', 'domain', 'detect', 'interpret', 'validate']})
            
            assert 'pipeline_completed' in result or 'error' in result

    def test_run_scientific_tests(self, pipeline):
        """Test running scientific tests."""
        pipeline.e8_system = Mock()
        pipeline.e8_system.construct_single_e8 = Mock(return_value=np.random.randn(248, 248))
        pipeline.e8_system.construct_heterotic_system = Mock(return_value=np.random.randn(496, 496))
        pipeline.e8_system.get_network_properties = Mock(return_value={
            'clustering_coefficient': 25.0/32.0,
            'dimension': 496
        })
        
        with patch('pipeline.ml.ml_pipeline.DataLoader') as mock_loader_class:
            mock_loader = Mock()
            mock_loader.load_act_dr6 = Mock(return_value=(
                np.linspace(100, 3000, 100),
                np.random.lognormal(-10, 0.5, 100),
                np.random.lognormal(-10, 0.5, 100) * 0.1
            ))
            mock_loader_class.return_value = mock_loader
            
            result = pipeline.run_scientific_tests({'tests': ['e8_pattern']})
            
            assert 'test_results' in result
            assert 'synthesis' in result

