"""
Tests for Void pipeline functionality.

Tests void analysis, blinding, systematic errors, and statistical validation.
"""
import pytest
import numpy as np
import pandas as pd

from pipeline.void.void_pipeline import VoidPipeline


class TestVoidPipeline:
    """Test Void pipeline functionality."""

    @pytest.fixture
    def void_pipeline(self, temp_output_dir):
        """Create Void pipeline instance."""
        return VoidPipeline(str(temp_output_dir))

    @pytest.fixture
    def mock_void_data(self, sample_void_catalog):
        """Create mock void data for testing."""
        return {
            'sdss': {'catalog': sample_void_catalog}
        }

    def test_pipeline_initialization(self, void_pipeline):
        """Test pipeline initialization."""
        assert void_pipeline.name == "void"
        assert len(void_pipeline.available_surveys) > 0

    def test_covariance_analysis(self, void_pipeline, mock_void_data):
        """Test void covariance matrix analysis."""
        covariance_analysis = void_pipeline._analyze_void_covariance_matrices(mock_void_data)

        assert 'individual_analyses' in covariance_analysis
        assert 'overall_assessment' in covariance_analysis

        # Check individual analysis for sdss
        sdss_analysis = covariance_analysis['individual_analyses']['sdss']
        assert 'covariance_matrix_properties' in sdss_analysis
        assert 'sample_size' in sdss_analysis

    def test_blinding_functionality(self, void_pipeline):
        """Test blinding implementation."""
        context = {'blinding_enabled': True}
        result = void_pipeline.run(context)

        assert 'blinding_info' in result
        blinding_info = result['blinding_info']
        assert blinding_info['blinding_status'] == 'blinded'

    def test_systematic_error_budget(self, void_pipeline):
        """Test systematic error budget."""
        result = void_pipeline.run()

        assert 'systematic_budget' in result
        budget = result['systematic_budget']

        assert 'total_systematic' in budget
        assert 'components' in budget

        # Check expected void systematic components
        components = budget['components']
        expected_components = ['void_finding_bias', 'selection_effects',
                             'tracer_density', 'redshift_precision',
                             'survey_geometry', 'cosmological_model',
                             'numerical_precision']
        for component in expected_components:
            assert component in components

    def test_validation_basic(self, void_pipeline):
        """Test basic validation."""
        result = void_pipeline.validate()

        assert 'null_hypothesis_test' in result
        nh_test = result['null_hypothesis_test']
        assert 'null_hypothesis' in nh_test
        assert 'p_value' in nh_test

    def test_validation_extended(self, void_pipeline):
        """Test extended validation."""
        context = {'n_bootstrap': 100, 'n_randomization': 100}  # Smaller for testing
        result = void_pipeline.validate_extended(context)

        assert 'bootstrap' in result
        assert 'monte_carlo' in result
        assert 'randomization' in result
        assert 'null_hypothesis' in result

    def test_run_complete_pipeline(self, void_pipeline):
        """Test complete pipeline run."""
        result = void_pipeline.run()

        required_keys = ['surveys_analyzed', 'void_data', 'e8_alignment',
                        'clustering_analysis', 'analysis_summary']
        for key in required_keys:
            assert key in result

    def test_alignment_analysis(self, void_pipeline, mock_void_data):
        """Test E8 alignment analysis."""
        with pytest.mock.patch.object(void_pipeline, '_load_void_data',
                                    return_value=mock_void_data):
            alignment_results = void_pipeline._analyze_e8_alignment(mock_void_data)

            assert 'alignment_strength' in alignment_results
            assert 'e8_correlation' in alignment_results
            assert 'statistical_significance' in alignment_results

    def test_clustering_analysis(self, void_pipeline, mock_void_data):
        """Test void clustering analysis."""
        with pytest.mock.patch.object(void_pipeline, '_load_void_data',
                                    return_value=mock_void_data):
            clustering_results = void_pipeline._analyze_clustering(mock_void_data)

            assert 'clustering_strength' in clustering_results
            assert 'void_correlations' in clustering_results
            assert 'scale_dependence' in clustering_results

    def test_null_hypothesis_testing(self, void_pipeline):
        """Test null hypothesis testing for voids."""
        result = void_pipeline._void_null_hypothesis_testing(100)

        assert 'test' in result
        assert 'p_value' in result
        assert 'n_simulations' in result
        assert 'null_hypothesis_adequate' in result

    def test_randomization_testing(self, void_pipeline):
        """Test randomization testing."""
        result = void_pipeline._randomization_testing(100)

        assert 'method' in result
        assert 'p_value' in result
        assert 'n_randomizations' in result

    def test_monte_carlo_validation(self, void_pipeline):
        """Test Monte Carlo validation."""
        result = void_pipeline._monte_carlo_validation(50)  # Small number for testing

        assert 'passed' in result
        assert 'method' in result
        assert 'n_simulations' in result
        assert 'null_hypothesis_consistent' in result

    def test_bootstrap_validation(self, void_pipeline):
        """Test bootstrap validation."""
        result = void_pipeline._bootstrap_void_validation(50)

        assert 'method' in result
        assert 'n_bootstrap' in result
        assert 'stability_ok' in result

    def test_different_surveys(self, void_pipeline):
        """Test pipeline with different survey combinations."""
        # Test with default surveys
        result = void_pipeline.run()
        assert len(result['surveys_analyzed']) > 0

    def test_data_processing(self, void_pipeline, sample_void_catalog):
        """Test void data processing."""
        # Mock the data loading
        mock_data = {'survey1': {'catalog': sample_void_catalog}}

        with pytest.mock.patch.object(void_pipeline.data_processor, 'process',
                                    return_value=mock_data):
            processed_data = void_pipeline.data_processor.process(['survey1'])

            assert 'survey1' in processed_data
            assert 'catalog' in processed_data['survey1']
            assert isinstance(processed_data['survey1']['catalog'], pd.DataFrame)

    def test_systematic_budget_calculation(self, void_pipeline):
        """Test systematic budget calculation."""
        budget = void_pipeline._create_void_systematic_budget()

        total = budget.get_total_systematic()
        assert total > 0

        breakdown = budget.get_budget_breakdown()
        assert 'total_systematic' in breakdown
        assert 'dominant_source' in breakdown

        # Check that components sum correctly in quadrature
        manual_total = np.sqrt(sum(v**2 for v in budget.components.values()))
        assert abs(total - manual_total) < 1e-10
