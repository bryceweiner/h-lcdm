"""
Tests for CMB pipeline functionality.

Tests CMB analysis, blinding, systematic errors, covariance matrices,
and statistical validation.
"""
import pytest
import numpy as np

from pipeline.cmb.cmb_pipeline import CMBPipeline


class TestCMBPipeline:
    """Test CMB pipeline functionality."""

    @pytest.fixture
    def cmb_pipeline(self, temp_output_dir, mock_data_loader):
        """Create CMB pipeline instance."""
        return CMB_Pipeline(str(temp_output_dir))

    def test_pipeline_initialization(self, cmb_pipeline):
        """Test pipeline initialization."""
        assert cmb_pipeline.name == "cmb"
        assert len(cmb_pipeline.available_methods) > 0

    def test_covariance_analysis(self, cmb_pipeline, sample_cmb_data):
        """Test CMB covariance matrix analysis."""
        covariance_analysis = cmb_pipeline._analyze_cmb_covariance_matrices(sample_cmb_data)

        assert 'individual_analyses' in covariance_analysis
        assert 'overall_assessment' in covariance_analysis

        # Check individual analyses
        for dataset_name in sample_cmb_data.keys():
            assert dataset_name in covariance_analysis['individual_analyses']
            analysis = covariance_analysis['individual_analyses'][dataset_name]
            assert 'covariance_matrix_properties' in analysis

    def test_blinding_functionality(self, cmb_pipeline):
        """Test blinding implementation."""
        context = {'blinding_enabled': True}
        result = cmb_pipeline.run(context)

        assert 'blinding_info' in result
        blinding_info = result['blinding_info']
        assert blinding_info['blinding_status'] == 'blinded'

    def test_systematic_error_budget(self, cmb_pipeline):
        """Test systematic error budget."""
        result = cmb_pipeline.run()

        assert 'systematic_budget' in result
        budget = result['systematic_budget']

        assert 'total_systematic' in budget
        assert 'components' in budget

        # Check expected CMB systematic components
        components = budget['components']
        expected_components = ['beam_leakage', 'foreground_residuals', 'calibration',
                             'temperature_leakage', 'point_sources', 'atmospheric',
                             'glitch_removal']
        for component in expected_components:
            assert component in components

    def test_validation_basic(self, cmb_pipeline):
        """Test basic validation."""
        result = cmb_pipeline.validate()

        assert 'null_hypothesis_test' in result
        nh_test = result['null_hypothesis_test']
        assert 'null_hypothesis' in nh_test
        assert 'p_value' in nh_test

    def test_validation_extended(self, cmb_pipeline):
        """Test extended validation."""
        context = {'n_null_simulations': 100}  # Smaller for testing
        result = cmb_pipeline.validate_extended(context)

        assert 'bootstrap' in result
        assert 'monte_carlo' in result
        assert 'null_hypothesis' in result

    def test_run_complete_pipeline(self, cmb_pipeline):
        """Test complete pipeline run."""
        result = cmb_pipeline.run()

        required_keys = ['methods_run', 'analysis_methods', 'synthesis', 'detection_summary']
        for key in required_keys:
            assert key in result

    def test_method_execution(self, cmb_pipeline):
        """Test individual CMB analysis methods."""
        methods = ['phase', 'topological']  # Test a subset
        context = {'methods': methods}

        result = cmb_pipeline.run(context)
        analysis_methods = result['analysis_methods']

        for method in methods:
            assert method in analysis_methods
            assert 'evidence' in analysis_methods[method]
            assert 'p_value' in analysis_methods[method]

    def test_synthesis_across_methods(self, cmb_pipeline):
        """Test synthesis of results across methods."""
        result = cmb_pipeline.run()
        synthesis = result['synthesis']

        assert 'combined_evidence' in synthesis
        assert 'method_consistency' in synthesis
        assert 'dominant_detection' in synthesis

    def test_null_hypothesis_testing(self, cmb_pipeline):
        """Test null hypothesis testing."""
        result = cmb_pipeline._null_hypothesis_testing(100)

        assert 'test' in result
        assert 'p_value' in result
        assert 'n_simulations' in result
        assert 'null_hypothesis_adequate' in result

    def test_numerical_stability_checks(self, cmb_pipeline):
        """Test numerical stability of covariance operations."""
        # Create a test covariance matrix
        cov_matrix = np.eye(10) + 0.1 * np.random.randn(10, 10)
        cov_matrix = cov_matrix @ cov_matrix.T  # Make positive definite

        stability = cmb_pipeline.check_numerical_stability(cov_matrix)

        assert stability['is_finite'] is True
        assert stability['matrix_invertible'] is True
        assert stability['stable_for_operation'] is True

    def test_convergence_diagnostics(self, cmb_pipeline):
        """Test MCMC convergence diagnostics."""
        # Create mock MCMC samples
        samples = {
            'param1': np.random.normal(1.0, 0.1, 1000),
            'param2': np.random.normal(2.0, 0.2, 1000)
        }

        convergence = cmb_pipeline.check_convergence(samples)

        assert 'parameter_convergence' in convergence
        assert 'overall_converged' in convergence
        assert convergence['method'] == 'gelman_rubin_r_hat'

    def test_multiple_testing_correction(self, cmb_pipeline):
        """Test multiple testing correction for CMB detections."""
        p_values = np.array([0.01, 0.02, 0.03, 0.04, 0.05])

        correction = cmb_pipeline.apply_multiple_testing_correction(p_values, 'bonferroni')

        assert correction['method'] == 'bonferroni'
        assert correction['n_tests'] == 5
        assert len(correction['corrected_p_values']) == 5
        assert correction['alpha_corrected'] == 0.05 / 5

    def test_systematic_budget_modification(self, cmb_pipeline):
        """Test systematic budget modification."""
        budget = cmb_pipeline._create_cmb_systematic_budget()

        # Test adding component
        budget.add_component('new_systematic', 0.005)
        assert 'new_systematic' in budget.components

        # Test total calculation updates
        total_before = budget.get_total_systematic()
        budget.add_component('another_systematic', 0.010)
        total_after = budget.get_total_systematic()
        assert total_after > total_before

        # Test removal
        budget.remove_component('beam_leakage')
        assert 'beam_leakage' not in budget.components
