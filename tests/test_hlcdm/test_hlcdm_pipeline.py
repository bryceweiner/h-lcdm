"""
Tests for H-ΛCDM pipeline functionality.

Tests meta-analysis, blinding, systematic errors, and statistical validation.
"""
import pytest
import numpy as np

from pipeline.hlcdm.hlcdm_pipeline import HLCDMPipeline


class TestHLambdaDMPipeline:
    """Test H-ΛCDM pipeline functionality."""

    @pytest.fixture
    def hlcdm_pipeline(self, temp_output_dir):
        """Create H-ΛCDM pipeline instance."""
        return HLambdaDM_Pipeline(str(temp_output_dir))

    def test_pipeline_initialization(self, hlcdm_pipeline):
        """Test pipeline initialization."""
        assert hlcdm_pipeline.name == "hlcdm"
        assert len(hlcdm_pipeline.available_tests) > 0

    def test_blinding_functionality(self, hlcdm_pipeline):
        """Test blinding implementation."""
        context = {'blinding_enabled': True}
        result = hlcdm_pipeline.run(context)

        assert 'blinding_info' in result
        blinding_info = result['blinding_info']
        assert blinding_info['blinding_status'] == 'blinded'

    def test_systematic_error_budget(self, hlcdm_pipeline):
        """Test systematic error budget."""
        result = hlcdm_pipeline.run()

        assert 'systematic_budget' in result
        budget = result['systematic_budget']

        assert 'total_systematic' in budget
        assert 'components' in budget

        # Check expected H-ΛCDM systematic components (propagated from probes)
        components = budget['components']
        expected_components = ['bao_systematics', 'cmb_systematics', 'void_systematics',
                             'gamma_systematics', 'cross_calibration', 'evidence_combination',
                             'test_selection_bias']
        for component in expected_components:
            assert component in components

    def test_validation_basic(self, hlcdm_pipeline):
        """Test basic validation."""
        result = hlcdm_pipeline.validate()

        assert 'null_hypothesis_test' in result
        nh_test = result['null_hypothesis_test']
        assert 'null_hypothesis' in nh_test
        assert 'p_value' in nh_test

    def test_validation_extended(self, hlcdm_pipeline):
        """Test extended validation."""
        context = {'n_bootstrap': 100, 'n_monte_carlo': 100}  # Smaller for testing
        result = hlcdm_pipeline.validate_extended(context)

        assert 'bootstrap' in result
        assert 'monte_carlo' in result
        assert 'loo_cv' in result

    def test_run_complete_pipeline(self, hlcdm_pipeline):
        """Test complete pipeline run."""
        result = hlcdm_pipeline.run()

        required_keys = ['test_results', 'synthesis', 'tests_run', 'overall_assessment']
        for key in required_keys:
            assert key in result

    def test_individual_test_execution(self, hlcdm_pipeline):
        """Test individual extension test execution."""
        # Test JWST analysis
        jwst_result = hlcdm_pipeline._run_jwst_analysis()
        assert 'evidence' in jwst_result
        assert 'p_value' in jwst_result

        # Test Lyman-α analysis
        ly_alpha_result = hlcdm_pipeline._run_lyman_alpha_analysis()
        assert 'evidence' in ly_alpha_result
        assert 'p_value' in ly_alpha_result

        # Test FRB analysis
        frb_result = hlcdm_pipeline._run_frb_analysis()
        assert 'evidence' in frb_result
        assert 'p_value' in frb_result

    def test_synthesis_across_tests(self, hlcdm_pipeline, sample_test_results):
        """Test synthesis of results across extension tests."""
        synthesis = hlcdm_pipeline._synthesize_hlcdm_results(sample_test_results)

        assert 'combined_evidence' in synthesis
        assert 'evidence_ratio' in synthesis
        assert 'most_significant_test' in synthesis
        assert 'consistency_across_tests' in synthesis

    def test_null_hypothesis_testing(self, hlcdm_pipeline):
        """Test null hypothesis testing for extension tests."""
        result = hlcdm_pipeline._test_null_hypothesis()

        assert 'test' in result
        assert 'null_hypothesis' in result
        assert 'p_value' in result
        assert 'tests_rejected' in result

    def test_individual_null_hypothesis_tests(self, hlcdm_pipeline):
        """Test individual null hypothesis tests for each extension."""
        test_cases = [
            ('jwst', {'evidence': 2.5, 'p_value': 0.012}),
            ('lyman_alpha', {'evidence': 1.8, 'p_value': 0.035}),
            ('frb', {'evidence': 3.2, 'p_value': 0.001}),
            ('e8_ml', {'evidence': 1.2, 'p_value': 0.230}),
            ('e8_chiral', {'evidence': 2.8, 'p_value': 0.005}),
            ('temporal_cascade', {'evidence': 1.5, 'p_value': 0.133})
        ]

        for test_name, test_result in test_cases:
            nh_result = hlcdm_pipeline._test_individual_null_hypothesis(test_name, test_result)
            assert 'null_hypothesis' in nh_result
            assert 'null_hypothesis_rejected' in nh_result

    def test_bootstrap_validation(self, hlcdm_pipeline):
        """Test bootstrap validation."""
        result = hlcdm_pipeline._bootstrap_hlcdm_validation(50)

        assert 'method' in result
        assert 'n_bootstrap' in result
        assert 'stability_ok' in result

    def test_monte_carlo_validation(self, hlcdm_pipeline):
        """Test Monte Carlo validation."""
        result = hlcdm_pipeline._monte_carlo_hlcdm_validation(50)

        assert 'method' in result
        assert 'n_simulations' in result
        assert 'evidence_distribution' in result

    def test_loo_cv_validation(self, hlcdm_pipeline):
        """Test Leave-One-Out Cross-Validation."""
        result = hlcdm_pipeline._loo_cv_validation()

        assert 'method' in result
        assert 'rmse' in result
        assert 'n_predictions' in result

    def test_jackknife_validation(self, hlcdm_pipeline):
        """Test jackknife validation."""
        result = hlcdm_pipeline._jackknife_validation()

        assert 'method' in result
        assert 'jackknife_std_error' in result
        assert 'bias_correction' in result

    def test_model_comparison(self, hlcdm_pipeline):
        """Test model comparison between H-ΛCDM and ΛCDM."""
        result = hlcdm_pipeline._perform_model_comparison()

        assert 'models_compared' in result
        assert 'preferred_model' in result
        assert 'evidence_ratio' in result

    def test_different_test_combinations(self, hlcdm_pipeline):
        """Test pipeline with different test combinations."""
        test_combinations = [
            ['jwst', 'lyman_alpha'],
            ['jwst', 'frb', 'e8_ml'],
            ['all']  # Test all available
        ]

        for tests in test_combinations:
            context = {'tests': tests}
            result = hlcdm_pipeline.run(context)

            if 'all' in tests:
                assert len(result['tests_run']) == len(hlcdm_pipeline.available_tests) - 1
            else:
                assert len(result['tests_run']) == len(tests)

    def test_overall_assessment(self, hlcdm_pipeline, sample_test_results):
        """Test overall assessment generation."""
        assessment = hlcdm_pipeline._generate_overall_assessment(
            hlcdm_pipeline._synthesize_hlcdm_results(sample_test_results)
        )

        assert 'evidence_strength' in assessment
        assert 'recommended_action' in assessment
        assert 'key_findings' in assessment

    def test_systematic_budget_propagation(self, hlcdm_pipeline):
        """Test that systematic errors are properly propagated from individual probes."""
        budget = hlcdm_pipeline._create_hlcdm_systematic_budget()

        # Check that all components are included
        expected_components = ['bao_systematics', 'cmb_systematics', 'void_systematics',
                             'gamma_systematics', 'cross_calibration', 'evidence_combination',
                             'test_selection_bias']

        for component in expected_components:
            assert component in budget.components
            assert budget.components[component] > 0

        # Test that total systematic is reasonable
        total = budget.get_total_systematic()
        assert 0.1 < total < 1.0  # Reasonable range for propagated systematics
