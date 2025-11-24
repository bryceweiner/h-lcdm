"""
Tests for Gamma pipeline functionality.

Tests theoretical gamma analysis, blinding, systematic errors,
and statistical validation.
"""
import pytest
import numpy as np

from pipeline.gamma.gamma_pipeline import GammaPipeline


class TestGammaPipeline:
    """Test Gamma pipeline functionality."""

    @pytest.fixture
    def gamma_pipeline(self, temp_output_dir):
        """Create Gamma pipeline instance."""
        return GammaPipeline(str(temp_output_dir))

    def test_pipeline_initialization(self, gamma_pipeline):
        """Test pipeline initialization."""
        assert gamma_pipeline.name == "gamma"
        assert gamma_pipeline.z_min == 0.0
        assert gamma_pipeline.z_max == 10.0
        assert gamma_pipeline.z_steps == 100

    def test_theoretical_calculations(self, gamma_pipeline):
        """Test theoretical gamma and Lambda calculations."""
        z_test = 1.0

        # Test gamma calculation
        from hlcdm.cosmology import HLCDMCosmology
        gamma_val = HLCDMCosmology.gamma_at_redshift(z_test)
        assert isinstance(gamma_val, (int, float))
        assert gamma_val > 0

        # Test Lambda evolution
        lambda_evolution = HLCDMCosmology.lambda_evolution(z_test)
        assert 'lambda_theoretical' in lambda_evolution

    def test_blinding_functionality(self, gamma_pipeline):
        """Test blinding implementation."""
        context = {'blinding_enabled': True}

        # Run with blinding
        result = gamma_pipeline.run(context)

        assert 'blinding_info' in result
        blinding_info = result['blinding_info']
        assert blinding_info['blinding_status'] == 'blinded'
        assert 'blinded_parameters' in blinding_info
        assert 'gamma_fundamental' in blinding_info['blinded_parameters']
        assert 'qtep_ratio' in blinding_info['blinded_parameters']

    def test_systematic_error_budget(self, gamma_pipeline):
        """Test systematic error budget."""
        result = gamma_pipeline.run()

        assert 'systematic_budget' in result
        budget = result['systematic_budget']

        assert 'total_systematic' in budget
        assert 'components' in budget
        assert isinstance(budget['total_systematic'], (int, float))
        assert budget['total_systematic'] > 0

        # Check expected components
        components = budget['components']
        expected_components = ['redshift_precision', 'model_implementation',
                             'numerical_precision', 'theoretical_approximations']
        for component in expected_components:
            assert component in components

    def test_validation_basic(self, gamma_pipeline):
        """Test basic validation."""
        result = gamma_pipeline.validate()

        assert 'null_hypothesis_test' in result
        assert 'mathematical_consistency' in result
        assert 'physical_bounds' in result
        assert 'qtep_verification' in result
        assert 'theory_self_consistency' in result

        # Check null hypothesis test
        nh_test = result['null_hypothesis_test']
        assert 'null_hypothesis' in nh_test
        assert 'null_hypothesis_rejected' in nh_test
        assert 'p_value' in nh_test

    def test_validation_extended(self, gamma_pipeline):
        """Test extended validation with large samples."""
        # Ensure results are available
        gamma_pipeline.run()
        
        context = {'n_monte_carlo': 1000, 'n_bootstrap': 100, 'random_seed': 42}  # Smaller for testing
        result = gamma_pipeline.validate_extended(context)

        assert 'monte_carlo' in result
        assert 'bootstrap' in result
        assert 'loo_cv' in result
        assert 'jackknife' in result
        assert 'model_comparison' in result
        
        mc_result = result['monte_carlo']
        assert 'test' in mc_result
        assert 'n_samples' in mc_result
        assert 'random_seed' in mc_result
        
        bootstrap_result = result['bootstrap']
        assert 'test' in bootstrap_result
        if bootstrap_result.get('passed', False):
            assert 'n_bootstrap' in bootstrap_result
            assert 'random_seed' in bootstrap_result

    def test_run_complete_pipeline(self, gamma_pipeline):
        """Test complete pipeline run."""
        result = gamma_pipeline.run()

        # Check required outputs
        required_keys = ['z_grid', 'gamma_values', 'lambda_evolution',
                        'qtep_ratio', 'theory_summary']
        for key in required_keys:
            assert key in result

        # Check data types and ranges
        assert len(result['gamma_values']) == len(result['z_grid'])
        assert all(g > 0 for g in result['gamma_values'])

        # Check redshift range
        z_grid = result['z_grid']
        assert min(z_grid) >= 0.0
        assert max(z_grid) <= 10.0

    def test_multiple_redshift_ranges(self, gamma_pipeline):
        """Test pipeline with different redshift ranges."""
        test_ranges = [
            {'z_min': 0.0, 'z_max': 1.0},
            {'z_min': 0.5, 'z_max': 1.5},
            {'z_min': 1.0, 'z_max': 2.0}
        ]

        for z_range in test_ranges:
            context = {**z_range, 'z_steps': 20}
            result = gamma_pipeline.run(context)

            z_grid = result['z_grid']
            assert min(z_grid) >= z_range['z_min']
            assert max(z_grid) <= z_range['z_max']
            assert len(z_grid) == 20

    def test_theory_validation(self, gamma_pipeline):
        """Test theoretical consistency validation."""
        result = gamma_pipeline.validate()

        # Check that all validation components are present
        assert 'mathematical_consistency' in result
        assert 'physical_bounds' in result
        assert 'qtep_verification' in result
        assert 'theory_self_consistency' in result
        
        # Check physical bounds validation
        physical_bounds = result['physical_bounds']
        assert 'passed' in physical_bounds
        assert 'gamma_today' in physical_bounds

    def test_bootstrap_validation(self, gamma_pipeline):
        """Test bootstrap validation method."""
        # Ensure results are available
        results = gamma_pipeline.run()
        gamma_pipeline.results = results
        
        result = gamma_pipeline._bootstrap_validation(10, random_seed=42)

        assert 'passed' in result
        assert 'test' in result
        if result.get('passed', False):
            assert 'n_bootstrap' in result
            assert 'bootstrap_mean' in result
            assert 'bootstrap_std' in result
            assert 'bootstrap_ci_95_lower' in result
            assert 'bootstrap_ci_95_upper' in result
            assert 'random_seed' in result

    def test_monte_carlo_validation(self, gamma_pipeline):
        """Test Monte Carlo validation method."""
        result = gamma_pipeline._monte_carlo_validation(100, random_seed=42)

        assert 'passed' in result
        assert 'test' in result
        assert 'n_samples' in result
        assert 'gamma_coefficient_of_variation' in result
        assert 'lambda_coefficient_of_variation' in result
        assert 'chi_squared' in result
        assert 'p_value' in result
        assert 'random_seed' in result

    def test_model_comparison(self, gamma_pipeline):
        """Test model comparison methods."""
        gamma_pipeline.run()
        result = gamma_pipeline._perform_model_comparison()

        assert 'comparison_available' in result
        if result.get('comparison_available'):
            assert 'lcdm' in result
            assert 'hlcdm' in result
            assert 'comparison' in result
            assert 'bayes_factor' in result['comparison']
            assert 'preferred_model' in result['comparison']
