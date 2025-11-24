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
        assert gamma_pipeline.theoretical_gamma > 0
        assert gamma_pipeline.qtep_ratio > 1

    def test_theoretical_calculations(self, gamma_pipeline):
        """Test theoretical gamma and Lambda calculations."""
        z_test = 1.0

        # Test gamma calculation
        gamma_val = gamma_pipeline.theoretical_gamma
        assert isinstance(gamma_val, (int, float))
        assert gamma_val > 0

        # Test Lambda evolution
        lambda_evolution = gamma_pipeline._calculate_lambda_evolution(z_test)
        assert 'lambda_theoretical' in lambda_evolution
        assert 'qtep_ratio' in lambda_evolution

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
        assert 'theory_validation' in result

        # Check null hypothesis test
        nh_test = result['null_hypothesis_test']
        assert 'null_hypothesis' in nh_test
        assert 'null_hypothesis_rejected' in nh_test
        assert 'p_value' in nh_test

    def test_validation_extended(self, gamma_pipeline):
        """Test extended validation with large samples."""
        context = {'n_monte_carlo': 1000}  # Smaller for testing
        result = gamma_pipeline.validate_extended(context)

        assert 'monte_carlo' in result
        mc_result = result['monte_carlo']
        assert 'method' in mc_result
        assert 'n_simulations' in mc_result

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
        assert max(z_grid) <= 2.0

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

        theory_validation = result['theory_validation']
        assert 'qtep_consistent' in theory_validation
        assert 'gamma_range_valid' in theory_validation
        assert 'lambda_evolution_smooth' in theory_validation

    def test_parameter_ranges(self, gamma_pipeline):
        """Test parameter range validation."""
        # Test with extreme parameter values
        gamma_pipeline.theoretical_gamma = 1e-20  # Very small
        result = gamma_pipeline.validate()

        # Should detect invalid parameters
        assert result['theory_validation']['gamma_range_valid'] is False

        # Reset to valid value
        gamma_pipeline.theoretical_gamma = 1e-10  # Valid range
        result = gamma_pipeline.validate()
        assert result['theory_validation']['gamma_range_valid'] is True
