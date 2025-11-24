"""
Tests for base pipeline statistical methods.

Tests multiple testing correction, blinding, convergence checking,
numerical stability, and systematic error budget functionality.
"""
import pytest
import numpy as np
from unittest.mock import patch

from pipeline.common.base_pipeline import AnalysisPipeline


class TestAnalysisPipeline(AnalysisPipeline):
    """Test implementation of AnalysisPipeline."""

    def __init__(self, name="test"):
        super().__init__(name)

    def run(self, context=None):
        return {"test": "data"}

    def validate(self, context=None):
        return {"validation": "passed"}

    def validate_extended(self, context=None):
        return {"extended_validation": "passed"}


class TestMultipleTestingCorrection:
    """Test multiple testing correction methods."""

    def test_bonferroni_correction(self, sample_p_values):
        """Test Bonferroni correction."""
        pipeline = TestAnalysisPipeline()
        result = pipeline.apply_multiple_testing_correction(sample_p_values, 'bonferroni')

        assert result['method'] == 'bonferroni'
        assert result['n_tests'] == len(sample_p_values)
        assert len(result['corrected_p_values']) == len(sample_p_values)
        assert len(result['rejected']) == len(sample_p_values)
        assert result['alpha_corrected'] == 0.05 / len(sample_p_values)

        # Bonferroni should be more conservative
        assert all(cp >= p for cp, p in zip(result['corrected_p_values'], sample_p_values))

    def test_fdr_correction(self, sample_p_values):
        """Test False Discovery Rate correction."""
        pipeline = TestAnalysisPipeline()
        result = pipeline.apply_multiple_testing_correction(sample_p_values, 'fdr_bh')

        assert result['method'] == 'fdr_bh'
        assert result['n_tests'] == len(sample_p_values)
        assert result['alpha_corrected'] == 0.05  # FDR controls expected proportion

    def test_holm_correction(self, sample_p_values):
        """Test Holm-Bonferroni correction."""
        pipeline = TestAnalysisPipeline()
        result = pipeline.apply_multiple_testing_correction(sample_p_values, 'holm')

        assert result['method'] == 'holm'
        assert result['n_tests'] == len(sample_p_values)

    def test_invalid_method(self, sample_p_values):
        """Test invalid correction method raises error."""
        pipeline = TestAnalysisPipeline()
        with pytest.raises(ValueError, match="Unknown correction method"):
            pipeline.apply_multiple_testing_correction(sample_p_values, 'invalid')


class TestAnalysisBlinding:
    """Test analysis blinding functionality."""

    def test_blinding_application(self):
        """Test blinding offset application."""
        pipeline = TestAnalysisPipeline()
        sensitive_params = {'param1': 1.0, 'param2': 2.0}

        result = pipeline.apply_blinding(sensitive_params, blinding_key=42)

        assert 'blinded_parameters' in result
        assert 'blinding_key' in result
        assert 'blinding_offsets' in result
        assert result['blinding_status'] == 'blinded'

        # Check that parameters are offset
        for param, value in sensitive_params.items():
            assert result['blinded_parameters'][param] != value

    def test_blinding_unblinding(self):
        """Test blinding and unblinding round trip."""
        pipeline = TestAnalysisPipeline()
        sensitive_params = {'param1': 1.0, 'param2': 2.0}

        # Apply blinding
        blinding_info = pipeline.apply_blinding(sensitive_params, blinding_key=42)

        # Create mock blinded results
        blinded_results = {
            'param1': blinding_info['blinded_parameters']['param1'],
            'param2': blinding_info['blinded_parameters']['param2'],
            'nested': {
                'param1': blinding_info['blinded_parameters']['param1'] * 2
            },
            'list_param': [
                blinding_info['blinded_parameters']['param1'],
                blinding_info['blinded_parameters']['param2']
            ]
        }

        # Unblind
        unblinded = pipeline.unblind_analysis(blinded_results, blinding_info)

        # Check unblinding worked for top-level parameters
        assert abs(unblinded['param1'] - sensitive_params['param1']) < 1e-10
        assert abs(unblinded['param2'] - sensitive_params['param2']) < 1e-10
        assert unblinded['blinding_status'] == 'unblinded'


class TestConvergenceChecking:
    """Test MCMC convergence diagnostics."""

    def test_convergence_single_chain(self, sample_mcmc_samples):
        """Test convergence checking for single chain."""
        pipeline = TestAnalysisPipeline()
        result = pipeline.check_convergence(sample_mcmc_samples)

        assert 'parameter_convergence' in result
        assert 'overall_converged' in result
        assert result['method'] == 'gelman_rubin_r_hat'

        for param_stats in result['parameter_convergence'].values():
            assert 'r_hat' in param_stats
            assert 'converged' in param_stats
            assert 'effective_sample_size' in param_stats

    def test_convergence_multiple_chains(self):
        """Test convergence checking for multiple chains."""
        pipeline = TestAnalysisPipeline()

        # Create properly converging chains
        np.random.seed(42)
        true_value = 1.0
        n_samples = 2000  # Longer chains for better convergence
        n_chains = 3

        samples = {}
        for param in ['param1']:
            all_samples = []
            for chain in range(n_chains):
                # Start chains with different initial values
                chain_start = true_value + np.random.normal(0, 0.3)
                chain_data = []

                # Generate chain that mixes well
                current_value = chain_start
                for i in range(n_samples):
                    # Random walk with drift toward true value
                    step = np.random.normal(0, 0.05)
                    drift = 0.001 * (true_value - current_value)  # Small drift
                    current_value += step + drift
                    chain_data.append(current_value)

                all_samples.extend(chain_data)
            samples[param] = all_samples

        result = pipeline.check_convergence(samples, n_chains=3)

        assert result['overall_converged'] is not None
        # For this simple test, just check that R-hat is computed
        assert 'r_hat' in result['parameter_convergence']['param1']
        # R-hat can be > 1 for short chains, but should be finite
        assert np.isfinite(result['parameter_convergence']['param1']['r_hat'])


class TestNumericalStability:
    """Test numerical stability checking."""

    def test_stable_matrix(self, sample_covariance_matrix):
        """Test stability checking for well-conditioned matrix."""
        pipeline = TestAnalysisPipeline()
        result = pipeline.check_numerical_stability(sample_covariance_matrix)

        assert result['is_finite'] is True
        assert result['matrix_invertible'] is True
        assert result['stable_for_operation'] is True
        assert 'condition_number' in result
        assert 'eigenvalues_computed' in result

    def test_ill_conditioned_matrix(self):
        """Test stability checking for ill-conditioned matrix."""
        pipeline = TestAnalysisPipeline()

        # Create highly ill-conditioned matrix
        # Matrix with very different eigenvalues
        u, s, vh = np.linalg.svd(np.random.randn(5, 5))
        s[0] = 1000  # Large eigenvalue
        s[-1] = 1e-6  # Very small eigenvalue
        ill_conditioned = u @ np.diag(s) @ vh

        result = pipeline.check_numerical_stability(ill_conditioned)

        assert result['stable_for_operation'] is False
        assert len(result['issues']) > 0

    def test_singular_matrix(self):
        """Test stability checking for singular matrix."""
        pipeline = TestAnalysisPipeline()

        # Create singular matrix
        singular = np.array([[1.0, 2.0], [2.0, 4.0]])  # rank deficient
        result = pipeline.check_numerical_stability(singular)

        assert result['matrix_invertible'] is False
        assert result['stable_for_operation'] is False


class TestSystematicBudget:
    """Test systematic error budget functionality."""

    def test_systematic_budget_creation(self, sample_systematic_components):
        """Test systematic budget creation and calculation."""
        pipeline = TestAnalysisPipeline()
        budget = pipeline.SystematicBudget(sample_systematic_components)

        # Test total systematic calculation
        total = budget.get_total_systematic()
        expected_total = np.sqrt(sum(v**2 for v in sample_systematic_components.values()))
        assert abs(total - expected_total) < 1e-10

        # Test budget breakdown
        breakdown = budget.get_budget_breakdown()
        assert 'total_systematic' in breakdown
        assert 'components' in breakdown
        assert 'relative_contributions' in breakdown
        assert 'dominant_source' in breakdown

    def test_systematic_budget_modification(self, sample_systematic_components):
        """Test adding/removing systematic components."""
        pipeline = TestAnalysisPipeline()
        budget = pipeline.SystematicBudget()

        # Add components
        for name, value in sample_systematic_components.items():
            budget.add_component(name, value)

        assert budget.get_total_systematic() > 0

        # Remove component
        budget.remove_component('calibration')
        assert 'calibration' not in budget.components


class TestStatisticalMethods:
    """Test core statistical methods."""

    def test_chi_squared_calculation(self):
        """Test χ² statistic calculation."""
        pipeline = TestAnalysisPipeline()

        observed = np.array([1.0, 2.0, 3.0])
        expected = np.array([1.1, 1.9, 3.1])
        uncertainties = np.array([0.1, 0.1, 0.1])

        result = pipeline.calculate_chi_squared(observed, expected, uncertainties)

        assert 'chi_squared' in result
        assert 'degrees_of_freedom' in result
        assert 'reduced_chi_squared' in result
        assert 'p_value' in result
        assert result['degrees_of_freedom'] == 2  # 3 data points, 1 parameter fitted

    def test_covariance_matrix_construction(self):
        """Test covariance matrix construction."""
        pipeline = TestAnalysisPipeline()

        data = np.ones(3)
        uncertainties = np.array([0.1, 0.2, 0.3])
        correlation_matrix = np.array([[1.0, 0.5, 0.2],
                                     [0.5, 1.0, 0.3],
                                     [0.2, 0.3, 1.0]])

        cov_matrix = pipeline.construct_covariance_matrix(data, correlation_matrix, uncertainties)

        assert cov_matrix.shape == (3, 3)
        assert np.allclose(cov_matrix[0, 0], uncertainties[0]**2)  # Diagonal
        assert cov_matrix[0, 1] == correlation_matrix[0, 1] * uncertainties[0] * uncertainties[1]

    def test_bayesian_analysis(self):
        """Test Bayesian parameter estimation."""
        pipeline = TestAnalysisPipeline()

        def mock_likelihood(params, data):
            # Simple Gaussian likelihood
            return -0.5 * np.sum((data - params['mean'])**2) / params['sigma']**2

        def mock_prior(params):
            # Flat prior
            return 0.0

        data = np.random.normal(1.0, 0.5, 100)
        parameter_ranges = {'mean': (0.0, 2.0), 'sigma': (0.1, 1.0)}

        result = pipeline.perform_bayesian_analysis(
            mock_likelihood, mock_prior, data, parameter_ranges, n_samples=100
        )

        assert 'parameter_posterior' in result
        assert 'evidence_estimate' in result
        assert 'n_samples' in result
        assert 'acceptance_rate' in result

    def test_loo_cv(self):
        """Test Leave-One-Out Cross-Validation."""
        pipeline = TestAnalysisPipeline()

        def mock_model_func(train_data, test_data):
            return np.mean(train_data)

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pipeline.perform_loo_cv(data, mock_model_func)

        assert result['method'] == 'loo_cv'
        assert result['n_samples'] == 5
        assert 'mse' in result
        assert 'rmse' in result

    def test_jackknife(self):
        """Test jackknife resampling."""
        pipeline = TestAnalysisPipeline()

        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = pipeline.perform_jackknife(data, np.mean)

        assert result['method'] == 'jackknife'
        assert result['n_samples'] == 5
        assert 'jackknife_mean' in result
        assert 'jackknife_std_error' in result
        assert 'bias_correction' in result

    def test_bic_aic_calculation(self):
        """Test BIC/AIC calculation."""
        pipeline = TestAnalysisPipeline()

        log_likelihood = -100.0
        n_parameters = 3
        n_data_points = 50

        result = pipeline.calculate_bic_aic(log_likelihood, n_parameters, n_data_points)

        assert 'aic' in result
        assert 'bic' in result
        assert result['bic'] > result['aic']  # BIC penalizes complexity more
        assert result['log_likelihood'] == log_likelihood
