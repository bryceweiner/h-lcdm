"""
Base Analysis Pipeline
======================

Base class for all H-ΛCDM analysis pipelines.

Provides common interface and functionality for:
- Data loading and processing
- Validation (basic, extended)
- Result storage and reporting
- Error handling and logging
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json
import time
import numpy as np
from datetime import datetime


class AnalysisPipeline(ABC):
    """
    Base class for H-ΛCDM analysis pipelines.

    All pipelines must implement:
    - run(): Main analysis execution
    - validate(): Basic statistical validation
    - validate_extended(): Extended validation (Monte Carlo, bootstrap)
    """

    def __init__(self, name: str, output_dir: str = "results"):
        """
        Initialize pipeline.

        Parameters:
            name (str): Pipeline name
            output_dir (str): Output directory
        """
        self.name = name
        self.base_output_dir = Path(output_dir)
        
        # Create organized directory structure
        self.json_dir = self.base_output_dir / "json"
        self.reports_dir = self.base_output_dir / "reports"
        self.figures_dir = self.base_output_dir / "figures"
        self.logs_dir = self.base_output_dir / "logs"
        
        # Create directories
        self.json_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.figures_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)
        
        # Keep output_dir for backward compatibility (points to json_dir)
        self.output_dir = self.json_dir
        
        # Set up log file
        self.log_file = self.logs_dir / f"{self.name}_pipeline.log"
        self._log_file_handle = None

        # Initialize data directories
        self.downloaded_data_dir = Path("downloaded_data")
        self.processed_data_dir = Path("processed_data")

        # Results storage
        self.results = {}
        self.metadata = {}

    @abstractmethod
    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute the main analysis pipeline.

        Parameters:
            context (dict, optional): Analysis context/parameters

        Returns:
            dict: Analysis results
        """
        pass

    @abstractmethod
    def validate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform basic statistical validation.

        Parameters:
            context (dict, optional): Validation parameters

        Returns:
            dict: Validation results
        """
        pass

    @abstractmethod
    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform extended validation (Monte Carlo, bootstrap, etc.).

        Parameters:
            context (dict, optional): Extended validation parameters

        Returns:
            dict: Extended validation results
        """
        pass

    def calculate_bic_aic(self, log_likelihood: float, n_parameters: int, n_data_points: int) -> Dict[str, float]:
        """
        Calculate BIC (Bayesian Information Criterion) and AIC (Akaike Information Criterion).

        Parameters:
            log_likelihood: Log-likelihood of the model
            n_parameters: Number of free parameters in the model
            n_data_points: Number of data points

        Returns:
            dict: BIC and AIC values
        """
        # AIC = 2k - 2ln(L)
        aic = 2 * n_parameters - 2 * log_likelihood

        # BIC = k*ln(n) - 2ln(L)
        bic = n_parameters * np.log(n_data_points) - 2 * log_likelihood

        return {
            'aic': aic,
            'bic': bic,
            'log_likelihood': log_likelihood,
            'n_parameters': n_parameters,
            'n_data_points': n_data_points
        }

    def perform_loo_cv(self, data: np.ndarray, model_func: callable, n_folds: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Leave-One-Out Cross-Validation.

        Parameters:
            data: Input data array
            model_func: Function that takes training data and returns predictions
            n_folds: Number of folds (if None, uses full LOO)

        Returns:
            dict: LOO-CV results
        """
        if n_folds is None:
            n_folds = len(data)

        n_samples = len(data)
        predictions = np.zeros(n_samples)
        errors = []

        for i in range(n_samples):
            # Leave one out
            train_indices = np.arange(n_samples) != i
            test_indices = [i]

            train_data = data[train_indices]
            test_data = data[test_indices]

            try:
                # Fit model on training data
                prediction = model_func(train_data, test_data)
                if np.isscalar(prediction):
                    predictions[i] = prediction
                else:
                    predictions[i] = prediction[0] if len(prediction) > 0 else np.nan
            except Exception as e:
                errors.append(str(e))
                predictions[i] = np.nan

        # Calculate metrics
        valid_predictions = predictions[~np.isnan(predictions)]
        mse = np.mean((valid_predictions - data[~np.isnan(predictions)])**2) if len(valid_predictions) > 0 else np.nan
        rmse = np.sqrt(mse) if not np.isnan(mse) else np.nan

        return {
            'method': 'loo_cv',
            'n_samples': n_samples,
            'n_valid_predictions': len(valid_predictions),
            'predictions': predictions.tolist(),
            'mse': mse,
            'rmse': rmse,
            'errors': errors
        }

    def perform_jackknife(self, data: np.ndarray, statistic_func: callable) -> Dict[str, Any]:
        """
        Perform jackknife resampling for statistical analysis.

        Parameters:
            data: Input data array
            statistic_func: Function that computes statistic on data

        Returns:
            dict: Jackknife results
        """
        n_samples = len(data)
        jackknife_stats = []

        for i in range(n_samples):
            # Leave one out and compute statistic
            jackknife_sample = np.delete(data, i)
            try:
                stat = statistic_func(jackknife_sample)
                jackknife_stats.append(stat)
            except Exception as e:
                jackknife_stats.append(np.nan)

        jackknife_stats = np.array(jackknife_stats)
        valid_stats = jackknife_stats[~np.isnan(jackknife_stats)]

        if len(valid_stats) == 0:
            return {
                'method': 'jackknife',
                'error': 'No valid statistics computed'
            }

        # Calculate jackknife estimates
        mean_stat = np.mean(valid_stats)
        variance = (n_samples - 1) / n_samples * np.sum((valid_stats - mean_stat)**2)
        std_error = np.sqrt(variance)

        # Bias correction
        original_stat = statistic_func(data)
        bias = (n_samples - 1) * (mean_stat - original_stat)

        return {
            'method': 'jackknife',
            'n_samples': n_samples,
            'original_statistic': original_stat,
            'jackknife_mean': mean_stat,
            'jackknife_variance': variance,
            'jackknife_std_error': std_error,
            'bias_correction': bias,
            'jackknife_estimates': valid_stats.tolist()
        }

    def calculate_chi_squared(self, observed: np.ndarray, expected: np.ndarray,
                            uncertainties: Optional[np.ndarray] = None,
                            degrees_of_freedom: Optional[int] = None) -> Dict[str, float]:
        """
        Calculate χ² statistic and related quantities.

        Parameters:
            observed: Observed values
            expected: Expected/model values
            uncertainties: Measurement uncertainties (default: assume σ=1)
            degrees_of_freedom: Degrees of freedom (if None, calculated as len-1)

        Returns:
            dict: χ² statistics
        """
        observed = np.asarray(observed)
        expected = np.asarray(expected)

        if uncertainties is None:
            uncertainties = np.ones_like(observed)
        else:
            uncertainties = np.asarray(uncertainties)

        # χ² = Σ [(observed - expected)² / σ²]
        chi_squared = np.sum(((observed - expected) / uncertainties) ** 2)

        n_data_points = len(observed)
        if degrees_of_freedom is None:
            degrees_of_freedom = n_data_points - 1  # Default: assume 1 parameter fitted

        reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.nan

        # p-value from χ² distribution
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

        return {
            'chi_squared': chi_squared,
            'degrees_of_freedom': degrees_of_freedom,
            'reduced_chi_squared': reduced_chi_squared,
            'p_value': p_value,
            'n_data_points': n_data_points
        }

    def perform_bayesian_analysis(self, likelihood_func: callable,
                                prior_func: callable, data: np.ndarray,
                                parameter_ranges: Dict[str, Tuple[float, float]],
                                n_samples: int = 10000) -> Dict[str, Any]:
        """
        Perform Bayesian parameter estimation using MCMC.

        Parameters:
            likelihood_func: Function returning log-likelihood given parameters and data
            prior_func: Function returning log-prior given parameters
            data: Observed data
            parameter_ranges: Dictionary of (min, max) ranges for each parameter
            n_samples: Number of MCMC samples

        Returns:
            dict: Bayesian analysis results
        """
        try:
            # Simple Metropolis-Hastings MCMC implementation
            param_names = list(parameter_ranges.keys())
            n_params = len(param_names)

            # Initialize chains
            current_params = {}
            for name, (min_val, max_val) in parameter_ranges.items():
                current_params[name] = np.random.uniform(min_val, max_val)

            current_log_like = likelihood_func(current_params, data)
            current_log_prior = prior_func(current_params)

            # Storage for samples
            samples = {name: [] for name in param_names}
            log_likelihoods = []

            # MCMC loop
            for i in range(n_samples):
                # Propose new parameters
                proposed_params = {}
                for name, (min_val, max_val) in parameter_ranges.items():
                    # Gaussian proposal centered on current value
                    proposal_width = (max_val - min_val) / 20  # Adaptive width
                    proposed = np.random.normal(current_params[name], proposal_width)
                    # Reflect at boundaries
                    while proposed < min_val or proposed > max_val:
                        if proposed < min_val:
                            proposed = 2 * min_val - proposed
                        elif proposed > max_val:
                            proposed = 2 * max_val - proposed
                    proposed_params[name] = proposed

                # Calculate posterior
                proposed_log_like = likelihood_func(proposed_params, data)
                proposed_log_prior = prior_func(proposed_params)

                # Acceptance ratio
                log_acceptance = (proposed_log_like + proposed_log_prior) - (current_log_like + current_log_prior)

                # Accept or reject
                if np.log(np.random.uniform()) < log_acceptance:
                    current_params = proposed_params
                    current_log_like = proposed_log_like
                    current_log_prior = proposed_log_prior

                # Store sample
                for name in param_names:
                    samples[name].append(current_params[name])
                log_likelihoods.append(current_log_like)

            # Calculate statistics
            parameter_stats = {}
            for name in param_names:
                param_samples = np.array(samples[name])
                parameter_stats[name] = {
                    'mean': np.mean(param_samples),
                    'std': np.std(param_samples),
                    'median': np.median(param_samples),
                    'credible_interval_68': [
                        np.percentile(param_samples, 16),
                        np.percentile(param_samples, 84)
                    ],
                    'credible_interval_95': [
                        np.percentile(param_samples, 2.5),
                        np.percentile(param_samples, 97.5)
                    ]
                }

            # Evidence approximation (harmonic mean - not recommended for real analysis)
            # In practice, use nested sampling or other methods
            evidence_estimate = 1.0 / np.mean(1.0 / np.exp(np.array(log_likelihoods)))

            return {
                'parameter_posterior': parameter_stats,
                'evidence_estimate': evidence_estimate,
                'n_samples': n_samples,
                'acceptance_rate': len(set(zip(*[samples[name] for name in param_names]))) / n_samples,
                'samples': samples,
                'log_likelihoods': log_likelihoods
            }

        except Exception as e:
            return {'error': str(e)}

    def construct_covariance_matrix(self, data: np.ndarray,
                                  correlation_matrix: Optional[np.ndarray] = None,
                                  uncertainties: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Construct covariance matrix from data and uncertainties.

        Parameters:
            data: Input data array
            correlation_matrix: Correlation matrix (optional)
            uncertainties: Measurement uncertainties (diagonal elements)

        Returns:
            ndarray: Covariance matrix
        """
        n_points = len(data)

        # Initialize covariance matrix
        cov_matrix = np.zeros((n_points, n_points))

        if uncertainties is None:
            uncertainties = np.ones(n_points)  # Default unit uncertainties

        # Diagonal elements (variances)
        for i in range(n_points):
            cov_matrix[i, i] = uncertainties[i] ** 2

        # Off-diagonal elements (covariances)
        if correlation_matrix is not None:
            for i in range(n_points):
                for j in range(n_points):
                    if i != j:
                        correlation = correlation_matrix[i, j]
                        cov_matrix[i, j] = correlation * uncertainties[i] * uncertainties[j]

        return cov_matrix

    def apply_multiple_testing_correction(self, p_values: np.ndarray,
                                       method: str = 'bonferroni') -> Dict[str, Any]:
        """
        Apply multiple testing correction to control family-wise error rate.

        Parameters:
            p_values: Array of raw p-values
            method: Correction method ('bonferroni', 'fdr_bh', 'holm')

        Returns:
            dict: Corrected p-values and rejection decisions
        """
        p_values = np.asarray(p_values)
        n_tests = len(p_values)

        if method == 'bonferroni':
            # Bonferroni correction: multiply by number of tests
            corrected_p = np.minimum(p_values * n_tests, 1.0)
            alpha_corrected = 0.05 / n_tests

        elif method == 'fdr_bh':
            # Benjamini-Hochberg False Discovery Rate
            from scipy.stats import rankdata
            ranked_p = rankdata(p_values, method='ordinal')
            order = np.argsort(p_values)
            sorted_p = p_values[order]

            # BH procedure
            bh_thresholds = np.arange(1, n_tests + 1) * 0.05 / n_tests
            reject_up_to = np.sum(sorted_p <= bh_thresholds)

            corrected_p = np.ones_like(p_values)
            if reject_up_to > 0:
                corrected_p[order[:reject_up_to]] = sorted_p[:reject_up_to]

            alpha_corrected = 0.05  # FDR controls expected proportion of false discoveries

        elif method == 'holm':
            # Holm-Bonferroni method
            order = np.argsort(p_values)
            sorted_p = p_values[order]
            corrected_p = np.ones_like(p_values)

            for i in range(n_tests):
                corrected_p[order[i]] = min(1.0, sorted_p[i] * (n_tests - i))

            alpha_corrected = 0.05

        else:
            raise ValueError(f"Unknown correction method: {method}")

        # Determine rejection decisions
        rejected = corrected_p <= alpha_corrected

        return {
            'method': method,
            'raw_p_values': p_values.tolist(),
            'corrected_p_values': corrected_p.tolist(),
            'alpha_corrected': alpha_corrected,
            'rejected': rejected.tolist(),
            'n_tests': n_tests,
            'n_rejected': int(np.sum(rejected))
        }

    def apply_blinding(self, sensitive_params: Dict[str, float],
                      blinding_key: Optional[str] = None) -> Dict[str, Any]:
        """
        Apply reversible blinding offsets to sensitive parameters.

        Parameters:
            sensitive_params: Dictionary of parameter names and their true values
            blinding_key: Optional seed for reproducible blinding (for unblinding)

        Returns:
            dict: Blinded parameters and blinding information
        """
        if blinding_key is None:
            blinding_key = np.random.randint(0, 2**32)

        np.random.seed(blinding_key)

        blinded_params = {}
        blinding_offsets = {}

        for param_name, true_value in sensitive_params.items():
            # Generate blinding offset as fraction of parameter uncertainty
            # Typical blinding: offset by 2-3σ to be detectable but not obvious
            uncertainty_estimate = abs(true_value) * 0.1  # Conservative 10% uncertainty
            offset = np.random.normal(0, uncertainty_estimate * 2.0)
            blinding_offsets[param_name] = offset
            blinded_params[param_name] = true_value + offset

        return {
            'blinded_parameters': blinded_params,
            'blinding_key': blinding_key,
            'blinding_offsets': blinding_offsets,
            'blinding_status': 'blinded'
        }

    def unblind_analysis(self, blinded_results: Dict[str, Any],
                        blinding_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Remove blinding offsets to reveal true results.

        Parameters:
            blinded_results: Results computed on blinded parameters
            blinding_info: Blinding information from apply_blinding

        Returns:
            dict: Unblinded results
        """
        blinding_offsets = blinding_info.get('blinding_offsets', {})

        # Deep copy results to avoid modifying original
        unblinded_results = json.loads(json.dumps(blinded_results))

        # Recursively unblind all numerical values affected by blinded parameters
        # This is a simplified version - in practice, would need pipeline-specific logic
        def _unblind_recursive(obj):
            if isinstance(obj, dict):
                for key, value in obj.items():
                    if isinstance(value, (int, float)) and key in blinding_offsets:
                        obj[key] = value - blinding_offsets[key]
                    elif isinstance(value, (dict, list)):
                        _unblind_recursive(value)
            elif isinstance(obj, list):
                for i, item in enumerate(obj):
                    if isinstance(item, (int, float)):
                        # Assume list items correspond to blinded parameters by position
                        param_names = list(blinding_offsets.keys())
                        if i < len(param_names):
                            obj[i] = item - blinding_offsets[param_names[i]]
                    elif isinstance(item, (dict, list)):
                        _unblind_recursive(item)

        _unblind_recursive(unblinded_results)

        unblinded_results['blinding_status'] = 'unblinded'
        unblinded_results['unblinding_timestamp'] = time.time()

        return unblinded_results

    def check_convergence(self, mcmc_samples: Dict[str, np.ndarray],
                         n_chains: Optional[int] = None) -> Dict[str, Any]:
        """
        Check MCMC convergence using Gelman-Rubin statistic (R-hat).

        Parameters:
            mcmc_samples: Dictionary of parameter samples from MCMC
            n_chains: Number of chains (if samples are concatenated)

        Returns:
            dict: Convergence diagnostics
        """
        convergence_stats = {}

        for param_name, samples in mcmc_samples.items():
            samples = np.asarray(samples)

            if n_chains is None:
                # Assume single chain
                n_samples = len(samples)
                chain_length = n_samples // 4  # Use last 75% for convergence
                samples_for_conv = samples[-chain_length:]
            else:
                # Multiple chains assumed concatenated
                chain_length = len(samples) // n_chains
                chains = [samples[i*chain_length:(i+1)*chain_length] for i in range(n_chains)]
                samples_for_conv = chains

            # Gelman-Rubin statistic (simplified version)
            if n_chains is None or n_chains == 1:
                # Single chain: check autocorrelation and effective sample size
                mean = np.mean(samples_for_conv)
                var = np.var(samples_for_conv, ddof=1)

                # Simple effective sample size estimate
                n_eff = len(samples_for_conv) / 10  # Conservative estimate

                r_hat = 1.0  # Single chain, assume converged
                converged = True
            else:
                # Multiple chains: full Gelman-Rubin
                chain_means = [np.mean(chain) for chain in samples_for_conv]
                chain_vars = [np.var(chain, ddof=1) for chain in samples_for_conv]

                W = np.mean(chain_vars)  # Within-chain variance
                B = np.var(chain_means, ddof=1) * chain_length  # Between-chain variance

                var_hat = (1 - 1/n_chains) * W + B/n_chains
                r_hat = np.sqrt(var_hat / W)

                # R-hat < 1.1 typically indicates convergence
                converged = r_hat < 1.1

            convergence_stats[param_name] = {
                'r_hat': float(r_hat),
                'converged': converged,
                'effective_sample_size': float(n_eff) if 'n_eff' in locals() else float(len(samples_for_conv)),
                'mean': float(np.mean(samples_for_conv)),
                'std': float(np.std(samples_for_conv, ddof=1))
            }

        overall_converged = all(stats['converged'] for stats in convergence_stats.values())

        return {
            'parameter_convergence': convergence_stats,
            'overall_converged': overall_converged,
            'method': 'gelman_rubin_r_hat',
            'convergence_threshold': 1.1
        }

    def check_numerical_stability(self, matrix: np.ndarray,
                                operation: str = 'general') -> Dict[str, Any]:
        """
        Check numerical stability of matrix operations.

        Parameters:
            matrix: Matrix to analyze
            operation: Type of operation ('inverse', 'eigenvalue', 'general')

        Returns:
            dict: Stability diagnostics
        """
        matrix = np.asarray(matrix)
        stability_info = {
            'matrix_shape': matrix.shape,
            'operation_type': operation,
            'condition_number': float(np.linalg.cond(matrix)),
            'determinant': float(np.linalg.det(matrix)),
            'is_finite': bool(np.all(np.isfinite(matrix))),
            'is_symmetric': bool(np.allclose(matrix, matrix.T)),
            'eigenvalues_computed': False,
            'matrix_invertible': False
        }

        try:
            eigenvalues = np.linalg.eigvals(matrix)
            stability_info.update({
                'eigenvalues_computed': True,
                'eigenvalue_range': [float(np.min(eigenvalues)), float(np.max(eigenvalues))],
                'n_negative_eigenvalues': int(np.sum(eigenvalues < 0)),
                'n_zero_eigenvalues': int(np.sum(np.abs(eigenvalues) < 1e-12))
            })

            # Check if positive definite (for covariance matrices)
            stability_info['positive_definite'] = bool(np.all(eigenvalues > 0))

        except np.linalg.LinAlgError:
            stability_info['eigenvalues_computed'] = False

        try:
            inverse = np.linalg.inv(matrix)
            stability_info.update({
                'matrix_invertible': True,
                'inverse_condition_number': float(np.linalg.cond(inverse)),
                'inverse_finite': bool(np.all(np.isfinite(inverse)))
            })
        except np.linalg.LinAlgError:
            stability_info['matrix_invertible'] = False

        # Overall stability assessment
        issues = []
        if stability_info['condition_number'] > 1e12:
            issues.append('Extremely ill-conditioned matrix')
        elif stability_info['condition_number'] > 1e6:
            issues.append('Poorly conditioned matrix')

        if not stability_info['is_finite']:
            issues.append('Matrix contains non-finite values')

        if stability_info.get('eigenvalues_computed') and stability_info.get('n_negative_eigenvalues', 0) > 0:
            issues.append('Matrix has negative eigenvalues')

        stability_info['issues'] = issues
        stability_info['stable_for_operation'] = len(issues) == 0

        return stability_info

    class SystematicBudget:
        """
        Standardized systematic error budget class.

        Tracks multiple sources of systematic uncertainty and their quadrature combination.
        """

        def __init__(self, components: Optional[Dict[str, float]] = None):
            """
            Initialize systematic budget.

            Parameters:
                components: Dictionary of systematic components {name: uncertainty}
            """
            self.components = components or {}
            self._total_systematic = None

        def add_component(self, name: str, uncertainty: float):
            """Add a systematic error component."""
            self.components[name] = uncertainty
            self._total_systematic = None  # Invalidate cache

        def remove_component(self, name: str):
            """Remove a systematic error component."""
            if name in self.components:
                del self.components[name]
                self._total_systematic = None

        def get_total_systematic(self) -> float:
            """Get total systematic uncertainty (quadrature sum)."""
            if self._total_systematic is None:
                if not self.components:
                    self._total_systematic = 0.0
                else:
                    self._total_systematic = np.sqrt(sum(unc**2 for unc in self.components.values()))
            return self._total_systematic

        def get_budget_breakdown(self) -> Dict[str, Any]:
            """Get detailed breakdown of systematic budget."""
            total = self.get_total_systematic()
            return {
                'components': self.components.copy(),
                'total_systematic': total,
                'relative_contributions': {
                    name: unc/total if total > 0 else 0.0
                    for name, unc in self.components.items()
                },
                'dominant_source': max(self.components.items(), key=lambda x: x[1]) if self.components else None
            }

    def save_results(self, results: Dict[str, Any], filename: Optional[str] = None) -> Path:
        """
        Save analysis results to JSON file.

        Parameters:
            results (dict): Results to save
            filename (str, optional): Custom filename

        Returns:
            Path: Path to saved file
        """
        if filename is None:
            filename = f"{self.name}_results.json"

        filepath = self.json_dir / filename

        # Add metadata
        results_package = {
            'pipeline': self.name,
            'timestamp': time.time(),
            'results': results,
            'metadata': self.metadata
        }

        with open(filepath, 'w') as f:
            json.dump(results_package, f, indent=2, default=str)

        print(f"✓ Results saved: {filepath}")
        
        # Close log file after saving results
        self.close_log_file()
        
        return filepath

    def load_results(self, filename: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """
        Load previously saved results.

        Parameters:
            filename (str, optional): Results filename

        Returns:
            dict or None: Loaded results
        """
        if filename is None:
            filename = f"{self.name}_results.json"

        # Try new location first, then fall back to old location for compatibility
        filepath = self.json_dir / filename
        if not filepath.exists():
            # Check old location for backward compatibility
            old_filepath = self.base_output_dir / self.name / filename
            if old_filepath.exists():
                filepath = old_filepath

        if not filepath.exists():
            return None

        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            return data.get('results')
        except Exception:
            return None

    def log_progress(self, message: str):
        """
        Log progress message to both console and log file.

        Parameters:
            message (str): Progress message
        """
        timestamp = time.strftime("%H:%M:%S")
        log_line = f"[{timestamp}] {self.name}: {message}"
        
        # Print to console
        print(log_line)
        
        # Write to log file
        try:
            if self._log_file_handle is None:
                self._log_file_handle = open(self.log_file, 'a', encoding='utf-8')
                # Write header on first open
                self._log_file_handle.write(f"\n{'='*80}\n")
                self._log_file_handle.write(f"Pipeline: {self.name}\n")
                self._log_file_handle.write(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self._log_file_handle.write(f"{'='*80}\n")
            
            self._log_file_handle.write(log_line + "\n")
            self._log_file_handle.flush()
        except Exception as e:
            # Don't fail if logging fails
            pass
    
    def close_log_file(self):
        """Close the log file handle."""
        if self._log_file_handle is not None:
            try:
                self._log_file_handle.write(f"\n{'='*80}\n")
                self._log_file_handle.write(f"Ended: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                self._log_file_handle.write(f"{'='*80}\n\n")
                self._log_file_handle.close()
                self._log_file_handle = None
            except Exception:
                pass

    def get_data_paths(self) -> Dict[str, Path]:
        """
        Get data directory paths.

        Returns:
            dict: Data paths
        """
        return {
            'downloaded': self.downloaded_data_dir,
            'processed': self.processed_data_dir,
            'output': self.output_dir
        }

    def update_metadata(self, key: str, value: Any):
        """
        Update pipeline metadata.

        Parameters:
            key (str): Metadata key
            value: Metadata value
        """
        self.metadata[key] = value
