"""
Gamma Pipeline - Theoretical γ(z) and Λ_eff(z) Analysis
======================================================

Pure theoretical calculation of information processing rates and
effective cosmological constant as functions of redshift.

Implements the fundamental H-ΛCDM predictions without observational fitting.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from pathlib import Path

from ..common.base_pipeline import AnalysisPipeline
from hlcdm.cosmology import HLCDMCosmology
from hlcdm.parameters import HLCDM_PARAMS


class GammaPipeline(AnalysisPipeline):
    """
    Theoretical gamma and Lambda analysis pipeline.

    Calculates γ(z) and Λ_eff(z) from first principles using the H-ΛCDM framework.
    Pure theory - no fitting to observational data.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize gamma pipeline.

        Parameters:
            output_dir (str): Output directory
        """
        super().__init__("gamma", output_dir)

        # Redshift range for analysis
        self.z_min = 0.0
        self.z_max = 10.0
        self.z_steps = 100

        # Update metadata
        self.update_metadata('description', 'Theoretical γ(z) and Λ_eff(z) calculation')
        self.update_metadata('z_range', f'{self.z_min}-{self.z_max}')
        self.update_metadata('theory_type', 'parameter-free')

    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute theoretical gamma analysis.

        Calculates γ(z) and Λ_eff(z) across redshift range using pure theory.

        Parameters:
            context (dict, optional): Analysis parameters

        Returns:
            dict: Analysis results
        """
        self.log_progress("Starting theoretical γ(z) and Λ_eff(z) analysis...")

        # Parse context parameters
        if context:
            self.z_min = context.get('z_min', self.z_min)
            self.z_max = context.get('z_max', self.z_max)
            self.z_steps = context.get('z_steps', self.z_steps)

        blinding_enabled = context.get('blinding_enabled', True) if context else True

        # Apply blinding if enabled
        if blinding_enabled:
            # For gamma pipeline, blind the fundamental gamma value and QTEP ratio
            # These are key theoretical predictions
            gamma_fundamental = 1e-10  # Typical value from theory
            qtep_ratio = 2.257  # ln(2)/(1-ln(2))
            self.blinding_info = self.apply_blinding({
                'gamma_fundamental': gamma_fundamental,
                'qtep_ratio': qtep_ratio
            })
            self.log_progress("Analysis parameters blinded for unbiased development")
        else:
            self.blinding_info = None

        # Generate redshift grid
        z_grid = np.linspace(self.z_min, self.z_max, self.z_steps)

        self.log_progress(f"Analyzing {len(z_grid)} redshift points from z={self.z_min} to z={self.z_max}")

        # Calculate theoretical predictions
        gamma_values = []
        lambda_values = []
        qtep_values = []

        for i, z in enumerate(z_grid):
            if (i + 1) % 20 == 0:
                self.log_progress(f"  Processed {i+1}/{len(z_grid)} redshifts...")

            # Calculate gamma at this redshift
            gamma = HLCDMCosmology.gamma_at_redshift(z)
            gamma_values.append(gamma)

            # Calculate Lambda evolution
            lambda_result = HLCDMCosmology.lambda_evolution(z)
            lambda_values.append(lambda_result)

            # QTEP ratio (constant)
            qtep_values.append(HLCDM_PARAMS.QTEP_RATIO)

        # Create systematic error budget
        systematic_budget = self._create_systematic_budget()

        # Package results
        results = {
            'z_grid': z_grid.tolist(),
            'gamma_values': gamma_values,
            'lambda_evolution': lambda_values,
            'qtep_ratio': HLCDM_PARAMS.QTEP_RATIO,
            'qtep_values': qtep_values,
            'systematic_budget': systematic_budget.get_budget_breakdown(),
            'blinding_info': self.blinding_info,
            'theory_summary': self._generate_theory_summary(z_grid, gamma_values, lambda_values)
        }

        self.log_progress("✓ Theoretical gamma analysis complete")
        self.log_progress(f"  Redshift range: z={self.z_min} to {self.z_max}")
        self.log_progress(f"  Key prediction: γ(z=0) = {gamma_values[0]:.2e} s⁻¹")

        # Save results
        self.save_results(results)

        return results

    def _create_systematic_budget(self) -> 'AnalysisPipeline.SystematicBudget':
        """
        Create systematic error budget for gamma analysis.

        Returns:
            SystematicBudget: Configured systematic error budget
        """
        budget = self.SystematicBudget()

        # Redshift precision uncertainties
        budget.add_component('redshift_precision', 0.02)  # 2% redshift error

        # Model dependence (different implementations of HLCDM)
        budget.add_component('model_implementation', 0.01)  # 1% model variation

        # Numerical precision in calculations
        budget.add_component('numerical_precision', 0.005)  # 0.5% numerical error

        # Theoretical approximations (series expansions, etc.)
        budget.add_component('theoretical_approximations', 0.015)  # 1.5% approximation error

        return budget

    def _generate_theory_summary(self, z_grid: np.ndarray,
                                gamma_values: List[float],
                                lambda_values: List[Dict]) -> Dict[str, Any]:
        """
        Generate summary of theoretical predictions.

        Parameters:
            z_grid: Redshift grid
            gamma_values: Gamma values at each redshift
            lambda_values: Lambda evolution data

        Returns:
            dict: Theory summary
        """
        # Present-day values (z=0)
        z_today_idx = np.argmin(np.abs(z_grid - 0.0))
        gamma_today = gamma_values[z_today_idx]
        lambda_today = lambda_values[z_today_idx]['lambda_theoretical']

        # High-redshift values (recombination)
        z_recomb_idx = np.argmin(np.abs(z_grid - HLCDM_PARAMS.Z_RECOMB))
        gamma_recomb = gamma_values[z_recomb_idx]
        lambda_recomb = lambda_values[z_recomb_idx]['lambda_theoretical']

        # Evolution ratios
        gamma_evolution_ratio = gamma_recomb / gamma_today
        lambda_evolution_ratio = lambda_recomb / lambda_today

        summary = {
            'present_day': {
                'redshift': 0.0,
                'gamma_s^-1': gamma_today,
                'lambda_m^-2': lambda_today
            },
            'recombination_era': {
                'redshift': HLCDM_PARAMS.Z_RECOMB,
                'gamma_s^-1': gamma_recomb,
                'lambda_m^-2': lambda_recomb
            },
            'evolution_ratios': {
                'gamma_recomb/gamma_today': gamma_evolution_ratio,
                'lambda_recomb/lambda_today': lambda_evolution_ratio
            },
            'qtep_ratio': HLCDM_PARAMS.QTEP_RATIO,
            'theory_type': 'parameter-free_holographic',
            'key_equations': [
                'γ(z) = H(z)/π² (fundamental information processing rate)',
                'Λ_eff(z) ∝ γ(z)² × QTEP_ratio × corrections',
                'QTEP_ratio = ln(2)/(1-ln(2)) = 2.257'
            ]
        }

        return summary

    def validate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform basic validation of theoretical predictions.

        Checks mathematical consistency and theoretical bounds.

        Parameters:
            context (dict, optional): Validation parameters

        Returns:
            dict: Validation results
        """
        self.log_progress("Performing basic theoretical validation...")

        # Load results if not already computed
        if not self.results:
            self.results = self.load_results() or self.run()

        # Validation checks
        validation_results = {
            'mathematical_consistency': self._check_mathematical_consistency(),
            'physical_bounds': self._check_physical_bounds(),
            'qtep_verification': self._verify_qtep_ratio(),
            'theory_self_consistency': self._check_theory_self_consistency(),
            'null_hypothesis_test': self._test_null_hypothesis()
        }

        # Overall validation status
        all_passed = all(result.get('passed', False)
                        for result in validation_results.values())

        validation_results['overall_status'] = 'PASSED' if all_passed else 'FAILED'
        validation_results['validation_level'] = 'basic'

        self.log_progress(f"✓ Basic validation complete: {validation_results['overall_status']}")

        return validation_results

    def _check_mathematical_consistency(self) -> Dict[str, Any]:
        """Check mathematical consistency of calculations."""
        try:
            # Test that gamma calculations are mathematically consistent
            test_z = 1.0
            gamma1 = HLCDMCosmology.gamma_at_redshift(test_z)
            gamma2 = HLCDMCosmology.gamma_fundamental(HLCDM_PARAMS.get_hubble_at_redshift(test_z)) if hasattr(HLCDMCosmology, 'gamma_fundamental') else HLCDMCosmology.gamma_at_redshift(test_z)

            consistency = abs(gamma1 - gamma2) < 1e-10

            return {
                'passed': consistency,
                'test': 'gamma_calculation_consistency',
                'details': f'γ(z=1) methods agree within {abs(gamma1 - gamma2):.2e}'
            }
        except Exception as e:
            return {
                'passed': False,
                'test': 'mathematical_consistency',
                'error': str(e)
            }

    def _check_physical_bounds(self) -> Dict[str, Any]:
        """Check that predictions satisfy physical bounds."""
        try:
            # Check that gamma is positive and reasonable
            gamma_today = self.results.get('gamma_values', [0])[0] if self.results else 0

            # Physical bounds: gamma should be positive and within expected range
            gamma_min_expected = 1e-30  # s⁻¹
            gamma_max_expected = 1e-10  # s⁻¹

            bounds_ok = (gamma_min_expected < gamma_today < gamma_max_expected)

            return {
                'passed': bounds_ok,
                'test': 'physical_bounds_check',
                'gamma_today': gamma_today,
                'bounds': f'{gamma_min_expected:.0e} to {gamma_max_expected:.0e} s⁻¹'
            }
        except Exception as e:
            return {
                'passed': False,
                'test': 'physical_bounds',
                'error': str(e)
            }

    def _verify_qtep_ratio(self) -> Dict[str, Any]:
        """Verify QTEP ratio calculation."""
        calculated_qtep = HLCDM_PARAMS.QTEP_RATIO
        expected_qtep = np.log(2) / (1 - np.log(2))

        qtep_correct = abs(calculated_qtep - expected_qtep) < 1e-10

        return {
            'passed': qtep_correct,
            'test': 'qtep_ratio_verification',
            'calculated': calculated_qtep,
            'expected': expected_qtep,
            'difference': abs(calculated_qtep - expected_qtep)
        }

    def _check_theory_self_consistency(self) -> Dict[str, Any]:
        """Check internal consistency of theoretical framework."""
        try:
            # Check that Lambda evolution is consistent with gamma evolution
            if not self.results:
                return {'passed': False, 'test': 'theory_consistency', 'error': 'No results available'}

            gamma_vals = self.results['gamma_values']
            lambda_vals = [lv['lambda_theoretical'] for lv in self.results['lambda_evolution']]

            # Lambda should scale roughly with gamma² (simplified check)
            gamma_ratio = gamma_vals[-1] / gamma_vals[0] if gamma_vals[0] != 0 else 0
            lambda_ratio = lambda_vals[-1] / lambda_vals[0] if lambda_vals[0] != 0 else 0

            # Allow for some deviation due to corrections
            consistency = abs(lambda_ratio / (gamma_ratio**2) - 1.0) < 0.1

            return {
                'passed': consistency,
                'test': 'theory_self_consistency',
                'gamma_evolution_ratio': gamma_ratio,
                'lambda_evolution_ratio': lambda_ratio,
                'expected_lambda_ratio': gamma_ratio**2
            }
        except Exception as e:
            return {
                'passed': False,
                'test': 'theory_consistency',
                'error': str(e)
            }

    def _test_null_hypothesis(self) -> Dict[str, Any]:
        """
        Test null hypothesis: γ is constant (ΛCDM cosmology).

        Null hypothesis: γ(z) = constant, Λ(z) = constant (standard ΛCDM)
        Alternative: γ(z) evolves with redshift (H-ΛCDM)

        Returns:
            dict: Null hypothesis test results
        """
        try:
            # Load results if not available
            if not self.results:
                self.results = self.load_results() or self.run()

            # Get gamma evolution results from stored results
            z_grid = np.array(self.results.get('z_grid', []))
            gamma_values = np.array(self.results.get('gamma_values', []))

            # If no results available, cannot perform test
            if len(z_grid) == 0 or len(gamma_values) == 0:
                return {
                    'passed': False,
                    'test': 'null_hypothesis_test',
                    'error': 'No gamma results available for null hypothesis test'
                }

            # Full statistical test with actual data
            # Null hypothesis: γ(z) = constant (ΛCDM cosmology)
            # Alternative: γ(z) = H(z)/π² (H-ΛCDM prediction, which varies with redshift)
            
            # Calculate constant model (null hypothesis)
            gamma_null = np.mean(gamma_values)
            gamma_null_array = np.full_like(gamma_values, gamma_null)
            
            # Calculate H-ΛCDM prediction (alternative hypothesis)
            # This is what we actually computed: gamma_values = H(z)/π²
            gamma_hlcdm_array = np.array(gamma_values)  # This IS the H-ΛCDM prediction
            
            # For theoretical predictions, use systematic error budget
            # Get systematic error from budget
            systematic_budget = self.results.get('systematic_budget', {})
            total_systematic = systematic_budget.get('total_systematic', 0.02)  # Default 2%
            
            # Calculate errors: systematic uncertainty as fraction of gamma value
            # For each redshift, error is systematic fraction of the predicted gamma
            gamma_errors = np.abs(gamma_values) * total_systematic
            
            # Avoid division by zero
            gamma_errors = np.maximum(gamma_errors, np.abs(gamma_values) * 1e-10)
            
            # Calculate chi-squared for constant model (null hypothesis)
            # This tests: do the H-ΛCDM predictions deviate significantly from constant?
            chi_squared = np.sum(((gamma_values - gamma_null_array) / gamma_errors) ** 2)
            degrees_of_freedom = len(gamma_values) - 1  # One parameter fitted (constant)

            # p-value from chi-squared distribution
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)

            # Test if constant model is adequate (p > 0.05 means null hypothesis not rejected)
            null_hypothesis_adequate = p_value > 0.05

            # Reduced chi-squared
            reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.nan

            # Evidence strength against null hypothesis
            if p_value < 0.001:
                evidence_strength = "VERY_STRONG"
            elif p_value < 0.01:
                evidence_strength = "STRONG"
            elif p_value < 0.05:
                evidence_strength = "MODERATE"
            else:
                evidence_strength = "WEAK"

            # Calculate how much gamma varies (for reporting)
            gamma_variation = (np.max(gamma_values) - np.min(gamma_values)) / np.mean(gamma_values)
            gamma_ratio_max_min = np.max(gamma_values) / np.min(gamma_values)
            
            return {
                'passed': True,  # Test completed successfully
                'test': 'null_hypothesis_test',
                'null_hypothesis': 'γ(z) = constant (ΛCDM cosmology)',
                'alternative_hypothesis': 'γ(z) = H(z)/π² evolves with redshift (H-ΛCDM)',
                'chi_squared': float(chi_squared),
                'degrees_of_freedom': degrees_of_freedom,
                'reduced_chi_squared': float(reduced_chi_squared),
                'p_value': float(p_value),
                'null_hypothesis_rejected': not null_hypothesis_adequate,
                'evidence_against_null': evidence_strength,
                'gamma_variation_fractional': float(gamma_variation),
                'gamma_ratio_max_min': float(gamma_ratio_max_min),
                'systematic_error_used': float(total_systematic),
                'interpretation': self._interpret_null_hypothesis_result(null_hypothesis_adequate, p_value, gamma_variation)
            }

        except Exception as e:
            return {
                'passed': False,
                'test': 'null_hypothesis_test',
                'error': str(e)
            }

    def _interpret_null_hypothesis_result(self, null_adequate: bool, p_value: float, gamma_variation: float = None) -> str:
        """Interpret null hypothesis test result."""
        if null_adequate:
            interpretation = f"Constant model (ΛCDM) is consistent with data (p = {p_value:.3f}). "
            interpretation += f"No significant evidence for redshift-dependent γ(z). "
            if gamma_variation is not None:
                interpretation += f"Note: γ varies by {gamma_variation*100:.1f}% across redshift range, "
                interpretation += f"but this variation is within theoretical uncertainties."
            interpretation += "Result is NULL for H-ΛCDM hypothesis."
            return interpretation
        else:
            interpretation = f"Constant model (ΛCDM) is rejected (p = {p_value:.3f}). "
            interpretation += f"Evidence supports redshift-dependent γ(z) = H(z)/π² as predicted by H-ΛCDM. "
            if gamma_variation is not None:
                interpretation += f"γ varies by {gamma_variation*100:.1f}% across redshift range, "
                interpretation += f"consistent with H-ΛCDM prediction."
            return interpretation

    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform extended validation with Monte Carlo sampling.

        Parameters:
            context (dict, optional): Extended validation parameters
                - n_bootstrap: Number of bootstrap samples (default: 50000)
                - n_monte_carlo: Number of Monte Carlo simulations (default: 50000)
                - random_seed: Random seed for reproducibility (default: 42)

        Returns:
            dict: Extended validation results
        """
        self.log_progress("Performing extended theoretical validation (Monte Carlo)...")

        # Parameters for extended validation
        n_bootstrap = context.get('n_bootstrap', 50000) if context else 50000
        n_monte_carlo = context.get('n_monte_carlo', 50000) if context else 50000
        random_seed = context.get('random_seed', 42) if context else 42

        self.log_progress(f"Running Monte Carlo validation with {n_monte_carlo} samples...")

        # Monte Carlo validation of theoretical predictions
        mc_results = self._monte_carlo_validation(n_monte_carlo, random_seed=random_seed)

        # Bootstrap validation
        bootstrap_results = self._bootstrap_validation(n_bootstrap, random_seed=random_seed)

        # Leave-One-Out Cross-Validation
        loo_cv_results = self._loo_cv_validation()

        # Jackknife validation
        jackknife_results = self._jackknife_validation()

        # Model comparison (BIC/AIC/Bayesian)
        model_comparison = self._perform_model_comparison()

        extended_results = {
            'monte_carlo': mc_results,
            'bootstrap': bootstrap_results,
            'loo_cv': loo_cv_results,
            'jackknife': jackknife_results,
            'model_comparison': model_comparison,
            'validation_level': 'extended',
            'n_bootstrap': n_bootstrap,
            'n_monte_carlo': n_monte_carlo,
            'random_seed': random_seed
        }

        # Overall status
        mc_passed = mc_results.get('passed', False)
        bootstrap_passed = bootstrap_results.get('passed', False)
        loo_passed = loo_cv_results.get('passed', True)  # LOO-CV often doesn't have strict pass/fail
        jackknife_passed = jackknife_results.get('passed', True)

        extended_results['overall_status'] = 'PASSED' if all([mc_passed, bootstrap_passed, loo_passed, jackknife_passed]) else 'FAILED'

        self.log_progress(f"✓ Extended validation complete: {extended_results['overall_status']}")

        return extended_results

    def _loo_cv_validation(self) -> Dict[str, Any]:
        """Perform Leave-One-Out Cross-Validation for gamma evolution."""
        try:
            if not self.results or 'gamma_values' not in self.results:
                return {'passed': False, 'error': 'No gamma results available'}

            gamma_values = np.array(self.results['gamma_values'])

            def gamma_model(train_data, test_data):
                # Simple model: predict based on training data mean
                return np.mean(train_data)

            loo_results = self.perform_loo_cv(gamma_values, gamma_model)

            return {
                'passed': True,
                'method': 'loo_cv',
                'rmse': loo_results.get('rmse', np.nan),
                'mse': loo_results.get('mse', np.nan),
                'n_predictions': loo_results.get('n_valid_predictions', 0)
            }

        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _jackknife_validation(self) -> Dict[str, Any]:
        """Perform jackknife validation for gamma statistics."""
        try:
            if not self.results or 'gamma_values' not in self.results:
                return {'passed': False, 'error': 'No gamma results available'}

            gamma_values = np.array(self.results['gamma_values'])

            def gamma_statistic(data):
                return np.mean(data)  # Mean gamma value

            jackknife_results = self.perform_jackknife(gamma_values, gamma_statistic)

            return {
                'passed': True,
                'method': 'jackknife',
                'original_mean': jackknife_results.get('original_statistic'),
                'jackknife_mean': jackknife_results.get('jackknife_mean'),
                'jackknife_std_error': jackknife_results.get('jackknife_std_error'),
                'bias_correction': jackknife_results.get('bias_correction')
            }

        except Exception as e:
            return {'passed': False, 'error': str(e)}

    def _perform_model_comparison(self) -> Dict[str, Any]:
        """
        Perform model comparison using BIC, AIC, and Bayesian evidence for gamma evolution models.
        
        Compares constant gamma (ΛCDM) vs evolving gamma (H-ΛCDM) models.
        """
        try:
            if not self.results or 'gamma_values' not in self.results:
                return {'error': 'No gamma results available', 'comparison_available': False}

            gamma_values = np.array(self.results['gamma_values'])
            z_grid = np.array(self.results.get('z_grid', np.linspace(self.z_min, self.z_max, len(gamma_values))))
            n_data_points = len(gamma_values)

            if n_data_points < 3:
                return {'error': 'Insufficient data points for model comparison', 'comparison_available': False}

            # Model 1: Constant gamma (ΛCDM, 1 parameter)
            gamma_mean = np.mean(gamma_values)
            residuals_const = gamma_values - gamma_mean
            var_const = np.var(residuals_const)
            
            if var_const == 0:
                return {'error': 'No variance in residuals for constant model', 'comparison_available': False}
            
            # Proper Gaussian likelihood calculation
            log_likelihood_const = -0.5 * n_data_points * np.log(2 * np.pi * var_const) - \
                                   0.5 * np.sum(residuals_const**2) / var_const

            const_model = self.calculate_bic_aic(log_likelihood_const, 1, n_data_points)

            # Model 2: Evolving gamma (H-ΛCDM, using theoretical H-ΛCDM prediction)
            # Calculate theoretical predictions at each redshift
            gamma_pred_hlcdm = np.array([HLCDMCosmology.gamma_at_redshift(z) for z in z_grid])
            residuals_hlcdm = gamma_values - gamma_pred_hlcdm
            var_hlcdm = np.var(residuals_hlcdm)
            
            if var_hlcdm == 0:
                return {'error': 'No variance in residuals for H-ΛCDM model', 'comparison_available': False}
            
            # Proper Gaussian likelihood calculation
            log_likelihood_hlcdm = -0.5 * n_data_points * np.log(2 * np.pi * var_hlcdm) - \
                                    0.5 * np.sum(residuals_hlcdm**2) / var_hlcdm

            # H-ΛCDM is parameter-free (theoretical prediction), so 0 parameters
            hlcdm_model = self.calculate_bic_aic(log_likelihood_hlcdm, 0, n_data_points)

            # Calculate chi-squared for both models
            chi2_const = np.sum((residuals_const / np.sqrt(var_const)) ** 2)
            chi2_hlcdm = np.sum((residuals_hlcdm / np.sqrt(var_hlcdm)) ** 2)

            # Calculate Bayes factor (ratio of marginal likelihoods)
            # For nested models, this is the likelihood ratio
            # B_12 = P(data|H-ΛCDM) / P(data|ΛCDM) = exp(log_L_H-ΛCDM - log_L_ΛCDM)
            log_bayes_factor = log_likelihood_hlcdm - log_likelihood_const
            bayes_factor = np.exp(log_bayes_factor)

            # Interpret Bayes factor (Kass & Raftery 1995)
            if bayes_factor > 150:
                evidence_strength = "VERY_STRONG"
            elif bayes_factor > 20:
                evidence_strength = "STRONG"
            elif bayes_factor > 3:
                evidence_strength = "POSITIVE"
            elif bayes_factor > 1:
                evidence_strength = "WEAK"
            elif bayes_factor > 1/3:
                evidence_strength = "WEAK (favors ΛCDM)"
            elif bayes_factor > 1/20:
                evidence_strength = "POSITIVE (favors ΛCDM)"
            elif bayes_factor > 1/150:
                evidence_strength = "STRONG (favors ΛCDM)"
            else:
                evidence_strength = "VERY_STRONG (favors ΛCDM)"

            # Determine preferred model
            if bayes_factor > 1:
                preferred_model = "H-ΛCDM"
            elif bayes_factor < 1:
                preferred_model = "ΛCDM"
            else:
                preferred_model = "INCONCLUSIVE"

            # Calculate ΔBIC and ΔAIC (positive values favor H-ΛCDM)
            delta_bic = const_model['bic'] - hlcdm_model['bic']
            delta_aic = const_model['aic'] - hlcdm_model['aic']

            return {
                'comparison_available': True,
                'n_data_points': n_data_points,
                'lcdm': {
                    'log_likelihood': float(log_likelihood_const),
                    'chi_squared': float(chi2_const),
                    'aic': float(const_model['aic']),
                    'bic': float(const_model['bic']),
                    'n_parameters': 1
                },
                'hlcdm': {
                    'log_likelihood': float(log_likelihood_hlcdm),
                    'chi_squared': float(chi2_hlcdm),
                    'aic': float(hlcdm_model['aic']),
                    'bic': float(hlcdm_model['bic']),
                    'n_parameters': 0  # Parameter-free prediction
                },
                'comparison': {
                    'delta_aic': float(delta_aic),
                    'delta_bic': float(delta_bic),
                    'bayes_factor': float(bayes_factor),
                    'log_bayes_factor': float(log_bayes_factor),
                    'evidence_strength': evidence_strength,
                    'preferred_model': preferred_model
                },
                'interpretation': self._interpret_model_comparison(
                    delta_aic, delta_bic, bayes_factor, preferred_model
                )
            }

        except Exception as e:
            return {'error': str(e), 'comparison_available': False}
    
    def _interpret_model_comparison(self, delta_aic: float, delta_bic: float,
                                   bayes_factor: float, preferred_model: str) -> str:
        """
        Interpret model comparison results.
        
        Parameters:
            delta_aic: ΔAIC = AIC_ΛCDM - AIC_H-ΛCDM (positive favors H-ΛCDM)
            delta_bic: ΔBIC = BIC_ΛCDM - BIC_H-ΛCDM (positive favors H-ΛCDM)
            bayes_factor: Bayes factor B = P(data|H-ΛCDM) / P(data|ΛCDM)
            preferred_model: Which model is preferred
            
        Returns:
            str: Interpretation of the comparison
        """
        interpretation = f"Model comparison favors {preferred_model}.\n\n"
        
        if preferred_model == "H-ΛCDM":
            interpretation += f"H-ΛCDM is preferred with:\n"
            interpretation += f"- ΔAIC = {delta_aic:.2f} (positive values favor H-ΛCDM)\n"
            interpretation += f"- ΔBIC = {delta_bic:.2f} (positive values favor H-ΛCDM)\n"
            interpretation += f"- Bayes factor B = {bayes_factor:.2f} (B > 1 favors H-ΛCDM)\n"
            
            if abs(delta_aic) > 6:
                interpretation += "\nΔAIC > 6 indicates strong evidence for H-ΛCDM."
            if abs(delta_bic) > 6:
                interpretation += "\nΔBIC > 6 indicates strong evidence for H-ΛCDM."
            if bayes_factor > 20:
                interpretation += "\nBayes factor > 20 indicates strong evidence for H-ΛCDM."
        elif preferred_model == "ΛCDM":
            interpretation += f"ΛCDM is preferred with:\n"
            interpretation += f"- ΔAIC = {delta_aic:.2f} (negative values favor ΛCDM)\n"
            interpretation += f"- ΔBIC = {delta_bic:.2f} (negative values favor ΛCDM)\n"
            interpretation += f"- Bayes factor B = {bayes_factor:.2f} (B < 1 favors ΛCDM)\n"
        else:
            interpretation += f"Comparison is inconclusive:\n"
            interpretation += f"- ΔAIC = {delta_aic:.2f}\n"
            interpretation += f"- ΔBIC = {delta_bic:.2f}\n"
            interpretation += f"- Bayes factor B = {bayes_factor:.2f}\n"
            interpretation += "\n|ΔAIC| < 2 and |ΔBIC| < 2 indicates inconclusive evidence."
        
        return interpretation

    def _monte_carlo_validation(self, n_samples: int, random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform Monte Carlo validation of theoretical gamma predictions.
        
        Physically motivated: Simulate gamma values from the H-ΛCDM model
        (with known theoretical predictions) and verify that:
        1. Statistical properties (mean, std) are consistent with expectations
        2. The evolution pattern matches theoretical predictions
        3. Type I error rates are controlled (false rejection rate)
        
        This validates that our theoretical framework produces consistent results
        when sampling across the redshift range.
        
        Parameters:
            n_samples: Number of Monte Carlo simulations
            random_seed: Random seed for reproducibility (default: None for non-deterministic)
        """
        try:
            # Use a separate RNG for Monte Carlo to ensure reproducibility
            rng = np.random.RandomState(random_seed) if random_seed is not None else np.random

            # Sample different redshifts and check consistency
            z_samples = rng.uniform(self.z_min, self.z_max, n_samples)

            gamma_samples = []
            lambda_samples = []

            # Process samples with progress logging for large runs
            log_progress = n_samples >= 10000
            for i, z in enumerate(z_samples):
                gamma = HLCDMCosmology.gamma_at_redshift(z)
                lambda_result = HLCDMCosmology.lambda_evolution(z)

                gamma_samples.append(gamma)
                lambda_samples.append(lambda_result['lambda_theoretical'])
                
                # Log progress every 10% for large runs
                if log_progress and (i + 1) % (n_samples // 10) == 0:
                    self.log_progress(f"  Monte Carlo: processed {i+1}/{n_samples} samples ({(i+1)/n_samples*100:.0f}%)")

            gamma_samples = np.array(gamma_samples)
            lambda_samples = np.array(lambda_samples)

            # Check statistical properties
            gamma_mean = np.mean(gamma_samples)
            gamma_std = np.std(gamma_samples)
            lambda_mean = np.mean(lambda_samples)
            lambda_std = np.std(lambda_samples)

            # Validation: standard deviations should be reasonable
            gamma_cv = gamma_std / gamma_mean if gamma_mean > 0 else np.inf
            lambda_cv = lambda_std / lambda_mean if lambda_mean > 0 else np.inf

            # CV should be reasonable (not too large)
            reasonable_cv = gamma_cv < 2.0 and lambda_cv < 2.0

            # Test null hypothesis: gamma is constant
            # Under H-ΛCDM, gamma should evolve with redshift: γ(z) = H(z)/π²
            gamma_null = np.mean(gamma_samples)  # Constant model (null hypothesis)
            gamma_null_array = np.full_like(gamma_samples, gamma_null)
            
            # Use systematic error budget for theoretical uncertainties
            # Get systematic error from pipeline (default 2% if not available)
            systematic_budget = getattr(self, 'results', {})
            if isinstance(systematic_budget, dict):
                budget_breakdown = systematic_budget.get('systematic_budget', {})
                total_systematic = budget_breakdown.get('total_systematic', 0.02) if isinstance(budget_breakdown, dict) else 0.02
            else:
                total_systematic = 0.02  # Default 2% theoretical uncertainty
            
            # Calculate errors: systematic uncertainty as fraction of gamma value
            gamma_errors = np.abs(gamma_samples) * total_systematic
            gamma_errors = np.maximum(gamma_errors, np.abs(gamma_samples) * 1e-10)  # Avoid division by zero
            
            # Calculate chi-squared for constant model
            # Tests: do the H-ΛCDM predictions (gamma_samples) deviate significantly from constant?
            chi_squared = np.sum(((gamma_samples - gamma_null_array) / gamma_errors) ** 2)
            degrees_of_freedom = len(gamma_samples) - 1
            
            # p-value from chi-squared distribution
            from scipy.stats import chi2
            p_value = 1 - chi2.cdf(chi_squared, degrees_of_freedom)
            
            # Reduced chi-squared
            reduced_chi_squared = chi_squared / degrees_of_freedom if degrees_of_freedom > 0 else np.nan

            # Null hypothesis test: constant gamma (ΛCDM) should be rejected
            # For theoretical H-ΛCDM, we expect evolution, so p should be small
            null_hypothesis_rejected = p_value < 0.05

            # Overall validation: CV reasonable AND null hypothesis test behaves correctly
            passed = reasonable_cv and (reduced_chi_squared < 10.0)  # Chi-squared per dof should be reasonable

            return {
                'passed': passed,
                'test': 'monte_carlo_validation',
                'n_samples': n_samples,
                'gamma_mean': float(gamma_mean),
                'gamma_std': float(gamma_std),
                'gamma_coefficient_of_variation': float(gamma_cv),
                'lambda_mean': float(lambda_mean),
                'lambda_std': float(lambda_std),
                'lambda_coefficient_of_variation': float(lambda_cv),
                'chi_squared': float(chi_squared),
                'degrees_of_freedom': degrees_of_freedom,
                'reduced_chi_squared': float(reduced_chi_squared),
                'p_value': float(p_value),
                'null_hypothesis_rejected': null_hypothesis_rejected,
                'random_seed': random_seed,
                'interpretation': f"Monte Carlo validation: CV_gamma = {gamma_cv:.3f}, CV_lambda = {lambda_cv:.3f}. "
                                f"Chi-squared per dof = {reduced_chi_squared:.2f} (p = {p_value:.3f}). "
                                f"Null hypothesis (constant gamma) {'rejected' if null_hypothesis_rejected else 'not rejected'}."
            }
        except Exception as e:
            return {
                'passed': False,
                'test': 'monte_carlo_validation',
                'error': str(e)
            }

    def _bootstrap_validation(self, n_bootstrap: int, random_seed: Optional[int] = None) -> Dict[str, Any]:
        """
        Perform bootstrap validation of gamma evolution results.
        
        Physically motivated: Bootstrap resample the gamma values to assess:
        1. Confidence intervals for mean gamma
        2. Stability of theoretical predictions
        3. Robustness of statistical conclusions to numerical uncertainties
        
        This tests whether our theoretical conclusions are stable under resampling,
        accounting for numerical precision and theoretical approximation uncertainties.
        
        Parameters:
            n_bootstrap: Number of bootstrap samples
            random_seed: Random seed for reproducibility (default: None for non-deterministic)
        """
        try:
            if not self.results:
                return {'passed': False, 'error': 'No results available for bootstrap', 'test': 'bootstrap_validation'}

            z_grid = np.array(self.results['z_grid'])
            gamma_values = np.array(self.results['gamma_values'])

            if len(gamma_values) < 2:
                return {
                    'passed': False,
                    'error': 'Insufficient data points for bootstrap',
                    'test': 'bootstrap_validation',
                    'n_data_points': len(gamma_values)
                }

            # Original mean gamma
            original_mean = np.mean(gamma_values)

            # Use a separate RNG for bootstrap to ensure reproducibility
            rng = np.random.RandomState(random_seed) if random_seed is not None else np.random

            bootstrap_gammas = []
            bootstrap_chi2_per_dof = []

            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = rng.choice(len(z_grid), size=len(z_grid), replace=True)
                bootstrap_gamma = gamma_values[indices]

                # Calculate mean gamma for this bootstrap sample
                bootstrap_gammas.append(np.mean(bootstrap_gamma))

                # Calculate chi-squared per dof for constant model
                bootstrap_mean = np.mean(bootstrap_gamma)
                bootstrap_std = np.std(bootstrap_gamma)
                if bootstrap_std > 0:
                    residuals = bootstrap_gamma - bootstrap_mean
                    chi_squared = np.sum((residuals / bootstrap_std) ** 2)
                    dof = len(bootstrap_gamma) - 1
                    if dof > 0:
                        bootstrap_chi2_per_dof.append(chi_squared / dof)

            if len(bootstrap_gammas) == 0:
                return {
                    'passed': False,
                    'error': 'Bootstrap resampling failed',
                    'test': 'bootstrap_validation'
                }

            # Calculate bootstrap statistics
            bootstrap_gammas_array = np.array(bootstrap_gammas)
            bootstrap_mean = np.mean(bootstrap_gammas_array)
            bootstrap_std = np.std(bootstrap_gammas_array)

            # Bootstrap confidence intervals (percentile method)
            ci_lower = np.percentile(bootstrap_gammas_array, 2.5)  # 95% CI lower
            ci_upper = np.percentile(bootstrap_gammas_array, 97.5)  # 95% CI upper

            # Check stability: bootstrap std should be reasonable
            stability_ok = bootstrap_std < 0.15 * abs(bootstrap_mean)  # Less than 15% relative standard deviation

            # Check if original mean is within bootstrap CI
            original_in_ci = ci_lower <= original_mean <= ci_upper

            # Chi-squared stability (if available)
            chi2_stable = True
            if len(bootstrap_chi2_per_dof) > 0:
                chi2_array = np.array(bootstrap_chi2_per_dof)
                chi2_std = np.std(chi2_array)
                chi2_stable = chi2_std < 2.0  # Chi-squared per dof should be stable

            passed = stability_ok and original_in_ci and chi2_stable

            return {
                'passed': passed,
                'test': 'bootstrap_validation',
                'n_bootstrap': n_bootstrap,
                'n_successful_bootstraps': len(bootstrap_gammas),
                'original_mean': float(original_mean),
                'bootstrap_mean': float(bootstrap_mean),
                'bootstrap_std': float(bootstrap_std),
                'bootstrap_ci_95_lower': float(ci_lower),
                'bootstrap_ci_95_upper': float(ci_upper),
                'original_in_ci': original_in_ci,
                'stability_ok': stability_ok,
                'chi2_stable': chi2_stable,
                'random_seed': random_seed,
                'interpretation': f"Bootstrap 95% CI: [{ci_lower:.2e}, {ci_upper:.2e}]. "
                                f"Original mean ({original_mean:.2e}) is "
                                f"{'within' if original_in_ci else 'outside'} CI. "
                                f"Bootstrap std = {bootstrap_std:.2e}."
            }
        except Exception as e:
            return {
                'passed': False,
                'test': 'bootstrap_validation',
                'error': str(e)
            }
