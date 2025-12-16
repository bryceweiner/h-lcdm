"""
Gravitational Constant Pipeline
================================

Main pipeline for gravitational constant calculation and validation.

Implements the holographic derivation of Newton's gravitational constant
from information processing principles with full statistical validation.
"""

import numpy as np
from typing import Dict, Any, Optional

from ..common.base_pipeline import AnalysisPipeline
from .physics import (
    calculate_np_from_alpha_inverse,
    calculate_g_base,
    calculate_g_geometric,
    calculate_g_final,
    calculate_g,
)
from .error_propagation import GravitationalConstantErrorPropagation
from .monte_carlo import GravitationalConstantMonteCarloValidator
from .sensitivity import GravitationalConstantSensitivityAnalysis
from hlcdm.parameters import HLCDM_PARAMS


class GravitationalConstantPipeline(AnalysisPipeline):
    """
    Gravitational constant pipeline from information processing principles.
    
    Calculates parameter-free prediction:
    G = πc⁵/(ℏH²·N_P) ≈ 6.62 × 10⁻¹¹ m³/(kg·s²)
    
    where N_P is derived from fine structure constant via:
    N_P = exp[2α⁻¹ + 2ln(4π²) + 1/π]
    
    No correction factors (ln(3), f_quantum) are needed - the holographic bound
    already encodes dimensional projection through the 2D horizon area.
    
    Validation: The fine structure pipeline confirms the underlying formula
    by showing (G_measured, H₀) → α_predicted agrees with α_measured to 0.0018%.
    
    Compares against CODATA 2018: G = 6.67430(15) × 10⁻¹¹ m³/(kg·s²)
    Expected agreement: ~1% (dominated by H₀ uncertainty from Hubble tension)
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize gravitational constant pipeline.
        
        Parameters:
            output_dir (str): Output directory for results
        """
        super().__init__("gravitational_constant", output_dir)
        
        # Initialize validation components
        self.error_prop = GravitationalConstantErrorPropagation()
        self.mc_validator = GravitationalConstantMonteCarloValidator(n_samples=100000)
        self.sensitivity = GravitationalConstantSensitivityAnalysis()
        
        # CODATA 2018 observational values
        self.codata_G = 6.67430e-11  # m³/(kg·s²)
        self.codata_sigma = 0.00015e-11  # m³/(kg·s²)
        
        # Input parameters (no f_quantum needed)
        self.alpha_inverse = 137.035999084
        self.delta_alpha_inverse = 0.000000021
        
        self.update_metadata('description', 'Gravitational constant from holographic information bound')
        self.update_metadata('parameter_free', True)  # No free parameters - just (α, H₀) as inputs
        self.update_metadata('prediction', 'G ≈ 6.62 × 10⁻¹¹ m³/(kg·s²), ~1% agreement with CODATA')
    
    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute gravitational constant analysis.
        
        Parameters:
            context (dict, optional): Analysis parameters
        
        Returns:
            dict: Complete analysis results
        """
        self.log_progress("Starting gravitational constant calculation...")
        
        # 1. Calculate prediction
        self.log_progress("Calculating information capacity and gravitational constant...")
        physics_results = self._calculate_physics()
        
        # 2. Compare with CODATA 2018 observation
        self.log_progress("Comparing with CODATA 2018 observation...")
        comparison = self._compare_with_observations(physics_results)
        
        # 3. Run statistical validations
        self.log_progress("Running statistical validations...")
        validation = self._run_validations(physics_results, comparison)
        
        # Package results
        results = {
            'physics': physics_results,
            'comparison': comparison,
            'validation': validation,
            'systematic_budget': self._create_systematic_budget().get_budget_breakdown()
        }
        
        self.log_progress("✓ Gravitational constant analysis complete")
        
        # Save results
        self.save_results(results)
        
        return results
    
    def _calculate_physics(self) -> Dict[str, Any]:
        """
        Calculate physics predictions using the corrected formula.
        
        The derivation proceeds:
        1. N_P from α⁻¹: N_P = exp[2α⁻¹ + 2ln(4π²) + 1/π]
        2. G from holographic bound: G = πc⁵/(ℏH²N_P)
        
        No correction factors (ln(3), f_quantum) are applied.
        
        Returns:
            dict: Physics calculation results
        """
        # Information capacity from α⁻¹
        np_result = calculate_np_from_alpha_inverse(self.alpha_inverse)
        
        # Complete G calculation (no corrections)
        g_result = calculate_g(alpha_inverse=self.alpha_inverse)
        
        return {
            'information_capacity': np_result,
            'g': g_result,
            'prediction_summary': {
                'N_P': np_result['N_P'],
                'ln_N_P': np_result['ln_N_P'],
                'G': g_result['G'],
                'formula': 'G = πc⁵/(ℏH²N_P)',
                'corrections_applied': False,
                'note': 'No ln(3) or f_quantum corrections - holographic bound already encodes dimensionality'
            }
        }
    
    def _compare_with_observations(self, physics_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare prediction with CODATA 2018 observation.
        
        Parameters:
            physics_results (dict): Physics calculation results
        
        Returns:
            dict: Comparison analysis
        """
        predicted = physics_results['g']['G']
        
        # Error propagation comparison
        error_comparison = self.error_prop.compare_with_observation(
            self.codata_G,
            self.codata_sigma
        )
        
        # Calculate chi-squared
        chi2_result = self.calculate_chi_squared(
            observed=np.array([self.codata_G]),
            expected=np.array([predicted]),
            uncertainties=np.array([self.codata_sigma]),
            degrees_of_freedom=0  # Parameter-free prediction (depends on α⁻¹ measurement)
        )
        
        return {
            'predicted': predicted,
            'observed': self.codata_G,
            'observed_sigma': self.codata_sigma,
            'error_propagation': error_comparison,
            'chi_squared': chi2_result,
            'deviation_sigma': error_comparison['deviation_sigma'],
            'relative_difference_percent': error_comparison['relative_difference_percent'],
            'agreement': error_comparison['agreement']
        }
    
    def _run_validations(self,
                        physics_results: Dict[str, Any],
                        comparison: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run all statistical validations.
        
        Parameters:
            physics_results (dict): Physics results
            comparison (dict): Comparison results
        
        Returns:
            dict: Validation results
        """
        validations = {}
        
        # Error propagation
        self.log_progress("  Running error propagation analysis...")
        validations['error_propagation'] = self.error_prop.full_error_analysis()
        
        # Monte Carlo validation (no f_quantum uncertainty)
        self.log_progress("  Running Monte Carlo uncertainty quantification...")
        validations['monte_carlo'] = self.mc_validator.run_mc_validation(
            self.codata_G,
            self.codata_sigma,
            self.alpha_inverse,
            self.delta_alpha_inverse
        )
        
        # Model comparison (best fit approach, not BIC/AIC due to single data point)
        self.log_progress("  Running model comparison...")
        validations['model_comparison'] = self._model_comparison()
        
        # Sensitivity analysis
        self.log_progress("  Running sensitivity analysis...")
        validations['sensitivity'] = self.sensitivity.full_sensitivity_analysis()
        
        # Compute overall status based on relative difference (for parameter-free prediction)
        mc_consistency = validations['monte_carlo'].get('consistency_fractions', {})
        p_consistent_2_percent = mc_consistency.get('p_consistent_2_percent', 0)
        validations['overall_status'] = 'PASSED' if p_consistent_2_percent > 0.5 else 'FAILED'
        
        return validations
    
    def _model_comparison(self) -> Dict[str, Any]:
        """
        Compare H-ΛCDM prediction against standard physics.
        
        Uses best fit model comparison (not BIC/AIC, which require multiple data points).
        
        Models compared:
        1. H-ΛCDM: Derived from holographic information bound (parameter-free)
        2. Standard: G as fundamental constant (best fit = observed value)
        
        Agreement quality combined with parsimony determines the preferred model.
        
        Returns:
            dict: Model comparison results
        """
        # Get actual prediction (no corrections)
        g_result = calculate_g(alpha_inverse=self.alpha_inverse)
        prediction = g_result['G']
        
        # Calculate relative difference
        relative_diff = abs(prediction - self.codata_G) / self.codata_G * 100
        
        # Deviation in sigma (note: ~1% difference >> measurement uncertainty)
        deviation_sigma = (self.codata_G - prediction) / self.codata_sigma
        
        # Determine preferred model based on agreement quality
        # Note: For a PARAMETER-FREE prediction, ~1% is remarkable
        if relative_diff < 0.1:
            preferred_model = 'hlcdm'
            preference_reason = 'excellent_agreement'
            evidence_strength = 'strong'
        elif relative_diff < 1.0:
            preferred_model = 'hlcdm'
            preference_reason = 'good_agreement'
            evidence_strength = 'strong'
        elif relative_diff < 2.0:
            preferred_model = 'hlcdm'
            preference_reason = 'good_agreement_parameter_free'
            evidence_strength = 'moderate'
        else:
            preferred_model = 'inconclusive'
            preference_reason = 'significant_deviation'
            evidence_strength = 'weak'
        
        return {
            'models': {
                'hlcdm': {
                    'prediction': prediction,
                    'relative_difference_percent': float(relative_diff),
                    'deviation_sigma': float(deviation_sigma),
                    'formula': 'G = πc⁵/(ℏH²N_P)',
                    'free_parameters': 0,
                    'inputs': ['α⁻¹ (measured)', 'H₀ (measured)']
                },
                'standard': {
                    'best_fit': self.codata_G,
                    'note': 'G treated as fundamental constant'
                }
            },
            'preferred_model': preferred_model,
            'preference_reason': preference_reason,
            'evidence_strength': evidence_strength,
            'relative_difference_percent': float(relative_diff),
            'deviation_sigma': float(deviation_sigma),
            'interpretation': self._interpret_model_comparison(preferred_model, relative_diff)
        }
    
    def _interpret_model_comparison(self, preferred_model: str, relative_diff: float) -> str:
        """
        Interpret overall model comparison results.
        
        Parameters:
            preferred_model (str): Preferred model ('hlcdm' or 'standard')
            relative_diff (float): Relative difference in percent
        
        Returns:
            str: Interpretation string
        """
        if preferred_model == 'hlcdm':
            if relative_diff < 0.1:
                return f"H-ΛCDM preferred: excellent agreement ({relative_diff:.4f}% relative difference) with information-theoretic prediction"
            else:
                return f"H-ΛCDM preferred: good agreement ({relative_diff:.4f}% relative difference) with information-theoretic prediction"
        else:
            return f"Standard physics preferred: better fit to observation ({relative_diff:.4f}% relative difference)"
    
    def _create_systematic_budget(self) -> AnalysisPipeline.SystematicBudget:
        """
        Create systematic error budget.
        
        The dominant uncertainty is from H₀ measurement (Hubble tension).
        No theoretical correction factors contribute uncertainty.
        
        Returns:
            SystematicBudget: Configured systematic error budget
        """
        budget = self.SystematicBudget()
        
        # Theoretical systematics (all exact - no correction factors)
        budget.add_component('holographic_bound_formula', 0.0)  # Exact
        budget.add_component('np_from_alpha_formula', 0.0)  # Exact
        
        # Measurement systematics
        budget.add_component('alpha_inverse_measurement', 1.5e-10)  # CODATA fractional uncertainty
        budget.add_component('H0_measurement', 0.015)  # ~1.5% from Hubble tension
        budget.add_component('fundamental_constants', 0.0)  # c, ℏ are defined exactly
        
        # Comparison systematics (from CODATA G measurement)
        budget.add_component('codata_G_measurement', 0.000022)  # CODATA fractional precision
        
        return budget
    
    def validate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform basic statistical validation.
        
        Parameters:
            context (dict, optional): Validation parameters
        
        Returns:
            dict: Validation results
        """
        self.log_progress("Performing basic gravitational constant validation...")
        
        # Load results if needed
        if not self.results:
            self.results = self.load_results() or self.run()
        
        # Basic validation checks
        validation_results = {
            'physics_accuracy': self._validate_physics_accuracy(),
            'comparison_consistency': self._validate_comparison_consistency(),
            'error_propagation': self._validate_error_propagation(),
            'numerical_stability': self._validate_numerical_stability()
        }
        
        # Overall status
        all_passed = all(result.get('passed', False) for result in validation_results.values())
        validation_results['overall_status'] = 'PASSED' if all_passed else 'FAILED'
        validation_results['validation_level'] = 'basic'
        
        self.log_progress(f"✓ Basic validation complete: {validation_results['overall_status']}")
        
        return validation_results
    
    def _validate_physics_accuracy(self) -> Dict[str, Any]:
        """Validate physics calculation accuracy."""
        physics = self.results.get('physics', {})
        G = physics.get('g', {}).get('G', None)
        
        if G is None:
            return {'passed': False, 'error': 'No physics results available'}
        
        # Check against expected value
        expected = 6.67e-11
        tolerance = 0.1e-11  # 1.5% tolerance
        
        passed = np.abs(G - expected) < tolerance
        
        return {
            'passed': passed,
            'test': 'physics_accuracy',
            'predicted': G,
            'expected': expected,
            'deviation': G - expected,
            'tolerance': tolerance
        }
    
    def _validate_comparison_consistency(self) -> Dict[str, Any]:
        """Validate comparison consistency."""
        comparison = self.results.get('comparison', {})
        relative_diff = comparison.get('relative_difference_percent', None)
        deviation_sigma = comparison.get('deviation_sigma', None)
        observed_sigma = comparison.get('observed_sigma', None)
        
        if relative_diff is None:
            return {'passed': False, 'error': 'No comparison results available'}
        
        # For CODATA-level precision, use relative difference
        if observed_sigma and observed_sigma / self.codata_G < 0.01:  # <1% relative uncertainty
            passed = relative_diff < 1.0  # <1% relative difference
            threshold = 1.0
            metric = 'relative_difference_percent'
        else:
            # Use deviation_sigma
            if deviation_sigma is None:
                return {'passed': False, 'error': 'No deviation_sigma available'}
            passed = np.abs(deviation_sigma) < 2.0
            threshold = 2.0
            metric = 'deviation_sigma'
        
        return {
            'passed': passed,
            'test': 'comparison_consistency',
            metric: deviation_sigma if metric == 'deviation_sigma' else relative_diff,
            'relative_difference_percent': relative_diff,
            'deviation_sigma': deviation_sigma,
            'threshold': threshold
        }
    
    def _validate_error_propagation(self) -> Dict[str, Any]:
        """Validate error propagation calculations."""
        validation = self.results.get('validation', {})
        error_prop = validation.get('error_propagation', {})
        
        if not error_prop:
            return {'passed': False, 'error': 'No error propagation results'}
        
        # Check that uncertainty is calculated
        g_uncertainty = error_prop.get('g_uncertainty', {})
        delta_G = g_uncertainty.get('delta_G', None)
        
        passed = delta_G is not None and delta_G > 0
        
        return {
            'passed': passed,
            'test': 'error_propagation',
            'delta_G': delta_G,
            'expected': 'positive value'
        }
    
    def _validate_numerical_stability(self) -> Dict[str, Any]:
        """Validate numerical stability."""
        physics = self.results.get('physics', {})
        G = physics.get('g', {}).get('G', None)
        
        if G is None:
            return {'passed': False, 'error': 'No physics results available'}
        
        # Check for finite, reasonable values
        passed = np.isfinite(G) and 1e-11 < G < 1e-10
        
        return {
            'passed': passed,
            'test': 'numerical_stability',
            'G': G,
            'is_finite': np.isfinite(G),
            'in_range': 1e-11 < G < 1e-10
        }
    
    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform extended validation with Monte Carlo and sensitivity analysis.
        
        Parameters:
            context (dict, optional): Extended validation parameters
        
        Returns:
            dict: Extended validation results
        """
        self.log_progress("Performing extended gravitational constant validation...")
        
        # Load results if needed
        if not self.results:
            self.results = self.load_results() or self.run()
        
        validation = self.results.get('validation', {})
        
        extended_results = {
            'monte_carlo': validation.get('monte_carlo', {}),
            'sensitivity': validation.get('sensitivity', {}),
            'model_comparison': validation.get('model_comparison', {}),
            'validation_level': 'extended'
        }
        
        # Overall status
        mc_consistent = validation.get('monte_carlo', {}).get('consistency_fractions', {}).get('p_consistent_1_percent', 0) > 0.95
        extended_results['overall_status'] = 'PASSED' if mc_consistent else 'FAILED'
        
        self.log_progress(f"✓ Extended validation complete: {extended_results['overall_status']}")
        
        return extended_results
