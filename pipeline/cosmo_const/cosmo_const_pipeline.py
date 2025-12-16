"""
Cosmological Constant Pipeline
================================

Main pipeline for cosmological constant calculation and validation.

Implements the holographic resolution of the cosmological constant problem
through causal diamond triality with full statistical validation.
"""

import numpy as np
from typing import Dict, Any, Optional
from scipy import stats

from ..common.base_pipeline import AnalysisPipeline
from .physics import (
    calculate_geometric_entropy,
    calculate_irreversibility_fraction,
    calculate_omega_lambda,
    calculate_lambda,
)
from .error_propagation import ErrorPropagation
from .monte_carlo import CosmoConstMonteCarloValidator
from .sensitivity import SensitivityAnalysis
from hlcdm.parameters import HLCDM_PARAMS


class CosmoConstPipeline(AnalysisPipeline):
    """
    Cosmological constant pipeline from causal diamond triality.
    
    Calculates parameter-free prediction:
    Ω_Λ = (1-e^{-1})(11\ln 2 - 3\ln 3)/4 = 0.6841
    
    Compares against Planck 2018: Ω_Λ = 0.6847 ± 0.0073
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize cosmological constant pipeline.
        
        Parameters:
            output_dir (str): Output directory for results
        """
        super().__init__("cosmo_const", output_dir)
        
        # Initialize validation components
        self.error_prop = ErrorPropagation()
        self.mc_validator = CosmoConstMonteCarloValidator(n_samples=100000)
        self.sensitivity = SensitivityAnalysis()
        
        # Planck 2018 observational values
        self.planck_omega_lambda = 0.6847
        self.planck_sigma = 0.0073
        
        self.update_metadata('description', 'Cosmological constant from causal diamond triality')
        self.update_metadata('parameter_free', True)
        self.update_metadata('prediction', 'Omega_Lambda = 0.6841')
    
    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute cosmological constant analysis.
        
        Parameters:
            context (dict, optional): Analysis parameters
        
        Returns:
            dict: Complete analysis results
        """
        self.log_progress("Starting cosmological constant calculation...")
        
        # 1. Calculate parameter-free prediction
        self.log_progress("Calculating geometric entropy and irreversibility fraction...")
        physics_results = self._calculate_physics()
        
        # 2. Compare with Planck 2018 observation
        self.log_progress("Comparing with Planck 2018 observation...")
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
        
        self.log_progress("✓ Cosmological constant analysis complete")
        
        # Save results
        self.save_results(results)
        
        return results
    
    def _calculate_physics(self) -> Dict[str, Any]:
        """
        Calculate physics predictions.
        
        Returns:
            dict: Physics calculation results
        """
        # Geometric entropy
        geom_result = calculate_geometric_entropy()
        
        # Irreversibility fraction
        irrev_result = calculate_irreversibility_fraction()
        
        # Dark energy fraction
        omega_result = calculate_omega_lambda()
        
        # Cosmological constant
        lambda_result = calculate_lambda()
        
        return {
            'geometric_entropy': geom_result,
            'irreversibility_fraction': irrev_result,
            'omega_lambda': omega_result,
            'lambda': lambda_result,
            'prediction_summary': {
                'S_geom': geom_result['S_geom'],
                'f_irrev': irrev_result['f_irrev'],
                'omega_lambda': omega_result['omega_lambda'],
                'lambda': lambda_result['lambda']
            }
        }
    
    def _compare_with_observations(self, physics_results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare prediction with Planck 2018 observation.
        
        Parameters:
            physics_results (dict): Physics calculation results
        
        Returns:
            dict: Comparison analysis
        """
        predicted = physics_results['omega_lambda']['omega_lambda']
        
        # Error propagation comparison
        error_comparison = self.error_prop.compare_with_observation(
            self.planck_omega_lambda,
            self.planck_sigma
        )
        
        # Calculate chi-squared
        chi2_result = self.calculate_chi_squared(
            observed=np.array([self.planck_omega_lambda]),
            expected=np.array([predicted]),
            uncertainties=np.array([self.planck_sigma]),
            degrees_of_freedom=0  # Parameter-free prediction
        )
        
        return {
            'predicted': predicted,
            'observed': self.planck_omega_lambda,
            'observed_sigma': self.planck_sigma,
            'error_propagation': error_comparison,
            'chi_squared': chi2_result,
            'deviation_sigma': error_comparison['deviation_sigma'],
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
        
        # Monte Carlo validation
        self.log_progress("  Running Monte Carlo uncertainty quantification...")
        validations['monte_carlo'] = self.mc_validator.run_mc_validation(
            self.planck_omega_lambda,
            self.planck_sigma
        )
        
        # Model comparison (best fit approach, not BIC/AIC due to single data point)
        self.log_progress("  Running model comparison...")
        validations['model_comparison'] = self._model_comparison()
        
        # Sensitivity analysis
        self.log_progress("  Running sensitivity analysis...")
        validations['sensitivity'] = self.sensitivity.full_sensitivity_analysis()
        
        # Compute overall status based on Monte Carlo consistency
        mc_consistency = validations['monte_carlo'].get('consistency_fractions', {})
        p_consistent_2sigma = mc_consistency.get('p_consistent_2sigma', 0)
        validations['overall_status'] = 'PASSED' if p_consistent_2sigma > 0.95 else 'FAILED'
        
        return validations
    
    def _model_comparison(self) -> Dict[str, Any]:
        """
        Compare H-ΛCDM parameter-free prediction against alternative models.
        
        Uses best fit model comparison (not BIC/AIC, which require multiple data points).
        
        Models compared:
        1. H-ΛCDM: 0 free parameters (prediction = 0.6841)
        2. ΛCDM: 1 free parameter (best fit = observed value)
        3. Quintessence: 2 free parameters (w, Ω_Λ)
        
        With a single data point, we use agreement quality combined with parsimony:
        - Excellent agreement (<0.1σ): H-ΛCDM preferred (parameter-free prediction matches observation)
        - Good agreement (<1σ): H-ΛCDM preferred (parsimony favors parameter-free model)
        - Poor agreement (≥1σ): ΛCDM preferred (better fit despite extra parameter)
        
        Returns:
            dict: Model comparison results
        """
        # Model 1: H-ΛCDM (0 parameters)
        prediction = 0.6841
        deviation_sigma = (self.planck_omega_lambda - prediction) / self.planck_sigma
        
        # Model 2: ΛCDM (1 parameter - best fit at observed value)
        # Model 3: Quintessence (2 parameters - best fit at observed value)
        
        # Determine preferred model based on agreement quality and parsimony
        if abs(deviation_sigma) < 0.1:
            preferred_model = 'hlcdm'
            preference_reason = 'excellent_agreement'
            evidence_strength = 'strong'
        elif abs(deviation_sigma) < 1.0:
            preferred_model = 'hlcdm'
            preference_reason = 'good_agreement_parsimony'
            evidence_strength = 'moderate'
        else:
            preferred_model = 'lambdacdm'
            preference_reason = 'better_fit'
            evidence_strength = 'weak'
        
        return {
            'models': {
                'hlcdm': {
                    'n_parameters': 0,
                    'prediction': prediction,
                    'deviation_sigma': float(deviation_sigma)
                },
                'lambdacdm': {
                    'n_parameters': 1,
                    'best_fit': self.planck_omega_lambda
                },
                'quintessence': {
                    'n_parameters': 2,
                    'best_fit': self.planck_omega_lambda
                }
            },
            'preferred_model': preferred_model,
            'preference_reason': preference_reason,
            'evidence_strength': evidence_strength,
            'deviation_sigma': float(deviation_sigma),
            'interpretation': self._interpret_model_comparison(preferred_model, abs(deviation_sigma))
        }
    
    def _interpret_model_comparison(self, preferred_model: str, deviation_sigma: float) -> str:
        """
        Interpret overall model comparison results.
        
        Parameters:
            preferred_model (str): Preferred model ('hlcdm' or 'lambdacdm')
            deviation_sigma (float): Deviation in sigma units
        
        Returns:
            str: Interpretation string
        """
        if preferred_model == 'hlcdm':
            if abs(deviation_sigma) < 0.1:
                return "H-ΛCDM preferred: excellent agreement (within 0.1σ) with parameter-free prediction"
            elif abs(deviation_sigma) < 1.0:
                return "H-ΛCDM preferred: good agreement (within 1σ) with parameter-free prediction favors parsimony"
            else:
                return "H-ΛCDM preferred: parameter-free prediction within acceptable range"
        else:
            return f"ΛCDM preferred: better fit to observation (deviation {abs(deviation_sigma):.2f}σ)"
    
    def _create_systematic_budget(self) -> AnalysisPipeline.SystematicBudget:
        """
        Create systematic error budget.
        
        Returns:
            SystematicBudget: Configured systematic error budget
        """
        budget = self.SystematicBudget()
        
        # Theoretical systematics
        budget.add_component('dimension_counting', 0.0)  # Exact: 3+3+2=8
        budget.add_component('entropy_formula', 0.0)  # Shannon entropy exact
        budget.add_component('decoherence_model', 0.001)  # Poisson assumption ~0.1%
        
        # Physical assumptions
        budget.add_component('flat_universe', 0.002)  # Curvature Ω_k < 0.002
        budget.add_component('hubble_time_scale', 0.001)  # t_H definition
        
        # Comparison systematics (from Planck)
        budget.add_component('planck_calibration', 0.002)  # Instrumental
        budget.add_component('cmb_foregrounds', 0.001)  # Foreground model
        
        return budget
    
    def validate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform basic statistical validation.
        
        Parameters:
            context (dict, optional): Validation parameters
        
        Returns:
            dict: Validation results
        """
        self.log_progress("Performing basic cosmological constant validation...")
        
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
        omega = physics.get('omega_lambda', {}).get('omega_lambda', None)
        
        if omega is None:
            return {'passed': False, 'error': 'No physics results available'}
        
        # Check against expected value
        expected = 0.6841
        tolerance = 1e-4
        
        passed = np.abs(omega - expected) < tolerance
        
        return {
            'passed': passed,
            'test': 'physics_accuracy',
            'predicted': omega,
            'expected': expected,
            'deviation': omega - expected,
            'tolerance': tolerance
        }
    
    def _validate_comparison_consistency(self) -> Dict[str, Any]:
        """Validate comparison consistency."""
        comparison = self.results.get('comparison', {})
        deviation_sigma = comparison.get('deviation_sigma', None)
        
        if deviation_sigma is None:
            return {'passed': False, 'error': 'No comparison results available'}
        
        # Should be within 2σ
        passed = np.abs(deviation_sigma) < 2.0
        
        return {
            'passed': passed,
            'test': 'comparison_consistency',
            'deviation_sigma': deviation_sigma,
            'threshold': 2.0
        }
    
    def _validate_error_propagation(self) -> Dict[str, Any]:
        """Validate error propagation calculations."""
        validation = self.results.get('validation', {})
        error_prop = validation.get('error_propagation', {})
        
        if not error_prop:
            return {'passed': False, 'error': 'No error propagation results'}
        
        # Check that theoretical uncertainty is zero
        omega_uncertainty = error_prop.get('omega_lambda_uncertainty', {})
        delta_theory = omega_uncertainty.get('delta_omega_lambda_theory', 1.0)
        
        passed = np.abs(delta_theory) < 1e-10
        
        return {
            'passed': passed,
            'test': 'error_propagation',
            'theoretical_uncertainty': delta_theory,
            'expected': 0.0
        }
    
    def _validate_numerical_stability(self) -> Dict[str, Any]:
        """Validate numerical stability."""
        physics = self.results.get('physics', {})
        omega = physics.get('omega_lambda', {}).get('omega_lambda', None)
        
        if omega is None:
            return {'passed': False, 'error': 'No physics results available'}
        
        # Check for finite, reasonable values
        passed = np.isfinite(omega) and 0 < omega < 1
        
        return {
            'passed': passed,
            'test': 'numerical_stability',
            'omega_lambda': omega,
            'is_finite': np.isfinite(omega),
            'in_range': 0 < omega < 1
        }
    
    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform extended validation with Monte Carlo and sensitivity analysis.
        
        Parameters:
            context (dict, optional): Extended validation parameters
        
        Returns:
            dict: Extended validation results
        """
        self.log_progress("Performing extended cosmological constant validation...")
        
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
        mc_consistent = validation.get('monte_carlo', {}).get('consistency_fractions', {}).get('p_consistent_2sigma', 0) > 0.95
        extended_results['overall_status'] = 'PASSED' if mc_consistent else 'FAILED'
        
        self.log_progress(f"✓ Extended validation complete: {extended_results['overall_status']}")
        
        return extended_results
