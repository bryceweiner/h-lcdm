"""
Error Budget Comparison for LCDM vs HLCDM Void Analysis
========================================================

Compares error budgets between LCDM and HLCDM pipelines to determine if
differences in C_obs are due to systematics or the holographic component (Λ(z)).

Key insight: The difference ΔC_obs = C_obs(HLCDM) - C_obs(LCDM) can be decomposed into:
1. Common systematics (affect both equally)
2. Model-specific systematics (different void-finding algorithms)
3. Holographic component signal (Λ(z) effects on void structure)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import stats


class ErrorBudgetComparison:
    """
    Compare error budgets between LCDM and HLCDM void analyses.
    """
    
    def __init__(self):
        """Initialize error budget comparison."""
        pass
    
    def calculate_comprehensive_error_budget(
        self,
        c_obs: float,
        statistical_std: float,
        systematic_components: Dict[str, float],
        model_type: str = 'lcdm'
    ) -> Dict[str, Any]:
        """
        Calculate comprehensive error budget including statistical and systematic uncertainties.
        
        Parameters:
            c_obs: Observed clustering coefficient
            statistical_std: Statistical uncertainty (from bootstrap/jackknife)
            systematic_components: Dictionary of systematic error components
            model_type: 'lcdm' or 'hlcdm'
            
        Returns:
            dict: Comprehensive error budget
        """
        # Statistical uncertainty
        stat_uncertainty = statistical_std
        
        # Systematic uncertainty (quadrature sum)
        sys_components = systematic_components.copy()
        
        # Add model-specific systematic components
        if model_type == 'hlcdm':
            # H-ZOBOV specific systematics
            # Λ(z) calculation uncertainty propagates to void identification
            sys_components['lambda_z_uncertainty'] = 0.015  # 1.5% from Λ(z) variations
            # Redshift binning effects (H-ZOBOV processes in bins)
            sys_components['redshift_binning'] = 0.010  # 1.0% from binning
            # Zone merging threshold variations (Λ(z)-dependent)
            sys_components['threshold_variation'] = 0.012  # 1.2% from threshold scaling
        
        elif model_type == 'lcdm':
            # Traditional void-finding specific systematics
            # Algorithm differences (VAST vs ZOBOV vs DESIVAST)
            sys_components['algorithm_differences'] = 0.020  # 2.0% from different algorithms
            # Fixed threshold assumption
            sys_components['fixed_threshold'] = 0.008  # 0.8% from not accounting for Λ(z)
        
        # Calculate total systematic uncertainty
        total_systematic = np.sqrt(sum(unc**2 for unc in sys_components.values()))
        
        # Combined uncertainty (statistical + systematic in quadrature)
        total_uncertainty = np.sqrt(stat_uncertainty**2 + total_systematic**2)
        
        return {
            'c_obs': c_obs,
            'statistical_uncertainty': stat_uncertainty,
            'systematic_components': sys_components,
            'total_systematic': total_systematic,
            'total_uncertainty': total_uncertainty,
            'model_type': model_type
        }
    
    def compare_error_budgets(
        self,
        lcdm_budget: Dict[str, Any],
        hlcdm_budget: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compare error budgets between LCDM and HLCDM analyses.
        
        Parameters:
            lcdm_budget: LCDM error budget
            hlcdm_budget: HLCDM error budget
            
        Returns:
            dict: Comparison results
        """
        # Extract values
        c_obs_lcdm = lcdm_budget['c_obs']
        c_obs_hlcdm = hlcdm_budget['c_obs']
        delta_c_obs = c_obs_hlcdm - c_obs_lcdm
        
        # Extract uncertainties
        stat_lcdm = lcdm_budget['statistical_uncertainty']
        stat_hlcdm = hlcdm_budget['statistical_uncertainty']
        sys_lcdm = lcdm_budget['total_systematic']
        sys_hlcdm = hlcdm_budget['total_systematic']
        total_lcdm = lcdm_budget['total_uncertainty']
        total_hlcdm = hlcdm_budget['total_uncertainty']
        
        # Calculate uncertainty in the difference
        # Assuming independent measurements: σ_diff = sqrt(σ_lcdm² + σ_hlcdm²)
        stat_diff = np.sqrt(stat_lcdm**2 + stat_hlcdm**2)
        sys_diff = np.sqrt(sys_lcdm**2 + sys_hlcdm**2)
        total_diff = np.sqrt(total_lcdm**2 + total_hlcdm**2)
        
        # Statistical significance of the difference
        z_score = abs(delta_c_obs) / total_diff if total_diff > 0 else 0.0
        
        # Calculate p-value (two-tailed test)
        p_value = 2 * (1 - stats.norm.cdf(z_score)) if total_diff > 0 else 1.0
        
        # Decompose systematic components
        # Common systematics (affect both equally)
        common_systematics = {
            'void_finding_bias': min(
                lcdm_budget['systematic_components'].get('void_finding_bias', 0),
                hlcdm_budget['systematic_components'].get('void_finding_bias', 0)
            ),
            'tracer_density': min(
                lcdm_budget['systematic_components'].get('tracer_density', 0),
                hlcdm_budget['systematic_components'].get('tracer_density', 0)
            ),
            'survey_geometry': min(
                lcdm_budget['systematic_components'].get('survey_geometry', 0),
                hlcdm_budget['systematic_components'].get('survey_geometry', 0)
            ),
            'selection_effects': min(
                lcdm_budget['systematic_components'].get('selection_effects', 0),
                hlcdm_budget['systematic_components'].get('selection_effects', 0)
            ),
            'redshift_precision': min(
                lcdm_budget['systematic_components'].get('redshift_precision', 0),
                hlcdm_budget['systematic_components'].get('redshift_precision', 0)
            )
        }
        
        # Model-specific systematics
        lcdm_specific = {
            k: v for k, v in lcdm_budget['systematic_components'].items()
            if k not in common_systematics
        }
        hlcdm_specific = {
            k: v for k, v in hlcdm_budget['systematic_components'].items()
            if k not in common_systematics
        }
        
        # Estimate holographic component contribution
        # This is the difference that cannot be explained by systematics
        # If |ΔC_obs| > total_diff, there's a signal beyond systematics
        holographic_signal = delta_c_obs if abs(delta_c_obs) > total_diff else 0.0
        holographic_uncertainty = max(0, abs(delta_c_obs) - total_diff)
        
        return {
            'delta_c_obs': delta_c_obs,
            'statistical_uncertainty_diff': stat_diff,
            'systematic_uncertainty_diff': sys_diff,
            'total_uncertainty_diff': total_diff,
            'z_score': z_score,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'common_systematics': common_systematics,
            'lcdm_specific_systematics': lcdm_specific,
            'hlcdm_specific_systematics': hlcdm_specific,
            'holographic_signal_estimate': holographic_signal,
            'holographic_uncertainty': holographic_uncertainty,
            'interpretation': self._interpret_comparison(
                delta_c_obs, total_diff, z_score, p_value
            )
        }
    
    def _interpret_comparison(
        self,
        delta_c_obs: float,
        total_uncertainty: float,
        z_score: float,
        p_value: float
    ) -> str:
        """
        Interpret the comparison results.
        
        Parameters:
            delta_c_obs: Difference in C_obs
            total_uncertainty: Combined uncertainty in the difference
            z_score: Statistical significance (z-score)
            p_value: p-value
            
        Returns:
            str: Interpretation
        """
        if abs(delta_c_obs) < total_uncertainty:
            return (
                f"The difference ΔC_obs = {delta_c_obs:.4f} is smaller than the measurement "
                f"uncertainty (σ_total = {total_uncertainty:.4f}). LCDM and HLCDM differences "
                f"are too subtle for current measurements to distinguish. The observed difference "
                f"is consistent with systematic uncertainties and statistical noise."
            )
        elif p_value < 0.05:
            return (
                f"The difference ΔC_obs = {delta_c_obs:.4f} is statistically significant "
                f"(z = {z_score:.2f}, p = {p_value:.4f}). This suggests a real difference "
                f"beyond systematic uncertainties, potentially due to the holographic component (Λ(z))."
            )
        else:
            return (
                f"The difference ΔC_obs = {delta_c_obs:.4f} is marginally significant "
                f"(z = {z_score:.2f}, p = {p_value:.4f}). Further analysis needed to distinguish "
                f"between systematic effects and holographic component."
            )
    
    def generate_comparison_report(
        self,
        lcdm_budget: Dict[str, Any],
        hlcdm_budget: Dict[str, Any],
        comparison: Dict[str, Any]
    ) -> str:
        """
        Generate a detailed comparison report.
        
        Parameters:
            lcdm_budget: LCDM error budget
            hlcdm_budget: HLCDM error budget
            comparison: Comparison results
            
        Returns:
            str: Formatted report
        """
        report = []
        report.append("="*80)
        report.append("ERROR BUDGET COMPARISON: LCDM vs HLCDM")
        report.append("="*80)
        report.append("")
        
        # Observed values
        report.append("OBSERVED VALUES:")
        report.append(f"  LCDM:  C_obs = {lcdm_budget['c_obs']:.4f} ± {lcdm_budget['total_uncertainty']:.4f}")
        report.append(f"  HLCDM: C_obs = {hlcdm_budget['c_obs']:.4f} ± {hlcdm_budget['total_uncertainty']:.4f}")
        report.append(f"  Difference: ΔC_obs = {comparison['delta_c_obs']:.4f}")
        report.append("")
        
        # Uncertainty breakdown
        report.append("UNCERTAINTY BREAKDOWN:")
        report.append(f"  Statistical (LCDM):  {lcdm_budget['statistical_uncertainty']:.4f}")
        report.append(f"  Statistical (HLCDM): {hlcdm_budget['statistical_uncertainty']:.4f}")
        report.append(f"  Systematic (LCDM):   {lcdm_budget['total_systematic']:.4f}")
        report.append(f"  Systematic (HLCDM):  {hlcdm_budget['total_systematic']:.4f}")
        report.append(f"  Total (LCDM):        {lcdm_budget['total_uncertainty']:.4f}")
        report.append(f"  Total (HLCDM):       {hlcdm_budget['total_uncertainty']:.4f}")
        report.append("")
        
        # Difference uncertainty
        report.append("DIFFERENCE UNCERTAINTY:")
        report.append(f"  Statistical: {comparison['statistical_uncertainty_diff']:.4f}")
        report.append(f"  Systematic:  {comparison['systematic_uncertainty_diff']:.4f}")
        report.append(f"  Total:       {comparison['total_uncertainty_diff']:.4f}")
        report.append("")
        
        # Statistical significance
        report.append("STATISTICAL SIGNIFICANCE:")
        report.append(f"  Z-score: {comparison['z_score']:.2f}")
        report.append(f"  p-value: {comparison['p_value']:.4f}")
        report.append(f"  Significant: {'Yes' if comparison['statistically_significant'] else 'No'}")
        report.append("")
        
        # Systematic components
        report.append("COMMON SYSTEMATIC COMPONENTS:")
        for name, value in comparison['common_systematics'].items():
            report.append(f"  {name}: {value:.4f}")
        report.append("")
        
        report.append("LCDM-SPECIFIC SYSTEMATIC COMPONENTS:")
        for name, value in comparison['lcdm_specific_systematics'].items():
            report.append(f"  {name}: {value:.4f}")
        report.append("")
        
        report.append("HLCDM-SPECIFIC SYSTEMATIC COMPONENTS:")
        for name, value in comparison['hlcdm_specific_systematics'].items():
            report.append(f"  {name}: {value:.4f}")
        report.append("")
        
        # Interpretation
        report.append("INTERPRETATION:")
        report.append(f"  {comparison['interpretation']}")
        report.append("")
        
        if comparison['holographic_signal_estimate'] != 0:
            report.append("HOLOGRAPHIC COMPONENT ESTIMATE:")
            report.append(f"  Signal: {comparison['holographic_signal_estimate']:.4f}")
            report.append(f"  Uncertainty: {comparison['holographic_uncertainty']:.4f}")
            report.append("")
        
        report.append("="*80)
        
        return "\n".join(report)

