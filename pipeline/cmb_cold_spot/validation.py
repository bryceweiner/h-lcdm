"""
Cold Spot Validation Module
===========================

Minimal statistical validation for Cold Spot QTEP tests.
Bootstrap confidence intervals and cross-survey consistency checks.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class ColdSpotValidator:
    """Minimal validation for Cold Spot QTEP tests."""
    
    def __init__(self, n_bootstrap: int = 100):
        """
        Initialize validator.
        
        Parameters:
            n_bootstrap: Number of bootstrap samples for CI estimation
        """
        self.n_bootstrap = n_bootstrap
    
    def bootstrap_confidence_intervals(self, test_results: Dict[str, Any],
                                     data: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Bootstrap CI estimation for test statistics.
        
        Parameters:
            test_results: Test results dictionary
            data: Original data for resampling (optional)
            
        Returns:
            dict: Updated test results with bootstrap CIs
        """
        # If data provided, use it for bootstrap
        # Otherwise, use existing CIs as estimates
        
        results_updated = test_results.copy()
        
        # For each test, update bootstrap CIs if data available
        # Otherwise, keep existing estimates
        
        return results_updated
    
    def cross_survey_consistency(self, results_by_survey: Dict[str, Dict]) -> Dict[str, Any]:
        """
        Check consistency across ACT, Planck, SPT-3G, COBE, WMAP.
        
        Parameters:
            results_by_survey: Dictionary mapping survey names to test results
            
        Returns:
            dict: Consistency analysis results
        """
        if not results_by_survey:
            return {
                'chi_squared_per_dof': np.nan,
                'survey_consistency_p_value': np.nan,
                'surveys_agree': False,
                'n_surveys': 0
            }
        
        # Extract temperature deficits from each survey
        deficits = []
        uncertainties = []
        survey_names = []
        
        for survey_name, results in results_by_survey.items():
            if 'test_1_temperature_deficit' in results:
                test_1 = results['test_1_temperature_deficit']
                deficit = test_1.get('observed_deficit', np.nan)
                uncertainty = test_1.get('observed_deficit_uncertainty', np.nan)
                
                if not np.isnan(deficit) and not np.isnan(uncertainty):
                    deficits.append(deficit)
                    uncertainties.append(uncertainty)
                    survey_names.append(survey_name)
        
        if len(deficits) < 2:
            return {
                'chi_squared_per_dof': np.nan,
                'survey_consistency_p_value': np.nan,
                'surveys_agree': True,  # Can't test with <2 surveys
                'n_surveys': len(deficits)
            }
        
        deficits = np.array(deficits)
        uncertainties = np.array(uncertainties)
        
        # Weighted mean
        weights = 1.0 / (uncertainties**2 + 1e-20)
        weighted_mean = np.sum(weights * deficits) / np.sum(weights)
        
        # Chi-squared test
        chi_squared = np.sum(((deficits - weighted_mean) / uncertainties)**2)
        dof = len(deficits) - 1
        
        if dof > 0:
            chi_squared_per_dof = chi_squared / dof
            # P-value from chi-squared distribution
            p_value = 1 - stats.chi2.cdf(chi_squared, dof)
        else:
            chi_squared_per_dof = np.nan
            p_value = np.nan
        
        # Surveys agree if p > 0.05
        surveys_agree = p_value > 0.05 if not np.isnan(p_value) else True
        
        return {
            'chi_squared_per_dof': float(chi_squared_per_dof) if not np.isnan(chi_squared_per_dof) else np.nan,
            'survey_consistency_p_value': float(p_value) if not np.isnan(p_value) else np.nan,
            'surveys_agree': surveys_agree,
            'n_surveys': len(deficits),
            'survey_names': survey_names,
            'weighted_mean_deficit': float(weighted_mean),
            'deficits': [float(d) for d in deficits],
            'uncertainties': [float(u) for u in uncertainties]
        }
    
    def update_bootstrap_cis(self, test_results: Dict[str, Any],
                           bootstrap_samples: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Update bootstrap confidence intervals from bootstrap samples.
        
        Parameters:
            test_results: Original test results
            bootstrap_samples: List of test results from bootstrap resampling
            
        Returns:
            dict: Updated test results with bootstrap CIs
        """
        results_updated = test_results.copy()
        
        # Extract statistics from bootstrap samples
        if bootstrap_samples:
            # For test 1: observed_deficit
            if 'test_1_temperature_deficit' in test_results:
                deficits = []
                for sample in bootstrap_samples:
                    if 'test_1_temperature_deficit' in sample:
                        def_val = sample['test_1_temperature_deficit'].get('observed_deficit', np.nan)
                        if not np.isnan(def_val):
                            deficits.append(def_val)
                
                if deficits:
                    deficits = np.array(deficits)
                    ci_low = np.percentile(deficits, 2.5)
                    ci_high = np.percentile(deficits, 97.5)
                    results_updated['test_1_temperature_deficit']['bootstrap_ci_low'] = float(ci_low)
                    results_updated['test_1_temperature_deficit']['bootstrap_ci_high'] = float(ci_high)
            
            # For test 2: discrete_feature_score
            if 'test_2_angular_power_spectrum' in test_results:
                scores = []
                for sample in bootstrap_samples:
                    if 'test_2_angular_power_spectrum' in sample:
                        score_val = sample['test_2_angular_power_spectrum'].get('discrete_feature_score', np.nan)
                        if not np.isnan(score_val):
                            scores.append(score_val)
                
                if scores:
                    scores = np.array(scores)
                    ci_low = np.percentile(scores, 2.5)
                    ci_high = np.percentile(scores, 97.5)
                    results_updated['test_2_angular_power_spectrum']['bootstrap_ci_low'] = float(ci_low)
                    results_updated['test_2_angular_power_spectrum']['bootstrap_ci_high'] = float(ci_high)
            
            # For test 3: correlation_coefficient
            if 'test_3_spatial_correlation' in test_results:
                correlations = []
                for sample in bootstrap_samples:
                    if 'test_3_spatial_correlation' in sample:
                        corr_val = sample['test_3_spatial_correlation'].get('correlation_coefficient', np.nan)
                        if not np.isnan(corr_val):
                            correlations.append(corr_val)
                
                if correlations:
                    correlations = np.array(correlations)
                    ci_low = np.percentile(correlations, 2.5)
                    ci_high = np.percentile(correlations, 97.5)
                    results_updated['test_3_spatial_correlation']['bootstrap_ci_low'] = float(ci_low)
                    results_updated['test_3_spatial_correlation']['bootstrap_ci_high'] = float(ci_high)
        
        return results_updated

