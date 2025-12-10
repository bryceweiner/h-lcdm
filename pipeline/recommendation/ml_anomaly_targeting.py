"""
ML Anomaly Targeting - Phase 6
===============================

Connects CMB analysis to ML-flagged anomalies.

The ML pipeline flagged specific samples (4, 6, 9, 25, 29, 33, 46, 48, 54, 80)
as H-ΛCDM candidates. This module verifies H-ΛCDM signatures at those ℓ-bins.
"""

import json
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


class MLAnomalyTargeting:
    """
    Target analysis at ML-flagged anomaly locations.
    
    For flagged ℓ-bins, verifies:
    1. Residuals exceed 2σ from ΛCDM expectation
    2. Cross-modal correlation ρ is elevated
    3. Pattern consistent with α ≈ -5.7
    """
    
    # ML-flagged samples from pipeline
    FLAGGED_SAMPLES = [4, 6, 9, 25, 29, 33, 46, 48, 54, 80]
    
    def __init__(self):
        """Initialize ML anomaly targeting."""
        pass
    
    def load_ml_anomaly_context(self) -> Dict[str, Any]:
        """
        Load anomaly scores, ℓ-bins, z-values from ML results.
        
        Returns:
            dict with:
                - anomaly_indices: List of flagged sample indices
                - ell_bins: Corresponding multipole bins
                - redshifts: Effective redshifts
                - anomaly_scores: ML anomaly scores
        """
        # Try to load from ML pipeline results
        stage3_path = Path("results") / "json" / "ml_pipeline" / "checkpoints" / "stage3_pattern_detection_results.json"
        fallback_path = Path("results") / "json" / "ml_results.json"
        path = stage3_path if stage3_path.exists() else fallback_path
        
        anomaly_data = {
            'anomaly_indices': self.FLAGGED_SAMPLES,
            'ell_bins': [],
            'redshifts': [],
            'anomaly_scores': [],
        }
        
        if path.exists():
            try:
                with open(path, "r") as f:
                    data = json.load(f)
                
                anomalies = data.get("top_anomalies") or data.get("results", {}).get("pattern_detection", {}).get("top_anomalies", [])
                
                for entry in anomalies:
                    sample_idx = entry.get("sample_index")
                    if sample_idx in self.FLAGGED_SAMPLES:
                        ctx = entry.get("context", {})
                        
                        # Extract ℓ-bin information
                        ell_bin = ctx.get("ell_bin") or ctx.get("multipole_range")
                        if ell_bin:
                            anomaly_data['ell_bins'].append(ell_bin)
                        
                        # Extract redshift
                        z_val = ctx.get("redshift") or ctx.get("redshift_regime")
                        if isinstance(z_val, (int, float)):
                            anomaly_data['redshifts'].append(float(z_val))
                        
                        # Extract anomaly score
                        score = entry.get("anomaly_score") or entry.get("score")
                        if score is not None:
                            anomaly_data['anomaly_scores'].append(float(score))
            except Exception as e:
                logger.warning(f"Could not load ML anomaly context: {e}")
        
        # Default values if not found
        if not anomaly_data['ell_bins']:
            # ACT DR6 covers ℓ ≈ 500-2500, so estimate bins
            # Assuming ~20 multipoles per sample
            anomaly_data['ell_bins'] = [
                (500 + i * 20, 520 + i * 20) for i in self.FLAGGED_SAMPLES
            ]
        
        if not anomaly_data['redshifts']:
            # Default to z ≈ 2.2 (from previous analysis)
            anomaly_data['redshifts'] = [2.2] * len(self.FLAGGED_SAMPLES)
        
        return anomaly_data
    
    def targeted_test(
        self,
        residuals: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        coherence_results: Dict[str, Any],
        alpha_results: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Verify H-ΛCDM signatures at ML-flagged ℓ-bins.
        
        For each flagged bin, checks:
        1. Residuals exceed 2σ from ΛCDM expectation
        2. Cross-modal correlation ρ is elevated
        3. Pattern consistent with α ≈ -5.7
        
        Parameters:
            residuals: Residuals from ResidualAnalyzer
            coherence_results: Cross-modal coherence results
            alpha_results: Optional amplitude consistency results
            
        Returns:
            dict with verification results for each flagged bin
        """
        ml_context = self.load_ml_anomaly_context()
        
        results = {}
        
        # Focus on ACT DR6 (primary survey for ML analysis)
        survey_name = 'act_dr6'
        if survey_name not in residuals:
            logger.warning(f"Survey {survey_name} not found in residuals")
            return {'error': f'Survey {survey_name} not available'}
        
        survey_residuals = residuals[survey_name]
        
        # Get coherence results for this survey
        survey_coherence = coherence_results.get('by_survey', {}).get(survey_name, {})
        
        for i, sample_idx in enumerate(self.FLAGGED_SAMPLES):
            # Get ℓ-bin for this sample
            if i < len(ml_context['ell_bins']):
                ell_bin = ml_context['ell_bins'][i]
                if isinstance(ell_bin, (list, tuple)) and len(ell_bin) == 2:
                    ell_min, ell_max = ell_bin
                else:
                    # Estimate from sample index
                    ell_min = 500 + sample_idx * 20
                    ell_max = ell_min + 20
            else:
                ell_min = 500 + sample_idx * 20
                ell_max = ell_min + 20
            
            sample_results = {
                'sample_index': sample_idx,
                'ell_bin': (ell_min, ell_max),
                'redshift': ml_context['redshifts'][i] if i < len(ml_context['redshifts']) else 2.2,
            }
            
            # Check residuals for each spectrum
            for spectrum in ['TT', 'TE', 'EE']:
                if spectrum not in survey_residuals or survey_residuals[spectrum] is None:
                    continue
                
                res_data = survey_residuals[spectrum]
                ell = res_data['ell']
                residual = res_data['residual']
                cl_err = res_data['cl_err']
                
                # Find data points in this ℓ-bin
                bin_mask = (ell >= ell_min) & (ell <= ell_max)
                if not bin_mask.any():
                    continue
                
                residual_bin = residual[bin_mask]
                cl_err_bin = cl_err[bin_mask]
                
                # Check if residuals exceed 2σ
                sigma_excess = np.abs(residual_bin) / cl_err_bin
                exceeds_2sigma = np.any(sigma_excess > 2.0)
                max_sigma = np.max(sigma_excess) if len(sigma_excess) > 0 else 0.0
                
                sample_results[f'{spectrum}_exceeds_2sigma'] = exceeds_2sigma
                sample_results[f'{spectrum}_max_sigma'] = max_sigma
                sample_results[f'{spectrum}_mean_residual'] = np.mean(residual_bin) if len(residual_bin) > 0 else np.nan
            
            # Check cross-modal coherence in this bin
            # Get correlation values near this ℓ-bin
            for pair_name in ['tt_te', 'tt_ee', 'te_ee']:
                if pair_name not in survey_coherence:
                    continue
                
                pair_data = survey_coherence[pair_name]
                if 'correlation' not in pair_data:
                    continue
                
                corr_data = pair_data['correlation']
                ell_corr = corr_data.get('ell', np.array([]))
                correlation = corr_data.get('correlation', np.array([]))
                
                if len(ell_corr) == 0 or len(correlation) == 0:
                    continue
                
                # Find correlation values in this bin
                corr_bin_mask = (ell_corr >= ell_min) & (ell_corr <= ell_max)
                if corr_bin_mask.any():
                    corr_bin = correlation[corr_bin_mask]
                    mean_corr = np.nanmean(corr_bin)
                    sample_results[f'{pair_name}_mean_correlation'] = mean_corr
                    sample_results[f'{pair_name}_elevated'] = mean_corr > 0.3  # Threshold for "elevated"
            
            # Check consistency with α ≈ -5.7
            # This would require more detailed analysis, but we can check if
            # the residual pattern matches expected H-ΛCDM scaling
            if alpha_results:
                consistency = alpha_results.get('consistency', {})
                alpha_mean = consistency.get('mean_alpha', np.nan)
                if not np.isnan(alpha_mean):
                    alpha_diff = abs(alpha_mean - (-5.7))
                    sample_results['alpha_consistent'] = alpha_diff < 2.0
                    sample_results['alpha_value'] = alpha_mean
            
            results[f'sample_{sample_idx}'] = sample_results
        
        # Summary statistics
        summary = self._compute_summary(results)
        
        return {
            'by_sample': results,
            'summary': summary,
        }
    
    def _compute_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics across all flagged samples."""
        n_samples = len(results)
        n_exceed_2sigma = 0
        n_elevated_correlation = 0
        all_max_sigmas = []
        
        for sample_key, sample_data in results.items():
            # Count samples with 2σ excess
            for spectrum in ['TT', 'TE', 'EE']:
                key = f'{spectrum}_exceeds_2sigma'
                if key in sample_data and sample_data[key]:
                    n_exceed_2sigma += 1
                    break  # Count each sample only once
            
            # Count samples with elevated correlation
            for pair in ['tt_te', 'tt_ee', 'te_ee']:
                key = f'{pair}_elevated'
                if key in sample_data and sample_data.get(key, False):
                    n_elevated_correlation += 1
                    break
            
            # Collect max sigma values
            for spectrum in ['TT', 'TE', 'EE']:
                key = f'{spectrum}_max_sigma'
                if key in sample_data:
                    val = sample_data[key]
                    if not np.isnan(val):
                        all_max_sigmas.append(val)
        
        return {
            'n_samples_tested': n_samples,
            'n_exceed_2sigma': n_exceed_2sigma,
            'n_elevated_correlation': n_elevated_correlation,
            'fraction_exceed_2sigma': n_exceed_2sigma / n_samples if n_samples > 0 else 0.0,
            'fraction_elevated_correlation': n_elevated_correlation / n_samples if n_samples > 0 else 0.0,
            'mean_max_sigma': np.mean(all_max_sigmas) if all_max_sigmas else np.nan,
        }

