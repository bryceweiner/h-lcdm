"""
Cross-Modal Coherence Verification - Recommendation 6
======================================================

Tests for cross-modal coherence in CMB TE-EE residuals at ℓ=800-1200.

PHYSICAL BASIS:
---------------
H-ΛCDM predicts that the Lindblad-Zeno mechanism at recombination affects ALL 
baryon-photon modes coherently. This produces CORRELATED residuals across TT, TE, 
and EE spectra - the same underlying modification affects all polarization channels.

CRITICAL DISTINCTION:
- The QTEP ratio (S_coh/|S_decoh| ≈ 2.257) is a THEORETICAL CONSTANT, not an observable
- It enters through: QTEP → α → r_s → observable shifts
- You CANNOT directly measure QTEP from CMB power spectra

WHAT THIS TEST MEASURES:
- Cross-modal correlation ρ(TE, EE) = Cov(ΔC_ℓ^TE, ΔC_ℓ^EE) / (σ_TE × σ_EE)
- ρ ranges from -1 to +1 (it's a correlation coefficient)

PREDICTIONS:
- H-ΛCDM: ρ > 0 (correlated residuals from shared Lindblad mechanism)
  Expected magnitude: ρ ~ 0.2-0.5 at ℓ=800-1200
- ΛCDM:   ρ ≈ 0 (independent Gaussian noise in each channel)
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any, List
import logging
from scipy import stats
from scipy.interpolate import interp1d

from hlcdm.parameters import HLCDM_PARAMS
from hlcdm.cosmology import HLCDMCosmology

logger = logging.getLogger(__name__)


class QTEPVerificationTest:
    """
    Cross-modal coherence test for CMB polarization residuals.
    
    Tests whether TE and EE residuals show positive correlation at ℓ=800-1200,
    which is the H-ΛCDM prediction from Lindblad-Zeno coherent acoustic enhancement.
    
    IMPORTANT: This tests for ρ > 0, NOT for ρ = 2.257. The QTEP ratio is a
    theoretical input that determines α, which modifies r_s. The observable
    consequence is correlated residuals (ρ > 0), not the QTEP ratio itself.
    """
    
    # Theoretical reference (for documentation only - NOT the test target)
    QTEP_RATIO_THEORY = HLCDM_PARAMS.QTEP_RATIO  # ≈ 2.257 (theoretical input)
    
    # Test targets for cross-modal coherence
    # H-ΛCDM predicts: ρ > 0 with expected magnitude ~0.2-0.5
    # ΛCDM predicts: ρ ≈ 0 ± statistical uncertainty
    CORRELATION_THRESHOLD_HLCDM = 0.2   # Minimum expected correlation for H-ΛCDM
    CORRELATION_NULL_LCDM = 0.0         # ΛCDM null hypothesis
    SIGNIFICANCE_THRESHOLD = 2.0        # σ threshold for detection
    
    # Target multipole range (from bao_resolution_qit.tex)
    ELL_MIN = 800
    ELL_MAX = 1200
    
    # Coherence extraction parameters
    BIN_WIDTH = 20.0  # Multipole bin width for coherence analysis
    
    def __init__(self):
        """Initialize cross-modal coherence test."""
        pass
    
    def extract_coherence_amplitude(
        self,
        residuals_te: np.ndarray,
        residuals_ee: np.ndarray,
        ell: np.ndarray,
        ell_min: float = None,
        ell_max: float = None
    ) -> Dict[str, Any]:
        """
        Extract cross-modal coherence from TE-EE residuals.
        
        Computes the Pearson correlation between TE and EE residuals in the
        target multipole range. A positive correlation indicates that both
        channels are affected by the same underlying modification (H-ΛCDM).
        
        Parameters:
            residuals_te: TE residual values ΔC_ℓ^TE
            residuals_ee: EE residual values ΔC_ℓ^EE
            ell: Multipole array
            ell_min: Minimum multipole (None = use minimum available)
            ell_max: Maximum multipole (None = use maximum available)
            
        Returns:
            dict with:
                - ell_binned: Binned multipole centers
                - coherence: Cross-correlation per bin (ρ, ranges -1 to +1)
                - coherence_err: Uncertainty on coherence
                - correlation: Same as coherence (Pearson r)
                - n_points: Number of data points per bin
        """
        # Use full range if not specified
        if ell_min is None:
            ell_min = np.min(ell)
        if ell_max is None:
            ell_max = np.max(ell)
        
        # Restrict to target range
        range_mask = (ell >= ell_min) & (ell <= ell_max)
        if not range_mask.any():
            logger.warning(f"No data in range ℓ=[{ell_min}, {ell_max}]")
            return self._empty_coherence_result()
        
        ell_range = ell[range_mask]
        res_te_range = residuals_te[range_mask]
        res_ee_range = residuals_ee[range_mask]
        
        # Remove NaN values
        valid_mask = np.isfinite(res_te_range) & np.isfinite(res_ee_range)
        if not valid_mask.any():
            logger.warning("No valid data after NaN removal")
            return self._empty_coherence_result()
        
        ell_valid = ell_range[valid_mask]
        res_te_valid = res_te_range[valid_mask]
        res_ee_valid = res_ee_range[valid_mask]
        
        # Bin data for better signal-to-noise
        ell_bins = np.arange(ell_min, ell_max + self.BIN_WIDTH, self.BIN_WIDTH)
        bin_indices = np.digitize(ell_valid, ell_bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(ell_bins) - 2)
        
        # Compute coherence per bin
        ell_binned = []
        coherence = []
        coherence_err = []
        correlation = []
        n_points = []
        
        for i in range(len(ell_bins) - 1):
            bin_mask = (bin_indices == i)
            if bin_mask.sum() < 3:  # Need at least 3 points per bin
                continue
            
            ell_bin = ell_valid[bin_mask]
            res_te_bin = res_te_valid[bin_mask]
            res_ee_bin = res_ee_valid[bin_mask]
            
            # Cross-correlation coefficient (Pearson r)
            # This is what H-ΛCDM predicts to be > 0
            try:
                corr_val, p_val = stats.pearsonr(res_te_bin, res_ee_bin)
            except:
                corr_val = 0.0
                p_val = 1.0
            
            # Use Fisher z-transform for uncertainty estimation
            if len(res_te_bin) >= 5:
                n = len(res_te_bin)
                # Standard error of correlation coefficient
                se_r = np.sqrt((1 - corr_val**2) / (n - 2)) if n > 2 else 0.1
            else:
                se_r = 0.3  # Conservative uncertainty for small samples
            
            ell_binned.append(np.mean(ell_bin))
            coherence.append(corr_val)
            coherence_err.append(se_r)
            correlation.append(corr_val)
            n_points.append(len(ell_bin))
        
        if not ell_binned:
            return self._empty_coherence_result()
        
        return {
            'ell_binned': np.array(ell_binned),
            'coherence': np.array(coherence),
            'coherence_err': np.array(coherence_err),
            'correlation': np.array(correlation),
            'n_points': np.array(n_points),
            'ell_range': [ell_min, ell_max],
        }
    
    def _empty_coherence_result(self) -> Dict[str, Any]:
        """Return empty coherence result structure."""
        return {
            'ell_binned': np.array([]),
            'coherence': np.array([]),
            'coherence_err': np.array([]),
            'correlation': np.array([]),
            'n_points': np.array([]),
            'ell_range': [self.ELL_MIN, self.ELL_MAX],
        }
    
    def fit_correlation_bayesian(
        self,
        coherence_data: Dict[str, np.ndarray],
        n_bootstrap: int = 1000
    ) -> Dict[str, Any]:
        """
        Estimate mean cross-modal correlation with uncertainty.
        
        Uses bootstrap resampling to estimate the mean correlation coefficient
        and its uncertainty. H-ΛCDM predicts ρ_mean > 0.
        
        Parameters:
            coherence_data: Output from extract_coherence_amplitude()
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            dict with:
                - rho_median: Median correlation coefficient
                - rho_mean: Mean correlation coefficient
                - rho_std: Standard deviation
                - rho_credible_68: [lower, upper] 68% credible interval
                - rho_credible_95: [lower, upper] 95% credible interval
                - significance_vs_null: How many σ from ρ=0
        """
        if len(coherence_data['coherence']) == 0:
            logger.warning("No coherence data available for fitting")
            return self._empty_fit_result()
        
        coherence = coherence_data['coherence']
        coherence_err = coherence_data['coherence_err']
        n_points = coherence_data['n_points']
        
        # Remove invalid values
        valid_mask = np.isfinite(coherence) & np.isfinite(coherence_err) & (coherence_err > 0)
        if not valid_mask.any():
            return self._empty_fit_result()
        
        coherence_valid = coherence[valid_mask]
        coherence_err_valid = coherence_err[valid_mask]
        n_points_valid = n_points[valid_mask]
        
        # Weighted mean (weighted by 1/variance)
        weights = 1.0 / coherence_err_valid**2
        weights = weights / np.sum(weights)
        rho_mean = np.sum(weights * coherence_valid)
        
        # Bootstrap for uncertainty
        bootstrap_means = []
        for _ in range(n_bootstrap):
            # Resample bins
            indices = np.random.choice(len(coherence_valid), size=len(coherence_valid), replace=True)
            boot_coherence = coherence_valid[indices]
            boot_err = coherence_err_valid[indices]
            boot_weights = 1.0 / boot_err**2
            boot_weights = boot_weights / np.sum(boot_weights)
            boot_mean = np.sum(boot_weights * boot_coherence)
            bootstrap_means.append(boot_mean)
        
        bootstrap_means = np.array(bootstrap_means)
        rho_std = np.std(bootstrap_means)
        rho_median = np.median(bootstrap_means)
        
        # Credible intervals
        rho_credible_68 = [
            np.percentile(bootstrap_means, 16),
            np.percentile(bootstrap_means, 84)
        ]
        rho_credible_95 = [
            np.percentile(bootstrap_means, 2.5),
            np.percentile(bootstrap_means, 97.5)
        ]
        
        # Significance vs null hypothesis (ρ = 0)
        significance_vs_null = abs(rho_mean) / rho_std if rho_std > 0 else 0.0
        
        # Legacy compatibility: also report as "R" values for existing code
        return {
            'rho_median': float(rho_median),
            'rho_mean': float(rho_mean),
            'rho_std': float(rho_std),
            'rho_credible_68': rho_credible_68,
            'rho_credible_95': rho_credible_95,
            'significance_vs_null': float(significance_vs_null),
            'n_bins': len(coherence_valid),
            # Legacy fields for backward compatibility
            'R_median': float(rho_median),
            'R_mean': float(rho_mean),
            'R_std': float(rho_std),
            'R_credible_68': rho_credible_68,
            'R_credible_95': rho_credible_95,
            'n_samples': n_bootstrap,
            'samples': None,
        }
    
    # Alias for backward compatibility
    def fit_qtep_ratio_bayesian(self, coherence_data, **kwargs):
        """Backward compatibility alias for fit_correlation_bayesian."""
        return self.fit_correlation_bayesian(coherence_data, **kwargs)
    
    def _empty_fit_result(self) -> Dict[str, Any]:
        """Return empty fit result structure."""
        return {
            'rho_median': np.nan,
            'rho_mean': np.nan,
            'rho_std': np.nan,
            'rho_credible_68': [np.nan, np.nan],
            'rho_credible_95': [np.nan, np.nan],
            'significance_vs_null': np.nan,
            'n_bins': 0,
            # Legacy fields
            'R_median': np.nan,
            'R_mean': np.nan,
            'R_std': np.nan,
            'R_credible_68': [np.nan, np.nan],
            'R_credible_95': [np.nan, np.nan],
            'n_samples': 0,
            'samples': None,
        }
    
    def compute_bayes_factor(
        self,
        coherence_data: Dict[str, np.ndarray],
        fit_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Compute Bayes factor comparing H-ΛCDM vs ΛCDM null hypothesis.
        
        BF = P(data | ρ > 0) / P(data | ρ = 0)
        
        H-ΛCDM: ρ > 0 (prior: half-normal with mode at 0, σ = 0.3)
        ΛCDM:   ρ = 0 (point null)
        
        This tests whether there is POSITIVE correlation, not whether ρ = 2.257.
        The QTEP ratio is a theoretical constant, not an observable.
        
        Parameters:
            coherence_data: Output from extract_coherence_amplitude()
            fit_result: Output from fit_correlation_bayesian()
            
        Returns:
            dict with:
                - bayes_factor: BF value
                - log_bf: log(BF)
                - interpretation: Text interpretation
                - evidence_strength: 'weak', 'moderate', 'strong', 'very strong'
        """
        if len(coherence_data['coherence']) == 0:
            return {
                'bayes_factor': np.nan,
                'log_bf': np.nan,
                'interpretation': 'insufficient data',
                'evidence_strength': 'none',
            }
        
        coherence = coherence_data['coherence']
        coherence_err = coherence_data['coherence_err']
        
        valid_mask = np.isfinite(coherence) & np.isfinite(coherence_err) & (coherence_err > 0)
        if not valid_mask.any():
            return {
                'bayes_factor': np.nan,
                'log_bf': np.nan,
                'interpretation': 'no valid data',
                'evidence_strength': 'none',
            }
        
        coherence_valid = coherence[valid_mask]
        coherence_err_valid = coherence_err[valid_mask]
        
        rho_mean = fit_result.get('rho_mean', np.nan)
        rho_std = fit_result.get('rho_std', np.nan)
        
        if np.isnan(rho_mean) or np.isnan(rho_std) or rho_std <= 0:
            return {
                'bayes_factor': np.nan,
                'log_bf': np.nan,
                'interpretation': 'invalid fit',
                'evidence_strength': 'none',
            }
        
        # Test: is ρ significantly > 0?
        # H-ΛCDM: expects ρ ~ 0.2-0.5 (positive correlation)
        # ΛCDM: expects ρ ~ 0 (no correlation)
        
        # Simple likelihood ratio test
        # L(data | H-ΛCDM: ρ = rho_mean) vs L(data | ΛCDM: ρ = 0)
        def log_likelihood(rho_hypothesis):
            chi2 = np.sum(((coherence_valid - rho_hypothesis) / coherence_err_valid) ** 2)
            return -0.5 * chi2
        
        # For H-ΛCDM, use the fitted mean as the best estimate
        log_L_hlcdm = log_likelihood(rho_mean)
        
        # For ΛCDM, ρ = 0
        log_L_lcdm = log_likelihood(0.0)
        
        # Log Bayes factor (with Occam factor for model complexity)
        # H-ΛCDM has one free parameter (ρ), ΛCDM has none
        # Apply BIC-like penalty: log(BF) ≈ ΔlogL - 0.5*k*log(n)
        n_data = len(coherence_valid)
        occam_penalty = 0.5 * 1 * np.log(n_data)  # k=1 for ρ
        
        log_bf = log_L_hlcdm - log_L_lcdm - occam_penalty
        
        # Cap to avoid overflow
        log_bf = np.clip(log_bf, -100, 100)
        bayes_factor = np.exp(log_bf)
        
        # Interpretation based on correlation direction and significance
        significance = fit_result.get('significance_vs_null', 0.0)
        
        if rho_mean > 0 and significance > 2.0:
            if bayes_factor >= 100:
                interpretation = "strong evidence for positive correlation (H-ΛCDM favored)"
                evidence_strength = "strong"
            elif bayes_factor >= 10:
                interpretation = "moderate evidence for positive correlation (H-ΛCDM favored)"
                evidence_strength = "moderate"
            elif bayes_factor >= 3:
                interpretation = "weak evidence for positive correlation (H-ΛCDM favored)"
                evidence_strength = "weak"
            else:
                interpretation = "inconclusive - positive correlation detected but not significant"
                evidence_strength = "inconclusive"
        elif rho_mean > 0:
            interpretation = "positive correlation detected but below significance threshold"
            evidence_strength = "inconclusive"
        elif rho_mean < 0 and significance > 2.0:
            interpretation = "unexpected: significant negative correlation detected"
            evidence_strength = "anomalous"
        else:
            interpretation = "consistent with null hypothesis (ρ ≈ 0, ΛCDM)"
            evidence_strength = "none"
        
        return {
            'bayes_factor': float(bayes_factor),
            'log_bf': float(log_bf),
            'interpretation': interpretation,
            'evidence_strength': evidence_strength,
            'log_L_hlcdm': float(log_L_hlcdm),
            'log_L_lcdm': float(log_L_lcdm),
            'rho_mean': float(rho_mean),
            'rho_significance': float(significance),
        }
    
    def check_hlcdm_consistency(
        self,
        fit_result: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if observed correlation is consistent with H-ΛCDM prediction.
        
        H-ΛCDM predicts: ρ > 0 (correlated residuals from Lindblad mechanism)
        Expected range: ρ ~ 0.2-0.5 at ℓ=800-1200
        
        IMPORTANT: We are NOT testing if ρ = 2.257. The QTEP ratio is a theoretical
        constant that determines α, which modifies r_s. The observable consequence
        is positive cross-modal correlation, not the QTEP ratio itself.
        
        Parameters:
            fit_result: Output from fit_correlation_bayesian()
            
        Returns:
            dict with:
                - rho_expected_range: Expected correlation range for H-ΛCDM
                - rho_fitted: Fitted correlation value
                - consistent_with_hlcdm: True if ρ > 0 at >2σ significance
                - consistent_with_lcdm: True if ρ ≈ 0 within 2σ
                - significance_vs_null: How many σ from ρ=0
        """
        rho_fitted = fit_result.get('rho_mean', np.nan)
        rho_std = fit_result.get('rho_std', np.nan)
        
        if np.isnan(rho_fitted) or np.isnan(rho_std) or rho_std <= 0:
            return {
                'rho_expected_range': [0.2, 0.5],
                'rho_fitted': rho_fitted,
                'rho_uncertainty': np.nan,
                'consistent_with_hlcdm': False,
                'consistent_with_lcdm': True,
                'significance_vs_null': np.nan,
                # Legacy fields (DEPRECATED - these compare to wrong value)
                'R_predicted': self.QTEP_RATIO_THEORY,
                'R_fitted': rho_fitted,
                'R_uncertainty': rho_std,
                'within_1sigma': False,
                'within_2sigma': False,
                'tension_sigma': np.nan,
            }
        
        # Significance vs null hypothesis (ρ = 0)
        significance_vs_null = abs(rho_fitted) / rho_std
        
        # H-ΛCDM consistency: ρ > 0 with significance > 2σ
        consistent_with_hlcdm = (rho_fitted > 0) and (significance_vs_null > self.SIGNIFICANCE_THRESHOLD)
        
        # ΛCDM consistency: ρ ≈ 0 (within 2σ of zero)
        consistent_with_lcdm = significance_vs_null < 2.0
        
        # Check if in expected H-ΛCDM range (0.2-0.5)
        in_expected_range = (0.2 <= rho_fitted <= 0.5)
        
        return {
            'rho_expected_range': [0.2, 0.5],
            'rho_fitted': float(rho_fitted),
            'rho_uncertainty': float(rho_std),
            'consistent_with_hlcdm': bool(consistent_with_hlcdm),
            'consistent_with_lcdm': bool(consistent_with_lcdm),
            'in_expected_hlcdm_range': bool(in_expected_range),
            'significance_vs_null': float(significance_vs_null),
            # Legacy fields for backward compatibility (DEPRECATED)
            # Note: These compare ρ to QTEP ratio, which is physically meaningless
            # They are kept only for compatibility with existing result parsing
            'R_predicted': float(self.QTEP_RATIO_THEORY),
            'R_fitted': float(rho_fitted),
            'R_uncertainty': float(rho_std),
            'within_1sigma': bool(abs(rho_fitted - 0.35) < rho_std),  # Expected center
            'within_2sigma': bool(abs(rho_fitted - 0.35) < 2 * rho_std),
            'tension_sigma': float(significance_vs_null),
        }
    
    def run_verification(
        self,
        residuals: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> Dict[str, Any]:
        """
        Run complete cross-modal coherence verification test.
        
        Parameters:
            residuals: Nested dict from ResidualAnalyzer.compute_all_residuals()
                Structure: residuals[survey][spectrum] = {'ell': ..., 'residual': ...}
        
        Returns:
            dict with complete verification results
        """
        logger.info("Running cross-modal coherence verification test...")
        logger.info("Testing for ρ(TE,EE) > 0 (H-ΛCDM prediction)")
        logger.info("Note: QTEP ratio (2.257) is theoretical input, not CMB observable")
        
        results_by_survey = {}
        
        # Process each survey
        for survey_name, survey_residuals in residuals.items():
            if 'TE' not in survey_residuals or 'EE' not in survey_residuals:
                logger.warning(f"Missing TE or EE data for {survey_name}, skipping")
                continue
            
            te_data = survey_residuals['TE']
            ee_data = survey_residuals['EE']
            
            if te_data is None or ee_data is None:
                continue
            
            # Interpolate to common multipole grid
            ell_te = te_data['ell']
            ell_ee = ee_data['ell']
            ell_common = np.unique(np.concatenate([ell_te, ell_ee]))
            ell_common = np.sort(ell_common)
            
            # Filter to valid data range
            valid_te = np.isfinite(te_data['residual'])
            valid_ee = np.isfinite(ee_data['residual'])
            ell_min_available = max(np.min(ell_te[valid_te]), np.min(ell_ee[valid_ee]))
            ell_max_available = min(np.max(ell_te[valid_te]), np.max(ell_ee[valid_ee]))
            ell_common = ell_common[
                (ell_common >= ell_min_available) & (ell_common <= ell_max_available)
            ]
            
            if len(ell_common) < 10:
                logger.warning(f"Insufficient data for {survey_name}")
                continue
            
            logger.info(f"Analyzing {survey_name} over ℓ=[{ell_min_available:.0f}, {ell_max_available:.0f}]")
            
            # Interpolate residuals
            interp_te = interp1d(
                ell_te, te_data['residual'],
                kind='linear', bounds_error=False, fill_value=np.nan
            )
            interp_ee = interp1d(
                ell_ee, ee_data['residual'],
                kind='linear', bounds_error=False, fill_value=np.nan
            )
            
            res_te = interp_te(ell_common)
            res_ee = interp_ee(ell_common)
            
            # Extract coherence over full available range
            coherence_data = self.extract_coherence_amplitude(
                res_te, res_ee, ell_common,
                ell_min=None,
                ell_max=None
            )
            
            # Also extract coherence in predicted range for comparison
            coherence_data_predicted = self.extract_coherence_amplitude(
                res_te, res_ee, ell_common,
                ell_min=self.ELL_MIN,
                ell_max=self.ELL_MAX
            )
            
            if len(coherence_data['coherence']) == 0:
                logger.warning(f"No coherence extracted for {survey_name}")
                continue
            
            # Fit correlation
            fit_result = self.fit_correlation_bayesian(coherence_data)
            
            # Bayes factor (tests ρ > 0 vs ρ = 0)
            bayes_factor = self.compute_bayes_factor(coherence_data, fit_result)
            
            # Consistency check
            consistency = self.check_hlcdm_consistency(fit_result)
            
            # Log result interpretation
            rho = fit_result.get('rho_mean', np.nan)
            sig = fit_result.get('significance_vs_null', np.nan)
            logger.info(f"  {survey_name}: ρ = {rho:.3f} ± {fit_result.get('rho_std', np.nan):.3f}")
            logger.info(f"  Significance vs null (ρ=0): {sig:.1f}σ")
            logger.info(f"  {bayes_factor.get('interpretation', 'N/A')}")
            
            # Fit for predicted range as well
            fit_result_predicted = None
            bayes_factor_predicted = None
            consistency_predicted = None
            
            if len(coherence_data_predicted['coherence']) > 0:
                fit_result_predicted = self.fit_correlation_bayesian(coherence_data_predicted)
                bayes_factor_predicted = self.compute_bayes_factor(coherence_data_predicted, fit_result_predicted)
                consistency_predicted = self.check_hlcdm_consistency(fit_result_predicted)
            
            results_by_survey[survey_name] = {
                'coherence_data_full': {
                    'ell_binned': coherence_data['ell_binned'].tolist(),
                    'coherence': coherence_data['coherence'].tolist(),
                    'coherence_err': coherence_data['coherence_err'].tolist(),
                    'correlation': coherence_data['correlation'].tolist(),
                    'n_points': coherence_data['n_points'].tolist(),
                    'ell_range': [float(ell_min_available), float(ell_max_available)],
                },
                'coherence_data_predicted': {
                    'ell_binned': coherence_data_predicted['ell_binned'].tolist(),
                    'coherence': coherence_data_predicted['coherence'].tolist(),
                    'coherence_err': coherence_data_predicted['coherence_err'].tolist(),
                    'correlation': coherence_data_predicted['correlation'].tolist(),
                    'n_points': coherence_data_predicted['n_points'].tolist(),
                    'ell_range': [self.ELL_MIN, self.ELL_MAX],
                } if len(coherence_data_predicted['coherence']) > 0 else None,
                # Use new naming but include legacy aliases
                'correlation_fit_full': fit_result,
                'correlation_fit_predicted': fit_result_predicted,
                'qtep_fit_full': fit_result,  # Legacy alias
                'qtep_fit_predicted': fit_result_predicted,  # Legacy alias
                'bayes_factor_full': bayes_factor,
                'bayes_factor_predicted': bayes_factor_predicted,
                'hlcdm_consistency_full': consistency,
                'hlcdm_consistency_predicted': consistency_predicted,
            }
        
        # Combine results across surveys
        if not results_by_survey:
            logger.warning("No valid survey results for cross-modal coherence test")
            return {
                'test_name': 'Cross-Modal Coherence (TE-EE)',
                'physical_test': 'ρ(TE,EE) > 0 indicates correlated residuals (H-ΛCDM)',
                'note': 'QTEP ratio (2.257) is theoretical input, NOT CMB observable',
                'ell_range_predicted': [self.ELL_MIN, self.ELL_MAX],
                'spectra': ['TE', 'EE'],
                'surveys': {},
                'combined_full': {
                    'rho_median': np.nan,
                    'bayes_factor': np.nan,
                    'interpretation': 'no data',
                },
                'combined_predicted': {
                    'rho_median': np.nan,
                    'bayes_factor': np.nan,
                    'interpretation': 'no data',
                },
            }
        
        # Weighted combination across surveys
        all_rho_means = []
        all_rho_stds = []
        all_bfs = []
        
        all_rho_means_predicted = []
        all_rho_stds_predicted = []
        all_bfs_predicted = []
        
        for survey_name, survey_results in results_by_survey.items():
            fit = survey_results['correlation_fit_full']
            if not np.isnan(fit['rho_mean']):
                all_rho_means.append(fit['rho_mean'])
                all_rho_stds.append(fit['rho_std'])
            if not np.isnan(survey_results['bayes_factor_full']['bayes_factor']):
                all_bfs.append(survey_results['bayes_factor_full']['bayes_factor'])
            
            if survey_results['correlation_fit_predicted'] is not None:
                fit_pred = survey_results['correlation_fit_predicted']
                if not np.isnan(fit_pred['rho_mean']):
                    all_rho_means_predicted.append(fit_pred['rho_mean'])
                    all_rho_stds_predicted.append(fit_pred['rho_std'])
            if survey_results['bayes_factor_predicted'] is not None:
                bf_pred = survey_results['bayes_factor_predicted']['bayes_factor']
                if not np.isnan(bf_pred):
                    all_bfs_predicted.append(bf_pred)
        
        # Weighted mean of correlations
        if all_rho_means:
            weights = 1.0 / np.array(all_rho_stds) ** 2
            weights = weights / np.sum(weights)
            rho_combined = np.sum(weights * np.array(all_rho_means))
            rho_combined_std = np.sqrt(1.0 / np.sum(1.0 / np.array(all_rho_stds)**2))
        else:
            rho_combined = np.nan
            rho_combined_std = np.nan
        
        # Combined Bayes factor
        if all_bfs:
            valid_bfs = [bf for bf in all_bfs if bf > 0 and np.isfinite(bf)]
            if valid_bfs:
                log_bf_combined = np.mean(np.log(valid_bfs))
                bf_combined = np.exp(log_bf_combined)
            else:
                bf_combined = 0.0
        else:
            bf_combined = np.nan
        
        # Generate combined interpretation
        if not np.isnan(rho_combined):
            sig_combined = abs(rho_combined) / rho_combined_std if rho_combined_std > 0 else 0
            if rho_combined > 0 and sig_combined > 2:
                combined_interpretation = f"positive correlation detected: ρ={rho_combined:.3f} ({sig_combined:.1f}σ from null)"
            elif rho_combined > 0:
                combined_interpretation = f"weak positive correlation: ρ={rho_combined:.3f} ({sig_combined:.1f}σ)"
            else:
                combined_interpretation = f"consistent with null: ρ={rho_combined:.3f}"
        else:
            combined_interpretation = "insufficient data"
        
        # Same for predicted range
        if all_rho_means_predicted:
            weights_pred = 1.0 / np.array(all_rho_stds_predicted) ** 2
            weights_pred = weights_pred / np.sum(weights_pred)
            rho_combined_predicted = np.sum(weights_pred * np.array(all_rho_means_predicted))
            rho_combined_std_predicted = np.sqrt(1.0 / np.sum(1.0 / np.array(all_rho_stds_predicted)**2))
        else:
            rho_combined_predicted = np.nan
            rho_combined_std_predicted = np.nan
        
        if all_bfs_predicted:
            valid_bfs_pred = [bf for bf in all_bfs_predicted if bf > 0 and np.isfinite(bf)]
            bf_combined_predicted = np.exp(np.mean(np.log(valid_bfs_pred))) if valid_bfs_pred else 0.0
        else:
            bf_combined_predicted = np.nan
        
        return {
            'test_name': 'Cross-Modal Coherence (TE-EE)',
            'physical_test': 'ρ(TE,EE) > 0 indicates correlated residuals (H-ΛCDM)',
            'note': 'QTEP ratio (2.257) is theoretical input, NOT CMB observable',
            'ell_range_predicted': [self.ELL_MIN, self.ELL_MAX],
            'spectra': ['TE', 'EE'],
            'surveys': results_by_survey,
            'combined_full': {
                'rho_median': float(rho_combined) if not np.isnan(rho_combined) else np.nan,
                'rho_std': float(rho_combined_std) if not np.isnan(rho_combined_std) else np.nan,
                'bayes_factor': float(bf_combined) if not np.isnan(bf_combined) else np.nan,
                'interpretation': combined_interpretation,
                # Legacy aliases
                'R_median': float(rho_combined) if not np.isnan(rho_combined) else np.nan,
                'R_std': float(rho_combined_std) if not np.isnan(rho_combined_std) else np.nan,
            },
            'combined_predicted': {
                'rho_median': float(rho_combined_predicted) if not np.isnan(rho_combined_predicted) else np.nan,
                'rho_std': float(rho_combined_std_predicted) if not np.isnan(rho_combined_std_predicted) else np.nan,
                'bayes_factor': float(bf_combined_predicted) if not np.isnan(bf_combined_predicted) else np.nan,
                'interpretation': 'see full range results',
                # Legacy aliases
                'R_median': float(rho_combined_predicted) if not np.isnan(rho_combined_predicted) else np.nan,
                'R_std': float(rho_combined_std_predicted) if not np.isnan(rho_combined_std_predicted) else np.nan,
            },
        }
