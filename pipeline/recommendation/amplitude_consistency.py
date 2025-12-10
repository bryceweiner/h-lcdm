"""
Amplitude Consistency Check - Phase 5
=====================================

Validates that α_CMB from TT, TE, EE agrees with theoretical prediction.

The BAO paper predicts α = -5.7 ± 2.0 from first principles.
This module fits α from CMB residuals and checks consistency.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from scipy.optimize import curve_fit
from scipy import stats

from .hlcdm_camb_model import HLCDMCAMBModel

logger = logging.getLogger(__name__)


class AmplitudeConsistencyChecker:
    """
    Check consistency of Lindblad-Zeno amplitude α across CMB spectra.
    
    Fits α from residuals and verifies:
    1. α_CMB agrees across TT, TE, EE within errors
    2. α_CMB falls within theoretical prior [-7.7, -3.7]
    3. α_CMB consistent with α_BAO ≈ -7.2
    """
    
    # Theoretical predictions
    ALPHA_THEORY = -5.7
    ALPHA_THEORY_ERR = 2.0
    ALPHA_PRIOR = (-7.7, -3.7)  # Theoretical bounds
    ALPHA_BAO = -7.2  # From BAO sensitivity analysis
    
    def __init__(self, hlcdm_model: Optional[HLCDMCAMBModel] = None):
        """
        Initialize amplitude consistency checker.
        
        Parameters:
            hlcdm_model: H-ΛCDM model instance (creates new if None)
        """
        self.hlcdm_model = hlcdm_model or HLCDMCAMBModel()
    
    def fit_alpha_from_cmb(
        self,
        residuals: np.ndarray,
        cl_lcdm: np.ndarray,
        ell: np.ndarray,
        cl_err: np.ndarray,
        params: Dict[str, float],
        spectrum: str = 'TT'
    ) -> Dict[str, float]:
        """
        Fit α from CMB residuals.
        
        From the relation:
        α_CMB = (H/γ) × (ΔC_ℓ/C_ℓ^ΛCDM) / (1 - αγ/H)
        
        For small corrections where αγ/H << 1, this simplifies to:
        α_CMB ≈ (H/γ) × (ΔC_ℓ/C_ℓ^ΛCDM)
        
        We compute α for each data point and take a weighted mean.
        
        Parameters:
            residuals: Residual values ΔC_ℓ
            cl_lcdm: ΛCDM theoretical spectrum C_ℓ^ΛCDM
            ell: Multipole array
            cl_err: Observational errors
            params: Cosmological parameters
            spectrum: Spectrum type ('TT', 'TE', 'EE')
            
        Returns:
            dict with:
                - alpha: Fitted α value (weighted mean)
                - alpha_err: Uncertainty on α
                - chi2: χ² of fit
                - reduced_chi2: Reduced χ²
        """
        # Remove NaN values
        valid_mask = np.isfinite(residuals) & np.isfinite(cl_lcdm) & np.isfinite(cl_err) & (cl_lcdm != 0)
        if not valid_mask.any() or valid_mask.sum() < 3:
            return {
                'alpha': np.nan,
                'alpha_err': np.nan,
                'chi2': np.nan,
                'reduced_chi2': np.nan,
            }
        
        ell_valid = ell[valid_mask]
        residuals_valid = residuals[valid_mask]
        cl_lcdm_valid = cl_lcdm[valid_mask]
        cl_err_valid = cl_err[valid_mask]
        
        # Compute fractional residuals: ΔC_ℓ/C_ℓ^ΛCDM
        residual_fraction = residuals_valid / cl_lcdm_valid
        
        # Compute H/γ ratio
        H0 = params.get('H0', 67.36)  # km/s/Mpc
        gamma = self.hlcdm_model.compute_gamma(H0)  # Same units as H0
        H_gamma_ratio = H0 / gamma
        
        # For each data point: α_i = (H/γ) × (ΔC_ℓ/C_ℓ^ΛCDM)_i
        alpha_per_point = H_gamma_ratio * residual_fraction
        
        # Compute weights from fractional errors
        # Error on residual_fraction: σ_fraction = σ_Cℓ / C_ℓ
        fractional_error = cl_err_valid / cl_lcdm_valid
        weights = 1.0 / (fractional_error ** 2)
        
        # Remove infinite weights
        finite_weight_mask = np.isfinite(weights) & (weights > 0)
        if not finite_weight_mask.any():
            return {
                'alpha': np.nan,
                'alpha_err': np.nan,
                'chi2': np.nan,
                'reduced_chi2': np.nan,
            }
        
        alpha_per_point = alpha_per_point[finite_weight_mask]
        weights = weights[finite_weight_mask]
        
        # Weighted mean: α = Σ(w_i × α_i) / Σ(w_i)
        weights_normalized = weights / np.sum(weights)
        alpha_fit = np.sum(weights_normalized * alpha_per_point)
        
        # Weighted standard deviation (uncertainty on mean)
        variance = np.sum(weights_normalized * (alpha_per_point - alpha_fit) ** 2)
        alpha_err = np.sqrt(variance)
        
        # Compute χ²: compare observed α_i to fitted α
        chi2 = np.sum(weights * (alpha_per_point - alpha_fit) ** 2)
        n_data = len(alpha_per_point)
        reduced_chi2 = chi2 / (n_data - 1) if n_data > 1 else np.nan
        
        return {
            'alpha': alpha_fit,
            'alpha_err': alpha_err,
            'chi2': chi2,
            'reduced_chi2': reduced_chi2,
            'n_data': n_data,
        }
    
    def check_all_spectra(
        self,
        residuals: Dict[str, Dict[str, Dict[str, np.ndarray]]],
        theoretical_spectra: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
        params: Dict[str, float]
    ) -> Dict[str, Any]:
        """
        Fit α from all spectra and check consistency.
        
        Parameters:
            residuals: Residuals from ResidualAnalyzer
            theoretical_spectra: Theoretical ΛCDM spectra
                Structure: theoretical_spectra[survey][spectrum] = (ell, cl)
            params: Cosmological parameters
            
        Returns:
            dict with α fits and consistency checks
        """
        alpha_results = {}
        
        for survey_name, survey_residuals in residuals.items():
            alpha_results[survey_name] = {}
            
            for spectrum in ['TT', 'TE', 'EE']:
                if spectrum not in survey_residuals or survey_residuals[spectrum] is None:
                    continue
                
                if survey_name not in theoretical_spectra or spectrum not in theoretical_spectra[survey_name]:
                    continue
                
                res_data = survey_residuals[spectrum]
                
                # Use the theoretical spectrum already computed in residual analysis
                # This is already interpolated to the observed multipoles
                cl_theory = res_data.get('cl_theory', None)
                if cl_theory is None:
                    # Fallback: interpolate from theoretical_spectra if needed
                    if survey_name in theoretical_spectra and spectrum in theoretical_spectra[survey_name]:
                        ell_theory, cl_theory_full = theoretical_spectra[survey_name][spectrum]
                        from scipy.interpolate import interp1d
                        interp_func = interp1d(ell_theory, cl_theory_full, kind='linear', bounds_error=False, fill_value=np.nan)
                        cl_theory = interp_func(res_data['ell'])
                    else:
                        continue
                
                # Fit α
                alpha_fit = self.fit_alpha_from_cmb(
                    res_data['residual'],
                    cl_theory,
                    res_data['ell'],
                    res_data['cl_err'],
                    params,
                    spectrum=spectrum
                )
                
                alpha_results[survey_name][spectrum] = alpha_fit
        
        # Check consistency
        consistency = self.check_consistency(alpha_results)
        
        return {
            'alpha_fits': alpha_results,
            'consistency': consistency,
        }
    
    def check_consistency(
        self,
        alpha_results: Dict[str, Dict[str, Dict[str, float]]]
    ) -> Dict[str, Any]:
        """
        Check consistency of α values across spectra.
        
        Verifies:
        1. α_CMB agrees across TT, TE, EE within errors
        2. α_CMB falls within [-7.7, -3.7]
        3. α_CMB consistent with α_BAO ≈ -7.2
        
        Parameters:
            alpha_results: Results from fit_alpha_from_cmb
            
        Returns:
            dict with consistency checks
        """
        all_alphas = []
        all_alpha_errs = []
        spectrum_labels = []
        
        # Collect all α values
        for survey_name, survey_results in alpha_results.items():
            for spectrum, alpha_fit in survey_results.items():
                if not np.isnan(alpha_fit.get('alpha', np.nan)):
                    all_alphas.append(alpha_fit['alpha'])
                    all_alpha_errs.append(alpha_fit.get('alpha_err', np.inf))
                    spectrum_labels.append(f"{survey_name}_{spectrum}")
        
        if not all_alphas:
            return {
                'mean_alpha': np.nan,
                'std_alpha': np.nan,
                'within_prior': False,
                'consistent_with_bao': False,
                'consistent_across_spectra': False,
                'n_measurements': 0,
            }
        
        all_alphas = np.array(all_alphas)
        all_alpha_errs = np.array(all_alpha_errs)
        
        # Weighted mean
        weights = 1.0 / (all_alpha_errs ** 2)
        weights = weights / np.sum(weights)
        mean_alpha = np.sum(weights * all_alphas)
        
        # Weighted standard deviation
        variance = np.sum(weights * (all_alphas - mean_alpha) ** 2)
        std_alpha = np.sqrt(variance)
        
        # Check if within theoretical prior
        within_prior = (self.ALPHA_PRIOR[0] <= mean_alpha <= self.ALPHA_PRIOR[1])
        
        # Check consistency with BAO value
        bao_diff = abs(mean_alpha - self.ALPHA_BAO)
        bao_consistent = bao_diff < 3.0 * std_alpha  # Within 3σ
        
        # Check consistency with theory
        theory_diff = abs(mean_alpha - self.ALPHA_THEORY)
        theory_consistent = theory_diff < max(3.0 * std_alpha, self.ALPHA_THEORY_ERR)
        
        # Check consistency across spectra (TT, TE, EE should agree)
        # Group by spectrum type
        tt_alphas = []
        te_alphas = []
        ee_alphas = []
        
        for i, label in enumerate(spectrum_labels):
            if '_TT' in label:
                tt_alphas.append(all_alphas[i])
            elif '_TE' in label:
                te_alphas.append(all_alphas[i])
            elif '_EE' in label:
                ee_alphas.append(all_alphas[i])
        
        consistent_across_spectra = True
        if len(tt_alphas) > 0 and len(te_alphas) > 0:
            tt_mean = np.mean(tt_alphas)
            te_mean = np.mean(te_alphas)
            if abs(tt_mean - te_mean) > 2.0 * std_alpha:
                consistent_across_spectra = False
        
        if len(tt_alphas) > 0 and len(ee_alphas) > 0:
            tt_mean = np.mean(tt_alphas)
            ee_mean = np.mean(ee_alphas)
            if abs(tt_mean - ee_mean) > 2.0 * std_alpha:
                consistent_across_spectra = False
        
        return {
            'mean_alpha': mean_alpha,
            'std_alpha': std_alpha,
            'within_prior': within_prior,
            'consistent_with_bao': bao_consistent,
            'consistent_with_theory': theory_consistent,
            'consistent_across_spectra': consistent_across_spectra,
            'n_measurements': len(all_alphas),
            'alpha_tt': np.mean(tt_alphas) if tt_alphas else np.nan,
            'alpha_te': np.mean(te_alphas) if te_alphas else np.nan,
            'alpha_ee': np.mean(ee_alphas) if ee_alphas else np.nan,
        }

