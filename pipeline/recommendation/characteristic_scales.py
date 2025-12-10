"""
Characteristic Scale Analysis - Phase 3
=======================================

Fourier analysis to detect H-ΛCDM signature at predicted scales.

From BAO paper: r_s/λ_Silk ≈ 15.4 implies characteristic multipole ℓ_char ≈ 1000
and characteristic width Δℓ ≈ 100-150.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from scipy import signal
from scipy.fft import fft, fftfreq

logger = logging.getLogger(__name__)


class CharacteristicScaleAnalyzer:
    """
    Analyze residuals for characteristic H-ΛCDM scales.
    
    The BAO paper predicts:
    - Characteristic multipole: ℓ_char ≈ 1000
    - Characteristic width: Δℓ ≈ 100-150
    """
    
    # From BAO paper: r_s/λ_Silk ≈ 15.4
    RS_LAMBDA_SILK_RATIO = 15.4
    
    # Predicted characteristic scales
    EXPECTED_CHAR_ELL = 1000.0
    EXPECTED_DELTA_ELL = 125.0  # Mean of 100-150 range
    DELTA_ELL_RANGE = (100, 150)
    
    # Typical angular diameter distance to last scattering
    DA_RS_RATIO = 14000.0  # Approximate D_A/r_s at z=1100
    
    def __init__(self):
        """Initialize characteristic scale analyzer."""
        pass
    
    def compute_characteristic_multipole(self, params: Optional[Dict] = None) -> float:
        """
        Compute characteristic multipole: ℓ_char ≈ π(D_A/r_s) × (r_s/λ_Silk) ≈ 1000
        
        Parameters:
            params: Optional cosmological parameters (uses defaults if None)
            
        Returns:
            float: Characteristic multipole
        """
        # ℓ_char = π × (D_A/r_s) × (r_s/λ_Silk)
        char_ell = np.pi * self.DA_RS_RATIO * (1.0 / self.RS_LAMBDA_SILK_RATIO)
        return char_ell
    
    def fourier_analysis(
        self,
        residuals: np.ndarray,
        ell: np.ndarray,
        spectrum: str = 'TT'
    ) -> Dict[str, np.ndarray]:
        """
        Perform Fourier analysis of residuals to detect characteristic features.
        
        Looks for power excess at Δℓ ≈ 100-150 (width from modular spectrum's
        QTEP-defined projection).
        
        Parameters:
            residuals: Residual values ΔC_ℓ
            ell: Multipole array
            spectrum: Spectrum type ('TT', 'TE', 'EE')
            
        Returns:
            dict with keys:
                - k: Wavenumber (1/Δℓ)
                - power: Power spectrum of residuals
                - peak_k: Wavenumber of maximum power
                - peak_power: Maximum power value
                - expected_k: Expected wavenumber from Δℓ prediction
        """
        # Remove NaN values
        valid_mask = np.isfinite(residuals)
        if not valid_mask.any() or valid_mask.sum() < 10:
            return {
                'k': np.array([]),
                'power': np.array([]),
                'peak_k': np.nan,
                'peak_power': np.nan,
                'expected_k': 1.0 / self.EXPECTED_DELTA_ELL,
            }
        
        ell_valid = ell[valid_mask]
        residuals_valid = residuals[valid_mask]
        
        # Ensure uniform spacing (interpolate if needed)
        ell_min = np.min(ell_valid)
        ell_max = np.max(ell_valid)
        ell_step = np.median(np.diff(np.sort(ell_valid)))
        
        # Create uniform grid
        ell_uniform = np.arange(ell_min, ell_max + ell_step, ell_step)
        interp_func = np.interp(ell_uniform, ell_valid, residuals_valid)
        
        # Compute FFT
        n = len(ell_uniform)
        fft_vals = fft(interp_func)
        power = np.abs(fft_vals) ** 2
        
        # Wavenumber in units of 1/Δℓ
        k = fftfreq(n, d=ell_step)
        k = k[:n//2]  # Only positive frequencies
        power = power[:n//2]
        
        # Find peak
        if len(power) > 0:
            peak_idx = np.argmax(power[1:]) + 1  # Skip DC component
            peak_k = k[peak_idx]
            peak_power = power[peak_idx]
        else:
            peak_k = np.nan
            peak_power = np.nan
        
        # Expected wavenumber from Δℓ prediction
        expected_k = 1.0 / self.EXPECTED_DELTA_ELL
        
        return {
            'k': k,
            'power': power,
            'peak_k': peak_k,
            'peak_power': peak_power,
            'expected_k': expected_k,
            'ell': ell_uniform,
            'residuals_interp': interp_func,
        }
    
    def detect_characteristic_features(
        self,
        residuals: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> Dict[str, Any]:
        """
        Search for H-ΛCDM signature at predicted scales.
        
        Parameters:
            residuals: Nested dict from ResidualAnalyzer.compute_all_residuals()
                Structure: residuals[survey][spectrum] = {'ell': ..., 'residual': ...}
        
        Returns:
            dict with detection results for each survey/spectrum combination
        """
        results = {}
        
        char_ell = self.compute_characteristic_multipole()
        
        for survey_name, survey_residuals in residuals.items():
            results[survey_name] = {}
            
            for spectrum in ['TT', 'TE', 'EE']:
                if spectrum not in survey_residuals or survey_residuals[spectrum] is None:
                    continue
                
                data = survey_residuals[spectrum]
                ell = data['ell']
                residual = data['residual']
                
                # Fourier analysis
                fourier = self.fourier_analysis(residual, ell, spectrum)
                
                # Check if peak is near expected scale
                peak_k = fourier['peak_k']
                expected_k = fourier['expected_k']
                
                if not np.isnan(peak_k) and not np.isnan(expected_k):
                    k_match = abs(peak_k - expected_k) / expected_k < 0.5  # Within 50%
                else:
                    k_match = False
                
                # Check for excess power near characteristic multipole
                ell_mask = (ell >= char_ell - 200) & (ell <= char_ell + 200)
                if ell_mask.any():
                    excess_near_char = np.nanmean(np.abs(residual[ell_mask]))
                    excess_away = np.nanmean(np.abs(residual[~ell_mask])) if (~ell_mask).any() else 0
                    excess_ratio = excess_near_char / excess_away if excess_away > 0 else np.nan
                else:
                    excess_ratio = np.nan
                
                results[survey_name][spectrum] = {
                    'characteristic_ell': char_ell,
                    'fourier_analysis': fourier,
                    'peak_matches_expected': k_match,
                    'excess_near_characteristic': excess_ratio,
                    'mean_residual_near_char': excess_near_char if ell_mask.any() else np.nan,
                }
        
        # Summary across all surveys/spectra
        summary = self._compute_summary(results)
        
        return {
            'by_survey': results,
            'summary': summary,
            'expected_characteristic_ell': char_ell,
            'expected_delta_ell': self.EXPECTED_DELTA_ELL,
        }
    
    def _compute_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics across all surveys/spectra."""
        all_peaks_match = []
        all_excess_ratios = []
        
        for survey_name, survey_results in results.items():
            for spectrum, spectrum_results in survey_results.items():
                if 'peak_matches_expected' in spectrum_results:
                    all_peaks_match.append(spectrum_results['peak_matches_expected'])
                if 'excess_near_characteristic' in spectrum_results:
                    excess = spectrum_results['excess_near_characteristic']
                    if not np.isnan(excess):
                        all_excess_ratios.append(excess)
        
        return {
            'n_peaks_matching_expected': sum(all_peaks_match) if all_peaks_match else 0,
            'n_total_tests': len(all_peaks_match) if all_peaks_match else 0,
            'mean_excess_ratio': np.nanmean(all_excess_ratios) if all_excess_ratios else np.nan,
            'fraction_matching': sum(all_peaks_match) / len(all_peaks_match) if all_peaks_match else 0.0,
        }

