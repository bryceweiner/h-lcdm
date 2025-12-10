"""
Cross-Modal Coherence Test - Phase 2
====================================

THE KEY DISCRIMINANT between H-ΛCDM and noise/systematics.

The Lindblad-Zeno mechanism predicts correlated residuals across TT, TE, EE
because the coherent acoustic enhancement modifies the same underlying
baryon-photon modes.

H-ΛCDM prediction: ρ > 0, peak at ℓ ≈ 800-1200
ΛCDM null: ρ ≈ 0 (residuals are uncorrelated noise)
"""

import numpy as np
from typing import Dict, Tuple, Optional, List, Any
import logging
from scipy import stats

logger = logging.getLogger(__name__)


class CrossModalCoherenceTest:
    """
    Test for cross-modal coherence in CMB residuals at acoustic harmonics.
    
    The smoking gun test: H-ΛCDM predicts correlated residuals across TT/TE/EE
    at ALL acoustic peak positions due to coherent enhancement of the same 
    underlying baryon-photon modes.
    
    Method:
    1. Identify actual acoustic peaks from ΛCDM theory spectrum
    2. Test if residuals correlate at those specific positions
    3. Compare peak vs off-peak correlation enhancement
    """
    
    # Window size around each peak for correlation analysis
    PEAK_WINDOW = 50  # ±50 multipoles
    
    # Minimum peak prominence for detection (relative to local baseline)
    PEAK_PROMINENCE = 0.1
    
    def __init__(self):
        """Initialize cross-modal coherence tester."""
        pass
    
    def identify_acoustic_peaks(
        self,
        ell: np.ndarray,
        cl_theory: np.ndarray,
        min_ell: float = 200,
        max_ell: float = 2500
    ) -> np.ndarray:
        """
        Identify actual acoustic peak positions from theory spectrum.
        
        Converts C_ℓ → D_ℓ to make acoustic oscillations prominent,
        then uses scipy.signal.find_peaks to locate maxima.
        
        Parameters:
            ell: Multipole array
            cl_theory: Theoretical power spectrum (C_ℓ in μK²)
            min_ell: Minimum multipole to search
            max_ell: Maximum multipole to search
            
        Returns:
            np.ndarray: Peak positions in ℓ
        """
        from scipy.signal import find_peaks
        
        # Restrict to search range
        search_mask = (ell >= min_ell) & (ell <= max_ell) & np.isfinite(cl_theory) & (ell > 0)
        ell_search = ell[search_mask]
        cl_search = cl_theory[search_mask]
        
        if len(ell_search) < 10:
            logger.warning("Insufficient data range for peak finding")
            return np.array([])
        
        # Convert C_ℓ to D_ℓ to make acoustic peaks prominent
        # D_ℓ = ℓ(ℓ+1) C_ℓ / (2π)
        dl_search = ell_search * (ell_search + 1) * cl_search / (2 * np.pi)
        
        # Find peaks in D_ℓ
        # Acoustic peaks occur at roughly harmonic intervals (Δℓ ~ 270-320)
        # Use prominence relative to local baseline, not global maximum
        peak_indices, properties = find_peaks(
            dl_search,
            prominence=100,  # Absolute prominence in μK² (catches peaks at all ℓ)
            distance=150  # Acoustic peaks separated by ~270-320 ℓ
        )
        
        if len(peak_indices) == 0:
            logger.warning("No acoustic peaks found in spectrum")
            return np.array([])
        
        peak_ells = ell_search[peak_indices]
        peak_dls = dl_search[peak_indices]
        
        logger.info(f"Identified {len(peak_ells)} acoustic peaks:")
        for i, (peak_ell, peak_dl) in enumerate(zip(peak_ells, peak_dls)):
            logger.info(f"  Peak {i+1}: ℓ={peak_ell:.0f}, D_ℓ={peak_dl:.2f} μK²")
        
        return peak_ells
    
    def compute_acoustic_peak_correlations(
        self,
        residuals_XY: np.ndarray,
        residuals_WZ: np.ndarray,
        ell: np.ndarray,
        cl_theory: np.ndarray
    ) -> Dict[str, Any]:
        """
        Compute correlations at ACTUAL acoustic peak positions from data.
        
        Tests the H-ΛCDM prediction: residuals should be correlated at ALL
        acoustic peaks identified from the theory spectrum.
        
        Parameters:
            residuals_XY: Residuals for first spectrum
            residuals_WZ: Residuals for second spectrum
            ell: Multipole array
            cl_theory: Theoretical spectrum to identify peaks from
            
        Returns:
            dict with acoustic peak correlation statistics
        """
        # Remove NaN values
        valid_mask = np.isfinite(residuals_XY) & np.isfinite(residuals_WZ) & np.isfinite(ell) & np.isfinite(cl_theory)
        if not valid_mask.any():
            return self._empty_acoustic_results()
        
        ell_valid = ell[valid_mask]
        res_XY = residuals_XY[valid_mask]
        res_WZ = residuals_WZ[valid_mask]
        cl_valid = cl_theory[valid_mask]
        
        # Identify actual acoustic peaks from theory spectrum
        peak_ells = self.identify_acoustic_peaks(ell_valid, cl_valid)
        
        if len(peak_ells) == 0:
            logger.warning("No acoustic peaks identified")
            return self._empty_acoustic_results()
        
        # Build masks for peak vs off-peak regions
        peak_mask = np.zeros(len(ell_valid), dtype=bool)
        for peak_ell in peak_ells:
            window = (ell_valid >= peak_ell - self.PEAK_WINDOW) & (ell_valid <= peak_ell + self.PEAK_WINDOW)
            peak_mask |= window
        
        offpeak_mask = ~peak_mask
        
        # GLOBAL correlation at ALL peak positions combined
        if np.sum(peak_mask) >= 3:
            try:
                rho_peak_global, p_peak_global = stats.pearsonr(res_XY[peak_mask], res_WZ[peak_mask])
            except:
                rho_peak_global = 0.0
                p_peak_global = 1.0
        else:
            rho_peak_global = 0.0
            p_peak_global = 1.0
        
        # GLOBAL correlation at ALL off-peak positions combined
        if np.sum(offpeak_mask) >= 3:
            try:
                rho_offpeak_global, p_offpeak_global = stats.pearsonr(res_XY[offpeak_mask], res_WZ[offpeak_mask])
            except:
                rho_offpeak_global = 0.0
                p_offpeak_global = 1.0
        else:
            rho_offpeak_global = 0.0
            p_offpeak_global = 1.0
        
        # Enhancement ratio: Is correlation STRONGER at peaks?
        # Use absolute values because sign doesn't matter for "correlation strength"
        abs_rho_peak = np.abs(rho_peak_global)
        abs_rho_offpeak = np.abs(rho_offpeak_global)
        enhancement = (abs_rho_peak / abs_rho_offpeak) if abs_rho_offpeak > 1e-10 else np.nan
        
        # Also compute per-peak correlations for detailed inspection
        peak_correlations = []
        peak_p_values = []
        for peak_ell in peak_ells:
            window = (ell_valid >= peak_ell - self.PEAK_WINDOW) & (ell_valid <= peak_ell + self.PEAK_WINDOW)
            if np.sum(window) >= 3:
                try:
                    rho, p_val = stats.pearsonr(res_XY[window], res_WZ[window])
                    peak_correlations.append(rho)
                    peak_p_values.append(p_val)
                except:
                    peak_correlations.append(np.nan)
                    peak_p_values.append(np.nan)
            else:
                peak_correlations.append(np.nan)
                peak_p_values.append(np.nan)
        
        n_significant = np.sum(np.array(peak_p_values) < 0.05)
        
        logger.info(f"Acoustic peak correlation test: {len(peak_ells)} peaks identified")
        logger.info(f"  Global ρ at ALL peak positions: {rho_peak_global:.4f} (p={p_peak_global:.4f})")
        logger.info(f"  Global ρ at off-peak positions: {rho_offpeak_global:.4f} (p={p_offpeak_global:.4f})")
        logger.info(f"  Enhancement ratio: {enhancement:.2f}×" if np.isfinite(enhancement) else "  Enhancement ratio: N/A")
        
        return {
            'acoustic_peaks': peak_ells.tolist(),
            'correlations_at_individual_peaks': peak_correlations,  # Per-peak for inspection
            'p_values_at_individual_peaks': peak_p_values,  # Per-peak for inspection
            'rho_peak_global': float(rho_peak_global),  # Global correlation at ALL peaks
            'p_peak_global': float(p_peak_global),
            'rho_offpeak_global': float(rho_offpeak_global),  # Global correlation off-peak
            'p_offpeak_global': float(p_offpeak_global),
            'n_peaks_identified': len(peak_ells),
            'n_peaks_significant': int(n_significant),  # How many individual peaks have p<0.05
            'enhancement_ratio': float(enhancement) if np.isfinite(enhancement) else None,
            'n_points_at_peaks': int(np.sum(peak_mask)),  # Sample size at peaks
            'n_points_offpeak': int(np.sum(offpeak_mask)),  # Sample size off-peak
        }
    
    def compute_ml_targeted_correlation(
        self,
        residuals_XY: np.ndarray,
        residuals_WZ: np.ndarray,
        ell: np.ndarray,
        ell_min: float = 800,
        ell_max: float = 1200
    ) -> Dict[str, Any]:
        """
        Test correlation at specific ℓ range identified by ML pipeline.
        
        This tests the ML recommendation: compute ρ(ℓ) at the specific
        multipole range where ML detected cross-modal coherence.
        
        Parameters:
            residuals_XY: Residuals for first spectrum (e.g., TT)
            residuals_WZ: Residuals for second spectrum (e.g., TE)
            ell: Multipole array
            ell_min: Minimum ℓ of target range
            ell_max: Maximum ℓ of target range
            
        Returns:
            dict with correlation in target range vs outside range
        """
        # Remove NaN values
        valid_mask = np.isfinite(residuals_XY) & np.isfinite(residuals_WZ) & np.isfinite(ell)
        if not valid_mask.any():
            return self._empty_ml_targeted_results()
        
        ell_valid = ell[valid_mask]
        res_XY = residuals_XY[valid_mask]
        res_WZ = residuals_WZ[valid_mask]
        
        # Target range mask (ML-identified range)
        target_mask = (ell_valid >= ell_min) & (ell_valid <= ell_max)
        outside_mask = ~target_mask
        
        # Correlation IN target range
        if np.sum(target_mask) >= 3:
            try:
                rho_target, p_target = stats.pearsonr(res_XY[target_mask], res_WZ[target_mask])
            except:
                rho_target = 0.0
                p_target = 1.0
        else:
            rho_target = 0.0
            p_target = 1.0
        
        # Correlation OUTSIDE target range
        if np.sum(outside_mask) >= 3:
            try:
                rho_outside, p_outside = stats.pearsonr(res_XY[outside_mask], res_WZ[outside_mask])
            except:
                rho_outside = 0.0
                p_outside = 1.0
        else:
            rho_outside = 0.0
            p_outside = 1.0
        
        # Enhancement: Is correlation stronger in target range?
        abs_rho_target = np.abs(rho_target)
        abs_rho_outside = np.abs(rho_outside)
        enhancement = (abs_rho_target / abs_rho_outside) if abs_rho_outside > 1e-10 else np.nan
        
        # ML prediction check
        ml_prediction_met = abs_rho_target > 0.3  # ML predicted ρ > 0.3 at this range
        
        logger.info(f"ML-targeted correlation test (ℓ={ell_min:.0f}-{ell_max:.0f}):")
        logger.info(f"  ρ in target range: {rho_target:.4f} (p={p_target:.4f})")
        logger.info(f"  ρ outside range: {rho_outside:.4f} (p={p_outside:.4f})")
        logger.info(f"  Enhancement: {enhancement:.2f}×" if np.isfinite(enhancement) else "  Enhancement: N/A")
        logger.info(f"  ML prediction (ρ > 0.3): {'✓ MET' if ml_prediction_met else '✗ NOT MET'}")
        
        return {
            'ell_range': [float(ell_min), float(ell_max)],
            'rho_in_range': float(rho_target),
            'p_in_range': float(p_target),
            'rho_outside_range': float(rho_outside),
            'p_outside_range': float(p_outside),
            'enhancement_ratio': float(enhancement) if np.isfinite(enhancement) else None,
            'n_points_in_range': int(np.sum(target_mask)),
            'n_points_outside': int(np.sum(outside_mask)),
            'ml_prediction_met': bool(ml_prediction_met),
            'ml_predicted_rho': 0.5,  # ML predicted ρ ~ 0.5 ± 0.2
            'ml_predicted_threshold': 0.3,
        }
    
    def _empty_acoustic_results(self) -> Dict[str, Any]:
        """Return empty results structure for acoustic peak analysis."""
        return {
            'acoustic_peaks': [],
            'correlations_at_individual_peaks': [],
            'p_values_at_individual_peaks': [],
            'rho_peak_global': 0.0,
            'p_peak_global': 1.0,
            'rho_offpeak_global': 0.0,
            'p_offpeak_global': 1.0,
            'n_peaks_identified': 0,
            'n_peaks_significant': 0,
            'enhancement_ratio': None,
            'n_points_at_peaks': 0,
            'n_points_offpeak': 0,
        }
    
    def _empty_ml_targeted_results(self) -> Dict[str, Any]:
        """Return empty results structure for ML-targeted test."""
        return {
            'ell_range': [800.0, 1200.0],
            'rho_in_range': 0.0,
            'p_in_range': 1.0,
            'rho_outside_range': 0.0,
            'p_outside_range': 1.0,
            'enhancement_ratio': None,
            'n_points_in_range': 0,
            'n_points_outside': 0,
            'ml_prediction_met': False,
            'ml_predicted_rho': 0.5,
            'ml_predicted_threshold': 0.3,
        }
    
    def compute_cross_spectrum(
        self,
        residuals_XY: np.ndarray,
        residuals_WZ: np.ndarray,
        ell: np.ndarray
    ) -> np.ndarray:
        """
        Compute cross-spectrum: C_ℓ^{ΔXY × ΔWZ} = <ΔC_ℓ^XY · ΔC_ℓ^WZ>
        
        Parameters:
            residuals_XY: Residuals for spectrum XY (e.g., TT)
            residuals_WZ: Residuals for spectrum WZ (e.g., TE)
            ell: Multipole array
            
        Returns:
            np.ndarray: Cross-spectrum values
        """
        return residuals_XY * residuals_WZ
    
    def compute_correlation(
        self,
        residuals_XY: np.ndarray,
        residuals_WZ: np.ndarray,
        ell: np.ndarray,
        ell_bins: Optional[np.ndarray] = None
    ) -> Dict[str, np.ndarray]:
        """
        Compute correlation coefficient: ρ_XY,WZ(ℓ) = Cov(ΔC_ℓ^XY, ΔC_ℓ^WZ) / (σ_XY · σ_WZ)
        
        Parameters:
            residuals_XY: Residuals for spectrum XY
            residuals_WZ: Residuals for spectrum WZ
            ell: Multipole array
            ell_bins: Optional binning for multipoles (if None, uses sliding window)
            
        Returns:
            dict with keys:
                - ell: Multipole centers
                - correlation: Correlation coefficient ρ
                - p_value: Statistical significance
                - cross_spectrum: Cross-spectrum values
        """
        # Remove NaN values
        valid_mask = np.isfinite(residuals_XY) & np.isfinite(residuals_WZ)
        if not valid_mask.any():
            return {
                'ell': ell,
                'correlation': np.full_like(ell, np.nan),
                'p_value': np.full_like(ell, np.nan),
                'cross_spectrum': np.full_like(ell, np.nan),
            }
        
        ell_valid = ell[valid_mask]
        res_XY_valid = residuals_XY[valid_mask]
        res_WZ_valid = residuals_WZ[valid_mask]
        
        # Compute cross-spectrum
        cross_spectrum = res_XY_valid * res_WZ_valid
        
        # Compute correlation
        if len(res_XY_valid) < 3:
            correlation = np.full_like(ell_valid, np.nan)
            p_value = np.full_like(ell_valid, np.nan)
        else:
            correlation_val, p_val = stats.pearsonr(res_XY_valid, res_WZ_valid)
            # For binned analysis, compute per-bin correlations
            if ell_bins is not None:
                correlation = np.full(len(ell_bins) - 1, correlation_val)
                p_value = np.full(len(ell_bins) - 1, p_val)
            else:
                correlation = np.full_like(ell_valid, correlation_val)
                p_value = np.full_like(ell_valid, p_val)
        
        return {
            'ell': ell_valid if ell_bins is None else (ell_bins[:-1] + ell_bins[1:]) / 2,
            'correlation': correlation,
            'p_value': p_value,
            'cross_spectrum': cross_spectrum,
        }
    
    def compute_binned_correlation(
        self,
        residuals_XY: np.ndarray,
        residuals_WZ: np.ndarray,
        ell: np.ndarray,
        bin_width: float = 50.0
    ) -> Dict[str, np.ndarray]:
        """
        Compute correlation in bins for better signal-to-noise.
        
        Parameters:
            residuals_XY: Residuals for spectrum XY
            residuals_WZ: Residuals for spectrum WZ
            ell: Multipole array
            bin_width: Width of multipole bins
            
        Returns:
            dict: Binned correlation results
        """
        # Create bins
        ell_min = np.min(ell)
        ell_max = np.max(ell)
        ell_bins = np.arange(ell_min, ell_max + bin_width, bin_width)
        
        # Remove NaN values
        valid_mask = np.isfinite(residuals_XY) & np.isfinite(residuals_WZ)
        ell_valid = ell[valid_mask]
        res_XY_valid = residuals_XY[valid_mask]
        res_WZ_valid = residuals_WZ[valid_mask]
        
        # Bin data
        bin_indices = np.digitize(ell_valid, ell_bins) - 1
        bin_indices = np.clip(bin_indices, 0, len(ell_bins) - 2)
        
        # Compute correlation per bin
        bin_centers = []
        bin_correlations = []
        bin_p_values = []
        bin_counts = []
        
        for i in range(len(ell_bins) - 1):
            bin_mask = (bin_indices == i)
            if bin_mask.sum() < 3:  # Need at least 3 points
                continue
            
            ell_bin = ell_valid[bin_mask]
            res_XY_bin = res_XY_valid[bin_mask]
            res_WZ_bin = res_WZ_valid[bin_mask]
            
            if len(res_XY_bin) < 3:
                continue
            
            corr, p_val = stats.pearsonr(res_XY_bin, res_WZ_bin)
            
            bin_centers.append(np.mean(ell_bin))
            bin_correlations.append(corr)
            bin_p_values.append(p_val)
            bin_counts.append(len(ell_bin))
        
        return {
            'ell': np.array(bin_centers),
            'correlation': np.array(bin_correlations),
            'p_value': np.array(bin_p_values),
            'counts': np.array(bin_counts),
        }
    
    def test_peak_significance(
        self,
        correlation_results: Dict[str, np.ndarray],
        peak_range: Tuple[float, float] = None
    ) -> Dict[str, float]:
        """
        Test if correlation peak is significant (LEGACY METHOD).
        
        Use compute_acoustic_peak_correlations() for actual acoustic peak analysis.
        
        Parameters:
            correlation_results: Results from compute_correlation
            peak_range: Expected peak range (optional)
            
        Returns:
            dict with:
                - peak_ell: Multipole of maximum correlation
                - peak_correlation: Maximum correlation value
                - peak_p_value: Significance of peak
                - in_expected_range: False (deprecated)
                - mean_correlation: Mean correlation outside peak range
        """
        if peak_range is None:
            peak_range = (800, 1200)  # Default legacy range
        
        ell = correlation_results['ell']
        correlation = correlation_results['correlation']
        p_value = correlation_results['p_value']
        
        # Find peak
        valid_mask = np.isfinite(correlation)
        if not valid_mask.any():
            return {
                'peak_ell': np.nan,
                'peak_correlation': np.nan,
                'peak_p_value': np.nan,
                'in_expected_range': False,
                'mean_correlation': np.nan,
            }
        
        peak_idx = np.nanargmax(correlation[valid_mask])
        peak_ell = ell[valid_mask][peak_idx]
        peak_correlation = correlation[valid_mask][peak_idx]
        peak_p_value = p_value[valid_mask][peak_idx] if 'p_value' in correlation_results else np.nan
        
        # Check if in expected range
        in_expected_range = (peak_range[0] <= peak_ell <= peak_range[1])
        
        # Mean correlation outside peak range
        outside_mask = valid_mask & ((ell < peak_range[0]) | (ell > peak_range[1]))
        mean_correlation = np.nanmean(correlation[outside_mask]) if outside_mask.any() else np.nan
        
        return {
            'peak_ell': peak_ell,
            'peak_correlation': peak_correlation,
            'peak_p_value': peak_p_value,
            'in_expected_range': in_expected_range,
            'mean_correlation': mean_correlation,
        }
    
    def run_coherence_test(
        self,
        residuals: Dict[str, Dict[str, Dict[str, np.ndarray]]]
    ) -> Dict[str, Any]:
        """
        Run full cross-modal coherence test for all spectrum pairs.
        
        Parameters:
            residuals: Nested dict from ResidualAnalyzer.compute_all_residuals()
                Structure: residuals[survey][spectrum] = {'ell': ..., 'residual': ...}
        
        Returns:
            dict with keys:
                - tt_te: TT×TE correlation results
                - tt_ee: TT×EE correlation results
                - te_ee: TE×EE correlation results
                - summary: Summary statistics
        """
        results = {}
        
        # For each survey, compute cross-correlations
        for survey_name, survey_residuals in residuals.items():
            if 'TT' not in survey_residuals or 'TE' not in survey_residuals or 'EE' not in survey_residuals:
                logger.warning(f"Incomplete data for {survey_name}, skipping")
                continue
            
            tt_data = survey_residuals['TT']
            te_data = survey_residuals['TE']
            ee_data = survey_residuals['EE']
            
            if tt_data is None or te_data is None or ee_data is None:
                continue
            
            # Interpolate to common multipole grid
            ell_common = np.unique(np.concatenate([
                tt_data['ell'],
                te_data['ell'],
                ee_data['ell']
            ]))
            ell_common = np.sort(ell_common)
            
            # Interpolate residuals to common grid
            from scipy.interpolate import interp1d
            
            interp_tt = interp1d(tt_data['ell'], tt_data['residual'], 
                                kind='linear', bounds_error=False, fill_value=np.nan)
            interp_te = interp1d(te_data['ell'], te_data['residual'],
                                kind='linear', bounds_error=False, fill_value=np.nan)
            interp_ee = interp1d(ee_data['ell'], ee_data['residual'],
                                kind='linear', bounds_error=False, fill_value=np.nan)
            
            res_tt = interp_tt(ell_common)
            res_te = interp_te(ell_common)
            res_ee = interp_ee(ell_common)
            
            # Interpolate theory spectrum for peak identification
            interp_cl_theory = interp1d(tt_data['ell'], tt_data['cl_theory'],
                                       kind='linear', bounds_error=False, fill_value=np.nan)
            cl_theory_common = interp_cl_theory(ell_common)
            
            # NEW: Test correlations at ACTUAL acoustic peak positions
            tt_te_acoustic = self.compute_acoustic_peak_correlations(res_tt, res_te, ell_common, cl_theory_common)
            tt_ee_acoustic = self.compute_acoustic_peak_correlations(res_tt, res_ee, ell_common, cl_theory_common)
            te_ee_acoustic = self.compute_acoustic_peak_correlations(res_te, res_ee, ell_common, cl_theory_common)
            
            # NEW: ML-targeted test at specific ℓ range where anomalies were detected
            tt_te_ml = self.compute_ml_targeted_correlation(res_tt, res_te, ell_common, ell_min=800, ell_max=1200)
            tt_ee_ml = self.compute_ml_targeted_correlation(res_tt, res_ee, ell_common, ell_min=800, ell_max=1200)
            te_ee_ml = self.compute_ml_targeted_correlation(res_te, res_ee, ell_common, ell_min=800, ell_max=1200)
            
            # LEGACY: Also compute binned correlations for compatibility
            tt_te = self.compute_binned_correlation(res_tt, res_te, ell_common)
            tt_ee = self.compute_binned_correlation(res_tt, res_ee, ell_common)
            te_ee = self.compute_binned_correlation(res_te, res_ee, ell_common)
            
            # Test peak significance (legacy)
            tt_te_peak = self.test_peak_significance(tt_te)
            tt_ee_peak = self.test_peak_significance(tt_ee)
            te_ee_peak = self.test_peak_significance(te_ee)
            
            results[survey_name] = {
                'tt_te': {
                    'correlation': tt_te,
                    'peak': tt_te_peak,
                    'acoustic_peaks': tt_te_acoustic,  # NEW: Actual peak positions
                    'ml_targeted': tt_te_ml,  # NEW: ML-recommended ℓ range test
                },
                'tt_ee': {
                    'correlation': tt_ee,
                    'peak': tt_ee_peak,
                    'acoustic_peaks': tt_ee_acoustic,
                    'ml_targeted': tt_ee_ml,
                },
                'te_ee': {
                    'correlation': te_ee,
                    'peak': te_ee_peak,
                    'acoustic_peaks': te_ee_acoustic,
                    'ml_targeted': te_ee_ml,
                },
            }
        
        # Compute summary across surveys
        summary = self._compute_summary(results)
        
        return {
            'by_survey': results,
            'summary': summary,
        }
    
    def _compute_summary(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Compute summary statistics across all surveys."""
        all_peaks = []
        all_correlations = []
        all_acoustic_results = []
        
        all_ml_targeted_results = []
        
        for survey_name, survey_results in results.items():
            for pair_name, pair_results in survey_results.items():
                # Legacy peak data
                peak = pair_results['peak']
                if not np.isnan(peak['peak_correlation']):
                    all_peaks.append(peak)
                    corr_data = pair_results['correlation']
                    if 'correlation' in corr_data:
                        valid_corr = corr_data['correlation'][np.isfinite(corr_data['correlation'])]
                        all_correlations.extend(valid_corr.tolist())
                
                # NEW: Acoustic peak data
                if 'acoustic_peaks' in pair_results:
                    acoustic = pair_results['acoustic_peaks']
                    if acoustic['n_peaks_identified'] > 0:
                        all_acoustic_results.append(acoustic)
                
                # NEW: ML-targeted data
                if 'ml_targeted' in pair_results:
                    ml_targeted = pair_results['ml_targeted']
                    if ml_targeted['n_points_in_range'] > 0:
                        all_ml_targeted_results.append(ml_targeted)
        
        # Legacy summary
        if all_peaks:
            peak_correlations = [p['peak_correlation'] for p in all_peaks]
            peaks_in_range = [p['in_expected_range'] for p in all_peaks]
            significant_peaks = [p['peak_p_value'] < 0.05 for p in all_peaks if not np.isnan(p['peak_p_value'])]
            legacy_summary = {
                'mean_peak_correlation': np.nanmean(peak_correlations),
                'mean_correlation': np.nanmean(all_correlations) if all_correlations else np.nan,
                'n_significant_peaks': sum(significant_peaks),
                'n_peaks_in_range': sum(peaks_in_range),
                'total_pairs_tested': len(all_peaks),
            }
        else:
            legacy_summary = {
                'mean_peak_correlation': np.nan,
                'mean_correlation': np.nan,
                'n_significant_peaks': 0,
                'n_peaks_in_range': 0,
                'total_pairs_tested': 0,
            }
        
        # NEW: Acoustic peak summary
        if all_acoustic_results:
            total_peaks_identified = sum(a['n_peaks_identified'] for a in all_acoustic_results)
            total_significant = sum(a['n_peaks_significant'] for a in all_acoustic_results)
            
            # Collect global correlations from each spectrum pair
            rho_peaks = [a['rho_peak_global'] for a in all_acoustic_results if a['n_peaks_identified'] > 0]
            rho_offpeaks = [a['rho_offpeak_global'] for a in all_acoustic_results if a['n_peaks_identified'] > 0]
            enhancements = [a['enhancement_ratio'] for a in all_acoustic_results 
                           if a['enhancement_ratio'] is not None and np.isfinite(a['enhancement_ratio'])]
            
            # Summary statistics
            mean_rho_peak = np.mean(np.abs(rho_peaks)) if rho_peaks else 0.0
            mean_rho_offpeak = np.mean(np.abs(rho_offpeaks)) if rho_offpeaks else 0.0
            median_enhancement = np.median(enhancements) if enhancements else None
            
            acoustic_summary = {
                'n_acoustic_peaks_identified': total_peaks_identified,
                'n_acoustic_peaks_significant': total_significant,
                'fraction_significant': total_significant / total_peaks_identified if total_peaks_identified > 0 else 0.0,
                'mean_abs_rho_peak_global': float(mean_rho_peak),  # Mean of |ρ_peak| across pairs
                'mean_abs_rho_offpeak_global': float(mean_rho_offpeak),  # Mean of |ρ_offpeak| across pairs
                'median_enhancement_ratio': float(median_enhancement) if median_enhancement is not None else None,
                'enhancement_per_pair': enhancements,  # Individual enhancements for each pair
                'n_pairs_tested': len(rho_peaks),
            }
        else:
            acoustic_summary = {
                'n_acoustic_peaks_identified': 0,
                'n_acoustic_peaks_significant': 0,
                'fraction_significant': 0.0,
                'mean_abs_rho_peak_global': 0.0,
                'mean_abs_rho_offpeak_global': 0.0,
                'median_enhancement_ratio': None,
                'enhancement_per_pair': [],
                'n_pairs_tested': 0,
            }
        
        # NEW: ML-targeted summary
        if all_ml_targeted_results:
            rhos_in_range = [ml['rho_in_range'] for ml in all_ml_targeted_results]
            rhos_outside = [ml['rho_outside_range'] for ml in all_ml_targeted_results]
            ml_predictions_met = [ml['ml_prediction_met'] for ml in all_ml_targeted_results]
            enhancements_ml = [ml['enhancement_ratio'] for ml in all_ml_targeted_results 
                              if ml['enhancement_ratio'] is not None and np.isfinite(ml['enhancement_ratio'])]
            
            mean_rho_in_range = np.mean(np.abs(rhos_in_range)) if rhos_in_range else 0.0
            mean_rho_outside = np.mean(np.abs(rhos_outside)) if rhos_outside else 0.0
            n_predictions_met = sum(ml_predictions_met)
            
            ml_targeted_summary = {
                'ell_range': [800.0, 1200.0],
                'mean_abs_rho_in_range': float(mean_rho_in_range),
                'mean_abs_rho_outside': float(mean_rho_outside),
                'median_enhancement': float(np.median(enhancements_ml)) if enhancements_ml else None,
                'n_pairs_tested': len(all_ml_targeted_results),
                'n_predictions_met': int(n_predictions_met),
                'fraction_predictions_met': float(n_predictions_met / len(all_ml_targeted_results)) if all_ml_targeted_results else 0.0,
                'ml_predicted_rho': 0.5,
                'ml_predicted_threshold': 0.3,
            }
        else:
            ml_targeted_summary = {
                'ell_range': [800.0, 1200.0],
                'mean_abs_rho_in_range': 0.0,
                'mean_abs_rho_outside': 0.0,
                'median_enhancement': None,
                'n_pairs_tested': 0,
                'n_predictions_met': 0,
                'fraction_predictions_met': 0.0,
                'ml_predicted_rho': 0.5,
                'ml_predicted_threshold': 0.3,
            }
        
        # Combine legacy and new summaries
        return {
            **legacy_summary,
            'acoustic_peaks': acoustic_summary,
            'ml_targeted': ml_targeted_summary,
        }

