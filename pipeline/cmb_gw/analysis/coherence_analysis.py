"""
Cross-Modal Coherence Analysis
================================

TEST 5: Compute correlation of ΛCDM residuals at acoustic harmonic positions.

This verifies that deviations from ΛCDM in TT, TE, and EE spectra are correlated
at acoustic peak positions, as expected if a single physical mechanism (evolving G)
is responsible.

CRITICAL: Peak positions are found FROM THE DATA, not hardcoded.
"""

import numpy as np
import pandas as pd
import logging
from scipy.signal import find_peaks
from scipy.stats import median_abs_deviation
from typing import Dict, Any, Optional, Tuple, List
from data.loader import DataLoader
from ..physics.cmb_spectra import (
    compute_cmb_residuals as compute_camb_residuals,
    compute_lcdm_cmb_spectrum,
    CAMB_AVAILABLE
)

logger = logging.getLogger(__name__)


def find_coherence_scale(
    residual_TT: np.ndarray,
    residual_TE: np.ndarray,
    residual_EE: np.ndarray,
    ell: np.ndarray
) -> Optional[Dict[str, Any]]:
    """
    Scan across ℓ to find where cross-modal correlations peak.
    
    This empirically identifies the acoustic scale by finding where TT/TE/EE
    residuals are most correlated, revealing harmonic structure if present.
    
    Parameters:
    -----------
    residual_XX : array
        (C_ℓ^obs - C_ℓ^ΛCDM) for each spectrum
    ell : array
        Multipole values
        
    Returns:
    --------
    dict or None
        Results containing:
        - 'peak_ells': List of ℓ positions where correlations peak
        - 'mean_spacing': Mean spacing between peaks (should be ~270-280 for acoustic scale)
        - 'correlation_scan': Full scan results as list of dicts
        - 'harmonic_structure_detected': Boolean indicating if ≥3 peaks found
    """
    if len(residual_TT) == 0 or len(residual_TE) == 0 or len(residual_EE) == 0:
        return None
    
    correlations = []
    # Scan from ℓ=100 to ℓ=2000 in steps of 20
    test_windows = np.arange(100, min(2000, ell.max()), 20)
    window_size = 50  # Half-width of window
    
    for ell_center in test_windows:
        mask = np.abs(ell - ell_center) < window_size
        if np.sum(mask) < 10:
            continue
        
        # Cross-correlations at this scale
        try:
            rho_TT_TE = np.corrcoef(residual_TT[mask], residual_TE[mask])[0, 1]
            rho_TT_EE = np.corrcoef(residual_TT[mask], residual_EE[mask])[0, 1]
            rho_TE_EE = np.corrcoef(residual_TE[mask], residual_EE[mask])[0, 1]
            
            # Handle NaN correlations
            if np.isnan(rho_TT_TE):
                rho_TT_TE = 0.0
            if np.isnan(rho_TT_EE):
                rho_TT_EE = 0.0
            if np.isnan(rho_TE_EE):
                rho_TE_EE = 0.0
            
            mean_rho = np.nanmean([rho_TT_TE, rho_TT_EE, rho_TE_EE])
            correlations.append({
                'ell': float(ell_center),
                'mean_rho': float(mean_rho),
                'rho_TT_TE': float(rho_TT_TE),
                'rho_TT_EE': float(rho_TT_EE),
                'rho_TE_EE': float(rho_TE_EE)
            })
        except Exception:
            continue
    
    if not correlations:
        return None
    
    # Find peaks in correlation function
    df = pd.DataFrame(correlations)
    threshold = df['mean_rho'].quantile(0.9)
    peak_ells = df.loc[df['mean_rho'] > threshold, 'ell'].values
    
    # Measure spacing between peaks (should be ~270-280 for acoustic scale)
    mean_spacing = np.nan
    if len(peak_ells) >= 2:
        spacings = np.diff(sorted(peak_ells))
        # Filter out noise (spacings < 100 are likely noise)
        valid_spacings = spacings[spacings > 100]
        if len(valid_spacings) > 0:
            mean_spacing = float(np.mean(valid_spacings))
    
    return {
        'peak_ells': peak_ells.tolist(),
        'mean_spacing': mean_spacing,
        'correlation_scan': correlations,
        'harmonic_structure_detected': len(peak_ells) >= 3,
        'threshold_used': float(threshold)
    }


def find_acoustic_peaks_from_data(Cl_spectrum: np.ndarray, ell: np.ndarray, 
                                   max_peaks: int = 10) -> List[float]:
    """
    Find acoustic peak positions directly from data.
    
    NO HARDCODED PEAK POSITIONS. Uses data-driven peak detection.
    
    Parameters:
    -----------
    Cl_spectrum : array
        Power spectrum (typically TT for clearest peaks)
    ell : array
        Multipole values
    max_peaks : int
        Maximum number of peaks to return
        
    Returns:
    --------
    list
        Peak positions in ell found from data
    """
    if len(Cl_spectrum) < 10 or len(ell) < 10:
        return []
    
    # Remove non-finite values
    valid = np.isfinite(Cl_spectrum) & np.isfinite(ell)
    if np.sum(valid) < 10:
        return []
    
    Cl_clean = Cl_spectrum[valid]
    ell_clean = ell[valid]
    
    # Sort by ell
    sort_idx = np.argsort(ell_clean)
    ell_sorted = ell_clean[sort_idx]
    Cl_sorted = Cl_clean[sort_idx]
    
    # Data-driven prominence: use MAD-based threshold
    mad_val = median_abs_deviation(Cl_sorted, nan_policy='omit')
    if mad_val == 0 or not np.isfinite(mad_val):
        mad_val = np.std(Cl_sorted)
    
    prominence = 2.0 * mad_val
    
    # Data-driven distance: estimate from autocorrelation
    if len(Cl_sorted) > 50:
        Cl_centered = Cl_sorted - np.mean(Cl_sorted)
        autocorr = np.correlate(Cl_centered, Cl_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        if autocorr[0] != 0:
            autocorr = autocorr / autocorr[0]
        
        # Find first minimum to estimate half-period
        for i in range(5, len(autocorr) - 1):
            if autocorr[i] < autocorr[i-1] and autocorr[i] < autocorr[i+1]:
                half_spacing = i
                break
        else:
            half_spacing = len(Cl_sorted) // 10
        
        distance = max(1, int(half_spacing * 0.5))
    else:
        distance = max(1, len(Cl_sorted) // 10)
    
    # Find peaks
    peaks, _ = find_peaks(Cl_sorted, prominence=prominence, distance=distance)
    
    # If not enough peaks, relax constraints
    if len(peaks) < 3:
        peaks, _ = find_peaks(Cl_sorted, prominence=prominence/2, distance=distance//2)
    
    if len(peaks) == 0:
        return []
    
    # Return peak positions in ell, sorted by ell
    peak_ells = ell_sorted[peaks]
    return sorted(peak_ells.tolist())[:max_peaks]


def cross_modal_coherence_at_harmonics(
    residual_TT: np.ndarray,
    residual_TE: np.ndarray,
    residual_EE: np.ndarray,
    ell: np.ndarray,
    acoustic_ells: Optional[List[float]] = None
) -> Dict[str, Any]:
    """
    Compute correlation of ΛCDM residuals at acoustic harmonic positions.
    
    This implements TEST 5 from docs/cmb_gw.md.
    
    CRITICAL: Uses empirical coherence scale finding to identify acoustic structure.
    If acoustic_ells is not provided, they are FOUND FROM THE DATA by scanning
    for correlation peaks across ℓ.
    
    Parameters:
    -----------
    residual_XX : array
        (C_ℓ^obs - C_ℓ^ΛCDM) for each spectrum
    ell : array
        Multipole values
    acoustic_ells : list, optional
        Peak positions. If None, found empirically from coherence scan.
        
    Returns:
    --------
    dict
        Coherence analysis results containing:
        - 'peak_correlations': List of correlations at each harmonic
        - 'mean_rho_at_peaks': Mean correlation at acoustic positions
        - 'mean_rho_off_peaks': Mean correlation away from peaks
        - 'enhancement_ratio': Ratio of peak to off-peak correlation
        - 'n_harmonics_tested': Number of harmonics analyzed
        - 'acoustic_ells_used': Peak positions used (data-derived or provided)
        - 'coherence_scale_results': Results from empirical coherence scan
        - 'interpretation': Interpretation of results
    """
    # DATA-DRIVEN: Find coherence scale empirically by scanning for correlation peaks
    coherence_results = find_coherence_scale(residual_TT, residual_TE, residual_EE, ell)
    
    # Use empirically-found peaks if not provided
    if acoustic_ells is None or len(acoustic_ells) == 0:
        if coherence_results and len(coherence_results.get('peak_ells', [])) > 0:
            acoustic_ells = coherence_results['peak_ells']
        else:
            # Fallback: use TT spectrum to find peaks
            acoustic_ells = find_acoustic_peaks_from_data(np.abs(residual_TT), ell)
    
    if len(acoustic_ells) == 0:
        return {
            'peak_correlations': [],
            'mean_rho_at_peaks': np.nan,
            'mean_rho_off_peaks': np.nan,
            'enhancement_ratio': np.nan,
            'n_harmonics_tested': 0,
            'acoustic_ells_used': []
        }
    
    # DATA-DRIVEN window width: scale with typical peak spacing
    # BUT ensure window is large enough to capture sufficient points for correlation
    if len(acoustic_ells) >= 2:
        spacings = np.diff(sorted(acoustic_ells))
        typical_spacing = np.median(spacings)
        # Start with 1/4 of typical spacing, but ensure minimum width
        window = max(typical_spacing / 4, 15.0)  # At least ±15 multipoles
    else:
        # Fallback: use fraction of ell range
        window = max((ell.max() - ell.min()) / 20, 15.0)
    
    # Minimum points required for correlation
    # Reduced from 1% to be more permissive with discrete ell sampling
    min_points = max(5, len(ell) // 200)  # At least 0.5% of data or 5 points
    
    # At-peak correlations
    peak_correlations = []
    for ell_peak in acoustic_ells:
        mask = np.abs(ell - ell_peak) < window
        if np.sum(mask) < min_points:
            continue
        
        # Compute pairwise correlations
        rho_TT_TE = np.corrcoef(residual_TT[mask], residual_TE[mask])[0, 1]
        rho_TT_EE = np.corrcoef(residual_TT[mask], residual_EE[mask])[0, 1]
        rho_TE_EE = np.corrcoef(residual_TE[mask], residual_EE[mask])[0, 1]
        
        # Handle NaN correlations (can occur if residuals are constant)
        if np.isnan(rho_TT_TE):
            rho_TT_TE = 0.0
        if np.isnan(rho_TT_EE):
            rho_TT_EE = 0.0
        if np.isnan(rho_TE_EE):
            rho_TE_EE = 0.0
        
        peak_correlations.append({
            'ell': ell_peak,
            'rho_TT_TE': rho_TT_TE,
            'rho_TT_EE': rho_TT_EE,
            'rho_TE_EE': rho_TE_EE,
            'mean_rho': np.mean([rho_TT_TE, rho_TT_EE, rho_TE_EE])
        })
    
    if not peak_correlations:
        return {
            'peak_correlations': [],
            'mean_rho_at_peaks': np.nan,
            'mean_rho_off_peaks': np.nan,
            'enhancement_ratio': np.nan,
            'n_harmonics_tested': 0,
            'acoustic_ells_used': acoustic_ells,
            'window_used': window
        }
    
    # Off-peak correlations (between harmonics)
    off_peak_mask = np.ones(len(ell), dtype=bool)
    for ell_peak in acoustic_ells:
        off_peak_mask &= np.abs(ell - ell_peak) > window
    
    if np.sum(off_peak_mask) < min_points:
        # If not enough off-peak points, use all points not in peak windows
        off_peak_mask = np.ones(len(ell), dtype=bool)
        for ell_peak in acoustic_ells:
            off_peak_mask &= np.abs(ell - ell_peak) > (window / 2)
    
    if np.sum(off_peak_mask) >= min_points:
        rho_TT_TE_off = np.corrcoef(residual_TT[off_peak_mask], residual_TE[off_peak_mask])[0, 1]
        rho_TT_EE_off = np.corrcoef(residual_TT[off_peak_mask], residual_EE[off_peak_mask])[0, 1]
        rho_TE_EE_off = np.corrcoef(residual_TE[off_peak_mask], residual_EE[off_peak_mask])[0, 1]
        
        # Handle NaN
        if np.isnan(rho_TT_TE_off):
            rho_TT_TE_off = 0.0
        if np.isnan(rho_TT_EE_off):
            rho_TT_EE_off = 0.0
        if np.isnan(rho_TE_EE_off):
            rho_TE_EE_off = 0.0
        
        mean_rho_off = np.mean([rho_TT_TE_off, rho_TT_EE_off, rho_TE_EE_off])
    else:
        mean_rho_off = 0.0
    
    mean_rho_peak = np.mean([p['mean_rho'] for p in peak_correlations])
    
    enhancement_ratio = mean_rho_peak / mean_rho_off if mean_rho_off != 0 else np.inf
    
    # Interpretation logic based on coherence results
    interpretation = "unknown"
    if coherence_results:
        mean_spacing = coherence_results.get('mean_spacing', np.nan)
        harmonic_detected = coherence_results.get('harmonic_structure_detected', False)
        peak_ells_found = coherence_results.get('peak_ells', [])
        
        if harmonic_detected and np.isfinite(mean_spacing):
            # Check if spacing matches acoustic scale (~270-280)
            if 250 < mean_spacing < 300:
                # Standard acoustic physics (ΛCDM or H-ΛCDM)
                interpretation = "standard_acoustic_physics"
            elif mean_spacing < 250 or mean_spacing > 300:
                # Modified acoustic scale - measure the shift
                # Δℓ_n/ℓ_n ≈ -Δr_s/r_s
                # For β ~ 0.2, r_s increases by ~2.5%, so ℓ shifts by ~-2.5%
                interpretation = "modified_acoustic_scale"
            else:
                interpretation = "acoustic_structure_detected"
        elif len(peak_ells_found) > 0:
            # Some peaks found but not harmonic structure
            interpretation = "non_harmonic_structure"
        else:
            # No peaks found
            interpretation = "no_coherence_peaks"
    else:
        # Coherence scan failed
        interpretation = "coherence_scan_failed"
    
    return {
        'peak_correlations': peak_correlations,
        'mean_rho_at_peaks': mean_rho_peak,
        'mean_rho_off_peaks': mean_rho_off,
        'enhancement_ratio': enhancement_ratio,
        'n_harmonics_tested': len(peak_correlations),
        'acoustic_ells_used': acoustic_ells,
        'window_used': float(window),
        'coherence_scale_results': coherence_results,
        'interpretation': interpretation
    }


def compute_cmb_residuals(
    dataset: str = 'planck_2018'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ΛCDM residuals for TT, TE, EE spectra.
    
    SCIENTIFICALLY RIGOROUS: Uses CAMB (full Boltzmann solver) to compute
    proper C_ℓ^ΛCDM predictions. Residuals = (C_ℓ^obs - C_ℓ^CAMB).
    
    Falls back to smooth fit only if CAMB unavailable (with warning).
    
    Parameters:
    -----------
    dataset : str
        Dataset name ('planck_2018', 'act_dr6', 'spt3g')
        
    Returns:
    --------
    tuple
        (residual_TT, residual_TE, residual_EE, ell) arrays
    """
    data_loader = DataLoader()
    
    try:
        if dataset == 'planck_2018':
            cmb_data = data_loader.load_planck_2018()
        elif dataset == 'act_dr6':
            cmb_data = data_loader.load_act_dr6()
        elif dataset == 'spt3g':
            cmb_data = data_loader.load_spt3g()
        else:
            cmb_data = data_loader.load_planck_2018()
    except Exception:
        # Return empty arrays if data loading fails
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    residuals = {}
    ell_arrays = {}
    
    # Try CAMB for PROPER residuals
    if CAMB_AVAILABLE:
        try:
            # Get CAMB ΛCDM spectrum
            camb_spectra = compute_lcdm_cmb_spectrum(
                H0=67.36,
                omega_b_h2=0.02237,
                omega_c_h2=0.1200,
                unit='cl'
            )
            logger.info("Using CAMB for proper C_ℓ^ΛCDM residuals (full Boltzmann solver)")
            use_camb = True
        except Exception as e:
            logger.warning(f"CAMB failed: {e}. Using smooth fit approximation.")
            use_camb = False
    else:
        logger.warning("CAMB not available. Using smooth fit approximation.")
        use_camb = False
    
    for spectrum in ['TT', 'TE', 'EE']:
        if spectrum in cmb_data:
            ell, Cl_obs, Cl_err = cmb_data[spectrum]
            
            if use_camb:
                # PROPER METHOD: CAMB ΛCDM prediction
                ell_lcdm, Cl_lcdm = camb_spectra[spectrum]
                Cl_model = np.interp(ell, ell_lcdm, Cl_lcdm)
            else:
                # FALLBACK: Smooth fit (NOT RIGOROUS)
                from scipy.interpolate import UnivariateSpline
                spline = UnivariateSpline(ell, Cl_obs, s=len(ell)*0.1)
                Cl_model = spline(ell)
            
            # Residuals
            residual = Cl_obs - Cl_model
            residuals[spectrum] = residual
            ell_arrays[spectrum] = ell
    
    # Find common ell range across all spectra
    # Use intersection of all ell arrays to ensure consistent indexing
    if not ell_arrays:
        return np.array([]), np.array([]), np.array([]), np.array([])
    
    # Find common ell values
    common_ell = None
    for ell_arr in ell_arrays.values():
        if common_ell is None:
            common_ell = set(ell_arr)
        else:
            common_ell = common_ell.intersection(set(ell_arr))
    
    if not common_ell:
        # Fallback: use TT ell array
        ell_ref = ell_arrays.get('TT', np.array([]))
        residual_TT = residuals.get('TT', np.zeros_like(ell_ref))
        residual_TE = residuals.get('TE', np.zeros_like(ell_ref))
        residual_EE = residuals.get('EE', np.zeros_like(ell_ref))
        return residual_TT, residual_TE, residual_EE, ell_ref
    
    # Convert to sorted array
    ell_ref = np.array(sorted(common_ell))
    
    # Interpolate residuals to common ell grid
    residual_TT = np.zeros_like(ell_ref)
    residual_TE = np.zeros_like(ell_ref)
    residual_EE = np.zeros_like(ell_ref)
    
    if 'TT' in residuals:
        from scipy.interpolate import interp1d
        ell_tt = ell_arrays['TT']
        res_tt = residuals['TT']
        if len(ell_tt) > 0:
            interp_tt = interp1d(ell_tt, res_tt, kind='linear', bounds_error=False, fill_value=0.0)
            residual_TT = interp_tt(ell_ref)
    
    if 'TE' in residuals:
        ell_te = ell_arrays['TE']
        res_te = residuals['TE']
        if len(ell_te) > 0:
            interp_te = interp1d(ell_te, res_te, kind='linear', bounds_error=False, fill_value=0.0)
            residual_TE = interp_te(ell_ref)
    
    if 'EE' in residuals:
        ell_ee = ell_arrays['EE']
        res_ee = residuals['EE']
        if len(ell_ee) > 0:
            interp_ee = interp1d(ell_ee, res_ee, kind='linear', bounds_error=False, fill_value=0.0)
            residual_EE = interp_ee(ell_ref)
    
    return residual_TT, residual_TE, residual_EE, ell_ref

