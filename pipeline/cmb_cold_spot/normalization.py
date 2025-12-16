"""
CMB Survey Normalization
========================

Handle survey-specific systematics for cross-survey CMB analysis.

Key systematics:
- Unit conversions (K, mK, μK, thermodynamic vs antenna temperature)
- Monopole/dipole removal
- Beam smoothing differences
- Calibration uncertainties
- Foreground residuals
- Frequency-dependent corrections
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False

logger = logging.getLogger(__name__)


# Survey-specific calibration and systematic parameters
SURVEY_SYSTEMATICS = {
    'planck_2018': {
        'frequency_ghz': 143,  # Effective frequency for SMICA
        'beam_fwhm_arcmin': 7.0,  # Effective beam
        'calibration_uncertainty': 0.0025,  # 0.25% absolute calibration
        'units': 'K_CMB',  # Thermodynamic temperature
        'unit_conversion_to_uK': 1e6,
        'monopole_removed': True,
        'dipole_removed': True,
        'notes': 'SMICA component separation, minimal foreground residuals'
    },
    'wmap': {
        'frequency_ghz': 61,  # K-band for ILC
        'beam_fwhm_arcmin': 13.0,  # Effective beam for ILC
        'calibration_uncertainty': 0.005,  # 0.5% absolute calibration
        'units': 'mK_CMB',
        'unit_conversion_to_uK': 1e3,
        'monopole_removed': True,
        'dipole_removed': True,
        'notes': 'ILC combination, some foreground residuals at low-ℓ'
    },
    'cobe': {
        'frequency_ghz': 53,  # DMR 53 GHz channel
        'beam_fwhm_arcmin': 420.0,  # 7° FWHM
        'calibration_uncertainty': 0.01,  # 1% absolute calibration
        'units': 'mK_CMB',
        'unit_conversion_to_uK': 1e3,
        'monopole_removed': True,
        'dipole_removed': True,
        'notes': 'Low resolution, significant beam smoothing'
    },
    'act_dr6': {
        'frequency_ghz': 150,
        'beam_fwhm_arcmin': 1.4,  # 150 GHz beam
        'calibration_uncertainty': 0.01,  # 1% calibration
        'units': 'uK_CMB',
        'unit_conversion_to_uK': 1.0,
        'monopole_removed': True,
        'dipole_removed': False,  # Partial-sky, dipole not fully removed
        'notes': 'Partial-sky, equatorial coverage'
    },
    'spt3g': {
        'frequency_ghz': 150,
        'beam_fwhm_arcmin': 1.0,  # High-res beam
        'calibration_uncertainty': 0.01,  # 1% calibration
        'units': 'uK_CMB',
        'unit_conversion_to_uK': 1.0,
        'monopole_removed': True,
        'dipole_removed': False,  # Partial-sky
        'notes': 'Partial-sky, 500d field'
    }
}


def normalize_cmb_map(temperature_map: np.ndarray,
                     survey: str,
                     nside: Optional[int] = None,
                     remove_monopole: bool = True,
                     remove_dipole: bool = True) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Normalize CMB temperature map accounting for survey-specific systematics.
    
    Applies:
    1. Unit conversion to μK thermodynamic temperature
    2. Monopole removal (if not already removed)
    3. Dipole removal (if not already removed)
    4. Calibration uncertainty accounting
    
    Parameters:
        temperature_map: Input CMB temperature map
        survey: Survey name ('planck_2018', 'wmap', 'cobe', 'act_dr6', 'spt3g')
        nside: HEALPix nside (inferred if None)
        remove_monopole: Remove monopole if True
        remove_dipole: Remove dipole if True
        
    Returns:
        tuple: (normalized_map, metadata_dict)
    """
    if not HEALPY_AVAILABLE:
        logger.warning("healpy not available, normalization limited")
        return temperature_map, {}
    
    # Get survey parameters
    if survey not in SURVEY_SYSTEMATICS:
        logger.warning(f"Unknown survey {survey}, using default normalization")
        systematics = {
            'unit_conversion_to_uK': 1.0,
            'monopole_removed': False,
            'dipole_removed': False,
            'calibration_uncertainty': 0.02
        }
    else:
        systematics = SURVEY_SYSTEMATICS[survey]
    
    # Infer nside if not provided
    if nside is None:
        nside = hp.npix2nside(len(temperature_map))
    
    # Start with copy of input map
    normalized_map = temperature_map.copy()
    
    metadata = {
        'survey': survey,
        'input_rms_original_units': float(np.std(temperature_map)),
        'input_mean_original_units': float(np.mean(temperature_map)),
    }
    
    # Step 1: Unit conversion to μK
    unit_factor = systematics.get('unit_conversion_to_uK', 1.0)
    normalized_map = normalized_map * unit_factor
    
    metadata['unit_conversion_factor'] = unit_factor
    metadata['output_units'] = 'uK_CMB'
    
    # Step 2: Remove monopole if needed
    if remove_monopole and not systematics.get('monopole_removed', False):
        monopole = np.mean(normalized_map)
        normalized_map = normalized_map - monopole
        metadata['monopole_removed'] = monopole
        logger.info(f"{survey}: Removed monopole = {monopole:.2f} μK")
    else:
        metadata['monopole_removed'] = 0.0
    
    # Step 3: Remove dipole if needed
    if remove_dipole and not systematics.get('dipole_removed', False):
        # Fit and remove dipole using healpy
        try:
            # Get dipole amplitude and direction
            monopole, dipole = hp.fit_dipole(normalized_map)
            # Remove fitted monopole and dipole
            npix = len(normalized_map)
            # Infer ACTUAL nside from map size (don't trust parameter!)
            actual_nside = hp.npix2nside(npix)
            theta, phi = hp.pix2ang(actual_nside, np.arange(npix))
            
            # Dipole pattern
            dipole_amplitude = np.sqrt(np.sum(dipole**2))
            if dipole_amplitude > 0:
                # Direction vector
                dipole_dir = dipole / dipole_amplitude
                
                # Compute dipole map
                vec = hp.ang2vec(theta, phi)
                dipole_map = dipole_amplitude * np.dot(vec, dipole_dir)
                
                # Remove
                normalized_map = normalized_map - dipole_map
                
                metadata['dipole_amplitude_uK'] = float(dipole_amplitude)
                metadata['dipole_removed'] = True
                logger.info(f"{survey}: Removed dipole amplitude = {dipole_amplitude:.2f} μK")
            else:
                metadata['dipole_amplitude_uK'] = 0.0
                metadata['dipole_removed'] = False
        except Exception as e:
            logger.warning(f"Failed to remove dipole: {e}")
            metadata['dipole_removed'] = False
            metadata['dipole_amplitude_uK'] = 0.0
    else:
        metadata['dipole_removed'] = False
        metadata['dipole_amplitude_uK'] = 0.0
    
    # Step 4: Record final statistics
    metadata['output_rms_uK'] = float(np.std(normalized_map))
    metadata['output_mean_uK'] = float(np.mean(normalized_map))
    metadata['calibration_uncertainty'] = systematics.get('calibration_uncertainty', 0.02)
    metadata['beam_fwhm_arcmin'] = systematics.get('beam_fwhm_arcmin', 'unknown')
    metadata['frequency_ghz'] = systematics.get('frequency_ghz', 'unknown')
    
    logger.info(f"{survey}: Normalized map RMS = {metadata['output_rms_uK']:.1f} μK")
    
    return normalized_map, metadata


def estimate_calibration_systematic(survey_results: Dict[str, Dict]) -> Dict[str, float]:
    """
    Estimate calibration systematic uncertainty from cross-survey comparison.
    
    Parameters:
        survey_results: Dictionary of results per survey
        
    Returns:
        dict: Systematic uncertainty estimates
    """
    if len(survey_results) < 2:
        return {'calibration_systematic': 0.0}
    
    # Extract deficits and uncertainties
    deficits = []
    stat_uncertainties = []
    cal_uncertainties = []
    
    for survey, data in survey_results.items():
        if survey in SURVEY_SYSTEMATICS:
            deficit = data.get('deficit', np.nan)
            uncertainty = data.get('uncertainty', np.nan)
            
            if not np.isnan(deficit) and not np.isnan(uncertainty):
                deficits.append(deficit)
                stat_uncertainties.append(uncertainty)
                
                # Calibration systematic: deficit × calibration_uncertainty
                cal_sys = abs(deficit) * SURVEY_SYSTEMATICS[survey]['calibration_uncertainty']
                cal_uncertainties.append(cal_sys)
    
    if len(deficits) < 2:
        return {'calibration_systematic': 0.0}
    
    deficits = np.array(deficits)
    stat_uncertainties = np.array(stat_uncertainties)
    cal_uncertainties = np.array(cal_uncertainties)
    
    # Estimate systematic from scatter beyond statistical uncertainties
    weighted_mean = np.average(deficits, weights=1.0/stat_uncertainties**2)
    residuals = deficits - weighted_mean
    
    # Excess variance beyond statistical
    stat_variance = np.mean(stat_uncertainties**2)
    total_variance = np.var(residuals)
    
    if total_variance > stat_variance:
        systematic_variance = total_variance - stat_variance
        systematic_uncertainty = np.sqrt(systematic_variance)
    else:
        systematic_uncertainty = 0.0
    
    # Compare to calibration uncertainties
    mean_cal_uncertainty = np.mean(cal_uncertainties)
    
    return {
        'calibration_systematic': float(systematic_uncertainty),
        'expected_cal_systematic': float(mean_cal_uncertainty),
        'systematic_to_statistical_ratio': float(systematic_uncertainty / np.mean(stat_uncertainties))
    }


def apply_beam_correction(C_ell: np.ndarray, 
                         beam_fwhm_arcmin: float,
                         deconvolve: bool = True) -> np.ndarray:
    """
    Apply or remove beam smoothing from power spectrum.
    
    Parameters:
        C_ell: Input power spectrum
        beam_fwhm_arcmin: Beam FWHM in arcminutes
        deconvolve: If True, deconvolve beam; if False, convolve
        
    Returns:
        np.ndarray: Beam-corrected power spectrum
    """
    if not HEALPY_AVAILABLE:
        return C_ell
    
    # Convert FWHM to radians
    beam_fwhm_rad = np.radians(beam_fwhm_arcmin / 60.0)
    
    # Gaussian beam window function
    ell = np.arange(len(C_ell))
    sigma = beam_fwhm_rad / np.sqrt(8.0 * np.log(2.0))
    beam_window = np.exp(-0.5 * ell * (ell + 1) * sigma**2)
    
    # Avoid division by very small numbers
    beam_window = np.maximum(beam_window, 1e-10)
    
    # Apply correction
    if deconvolve:
        # Deconvolve beam (sharpen)
        C_ell_corrected = C_ell / (beam_window**2)
    else:
        # Convolve beam (smooth)
        C_ell_corrected = C_ell * (beam_window**2)
    
    return C_ell_corrected

