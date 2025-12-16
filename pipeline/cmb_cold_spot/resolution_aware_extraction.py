"""
Resolution-Aware Cold Spot Extraction
======================================

Extract Cold Spot respecting each survey's native characteristics:
- Resolution (beam FWHM)
- Pixelization scheme
- Systematic uncertainties

Principle: "Meet the data where it is" - don't force COBE (1° beam) to match
Planck (13 arcmin resolution). Compare at matched angular scales only.

References:
- Planck Collaboration (2020): Beam and resolution characteristics
- WMAP 9-year: Bennett et al. (2013)
- COBE DMR: Smoot et al. (1992)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False

logger = logging.getLogger(__name__)

# Survey characteristics from literature
SURVEY_SPECS = {
    'planck_2018': {
        'beam_fwhm_arcmin': 13.0,  # Planck SMICA effective
        'native_nside': 2048,
        'recommended_nside': 256,
        'systematic_floor_uK': 2.0,
        'reference': 'Planck Collaboration (2020) A&A 641, A6'
    },
    'wmap': {
        'beam_fwhm_arcmin': 30.0,  # WMAP ILC effective  
        'native_nside': 512,
        'recommended_nside': 128,
        'systematic_floor_uK': 5.0,
        'reference': 'Bennett et al. (2013) ApJS 208, 20'
    },
    'cobe': {
        'beam_fwhm_arcmin': 60.0,  # COBE DMR 7° → 1° effective after smoothing
        'native_nside': 16,  # Reprojected from quad-cube
        'recommended_nside': 16,
        'systematic_floor_uK': 30.0,
        'reference': 'Smoot et al. (1992) ApJ 396, L1'
    },
    'act_dr6': {
        'beam_fwhm_arcmin': 1.4,  # ACT 150 GHz
        'native_nside': 8192,  # Very high res
        'recommended_nside': 512,
        'systematic_floor_uK': 10.0,  # Partial sky, higher noise
        'reference': 'Aiola et al. (2020) JCAP 12, 047'
    },
    'spt3g': {
        'beam_fwhm_arcmin': 1.2,  # SPT-3G 150 GHz
        'native_nside': 8192,
        'recommended_nside': 512,
        'systematic_floor_uK': 10.0,
        'reference': 'Dutcher et al. (2021) Phys. Rev. D 104, 022003'
    }
}


def get_effective_resolution_deg(survey_name: str) -> float:
    """
    Get effective angular resolution in degrees.
    
    For Cold Spot analysis (10° scale), this determines:
    - How well the survey can resolve internal structure
    - Appropriate uncertainty scaling
    
    Parameters:
        survey_name: Survey identifier
        
    Returns:
        Effective resolution in degrees
    """
    specs = SURVEY_SPECS.get(survey_name, {})
    beam_arcmin = specs.get('beam_fwhm_arcmin', 30.0)
    return beam_arcmin / 60.0  # Convert arcmin → degrees


def get_beam_smearing_factor(survey_name: str,
                             feature_scale_deg: float = 10.0) -> float:
    """
    Calculate beam smearing factor for feature at given scale.
    
    For a Gaussian beam:
    smearing_factor = exp(-ℓ²σ²/2) where σ = FWHM/(2√(2ln2))
    
    For large features (10° Cold Spot):
    - Planck (13 arcmin): negligible smearing (<1%)
    - WMAP (30 arcmin): minor smearing (~2%)
    - COBE (60 arcmin): moderate smearing (~8%)
    
    Parameters:
        survey_name: Survey identifier
        feature_scale_deg: Angular scale of feature in degrees
        
    Returns:
        Smearing factor (1.0 = no smearing, <1.0 = attenuated)
    """
    beam_fwhm_deg = get_effective_resolution_deg(survey_name)
    
    # Convert scale to multipole
    ell = 180.0 / feature_scale_deg
    
    # Gaussian beam window function
    # B(ℓ) = exp(-ℓ(ℓ+1)σ²/2) where σ = FWHM/(2√(2ln2))
    sigma_rad = np.radians(beam_fwhm_deg) / (2 * np.sqrt(2 * np.log(2)))
    beam_window = np.exp(-ell * (ell + 1) * sigma_rad**2 / 2)
    
    return beam_window


def calculate_resolution_corrected_uncertainty(base_uncertainty: float,
                                              survey_name: str,
                                              feature_scale_deg: float = 10.0) -> float:
    """
    Correct uncertainty for resolution effects.
    
    Lower resolution → larger effective uncertainty for fine structure
    But for 10° Cold Spot, all surveys can resolve it, so correction is small
    
    Parameters:
        base_uncertainty: Statistical uncertainty from RMS
        survey_name: Survey identifier
        feature_scale_deg: Feature scale in degrees
        
    Returns:
        Resolution-corrected uncertainty
    """
    specs = SURVEY_SPECS.get(survey_name, {})
    systematic_floor = specs.get('systematic_floor_uK', 5.0)
    
    # For large-scale features, add systematic floor in quadrature
    # (beam smearing uncertainty + calibration uncertainty)
    total_uncertainty = np.sqrt(base_uncertainty**2 + systematic_floor**2)
    
    return total_uncertainty


def extract_cold_spot_resolution_aware(cmb_map: np.ndarray,
                                       survey_name: str,
                                       nside: Optional[int] = None,
                                       center_galactic: Tuple[float, float] = (209.0, -57.0),
                                       radius_deg: float = 5.0) -> Dict[str, Any]:
    """
    Extract Cold Spot with resolution-aware treatment.
    
    Key principle: Meet the data where it is!
    - Accept map at its CURRENT nside (don't resample)
    - Apply beam smearing corrections appropriate for survey
    - Add systematic floor to uncertainties
    - Report effective resolution
    
    Parameters:
        cmb_map: Full-sky CMB map at its native/normalized nside
        survey_name: Survey identifier
        nside: HEALPix nside (if None, inferred from map size)
        center_galactic: (l, b) in degrees
        radius_deg: Extraction radius
        
    Returns:
        Dictionary with resolution-aware extraction
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required")
    
    # Get survey specifications
    specs = SURVEY_SPECS.get(survey_name, {})
    
    # Infer nside from map size (don't force to recommended!)
    if nside is None:
        npix = len(cmb_map)
        nside = hp.npix2nside(npix)
        logger.info(f"{survey_name}: Using map's native nside={nside}")
    
    # Get effective resolution
    effective_res_deg = get_effective_resolution_deg(survey_name)
    beam_smearing = get_beam_smearing_factor(survey_name, 2 * radius_deg)
    
    logger.info(f"{survey_name}: Extracting at nside={nside}, resolution={effective_res_deg:.2f}°")
    logger.info(f"{survey_name}: Beam smearing factor = {beam_smearing:.4f} for {2*radius_deg}° feature")
    
    # Standard extraction
    l_rad = np.radians(center_galactic[0])
    b_rad = np.radians(center_galactic[1])
    vec = hp.ang2vec(np.pi/2 - b_rad, l_rad)
    radius_rad = np.radians(radius_deg)
    ipix_disc = hp.query_disc(nside, vec, radius_rad)
    
    # Extract values
    mask = np.zeros(hp.nside2npix(nside), dtype=bool)
    mask[ipix_disc] = True
    temperature_map = cmb_map[mask]
    
    # Calculate statistics
    full_sky_mean = np.mean(cmb_map)
    full_sky_rms = np.std(cmb_map)
    
    # Determine units (μK or K)
    if full_sky_rms > 1e-3:
        unit_label = "μK"
        T_norm = 2.725e6  # μK
    else:
        unit_label = "K"
        T_norm = 2.725  # K
    
    # Cold Spot statistics
    mean_temp = np.mean(temperature_map)
    deficit = (mean_temp - full_sky_mean)
    normalized_deficit = deficit / T_norm
    
    # Resolution-corrected uncertainty
    base_uncertainty = full_sky_rms / np.sqrt(len(temperature_map))
    corrected_uncertainty = calculate_resolution_corrected_uncertainty(
        base_uncertainty, survey_name, 2 * radius_deg
    )
    normalized_uncertainty = corrected_uncertainty / T_norm
    
    # Beam correction to deficit (if significant smearing)
    beam_corrected_deficit = deficit / beam_smearing if beam_smearing < 0.99 else deficit
    
    logger.info(f"{survey_name}: Deficit = {deficit:.2f} {unit_label}, "
               f"Normalized = {normalized_deficit:.2e}")
    logger.info(f"{survey_name}: Uncertainty (corrected) = {corrected_uncertainty:.2f} {unit_label}")
    
    # Create metadata dictionary
    metadata = {
        'survey_name': survey_name,
        'center_galactic_l': center_galactic[0],
        'center_galactic_b': center_galactic[1],
        'radius_deg': radius_deg,
        'nside': nside,
        'npix_extracted': len(temperature_map),
        'npix_total': hp.nside2npix(nside),
        'full_sky_mean': full_sky_mean,
        'full_sky_rms': full_sky_rms,
        'map_units': unit_label,
        'normalization_temperature': T_norm,
        'extraction_fraction': len(temperature_map) / hp.nside2npix(nside),
        'effective_resolution_deg': effective_res_deg,
        'beam_fwhm_arcmin': specs.get('beam_fwhm_arcmin', 30.0),
        'beam_smearing_factor': float(beam_smearing)
    }
    
    return {
        'survey_name': survey_name,
        'temperature_map': temperature_map,
        'mask': mask,  # Full-sky mask (needed for visualization)
        'mean_deficit': deficit,
        'std_deficit': np.std(temperature_map),
        'deficit': deficit,
        'normalized_deficit': float(normalized_deficit),
        'normalized_deficit_uncertainty': float(normalized_uncertainty),  # Match test_1 expectation!
        'base_uncertainty': base_uncertainty,
        'corrected_uncertainty': corrected_uncertainty,
        'normalized_uncertainty': float(normalized_uncertainty),  # Also keep for reference
        'beam_smearing_factor': float(beam_smearing),
        'beam_corrected_deficit': beam_corrected_deficit,
        'effective_resolution_deg': effective_res_deg,
        'nside_used': nside,
        'npix_extracted': len(temperature_map),
        'full_sky_rms': full_sky_rms,
        'unit': unit_label,
        'specs': specs,
        'metadata': metadata
    }

