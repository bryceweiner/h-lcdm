"""
Cold Spot Extraction Module
============================

Extract the CMB Cold Spot region from full-sky CMB temperature maps.
Handles HEALPix format and coordinate transformations.
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
import logging

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False
    logging.warning("healpy not available. Cold Spot extraction will be limited.")

try:
    from astropy.coordinates import SkyCoord
    from astropy import units as u
    ASTROPY_AVAILABLE = True
except ImportError:
    ASTROPY_AVAILABLE = False
    logging.warning("astropy not available. Coordinate transformations will be limited.")


logger = logging.getLogger(__name__)


# Cold Spot location in Galactic coordinates (Eridanus)
COLD_SPOT_CENTER_GALACTIC = (209.6, -57.0)  # (l, b) in degrees
COLD_SPOT_RADIUS_DEG = 10.0  # Extraction radius in degrees


def extract_cold_spot_region(cmb_map: np.ndarray,
                            nside: Optional[int] = None,
                            center_galactic: Tuple[float, float] = COLD_SPOT_CENTER_GALACTIC,
                            radius_deg: float = COLD_SPOT_RADIUS_DEG) -> Dict[str, Any]:
    """
    Extract Cold Spot region from CMB temperature map.
    
    Parameters:
        cmb_map: Full-sky CMB temperature map (HEALPix format)
        nside: HEALPix resolution parameter (if None, inferred from map size)
        center_galactic: (l, b) in degrees (default: Eridanus Cold Spot)
        radius_deg: Extraction radius in degrees
        
    Returns:
        dict: Extracted region data with keys:
            - temperature_map: Extracted region pixels
            - mask: Valid pixels mask
            - mean_deficit: Average temperature deficit
            - std_deficit: Standard deviation
            - metadata: Extraction parameters
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required for Cold Spot extraction")
    
    # Infer nside from map size if not provided
    if nside is None:
        npix = len(cmb_map)
        nside = hp.npix2nside(npix)
    else:
        npix = hp.nside2npix(nside)
    
    if len(cmb_map) != npix:
        raise ValueError(f"Map size {len(cmb_map)} does not match nside={nside} (expected {npix} pixels)")
    
    # Convert Galactic coordinates to HEALPix pixel indices
    l_rad = np.radians(center_galactic[0])
    b_rad = np.radians(center_galactic[1])
    
    # Get pixels within radius
    vec = hp.ang2vec(np.pi/2 - b_rad, l_rad)  # (theta, phi) where theta is colatitude
    radius_rad = np.radians(radius_deg)
    
    # Find pixels within radius
    ipix_disc = hp.query_disc(nside, vec, radius_rad)
    
    # Create mask
    mask = np.zeros(npix, dtype=bool)
    mask[ipix_disc] = True
    
    # Extract temperature values
    temperature_map = cmb_map[mask]
    
    # Calculate statistics relative to full-sky mean
    full_sky_mean = np.mean(cmb_map)
    mean_deficit = np.mean(temperature_map) - full_sky_mean
    std_deficit = np.std(temperature_map)
    
    # Normalized deficit: δT/T
    # For CMB: δT/T = (temperature fluctuation) / (CMB monopole temperature T₀)
    # T₀ = 2.725 K = 2.725e6 μK
    # Maps from synfast are fluctuations (monopole removed), so we normalize by T₀
    
    # CMB monopole temperature in Kelvin
    T_CMB = 2.725  # Kelvin
    
    # Determine units of the map by checking RMS
    # CMB fluctuations are typically ~100 μK RMS at degree scales
    # If RMS ~ 100, map is in μK; if RMS ~ 1, map is in K
    full_sky_rms = np.std(cmb_map)
    
    if full_sky_rms > 10:
        # Map is likely in μK
        T_norm = T_CMB * 1e6  # Convert to μK
        unit_label = 'μK'
    else:
        # Map is in K or arbitrary units
        T_norm = T_CMB
        unit_label = 'K'
    
    if T_norm > 0:
        # δT/T = (mean_deficit) / T₀
        normalized_deficit = mean_deficit / T_norm
        # Uncertainty in normalized deficit
        n_pixels_extracted = len(temperature_map)
        std_error_mean = std_deficit / np.sqrt(n_pixels_extracted) if n_pixels_extracted > 0 else std_deficit
        normalized_uncertainty = std_error_mean / T_norm
    else:
        normalized_deficit = 0.0
        normalized_uncertainty = 0.0
    
    metadata = {
        'center_galactic_l': center_galactic[0],
        'center_galactic_b': center_galactic[1],
        'radius_deg': radius_deg,
        'nside': nside,
        'npix_extracted': len(temperature_map),
        'npix_total': npix,
        'full_sky_mean': full_sky_mean,
        'full_sky_rms': full_sky_rms,
        'map_units': unit_label,
        'normalization_temperature': T_norm,
        'extraction_fraction': len(temperature_map) / npix
    }
    
    return {
        'temperature_map': temperature_map,
        'mask': mask,
        'mean_deficit': mean_deficit,
        'std_deficit': std_deficit,
        'normalized_deficit': normalized_deficit,
        'normalized_deficit_uncertainty': normalized_uncertainty,
        'full_sky_rms': full_sky_rms,
        'metadata': metadata,
        'pixel_indices': ipix_disc
    }


def extract_cold_spot_from_power_spectrum(ell: np.ndarray,
                                        C_ell: np.ndarray,
                                        center_galactic: Tuple[float, float] = COLD_SPOT_CENTER_GALACTIC,
                                        radius_deg: float = COLD_SPOT_RADIUS_DEG) -> Dict[str, Any]:
    """
    Extract Cold Spot region from power spectrum data.
    
    This is a simplified version that works with power spectrum data
    rather than full-sky maps. Uses spherical harmonic decomposition.
    
    Parameters:
        ell: Multipole array
        C_ell: Power spectrum values
        center_galactic: (l, b) in degrees
        radius_deg: Angular radius in degrees
        
    Returns:
        dict: Extracted region data
    """
    # Convert radius to multipole scale
    # Rough approximation: ℓ ~ 180° / θ (degrees)
    ell_scale = 180.0 / radius_deg if radius_deg > 0 else 1000
    
    # Find relevant multipole range
    ell_min = max(2, int(ell_scale * 0.5))
    ell_max = min(int(ell_scale * 2), len(ell) - 1)
    
    # Extract power spectrum in relevant range
    mask = (ell >= ell_min) & (ell <= ell_max)
    ell_extracted = ell[mask]
    C_ell_extracted = C_ell[mask]
    
    # Calculate mean power in this range
    mean_power = np.mean(C_ell_extracted)
    std_power = np.std(C_ell_extracted)
    
    # Compare with full spectrum mean
    full_mean = np.mean(C_ell)
    power_deficit = mean_power - full_mean
    
    metadata = {
        'center_galactic_l': center_galactic[0],
        'center_galactic_b': center_galactic[1],
        'radius_deg': radius_deg,
        'ell_scale': ell_scale,
        'ell_min': ell_min,
        'ell_max': ell_max,
        'n_multipoles': len(ell_extracted)
    }
    
    return {
        'ell': ell_extracted,
        'C_ell': C_ell_extracted,
        'mean_power': mean_power,
        'std_power': std_power,
        'power_deficit': power_deficit,
        'normalized_deficit': power_deficit / full_mean if full_mean != 0 else 0.0,
        'metadata': metadata
    }

