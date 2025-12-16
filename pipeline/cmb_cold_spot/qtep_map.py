"""
QTEP Efficiency Map Generator
==============================

Generate predicted QTEP efficiency map for CMB last-scattering surface.
Uses QTEP framework to predict spatial variations in information processing efficiency.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
import logging

try:
    import healpy as hp
    HEALPY_AVAILABLE = True
except ImportError:
    HEALPY_AVAILABLE = False
    logging.warning("healpy not available. QTEP map generation will be limited.")

from hlcdm.parameters import HLCDM_PARAMS, QTEP_RATIO

logger = logging.getLogger(__name__)


def generate_qtep_efficiency_map(cosmology_params: Optional[Dict[str, Any]] = None,
                                redshift: float = 1089,
                                nside: int = 256,
                                seed: Optional[int] = None,
                                observed_cmb_map: Optional[np.ndarray] = None) -> Dict[str, Any]:
    """
    Generate predicted QTEP efficiency map at recombination epoch (z~1089).
    
    Literature-standard approach following CMB analysis conventions:
    1. QTEP predicts: δη/η = δT/T at last scattering
    2. δT/T arises from primordial curvature perturbations via Sachs-Wolfe
    3. Therefore: η(θ,φ) should correlate with observed CMB temperature field
    
    This follows the standard procedure for testing alternative theories against CMB:
    - Use observed temperature field as tracer of primordial perturbations
    - Predict how alternative physics (QTEP) couples to same perturbations
    - Test via angular cross-power spectrum (not pixel-by-pixel correlation)
    
    References:
    - Sachs & Wolfe (1967) on temperature-potential relation
    - Hu & Dodelson (2002) Ann.Rev.Astron.Astrophys. for CMB theory
    - Planck Collaboration (2020) A&A 641, A6 for analysis methods
    
    Parameters:
        cosmology_params: Cosmological parameters (optional, uses HLCDM defaults)
        redshift: Redshift of last-scattering (default: 1089 for recombination)
        nside: HEALPix resolution parameter
        seed: Random seed for stochastic component
        observed_cmb_map: Observed CMB temperature map (if provided, use for correlation)
        
    Returns:
        dict: Efficiency map data with keys:
            - efficiency_map: HEALPix map of η(θ,φ)
            - mean_efficiency: Cosmic mean η₀ ≈ 2.257
            - coupling_strength: δη/δT coupling from QTEP
            - method: 'observed_correlated' or 'theoretical_realization'
    """
    if not HEALPY_AVAILABLE:
        raise ImportError("healpy required for QTEP map generation")
    
    npix = hp.nside2npix(nside)
    ell_max = 3 * nside
    
    # Cosmic mean efficiency from QTEP framework
    mean_efficiency = QTEP_RATIO  # η₀ ≈ 2.257
    
    if observed_cmb_map is not None and len(observed_cmb_map) == npix:
        # METHOD 1: Use observed CMB to predict QTEP efficiency (literature standard)
        # QTEP framework: δη/η = δT/T
        # This creates a map where efficiency variations mirror temperature variations
        
        # Normalize CMB map: δT/T
        T_CMB = 2.725  # K (CMB monopole)
        delta_T_over_T = observed_cmb_map / (T_CMB * 1e6)  # Convert μK to K
        
        # Predict efficiency variations: δη = η₀ × (δT/T)
        # This is the QTEP prediction at recombination epoch
        efficiency_map = mean_efficiency * (1.0 + delta_T_over_T)
        
        method = 'observed_correlated'
        coupling_strength = 1.0  # Perfect coupling in QTEP: δη/η = δT/T
        
        logger.info("Generated QTEP efficiency map from observed CMB (z=1089)")
        logger.info("QTEP coupling: δη/η = δT/T (direct prediction)")
        
    else:
        # METHOD 2: Generate theoretical realization from ΛCDM+QTEP power spectrum
        # Use standard ΛCDM primordial power spectrum at recombination
        
        if seed is not None:
            np.random.seed(seed)
        
        ell = np.arange(ell_max + 1)
        
        # Standard ΛCDM power spectrum shape at recombination
        # C_ℓ^TT for primordial perturbations (before ISW, lensing, etc.)
        # Follow Sachs-Wolfe plateau + acoustic peaks structure
        
        # Primordial amplitude: A_s ≈ 2.1×10⁻⁹ (Planck 2018)
        # At recombination: δT/T ~ 10⁻⁵ RMS
        A_s = 2.1e-9  # Dimensionless primordial amplitude
        T_CMB_uK = 2.725e6  # CMB temperature in μK
        
        # Temperature power spectrum in (μK)²
        # Simplified: Sachs-Wolfe plateau + damping at high ℓ
        C_ell_temperature = np.zeros(ell_max + 1)
        
        # Large scales (ℓ < 50): Sachs-Wolfe plateau
        # C_ℓ^TT ≈ (2π/9) × A_s × T_CMB² × (ℓ(ℓ+1))⁻¹ for ℓ < 50
        mask_SW = (ell >= 2) & (ell < 50)
        C_ell_temperature[mask_SW] = (2*np.pi/9) * A_s * (T_CMB_uK**2) / (ell[mask_SW] * (ell[mask_SW] + 1))
        
        # Acoustic scales (50 < ℓ < 1000): Oscillations + damping
        # Simplified: exponential damping × oscillations
        mask_acoustic = (ell >= 50) & (ell <= ell_max)
        ell_acoustic = ell[mask_acoustic]
        ell_peak = 220  # First acoustic peak location
        damping_scale = 1400  # Silk damping scale
        
        # Acoustic oscillations with damping
        oscillation = 1.0 + 0.3 * np.cos(np.pi * (ell_acoustic - ell_peak) / 60)
        damping = np.exp(-(ell_acoustic / damping_scale)**2)
        C_ell_temperature[mask_acoustic] = (
            C_ell_temperature[49] * oscillation * damping * (50.0 / ell_acoustic)**2
        )
        
        C_ell_temperature[0:2] = 0  # Remove monopole and dipole
        
        # QTEP efficiency power spectrum: C_ℓ^ηη = η₀² × C_ℓ^TT / T_CMB²
        # Since δη/η = δT/T, the fractional power spectra are identical
        C_ell_efficiency = (mean_efficiency / T_CMB_uK)**2 * C_ell_temperature
        
        # Generate efficiency map from power spectrum
        try:
            delta_efficiency_map = hp.synfast(C_ell_efficiency, nside, new=True)
            efficiency_map = mean_efficiency + delta_efficiency_map
        except Exception as e:
            logger.warning(f"synfast failed, using simpler fallback: {e}")
            # Fallback: scale observed CMB if available, else white noise
            std_eff = mean_efficiency * 3e-5  # ~3×10⁻⁵ RMS (typical CMB)
            efficiency_map = np.random.normal(mean_efficiency, std_eff, npix)
        
        method = 'theoretical_realization'
        coupling_strength = 1.0
        
        logger.info("Generated QTEP efficiency map from theoretical power spectrum")
        logger.info("Using ΛCDM+QTEP prediction at z=1089")
    
    # Ensure physical bounds (efficiency must be positive)
    efficiency_map = np.maximum(efficiency_map, 0.1 * mean_efficiency)
    
    # Calculate statistics
    actual_mean = np.mean(efficiency_map)
    actual_std = np.std(efficiency_map)
    relative_variance = actual_std / actual_mean
    
    metadata = {
        'redshift': redshift,
        'nside': nside,
        'npix': npix,
        'theoretical_mean': mean_efficiency,
        'actual_mean': actual_mean,
        'actual_std': actual_std,
        'relative_variance': relative_variance,
        'method': method,
        'coupling_strength': coupling_strength,
        'epoch': 'recombination (z~1089)',
        'physics': 'QTEP prediction: δη/η = δT/T',
        'seed': seed
    }
    
    logger.info(f"QTEP map statistics: mean={actual_mean:.4f}, std={actual_std:.6f}, "
               f"relative_var={relative_variance:.2e}")
    
    return {
        'efficiency_map': efficiency_map,
        'mean_efficiency': actual_mean,
        'std_efficiency': actual_std,
        'theoretical_mean': mean_efficiency,
        'coupling_strength': coupling_strength,
        'method': method,
        'nside': nside,
        'metadata': metadata
    }


def calculate_qtep_efficiency_variation(cosmology_params: Optional[Dict[str, Any]] = None) -> Dict[str, float]:
    """
    Calculate expected QTEP efficiency variation for Cold Spot.
    
    Based on the framework prediction: δT/T = δη/η
    
    Parameters:
        cosmology_params: Cosmological parameters (optional)
        
    Returns:
        dict: Expected efficiency variation parameters
    """
    # Mean efficiency
    eta_mean = QTEP_RATIO  # ≈ 2.257
    
    # Observed Cold Spot temperature deficit
    # Typical values: δT/T ≈ -2.6×10⁻⁵ to -5.1×10⁻⁵
    delta_T_over_T_observed = -2.6e-5  # Conservative estimate
    
    # Predicted efficiency variation
    delta_eta_over_eta = delta_T_over_T_observed
    
    # Absolute efficiency change
    delta_eta = eta_mean * delta_eta_over_eta
    
    return {
        'eta_mean': eta_mean,
        'delta_T_over_T_observed': delta_T_over_T_observed,
        'delta_eta_over_eta': delta_eta_over_eta,
        'delta_eta': delta_eta,
        'eta_predicted': eta_mean + delta_eta
    }

