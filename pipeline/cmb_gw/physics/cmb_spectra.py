"""
CMB Power Spectra with CAMB
============================

Proper ΛCDM CMB power spectra using full Boltzmann solver (CAMB).
No approximations - this is the gold standard for CMB predictions.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging

logger = logging.getLogger(__name__)

try:
    import camb
    CAMB_AVAILABLE = True
except ImportError:
    CAMB_AVAILABLE = False
    logger.warning("CAMB not available for CMB spectrum calculation")


def compute_lcdm_cmb_spectrum(
    H0: float = 67.36,
    omega_b_h2: float = 0.02237,
    omega_c_h2: float = 0.1200,
    tau: float = 0.054,
    As: float = 2.1e-9,
    ns: float = 0.9649,
    lmax: int = 2500,
    unit: str = 'muK'
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute ΛCDM CMB power spectra using CAMB (full Boltzmann solver).
    
    This is the proper, rigorous method with no approximations.
    
    Parameters:
    -----------
    H0 : float
        Hubble constant in km/s/Mpc (default: 67.36, Planck 2018)
    omega_b_h2 : float
        Physical baryon density (default: 0.02237, Planck 2018)
    omega_c_h2 : float
        Physical CDM density (default: 0.1200, Planck 2018)
    tau : float
        Optical depth to reionization (default: 0.054)
    As : float
        Scalar amplitude (default: 2.1e-9)
    ns : float
        Scalar spectral index (default: 0.9649)
    lmax : int
        Maximum multipole (default: 2500)
    unit : str
        'muK' for μK², 'camb' for CAMB internal units, or 'dl' for D_ℓ
        
    Returns:
    --------
    dict
        Dictionary with keys 'TT', 'TE', 'EE' containing (ell, C_ell) tuples.
        C_ell values are in specified units.
        
    Raises:
    -------
    ImportError
        If CAMB is not installed
        
    Notes:
    ------
    - Uses unlensed scalar spectra (primordial, before lensing)
    - For lensed spectra, use lensed=True parameter
    - CAMB units: ℓ(ℓ+1)C_ℓ/(2π) in (μK)²
    - To get C_ℓ: multiply by 2π/(ℓ(ℓ+1))
    """
    if not CAMB_AVAILABLE:
        raise ImportError(
            "CAMB is required for rigorous CMB spectrum calculation. "
            "Install with: pip install camb"
        )
    
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0,
        ombh2=omega_b_h2,
        omch2=omega_c_h2,
        omk=0,  # Flat universe
        tau=tau
    )
    pars.InitPower.set_params(As=As, ns=ns)
    pars.set_for_lmax(lmax, lens_potential_accuracy=0)
    
    # Get results
    results = camb.get_results(pars)
    
    # Get power spectra
    # CMB_unit='muK' gives ℓ(ℓ+1)C_ℓ/(2π) in (μK)²
    powers = results.get_cmb_power_spectra(pars, CMB_unit='muK')
    
    # Extract unlensed scalar spectra
    cl_unlensed = powers['unlensed_scalar']
    ell = np.arange(cl_unlensed.shape[0])
    
    # Columns: TT, EE, BB, TE (in that order)
    Dl_TT = cl_unlensed[:, 0]  # ℓ(ℓ+1)C_ℓ/(2π) in (μK)²
    Dl_EE = cl_unlensed[:, 1]
    Dl_TE = cl_unlensed[:, 3]
    
    if unit == 'muK':
        # Return D_ℓ = ℓ(ℓ+1)C_ℓ/(2π) in (μK)²
        return {
            'TT': (ell, Dl_TT),
            'EE': (ell, Dl_EE),
            'TE': (ell, Dl_TE),
            'unit': 'μK² (D_ℓ)',
            'note': 'D_ℓ = ℓ(ℓ+1)C_ℓ/(2π)'
        }
    elif unit == 'cl':
        # Convert to C_ℓ in (μK)²
        with np.errstate(divide='ignore', invalid='ignore'):
            Cl_TT = Dl_TT * 2 * np.pi / (ell * (ell + 1))
            Cl_EE = Dl_EE * 2 * np.pi / (ell * (ell + 1))
            Cl_TE = Dl_TE * 2 * np.pi / (ell * (ell + 1))
        
        # Set ell=0,1 to zero (undefined)
        Cl_TT[:2] = 0
        Cl_EE[:2] = 0
        Cl_TE[:2] = 0
        
        return {
            'TT': (ell, Cl_TT),
            'EE': (ell, Cl_EE),
            'TE': (ell, Cl_TE),
            'unit': 'μK² (C_ℓ)',
            'note': 'C_ℓ power spectrum'
        }
    else:
        raise ValueError(f"Unknown unit: {unit}")


def compute_cmb_residuals(
    observed_spectra: Dict[str, Tuple[np.ndarray, np.ndarray]],
    H0: float = 67.36,
    omega_b_h2: float = 0.02237,
    omega_c_h2: float = 0.1200
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    Compute CMB residuals: (Observed - ΛCDM) using CAMB for ΛCDM prediction.
    
    This is the PROPER method - no smoothing approximations.
    
    Parameters:
    -----------
    observed_spectra : dict
        Dictionary with 'TT', 'TE', 'EE' keys containing (ell, C_ell) tuples
    H0, omega_b_h2, omega_c_h2 : float
        Cosmological parameters for ΛCDM model
        
    Returns:
    --------
    dict
        Residuals for TT, TE, EE as (ell, residual) tuples
        
    Notes:
    ------
    - Observed and CAMB spectra must be in same units
    - Interpolates CAMB prediction to observed ell grid
    """
    # Get ΛCDM prediction from CAMB
    lcdm_spectra = compute_lcdm_cmb_spectrum(
        H0=H0,
        omega_b_h2=omega_b_h2,
        omega_c_h2=omega_c_h2,
        unit='cl'  # Use C_ℓ for residuals
    )
    
    residuals = {}
    
    for spectrum_type in ['TT', 'TE', 'EE']:
        if spectrum_type not in observed_spectra:
            continue
        
        ell_obs, Cl_obs = observed_spectra[spectrum_type]
        ell_lcdm, Cl_lcdm = lcdm_spectra[spectrum_type]
        
        # Interpolate LCDM prediction to observed ell grid
        Cl_lcdm_interp = np.interp(ell_obs, ell_lcdm, Cl_lcdm)
        
        # Compute residual
        residual = Cl_obs - Cl_lcdm_interp
        
        residuals[spectrum_type] = (ell_obs, residual)
    
    return residuals


def get_acoustic_peak_predictions(
    H0: float = 67.36,
    omega_b_h2: float = 0.02237,
    omega_c_h2: float = 0.1200
) -> Dict[str, float]:
    """
    Get ΛCDM predictions for acoustic peak positions and heights from CAMB.
    
    Returns:
    --------
    dict
        Peak positions (ell) and height ratios R21, R31
    """
    # Get CAMB spectrum
    spectra = compute_lcdm_cmb_spectrum(
        H0=H0,
        omega_b_h2=omega_b_h2,
        omega_c_h2=omega_c_h2,
        unit='muK'  # D_ℓ for peak finding
    )
    
    ell, Dl_TT = spectra['TT']
    
    # Find peaks in D_ℓ (more pronounced than C_ℓ)
    # Restrict to ell > 50 to avoid low-ell features
    mask = (ell > 50) & (ell < 1500)
    ell_range = ell[mask]
    Dl_range = Dl_TT[mask]
    
    from scipy.signal import find_peaks
    
    # Find peaks with reasonable prominence
    peaks_idx, properties = find_peaks(
        Dl_range,
        prominence=np.std(Dl_range) * 0.5,
        distance=100  # Peaks separated by at least 100 multipoles
    )
    
    if len(peaks_idx) < 3:
        logger.warning("Could not find 3 acoustic peaks in CAMB spectrum")
        return {
            'ell_peaks': [],
            'heights': [],
            'R21': np.nan,
            'R31': np.nan
        }
    
    # Get first 3 peaks
    peak_ells = ell_range[peaks_idx[:3]]
    peak_heights = Dl_range[peaks_idx[:3]]
    
    # Compute height ratios
    h1, h2, h3 = peak_heights[:3]
    R21 = h2 / h1
    R31 = h3 / h1
    
    return {
        'ell_peaks': peak_ells.tolist(),
        'heights': peak_heights.tolist(),
        'R21': float(R21),
        'R31': float(R31)
    }

