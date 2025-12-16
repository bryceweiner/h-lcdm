"""
Sound Horizon Calculation with Evolving G
==========================================

Implements TEST 1: Sound horizon r_s calculation accounting for evolving G(z).

The sound horizon is the comoving distance sound waves travel from z_drag to z=∞:
r_s = ∫_{z_drag}^∞ c_s(z) / H(z) dz

With evolving G, H(z) is modified, affecting the integrated sound horizon.

For ΛCDM baseline, uses CAMB (Code for Anisotropies in the Microwave Background)
which provides full Boltzmann solver accuracy including:
- Precise recombination physics (RECFAST/HyRec)
- Neutrino effects (N_eff = 3.046)
- Accurate z_drag calculation
- ~0.1% precision on r_s

For evolving G cases, uses semi-analytic integration with CAMB-derived baseline
for comparison, ensuring the β=0 limit recovers the exact ΛCDM value.
"""

import numpy as np
from scipy.integrate import quad
from typing import Optional, Tuple, Dict
import logging

from hlcdm.parameters import HLCDM_PARAMS
from .evolving_g import H_evolving_G, c_s_baryon_photon, OMEGA_R

logger = logging.getLogger(__name__)

# Try to import CAMB for precise baseline calculation
try:
    import camb
    CAMB_AVAILABLE = True
except ImportError:
    CAMB_AVAILABLE = False
    logger.warning("CAMB not available; using semi-analytic approximations for sound horizon")

# Try to import CAMB-based evolving G module
try:
    from .camb_evolving_g import sound_horizon_evolving_G_camb, CAMB_AVAILABLE as CAMB_EG_AVAILABLE
    USE_CAMB_EVOLVING_G = CAMB_EG_AVAILABLE
except ImportError:
    USE_CAMB_EVOLVING_G = False
    logger.info("CAMB evolving G module not available; using semi-analytic integration")


def sound_horizon_camb(
    H0: float = 67.36,
    omega_b_h2: float = 0.02237,
    omega_c_h2: float = 0.1200,
    tau: float = 0.054,
    ns: float = 0.9649,
    As: float = 2.1e-9
) -> Dict[str, float]:
    """
    Calculate ΛCDM sound horizon using CAMB (full Boltzmann solver).
    
    This is the gold-standard calculation with ~0.1% precision, including:
    - Full recombination physics (RECFAST/HyRec)
    - Neutrino contributions (N_eff = 3.046)
    - Precise drag epoch timing
    
    Parameters:
    -----------
    H0 : float
        Hubble constant in km/s/Mpc (default: 67.36, Planck 2018)
    omega_b_h2 : float
        Physical baryon density Ω_b h² (default: 0.02237, Planck 2018)
    omega_c_h2 : float
        Physical CDM density Ω_c h² (default: 0.1200, Planck 2018)
    tau : float
        Optical depth to reionization (default: 0.054)
    ns : float
        Scalar spectral index (default: 0.9649)
    As : float
        Scalar amplitude (default: 2.1e-9)
        
    Returns:
    --------
    dict
        Dictionary containing:
        - 'r_s': Sound horizon at drag epoch in Mpc
        - 'z_drag': Redshift of drag epoch
        - 'z_star': Redshift of recombination
        - 'theta_star': Angular scale of sound horizon
        
    Raises:
    -------
    ImportError
        If CAMB is not installed
        
    Notes:
    ------
    - Default parameters match Planck 2018 best-fit ΛCDM
    - Returns r_s = 147.09 ± 0.01 Mpc with default parameters
    - z_drag ≈ 1059.94 (baryon decoupling), distinct from z_* ≈ 1089.92 (photon decoupling)
    """
    if not CAMB_AVAILABLE:
        raise ImportError("CAMB is required for precise sound horizon calculation. "
                         "Install with: pip install camb")
    
    # Set up CAMB parameters
    pars = camb.CAMBparams()
    pars.set_cosmology(
        H0=H0,
        ombh2=omega_b_h2,
        omch2=omega_c_h2,
        omk=0,  # Flat universe
        tau=tau
    )
    pars.InitPower.set_params(ns=ns, As=As)
    
    # Get background evolution
    results = camb.get_background(pars)
    derived = results.get_derived_params()
    
    return {
        'r_s': derived['rdrag'],
        'z_drag': derived['zdrag'],
        'z_star': derived['zstar'],
        'theta_star': derived['thetastar'],
        'r_s_star': derived.get('rstar', derived['rdrag']),  # Sound horizon at z_*
        'source': 'CAMB'
    }


def z_drag_eisenstein_hu(omega_b_h2: float, omega_m_h2: float) -> float:
    """
    Calculate drag epoch redshift using Eisenstein & Hu (1998) fitting formula.
    
    This is the analytic approximation for z_drag accurate to ~0.1%.
    
    Parameters:
    -----------
    omega_b_h2 : float
        Physical baryon density Ω_b h²
    omega_m_h2 : float
        Physical matter density Ω_m h² = Ω_b h² + Ω_c h²
        
    Returns:
    --------
    float
        Drag epoch redshift z_drag
        
    References:
    -----------
    Eisenstein & Hu (1998), ApJ 496, 605
    Equation 4 in their paper
    """
    b1 = 0.313 * omega_m_h2**(-0.419) * (1 + 0.607 * omega_m_h2**0.674)
    b2 = 0.238 * omega_m_h2**0.223
    z_d = 1291 * omega_m_h2**0.251 / (1 + 0.659 * omega_m_h2**0.828) * (1 + b1 * omega_b_h2**b2)
    return z_d


def sound_horizon_evolving_G(
    omega_b: float,
    omega_m: Optional[float] = None,
    H0: Optional[float] = None,
    beta: float = 0.0,
    z_drag: Optional[float] = None,
    use_camb: bool = True
) -> float:
    """
    Calculate sound horizon with evolving G: r_s = ∫_{z_drag}^∞ c_s(z) / H(z) dz.
    
    **RIGOROUS METHOD (use_camb=True):** Uses CAMB with phenomenological scaling
    **APPROXIMATE METHOD (use_camb=False):** Semi-analytic integration
    
    Parameters:
    -----------
    omega_b : float
        Baryon density parameter
    omega_m : float, optional
        Matter density parameter. If None, uses HLCDM_PARAMS.OMEGA_M
    H0 : float, optional
        Hubble constant in km/s/Mpc. If None, converts from HLCDM_PARAMS.H0 (s⁻¹)
    beta : float
        G evolution coupling strength (default: 0.0 for ΛCDM)
    z_drag : float, optional
        Drag epoch redshift. If None, uses HLCDM_PARAMS.Z_DRAG
    use_camb : bool
        If True, use CAMB-based calculation (more rigorous). Default: True
        
    Returns:
    --------
    float
        Sound horizon in Mpc
        
    Notes:
    ------
    - CAMB method (use_camb=True, default): ~2% accuracy for |β| < 0.3
    - Semi-analytic method (use_camb=False): ~5% accuracy, faster
    - For production analysis, recommend full CAMB modification at Fortran level
    """
    # Use CAMB-based method if available and requested
    if use_camb and USE_CAMB_EVOLVING_G:
        # Use vetted constants
        if omega_m is None:
            omega_m = HLCDM_PARAMS.OMEGA_M
        if H0 is None:
            H0_s_per_s = HLCDM_PARAMS.H0
            H0 = H0_s_per_s / 3.24e-20  # Convert to km/s/Mpc
        
        # Calculate omega_b_h2 and omega_c_h2
        h = H0 / 100.0
        omega_b_h2 = omega_b * h**2
        omega_c_h2 = (omega_m - omega_b) * h**2
        
        logger.info(f"Using CAMB-based evolving G calculation for β={beta:.4f}")
        return sound_horizon_evolving_G_camb(beta, H0, omega_b_h2, omega_c_h2)
    
    # Fallback to semi-analytic integration
    logger.info(f"Using semi-analytic integration for β={beta:.4f}")
    
    # Use vetted constants from HLCDM_PARAMS
    if omega_m is None:
        omega_m = HLCDM_PARAMS.OMEGA_M
    
    if z_drag is None:
        z_drag = HLCDM_PARAMS.Z_DRAG
    
    # Convert H0 from s⁻¹ to km/s/Mpc if needed
    if H0 is None:
        # HLCDM_PARAMS.H0 is in s⁻¹, convert to km/s/Mpc
        # H0 = 67.4 km/s/Mpc = 2.18e-18 s⁻¹
        # Conversion: 1 km/s/Mpc = 1e3 m/s / (3.086e22 m) = 3.24e-20 s⁻¹
        H0_s_per_s = HLCDM_PARAMS.H0
        H0_km_s_Mpc = H0_s_per_s / 3.24e-20
    else:
        H0_km_s_Mpc = H0
    
    # Speed of light in km/s
    c_km_s = HLCDM_PARAMS.C / 1000.0
    
    def integrand(z):
        """Integrand: c_s(z) / H(z)"""
        # Sound speed in baryon-photon fluid
        c_s = c_s_baryon_photon(z, omega_b)
        
        # Modified Hubble parameter with evolving G
        # Convert H from s⁻¹ to km/s/Mpc for integration
        H_s_per_s = H_evolving_G(z, beta)
        H_km_s_Mpc = H_s_per_s / 3.24e-20
        
        return c_s / H_km_s_Mpc
    
    # Integrate from z_drag to infinity
    # Use large upper limit (z_max) instead of infinity for numerical integration
    # Sound horizon integral converges slowly; need z_max ~ 10^5 for <1% accuracy
    # At very high z, c_s → c/√3 and H ∝ (1+z)², so integrand ∝ 1/(1+z)² → converges
    z_max = 1e5
    
    r_s, _ = quad(integrand, z_drag, z_max, limit=1000, epsabs=1e-3, epsrel=1e-3)
    
    return r_s


def sound_horizon_lcdm(
    omega_b: float = None,
    omega_m: Optional[float] = None,
    H0: Optional[float] = None,
    z_drag: Optional[float] = None,
    use_camb: bool = True
) -> float:
    """
    Calculate standard ΛCDM sound horizon.
    
    When CAMB is available (default), uses full Boltzmann solver for ~0.1% precision.
    Otherwise falls back to semi-analytic integration.
    
    Parameters:
    -----------
    omega_b : float, optional
        Baryon density parameter Ω_b. If None, uses Planck 2018 value.
    omega_m : float, optional
        Matter density parameter Ω_m. If None, uses HLCDM_PARAMS.OMEGA_M
    H0 : float, optional
        Hubble constant in km/s/Mpc. If None, uses 67.36 (Planck 2018)
    z_drag : float, optional
        Drag epoch redshift. Only used for semi-analytic calculation.
    use_camb : bool
        If True (default), use CAMB when available. If False, use semi-analytic.
        
    Returns:
    --------
    float
        Sound horizon at drag epoch in Mpc (ΛCDM value)
        
    Notes:
    ------
    - With Planck 2018 parameters, CAMB returns r_s = 147.09 ± 0.01 Mpc
    - Semi-analytic approximation gives ~1-2% accuracy
    - For evolving G analysis, the semi-analytic method is used with CAMB baseline
    """
    # Default parameters (Planck 2018)
    if H0 is None:
        H0 = 67.36
    if omega_m is None:
        omega_m = HLCDM_PARAMS.OMEGA_M
    
    h = H0 / 100.0
    
    # Use CAMB for gold-standard precision if available
    if use_camb and CAMB_AVAILABLE:
        # Convert density parameters to physical densities for CAMB
        # CAMB takes Ω_b h² and Ω_c h² (physical densities)
        if omega_b is None:
            omega_b_h2 = 0.02237  # Planck 2018
            omega_c_h2 = 0.1200   # Planck 2018
        else:
            omega_b_h2 = omega_b * h**2
            # CDM density: Ω_c = Ω_m - Ω_b
            omega_c = omega_m - omega_b
            omega_c_h2 = omega_c * h**2
        
        try:
            result = sound_horizon_camb(
                H0=H0,
                omega_b_h2=omega_b_h2,
                omega_c_h2=omega_c_h2
            )
            return result['r_s']
        except Exception as e:
            logger.warning(f"CAMB calculation failed: {e}. Falling back to semi-analytic.")
    
    # Fall back to semi-analytic calculation
    if omega_b is None:
        omega_b = 0.02237 / h**2  # Planck 2018
    
    return sound_horizon_evolving_G(omega_b, omega_m, H0, beta=0.0, z_drag=z_drag)

