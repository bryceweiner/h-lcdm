"""
Luminosity Distance Calculation with Evolving G
================================================

Implements TEST 3: Luminosity distance d_L(z) with evolving G(z).

The luminosity distance is:
d_L(z) = (1+z) × ∫₀ᶻ c dz' / H(z')

With evolving G, H(z) is modified, affecting the integrated distance.
"""

import numpy as np
from scipy.integrate import quad
from typing import Optional, Union
from hlcdm.parameters import HLCDM_PARAMS
from .evolving_g import H_evolving_G


def luminosity_distance_evolving_G(
    z: Union[float, np.ndarray],
    omega_m: Optional[float] = None,
    H0: Optional[float] = None,
    beta: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate luminosity distance with evolving G: d_L(z) = (1+z) × ∫₀ᶻ c dz' / H(z').
    
    Parameters:
    -----------
    z : float or array
        Redshift(s)
    omega_m : float, optional
        Matter density parameter. If None, uses HLCDM_PARAMS.OMEGA_M
    H0 : float, optional
        Hubble constant in km/s/Mpc. If None, converts from HLCDM_PARAMS.H0
    beta : float
        G evolution coupling strength (default: 0.0 for ΛCDM)
        
    Returns:
    --------
    float or array
        Luminosity distance in Mpc
        
    Notes:
    ------
    - Uses HLCDM_PARAMS.C for speed of light
    - At β=0, this recovers standard ΛCDM luminosity distance
    - At β>0, d_L is modified because H(z) is reduced in early universe
    """
    z = np.asarray(z)
    scalar_input = isinstance(z, (int, float))
    z = np.atleast_1d(z)
    
    if omega_m is None:
        omega_m = HLCDM_PARAMS.OMEGA_M
    
    # Convert H0 from s⁻¹ to km/s/Mpc if needed
    if H0 is None:
        H0_s_per_s = HLCDM_PARAMS.H0
        H0_km_s_Mpc = H0_s_per_s / 3.24e-20
    else:
        H0_km_s_Mpc = H0
    
    # Speed of light in km/s
    c_km_s = HLCDM_PARAMS.C / 1000.0
    
    def integrand(zp):
        """Integrand: c / H(z')"""
        # Modified Hubble parameter with evolving G
        # Convert H from s⁻¹ to km/s/Mpc for integration
        H_s_per_s = H_evolving_G(zp, beta)
        H_km_s_Mpc = H_s_per_s / 3.24e-20
        return c_km_s / H_km_s_Mpc
    
    d_L_array = np.zeros_like(z, dtype=float)
    
    for i, z_val in enumerate(z):
        # Comoving distance: ∫₀ᶻ c dz' / H(z')
        try:
            comoving_dist, _ = quad(integrand, 0, z_val, limit=1000, epsabs=1e-2, epsrel=1e-2)
            
            # Luminosity distance: d_L = (1+z) × comoving distance
            d_L_array[i] = (1 + z_val) * comoving_dist
        except Exception:
            d_L_array[i] = np.nan
    
    if scalar_input:
        return float(d_L_array[0]) if np.isfinite(d_L_array[0]) else np.nan
    return d_L_array


def dL_residual(
    z: Union[float, np.ndarray],
    omega_m: Optional[float] = None,
    H0: Optional[float] = None,
    beta: float = 0.0
) -> Union[float, np.ndarray]:
    """
    Calculate fractional deviation from ΛCDM: [d_L(evolving G) - d_L(ΛCDM)] / d_L(ΛCDM).
    
    Parameters:
    -----------
    z : float or array
        Redshift(s)
    omega_m : float, optional
        Matter density parameter. If None, uses HLCDM_PARAMS.OMEGA_M
    H0 : float, optional
        Hubble constant in km/s/Mpc. If None, converts from HLCDM_PARAMS.H0
    beta : float
        G evolution coupling strength
        
    Returns:
    --------
    float or array
        Fractional residual: (d_L_evolving - d_L_lcdm) / d_L_lcdm
    """
    dL_evolving = luminosity_distance_evolving_G(z, omega_m, H0, beta)
    dL_lcdm = luminosity_distance_evolving_G(z, omega_m, H0, 0.0)
    
    return (dL_evolving - dL_lcdm) / dL_lcdm

