"""
Evolving Gravitational Constant Physics
========================================

Implements the core physics for evolving G_eff(z) = G_0 × [1 - β × f(z)].

This module provides:
- G_ratio(z, β): Effective gravitational constant relative to present-day value
- H_evolving_G(z, β): Modified Hubble parameter accounting for evolving G
- c_s_baryon_photon(z): Sound speed in baryon-photon fluid

All calculations use vetted constants from HLCDM_PARAMS.
"""

import numpy as np
from typing import Union
from hlcdm.parameters import HLCDM_PARAMS

# Radiation density parameter (CMB photons + relativistic neutrinos)
# From Planck 2018: Ω_r h² = 4.18e-5 (photons) + 3 × 1.68e-5 (neutrinos) ≈ 9.24e-5
# This is the only new constant required; all others from HLCDM_PARAMS
OMEGA_R = 9.24e-5


def G_ratio(z: Union[float, np.ndarray], beta: float) -> Union[float, np.ndarray]:
    """
    Calculate G_eff(z) / G_0 = 1 - β × f(z) where f(z) = Ω_r(z) / [Ω_r(z) + Ω_m(z)].
    
    This implements Equation 1 from docs/cmb_gw.md.
    
    Parameters:
    -----------
    z : float or array
        Redshift(s)
    beta : float
        Coupling strength to information-theoretic vacuum departure
        
    Returns:
    --------
    float or array
        G_eff(z) / G_0 ratio. At z=0, this equals 1.0 regardless of β.
        
    Notes:
    ------
    - Uses HLCDM_PARAMS.OMEGA_M for matter density
    - At z=0: f(0) = Ω_r / (Ω_r + Ω_m) ≈ 0, so G_ratio = 1.0
    - At z >> 3400 (radiation era): f(z) → 1, so G_ratio → 1 - β
    - At z ≈ 1100 (recombination): f(z) ≈ 0.24, so G_ratio ≈ 1 - 0.24β
    """
    z = np.asarray(z)
    
    # Redshift-dependent densities
    Om_r_z = OMEGA_R * (1 + z)**4
    Om_m_z = HLCDM_PARAMS.OMEGA_M * (1 + z)**3
    
    # Transition function f(z) = Ω_r(z) / [Ω_r(z) + Ω_m(z)]
    f_z = Om_r_z / (Om_r_z + Om_m_z)
    
    # G_eff(z) / G_0 = 1 - β × f(z)
    ratio = 1 - beta * f_z
    
    # Return scalar if input was scalar
    if isinstance(z, (int, float)):
        return float(ratio)
    return ratio


def H_evolving_G(z: Union[float, np.ndarray], beta: float) -> Union[float, np.ndarray]:
    """
    Calculate Hubble parameter with evolving G: H² ∝ G_eff × ρ, so H ∝ √G_eff.
    
    The Hubble parameter is modified because H² = (8πG/3)ρ, so with evolving G:
    H_evolving² = H_lcdm² × (G_eff / G_0)
    
    Parameters:
    -----------
    z : float or array
        Redshift(s)
    beta : float
        Coupling strength
        
    Returns:
    --------
    float or array
        Modified Hubble parameter in s⁻¹
        
    Notes:
    ------
    - Computes full ΛCDM Hubble including radiation (critical for early universe)
    - At β=0, this matches standard ΛCDM Hubble parameter
    - At β>0, H is reduced in the early universe (weaker G → slower expansion)
    """
    z = np.asarray(z)
    
    # Compute full ΛCDM Hubble parameter INCLUDING RADIATION
    # H(z) = H0 × sqrt(Ω_r(1+z)⁴ + Ω_m(1+z)³ + Ω_Λ)
    # Note: HLCDM_PARAMS.get_hubble_at_redshift() omits radiation, which is critical
    # for early universe calculations (sound horizon, etc.)
    
    Om_r_z = OMEGA_R * (1 + z)**4
    Om_m_z = HLCDM_PARAMS.OMEGA_M * (1 + z)**3
    Om_L = HLCDM_PARAMS.OMEGA_LAMBDA
    
    # Standard ΛCDM Hubble
    H_lcdm = HLCDM_PARAMS.H0 * np.sqrt(Om_r_z + Om_m_z + Om_L)
    
    # Apply G evolution: H ∝ √G_eff
    G_eff_ratio = G_ratio(z, beta)
    H_modified = H_lcdm * np.sqrt(G_eff_ratio)
    
    # Return scalar if input was scalar
    if isinstance(z, (int, float)):
        return float(H_modified)
    return H_modified


def c_s_baryon_photon(z: Union[float, np.ndarray], omega_b: float = None) -> Union[float, np.ndarray]:
    """
    Calculate sound speed in baryon-photon fluid.
    
    c_s = c / √(3(1 + R)) where R = 3ρ_b / (4ρ_γ)
    
    Parameters:
    -----------
    z : float or array
        Redshift(s)
    omega_b : float, optional
        Baryon density parameter. If None, uses HLCDM_PARAMS value if available,
        otherwise defaults to 0.049 (Planck 2018).
        
    Returns:
    --------
    float or array
        Sound speed in km/s
        
    Notes:
    ------
    - Uses HLCDM_PARAMS.C for speed of light
    - R = 3ρ_b / (4ρ_γ) = (3/4) × (Ω_b / Ω_γ) × (1+z)⁻¹
    - At high z, R → 0 and c_s → c/√3
    """
    z = np.asarray(z)
    
    # Baryon density parameter (use default if not provided)
    if omega_b is None:
        # Try to get from HLCDM_PARAMS if available, otherwise use Planck 2018 value
        omega_b = getattr(HLCDM_PARAMS, 'OMEGA_B', 0.049)
    
    # Speed of light in km/s (convert from m/s)
    c_km_s = HLCDM_PARAMS.C / 1000.0
    
    # Baryon-to-photon ratio R = 3ρ_b / (4ρ_γ)
    # Standard CMB physics formula:
    # R = (3/4) × (Ω_b / Ω_γ) × (1+z)⁻¹
    # CRITICAL: Both densities must be in same units (density parameters, not physical)
    # 
    # Photon density from T_CMB = 2.7255 K:
    # Ω_γ h² = 2.47e-5 → Ω_γ = 2.47e-5 / h²
    # Using h = 0.674 (Planck 2018): Ω_γ = 5.44e-5
    # 
    # Note: OMEGA_R = 9.24e-5 includes neutrinos; for sound speed we need photons only
    h_planck = 0.674  # Planck 2018 value
    omega_gamma_h2 = 2.47e-5  # Ω_γ h² (photon density in physical units)
    omega_gamma = omega_gamma_h2 / (h_planck**2)  # Convert to density parameter Ω_γ
    
    R = (3.0 * omega_b / (4.0 * omega_gamma)) / (1 + z)
    
    # Sound speed: c_s = c / √(3(1 + R))
    c_s = c_km_s / np.sqrt(3.0 * (1 + R))
    
    # Return scalar if input was scalar
    if isinstance(z, (int, float)):
        return float(c_s)
    return c_s

