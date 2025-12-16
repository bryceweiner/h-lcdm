"""
Growth Factor Calculation with Evolving G
==========================================

Implements TEST 2: Linear growth factor D(z) with evolving G(z).

The linear growth equation is:
δ'' + 2Hδ' = 4πG_eff ρ_m δ

With evolving G, the growth factor is suppressed in the early universe.
"""

import numpy as np
from scipy.integrate import odeint
from typing import Optional, Union
from hlcdm.parameters import HLCDM_PARAMS
from .evolving_g import G_ratio, H_evolving_G, OMEGA_R


def growth_factor_evolving_G(
    z_array: np.ndarray,
    omega_m: Optional[float] = None,
    beta: float = 0.0
) -> np.ndarray:
    """
    Solve growth equation with evolving G.
    
    Standard linear growth equation (Hamilton 2001, Dodelson 2003):
    D'' + 2H D' + (k²c_s² - 4πG_eff ρ_m) D = 0
    
    For scales inside horizon and pressureless matter: k²c_s² << 4πG ρ_m, giving:
    D'' + 2H D' = 4πG_eff ρ_m D
    
    This is solved numerically using scale factor a as the integration variable.
    
    Parameters:
    -----------
    z_array : array
        Redshifts at which to evaluate growth factor
    omega_m : float, optional
        Matter density parameter. If None, uses HLCDM_PARAMS.OMEGA_M
    beta : float
        G evolution coupling strength (default: 0.0 for ΛCDM)
        
    Returns:
    --------
    array
        Growth factor D(z) normalized to D(0) = 1
        
    Notes:
    ------
    - Uses Carroll & Ostlie (2007) Eq. 29.42 and Dodelson (2003) Eq. 7.77
    - Initial conditions set deep in matter era (z ~ 100) where D ∝ a
    - At β=0, recovers standard ΛCDM growth factor
    - At β>0, growth is suppressed in early universe (weaker G → less structure growth)
    
    References:
    - Hamilton (2001), MNRAS 322, 419: "Linear growth rate"
    - Dodelson (2003), "Modern Cosmology", Eq. 7.77
    - Carroll & Ostlie (2007), "Introduction to Modern Astrophysics", Eq. 29.42
    """
    if omega_m is None:
        omega_m = HLCDM_PARAMS.OMEGA_M
    
    # Ensure z_array is sorted in descending order for integration
    z_sorted = np.sort(z_array)[::-1]
    a_sorted = 1.0 / (1.0 + z_sorted)
    
    def E_squared(a):
        """(H/H0)² with evolving G"""
        z = 1.0/a - 1.0
        Om_r_a = OMEGA_R / a**4
        Om_m_a = omega_m / a**3
        Om_L = 1 - omega_m - OMEGA_R
        
        # Standard ΛCDM E²
        E2_lcdm = Om_r_a + Om_m_a + Om_L
        
        # Apply G evolution: H² ∝ G_eff
        G_eff_ratio = G_ratio(z, beta)
        return E2_lcdm * G_eff_ratio
    
    def growth_ode(y, a):
        """
        Growth ODE in scale factor a.
        
        Standard form (Dodelson 2003, Eq. 7.77):
        a² d²D/da² + a[3 - d ln(a³E)/d ln a] dD/da - (3/2) Ω_m(a) G_eff D = 0
        
        Rearranging:
        d²D/da² = (3/2) Ω_m(a) G_eff D / a² - [3 - d ln(a³E)/d ln a] dD/da / a
        """
        D, dD_da = y
        z = 1.0/a - 1.0
        
        # Hubble parameter E(a) = H(a)/H₀
        E2 = E_squared(a)
        
        # Ω_m(a) = Ω_m,0 / (a³ E²)
        Om_m_a = omega_m / (a**3 * E2)
        
        # G_eff ratio
        G_eff = G_ratio(z, beta)
        
        # Compute d ln(a³E)/d ln a = 3 + d ln E/d ln a
        # d ln E²/d ln a = a dE²/da / E² = a × [derivative of E²] / E²
        Om_r_a = OMEGA_R / a**4
        Om_L = 1.0 - omega_m - OMEGA_R
        dE2_da = -4 * Om_r_a / a - 3 * omega_m / a**4
        dlnE2_dlna = a * dE2_da / E2
        dlnE_dlna = 0.5 * dlnE2_dlna
        dlna3E_dlna = 3.0 + dlnE_dlna
        
        # Growth equation (Dodelson form):
        d2D_da2 = (3.0/2.0) * Om_m_a * G_eff * D / a**2 - (3.0 - dlna3E_dlna) * dD_da / a
        
        return [dD_da, d2D_da2]
    
    # Initial conditions deep in matter era (z = 100)
    # In matter domination: D ∝ a, so dD/da = constant
    a_init = 1.0 / 101.0  # a = 1/(1+z) for z=100
    D_init = a_init  # D ∝ a in matter era (unnormalized)
    dD_da_init = 1.0  # dD/da = const when D ∝ a
    
    solution = odeint(growth_ode, [D_init, dD_da_init], a_sorted)
    D_sorted = solution[:, 0]
    
    # Map back to original z_array order
    sort_indices = np.argsort(z_array)[::-1]
    inverse_indices = np.argsort(sort_indices)
    D_array = D_sorted[inverse_indices]
    
    # Normalize to D(z=0) = 1
    idx_z0 = np.argmin(np.abs(z_array))
    D_array = D_array / D_array[idx_z0]
    
    return D_array


def void_size_ratio(
    z: float,
    omega_m: Optional[float] = None,
    beta: float = 0.0
) -> float:
    """
    DEPRECATED: Use void_size_ratio_literature() instead.
    
    This function is kept for backward compatibility but is deprecated.
    The new literature-based calibration (Pisani+ 2015) is more rigorous
    and based on billion-particle N-body simulations.
    
    See: pipeline.cmb_gw.physics.void_scaling_literature.void_size_ratio_literature()
    
    Estimate ratio of void radius in evolving-G model to ΛCDM.
    
    Physical basis: Voids expand as surrounding matter flows out.
    The expansion rate relative to background depends on growth rate.
    Suppressed growth → voids expand more relative to ΛCDM.
    
    This is an approximation. Accurate void statistics require N-body simulations.
    
    Parameters:
    -----------
    z : float
        Redshift at which void forms
    omega_m : float, optional
        Matter density parameter. If None, uses HLCDM_PARAMS.OMEGA_M
    beta : float
        G evolution coupling strength
        
    Returns:
    --------
    float
        R_v(evolving G) / R_v(ΛCDM) ratio
        
    Notes:
    ------
    - First-order approximation: ΔR_v/R_v ≈ -Δδ_v/3 ≈ (1 - suppression)/3
    - Suppression = D_evolving / D_lcdm at formation redshift
    - This is approximate; full void statistics require N-body simulations
    - DEPRECATED: Use void_size_ratio_literature() for production analysis
    """
    import warnings
    warnings.warn(
        "void_size_ratio() is deprecated. Use void_size_ratio_literature() "
        "from pipeline.cmb_gw.physics.void_scaling_literature for production analysis.",
        DeprecationWarning,
        stacklevel=2
    )
    if omega_m is None:
        omega_m = HLCDM_PARAMS.OMEGA_M
    
    z_arr = np.array([z, 0.0])
    
    D_evolving = growth_factor_evolving_G(z_arr, omega_m, beta)
    D_lcdm = growth_factor_evolving_G(z_arr, omega_m, 0.0)
    
    # Growth suppression factor at formation redshift
    suppression = D_evolving[0] / D_lcdm[0]
    
    # Void radius scales inversely with growth (approximately)
    # R_v ∝ (1 - δ_v)^(1/3) where δ_v is void underdensity
    # Suppressed growth → less negative δ_v → larger R_v
    # First-order approximation: ΔR_v/R_v ≈ -Δδ_v/3 ≈ (1 - suppression)/3
    
    return 1.0 + (1.0 - suppression) / 3.0

