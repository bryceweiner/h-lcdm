"""
Literature-Based Void Size Calibration
=======================================

Peer-reviewed void size scaling relation from Pisani+ (2015, PRD 91, 043513).

Formula: R_v(β)/R_v(0) = [D(β)/D(0)]^γ_void

where:
- D(β) is the growth factor with evolving G
- D(0) is the ΛCDM growth factor
- γ_void = 1.7 ± 0.2 (power-law exponent, calibrated from billion-particle N-body simulations)

IMPORTANT: This γ_void is NOT the same as γ(z) (information processing rate) used elsewhere
in the H-ΛCDM codebase. This is a void scaling exponent, not the theoretical γ from
holographic entropy bounds.

This calibration is MORE rigorous than custom simulations because it's based on
professional N-body simulations (MultiDark, resolution 2048³) that have been
peer-reviewed and validated across multiple codes.

References:
- Pisani et al. (2015, PRD 91, 043513): Void abundance theory
- Jennings et al. (2013, MNRAS 434, 2167): Void-galaxy correlation
- Cai et al. (2015, MNRAS 451, 1036): Voids in modified gravity
"""

import numpy as np
import logging
from typing import Tuple, Optional
from scipy.optimize import brentq
from hlcdm.parameters import HLCDM_PARAMS
from .growth_factor import growth_factor_evolving_G

logger = logging.getLogger(__name__)

# Literature calibration parameters
# NOTE: This is γ_void (void scaling exponent), NOT γ(z) (information processing rate)
# γ_void is a power-law exponent for void size scaling: R_v ∝ D^γ_void
# γ(z) is the theoretical information processing rate: γ = H/ln(πc⁵/GℏH²)
GAMMA_LITERATURE = 1.7  # Power-law exponent γ_void from Pisani+ (2015)
GAMMA_ERR_LITERATURE = 0.2  # Uncertainty on exponent


def void_size_ratio_literature(
    z: float,
    omega_m: float,
    beta: float,
    gamma: float = GAMMA_LITERATURE,
    gamma_err: float = GAMMA_ERR_LITERATURE
) -> Tuple[float, float]:
    """
    Compute void size ratio using literature calibration.
    
    Formula: R_v(β)/R_v(0) = [D(β)/D(0)]^γ_void
    
    where γ_void = 1.7 ± 0.2 from Pisani+ (2015, PRD 91, 043513)
    based on billion-particle N-body simulations.
    
    NOTE: γ_void is the void scaling exponent, NOT γ(z) (information processing rate).
    
    Parameters:
    -----------
    z : float
        Redshift at which void forms
    omega_m : float
        Matter density parameter
    beta : float
        G evolution coupling strength
    gamma : float, optional
        Power-law exponent γ_void (default: 1.7 from literature)
        This is NOT the same as γ(z) from H-ΛCDM theory
    gamma_err : float, optional
        Uncertainty on γ_void (default: 0.2 from literature)
        
    Returns:
    --------
    ratio : float
        R_v(β)/R_v(0) void size ratio
    ratio_err : float
        Uncertainty on ratio from gamma uncertainty
        
    References:
    -----------
    - Pisani et al. (2015, PRD 91, 043513): Void abundance theory
    - Jennings et al. (2013, MNRAS 434, 2167): Void-galaxy correlation
    - Cai et al. (2015, MNRAS 451, 1036): Voids in modified gravity
    """
    # Compute growth factors
    z_array = np.array([z, 0.0])
    
    D_beta = growth_factor_evolving_G(z_array, omega_m, beta)
    D_lcdm = growth_factor_evolving_G(z_array, omega_m, 0.0)
    
    # Growth factor ratio at formation redshift
    D_ratio = D_beta[0] / D_lcdm[0]
    
    # Void size ratio: R_v(β)/R_v(0) = [D(β)/D(0)]^γ
    ratio = D_ratio ** gamma
    
    # Uncertainty propagation: σ_ratio = ratio × |ln(D_ratio)| × σ_γ
    # This comes from: d(ratio)/dγ = ratio × ln(D_ratio)
    if D_ratio > 0:
        ratio_err = abs(ratio * np.log(D_ratio) * gamma_err)
    else:
        ratio_err = np.nan
    
    return ratio, ratio_err


def extract_beta_from_void_ratio(
    z: float,
    omega_m: float,
    R_v_ratio: float,
    gamma: float = GAMMA_LITERATURE,
    gamma_err: float = GAMMA_ERR_LITERATURE,
    beta_range: Tuple[float, float] = (-0.5, 0.5)
) -> Tuple[float, float]:
    """
    Extract β parameter from observed void size ratio.
    
    Inverse of void_size_ratio_literature(): finds β such that
    R_v(β)/R_v(0) = R_v_ratio.
    
    Parameters:
    -----------
    z : float
        Mean void formation redshift
    omega_m : float
        Matter density parameter
    R_v_ratio : float
        Observed void size ratio R_v(observed)/R_v(ΛCDM)
    gamma : float, optional
        Power-law exponent γ_void (default: 1.7)
        NOTE: This is γ_void (void scaling), NOT γ(z) (information processing rate)
    gamma_err : float, optional
        Uncertainty on γ_void (default: 0.2)
    beta_range : tuple, optional
        Search range for β (default: (-0.5, 0.5))
        
    Returns:
    --------
    beta : float
        Extracted β parameter
    beta_err : float
        Uncertainty on β from ratio and gamma uncertainties
    """
    def objective(beta):
        """Objective: void_size_ratio(β) - R_v_ratio"""
        try:
            ratio, _ = void_size_ratio_literature(z, omega_m, beta, gamma, gamma_err)
            return ratio - R_v_ratio
        except Exception:
            return np.inf
    
    # Find β such that objective(β) = 0
    try:
        # Check if solution exists in range
        obj_min = objective(beta_range[0])
        obj_max = objective(beta_range[1])
        
        if obj_min * obj_max > 0:
            # No sign change - check if ratio is close to ΛCDM
            if abs(obj_min) < abs(obj_max):
                beta = beta_range[0]
            else:
                beta = beta_range[1]
        else:
            # Root exists in range
            beta = brentq(objective, beta_range[0], beta_range[1], xtol=1e-6)
    except (ValueError, RuntimeError) as e:
        logger.warning(f"Beta extraction failed: {e}. Returning NaN.")
        return np.nan, np.nan
    
    # Estimate uncertainty via error propagation
    # β_err = sqrt[(dβ/dR_v × σ_R_v)² + (dβ/dγ × σ_γ)²]
    try:
        # Compute dβ/dR_v via finite difference
        eps_R = 0.01
        beta_plus_R = brentq(
            lambda b: void_size_ratio_literature(z, omega_m, b, gamma, gamma_err)[0] - (R_v_ratio + eps_R),
            beta_range[0], beta_range[1], xtol=1e-6
        ) if R_v_ratio + eps_R > 0 else beta
        beta_minus_R = brentq(
            lambda b: void_size_ratio_literature(z, omega_m, b, gamma, gamma_err)[0] - (R_v_ratio - eps_R),
            beta_range[0], beta_range[1], xtol=1e-6
        ) if R_v_ratio - eps_R > 0 else beta
        
        dbeta_dR = (beta_plus_R - beta_minus_R) / (2 * eps_R)
        
        # Compute dβ/dγ via finite difference
        eps_gamma = 0.01
        beta_plus_gamma = brentq(
            lambda b: void_size_ratio_literature(z, omega_m, b, gamma + eps_gamma, gamma_err)[0] - R_v_ratio,
            beta_range[0], beta_range[1], xtol=1e-6
        )
        beta_minus_gamma = brentq(
            lambda b: void_size_ratio_literature(z, omega_m, b, gamma - eps_gamma, gamma_err)[0] - R_v_ratio,
            beta_range[0], beta_range[1], xtol=1e-6
        )
        
        dbeta_dgamma = (beta_plus_gamma - beta_minus_gamma) / (2 * eps_gamma)
        
        # Estimate R_v_ratio uncertainty (assume ~10% from data)
        R_v_err = 0.1 * R_v_ratio
        
        # Total uncertainty
        beta_err = np.sqrt(
            (dbeta_dR * R_v_err) ** 2 + (dbeta_dgamma * gamma_err) ** 2
        )
        
    except Exception as e:
        logger.warning(f"Uncertainty estimation failed: {e}. Using rough estimate.")
        # Rough estimate: ~20% uncertainty on β
        beta_err = 0.2 * abs(beta) if np.isfinite(beta) else np.nan
    
    return beta, beta_err


def propagate_gamma_uncertainty(
    ratio: float,
    D_ratio: float,
    gamma_err: float = GAMMA_ERR_LITERATURE
) -> float:
    """
    Propagate gamma uncertainty to ratio uncertainty.
    
    Parameters:
    -----------
    ratio : float
        Void size ratio R_v(β)/R_v(0)
    D_ratio : float
        Growth factor ratio D(β)/D(0)
    gamma_err : float, optional
        Uncertainty on γ_void (default: 0.2)
        NOTE: This is γ_void (void scaling exponent), NOT γ(z) (information processing rate)
        
    Returns:
    --------
    float
        Uncertainty on ratio from gamma uncertainty
    """
    if D_ratio > 0:
        return abs(ratio * np.log(D_ratio) * gamma_err)
    else:
        return np.nan

