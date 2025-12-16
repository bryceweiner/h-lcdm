"""
Cosmological Constant Physics Calculations
==========================================

Implements the causal diamond triality calculation from cosmological_constant_resolution.tex.

The calculation proceeds from two information-theoretic principles:
1. Dimension-weighted entropy of causal diamond triality (S_geom)
2. Irreversible precipitation fraction (f_irrev)

Result: Ω_Λ = S_geom × f_irrev = 0.6841 (parameter-free prediction)
"""

import numpy as np
from typing import Dict, Any
from hlcdm.parameters import HLCDM_PARAMS


def calculate_geometric_entropy() -> Dict[str, Any]:
    """
    Calculate dimension-weighted geometric entropy S_geom.
    
    From cosmological_constant_resolution.tex Eqs. 75-98:
    - Causal diamond has three sectors: N^+ (3D), N^- (3D), σ (2D)
    - Dimension weights: w(N^+) = w(N^-) = 3, w(σ) = 2
    - Total weight: w_tot = 8
    - Probabilities: p(N^+) = p(N^-) = 3/8, p(σ) = 1/4
    - Shannon entropy: S_geom = -Σ p_i ln(p_i)
    
    Returns:
        dict: Geometric entropy calculation with intermediate steps
    """
    # Dimension weights (Eq. 68-70)
    w_N_plus = 3.0  # 3D null hypersurface (future null cone)
    w_N_minus = 3.0  # 3D null hypersurface (past null cone)
    w_sigma = 2.0  # 2D spacelike surface (holographic screen)
    
    w_tot = w_N_plus + w_N_minus + w_sigma  # = 8
    
    # Normalized probabilities (Eq. 75-76)
    p_N_plus = w_N_plus / w_tot  # = 3/8
    p_N_minus = w_N_minus / w_tot  # = 3/8
    p_sigma = w_sigma / w_tot  # = 1/4
    
    # Shannon entropy calculation (Eq. 82-98)
    # S_geom = -2 × (3/8)ln(3/8) - (1/4)ln(1/4)
    # Expanding: -3/8(ln 3 - 3ln 2) = -3/8 ln 3 + 9/8 ln 2
    # Final: S_geom = (11 ln 2 - 3 ln 3)/4
    
    term_N = -p_N_plus * np.log(p_N_plus)  # = -3/8 ln(3/8)
    term_sigma = -p_sigma * np.log(p_sigma)  # = -1/4 ln(1/4)
    
    S_geom = 2 * term_N + term_sigma  # Two null cones + screen
    
    # Closed form: (11 ln 2 - 3 ln 3)/4
    S_geom_closed = (11 * np.log(2) - 3 * np.log(3)) / 4
    
    # Verify numerical agreement
    assert np.abs(S_geom - S_geom_closed) < 1e-10, "Geometric entropy calculation mismatch"
    
    return {
        'S_geom': float(S_geom),
        'S_geom_numerical': float(S_geom),
        'S_geom_closed_form': float(S_geom_closed),
        'dimension_weights': {
            'w_N_plus': w_N_plus,
            'w_N_minus': w_N_minus,
            'w_sigma': w_sigma,
            'w_tot': w_tot
        },
        'probabilities': {
            'p_N_plus': float(p_N_plus),
            'p_N_minus': float(p_N_minus),
            'p_sigma': float(p_sigma)
        },
        'entropy_terms': {
            'term_N': float(term_N),
            'term_sigma': float(term_sigma)
        }
    }


def calculate_irreversibility_fraction() -> Dict[str, Any]:
    """
    Calculate irreversibility fraction f_irrev from Poisson decoherence.
    
    From cosmological_constant_resolution.tex Eqs. 125-127:
    - Decoherence rate: Γ_decoh = H (Hubble parameter)
    - Precipitation probability: P_classical(t) = 1 - exp(-Γ t)
    - At Hubble timescale t_H = H^{-1}: f_irrev = 1 - exp(-1)
    
    Returns:
        dict: Irreversibility fraction calculation
    """
    # Decoherence rate (Eq. 111)
    # Γ_decoh = H (set by cosmological horizon)
    
    # Precipitation probability at Hubble timescale (Eq. 125-127)
    # f_irrev = P_classical(t_H) = 1 - exp(-H × H^{-1}) = 1 - exp(-1)
    f_irrev = 1 - np.exp(-1)
    
    return {
        'f_irrev': float(f_irrev),
        'decoherence_rate': 'H',  # Hubble parameter
        'timescale': 't_H = H^{-1}',
        'formula': '1 - exp(-1)'
    }


def calculate_omega_lambda() -> Dict[str, Any]:
    """
    Calculate predicted dark energy fraction Ω_Λ.
    
    From cosmological_constant_resolution.tex Eq. 145-147:
    Ω_Λ = S_geom × f_irrev
    
    This is the primary parameter-free prediction of the framework.
    
    Returns:
        dict: Complete Ω_Λ calculation with all components
    """
    # Get component calculations
    geom_result = calculate_geometric_entropy()
    irrev_result = calculate_irreversibility_fraction()
    
    S_geom = geom_result['S_geom']
    f_irrev = irrev_result['f_irrev']
    
    # Primary prediction (Eq. 145-147)
    Omega_Lambda = S_geom * f_irrev
    
    return {
        'omega_lambda': float(Omega_Lambda),
        'S_geom': S_geom,
        'f_irrev': f_irrev,
        'formula': 'S_geom × f_irrev',
        'components': {
            'geometric_entropy': geom_result,
            'irreversibility_fraction': irrev_result
        }
    }


def calculate_lambda(H0: float = None) -> Dict[str, Any]:
    """
    Calculate cosmological constant Λ from Ω_Λ prediction.
    
    From cosmological_constant_resolution.tex Eq. 153-160:
    Λ = 3Ω_Λ H²/c²
    
    Parameters:
        H0 (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
    
    Returns:
        dict: Cosmological constant calculation
    """
    if H0 is None:
        H0 = HLCDM_PARAMS.H0
    
    # Get Ω_Λ prediction
    omega_result = calculate_omega_lambda()
    Omega_Lambda = omega_result['omega_lambda']
    
    # Calculate Λ (Eq. 153-160)
    # Λ = 3Ω_Λ H²/c²
    Lambda = 3 * Omega_Lambda * H0**2 / HLCDM_PARAMS.C**2
    
    return {
        'lambda': float(Lambda),
        'lambda_units': 'm^-2',
        'omega_lambda': Omega_Lambda,
        'H0': float(H0),
        'c': float(HLCDM_PARAMS.C),
        'formula': '3Ω_Λ H²/c²',
        'omega_lambda_calculation': omega_result
    }
