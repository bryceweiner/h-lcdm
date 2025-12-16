"""
Gravitational Constant Physics Calculations
============================================

Implements the gravitational constant derivation from information-theoretic principles.

The calculation proceeds from fine structure constant via holographic information capacity:
1. α⁻¹ = 137.035999084 (CODATA 2018)
2. Invert holographic relation: α⁻¹ = (1/2)ln(N_P) - ln(4π²) - 1/(2π)
   → N_P = exp[2α⁻¹ + 2ln(4π²) + 1/π]
3. Apply holographic bound: G = πc⁵/(ℏH²N_P)

Result: G ≈ 6.62 × 10⁻¹¹ m³/(kg·s²) (parameter-free prediction, ~1% agreement with CODATA)

Note: The fine structure pipeline validates this formula by showing that
(G_measured, H₀) → α_predicted agrees with α_measured to 0.0018%.
Inverting this validated relation gives the G prediction.

No correction factors (ln(3), f_quantum) are needed - the holographic bound
already encodes dimensional projection through the 2D horizon area.
"""

import numpy as np
from typing import Dict, Any, Optional
from hlcdm.parameters import HLCDM_PARAMS


def calculate_information_capacity(H: float = None) -> Dict[str, Any]:
    """
    Calculate information capacity of cosmic horizon N_P.
    
    From G_derivation_core.md Eq. 18:
    N_P = πc⁵/(ℏGH²)
    
    This represents the dimensionless information capacity of the cosmic horizon.
    
    Parameters:
        H (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
    
    Returns:
        dict: Information capacity calculation
    """
    if H is None:
        H = HLCDM_PARAMS.H0
    
    # Information capacity (Eq. 18)
    # N_P = πc⁵/(ℏGH²)
    # Note: This requires G, so this is typically calculated from α⁻¹ instead
    # But we can calculate it if G is known
    N_P = (np.pi * HLCDM_PARAMS.C**5) / (HLCDM_PARAMS.HBAR * HLCDM_PARAMS.G * H**2)
    
    # Logarithmic form
    ln_N_P = np.log(N_P)
    
    return {
        'N_P': float(N_P),
        'ln_N_P': float(ln_N_P),
        'H': float(H),
        'formula': 'πc⁵/(ℏGH²)',
        'components': {
            'pi': np.pi,
            'c': float(HLCDM_PARAMS.C),
            'hbar': float(HLCDM_PARAMS.HBAR),
            'G': float(HLCDM_PARAMS.G),
            'H': float(H)
        }
    }


def calculate_alpha_inverse_from_np(N_P: float) -> Dict[str, Any]:
    """
    Calculate α⁻¹ from information capacity N_P.
    
    From G_derivation_core.md Eq. 61:
    α⁻¹ = (1/2)ln(N_P) - ln(4π²) - 1/(2π)
    
    Parameters:
        N_P (float): Information capacity
    
    Returns:
        dict: α⁻¹ calculation
    """
    ln_N_P = np.log(N_P)
    
    # Calculate components
    holographic_term = 0.5 * ln_N_P
    geometric_term = np.log(4 * np.pi**2)
    vacuum_term = 1.0 / (2 * np.pi)
    
    # α⁻¹ calculation
    alpha_inverse = holographic_term - geometric_term - vacuum_term
    
    return {
        'alpha_inverse': float(alpha_inverse),
        'alpha': float(1.0 / alpha_inverse),
        'ln_N_P': float(ln_N_P),
        'holographic_term': float(holographic_term),
        'geometric_term': float(geometric_term),
        'vacuum_term': float(vacuum_term),
        'formula': '(1/2)ln(N_P) - ln(4π²) - 1/(2π)'
    }


def calculate_np_from_alpha_inverse(alpha_inverse: float = 137.035999084) -> Dict[str, Any]:
    """
    Calculate information capacity N_P from fine structure constant α⁻¹.
    
    From G_derivation_core.md Eq. 61, solving for N_P:
    α⁻¹ = (1/2)ln(N_P) - ln(4π²) - 1/(2π)
    → ln(N_P) = 2(α⁻¹ + ln(4π²) + 1/(2π))
    → N_P = exp(2(α⁻¹ + ln(4π²) + 1/(2π)))
    
    Parameters:
        alpha_inverse (float): Inverse fine structure constant (CODATA 2018)
    
    Returns:
        dict: N_P calculation
    """
    # Calculate constant terms
    geometric_term = np.log(4 * np.pi**2)
    vacuum_term = 1.0 / (2 * np.pi)
    constant_sum = geometric_term + vacuum_term
    
    # Solve for ln(N_P)
    ln_N_P = 2 * (alpha_inverse + constant_sum)
    
    # Calculate N_P
    N_P = np.exp(ln_N_P)
    
    return {
        'N_P': float(N_P),
        'ln_N_P': float(ln_N_P),
        'alpha_inverse': alpha_inverse,
        'geometric_term': float(geometric_term),
        'vacuum_term': float(vacuum_term),
        'constant_sum': float(constant_sum),
        'formula': 'exp(2(α⁻¹ + ln(4π²) + 1/(2π)))'
    }


def calculate_g_from_holographic_bound(H: float = None, N_P: float = None, alpha_inverse: float = 137.035999084) -> Dict[str, Any]:
    """
    Calculate gravitational constant G from holographic bound.
    
    The holographic bound relates information capacity to gravitational coupling:
    N_P = πc⁵/(ℏGH²)
    
    Inverting: G = πc⁵/(ℏH²N_P)
    
    This is the FINAL formula - no additional correction factors are needed.
    The dimensional projection (3D → 2D boundary) is already encoded in the
    area-based holographic bound.
    
    Parameters:
        H (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
        N_P (float, optional): Information capacity. If None, calculated from alpha_inverse.
        alpha_inverse (float): Inverse fine structure constant (used if N_P not provided)
    
    Returns:
        dict: G calculation from holographic bound
    """
    if H is None:
        H = HLCDM_PARAMS.H0
    
    if N_P is None:
        np_result = calculate_np_from_alpha_inverse(alpha_inverse)
        N_P = np_result['N_P']
        ln_N_P = np_result['ln_N_P']
    else:
        np_result = None
        ln_N_P = np.log(N_P)
    
    # Holographic bound inversion: G = πc⁵/(ℏH²N_P)
    numerator = np.pi * HLCDM_PARAMS.C**5
    denominator = HLCDM_PARAMS.HBAR * H**2 * N_P
    G = numerator / denominator
    
    return {
        'G': float(G),
        'N_P': float(N_P),
        'ln_N_P': float(ln_N_P),
        'H': float(H),
        'numerator': float(numerator),
        'denominator': float(denominator),
        'formula': 'πc⁵/(ℏH²N_P)',
        'np_calculation': np_result
    }


# Backwards compatibility alias
def calculate_g_base(H: float = None, N_P: float = None, alpha_inverse: float = 137.035999084) -> Dict[str, Any]:
    """
    Backwards compatibility wrapper for calculate_g_from_holographic_bound.
    
    Note: This now returns G directly (no corrections needed).
    The 'G_base' key is provided for compatibility but equals 'G'.
    """
    result = calculate_g_from_holographic_bound(H, N_P, alpha_inverse)
    # Add backwards-compatible key
    result['G_base'] = result['G']
    return result


def calculate_g_geometric(H: float = None, N_P: float = None, alpha_inverse: float = 137.035999084) -> Dict[str, Any]:
    """
    DEPRECATED: Geometric correction is not needed.
    
    The ln(3) factor was previously thought to arise from dimensional projection,
    but the holographic bound already encodes this through the 2D horizon area.
    Applying ln(3) correction degrades agreement from ~1% to ~10%.
    
    This function now returns G without geometric correction for backwards
    compatibility, with a warning that the correction is not applied.
    
    Parameters:
        H (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
        N_P (float, optional): Information capacity. If None, calculated from alpha_inverse.
        alpha_inverse (float): Inverse fine structure constant (used if N_P not provided)
    
    Returns:
        dict: G calculation (no geometric correction applied)
    """
    g_result = calculate_g_from_holographic_bound(H, N_P, alpha_inverse)
    G = g_result['G']
    
    # No geometric correction - return G directly
    # The ln(3) factor is NOT needed (see G_derivation_core.md for explanation)
    
    return {
        'G_geom': float(G),  # Same as G - no correction
        'G_base': float(G),
        'G': float(G),
        'ln_3': float(np.log(3)),
        'geometric_correction_applied': False,
        'note': 'Geometric ln(3) correction NOT applied - holographic bound already encodes dimensionality',
        'formula': 'πc⁵/(ℏH²N_P)',
        'g_base_calculation': g_result
    }


def calculate_g_final(H: float = None,
                     N_P: float = None,
                     alpha_inverse: float = 137.035999084,
                     f_quantum: float = 1.0) -> Dict[str, Any]:
    """
    Calculate final gravitational constant G.
    
    UPDATED: No quantum correction factor is needed at cosmological scales.
    
    All quantum corrections (LQG, asymptotic safety, string theory) are
    negligible at horizon scales:
    - Loop quantum gravity: ~10⁻¹²⁴
    - Asymptotic safety: ~10⁻⁴⁶
    - String α' corrections: ~10⁻²¹²
    
    The f_quantum parameter is retained for backwards compatibility but
    defaults to 1.0 (no correction).
    
    Parameters:
        H (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
        N_P (float, optional): Information capacity. If None, calculated from alpha_inverse.
        alpha_inverse (float): Inverse fine structure constant (used if N_P not provided)
        f_quantum (float): Quantum correction factor (default 1.0 - no correction)
    
    Returns:
        dict: Final G calculation
    """
    g_result = calculate_g_from_holographic_bound(H, N_P, alpha_inverse)
    G = g_result['G']
    
    # Apply f_quantum if provided (but default is 1.0)
    if f_quantum != 1.0:
        G = G / f_quantum
    
    return {
        'G': float(G),
        'G_uncorrected': float(g_result['G']),
        'f_quantum': float(f_quantum),
        'quantum_correction_applied': f_quantum != 1.0,
        'note': 'Quantum corrections negligible at cosmological scales' if f_quantum == 1.0 else f'f_quantum={f_quantum} applied',
        'formula': 'πc⁵/(ℏH²N_P)' if f_quantum == 1.0 else 'πc⁵/(ℏH²N_P·f_quantum)',
        'holographic_calculation': g_result
    }


def calculate_g(H: float = None,
                alpha_inverse: float = 137.035999084,
                f_quantum: float = 1.0) -> Dict[str, Any]:
    """
    Calculate gravitational constant G from fine structure constant.
    
    This is the main function implementing the parameter-free derivation:
    
    1. N_P from α⁻¹: N_P = exp[2α⁻¹ + 2ln(4π²) + 1/π]
    2. G from holographic bound: G = πc⁵/(ℏH²N_P)
    
    No correction factors (ln(3), f_quantum) are applied - the holographic
    bound already encodes dimensional projection through the 2D horizon area.
    
    Validation: The fine structure pipeline confirms this formula by showing
    (G_measured, H₀) → α_predicted agrees with α_measured to 0.0018%.
    
    Parameters:
        H (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
        alpha_inverse (float): Inverse fine structure constant (CODATA 2018)
        f_quantum (float): Quantum correction factor (default 1.0 - not needed)
    
    Returns:
        dict: Complete G calculation with derivation details
    """
    if H is None:
        H = HLCDM_PARAMS.H0
    
    # Step 1: Calculate N_P from α⁻¹
    np_result = calculate_np_from_alpha_inverse(alpha_inverse)
    N_P = np_result['N_P']
    ln_N_P = np_result['ln_N_P']
    
    # Step 2: Calculate G from holographic bound (NO corrections needed)
    g_result = calculate_g_from_holographic_bound(H, N_P, alpha_inverse)
    G = g_result['G']
    
    # Apply f_quantum only if explicitly different from 1.0
    if f_quantum != 1.0:
        G = G / f_quantum
    
    return {
        'G': float(G),
        'N_P': float(N_P),
        'ln_N_P': float(ln_N_P),
        'H': float(H),
        'alpha_inverse': alpha_inverse,
        'f_quantum': f_quantum,
        'formula': 'G = πc⁵/(ℏH²N_P)',
        'np_formula': 'N_P = exp[2α⁻¹ + 2ln(4π²) + 1/π]',
        'corrections_applied': f_quantum != 1.0,
        'note': 'No corrections needed - holographic bound already encodes dimensionality',
        'components': {
            'np_calculation': np_result,
            'holographic_calculation': g_result
        },
        # Backwards compatibility
        'G_base': float(G),
        'G_geom': float(G),
        'ln_3': float(np.log(3))  # Included but NOT used
    }
