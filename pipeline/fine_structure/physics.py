"""
Fine Structure Constant Physics Calculations
============================================

Implements the fine structure constant derivation from fine_structure_derivation.tex.

The calculation proceeds from information processing constraints:
1. Bekenstein-Hawking entropy of causal horizon: S_H = πc⁵/(ℏGH²)
2. Information processing rate: γ = H/ln(S_H)
3. Inverse fine structure constant: α⁻¹ = (1/2)ln(S_H) - ln(4π²) - 1/(2π)

Result: α⁻¹ = 137.032 (parameter-free prediction)
"""

import numpy as np
from typing import Dict, Any
from hlcdm.parameters import HLCDM_PARAMS


def calculate_bekenstein_hawking_entropy(H: float = None) -> Dict[str, Any]:
    """
    Calculate Bekenstein-Hawking entropy of causal horizon.
    
    From fine_structure_derivation.tex Eq. 129:
    S_H = πc⁵/(ℏGH²)
    
    This represents the maximum information capacity of the causal horizon.
    
    Parameters:
        H (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
    
    Returns:
        dict: Bekenstein-Hawking entropy calculation
    """
    if H is None:
        H = HLCDM_PARAMS.H0
    
    # Bekenstein-Hawking entropy (Eq. 129)
    # S_H = πc⁵/(ℏGH²)
    S_H = (np.pi * HLCDM_PARAMS.C**5) / (HLCDM_PARAMS.HBAR * HLCDM_PARAMS.G * H**2)
    
    # Logarithmic information content
    ln_S_H = np.log(S_H)
    
    return {
        'S_H': float(S_H),
        'ln_S_H': float(ln_S_H),
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


def calculate_information_processing_rate(H: float = None) -> Dict[str, Any]:
    """
    Calculate fundamental information processing rate γ.
    
    From fine_structure_derivation.tex Eq. 81:
    γ = H/ln(S_H)
    
    This represents the maximum frequency of discrete information processing
    events allowed by the Bekenstein bound.
    
    Parameters:
        H (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
    
    Returns:
        dict: Information processing rate calculation
    """
    if H is None:
        H = HLCDM_PARAMS.H0
    
    # Get Bekenstein-Hawking entropy
    entropy_result = calculate_bekenstein_hawking_entropy(H)
    ln_S_H = entropy_result['ln_S_H']
    
    # Information processing rate (Eq. 81)
    # γ = H/ln(S_H)
    gamma = H / ln_S_H
    
    return {
        'gamma': float(gamma),
        'H': float(H),
        'ln_S_H': ln_S_H,
        'formula': 'H/ln(S_H)',
        'entropy_calculation': entropy_result
    }


def calculate_alpha_inverse(H: float = None) -> Dict[str, Any]:
    """
    Calculate inverse fine structure constant α⁻¹.
    
    From fine_structure_derivation.tex Eq. 93:
    α⁻¹ = (1/2)ln(S_H) - ln(4π²) - 1/(2π)
    
    This is the primary parameter-free prediction of the framework.
    
    Parameters:
        H (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
    
    Returns:
        dict: Complete α⁻¹ calculation with all components
    """
    if H is None:
        H = HLCDM_PARAMS.H0
    
    # Get Bekenstein-Hawking entropy
    entropy_result = calculate_bekenstein_hawking_entropy(H)
    ln_S_H = entropy_result['ln_S_H']
    
    # Get information processing rate
    gamma_result = calculate_information_processing_rate(H)
    gamma = gamma_result['gamma']
    
    # Calculate components (Eq. 93)
    # α⁻¹ = (1/2)ln(S_H) - ln(4π²) - 1/(2π)
    
    # Component 1: Holographic information content
    holographic_term = 0.5 * ln_S_H
    
    # Component 2: Geometric phase space
    geometric_term = np.log(4 * np.pi**2)
    
    # Component 3: Vacuum topology
    vacuum_term = 1.0 / (2 * np.pi)
    
    # Primary prediction
    alpha_inverse = holographic_term - geometric_term - vacuum_term
    
    # Alternative form using γ (Eq. 109)
    # α⁻¹ = H/(2γ) - ln(4π²) - 1/(2π)
    alpha_inverse_alt = (H / (2 * gamma)) - geometric_term - vacuum_term
    
    # Verify both forms agree
    assert np.abs(alpha_inverse - alpha_inverse_alt) < 1e-10, "Alpha inverse calculation mismatch"
    
    return {
        'alpha_inverse': float(alpha_inverse),
        'alpha': float(1.0 / alpha_inverse),
        'holographic_term': float(holographic_term),
        'geometric_term': float(geometric_term),
        'vacuum_term': float(vacuum_term),
        'formula': '(1/2)ln(S_H) - ln(4π²) - 1/(2π)',
        'alternative_formula': 'H/(2γ) - ln(4π²) - 1/(2π)',
        'components': {
            'bekenstein_hawking_entropy': entropy_result,
            'information_processing_rate': gamma_result,
            'holographic_term': holographic_term,
            'geometric_term': geometric_term,
            'vacuum_term': vacuum_term
        }
    }


def calculate_alpha(H: float = None) -> Dict[str, Any]:
    """
    Calculate fine structure constant α.
    
    Parameters:
        H (float, optional): Hubble parameter in s^-1. Defaults to HLCDM_PARAMS.H0.
    
    Returns:
        dict: Fine structure constant calculation
    """
    alpha_inv_result = calculate_alpha_inverse(H)
    
    return {
        'alpha': alpha_inv_result['alpha'],
        'alpha_inverse': alpha_inv_result['alpha_inverse'],
        'alpha_inverse_calculation': alpha_inv_result
    }
