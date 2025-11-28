"""
H-ΛCDM Integration for H-ZOBOV
==============================

Wrapper module for accessing holographic cosmological constant Λ(z) from
first principles using the existing hlcdm/cosmology.py implementation.

This module provides redshift-dependent Lambda calculations for void significance
evaluation and zone merging criteria in the H-ZOBOV algorithm.
"""

import numpy as np
from typing import Dict, Any, Optional, Union
import logging

from hlcdm.cosmology import HLCDMCosmology

logger = logging.getLogger(__name__)


class HZOBOVLambdaError(Exception):
    """Error in Lambda calculation for H-ZOBOV."""
    pass


def get_lambda_at_redshift(z: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calculate holographic cosmological constant Λ(z) at given redshift(s).
    
    Uses the complete H-ΛCDM calculation from cosmological_constant_resolution.tex Eq. 208:
    
    Λ(z) = (8πG/c²) × ρ_P × (γ(z) × t_P)² × (S_coh/|S_decoh|) × f_quantum × f_geometric
    
    Where:
    - γ(z) = H(z)/π² (information processing rate at redshift z)
    - H(z) = H₀ × √(Ω_m(1+z)³ + Ω_Λ) (Hubble parameter evolution)
    - S_coh/|S_decoh| = ln(2)/(1-ln(2)) ≈ 2.257 (QTEP ratio)
    
    Parameters:
        z: Redshift(s) - can be scalar or array
        
    Returns:
        Lambda value(s) in m⁻² - same shape as input z
        
    Raises:
        HZOBOVLambdaError: If calculation fails or invalid redshift provided
    """
    try:
        # Handle both scalar and array inputs
        is_scalar = np.isscalar(z)
        z_array = np.atleast_1d(z)
        
        # Validate redshifts
        if np.any(z_array < 0):
            raise HZOBOVLambdaError(f"Invalid redshift(s): {z_array[z_array < 0]}. Redshift must be >= 0.")
        
        # Calculate Lambda for each redshift
        lambda_values = []
        for z_val in z_array:
            lambda_dict = HLCDMCosmology.lambda_evolution(float(z_val))
            lambda_values.append(lambda_dict['lambda_theoretical'])
        
        lambda_array = np.array(lambda_values)
        
        # Return scalar if input was scalar
        if is_scalar:
            return float(lambda_array[0])
        return lambda_array
        
    except Exception as e:
        raise HZOBOVLambdaError(f"Failed to calculate Lambda at redshift {z}: {e}") from e


def get_lambda_evolution(z_array: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Calculate complete Lambda evolution information for array of redshifts.
    
    Parameters:
        z_array: Array of redshifts
        
    Returns:
        Dictionary with:
        - 'lambda': Lambda values in m⁻²
        - 'rho_lambda': Vacuum energy density in kg/m³
        - 'gamma': Information processing rate γ(z) in s⁻¹
        - 'hubble': Hubble parameter H(z) in s⁻¹
        - 'redshift': Input redshifts
    """
    try:
        z_array = np.atleast_1d(z_array)
        
        if np.any(z_array < 0):
            raise HZOBOVLambdaError(f"Invalid redshifts: {z_array[z_array < 0]}")
        
        lambda_vals = []
        rho_vals = []
        gamma_vals = []
        hubble_vals = []
        
        for z_val in z_array:
            lambda_dict = HLCDMCosmology.lambda_evolution(float(z_val))
            lambda_vals.append(lambda_dict['lambda_theoretical'])
            rho_vals.append(lambda_dict['rho_lambda'])
            gamma_vals.append(lambda_dict['gamma_fundamental'])
            hubble_vals.append(lambda_dict['hubble_parameter'])
        
        return {
            'lambda': np.array(lambda_vals),
            'rho_lambda': np.array(rho_vals),
            'gamma': np.array(gamma_vals),
            'hubble': np.array(hubble_vals),
            'redshift': z_array
        }
        
    except Exception as e:
        raise HZOBOVLambdaError(f"Failed to calculate Lambda evolution: {e}") from e


def compare_lambda_models(z: Union[float, np.ndarray], 
                          lambda_constant: Optional[float] = None) -> Dict[str, Any]:
    """
    Compare H-ΛCDM Lambda(z) with constant Lambda model.
    
    Parameters:
        z: Redshift(s) for comparison
        lambda_constant: Constant Lambda value (default: observed Λ₀ ≈ 1.1×10⁻⁵² m⁻²)
        
    Returns:
        Dictionary with comparison statistics
    """
    from hlcdm.parameters import HLCDM_PARAMS
    
    if lambda_constant is None:
        lambda_constant = HLCDM_PARAMS.LAMBDA_OBS
    
    is_scalar = np.isscalar(z)
    z_array = np.atleast_1d(z)
    
    lambda_hzobov = get_lambda_at_redshift(z_array)
    
    # Calculate relative difference
    relative_diff = (lambda_hzobov - lambda_constant) / lambda_constant
    
    result = {
        'redshift': z_array,
        'lambda_hzobov': lambda_hzobov,
        'lambda_constant': lambda_constant,
        'relative_difference': relative_diff,
        'absolute_difference': lambda_hzobov - lambda_constant,
    }
    
    if is_scalar:
        # Return scalar values for scalar input
        for key in ['lambda_hzobov', 'lambda_constant', 'relative_difference', 'absolute_difference']:
            if isinstance(result[key], np.ndarray):
                result[key] = float(result[key][0])
    
    return result

