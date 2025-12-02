"""
Moiré Physics Engine
====================

Models moiré superlattice effects in twisted TMDC layers.
"""

import numpy as np
from typing import Dict, Any

from hlcdm.parameters import HLCDM_PARAMS
from hlcdm.tmdc_parameters import WSe2Parameters

# Physical Constants for WSe2 (from tmdc_parameters.py)
LATTICE_CONSTANT_A = WSe2Parameters.LATTICE_CONSTANT_A_NM
INTERLAYER_DISTANCE_D = WSe2Parameters.INTERLAYER_DISTANCE_D_NM
VDW_COUPLING_GAMMA0 = WSe2Parameters.VDW_COUPLING_GAMMA0_EV
EFFECTIVE_MASS_M_STAR = WSe2Parameters.EFFECTIVE_MASS


def _calculate_commensurability_factor(theta_rad: float) -> float:
    """
    Calculate commensurability modulation factor.
    
    Peaks at angles corresponding to commensurate superlattices (m, n).
    cos(theta) = (m^2 + 4mn + n^2) / (2(m^2 + mn + n^2))
    
    Args:
        theta_rad: Twist angle in radians
        
    Returns:
        float: Modulation factor (1.0 = baseline, >1.0 = resonant)
    """
    theta_deg = np.degrees(theta_rad)
    
    # Known commensurate angles for hexagonal lattice (up to m=1, n=50)
    # For WSe2, small angles 1-3 deg are relevant
    commensurate_angles = [
        # (n, 1) series for small angles
        21.79, 13.17, 9.43, 7.34, 6.01, 5.09, 4.41, 3.89, 3.48, 3.15, # n=2..11
        2.88, 2.65, 2.45, 2.28, 2.13, 2.00, 1.89, 1.79, 1.70, 1.61, # n=12..21
        1.54, 1.47, 1.41, 1.35, 1.30, 1.25, 1.20, 1.16, 1.12, 1.08  # n=22..31
    ]
    
    # Base modulation
    modulation = 1.0
    
    # Add Lorentzian peaks at commensurate angles
    width = 0.05  # Width of resonance in degrees
    
    for angle in commensurate_angles:
        # Lorentzian: 1 / (1 + ((x-x0)/(width/2))^2)
        # Amplitude scales with 1/angle (smaller angles = larger supercells = stronger moire potential)
        amplitude = 0.5 / (angle + 0.1) 
        
        delta = theta_deg - angle
        peak = amplitude / (1.0 + (delta / (width / 2.0))**2)
        modulation += peak
        
    return modulation

def calculate_moire_properties(theta_i: float, theta_j: float, material: str = 'WSe2') -> Dict[str, float]:
    """
    Calculate moiré superlattice properties between layers i and j.

    Args:
        theta_i: Twist angle of layer i (radians)
        theta_j: Twist angle of layer j (radians)
        material: TMDC material type (default: 'WSe2')

    Returns:
        dict: {
            'period': moiré period (nm),
            'coupling_strength': interlayer coupling (eV),
            'band_offset': energy offset (eV)
        }
    """
    # Validate material
    if material != 'WSe2':
        # Allow generic TMDC but warn? Or just use params
        pass 

    delta_theta = abs(theta_i - theta_j)
    
    # Avoid division by zero for identical angles
    if delta_theta < 1e-10:
        moire_period = 1e9 # Very large number
    else:
        # Moiré period: L_moiré = a / (2 * sin(|θ_i - θ_j| / 2))
        moire_period = LATTICE_CONSTANT_A / (2 * np.sin(delta_theta / 2))

    # Model interlayer van der Waals coupling
    reference_coupling = VDW_COUPLING_GAMMA0
    
    if moire_period > 1e6:  # Effectively infinite (aligned layers)
        coupling_strength = VDW_COUPLING_GAMMA0 * 0.01  # Very weak coupling
    else:
        # Magic angle coupling model (updated for WSe2)
        # Primary peak at ~1.2° (Devakul et al.)
        # Broad flat-band window 1° - 3° (An et al., Zhang et al.)
        
        magic_angle_deg = WSe2Parameters.MAGIC_ANGLE_PRIMARY_DEG
        fwhm_deg = 0.4 # Narrower peak for primary magic angle
        sigma = fwhm_deg / 2.355
        
        # Calculate effective angle due to lattice relaxation
        # Small angles relax significantly
        delta_theta_eff = calculate_lattice_relaxation(delta_theta)
        delta_theta_deg_eff = np.degrees(delta_theta_eff)
        
        # Gaussian peak at magic angle
        magic_peak = np.exp(-0.5 * ((delta_theta_deg_eff - magic_angle_deg) / sigma)**2)
        
        # Broader flat-band window envelope (1-3 deg)
        # Model as a super-Gaussian or wider Gaussian centered at 2.0
        flat_band_center = 2.0
        flat_band_sigma = 1.0 
        flat_band_envelope = 0.5 * np.exp(-0.5 * ((delta_theta_deg_eff - flat_band_center) / flat_band_sigma)**4)
        
        # Combined coupling modulation
        # Peak resonance + broad window support
        coupling_strength = reference_coupling * (0.1 + 0.8 * magic_peak + 0.4 * flat_band_envelope)
        # Cap at sensible max
        coupling_strength = min(coupling_strength, reference_coupling * 2.0)

    # Band offset
    # Strain approx proportional to theta^2 for small angles
    strain_energy = 0.05 * (delta_theta ** 2) # eV, phenomenological
    band_offset = strain_energy

    return {
        'period': float(moire_period),
        'coupling_strength': float(coupling_strength),
        'band_offset': float(band_offset),
        'delta_theta': float(delta_theta)
    }


def calculate_lattice_relaxation(theta_rad: float) -> float:
    """
    Calculate effective twist angle after lattice relaxation.
    
    Small angles relax significantly towards alignment (0 degrees).
    Reconstruction is strong for theta < 2 degrees (Rosenberger et al.).
    
    Args:
        theta_rad: Geometric twist angle in radians
        
    Returns:
        float: Effective twist angle
    """
    theta_deg = np.degrees(theta_rad)
    
    # Relaxation parameter
    # Critical angle for reconstruction ~2 deg for WSe2 (Rosenberger et al.)
    theta_c = 2.0 
    amplitude = 0.6 # Strong relaxation
    
    # Relaxation reduces the effective angle
    relaxation_factor = 1.0 - amplitude * np.exp(-(theta_deg / theta_c)**2)
    
    theta_eff = theta_rad * relaxation_factor
    return theta_eff


def calculate_wse2_strain_energy(angles_vector_rad: np.ndarray) -> float:
    """
    Calculate accumulated strain energy for WSe2 stack.
    Based on Rosenberger et al. (2020) reconstruction model.
    
    Args:
        angles_vector_rad: Array of twist angles in radians
        
    Returns:
        float: Total strain energy penalty factor (dimensionless)
    """
    total_strain = 0.0
    
    # Iterate over layer pairs
    # We assume input is absolute angles, so we take differences
    for i in range(len(angles_vector_rad) - 1):
        diff = abs(angles_vector_rad[i+1] - angles_vector_rad[i])
        diff_deg = np.degrees(diff)
        
        # Layer depth penalty (growing with stack height)
        # Domains allow some relaxation, so scaling is softer than rigid body z^2
        # Using z^1.5 as a middle ground
        layer_penalty = (i + 1) ** 1.5
        
        # Twist cost profile derived from WSe2 physics
        if diff_deg < 1.0:
            # Near-aligned: strong reconstruction cost
            # High cost for very small angles as domains are large and rigid
            twist_cost = 1.0 / (diff_deg + 0.1) 
        elif 1.0 <= diff_deg <= 3.0:
            # Flat-band window: domains stabilize, sweet spot
            twist_cost = 0.2 
        else:
            # Large angle: elastic mismatch dominates
            twist_cost = 0.1 * diff_deg
            
        total_strain += layer_penalty * twist_cost
        
    # Scaling factor to keep penalty exp(-E) reasonable
    # Calibrated so that a "good" 7-layer stack has E ~ 1-3 (penalty 0.3 - 0.05)
    # rather than E ~ 100.
    return total_strain * 0.1


def calculate_all_moire_pairs(angles_vector: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate moiré properties for all adjacent layer pairs.
    
    Args:
        angles_vector: Array of twist angles
        
    Returns:
        Dictionary mapping pair index to properties
    """
    moire_data = {}
    for k in range(len(angles_vector) - 1):
        props = calculate_moire_properties(angles_vector[k], angles_vector[k+1])
        moire_data[f"{k}_{k+1}"] = props
    return moire_data

