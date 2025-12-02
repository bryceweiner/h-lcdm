"""
Objective Function
==================

Defines the fitness metric for twist angle optimization.
"""

import numpy as np
import logging
from typing import Dict, Any, Tuple, Optional

from pipeline.tmdc.physics.moire import calculate_all_moire_pairs, calculate_wse2_strain_energy
from pipeline.tmdc.physics.tight_binding import construct_tb_hamiltonian, solve_eigenvalue_problem
from pipeline.tmdc.physics.exciton_floquet import calculate_exciton_floquet_coupling
from pipeline.tmdc.physics.qtep_dynamics import qtep_dynamics
from hlcdm.cosmology import HLCDMCosmology

logger = logging.getLogger(__name__)

def expand_twist_vector(interlayer_angles_deg: np.ndarray,
                        n_layers: Optional[int] = None) -> np.ndarray:
    """
    Expand 6D interlayer twist vector to 7D absolute angle vector.
    Fixes layer 1 at 0 degrees.
    theta_{i+1} = theta_i + delta_theta_i
    
    Args:
        interlayer_angles_deg: Array of 6 interlayer twist angles (degrees)
        
    Returns:
        Array of 7 absolute twist angles (degrees)
    """
    # Ensure input is 1D array
    deltas = np.atleast_1d(interlayer_angles_deg).astype(float)
    if deltas.ndim != 1:
        deltas = deltas.flatten()
    
    if n_layers is None:
        n_layers = len(deltas) + 1
    elif len(deltas) != n_layers - 1:
        raise ValueError(f"Expected {n_layers - 1} interlayer angles, got {len(deltas)}")

    # Initialize with 0 for first layer
    angles_deg = np.zeros(n_layers)
    
    # Cumulative sum
    current_angle = 0.0
    for i, delta in enumerate(deltas):
        current_angle += delta
        angles_deg[i+1] = current_angle
        
    return angles_deg


def objective_function(interlayer_angles_deg: np.ndarray,
                       n_layers: Optional[int] = None,
                       debug: bool = False) -> float:
    """
    Evaluate total coherence amplification for given twist angle configuration.
    
    Args:
        interlayer_angles_deg: [Δθ₁, ..., Δθ₆] in degrees (relative twists)
        debug: Whether to log detailed diagnostics
        
    Returns:
        float: Total coherence amplification factor
    """
    try:
        if n_layers is None:
            inferred_layers = len(np.atleast_1d(interlayer_angles_deg)) + 1
            n_layers = inferred_layers
        elif len(np.atleast_1d(interlayer_angles_deg)) != n_layers - 1:
            raise ValueError(f"Expected {n_layers - 1} interlayer angles, "
                             f"got {len(np.atleast_1d(interlayer_angles_deg))}")

        # Expand to absolute angles for physics engine
        angles_vector_deg = expand_twist_vector(interlayer_angles_deg, n_layers=n_layers)
        
        # Convert degrees to radians for physics calculations
        angles_vector_rad = np.deg2rad(angles_vector_deg)
        
        # Step 1: Calculate moiré properties for all layer pairs
        moire_data = calculate_all_moire_pairs(angles_vector_rad)
        
        # Step 2: Construct and diagonalize Hamiltonian
        # Using fewer k-points for speed during optimization loop
        hamiltonian = construct_tb_hamiltonian(angles_vector_rad, moire_data, k_points=50)
        band_structure = solve_eigenvalue_problem(hamiltonian)
        
        # Step 3: Compute exciton-Floquet coupling
        # Use calculated gamma from cosmology engine (z=0 for present-day device)
        gamma_rate = HLCDMCosmology.gamma_at_redshift(0.0)
        composite_data = calculate_exciton_floquet_coupling(band_structure, gamma_rate=gamma_rate)
        
        # Step 4: Run QTEP dynamics simulation
        # Pass moire data for layer-specific coupling constants
        qtep_results = qtep_dynamics(band_structure,
                                     composite_data,
                                     moire_data=moire_data,
                                     n_layers=n_layers)
        
        # Step 5: Extract total coherence amplification from QTEP cascade
        # This is the "base" QTEP amplification before device-level penalties
        base_amplification = qtep_results['total_amplification']
        
        # Step 6: Apply chain continuity bonus/penalty
        # A robust QTEP cascade requires uniform, strong coupling across the chain
        # Extract coupling strengths for the 6 interfaces
        moire_couplings = [moire_data[f"{i}_{i+1}"]['coupling_strength'] for i in range(len(angles_vector_deg) - 1)]
        
        # Penalty for weak links (breaks cascade)
        # Threshold 0.01 roughly corresponds to decoupled layers
        min_coupling = min(moire_couplings) if moire_couplings else 0.0
        if min_coupling < 0.01:
            # Weak link significantly degrades performance
            chain_penalty = 0.5 # 50% reduction
        else:
            chain_penalty = 1.0
            
        # Calculate accumulated strain penalty (Physical Realism for WSe2)
        # Deep stacks accumulate strain which degrades quantum coherence
        total_strain_energy = calculate_wse2_strain_energy(angles_vector_rad)
        strain_penalty_factor = np.exp(-total_strain_energy)
        
        # Add random thermal/experimental noise to prevent "perfect" integer convergence
        # Signal-to-noise ratio ~ 100 (1% noise)
        # This forces the optimizer to handle realistic experimental uncertainty
        noise_factor = 1.0 + np.random.normal(0, 0.01)
        
        total_amplification = base_amplification * chain_penalty * strain_penalty_factor * noise_factor
        
        # Diagnostics logging
        # Log randomly (1%) or if debug is True
        if debug or np.random.random() < 0.01:
            logger.info(f"Layers: {n_layers}")
            logger.info(f"Interlayer: {[f'{a:.2f}' for a in interlayer_angles_deg]}")
            logger.info(f"Absolute: {[f'{a:.2f}' for a in angles_vector_deg]}")
            logger.info(f"Couplings: {[f'{c:.3f}' for c in moire_couplings]}")
            logger.info(f"Base Amp: {base_amplification:.2f}, Penalty: {chain_penalty}, Strain: {strain_penalty_factor:.3f}")
            logger.info(f"Total Amp: {total_amplification:.2f}")
        
        return float(total_amplification)
        
    except Exception as e:
        logger.error(f"Error in objective function evaluation: {e}")
        # Return a very low value to discourage this region (penalty)
        return -1.0


def objective_diagnostics(interlayer_angles_deg: np.ndarray,
                          n_layers: Optional[int] = None) -> Dict[str, Any]:
    """
    Deterministic decomposition of the TMDC objective for a given configuration.
    
    This function mirrors the physics used in ``objective_function`` but returns
    the individual physical factors without stochastic noise:
    
    - Base QTEP amplification (from ``qtep_dynamics``)
    - Chain continuity penalty
    - Total accumulated strain energy and associated penalty factor
    - Inter-layer moiré couplings for the optimal stack
    
    Args:
        interlayer_angles_deg: [Δθ₁, ..., Δθ₆] in degrees
        
    Returns:
        dict with keys:
            - base_amplification
            - chain_penalty
            - total_strain_energy
            - strain_penalty_factor
            - moire_couplings
            - min_coupling
            - max_coupling
            - mean_coupling
            - absolute_angles
    """
    if n_layers is None:
        n_layers = len(np.atleast_1d(interlayer_angles_deg)) + 1
    elif len(np.atleast_1d(interlayer_angles_deg)) != n_layers - 1:
        raise ValueError(f"Expected {n_layers - 1} interlayer angles, "
                         f"got {len(np.atleast_1d(interlayer_angles_deg))}")

    # Expand to absolute angles
    angles_vector_deg = expand_twist_vector(interlayer_angles_deg, n_layers=n_layers)
    
    # Convert degrees to radians for physics calculations
    angles_vector_rad = np.deg2rad(angles_vector_deg)
    
    # Step 1: Moiré properties
    moire_data = calculate_all_moire_pairs(angles_vector_rad)
    
    # Step 2: Tight-binding Hamiltonian and band structure
    hamiltonian = construct_tb_hamiltonian(angles_vector_rad, moire_data, k_points=50)
    band_structure = solve_eigenvalue_problem(hamiltonian)
    
    # Step 3: Exciton-Floquet composite
    gamma_rate = HLCDMCosmology.gamma_at_redshift(0.0)
    composite_data = calculate_exciton_floquet_coupling(band_structure, gamma_rate=gamma_rate)
    
    # Step 4: QTEP dynamics (base amplification)
    qtep_results = qtep_dynamics(band_structure,
                                 composite_data,
                                 moire_data=moire_data,
                                 n_layers=n_layers)
    base_amplification = float(qtep_results['total_amplification'])
    
    # Step 5: Chain penalty from weakest interface
    moire_couplings = [
        moire_data[f"{i}_{i+1}"]['coupling_strength']
        for i in range(len(angles_vector_deg) - 1)
    ]
    
    min_coupling = min(moire_couplings) if moire_couplings else 0.0
    max_coupling = max(moire_couplings) if moire_couplings else 0.0
    mean_coupling = float(np.mean(moire_couplings)) if moire_couplings else 0.0
    
    if min_coupling < 0.01:
        chain_penalty = 0.5
    else:
        chain_penalty = 1.0
    
    # Step 6: Strain penalty (no noise)
    total_strain_energy = float(calculate_wse2_strain_energy(angles_vector_rad))
    strain_penalty_factor = float(np.exp(-total_strain_energy))
    
    return {
        'base_amplification': base_amplification,
        'chain_penalty': float(chain_penalty),
        'total_strain_energy': total_strain_energy,
        'strain_penalty_factor': strain_penalty_factor,
        'moire_couplings': [float(c) for c in moire_couplings],
        'min_coupling': float(min_coupling),
        'max_coupling': float(max_coupling),
        'mean_coupling': mean_coupling,
        'absolute_angles': angles_vector_deg.tolist()
    }
