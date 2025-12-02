"""
Tight-Binding Hamiltonian Engine
================================

Computes electronic band structure for twisted multilayers.
"""

import numpy as np
import scipy.linalg as la
from typing import Dict, Any, Tuple, List

from hlcdm.tmdc_parameters import WSe2Parameters

# Physics Constants for WSe2
HOPPING_T1 = WSe2Parameters.HOPPING_T1_EV      # eV (Nearest-neighbor)
HOPPING_T2 = WSe2Parameters.HOPPING_T2_EV      # eV (Next-nearest neighbor)
SOC_LAMBDA = WSe2Parameters.SOC_LAMBDA_EV      # eV (Spin-orbit coupling)
LAYER_GAP = 1.6       # eV (Approximate band gap for WSe2)


def construct_tb_hamiltonian(angles_array: np.ndarray, moire_properties: Dict[str, Dict[str, float]], 
                             k_points: int = 100) -> np.ndarray:
    """
    Build tight-binding Hamiltonian for 7-layer twisted system.
    
    Generates a set of Hamiltonians for a path in k-space.
    
    Args:
        angles_array: [θ₁, θ₂, ..., θ₇] in radians
        moire_properties: output from calculate_moire_properties
        k_points: Number of k-points to sample along high-symmetry path
    
    Returns:
        numpy.ndarray: Hamiltonian matrices of shape (k_points, N_bands, N_bands)
    """
    n_layers = len(angles_array)
    orbitals_per_layer = 4 # 2 bands (c, v) * 2 spins
    n_dim = n_layers * orbitals_per_layer
    
    # Initialize Hamiltonian array
    hamiltonians = np.zeros((k_points, n_dim, n_dim), dtype=np.complex128)
    
    # Generate k-path (simplified Gamma-K-M-Gamma for effective model)
    # We'll map k to [0, 1] for the variation
    k_path = np.linspace(0, 4*np.pi, k_points) # Arbitrary path scaling
    
    for k_idx, k_val in enumerate(k_path):
        H_k = np.zeros((n_dim, n_dim), dtype=np.complex128)
        
        # Fill diagonal blocks (Intralayer terms)
        for l in range(n_layers):
            base_idx = l * orbitals_per_layer
            
            # Intralayer Hamiltonian H_l
            # Twist angle affects the effective hopping strength
            # Small twist angles reduce effective hopping due to moiré modulation
            twist_angle = angles_array[l]
            
            # Angle-dependent modulation: hopping is reduced at small angles
            # This models the moiré-induced band flattening
            angle_factor = 1.0 - 0.3 * np.exp(-twist_angle / 0.05)  # Reduces hopping at small angles
            
            # Dispersion f(k) ~ cos(k) - approximated, modulated by twist angle
            dispersion = angle_factor * (-HOPPING_T1 * np.cos(k_val) - HOPPING_T2 * np.cos(2*k_val))
            
            # Spin-orbit coupling (split bands)
            # SOC acts differently on valence/conduction and spin up/down
            # Simple model: +/- lambda/2
            
            # Block structure: |c, up>, |c, down>, |v, up>, |v, down>
            
            # Conduction band (Spin Up)
            H_k[base_idx, base_idx] = LAYER_GAP/2 + dispersion + SOC_LAMBDA/2
            # Conduction band (Spin Down)
            H_k[base_idx+1, base_idx+1] = LAYER_GAP/2 + dispersion - SOC_LAMBDA/2
            # Valence band (Spin Up)
            H_k[base_idx+2, base_idx+2] = -LAYER_GAP/2 - dispersion + SOC_LAMBDA/2
            # Valence band (Spin Down)
            H_k[base_idx+3, base_idx+3] = -LAYER_GAP/2 - dispersion - SOC_LAMBDA/2
            
        # Fill off-diagonal blocks (Interlayer coupling)
        for l in range(n_layers - 1):
            idx_i = l
            idx_j = l + 1
            
            base_i = idx_i * orbitals_per_layer
            base_j = idx_j * orbitals_per_layer
            
            # Get moiré properties for this pair
            pair_key = f"{idx_i}_{idx_j}"
            props = moire_properties.get(pair_key, {})
            coupling = props.get('coupling_strength', 0.15)
            moire_period = props.get('period', 1e9)
            
            # Interlayer coupling matrix T_ij
            # Tunneling strength depends on moiré period and relative twist angle
            # Smaller moiré period (smaller angle difference) = stronger coupling
            
            # k-dependent modulation: coupling varies with momentum
            # At small k (near Gamma point), coupling is strongest
            k_modulation = np.exp(-k_val / (2 * np.pi))  # Decay with k
            
            # Period-dependent scaling: normalize by reference period
            period_factor = 5.0 / min(moire_period, 100.0)  # Reference 5 nm
            
            # Effective coupling strength
            effective_coupling = coupling * period_factor * k_modulation * 0.1
            
            T_ij = effective_coupling * np.eye(orbitals_per_layer)
            
            # Fill H_ij and H_ji (Hermitian)
            H_k[base_i:base_i+orbitals_per_layer, base_j:base_j+orbitals_per_layer] = T_ij
            H_k[base_j:base_j+orbitals_per_layer, base_i:base_i+orbitals_per_layer] = T_ij.conj().T
            
        # Fill Next-Nearest Neighbor (NNN) blocks (Layer i to i+2)
        # Perturbation: ~10% of NN coupling
        # This adds realistic complexity to the Hamiltonian, breaking perfect symmetries
        for l in range(n_layers - 2):
            idx_i = l
            idx_j = l + 2
            
            base_i = idx_i * orbitals_per_layer
            base_j = idx_j * orbitals_per_layer
            
            # NNN Coupling
            # Simple model: 10% of base interlayer coupling
            nnn_coupling_strength = WSe2Parameters.VDW_COUPLING_GAMMA0_EV * 0.1 
            
            T_nnn = nnn_coupling_strength * np.eye(orbitals_per_layer)
            
            H_k[base_i:base_i+orbitals_per_layer, base_j:base_j+orbitals_per_layer] = T_nnn
            H_k[base_j:base_j+orbitals_per_layer, base_i:base_i+orbitals_per_layer] = T_nnn.conj().T
            
        hamiltonians[k_idx] = H_k
        
    return hamiltonians


def solve_eigenvalue_problem(hamiltonians: np.ndarray) -> Dict[str, Any]:
    """
    Diagonalize Hamiltonians to get band structure.
    
    Args:
        hamiltonians: Array of shape (k_points, N_dim, N_dim)
        
    Returns:
        dict: {
            'eigenvalues': (k_points, N_dim),
            'eigenvectors': (k_points, N_dim, N_dim),
            'fermi_energy': float,
            'flatness_metric': float
        }
    """
    k_points, n_dim, _ = hamiltonians.shape
    
    eigenvalues = np.zeros((k_points, n_dim))
    eigenvectors = np.zeros((k_points, n_dim, n_dim), dtype=np.complex128)
    
    for k in range(k_points):
        # eigh for Hermitian matrices
        w, v = la.eigh(hamiltonians[k])
        eigenvalues[k] = w
        eigenvectors[k] = v
        
    # Calculate flatness metric (inverse of bandwidth of relevant bands)
    # Look at bands near Fermi level (assuming 0 for now or mid-gap)
    # Here we just take variance of the middle bands
    
    # Identify flat bands: bands with minimal dispersion
    band_dispersions = np.std(eigenvalues, axis=0)
    flatness_metric = 1.0 / (np.min(band_dispersions) + 1e-6)
    
    return {
        'eigenvalues': eigenvalues,
        'eigenvectors': eigenvectors,
        'fermi_energy': 0.0, # Simplified
        'flatness_metric': flatness_metric
    }

