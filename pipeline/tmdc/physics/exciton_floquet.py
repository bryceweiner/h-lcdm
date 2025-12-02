"""
Exciton-Floquet Coupling Engine
===============================

Models exciton-Floquet composite formation based on Park et al. research
and QTEP information-theoretic framework.
"""

import numpy as np
from typing import Dict, Any

from hlcdm.parameters import HLCDM_PARAMS
from hlcdm.cosmology import HLCDMCosmology

# Material-specific constants (WSe2)
# Approx 0.3-0.7 eV depending on dielectric environment. 0.4 eV is typical.
EXCITON_BINDING_ENERGY_EB = 0.4  # eV (WSe2)


def calculate_exciton_floquet_coupling(band_structure: Dict[str, Any], 
                                       gamma_rate: float = None) -> Dict[str, Any]:
    """
    Model exciton-Floquet composite formation and coupling strength.
    
    Args:
        band_structure: eigenvalues and eigenvectors from tight-binding
        gamma_rate: universal information processing rate (s^-1). 
                   If None, uses calculated gamma from cosmology engine at z=0.
        
    Returns:
        dict: {
            'composite_strength': coupling strength array,
            'floquet_frequency': driving frequency (Hz),
            'mixing_coefficients': {'alpha': array, 'beta': array}
        }
    """
    # Use calculated gamma from cosmology engine, not empirical values
    if gamma_rate is None:
        # Calculate gamma at z=0 using the theoretical engine
        gamma_rate = HLCDMCosmology.gamma_at_redshift(0.0)
    
    eigenvalues = band_structure['eigenvalues'] # (k_points, n_dim)
    
    # Floquet frequency: ω_F = 2πγ
    floquet_frequency = 2 * np.pi * gamma_rate # rad/s
    
    # Note: This frequency is extremely low (~10^-28 Hz). 
    # In standard Floquet theory, sidebands are at E +/- n*hbar*omega.
    # Here, the splitting is negligible. 
    # However, in the QTEP framework, this might represent a coupling to an information bath.
    
    # Calculate exciton binding energies
    # Simplified: Assume constant binding energy modified by band curvature (flatness)
    # E_b_effective = E_b * (1 + flatness_metric)
    flatness = band_structure.get('flatness_metric', 1.0)
    effective_binding_energy = EXCITON_BINDING_ENERGY_EB * (1.0 + 0.1 * flatness)
    
    # Compute exchange coupling through Coulomb interactions
    # Exchange interaction J depends on wavefunction overlap. 
    # We'll approximate it based on the inverse of the band gap average (perturbation theory)
    avg_gap = np.mean(np.abs(np.diff(eigenvalues, axis=1)))
    exchange_coupling = 0.1 / (avg_gap + 1e-6) # Phenomenological scaling
    
    # Model mixing coefficients α, β for |ψ⟩ = α|exciton⟩ + β|Floquet⟩
    # This looks like a two-level system hybridization.
    # Hamiltonian matrix for the composite:
    # H_comp = [ E_exciton    V_coupling ]
    #          [ V_coupling   E_Floquet  ]
    # E_Floquet might be related to the photon energy (very small here) or a Floquet state energy.
    # Let's assume resonance condition or effective coupling parameterization.
    
    # Based on "Park et al." and prompt context, we calculate a coupling strength.
    # Let's assume the coupling strength is enhanced by the QTEP ratio in some way or 
    # simply derived from the parameters.
    
    # Composite strength ~ Exchange * Overlap
    # We'll return a mock array of strengths for each k-point
    k_points = eigenvalues.shape[0]
    composite_strength = np.full(k_points, exchange_coupling * effective_binding_energy)
    
    # Calculate mixing coefficients
    # If we assume maximal mixing (resonance) or some distribution
    # alpha^2 + beta^2 = 1
    # Let's model beta (Floquet component) as proportional to coupling / detuning
    # If detuning is small, beta is large.
    
    # For this simulation, we'll assign alpha, beta based on band flatness
    # Flatter bands -> stronger correlation -> higher beta?
    beta = 0.5 * (flatness / (flatness + 10.0)) # Saturates at 0.5
    alpha = np.sqrt(1 - beta**2)
    
    return {
        'composite_strength': composite_strength,
        'floquet_frequency': floquet_frequency,
        'mixing_coefficients': {
            'alpha': np.full(k_points, alpha),
            'beta': np.full(k_points, beta)
        },
        'effective_binding_energy': effective_binding_energy
    }

