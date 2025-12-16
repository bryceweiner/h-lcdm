"""
Gravitational Constant Pipeline
================================

Holographic derivation of Newton's gravitational constant G from information processing principles.

The framework yields a prediction:
G = πc⁵/(ℏH²·N_P·ln(3)·f_quantum) = 6.67 × 10⁻¹¹ m³/(kg·s²)

where N_P is derived from the fine structure constant via:
α⁻¹ = (1/2)ln(N_P) - ln(4π²) - 1/(2π)

Compared against CODATA 2018: G = 6.67430(15) × 10⁻¹¹ m³/(kg·s²)
"""

from .gravitational_constant_pipeline import GravitationalConstantPipeline
from .physics import (
    calculate_information_capacity,
    calculate_alpha_inverse_from_np,
    calculate_np_from_alpha_inverse,
    calculate_g_base,
    calculate_g_geometric,
    calculate_g_final,
    calculate_g,
)

__all__ = [
    'GravitationalConstantPipeline',
    'calculate_information_capacity',
    'calculate_alpha_inverse_from_np',
    'calculate_np_from_alpha_inverse',
    'calculate_g_base',
    'calculate_g_geometric',
    'calculate_g_final',
    'calculate_g',
]
