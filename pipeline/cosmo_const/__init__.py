"""
Cosmological Constant Pipeline
==============================

Holographic resolution of the cosmological constant problem through causal diamond triality.

The framework yields a parameter-free prediction:
Ω_Λ = (1-e^{-1})(11\ln 2 - 3\ln 3)/4 = 0.6841

Compared against Planck 2018: Ω_Λ = 0.6847 ± 0.0073
"""

from .cosmo_const_pipeline import CosmoConstPipeline
from .physics import (
    calculate_geometric_entropy,
    calculate_irreversibility_fraction,
    calculate_omega_lambda,
    calculate_lambda,
)

__all__ = [
    'CosmoConstPipeline',
    'calculate_geometric_entropy',
    'calculate_irreversibility_fraction',
    'calculate_omega_lambda',
    'calculate_lambda',
]
