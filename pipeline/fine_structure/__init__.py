"""
Fine Structure Constant Pipeline
=================================

Holographic derivation of the fine structure constant from information processing principles.

The framework yields a parameter-free prediction:
α⁻¹ = (1/2)ln(S_H) - ln(4π²) - 1/(2π) = 137.032

Compared against CODATA 2018: α⁻¹ = 137.035999084(21)
"""

from .fine_structure_pipeline import FineStructurePipeline
from .physics import (
    calculate_bekenstein_hawking_entropy,
    calculate_information_processing_rate,
    calculate_alpha_inverse,
    calculate_alpha,
)

__all__ = [
    'FineStructurePipeline',
    'calculate_bekenstein_hawking_entropy',
    'calculate_information_processing_rate',
    'calculate_alpha_inverse',
    'calculate_alpha',
]
