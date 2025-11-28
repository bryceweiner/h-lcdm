"""
H-ZOBOV: Holographic ZOBOV Void Finding Algorithm
==================================================

MPS-accelerated implementation of the ZOBOV algorithm (Neyrinck 2008) with
integration of H-ΛCDM cosmological constant Λ(z) computed from first principles.

This module implements a parameter-free void-finding algorithm that uses:
- Voronoi tessellation for density field estimation (DTFE)
- Watershed algorithm for zone identification
- Zone merging with Λ(z)-dependent significance thresholds
- Apple Silicon MPS acceleration for large datasets

References:
    Neyrinck, M. C. (2008). ZOBOV: a parameter-free void-finding algorithm.
    Monthly Notices of the Royal Astronomical Society, 386(4), 2101-2109.
"""

from .zobov_pipeline import HZOBOVPipeline
from .zobov_core import ZOBOVCore
from .zobov_mps import ZOBOVMPS
from .hlcdm_integration import get_lambda_at_redshift

__all__ = [
    'HZOBOVPipeline',
    'ZOBOVCore',
    'ZOBOVMPS',
    'get_lambda_at_redshift',
]

