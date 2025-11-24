"""
Holographic Lambda Model (H-ΛCDM) - Core Library
=================================================

This package provides the core theoretical framework and parameters
for the Holographic Lambda Model (H-ΛCDM) analysis.

Modules:
    parameters: Single point of truth for all physical constants and parameters
    cosmology: Theoretical calculations including gamma, Lambda, and QTEP
    e8: Immutable E8 heterotic string theory mathematics
"""

from .parameters import HLCDMParameters, HLCDM_PARAMS
from .cosmology import HLCDMCosmology
from .e8.e8_heterotic_core import E8HeteroticSystem

__version__ = HLCDM_PARAMS.version
__author__ = HLCDM_PARAMS.author
__paper__ = HLCDM_PARAMS.paper

# Expose key classes and instances
__all__ = [
    'HLCDMParameters',
    'HLCDM_PARAMS',
    'HLCDMCosmology',
    'E8HeteroticSystem',
    '__version__',
    '__author__',
    '__paper__'
]
