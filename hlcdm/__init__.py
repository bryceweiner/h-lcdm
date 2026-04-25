"""
Holographic Lambda Model (H-ΛCDM) - Core Library
=================================================

This package provides the core theoretical framework and parameters
for the Holographic Lambda Model (H-ΛCDM) analysis.

Modules:
    parameters: Single point of truth for all physical constants and parameters
    cosmology: Theoretical calculations including gamma, Lambda, and QTEP

E8 heterotic string-theory mathematics is provided by the external
``e8-heterotic-network`` package
(https://github.com/bryceweiner/e8-heterotic-network). The pipeline
uses **Convention A** (canonical root-system graph, ⟨α,β⟩ = +1) for
adjacency-derived quantities such as the clustering coefficient C(G).

Convention A clustering coefficient: 27/55 ≈ 0.4909.
The 25/32 = 0.78125 value previously used in the local hlcdm/e8/
package was a literature claim that did not match the actual
adjacency-graph computation; it is preserved as
``e8_heterotic.E8_CLUSTERING_LITERATURE_FRACTION`` for reference but
is no longer used by the projection formula.
"""

from .parameters import HLCDMParameters, HLCDM_PARAMS
from .cosmology import HLCDMCosmology

__version__ = HLCDM_PARAMS.version
__author__ = HLCDM_PARAMS.author
__paper__ = HLCDM_PARAMS.paper

# Expose key classes and instances
__all__ = [
    'HLCDMParameters',
    'HLCDM_PARAMS',
    'HLCDMCosmology',
    '__version__',
    '__author__',
    '__paper__'
]
