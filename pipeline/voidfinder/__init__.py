"""
H-ΛCDM VoidFinder Pipeline
==========================

VAST VoidFinder pipeline for generating void catalogs from galaxy surveys.

Algorithm: VoidFinder (Hoyle & Vogeley 2002)
- Grid-based sphere-growing algorithm
- Imposes cubic grid over galaxy distribution
- Grows spheres from empty grid cells until bounded by galaxies
- Combines overlapping spheres into discrete voids
- Identifies maximal spheres (largest sphere in each void)

This pipeline implements:
- Galaxy catalog downloading (SDSS DR16, extensible to others)
- VAST VoidFinder algorithm with CPU parallelization
- Checkpointing for large datasets
- Integration with existing void analysis pipeline
"""

from .voidfinder_pipeline import VoidFinderPipeline
from .zobov.zobov_pipeline import HZOBOVPipeline

__all__ = ['VoidFinderPipeline', 'HZOBOVPipeline']

