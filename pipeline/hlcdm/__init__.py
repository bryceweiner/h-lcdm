"""
H-ΛCDM Extension Tests Pipeline
===============================

Specialized H-ΛCDM extension tests that don't fit into other pipelines.

This pipeline includes:
- JWST early galaxy formation analysis
- Lyman-alpha phase transition mapping
- CMB Zeno transition tests
- FRB Little Bang analysis

These are parameter-free predictions testing fundamental H-ΛCDM signatures.
"""

from .hlcdm_pipeline import HLCDMPipeline

__all__ = ['HLCDMPipeline']
