"""
H-ΛCDM Void Pipeline
===================

Cosmic void structure analysis for E8×E8 heterotic alignment.

This pipeline implements comprehensive void analysis including:
- Multi-survey void catalog processing
- 17-angle hierarchical E8 alignment detection
- Clustering discovery analysis
- Statistical validation (bootstrap, randomization)
- Model comparison and cross-validation
"""

from .void_pipeline import VoidPipeline

__all__ = ['VoidPipeline']
