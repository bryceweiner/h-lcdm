"""
H-ΛCDM CMB Pipeline
==================

Comprehensive CMB E-mode analysis with information-theoretic methods.

This pipeline implements the complete CMB analysis including:
- Wavelet analysis
- Bispectrum analysis
- Topological analysis
- Phase coherence analysis
- CMB-void cross-correlation
- Scale-dependent power analysis
- ML pattern recognition
"""

from .cmb_pipeline import CMBPipeline

__all__ = ['CMBPipeline']