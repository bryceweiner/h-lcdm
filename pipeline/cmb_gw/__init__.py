"""
H-ΛCDM CMB-GW Pipeline - Evolving Gravitational Constant Test Suite
====================================================================

Tests the evolving gravitational constant hypothesis G_eff(z) = G_0 × [1 - β × f(z)]
through five independent observational probes with joint parameter consistency analysis.

This pipeline implements the protocol from docs/cmb_gw.md:
- TEST 1: Sound horizon enhancement from BAO
- TEST 2: Void size distribution
- TEST 3: Standard siren luminosity distances
- TEST 4: CMB peak height ratios
- TEST 5: Cross-modal coherence at acoustic scale
"""

from .cmb_gw_pipeline import CMBGWPipeline

__all__ = ['CMBGWPipeline']

