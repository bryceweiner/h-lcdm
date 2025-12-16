"""
CMB Cold Spot QTEP Analysis Pipeline
====================================

Direct test of whether the CMB Cold Spot arises from QTEP efficiency variations
as predicted by the information-theoretic gravity framework.

Tests three independent hypotheses:
1. Temperature deficit vs QTEP prediction
2. Angular power spectrum structure (discrete vs continuous)
3. Spatial correlation with QTEP efficiency map
"""

from .cmb_cold_spot_pipeline import CMBColdSpotPipeline

__all__ = ['CMBColdSpotPipeline']

