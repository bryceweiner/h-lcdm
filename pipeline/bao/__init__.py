"""
H-ΛCDM BAO Pipeline
==================

Baryon Acoustic Oscillation analysis and prediction testing.

This pipeline implements comprehensive BAO analysis including:
- Multi-dataset validation (BOSS, eBOSS, 6dFGS, WiggleZ, DESI)
- α consistency analysis
- Model comparison (BIC/AIC/Bayes)
- Forward predictions for DESI Y3
- Alternative model comparison
"""

from .bao_pipeline import BAOPipeline

__all__ = ['BAOPipeline']
