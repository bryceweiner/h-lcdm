"""
TRGB Comparative Analysis pipeline.

Paired comparative analysis of CCHP TRGB H₀ measurements:
- Case A: Freedman 2019/2020 HST, LMC-anchored (d_local = 0.05 Mpc).
- Case B: Freedman 2024/2025 JWST, NGC 4258-anchored (d_local = 7.58 Mpc).

Each Freedman methodology is reproduced from public data on its own branch;
the H-ΛCDM framework's holographic projection formula is applied as a pure
forward prediction for each d_local. Unconditional reporting: both cases
always run to completion and both results are always published regardless
of agreement with framework predictions.
"""

from .trgb_pipeline import TRGBComparativePipeline

__all__ = ["TRGBComparativePipeline"]
