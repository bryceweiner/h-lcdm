"""
H-ΛCDM Expansion Enhancement Falsifiability Test
=================================================

Joint-likelihood fit of DESI DR1 BAO + Pantheon+ SNe + Planck θ* against two
cosmological models:

    Model A (standard ΛCDM):       (H₀, Ω_m) free, r_d = 147.5 Mpc fixed, ε = 0.
    Model B (framework):            (H₀, Ω_m, ε) free, r_d = 150.71 Mpc fixed.

Tests the framework's falsifiable prediction that H(z) for z ≤ z_rec is enhanced
by a factor (1+ε) from a persistent Zeno back-reaction deformation of the causal
diamond at recombination.

Two ε(z) prescriptions are run:
    - Constant ε applied as a hard step at z_rec = 1100.
    - QTEP-motivated ε(z) tied to the framework's γ(z) in hlcdm/cosmology.py.
"""

from .expansion_pipeline import ExpansionPipeline

__all__ = ["ExpansionPipeline"]
