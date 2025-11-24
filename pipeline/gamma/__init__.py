"""
H-ΛCDM Gamma Pipeline
=====================

Theoretical γ(z) and Λ_eff(z) calculation from first principles.

This pipeline implements the pure theoretical calculation of the information
processing rate γ and effective cosmological constant Λ as functions of redshift,
without any fitting to observational data.
"""

from .gamma_pipeline import GammaPipeline

__all__ = ['GammaPipeline']
