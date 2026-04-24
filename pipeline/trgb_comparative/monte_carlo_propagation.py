"""
Monte Carlo uncertainty propagation for framework forward predictions.

The framework has no likelihood — it only predicts. So uncertainty
propagation here is NOT emcee MCMC; it is a Monte Carlo draw over the
inputs (H_CMB from Planck, d_local from its geometric anchor), run through
:func:`pipeline.trgb_comparative.projection_formula.propagate_projection_uncertainty`.

MLX acceleration, when available, applies to the vectorized formula
evaluation. See ``compute_backend.py``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .projection_formula import (
    PerturbativeBreakdownWarning,
    propagate_projection_uncertainty,
)


@dataclass(frozen=True)
class PlanckH0Posterior:
    """Compact summary of the Planck 2018 H₀ posterior used as input.

    Source: Planck 2018 VI, A&A 641 A6, Table 2 (TT,TE,EE+lowE+lensing):
    H₀ = 67.36 ± 0.54 km/s/Mpc. Treated as a Gaussian.
    """

    mean: float = 67.36
    sigma: float = 0.54

    def sample(self, size: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(self.mean, self.sigma, size=size)


def sample_d_local(d_local: float, sigma_d_local: float, size: int, seed: Optional[int] = None) -> np.ndarray:
    """Gaussian draw over the geometric-anchor distance."""
    rng = np.random.default_rng(seed)
    return rng.normal(d_local, sigma_d_local, size=size)


def run_monte_carlo(
    d_local_mpc: float,
    sigma_d_local_mpc: float,
    H_cmb_posterior: Optional[PlanckH0Posterior] = None,
    n_samples: int = 50_000,
    seed: int = 42,
):
    """Sample H₀ predictions from the projection formula.

    Returns (H0_samples, per_draw_results) — the same tuple as
    :func:`propagate_projection_uncertainty` but with inputs drawn
    internally from Planck H_CMB and a Gaussian d_local.
    """
    if H_cmb_posterior is None:
        H_cmb_posterior = PlanckH0Posterior()

    rng_seed_h = int(seed) + 1
    rng_seed_d = int(seed) + 2

    H_cmb_samples = H_cmb_posterior.sample(n_samples, seed=rng_seed_h)
    d_local_samples = sample_d_local(d_local_mpc, sigma_d_local_mpc, n_samples, seed=rng_seed_d)
    # Guard against any negative draws from a Gaussian in d_local.
    d_local_samples = np.maximum(d_local_samples, 1e-6)

    import warnings as _warnings

    with _warnings.catch_warnings():
        _warnings.simplefilter("ignore", PerturbativeBreakdownWarning)
        return propagate_projection_uncertainty(
            H_cmb_samples=H_cmb_samples,
            d_local_samples=d_local_samples,
            emit_warning=False,
        )


__all__ = ["PlanckH0Posterior", "run_monte_carlo", "sample_d_local"]
