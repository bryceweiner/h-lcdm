"""
Planck-residual-shaped ε(z) for Model B_residuals.

Pipeline:
    1. Download Planck 2018 observed + best-fit theory D_ℓ for TT/TE/EE.
    2. Per-channel signed residual: Δ_X(ℓ) = D^obs_X − D^thy_X.
    3. Combine channels by inverse-variance weighting → signed unitless
       z_combined(ℓ) (in σ-units):

          z(ℓ) = Σ_X Δ_X(ℓ)/σ_X² / √(Σ_X 1/σ_X²)

    4. Map ℓ → z via Option A (post-recombination horizon re-entry):

          ℓ(z) = π · D_M(z*) / (D_M(z*) − D_M(z))

       evaluated in a FIXED reference ΛCDM cosmology so the shape is
       deterministic across MCMC samples (only ε_amp floats).

    5. Interpolate z_combined onto a z-grid, **zero-extended** outside the
       Planck ℓ-coverage (ℓ<2 ↔ z≈0 and ℓ>2508 ↔ z≈z_rec).

The returned callable ``epsilon_shape(z)`` is pure numpy, vectorized, and
returns a unitless signed number of O(1). The free MCMC parameter ``ε_amp``
scales it into H(z) enhancement: ε(z) = ε_amp · epsilon_shape(z).

⚠ Monotonicity caveat: the signed combined residual oscillates about zero, so
ε(z) can be negative locally — violating the framework's "screen cannot
reverse" monotonicity claim. This is methodologically intentional, per user
direction: the residual pattern is *data-driven*, and local sign violations
are treated as fluctuations about a positive-mean tail rather than
disqualifying. Any resulting publication must flag this.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from data.loader import DataLoader

from .cosmology import D_M, Z_REC, make_H_callable

logger = logging.getLogger(__name__)


# Reference ΛCDM cosmology used to freeze the ℓ(z) mapping so ε_shape(z) is
# a deterministic function of z across all MCMC samples.
_REF_H0: float = 67.36
_REF_OM: float = 0.3153

# Planck 2018 photon-decoupling redshift (same as data_loaders.load_cmb default).
_Z_STAR: float = 1089.80


# -----------------------------------------------------------------------------
# Residual loader + combination
# -----------------------------------------------------------------------------

@dataclass(frozen=True)
class PlanckResidualSignature:
    """Output of combining TT+TE+EE Planck residuals into one signed σ-score."""

    ell: np.ndarray            # (N,) multipoles covered by Planck
    z_combined: np.ndarray     # (N,) signed inverse-variance-weighted z-score


def load_combined_residuals(loader: Optional[DataLoader] = None) -> PlanckResidualSignature:
    """Load Planck 2018 observed − theory D_ℓ and combine into a signed z-score.

    Both observed and theory spectra are downloaded from the ESA Planck Legacy
    Archive (see ``data/loader.py``) and cached under ``downloaded_data/``.
    """
    if loader is None:
        loader = DataLoader()
    obs = loader.load_planck_2018_full_spectra_dell()
    thy = loader.load_planck_2018_best_fit_theory()

    # Planck ships TE/EE only out to ℓ=1996 while TT goes to 2508; theory goes
    # to 2508 for all three. Take the INTERSECTION of all six ℓ-grids so the
    # inverse-variance sum is well-defined at every multipole we use.
    ell_ref = obs['TT'][0]
    for channel in ('TE', 'EE'):
        ell_ref = np.intersect1d(ell_ref, obs[channel][0], assume_unique=True)
    for channel in ('TT', 'TE', 'EE'):
        ell_ref = np.intersect1d(ell_ref, thy[channel][0], assume_unique=True)
    ell_ref = ell_ref.astype(float)

    def _on_grid(arr_ell: np.ndarray, arr_y: np.ndarray) -> np.ndarray:
        # Assumes integer-ish ℓ grids on both sides.
        idx = np.searchsorted(arr_ell, ell_ref)
        return arr_y[idx]

    # Signed IVW combination: z(ℓ) = Σ_X Δ_X/σ_X² / sqrt(Σ_X 1/σ_X²).
    numer = np.zeros_like(ell_ref, dtype=float)
    inv_var_sum = np.zeros_like(ell_ref, dtype=float)
    for channel in ('TT', 'TE', 'EE'):
        ell_obs, D_obs_raw, sigma_raw = obs[channel]
        ell_thy, D_thy_raw = thy[channel]
        D_obs = _on_grid(ell_obs, D_obs_raw)
        sigma = _on_grid(ell_obs, sigma_raw)
        D_thy = _on_grid(ell_thy, D_thy_raw)
        delta = D_obs - D_thy
        # Guard against σ==0 rows (shouldn't happen for full spectra but defensive).
        safe = sigma > 0
        inv_var = np.zeros_like(sigma)
        inv_var[safe] = 1.0 / (sigma[safe] ** 2)
        numer += delta * inv_var
        inv_var_sum += inv_var

    with np.errstate(divide='ignore', invalid='ignore'):
        z_combined = np.where(
            inv_var_sum > 0,
            numer / np.sqrt(inv_var_sum),
            0.0,
        )

    logger.info(
        f"Combined Planck residuals: ℓ=[{int(ell_ref.min())}, {int(ell_ref.max())}], "
        f"N={ell_ref.size}; z-score range [{z_combined.min():.2f}, {z_combined.max():.2f}] σ; "
        f"rms {np.sqrt((z_combined**2).mean()):.2f} σ"
    )
    return PlanckResidualSignature(ell=ell_ref.astype(float), z_combined=z_combined.astype(float))


# -----------------------------------------------------------------------------
# ℓ(z) mapping — Option A, post-recombination horizon
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _ref_DM_to_z_star() -> float:
    H = make_H_callable(_REF_H0, _REF_OM, 0.0, mode="constant")
    # High n_steps not needed: D_M(z_star) is done adaptively by ``theta_star``
    # elsewhere; here a simple trapezoid with 4096 points is adequate for the
    # mapping precision (sub-Mpc).
    return float(np.atleast_1d(D_M(_Z_STAR, H, n_steps=4096))[0])


@lru_cache(maxsize=1)
def _ref_DM_grid(n_linear: int = 4096, n_log: int = 4096) -> Tuple[np.ndarray, np.ndarray]:
    """Precomputed (z_grid, D_M_grid) under the reference cosmology.

    Grid design: dense linear sampling in z ∈ [0, 1] (captures the low-ℓ
    end), then log-spaced sampling in (1+z) ∈ [2, 1+z_star] to put most
    points close to z_rec where 1/(D_M(z*)-D_M(z)) → ∞. High-ℓ Planck modes
    (ℓ > 1000) correspond to z very close to z_rec and need the dense tail.
    """
    H = make_H_callable(_REF_H0, _REF_OM, 0.0, mode="constant")
    lin = np.linspace(0.0, 1.0, n_linear)
    log = np.expm1(np.linspace(np.log1p(1.0), np.log1p(_Z_STAR - 1e-6), n_log))
    z_grid = np.unique(np.concatenate([lin, log]))
    dm_grid = np.asarray(D_M(z_grid, H, n_steps=4096), dtype=float)
    # Enforce monotone increasing via running-max (numerical noise guard).
    dm_grid = np.maximum.accumulate(dm_grid)
    return z_grid, dm_grid


def ell_of_z(z: np.ndarray) -> np.ndarray:
    """Option A mapping: ℓ(z) = π D_M(z*) / (D_M(z*) − D_M(z))."""
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    H = make_H_callable(_REF_H0, _REF_OM, 0.0, mode="constant")
    dm = np.asarray(D_M(z_arr, H, n_steps=4096), dtype=float)
    dm_star = _ref_DM_to_z_star()
    post_horizon = dm_star - dm
    # At z=0 (dm=0), post_horizon = dm_star → ℓ = π (~3.14).
    # At z→z_star, post_horizon → 0 → ℓ → ∞.
    with np.errstate(divide='ignore', invalid='ignore'):
        ell = np.where(post_horizon > 0, np.pi * dm_star / post_horizon, np.inf)
    return ell


def _z_of_ell(ell: np.ndarray) -> np.ndarray:
    """Inverse mapping: z(ℓ) = D_M^{-1}(D_M(z*) · (1 − π/ℓ))."""
    z_grid, dm_grid = _ref_DM_grid()
    dm_star = _ref_DM_to_z_star()
    ell_arr = np.atleast_1d(np.asarray(ell, dtype=float))
    target_dm = dm_star * (1.0 - np.pi / ell_arr)
    # Clip target_dm into the interpolation range: [0, dm_grid[-1]].
    target_dm = np.clip(target_dm, 0.0, dm_grid[-1])
    # Inverse interpolate: dm_grid is monotonic → np.interp handles this
    # directly (np.interp requires x-values be increasing, which dm_grid is).
    return np.interp(target_dm, dm_grid, z_grid)


# -----------------------------------------------------------------------------
# ε_shape(z) builder
# -----------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _build_epsilon_shape_table() -> Tuple[np.ndarray, np.ndarray]:
    """Precompute the (z, ε_shape) interpolation table from Planck residuals."""
    sig = load_combined_residuals()
    z_for_ell = _z_of_ell(sig.ell)

    # Planck ℓ ≥ 2; ℓ=2,3 map to z ≈ 0-0.1 region (post-recomb horizon ≈ D_M(z*)).
    # ℓ → large maps to z → z_star. No filter needed; all points are valid.
    # But the mapping is non-monotone at ℓ=2 (at the boundary ℓ=π, z=0 exactly).
    # Sort by z to ensure monotonicity for interp.
    order = np.argsort(z_for_ell)
    z_sorted = z_for_ell[order]
    shape_sorted = sig.z_combined[order]
    return z_sorted, shape_sorted


def epsilon_shape(z: np.ndarray) -> np.ndarray:
    """Signed unitless ε-shape, interpolated from Planck residuals; 0 outside
    the Planck ℓ-coverage (zero-extension)."""
    z_data, shape_data = _build_epsilon_shape_table()
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    # Zero-extend: np.interp with left/right=0 returns 0 outside [z_data[0], z_data[-1]].
    return np.interp(z_arr, z_data, shape_data, left=0.0, right=0.0)
