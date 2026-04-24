"""
Joint χ² / log-posterior for BAO + SN + CMB distance constraints.

Parameterization
----------------
Model A (standard ΛCDM):   θ = (H₀, Ω_m).                r_d = 147.5 Mpc, ε = 0.
Model B (framework):        θ = (H₀, Ω_m, ε).             r_d = 150.71 Mpc.
Model B_qtep: identical to Model B but ε(z) = ε·γ(z)/γ(0) rather than constant.

Pantheon+ absolute-magnitude nuisance M is analytically profiled out so the SN
block contributes shape (not overall normalization) information. See ``chi2_sn``
below for the closed-form profile.

Priors
------
Uniform: H₀ ∈ [60, 80] km/s/Mpc, Ω_m ∈ [0.2, 0.4], ε ∈ [0, 0.1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from scipy.integrate import quad

from .cosmology import (
    C_KMS,
    D_H,
    D_M,
    D_V,
    EpsilonMode,
    make_H_callable,
    mu_model,
)
from .data_loaders import BAOData, CMBDistancePrior, ExpansionDataBundle, SNData


# -----------------------------------------------------------------------------
# Model configuration
# -----------------------------------------------------------------------------

R_D_LCDM: float = 147.5      # pipeline/bao/bao_pipeline.py:57-61
R_D_FRAMEWORK: float = 150.71  # pipeline/bao/bao_pipeline.py:57-61, docs/bao_resolution_qit.tex


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for one of the three models under comparison."""

    name: str
    r_d: float
    epsilon_mode: EpsilonMode    # ignored when has_epsilon=False
    has_epsilon: bool            # True for Model B variants, False for Model A

    @property
    def n_parameters(self) -> int:
        return 3 if self.has_epsilon else 2

    @property
    def param_names(self) -> list:
        return ["H0", "Om", "eps"] if self.has_epsilon else ["H0", "Om"]


MODEL_A = ModelConfig(name="A_lcdm", r_d=R_D_LCDM, epsilon_mode="constant", has_epsilon=False)
MODEL_B_CONST = ModelConfig(name="B_const", r_d=R_D_FRAMEWORK, epsilon_mode="constant", has_epsilon=True)
MODEL_B_QTEP = ModelConfig(name="B_qtep", r_d=R_D_FRAMEWORK, epsilon_mode="qtep", has_epsilon=True)
# Residual-shape model: ε(z) = ε_amp × signed IVW Planck residual, via
# post-recombination horizon mapping. See ``cmb_residuals.py``.
MODEL_B_RESIDUALS = ModelConfig(
    name="B_residuals", r_d=R_D_FRAMEWORK, epsilon_mode="residuals", has_epsilon=True,
)


# Priors are uniform (box) unless otherwise noted. ε for the residual-shape
# model is an amplitude multiplying a signed unitless z-score, so we allow it
# to range over a wider interval than the raw-constant ε prior.
PRIORS = {
    "H0": (60.0, 80.0),
    "Om": (0.2, 0.4),
    "eps": (0.0, 0.1),
}

# Model-specific prior overrides. Residual-shape ε_amp needs a broader window
# because the z-score magnitude is O(1-3) σ and the physics prediction maps
# to ε(z) ~ 0.02 at peak — so ε_amp ~ 0.005-0.03 is the natural range. We
# allow the amplitude into a slightly wider interval to let the data speak.
PRIORS_BY_MODEL = {
    "B_residuals": {
        "H0": (60.0, 80.0),
        "Om": (0.2, 0.4),
        "eps": (0.0, 0.05),
    },
}


def _unpack(theta: np.ndarray, cfg: ModelConfig) -> Tuple[float, float, float]:
    if cfg.has_epsilon:
        H0, Om, eps = float(theta[0]), float(theta[1]), float(theta[2])
    else:
        H0, Om = float(theta[0]), float(theta[1])
        eps = 0.0
    return H0, Om, eps


def _priors_for(cfg: ModelConfig) -> dict:
    """Merge global priors with per-model overrides."""
    return {**PRIORS, **PRIORS_BY_MODEL.get(cfg.name, {})}


def _in_prior(theta: np.ndarray, cfg: ModelConfig) -> bool:
    H0, Om, eps = _unpack(theta, cfg)
    p = _priors_for(cfg)
    H0_lo, H0_hi = p["H0"]
    Om_lo, Om_hi = p["Om"]
    if not (H0_lo <= H0 <= H0_hi):
        return False
    if not (Om_lo <= Om <= Om_hi):
        return False
    if cfg.has_epsilon:
        eps_lo, eps_hi = p["eps"]
        if not (eps_lo <= eps <= eps_hi):
            return False
    return True


# -----------------------------------------------------------------------------
# Per-dataset χ²
# -----------------------------------------------------------------------------

def _bao_model_vector(data: BAOData, H_func, r_d: float) -> np.ndarray:
    """Model prediction for every DESI DR1 row, in the order of ``data.kind``."""
    predictions = np.zeros_like(data.value)
    for i, (zi, kind) in enumerate(zip(data.z, data.kind)):
        if kind == "D_M/r_d":
            predictions[i] = float(np.atleast_1d(D_M(zi, H_func))[0]) / r_d
        elif kind == "D_H/r_d":
            predictions[i] = float(np.atleast_1d(D_H(zi, H_func))[0]) / r_d
        elif kind == "D_V/r_d":
            predictions[i] = float(np.atleast_1d(D_V(zi, H_func))[0]) / r_d
        else:
            raise ValueError(f"Unknown BAO observable: {kind!r}")
    return predictions


def chi2_bao(theta: np.ndarray, cfg: ModelConfig, data: BAOData) -> float:
    H0, Om, eps = _unpack(theta, cfg)
    H = make_H_callable(H0, Om, eps, mode=cfg.epsilon_mode)
    model = _bao_model_vector(data, H, cfg.r_d)
    delta = data.value - model
    return float(delta @ data.inv_cov @ delta)


def chi2_sn(theta: np.ndarray, cfg: ModelConfig, data: SNData) -> float:
    """Pantheon+ χ² with analytic marginalization over the M nuisance.

    Model: μ_obs = μ_theory(z; θ) + M. With a flat prior on M, the profile
    likelihood gives

        χ²(θ) = Δ₀ᵀ C⁻¹ Δ₀  −  (1ᵀ C⁻¹ Δ₀)² / (1ᵀ C⁻¹ 1)

    where Δ₀ = μ_obs − μ_theory. This removes the absolute-magnitude
    degeneracy so the SN block contributes shape (Ω_m-sensitive) constraints
    but not H₀ directly.
    """
    H0, Om, eps = _unpack(theta, cfg)
    H = make_H_callable(H0, Om, eps, mode=cfg.epsilon_mode)
    mu_th = mu_model(data.z, H)

    delta0 = data.mu - mu_th
    inv = data.inv_cov
    # 1ᵀ C⁻¹ Δ₀ and 1ᵀ C⁻¹ 1 — both scalars.
    one = np.ones_like(delta0)
    a = delta0 @ inv @ delta0
    b = one @ inv @ delta0
    c = one @ inv @ one
    return float(a - (b * b) / c)


def chi2_cmb(theta: np.ndarray, cfg: ModelConfig, data: CMBDistancePrior) -> float:
    """CMB distance-prior χ² on D_M(z*).

    Directly compares model-predicted D_M(z_rec) to the Planck 2018 derived
    value (13869.7 ± 4.4 Mpc). Independent of the framework's r_d choice —
    see ``load_planck_2018_theta_star`` docstring for why this is preferred
    over a θ* constraint.

    Uses the vectorized ``D_M`` integrator with n_steps=16384 rather than
    scipy.quad: (i) 0.3ms vs 0.5ms per call, and (ii) H_func gets called once
    on a 16385-point grid instead of scalar-at-a-time, which is critical for
    the ``residuals`` ε mode whose shape function involves np.interp.
    """
    H0, Om, eps = _unpack(theta, cfg)
    H = make_H_callable(H0, Om, eps, mode=cfg.epsilon_mode)
    dm = float(np.atleast_1d(D_M(data.z_rec, H, n_steps=16384))[0])
    r = (data.D_M_z_rec - dm) / data.sigma_D_M
    return float(r * r)


# -----------------------------------------------------------------------------
# Log-posterior (what emcee calls)
# -----------------------------------------------------------------------------

def chi2_total(
    theta: np.ndarray,
    cfg: ModelConfig,
    bundle: ExpansionDataBundle,
) -> dict:
    """Return per-dataset and total χ². Useful for diagnostics + reporting."""
    b = chi2_bao(theta, cfg, bundle.bao)
    s = chi2_sn(theta, cfg, bundle.sn)
    c = chi2_cmb(theta, cfg, bundle.cmb)
    return {"bao": b, "sn": s, "cmb": c, "total": b + s + c}


def log_posterior(
    theta: np.ndarray,
    cfg: ModelConfig,
    bundle: ExpansionDataBundle,
) -> float:
    """log P(θ | data) ∝ -½ χ² inside the box prior, -inf outside."""
    theta = np.asarray(theta, dtype=float)
    if not _in_prior(theta, cfg):
        return -np.inf
    try:
        chi2 = (
            chi2_bao(theta, cfg, bundle.bao)
            + chi2_sn(theta, cfg, bundle.sn)
            + chi2_cmb(theta, cfg, bundle.cmb)
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return -np.inf
    if not np.isfinite(chi2):
        return -np.inf
    return -0.5 * chi2
