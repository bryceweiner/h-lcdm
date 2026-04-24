"""
Joint log-posterior for the Freedman reproduction MCMC runs.

Free parameters (4 sampled + 1 marginalized):

+-----------+---------------------------------------------------------------+
| H0        | local Hubble constant [km/s/Mpc] — sampled, uniform box        |
+-----------+---------------------------------------------------------------+
| M_TRGB    | absolute TRGB magnitude in the primary band (F814W for 2020,  |
|           | F150W for 2024) — sampled with a Gaussian-tailed prior        |
|           | centered on the published literature value                    |
+-----------+---------------------------------------------------------------+
| E(B-V)    | reddening nuisance — sampled, Gaussian prior whose σ folds    |
|           | in the extinction systematic budget (SFD scaling +            |
|           | zero-point), *not* a separate parameter                       |
+-----------+---------------------------------------------------------------+
| β         | metallicity color slope — sampled, uniform box                |
+-----------+---------------------------------------------------------------+
| M_B       | SN Ia absolute magnitude — analytically marginalized, never   |
|           | sampled (same treatment as expansion_enhancement/chi2_sn)     |
+-----------+---------------------------------------------------------------+

Numerical prior box ranges are frozen in Stage 1 preregistration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Tuple

import numpy as np

from pipeline.expansion_enhancement.cosmology import make_H_callable, mu_model


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PriorBox:
    """Uniform box prior plus optional Gaussian prior on top (for E(B-V))."""

    name: str
    lo: float
    hi: float
    mean: float = 0.0
    sigma: float = 0.0      # 0.0 means pure box, no Gaussian prior

    def log_prior(self, x: float) -> float:
        if not (self.lo <= x <= self.hi):
            return -np.inf
        if self.sigma <= 0.0:
            return 0.0
        return -0.5 * ((x - self.mean) / self.sigma) ** 2


@dataclass(frozen=True)
class FreedmanModelConfig:
    """Freedman reproduction model config (Case A or Case B)."""

    name: str                               # "freedman_2020" or "freedman_2024"
    priors: Tuple[PriorBox, ...]
    param_names: Tuple[str, ...]
    published_H0: float
    published_sigma_stat: float
    published_sigma_sys: float

    @property
    def n_parameters(self) -> int:
        return len(self.param_names)


# Published values from the two CCHP papers.
FREEDMAN_2020_MODEL = FreedmanModelConfig(
    name="freedman_2020",
    param_names=("H0", "M_TRGB", "EBV", "beta"),
    priors=(
        PriorBox("H0", 55.0, 85.0),
        PriorBox("M_TRGB", -5.0, -3.5, mean=-4.047, sigma=0.045),
        PriorBox("EBV", -0.10, 0.30, mean=0.07, sigma=0.03),
        PriorBox("beta", -0.2, 0.6, mean=0.2, sigma=0.1),
    ),
    published_H0=69.8,
    published_sigma_stat=0.8,
    published_sigma_sys=1.7,
)

FREEDMAN_2024_MODEL = FreedmanModelConfig(
    name="freedman_2024",
    param_names=("H0", "M_TRGB", "EBV", "beta"),
    priors=(
        PriorBox("H0", 55.0, 85.0),
        PriorBox("M_TRGB", -5.5, -4.0, mean=-4.362, sigma=0.05),
        PriorBox("EBV", -0.10, 0.30, mean=0.07, sigma=0.03),
        PriorBox("beta", -0.2, 0.6, mean=0.08, sigma=0.1),
    ),
    published_H0=70.39,
    published_sigma_stat=1.22,
    published_sigma_sys=1.33,
)


# ---------------------------------------------------------------------------
# Likelihood ingredients
# ---------------------------------------------------------------------------


@dataclass
class FreedmanLikelihoodInputs:
    """Everything the log-posterior needs, pre-computed per Freedman case."""

    case: str

    # Anchor distance modulus (prior, with σ quadrature of stat + sys).
    mu_anchor: float
    sigma_mu_anchor: float

    # TRGB tip observed in the anchor galaxy (post-extinction).
    I_TRGB_anchor: float
    sigma_I_TRGB_anchor: float

    # Per-host observed I_TRGB (post-extinction) and median color.
    host_names: np.ndarray
    I_TRGB_hosts: np.ndarray
    sigma_I_TRGB_hosts: np.ndarray
    median_color_hosts: np.ndarray
    pivot_color: float

    # SN calibrator apparent magnitudes per host (Pantheon+).
    mB_calibrators: np.ndarray
    sigma_mB_calibrators: np.ndarray
    host_to_calibrator_index: Dict[str, int]

    # Hubble-flow SNe.
    z_flow: np.ndarray
    mu_flow: np.ndarray
    inv_cov_flow: np.ndarray
    Om_flow: float = 0.315

    def n_data(self) -> int:
        return int(self.z_flow.size + self.I_TRGB_hosts.size + 1)


# ---------------------------------------------------------------------------
# Prior / posterior
# ---------------------------------------------------------------------------


def _unpack(theta: np.ndarray) -> Tuple[float, float, float, float]:
    return float(theta[0]), float(theta[1]), float(theta[2]), float(theta[3])


def log_prior(theta: np.ndarray, cfg: FreedmanModelConfig) -> float:
    total = 0.0
    for value, prior in zip(theta, cfg.priors):
        contrib = prior.log_prior(float(value))
        if not np.isfinite(contrib):
            return -np.inf
        total += contrib
    return float(total)


def _anchor_loglike(mu_anchor: float, I_TRGB_anchor: float, M_TRGB: float,
                    sigma_mu_anchor: float, sigma_I: float) -> float:
    """Anchor: μ_anchor = I_TRGB_anchor − M_TRGB (Gaussian)."""
    mu_pred = I_TRGB_anchor - M_TRGB
    sigma = float(np.sqrt(sigma_mu_anchor ** 2 + sigma_I ** 2))
    delta = mu_anchor - mu_pred
    return -0.5 * (delta / sigma) ** 2 - np.log(sigma)


def _hosts_sn_chi2_marginalized(
    theta: np.ndarray,
    cfg: FreedmanModelConfig,
    inputs: FreedmanLikelihoodInputs,
) -> float:
    """SN calibrators + Hubble-flow SNe with M_B analytically marginalized.

    For each calibrator host, the TRGB-derived distance modulus is::

        μ_host = (I_TRGB_host_observed − M_TRGB) − β (color_host − color_pivot)

    This gives ``μ_calib``. The SN M_B is then fit from the joint likelihood
    of (calibrator Δ(m_B − μ) offset + Hubble-flow SNe with m_B = μ_th + M_B),
    analytically profiled over M_B exactly as in expansion_enhancement.
    """
    H0, M_TRGB, EBV, beta = _unpack(theta)

    # Anchor constraint.
    anchor = _anchor_loglike(
        inputs.mu_anchor,
        inputs.I_TRGB_anchor,
        M_TRGB,
        inputs.sigma_mu_anchor,
        inputs.sigma_I_TRGB_anchor,
    )

    # Host TRGB distance moduli.
    mu_calib_pred = (
        inputs.I_TRGB_hosts
        - M_TRGB
        - beta * (inputs.median_color_hosts - inputs.pivot_color)
    )
    sigma_calib = np.sqrt(inputs.sigma_I_TRGB_hosts ** 2 + 0.05 ** 2)
    # (0.05 mag floor absorbs residual host-to-host systematics beyond TRGB
    #  statistical uncertainty; frozen in preregistration.)

    # Match Pantheon+ calibrators to hosts.
    idx = np.array(
        [inputs.host_to_calibrator_index[h] for h in inputs.host_names], dtype=int
    )
    mB_calib = inputs.mB_calibrators[idx]
    sigma_mB_calib = inputs.sigma_mB_calibrators[idx]

    # μ = m_B − M_B; SN part contributes Δ_calib = (m_B − M_B) − μ_calib_pred
    # i.e. delta = (m_B - μ_calib_pred) - M_B. Combine with Hubble-flow:
    # m_B_flow − M_B = μ_th_flow. We profile M_B analytically.

    # Build the joint chi2 piece-by-piece.
    H_func = make_H_callable(H0, float(inputs.Om_flow), 0.0, mode="constant")
    mu_th_flow = mu_model(inputs.z_flow, H_func)

    # Calibrator vector: delta0 = m_B_calib - mu_calib_pred  (⇒ M_B part)
    delta_calib = mB_calib - mu_calib_pred
    sigma_calib_total = np.sqrt(sigma_calib ** 2 + sigma_mB_calib ** 2)

    # Flow vector: delta0 = m_B_flow - (mu_th_flow + ?)   — we actually
    # don't have m_B_flow, we have μ_flow = m_B − M_B already. So the
    # Pantheon+ μ values already absorb M_B; the calibrator anchoring is
    # what determines M_B. Equivalently: we fit H₀ against μ_flow (with
    # analytic M offset), and simultaneously require the TRGB-derived
    # μ_calib for calibrator hosts to match μ_obs_calib with the same M.
    #
    # In the Pantheon+SH0ES release, μ_obs is calibrated to include a
    # consistent M. So the calibrator row's ``μ_obs − μ_pred`` is
    # informative about M_B offset; the flow row's ``μ_obs − μ_th`` is
    # informative about H₀. Marginalize the common offset analytically.

    # Jointly marginalize over the common M (distance-modulus offset that
    # is shared between the Pantheon+ flow block and the TRGB-calibrator
    # block). For a Gaussian likelihood with a shared scalar offset M:
    #
    #   χ²(θ, M) = Δ_flowᵀ C⁻¹ Δ_flow − 2 M·(1ᵀ C⁻¹ Δ_flow) + M² (1ᵀ C⁻¹ 1)
    #             + Σᵢ (Δ_calib,i − M)² / σ_calib,i²
    #
    # Minimising over M gives
    #   M̂ = (1ᵀ C⁻¹ Δ_flow + Σᵢ Δ_calib,i / σ_calib,i²)
    #        / (1ᵀ C⁻¹ 1 + Σᵢ 1/σ_calib,i²)
    # and the profiled χ² is obtained by substituting M̂ back in. Both
    # blocks contribute to M̂ so the calibrator residual is not
    # double-counted against a flow-only profile.
    delta_flow = inputs.mu_flow - mu_th_flow
    inv_cov_flow = inputs.inv_cov_flow
    one_flow = np.ones_like(delta_flow)

    a_flow = float(delta_flow @ inv_cov_flow @ delta_flow)
    b_flow = float(one_flow @ inv_cov_flow @ delta_flow)
    c_flow = float(one_flow @ inv_cov_flow @ one_flow)

    w_calib = 1.0 / np.maximum(sigma_calib_total ** 2, 1e-12)
    a_calib = float(np.sum(w_calib * delta_calib ** 2))
    b_calib = float(np.sum(w_calib * delta_calib))
    c_calib = float(np.sum(w_calib))

    denom = c_flow + c_calib
    M_hat = (b_flow + b_calib) / denom if denom > 0.0 else 0.0
    # Profiled χ² (completing the square):
    chi2_profiled = (a_flow + a_calib) - (b_flow + b_calib) ** 2 / denom

    log_norm = float(-0.5 * np.sum(np.log(2 * np.pi * sigma_calib_total ** 2)))

    return anchor - 0.5 * chi2_profiled + log_norm


def log_posterior(
    theta: np.ndarray,
    cfg: FreedmanModelConfig,
    inputs: FreedmanLikelihoodInputs,
) -> float:
    lp = log_prior(np.asarray(theta, dtype=float), cfg)
    if not np.isfinite(lp):
        return -np.inf
    try:
        ll = _hosts_sn_chi2_marginalized(np.asarray(theta, dtype=float), cfg, inputs)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return -np.inf
    if not np.isfinite(ll):
        return -np.inf
    return float(lp + ll)


__all__ = [
    "FREEDMAN_2020_MODEL",
    "FREEDMAN_2024_MODEL",
    "FreedmanLikelihoodInputs",
    "FreedmanModelConfig",
    "PriorBox",
    "log_posterior",
    "log_prior",
]
