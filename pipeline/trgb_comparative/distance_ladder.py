"""
Distance-ladder assembly and Hubble-flow fitting for the TRGB analysis.

Reuses the distance-integration primitives from
``pipeline.expansion_enhancement.cosmology`` — the same code path that the
expansion_enhancement pipeline uses. We do NOT reimplement D_L / D_M / D_A.

The distance chain itself is case-specific:

- **Case A (Freedman 2019/2020, LMC anchor)**:
  ``LMC DEB → LMC TRGB absolute magnitude → SN host TRGB distances →
   Hubble-flow SNe``.
- **Case B (Freedman 2024/2025, NGC 4258 anchor)**:
  ``NGC 4258 maser → NGC 4258 TRGB absolute magnitude → SN host TRGB
   distances → Hubble-flow SNe``.

The Hubble-flow fit uses the Pantheon+SH0ES sample (already loaded by the
main DataLoader). M_B is analytically marginalized just as in
``pipeline.expansion_enhancement.likelihood.chi2_sn``.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

# Reuse distance functions from expansion_enhancement.
from pipeline.expansion_enhancement.cosmology import (
    C_KMS,
    D_L,
    H_LCDM,
    make_H_callable,
    mu_model,
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class GeometricAnchor:
    """Direct geometric distance to a zero-point galaxy (LMC or NGC 4258)."""

    name: str                # e.g. "LMC" or "NGC_4258"
    mu: float                # distance modulus
    sigma_mu_stat: float
    sigma_mu_sys: float
    reference: str

    @property
    def distance_mpc(self) -> float:
        """Distance in Mpc from the distance modulus."""
        return float(10.0 ** ((self.mu + 5.0) / 5.0) / 1.0e6)

    @property
    def sigma_mu_total(self) -> float:
        return float(np.sqrt(self.sigma_mu_stat ** 2 + self.sigma_mu_sys ** 2))


@dataclass(frozen=True)
class TRGBHostDistance:
    """Distance modulus to a single SN Ia host galaxy from TRGB photometry."""

    host: str
    mu_TRGB: float
    sigma_mu_stat: float
    sigma_mu_sys: float
    M_TRGB: float               # absolute magnitude used for this host
    anchor: str                 # "LMC" or "NGC_4258"
    reference: str = ""

    @property
    def sigma_mu_total(self) -> float:
        return float(np.sqrt(self.sigma_mu_stat ** 2 + self.sigma_mu_sys ** 2))


@dataclass(frozen=True)
class DistanceChain:
    case: str                       # "case_a" or "case_b"
    geometric_anchor: GeometricAnchor
    M_TRGB_absolute: float          # absolute TRGB magnitude from the anchor
    sigma_M_TRGB: float
    host_distances: Tuple[TRGBHostDistance, ...]

    def as_arrays(self) -> Dict[str, np.ndarray]:
        return {
            "host": np.array([h.host for h in self.host_distances]),
            "mu": np.array([h.mu_TRGB for h in self.host_distances], dtype=float),
            "sigma_mu_stat": np.array(
                [h.sigma_mu_stat for h in self.host_distances], dtype=float
            ),
            "sigma_mu_sys": np.array(
                [h.sigma_mu_sys for h in self.host_distances], dtype=float
            ),
        }


# ---------------------------------------------------------------------------
# Chain assembly
# ---------------------------------------------------------------------------


def absolute_trgb_from_anchor_photometry(
    I_TRGB_observed: float,
    sigma_I_TRGB: float,
    anchor: GeometricAnchor,
) -> Tuple[float, float]:
    """Convert observed I_TRGB at the anchor + geometric distance to M_TRGB.

    M_TRGB = I_TRGB_observed − μ_anchor (after extinction & metallicity
    corrections have already been applied upstream).

    Returns (M_TRGB, σ_M_TRGB).
    """
    M = I_TRGB_observed - anchor.mu
    sigma = float(np.sqrt(sigma_I_TRGB ** 2 + anchor.sigma_mu_total ** 2))
    return float(M), sigma


def host_distance_modulus_from_trgb(
    I_TRGB_host: float,
    sigma_I_TRGB_host: float,
    M_TRGB: float,
    sigma_M_TRGB: float,
    anchor_name: str,
    host_name: str,
    reference: str = "",
) -> TRGBHostDistance:
    """μ_host = I_TRGB_host − M_TRGB."""
    mu = I_TRGB_host - M_TRGB
    sigma_stat = float(sigma_I_TRGB_host)
    sigma_sys = float(sigma_M_TRGB)
    return TRGBHostDistance(
        host=host_name,
        mu_TRGB=float(mu),
        sigma_mu_stat=sigma_stat,
        sigma_mu_sys=sigma_sys,
        M_TRGB=float(M_TRGB),
        anchor=anchor_name,
        reference=reference,
    )


def assemble_distance_chain(
    case: str,
    anchor: GeometricAnchor,
    I_TRGB_anchor: float,
    sigma_I_TRGB_anchor: float,
    host_tips: List[Tuple[str, float, float]],
) -> DistanceChain:
    """Build a :class:`DistanceChain` for Case A or Case B.

    ``host_tips`` is a list of (host_name, I_TRGB_host, sigma_I_TRGB_host)
    tuples for the SN Ia host galaxies.
    """
    if case not in ("case_a", "case_b"):
        raise ValueError(f"case must be 'case_a' or 'case_b'; got {case!r}")

    M_TRGB, sigma_M_TRGB = absolute_trgb_from_anchor_photometry(
        I_TRGB_anchor, sigma_I_TRGB_anchor, anchor
    )

    host_distances = tuple(
        host_distance_modulus_from_trgb(
            I_TRGB_host=I_t,
            sigma_I_TRGB_host=sig,
            M_TRGB=M_TRGB,
            sigma_M_TRGB=sigma_M_TRGB,
            anchor_name=anchor.name,
            host_name=name,
        )
        for name, I_t, sig in host_tips
    )

    return DistanceChain(
        case=case,
        geometric_anchor=anchor,
        M_TRGB_absolute=M_TRGB,
        sigma_M_TRGB=sigma_M_TRGB,
        host_distances=host_distances,
    )


# ---------------------------------------------------------------------------
# Hubble-flow fit
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class HubbleFlowFit:
    """Result of a SN Ia Hubble-flow fit for a single H₀ value.

    The fit analytically marginalizes SN M_B just as the expansion_enhancement
    pipeline does. The returned ``H0`` is the maximum-likelihood value from
    a simple 1-parameter grid/bisection — full MCMC with nuisance parameters
    lives in :mod:`pipeline.trgb_comparative.mcmc_runner`.
    """

    H0: float
    sigma_H0_stat: float
    z_flow: np.ndarray
    mu_flow: np.ndarray
    residuals: np.ndarray
    n_sn: int
    Om: float


def _sn_chi2_marginalized_M(
    z: np.ndarray,
    mu_obs: np.ndarray,
    inv_cov: np.ndarray,
    H_func: Callable,
) -> float:
    """SN χ² with M analytically marginalized (same form as expansion_enhancement)."""
    mu_th = mu_model(z, H_func)
    delta0 = mu_obs - mu_th
    one = np.ones_like(delta0)
    a = float(delta0 @ inv_cov @ delta0)
    b = float(one @ inv_cov @ delta0)
    c = float(one @ inv_cov @ one)
    return a - (b * b) / c


def fit_hubble_flow(
    z_flow: np.ndarray,
    mu_flow: np.ndarray,
    inv_cov: np.ndarray,
    Om: float = 0.315,
    H0_grid: Optional[np.ndarray] = None,
) -> HubbleFlowFit:
    """Fit H₀ to the Hubble-flow SN sample with M analytically marginalized.

    Uses a flat-ΛCDM H(z) with Ω_m fixed at the Planck 2018 central value.
    A fine 1-D grid over H₀ is minimized; the 1σ stat uncertainty comes
    from the Δχ² = 1 crossings.
    """
    if H0_grid is None:
        H0_grid = np.linspace(60.0, 85.0, 1001)

    chi2 = np.empty_like(H0_grid)
    for i, H0 in enumerate(H0_grid):
        H_func = make_H_callable(float(H0), float(Om), 0.0, mode="constant")
        chi2[i] = _sn_chi2_marginalized_M(z_flow, mu_flow, inv_cov, H_func)

    i_min = int(np.argmin(chi2))
    H0_best = float(H0_grid[i_min])
    chi2_min = float(chi2[i_min])

    # Δχ² = 1 crossings for 1σ uncertainty.
    above = chi2 - chi2_min > 1.0
    if not above.any():
        sigma_H0 = float(H0_grid[-1] - H0_grid[0]) / 2.0
    else:
        # Find closest lower/upper crossing around i_min.
        lo_idx = np.where(above[:i_min])[0]
        hi_idx = np.where(above[i_min:])[0]
        lo = H0_grid[lo_idx[-1]] if lo_idx.size else H0_grid[0]
        hi = H0_grid[i_min + hi_idx[0]] if hi_idx.size else H0_grid[-1]
        sigma_H0 = 0.5 * float(hi - lo)

    # Residuals at best fit.
    H_func_best = make_H_callable(H0_best, float(Om), 0.0, mode="constant")
    mu_th = mu_model(z_flow, H_func_best)
    delta = mu_flow - mu_th
    one = np.ones_like(delta)
    M_hat = float(one @ inv_cov @ delta) / float(one @ inv_cov @ one)
    residuals = delta - M_hat

    return HubbleFlowFit(
        H0=H0_best,
        sigma_H0_stat=float(sigma_H0),
        z_flow=np.asarray(z_flow),
        mu_flow=np.asarray(mu_flow),
        residuals=residuals,
        n_sn=int(z_flow.size),
        Om=float(Om),
    )


# ---------------------------------------------------------------------------
# Calibrator combination: TRGB-anchored SN absolute magnitude
# ---------------------------------------------------------------------------


def combine_calibrator_distances_for_MB(
    mu_calibrators: np.ndarray,
    sigma_mu: np.ndarray,
    mb_calibrators: np.ndarray,
    sigma_mb: np.ndarray,
) -> Tuple[float, float]:
    """Weighted mean M_B = <m_B − μ_TRGB> over calibrator hosts.

    σ_M_B is the formal uncertainty on the weighted mean accounting for
    per-host σ(m_B − μ_TRGB) in quadrature.
    """
    delta = mb_calibrators - mu_calibrators
    sigma_delta = np.sqrt(sigma_mb ** 2 + sigma_mu ** 2)
    weights = 1.0 / np.maximum(sigma_delta ** 2, 1e-12)
    M_B = float(np.sum(weights * delta) / np.sum(weights))
    sigma_M_B = float(np.sqrt(1.0 / np.sum(weights)))
    return M_B, sigma_M_B


def H0_from_calibrated_MB(
    M_B: float,
    sigma_M_B: float,
    hubble_fit: HubbleFlowFit,
    pantheon_intercept: float,
    sigma_intercept: float,
) -> Tuple[float, float]:
    """Convert a calibrator-derived SN M_B into an H₀ value.

    The Hubble-flow intercept ``a_B = log₁₀(cz) − 0.2 m_B`` relates M_B
    to H₀ via::

        5 log₁₀ H₀ = M_B + 5 a_B + 25

    Returns (H0, sigma_H0).
    """
    log10_H0 = (M_B + 5.0 * pantheon_intercept + 25.0) / 5.0
    H0 = float(10.0 ** log10_H0)
    # Linear error propagation through 5 log₁₀ H₀ = M_B + 5 a_B + 25.
    sigma_log = float(np.sqrt(sigma_M_B ** 2 + 25.0 * sigma_intercept ** 2) / 5.0)
    sigma_H0 = float(H0 * np.log(10.0) * sigma_log)
    return H0, sigma_H0


__all__ = [
    "DistanceChain",
    "GeometricAnchor",
    "HubbleFlowFit",
    "TRGBHostDistance",
    "absolute_trgb_from_anchor_photometry",
    "assemble_distance_chain",
    "combine_calibrator_distances_for_MB",
    "fit_hubble_flow",
    "H0_from_calibrated_MB",
    "host_distance_modulus_from_trgb",
]
