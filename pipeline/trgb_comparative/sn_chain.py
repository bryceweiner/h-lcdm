"""Per-SN-system MCMC chains for each TRGB case.

For each (case, SN photometric system) pair, this module builds a
self-contained chain that samples H₀ alone (profiling M_B analytically)
against the calibrator + Hubble-flow data in a single photometric system.

The public API is a single entry point ``run_chains_for_case`` that loops
over the four SN systems for a given case, returning an ordered
``Dict[system_id, SNChainResult]``. Each result holds a pipeline-computed
MCMC H₀ posterior with its own R̂ convergence diagnostic.

Systems (4 per case, 8 chains total across both cases):

================   ==========================================================
System label       Photometric system / Hubble flow source
================   ==========================================================
``csp_i``          CSP-I reduction (Uddin 2023 h0csp, ``sample=CSPI`` rows)
``csp_ii``         CSP-II reduction (Uddin 2023 h0csp, ``sample=CSPII`` rows)
``supercal``       Scolnic 2015 SuperCal system (Freedman 2019 Table 3
                   ``m_B^SuperCal`` calibrators + Pantheon 2018 Hubble flow,
                   which is built on the SuperCal cross-calibration)
``pantheon_plus``  Pantheon+SH0ES (Brout 2022, Scolnic 2022) calibrators +
                   Pantheon+SH0ES non-calibrator Hubble flow
================   ==========================================================

Case selection (A = LMC anchor, B = NGC 4258 anchor) only changes the
TRGB distance modulus assigned to each calibrator:

- Case A uses the Freedman 2019 Table 3 LMC-anchored μ_TRGB values;
  Uddin's ``calibrators_trgb_f19.csv`` reproduces these.
- Case B uses the Freedman 2025 Table 2 NGC 4258-anchored μ_TRGB values.

Likelihood discipline (shared across all 8 chains):

- Sampled parameter: **H₀ only** (range [55, 85] km/s/Mpc, uniform box).
- **M_B analytically marginalized** (not sampled) — the SN absolute
  magnitude drops out of the joint calibrator+flow posterior once we
  profile over the common distance-modulus offset.
- Intrinsic scatter σ_int fixed at 0.10 mag (frozen in Stage 1
  preregistration amendment).

The convergence gate is R̂ < 1.01. A chain that fails to converge is
reported with its R̂ value and the ``converged=False`` flag — the
pipeline never silently promotes a non-converged posterior.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from data.loader import DataLoader, DataUnavailableError

from .data_loaders import SN_TO_HOST
from .mcmc_runner import MCMCSettings

logger = logging.getLogger(__name__)


C_KMS: float = 299_792.458

# --- Hubble flow redshift cuts (Freedman 2019 §6.3) --------------------------
HUBBLE_FLOW_Z_MIN: float = 0.023
HUBBLE_FLOW_Z_MAX: float = 0.15

# Peculiar-velocity floor added in quadrature to σ_m_B for flow SNe.
SIGMA_VPEC_KMS: float = 250.0

# Intrinsic SN Ia scatter (mag) — frozen in the Stage 1 preregistration
# amendment that introduces the per-system chains. This is the common value
# adopted by Freedman 2019/2025 and Hoyt 2025.
SIGMA_INT_MAG: float = 0.10

# Matter density used for the low-z luminosity-distance ΛCDM correction. The
# Hubble flow sample sits at z < 0.15 where Ω_m enters at the ≲ 0.5 % level;
# this parameter is preregistered at Planck 2018's central value.
OMEGA_M_FLOW: float = 0.315


# =============================================================================
# Chain data container
# =============================================================================


@dataclass(frozen=True)
class SNChainData:
    """Inputs for a single (case, SN system) MCMC chain.

    All magnitudes are on the same photometric system. ``mu_TRGB`` values
    are in the appropriate geometric-anchor scale (LMC for Case A, NGC
    4258 for Case B).
    """

    case: str                          # 'case_a' or 'case_b'
    system: str                        # 'csp_i', 'csp_ii', 'supercal', 'pantheon_plus'
    system_label: str                  # human-readable description

    # Calibrators (TRGB-distance-calibrated SNe):
    calibrator_sn_names: np.ndarray    # (N_cal,) strings
    calibrator_hosts: np.ndarray       # (N_cal,) strings
    calibrator_mB: np.ndarray          # (N_cal,) standardized peak B mag
    calibrator_sigma_mB: np.ndarray    # (N_cal,) 1σ
    calibrator_mu_TRGB: np.ndarray     # (N_cal,) TRGB distance moduli
    calibrator_sigma_mu_TRGB: np.ndarray

    # Hubble flow SNe:
    flow_sn_names: np.ndarray          # (N_flow,) strings
    flow_zcmb: np.ndarray              # (N_flow,) CMB-frame redshift
    flow_mB: np.ndarray                # (N_flow,) standardized peak B mag
    flow_sigma_mB: np.ndarray          # (N_flow,) 1σ

    # Metadata for the report:
    published_target_H0: float = 0.0
    published_sigma_stat: float = 0.0
    published_sigma_sys: float = 0.0
    notes: str = ""

    def n_calibrators(self) -> int:
        return int(self.calibrator_mB.size)

    def n_flow(self) -> int:
        return int(self.flow_mB.size)

    def n_data(self) -> int:
        return self.n_calibrators() + self.n_flow()


# =============================================================================
# Physics — luminosity distance (low-z ΛCDM)
# =============================================================================


def _E_flat_LCDM(z: np.ndarray, Om: float) -> np.ndarray:
    return np.sqrt(Om * (1.0 + z) ** 3 + (1.0 - Om))


def _d_L_Mpc(z: np.ndarray, H0: float, Om: float = OMEGA_M_FLOW) -> np.ndarray:
    """Luminosity distance in Mpc under a flat ΛCDM cosmology.

    Uses a simple trapezoid integration on a fine grid; at z < 0.15 this
    is numerically indistinguishable from a quadrature evaluation.
    """
    z = np.atleast_1d(np.asarray(z, dtype=float))
    zmax = float(z.max()) if z.size else 0.0
    if zmax <= 0.0:
        return np.zeros_like(z)
    n_grid = max(400, int(zmax / 1e-4))
    zg = np.linspace(0.0, zmax, n_grid)
    integrand = 1.0 / _E_flat_LCDM(zg, Om)
    # Cumulative comoving distance in units of c/H₀.
    chi = np.concatenate(([0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(zg))))
    chi_z = np.interp(z, zg, chi)                              # (c/H₀) factor pulled out
    d_C = (C_KMS / H0) * chi_z
    return (1.0 + z) * d_C


def _mu_model(z: np.ndarray, H0: float) -> np.ndarray:
    dL = _d_L_Mpc(z, H0)
    with np.errstate(divide="ignore"):
        return 5.0 * np.log10(np.maximum(dL, 1e-6)) + 25.0


# =============================================================================
# Likelihood
# =============================================================================


def log_posterior_sn_chain(theta: np.ndarray, chain: SNChainData) -> float:
    """log-posterior for one per-system chain.

    Sampled parameter vector is (H0,). M_B is analytically marginalized
    over the joint (calibrator + flow) block.

    Model:
        calibrator i : m_B,i = μ_TRGB,i + M_B + ε,  σ_i² = σ_mB,i² + σ_μ,i² + σ_int²
        flow j       : m_B,j = μ_th(z_j, H0) + M_B + δ,
                       σ_j² = σ_mB,j² + (5/ln 10)² · σ_vpec/cz² + σ_int²

    Marginalizing M_B analytically profiles the common-offset term away.
    Result is a 1-D posterior on H₀ alone.
    """
    theta = np.atleast_1d(np.asarray(theta, dtype=float))
    H0 = float(theta[0])
    if not (55.0 <= H0 <= 85.0):
        return -np.inf

    # Calibrators: residual Δ_cal,i = m_B,i − μ_TRGB,i — informative about M_B.
    delta_cal = chain.calibrator_mB - chain.calibrator_mu_TRGB
    sigma_cal2 = (
        chain.calibrator_sigma_mB ** 2
        + chain.calibrator_sigma_mu_TRGB ** 2
        + SIGMA_INT_MAG ** 2
    )
    w_cal = 1.0 / np.maximum(sigma_cal2, 1e-12)

    # Flow: residual Δ_flow,j = m_B,j − μ_th(z_j, H0) — jointly informative about
    # (M_B, H₀). σ includes a peculiar-velocity floor translated to magnitude
    # space via σ_mu = (5/ln 10) · σ_v / cz.
    mu_th = _mu_model(chain.flow_zcmb, H0)
    delta_flow = chain.flow_mB - mu_th
    sigma_vpec_mag = (5.0 / np.log(10.0)) * SIGMA_VPEC_KMS / (C_KMS * chain.flow_zcmb)
    sigma_flow2 = (
        chain.flow_sigma_mB ** 2
        + sigma_vpec_mag ** 2
        + SIGMA_INT_MAG ** 2
    )
    w_flow = 1.0 / np.maximum(sigma_flow2, 1e-12)

    # Analytic M_B marginalization. With χ² = Σ (Δ - M_B)² · w for both blocks,
    # the profiled M̂_B = Σ Δ·w / Σ w. Substituting back:
    # χ²_profiled = Σ Δ²·w − (Σ Δ·w)² / (Σ w).
    sw = float(w_cal.sum() + w_flow.sum())
    if sw <= 0.0:
        return -np.inf
    sdw = float((delta_cal * w_cal).sum() + (delta_flow * w_flow).sum())
    sd2w = float(((delta_cal ** 2) * w_cal).sum() + ((delta_flow ** 2) * w_flow).sum())

    chi2 = sd2w - (sdw ** 2) / sw
    # Gaussian normalization (M_B marginalization adds a -0.5 · log(sw) term
    # from the profiled-likelihood Laplace approximation):
    log_norm = -0.5 * float(np.log(sigma_cal2).sum() + np.log(sigma_flow2).sum())
    log_norm += -0.5 * float(np.log(sw))

    return -0.5 * chi2 + log_norm


# =============================================================================
# MCMC runner
# =============================================================================


@dataclass
class SNChainResult:
    """Result of one (case, system) MCMC chain."""

    case: str
    system: str
    system_label: str
    H0_median: float
    H0_sigma: float                    # 68 % half-width
    H0_lo: float
    H0_hi: float
    rhat_H0: float
    converged: bool
    convergence_gate: float = 1.01
    n_walkers: int = 0
    n_steps: int = 0
    n_burnin: int = 0
    n_data: int = 0
    mean_acceptance: float = 0.0
    samples: Optional[np.ndarray] = None
    published_target_H0: float = 0.0
    published_sigma_stat: float = 0.0
    published_sigma_sys: float = 0.0
    notes: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "case": self.case,
            "system": self.system,
            "system_label": self.system_label,
            "H0_median": float(self.H0_median),
            "H0_sigma": float(self.H0_sigma),
            "H0_lo": float(self.H0_lo),
            "H0_hi": float(self.H0_hi),
            "rhat_H0": float(self.rhat_H0),
            "converged": bool(self.converged),
            "convergence_gate": float(self.convergence_gate),
            "n_walkers": int(self.n_walkers),
            "n_steps": int(self.n_steps),
            "n_burnin": int(self.n_burnin),
            "n_data": int(self.n_data),
            "mean_acceptance": float(self.mean_acceptance),
            "published_target_H0": float(self.published_target_H0),
            "published_sigma_stat": float(self.published_sigma_stat),
            "published_sigma_sys": float(self.published_sigma_sys),
            "notes": self.notes,
        }


def run_sn_chain(
    chain: SNChainData,
    settings: MCMCSettings,
    chain_out_path: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> SNChainResult:
    """Run a single per-system chain with emcee.

    Returns an :class:`SNChainResult` carrying median, 68 % interval,
    Gelman-Rubin R̂, and convergence state. The ``samples`` field is the
    flat post-burn-in 1-D H₀ posterior for downstream analysis.
    """
    import emcee

    _log = log_fn or (lambda m: logger.info(m))

    n_cal = chain.n_calibrators()
    n_flow = chain.n_flow()
    if n_cal < 3:
        raise ValueError(
            f"Chain {chain.case}/{chain.system}: need ≥ 3 calibrators; got {n_cal}."
        )
    if n_flow < 10:
        raise ValueError(
            f"Chain {chain.case}/{chain.system}: need ≥ 10 flow SNe; got {n_flow}."
        )

    rng = np.random.default_rng(settings.seed)
    n_walkers = int(settings.n_walkers)
    # Initialize H₀ walkers on a tight Gaussian around 72 km/s/Mpc (mid of
    # the 55–85 prior; neither Freedman-2019 nor SH0ES-style to avoid
    # steering the start point).
    p0 = 72.0 + 2.0 * rng.standard_normal((n_walkers, 1))
    p0 = np.clip(p0, 56.0, 84.0)

    _log(
        f"[{chain.case}/{chain.system}] emcee: {n_walkers} walkers × "
        f"{settings.n_steps} steps ({settings.n_burnin} burn-in); "
        f"cal N={n_cal}, flow N={n_flow}"
    )

    sampler = emcee.EnsembleSampler(
        n_walkers, 1, log_posterior_sn_chain, args=(chain,),
    )

    t0 = time.time()
    sampler.run_mcmc(p0, settings.n_steps, progress=settings.progress)
    dt = time.time() - t0
    mean_acc = float(sampler.acceptance_fraction.mean())
    _log(
        f"[{chain.case}/{chain.system}] chain ran in {dt:.1f}s; "
        f"mean acceptance = {mean_acc:.3f}"
    )

    chain_full = sampler.get_chain()                    # (n_steps, n_walkers, 1)
    post = chain_full[settings.n_burnin:]               # (n_post, n_walkers, 1)
    flat = post.reshape(-1)

    # Gelman-Rubin on H₀ only (single parameter).
    n_steps_post, n_w, _ = post.shape
    chain_means = post[:, :, 0].mean(axis=0)            # (n_walkers,)
    chain_vars = post[:, :, 0].var(axis=0, ddof=1)
    grand = chain_means.mean()
    B = n_steps_post * ((chain_means - grand) ** 2).sum() / (n_w - 1)
    W = chain_vars.mean()
    var_hat = (n_steps_post - 1) / n_steps_post * W + B / n_steps_post
    rhat = float(np.sqrt(var_hat / W)) if W > 0 else float("nan")
    converged = bool(rhat < 1.01)

    lo, med, hi = np.percentile(flat, [16.0, 50.0, 84.0])
    sigma = 0.5 * (hi - lo)

    _log(
        f"[{chain.case}/{chain.system}] H₀ = {med:.3f} ({lo:.3f} - {hi:.3f}); "
        f"σ = {sigma:.3f}; R̂ = {rhat:.4f} "
        f"({'CONVERGED' if converged else 'NOT CONVERGED'})"
    )

    if chain_out_path is not None:
        chain_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            chain_out_path,
            chain=chain_full,
            post_flat=flat,
            n_burnin=settings.n_burnin,
            n_walkers=n_walkers,
            n_steps=settings.n_steps,
            rhat=rhat,
            system=chain.system,
            case=chain.case,
            cal_sn_names=chain.calibrator_sn_names,
            cal_mB=chain.calibrator_mB,
            cal_sigma_mB=chain.calibrator_sigma_mB,
            cal_mu_TRGB=chain.calibrator_mu_TRGB,
            cal_sigma_mu_TRGB=chain.calibrator_sigma_mu_TRGB,
            flow_zcmb=chain.flow_zcmb,
            flow_mB=chain.flow_mB,
            flow_sigma_mB=chain.flow_sigma_mB,
        )
        _log(f"[{chain.case}/{chain.system}] chain saved → {chain_out_path}")

    return SNChainResult(
        case=chain.case,
        system=chain.system,
        system_label=chain.system_label,
        H0_median=float(med),
        H0_sigma=float(sigma),
        H0_lo=float(lo),
        H0_hi=float(hi),
        rhat_H0=rhat,
        converged=converged,
        convergence_gate=1.01,
        n_walkers=n_walkers,
        n_steps=int(settings.n_steps),
        n_burnin=int(settings.n_burnin),
        n_data=chain.n_data(),
        mean_acceptance=mean_acc,
        samples=flat,
        published_target_H0=chain.published_target_H0,
        published_sigma_stat=chain.published_sigma_stat,
        published_sigma_sys=chain.published_sigma_sys,
        notes=chain.notes,
    )


# =============================================================================
# Host-name normalization (cross-system crosswalk)
# =============================================================================


def _normalize_host(host: str) -> str:
    """Canonicalize host galaxy names across archives.

    Uddin uses compact forms like 'N1316', Freedman 2019 Table 3 uses
    'NGC 1316' (our canonical), Pantheon+ stores SN names only.
    """
    if not isinstance(host, str):
        return ""
    s = host.strip()
    if s.upper().startswith("N") and s[1:].isdigit():
        return f"NGC {s[1:]}"
    return s.replace("NGC", "NGC ").replace("  ", " ").strip()


__all__ = [
    "HUBBLE_FLOW_Z_MAX",
    "HUBBLE_FLOW_Z_MIN",
    "OMEGA_M_FLOW",
    "SIGMA_INT_MAG",
    "SIGMA_VPEC_KMS",
    "SNChainData",
    "SNChainResult",
    "log_posterior_sn_chain",
    "run_sn_chain",
]
