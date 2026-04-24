"""Uddin 8-parameter CSP MCMC chain — faithful reproduction of Uddin 2023.

Reproduces the H₀ MCMC chain in
:file:`scripts/H0CSP.py` of the Uddin 2023 repository
(https://github.com/syeduddin/h0csp), using the unified
``B_trgb_update3.csv`` input table.

Sampled parameters (8):
    M_B, p1, p2, β, α, σ_int, v_pec, H0

Likelihood (per SN):
    μ_obs(i) = Mmax(i) − M_B − p1·(st(i)−1) − p2·(st(i)−1)² − β·BV(i)
               − α·(m_host(i) − median(m_host))

    μ_model(i) = dist(i)                             if calibrator (dist > 0)
                = 5·log10[(1+zhel)/(1+zcmb) · c·zcmb/H₀ · (1 + (1−q)·zcmb/2)] + 25
                  with q = -0.53                      if Hubble flow (dist < 0)

    σ²(i)     = eMmax² + ((p1+2·p2·(st−1))·est)² + (β·eBV)²
                − 2·(p1+2·p2·(st−1))·covMs
                + 2·β·(p1+2·p2·(st−1))·covBVs
                − 2·(p1+2·p2·(st−1))·covBV_M
                + (α·em_host)² + σ_int²
                + peculiar-velocity term for flow:
                  ((5/ln 10)·v_pec/(c·zcmb))²
                or TRGB edist² for calibrators

Case switching (LMC → NGC 4258) happens upstream by updating ``dist`` in
the calibrator block to the Freedman 2025 Table 2 μ_bar value for each
host (Case B); Case A uses the file's native LMC-anchored μ_TRGB.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .mcmc_runner import MCMCSettings

logger = logging.getLogger(__name__)

C_KMS: float = 299_792.458
Q0: float = -0.53                                      # Uddin's fixed q₀


# =============================================================================
# Inputs
# =============================================================================


@dataclass
class UddinCSPInputs:
    """All arrays the Uddin 8-parameter likelihood needs.

    Calibrators + flow are concatenated into a single block; ``is_cal`` is
    a boolean mask marking which rows are TRGB calibrators.
    """

    sn_names: np.ndarray
    zhel: np.ndarray
    zcmb: np.ndarray
    st: np.ndarray
    est: np.ndarray
    Mmax: np.ndarray
    eMmax: np.ndarray
    BV: np.ndarray
    eBV: np.ndarray
    covMs: np.ndarray            # cov(Mmax, s_BV)
    covBV_M: np.ndarray          # cov(BV, Mmax)
    covBVs: np.ndarray           # cov(BV, s_BV)
    m_hostmass: np.ndarray       # log stellar host mass
    ml_hostmass: np.ndarray
    mu_hostmass: np.ndarray
    dist: np.ndarray             # TRGB μ for calibrators; arbitrary for flow
    edist: np.ndarray
    is_cal: np.ndarray           # bool
    median_mhost: float
    # The error on m_hostmass (symmetric approximation of the upper-lower
    # bounds); precomputed to avoid recomputing inside the log-likelihood.
    em_hostmass: np.ndarray

    @property
    def n_total(self) -> int:
        return int(self.Mmax.size)

    @property
    def n_cal(self) -> int:
        return int(self.is_cal.sum())

    @property
    def n_flow(self) -> int:
        return int((~self.is_cal).sum())


def build_uddin_inputs_from_loader_dataset(
    uddin_dataset: Dict[str, Dict[str, np.ndarray]],
    *,
    case_b_host_mu_override: Optional[Dict[str, Tuple[float, float]]] = None,
    flow_sample_filter: Optional[str] = None,
) -> UddinCSPInputs:
    """Assemble UddinCSPInputs from ``DataLoader.load_uddin_h0csp_trgb_dataset``.

    Parameters
    ----------
    uddin_dataset : dict
        The return value of
        :py:meth:`data.loader.DataLoader.load_uddin_h0csp_trgb_dataset`.
    case_b_host_mu_override : mapping, optional
        If provided, a ``{canonical_host: (mu_NGC4258, sigma)}`` mapping.
        For each calibrator whose host is in the mapping, its ``dist`` is
        replaced with the NGC 4258-anchored value (Case B). Calibrators
        whose host is not in the mapping are **dropped** so Case B only
        sees NGC 4258-calibrated rows.
    flow_sample_filter : {'CSPI', 'CSPII', 'both', None}
        Restrict the Hubble-flow block to one CSP release. ``None`` or
        ``'both'`` keeps all flow rows.
    """
    cal = uddin_dataset["calibrators"]
    flow = uddin_dataset["flow"]

    # Merge the two blocks so the likelihood sees a single concatenated
    # array; is_cal mask distinguishes rows.
    def _vec(d, k):
        return np.asarray(d[k])

    keys_num = [
        "zhel", "zcmb", "st", "est", "Mmax", "eMmax", "BV", "eBV",
        "covMs", "covBV_M", "covBVs", "ml", "m_hostmass", "mu_hostmass",
        "dist", "edist",
    ]
    keys_str = ["SN_name", "host", "sample", "caltype"]

    def _select_flow(f: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if flow_sample_filter is None or flow_sample_filter == "both":
            return f
        if flow_sample_filter not in ("CSPI", "CSPII"):
            raise ValueError(
                f"flow_sample_filter must be 'CSPI', 'CSPII', 'both', or None; "
                f"got {flow_sample_filter!r}"
            )
        mask = _vec(f, "sample") == flow_sample_filter
        return {k: np.asarray(v)[mask] for k, v in f.items()}

    flow = _select_flow(flow)

    # Apply Case B override to calibrators.
    if case_b_host_mu_override is not None:
        hosts_cal = _vec(cal, "host")
        keep_mask = np.zeros(hosts_cal.size, dtype=bool)
        new_dist = np.asarray(cal["dist"], dtype=float).copy()
        new_edist = np.asarray(cal["edist"], dtype=float).copy()
        for i, h in enumerate(hosts_cal):
            hs = str(h).strip()
            hc = (
                f"NGC {hs[1:]}" if hs.upper().startswith("N") and hs[1:].isdigit()
                else hs
            )
            rec = case_b_host_mu_override.get(hc)
            if rec is None:
                continue
            keep_mask[i] = True
            new_dist[i], new_edist[i] = rec
        cal = {k: (np.asarray(v)[keep_mask] if k not in ("SN_name", "host", "sample",
                                                          "subtype", "caltype")
                   else np.asarray(v)[keep_mask])
               for k, v in cal.items()}
        cal["dist"] = new_dist[keep_mask]
        cal["edist"] = new_edist[keep_mask]

    # Concatenate numeric columns, tracking is_cal.
    n_cal = int(np.asarray(cal["zcmb"]).size)
    n_flow = int(np.asarray(flow["zcmb"]).size)
    is_cal = np.concatenate([np.ones(n_cal, dtype=bool), np.zeros(n_flow, dtype=bool)])

    cat = {}
    for k in keys_num:
        cat[k] = np.concatenate([
            np.asarray(cal[k], dtype=float),
            np.asarray(flow[k], dtype=float),
        ])
    sn_names = np.concatenate([_vec(cal, "SN_name"), _vec(flow, "SN_name")])

    # Host-mass error (symmetric avg of upper/lower 1σ-ish quantities).
    em = 0.5 * ((cat["mu_hostmass"] - cat["m_hostmass"])
                + (cat["m_hostmass"] - cat["ml"]))
    em = np.where(em == 0.0, 0.005, em)                # matches H0CSP.py

    median_mhost = float(np.median(cat["m_hostmass"][np.isfinite(cat["m_hostmass"])]))

    return UddinCSPInputs(
        sn_names=sn_names,
        zhel=cat["zhel"], zcmb=cat["zcmb"],
        st=cat["st"], est=cat["est"],
        Mmax=cat["Mmax"], eMmax=cat["eMmax"],
        BV=cat["BV"], eBV=cat["eBV"],
        covMs=cat["covMs"], covBV_M=cat["covBV_M"], covBVs=cat["covBVs"],
        m_hostmass=cat["m_hostmass"],
        ml_hostmass=cat["ml"], mu_hostmass=cat["mu_hostmass"],
        dist=cat["dist"], edist=cat["edist"],
        is_cal=is_cal,
        median_mhost=median_mhost,
        em_hostmass=em,
    )


# =============================================================================
# Likelihood — exact Uddin H0CSP.py form
# =============================================================================


def _distmod_uddin(H0: float, zhel: np.ndarray, zcmb: np.ndarray) -> np.ndarray:
    """Uddin's Eq. 9 form of the ΛCDM distance modulus with q₀ = -0.53."""
    t1 = (1.0 + zhel) / (1.0 + zcmb)
    t2 = (C_KMS * zcmb) / H0
    t3 = 1.0 + ((1.0 - Q0) / 2.0) * zcmb
    return 5.0 * np.log10(t1 * t2 * t3) + 25.0


def log_posterior_uddin(theta: np.ndarray, inputs: UddinCSPInputs) -> float:
    """8-parameter log-posterior, exact Uddin H0CSP.py form."""
    theta = np.asarray(theta, dtype=float)
    if theta.size != 8:
        return -np.inf
    M_B, p1, p2, beta, alpha, sig_int, v_pec, H0 = theta

    # Priors (box).
    if not (-25.0 < M_B < 14.0):
        return -np.inf
    if not (-10.0 < p1 < 10.0):
        return -np.inf
    if not (-10.0 < p2 < 10.0):
        return -np.inf
    if not (0.0 < beta < 10.0):
        return -np.inf
    if not (-1.0 < alpha < 1.0):
        return -np.inf
    if not (0.0 < sig_int < 1.0):
        return -np.inf
    if not (0.0 < v_pec < 1000.0):
        return -np.inf
    if not (55.0 < H0 < 85.0):
        return -np.inf

    st_minus_1 = inputs.st - 1.0
    fac = p1 + 2.0 * p2 * st_minus_1                   # d(mu_obs)/d(s_BV)

    # Host-mass centered.
    m_centered = inputs.m_hostmass - inputs.median_mhost

    # Observed distance modulus (per Uddin Eq. 8).
    mu_obs = (
        inputs.Mmax - M_B
        - p1 * st_minus_1
        - p2 * st_minus_1 ** 2
        - beta * inputs.BV
        - alpha * m_centered
    )

    # Model distance modulus.
    mu_flow = _distmod_uddin(H0, inputs.zhel, inputs.zcmb)
    mu_model = np.where(inputs.is_cal, inputs.dist, mu_flow)

    # Flow-sample variance (incl. v_pec term):
    err_flow2 = (
        inputs.eMmax ** 2
        + (fac * inputs.est) ** 2
        + (beta * inputs.eBV) ** 2
        - 2.0 * fac * inputs.covMs
        + 2.0 * beta * fac * inputs.covBVs
        - 2.0 * fac * inputs.covBV_M
        + (alpha * inputs.em_hostmass) ** 2
        + sig_int ** 2
        + (5.0 / np.log(10.0)) ** 2 * (v_pec / (C_KMS * inputs.zcmb)) ** 2
    )
    # Calibrator variance (substitute edist² for the v_pec term):
    err_cal2 = (
        inputs.eMmax ** 2
        + (fac * inputs.est) ** 2
        + (beta * inputs.eBV) ** 2
        - 2.0 * fac * inputs.covMs
        + 2.0 * beta * fac * inputs.covBVs
        - 2.0 * fac * inputs.covBV_M
        + (alpha * inputs.em_hostmass) ** 2
        + sig_int ** 2
        + inputs.edist ** 2
    )
    mu_stat2 = np.where(inputs.is_cal, err_cal2, err_flow2)
    if np.any(mu_stat2 <= 0.0) or np.any(~np.isfinite(mu_stat2)):
        return -np.inf

    dmu = mu_obs - mu_model
    chi2 = float(np.sum(dmu ** 2 / mu_stat2))
    log_norm = -0.5 * float(np.sum(np.log(2.0 * np.pi * mu_stat2)))
    return -0.5 * chi2 + log_norm


# =============================================================================
# Runner
# =============================================================================


@dataclass
class UddinChainResult:
    """Result of one Uddin 8-parameter MCMC chain."""
    case: str
    system: str
    system_label: str
    n_calibrators: int
    n_flow: int
    samples: np.ndarray                      # (N_post, 8)
    param_names: List[str] = field(default_factory=lambda: [
        "M_B", "p1", "p2", "beta", "alpha", "sigma_int", "v_pec", "H0",
    ])
    # Derived summaries for H₀:
    H0_median: float = float("nan")
    H0_sigma: float = float("nan")
    H0_lo: float = float("nan")
    H0_hi: float = float("nan")
    rhat_max: float = float("nan")
    rhat_H0: float = float("nan")
    converged: bool = False
    convergence_gate: float = 1.01
    rhat_per_param: Dict[str, float] = field(default_factory=dict)
    credible_intervals: Dict[str, tuple] = field(default_factory=dict)
    n_walkers: int = 0
    n_steps: int = 0
    n_burnin: int = 0
    mean_acceptance: float = 0.0
    published_target_H0: float = 0.0
    published_sigma_stat: float = 0.0
    published_sigma_sys: float = 0.0
    notes: str = ""

    def as_dict(self) -> Dict[str, object]:
        return {
            "case": self.case,
            "system": self.system,
            "system_label": self.system_label,
            "n_calibrators": int(self.n_calibrators),
            "n_flow": int(self.n_flow),
            "H0_median": float(self.H0_median),
            "H0_sigma": float(self.H0_sigma),
            "H0_lo": float(self.H0_lo),
            "H0_hi": float(self.H0_hi),
            "rhat_max": float(self.rhat_max),
            "rhat_H0": float(self.rhat_H0),
            "converged": bool(self.converged),
            "convergence_gate": float(self.convergence_gate),
            "rhat_per_param": {k: float(v) for k, v in self.rhat_per_param.items()},
            "credible_intervals": {
                k: [float(x) for x in v] for k, v in self.credible_intervals.items()
            },
            "n_walkers": int(self.n_walkers),
            "n_steps": int(self.n_steps),
            "n_burnin": int(self.n_burnin),
            "mean_acceptance": float(self.mean_acceptance),
            "published_target_H0": float(self.published_target_H0),
            "published_sigma_stat": float(self.published_sigma_stat),
            "published_sigma_sys": float(self.published_sigma_sys),
            "notes": self.notes,
        }


def _gelman_rubin_multi(post: np.ndarray) -> np.ndarray:
    """Rhat per parameter; post shape = (n_steps, n_walkers, n_params)."""
    n_steps, n_w, _ = post.shape
    if n_steps < 2 or n_w < 2:
        return np.full(post.shape[-1], np.nan)
    means = post.mean(axis=0)
    vars_ = post.var(axis=0, ddof=1)
    grand = means.mean(axis=0)
    B = n_steps * ((means - grand) ** 2).sum(axis=0) / (n_w - 1)
    W = vars_.mean(axis=0)
    var_hat = (n_steps - 1) / n_steps * W + B / n_steps
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.sqrt(np.where(W > 0, var_hat / W, np.nan))


def run_uddin_csp_chain(
    inputs: UddinCSPInputs,
    settings: MCMCSettings,
    case: str,
    system: str,
    system_label: str,
    *,
    published_target_H0: float = 0.0,
    published_sigma_stat: float = 0.0,
    published_sigma_sys: float = 0.0,
    notes: str = "",
    chain_out_path: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> UddinChainResult:
    """Run the 8-parameter Uddin MCMC for one (case, CSP-sub-sample)."""
    import emcee

    _log = log_fn or (lambda m: logger.info(m))

    rng = np.random.default_rng(settings.seed)
    n_walkers = max(int(settings.n_walkers), 32)        # 8-dim sampling benefits from ≥ 32 walkers
    n_dim = 8

    # Initial ball centered on Uddin's published best fit, with small spread.
    p0_centre = np.array([-19.18, -1.09, -0.54, 2.87, -0.004, 0.18, 420.0, 72.0])
    p0_scale = np.array([0.02, 0.10, 0.30, 0.10, 0.02, 0.02, 60.0, 1.0])
    p0 = p0_centre + p0_scale * rng.standard_normal((n_walkers, n_dim))
    # Clip to box priors:
    lo = np.array([-25.0, -10.0, -10.0, 0.01, -1.0, 1e-4, 1.0, 56.0])
    hi = np.array([14.0, 10.0, 10.0, 10.0, 1.0, 1.0, 1000.0, 84.0])
    p0 = np.clip(p0, lo + 1e-6, hi - 1e-6)

    _log(
        f"[{case}/{system}] Uddin-8 emcee: {n_walkers} walkers × "
        f"{settings.n_steps} steps ({settings.n_burnin} burn-in); "
        f"cal N={inputs.n_cal}, flow N={inputs.n_flow}"
    )

    sampler = emcee.EnsembleSampler(
        n_walkers, n_dim, log_posterior_uddin, args=(inputs,),
    )
    t0 = time.time()
    sampler.run_mcmc(p0, settings.n_steps, progress=settings.progress)
    dt = time.time() - t0
    mean_acc = float(sampler.acceptance_fraction.mean())
    _log(
        f"[{case}/{system}] chain ran in {dt:.1f}s; "
        f"mean acceptance = {mean_acc:.3f}"
    )

    chain = sampler.get_chain()                         # (n_steps, n_w, n_dim)
    post = chain[settings.n_burnin:]
    flat = post.reshape(-1, n_dim)
    names = ["M_B", "p1", "p2", "beta", "alpha", "sigma_int", "v_pec", "H0"]

    rhats = _gelman_rubin_multi(post)
    rhat_per_param = {n: float(r) for n, r in zip(names, rhats)}
    rhat_max = float(np.nanmax(rhats))
    rhat_H0 = float(rhats[-1])
    converged = bool(np.isfinite(rhat_max) and rhat_max < 1.01)

    ci = {}
    for j, n in enumerate(names):
        lo_j, med_j, hi_j = np.percentile(flat[:, j], [16.0, 50.0, 84.0])
        ci[n] = (float(lo_j), float(med_j), float(hi_j))

    H0_samples = flat[:, -1]
    H0_lo, H0_med, H0_hi = np.percentile(H0_samples, [16.0, 50.0, 84.0])
    sigma = 0.5 * (H0_hi - H0_lo)

    _log(
        f"[{case}/{system}] H₀ = {H0_med:.3f} ({H0_lo:.3f}–{H0_hi:.3f}); "
        f"σ = {sigma:.3f}; R̂_max = {rhat_max:.4f} "
        f"({'CONVERGED' if converged else 'NOT CONVERGED'})"
    )

    if chain_out_path is not None:
        chain_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            chain_out_path,
            chain=chain,
            post_flat=flat,
            n_burnin=settings.n_burnin,
            n_walkers=n_walkers,
            n_steps=settings.n_steps,
            rhat_per_param=rhats,
            param_names=np.array(names),
            case=case,
            system=system,
            cal_sn_names=inputs.sn_names[inputs.is_cal],
            cal_dist=inputs.dist[inputs.is_cal],
            cal_edist=inputs.edist[inputs.is_cal],
            flow_sn_names=inputs.sn_names[~inputs.is_cal],
            flow_zcmb=inputs.zcmb[~inputs.is_cal],
        )
        _log(f"[{case}/{system}] chain saved → {chain_out_path}")

    return UddinChainResult(
        case=case,
        system=system,
        system_label=system_label,
        n_calibrators=inputs.n_cal,
        n_flow=inputs.n_flow,
        samples=flat,
        H0_median=float(H0_med),
        H0_sigma=float(sigma),
        H0_lo=float(H0_lo),
        H0_hi=float(H0_hi),
        rhat_max=rhat_max,
        rhat_H0=rhat_H0,
        converged=converged,
        convergence_gate=1.01,
        rhat_per_param=rhat_per_param,
        credible_intervals=ci,
        n_walkers=n_walkers,
        n_steps=int(settings.n_steps),
        n_burnin=int(settings.n_burnin),
        mean_acceptance=mean_acc,
        published_target_H0=published_target_H0,
        published_sigma_stat=published_sigma_stat,
        published_sigma_sys=published_sigma_sys,
        notes=notes,
    )


__all__ = [
    "C_KMS",
    "Q0",
    "UddinChainResult",
    "UddinCSPInputs",
    "build_uddin_inputs_from_loader_dataset",
    "log_posterior_uddin",
    "run_uddin_csp_chain",
]
