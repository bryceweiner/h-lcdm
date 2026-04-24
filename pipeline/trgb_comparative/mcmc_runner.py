"""
emcee wrapper for the Freedman reproduction MCMC runs.

Mirrors the pattern in
:mod:`pipeline.expansion_enhancement.mcmc_runner` — 32 walkers × 10k steps
× 2k burn-in, Gelman-Rubin R̂ < 1.01 target, chains saved as .npz.

Each Freedman case (2020 HST / 2024 JWST) is an independent MCMC run
producing its own posterior.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .likelihood import (
    FreedmanLikelihoodInputs,
    FreedmanModelConfig,
    log_posterior,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration / result structs
# ---------------------------------------------------------------------------


@dataclass
class MCMCSettings:
    n_walkers: int = 32
    n_steps: int = 10_000
    n_burnin: int = 2_000
    seed: int = 42
    progress: bool = True

    @classmethod
    def short(cls) -> "MCMCSettings":
        return cls(n_walkers=16, n_steps=400, n_burnin=100, progress=False)


@dataclass
class MCMCResult:
    case_name: str
    param_names: List[str]
    samples: np.ndarray             # (n_kept, n_params)
    chain: np.ndarray               # (n_steps, n_walkers, n_params)
    log_prob: np.ndarray
    best_fit: Dict[str, float]
    credible_intervals: Dict[str, tuple]
    r_hat: Dict[str, float]
    mean_acceptance_fraction: float
    autocorr_time: Optional[np.ndarray] = None
    n_data: int = 0

    def as_dict(self) -> dict:
        return {
            "case_name": self.case_name,
            "param_names": list(self.param_names),
            "best_fit": self.best_fit,
            "credible_intervals": {k: list(v) for k, v in self.credible_intervals.items()},
            "r_hat": self.r_hat,
            "mean_acceptance_fraction": float(self.mean_acceptance_fraction),
            "autocorr_time": (self.autocorr_time.tolist() if self.autocorr_time is not None else None),
            "n_samples": int(self.samples.shape[0]),
            "n_data": int(self.n_data),
            "n_parameters": int(self.samples.shape[1]),
        }


# ---------------------------------------------------------------------------
# Initial ball
# ---------------------------------------------------------------------------


def _initial_ball(
    cfg: FreedmanModelConfig,
    n_walkers: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian ball centered on each prior's mean (or box midpoint)."""
    centres = np.empty(cfg.n_parameters)
    widths = np.empty(cfg.n_parameters)
    for i, prior in enumerate(cfg.priors):
        if prior.sigma > 0.0:
            centres[i] = prior.mean
            widths[i] = 0.5 * prior.sigma
        else:
            centres[i] = 0.5 * (prior.lo + prior.hi)
            widths[i] = 0.05 * (prior.hi - prior.lo)
    ball = centres + widths * rng.standard_normal((n_walkers, cfg.n_parameters))
    # Clip to stay inside box.
    for i, prior in enumerate(cfg.priors):
        ball[:, i] = np.clip(ball[:, i], prior.lo + 1e-6, prior.hi - 1e-6)
    return ball


# ---------------------------------------------------------------------------
# Gelman-Rubin
# ---------------------------------------------------------------------------


def _gelman_rubin(chain_post_burnin: np.ndarray) -> np.ndarray:
    n_steps, n_walkers, n_params = chain_post_burnin.shape
    if n_steps < 2 or n_walkers < 2:
        return np.full(n_params, np.nan)
    chain_means = chain_post_burnin.mean(axis=0)
    chain_vars = chain_post_burnin.var(axis=0, ddof=1)
    grand_mean = chain_means.mean(axis=0)
    between = n_steps * ((chain_means - grand_mean) ** 2).sum(axis=0) / (n_walkers - 1)
    within = chain_vars.mean(axis=0)
    var_hat = (n_steps - 1) / n_steps * within + between / n_steps
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(np.where(within > 0, var_hat / within, np.nan))
    return rhat


def _summarize(samples: np.ndarray, names) -> Dict[str, tuple]:
    out = {}
    for j, p in enumerate(names):
        lo, med, hi = np.percentile(samples[:, j], [16, 50, 84])
        out[p] = (float(lo), float(med), float(hi))
    return out


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def run_freedman_case(
    cfg: FreedmanModelConfig,
    inputs: FreedmanLikelihoodInputs,
    settings: MCMCSettings,
    chain_out_path: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> MCMCResult:
    """Run emcee for a single Freedman case."""
    import emcee

    _log = log_fn or (lambda m: logger.info(m))

    rng = np.random.default_rng(settings.seed)
    p0 = _initial_ball(cfg, settings.n_walkers, rng)

    _log(
        f"[{cfg.name}] emcee: {settings.n_walkers} walkers × {settings.n_steps} steps "
        f"({settings.n_burnin} burn-in); n_params={cfg.n_parameters}"
    )

    sampler = emcee.EnsembleSampler(
        settings.n_walkers,
        cfg.n_parameters,
        log_posterior,
        args=(cfg, inputs),
    )

    t0 = time.time()
    sampler.run_mcmc(p0, settings.n_steps, progress=settings.progress)
    dt = time.time() - t0
    _log(
        f"[{cfg.name}] chain ran in {dt:.1f}s; "
        f"mean acceptance = {sampler.acceptance_fraction.mean():.3f}"
    )

    chain_full = sampler.get_chain()
    log_prob_full = sampler.get_log_prob()
    post = chain_full[settings.n_burnin:]
    post_lp = log_prob_full[settings.n_burnin:]

    rhat = _gelman_rubin(post)
    rhat_dict = {p: float(r) for p, r in zip(cfg.param_names, rhat)}
    _log(
        f"[{cfg.name}] Gelman-Rubin R̂: "
        + ", ".join(f"{k}={v:.4f}" for k, v in rhat_dict.items())
    )

    autocorr = None
    try:
        autocorr = sampler.get_autocorr_time(tol=0)
    except Exception as exc:  # emcee.autocorr.AutocorrError
        _log(f"[{cfg.name}] autocorr time unavailable: {exc}")

    flat = post.reshape(-1, cfg.n_parameters)
    flat_lp = post_lp.reshape(-1)

    idx_best = int(np.argmax(flat_lp))
    theta_best = flat[idx_best]
    best_fit = {p: float(v) for p, v in zip(cfg.param_names, theta_best)}
    credible = _summarize(flat, cfg.param_names)

    _log(
        f"[{cfg.name}] best-fit: "
        + ", ".join(f"{k}={v:.4f}" for k, v in best_fit.items())
    )

    if chain_out_path is not None:
        chain_out_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            chain_out_path,
            chain=chain_full,
            log_prob=log_prob_full,
            post=flat,
            post_log_prob=flat_lp,
            param_names=np.array(cfg.param_names),
            best_fit=theta_best,
        )
        _log(f"[{cfg.name}] chain saved → {chain_out_path}")

    return MCMCResult(
        case_name=cfg.name,
        param_names=list(cfg.param_names),
        samples=flat,
        chain=chain_full,
        log_prob=log_prob_full,
        best_fit=best_fit,
        credible_intervals=credible,
        r_hat=rhat_dict,
        mean_acceptance_fraction=float(sampler.acceptance_fraction.mean()),
        autocorr_time=autocorr,
        n_data=inputs.n_data(),
    )


__all__ = ["MCMCResult", "MCMCSettings", "run_freedman_case"]
