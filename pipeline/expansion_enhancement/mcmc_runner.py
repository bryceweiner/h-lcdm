"""
emcee wrapper: run one MCMC chain per ModelConfig, return posterior samples
plus convergence diagnostics.

Gelman–Rubin R̂ is evaluated across emcee walkers (each walker = one chain;
convergence diagnostic is applied to the post-burnin walker traces). Targeting
R̂ < 1.01 per task spec.

Designed to be cheap enough for a smoke run (``n_steps=200, n_walkers=16``)
and thorough enough for the full run (``n_steps=10000, n_walkers=32``).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np

from .data_loaders import ExpansionDataBundle
from .likelihood import ModelConfig, _priors_for, chi2_total, log_posterior

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------

@dataclass
class MCMCSettings:
    n_walkers: int = 32
    n_steps: int = 10_000
    n_burnin: int = 2_000
    seed: int = 42
    progress: bool = True
    n_processes: int = 1  # 1 = no multiprocessing; emcee's Pool is fork-based

    @classmethod
    def short(cls) -> "MCMCSettings":
        return cls(n_walkers=16, n_steps=400, n_burnin=100, progress=False)


@dataclass
class MCMCResult:
    model_name: str
    param_names: List[str]
    samples: np.ndarray                 # (n_kept, n_params) — post-burnin, flattened
    chain: np.ndarray                   # (n_steps, n_walkers, n_params) — full walker history
    log_prob: np.ndarray                # (n_steps, n_walkers)
    best_fit: Dict[str, float]          # maximum-posterior sample
    best_fit_chi2: Dict[str, float]     # per-dataset χ² at best-fit
    credible_intervals: Dict[str, tuple]  # 68% two-sided per parameter
    r_hat: Dict[str, float]
    mean_acceptance_fraction: float
    autocorr_time: Optional[np.ndarray] = None
    n_data: int = 0

    def as_dict(self) -> dict:
        return {
            "model_name": self.model_name,
            "param_names": list(self.param_names),
            "best_fit": self.best_fit,
            "best_fit_chi2": self.best_fit_chi2,
            "credible_intervals": {k: list(v) for k, v in self.credible_intervals.items()},
            "r_hat": self.r_hat,
            "mean_acceptance_fraction": float(self.mean_acceptance_fraction),
            "autocorr_time": (self.autocorr_time.tolist() if self.autocorr_time is not None else None),
            "n_samples": int(self.samples.shape[0]),
            "n_data": int(self.n_data),
            "n_parameters": int(self.samples.shape[1]),
        }


# -----------------------------------------------------------------------------
# Initial walker ball
# -----------------------------------------------------------------------------

def _initial_guess(cfg: ModelConfig) -> np.ndarray:
    """Planck-like central values, plus ε₀ near zero if Model B."""
    base = {"H0": 67.4, "Om": 0.315, "eps": 0.02}
    return np.array([base[p] for p in cfg.param_names])


def _initial_ball(cfg: ModelConfig, n_walkers: int, rng: np.random.Generator) -> np.ndarray:
    """Tight Gaussian ball around initial guess, clipped to the prior box."""
    centre = _initial_guess(cfg)
    scales = {"H0": 1.0, "Om": 0.01, "eps": 0.005}
    widths = np.array([scales[p] for p in cfg.param_names])
    ball = centre + widths * rng.standard_normal((n_walkers, cfg.n_parameters))
    # Clip each param inside its prior box — prevents walkers starting at -inf.
    priors = _priors_for(cfg)
    for j, p in enumerate(cfg.param_names):
        lo, hi = priors[p]
        ball[:, j] = np.clip(ball[:, j], lo + 1e-6, hi - 1e-6)
    return ball


# -----------------------------------------------------------------------------
# Gelman–Rubin across walkers
# -----------------------------------------------------------------------------

def _gelman_rubin(chain_post_burnin: np.ndarray) -> np.ndarray:
    """R̂ for each parameter, computed across emcee walkers.

    ``chain_post_burnin`` has shape (n_steps_kept, n_walkers, n_params). Each
    walker is treated as an independent chain. Returns a 1-D array of R̂s.
    """
    n_steps, n_walkers, n_params = chain_post_burnin.shape
    if n_steps < 2 or n_walkers < 2:
        return np.full(n_params, np.nan)

    # Chain means and variances (treat each walker as one chain).
    chain_means = chain_post_burnin.mean(axis=0)                      # (n_walkers, n_params)
    chain_vars = chain_post_burnin.var(axis=0, ddof=1)                # (n_walkers, n_params)
    grand_mean = chain_means.mean(axis=0)                             # (n_params,)

    between = n_steps * ((chain_means - grand_mean) ** 2).sum(axis=0) / (n_walkers - 1)
    within = chain_vars.mean(axis=0)

    var_hat = (n_steps - 1) / n_steps * within + between / n_steps
    # Guard against pathological zero within-chain variance.
    with np.errstate(divide="ignore", invalid="ignore"):
        rhat = np.sqrt(np.where(within > 0, var_hat / within, np.nan))
    return rhat


# -----------------------------------------------------------------------------
# Summary statistics
# -----------------------------------------------------------------------------

def _summarize_samples(samples: np.ndarray, names: List[str]) -> Dict[str, tuple]:
    """Median + 68% credible interval per parameter."""
    out: Dict[str, tuple] = {}
    for j, p in enumerate(names):
        lo, med, hi = np.percentile(samples[:, j], [16, 50, 84])
        out[p] = (float(lo), float(med), float(hi))
    return out


# -----------------------------------------------------------------------------
# Main driver
# -----------------------------------------------------------------------------

def run_model(
    cfg: ModelConfig,
    bundle: ExpansionDataBundle,
    settings: MCMCSettings,
    chain_out_path: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> MCMCResult:
    """Run emcee for a single ModelConfig against the joint data bundle."""
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
        args=(cfg, bundle),
    )

    t0 = time.time()
    sampler.run_mcmc(p0, settings.n_steps, progress=settings.progress)
    dt = time.time() - t0
    _log(f"[{cfg.name}] chain ran in {dt:.1f}s; mean acceptance = {sampler.acceptance_fraction.mean():.3f}")

    chain_full = sampler.get_chain()                # (n_steps, n_walkers, n_params)
    log_prob_full = sampler.get_log_prob()          # (n_steps, n_walkers)
    post = chain_full[settings.n_burnin:]
    post_lp = log_prob_full[settings.n_burnin:]

    rhat = _gelman_rubin(post)
    rhat_dict = {p: float(r) for p, r in zip(cfg.param_names, rhat)}
    _log(f"[{cfg.name}] Gelman–Rubin R̂: " + ", ".join(f"{k}={v:.4f}" for k, v in rhat_dict.items()))

    # Autocorrelation — emcee's estimate can throw on short chains; be tolerant.
    autocorr = None
    try:
        autocorr = sampler.get_autocorr_time(tol=0)
    except Exception as exc:  # emcee.autocorr.AutocorrError
        _log(f"[{cfg.name}] autocorr time unavailable: {exc}")

    flat = post.reshape(-1, cfg.n_parameters)
    flat_lp = post_lp.reshape(-1)

    # Best-fit = maximum-a-posteriori sample in the post-burnin subset.
    idx_best = int(np.argmax(flat_lp))
    theta_best = flat[idx_best]
    best_fit = {p: float(v) for p, v in zip(cfg.param_names, theta_best)}
    best_fit_chi2 = chi2_total(theta_best, cfg, bundle)

    credible = _summarize_samples(flat, cfg.param_names)
    _log(f"[{cfg.name}] best-fit: " + ", ".join(f"{k}={v:.4f}" for k, v in best_fit.items()))
    _log(f"[{cfg.name}] χ² (best-fit): " + ", ".join(f"{k}={v:.2f}" for k, v in best_fit_chi2.items()))

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
        model_name=cfg.name,
        param_names=list(cfg.param_names),
        samples=flat,
        chain=chain_full,
        log_prob=log_prob_full,
        best_fit=best_fit,
        best_fit_chi2=best_fit_chi2,
        credible_intervals=credible,
        r_hat=rhat_dict,
        mean_acceptance_fraction=float(sampler.acceptance_fraction.mean()),
        autocorr_time=autocorr,
        n_data=bundle.n_data,
    )
