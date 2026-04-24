"""
Plots for the expansion-enhancement test.

Every figure writes to ``figures_dir`` under the pipeline's output root.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

from ..common.visualization import HLCDM_COLORS
from .cosmology import C_KMS, D_H, D_M, D_V, H_LCDM, H_framework, make_H_callable
from .data_loaders import BAOData, ExpansionDataBundle
from .likelihood import ModelConfig, _bao_model_vector
from .mcmc_runner import MCMCResult

logger = logging.getLogger(__name__)

_MODEL_COLORS = {
    "A_lcdm": HLCDM_COLORS["neutral"],
    "B_const": HLCDM_COLORS["primary"],
    "B_qtep": HLCDM_COLORS["accent"],
    "B_residuals": HLCDM_COLORS["qtep_purple"],
}


def _model_H(cfg: ModelConfig, result: MCMCResult):
    H0 = result.best_fit["H0"]
    Om = result.best_fit["Om"]
    eps = result.best_fit.get("eps", 0.0)
    return make_H_callable(H0, Om, eps, mode=cfg.epsilon_mode)


# -----------------------------------------------------------------------------
# BAO residuals
# -----------------------------------------------------------------------------

def plot_bao_residuals(
    bao: BAOData,
    configs: List[ModelConfig],
    results: List[MCMCResult],
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Offset measurements slightly by kind for readability.
    markers = {"D_M/r_d": "o", "D_H/r_d": "s", "D_V/r_d": "^"}

    for cfg, result in zip(configs, results):
        H = _model_H(cfg, result)
        model = _bao_model_vector(bao, H, cfg.r_d)
        residuals = (bao.value - model) / bao.error
        for i, (zi, kind) in enumerate(zip(bao.z, bao.kind)):
            ax.errorbar(
                zi,
                residuals[i],
                yerr=1.0,
                fmt=markers.get(kind, "o"),
                color=_MODEL_COLORS.get(cfg.name, HLCDM_COLORS["neutral"]),
                alpha=0.8,
                label=f"{cfg.name}" if i == 0 else None,
            )

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"$z_{\rm eff}$")
    ax.set_ylabel(r"(data $-$ model) / $\sigma$")
    ax.set_title("DESI DR1 BAO residuals")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# SN Hubble-diagram residuals
# -----------------------------------------------------------------------------

def plot_sn_residuals(
    bundle: ExpansionDataBundle,
    configs: List[ModelConfig],
    results: List[MCMCResult],
    out_path: Path,
    z_bins: int = 40,
) -> None:
    from .cosmology import mu_model

    sn = bundle.sn
    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Bin residuals in log(z) for readability with 1700 points.
    z = sn.z
    log_z_edges = np.linspace(np.log10(max(z.min(), 1e-3)), np.log10(z.max()), z_bins + 1)
    z_centres = 10 ** (0.5 * (log_z_edges[:-1] + log_z_edges[1:]))

    for cfg, result in zip(configs, results):
        H = _model_H(cfg, result)
        mu_th = mu_model(z, H)
        # Profile-M: subtract the best-fit constant offset.
        one = np.ones_like(mu_th)
        inv = sn.inv_cov
        M_hat = float(one @ inv @ (sn.mu - mu_th)) / float(one @ inv @ one)
        resid = sn.mu - (mu_th + M_hat)

        means, stds = np.zeros(z_bins), np.zeros(z_bins)
        for b in range(z_bins):
            mask = (z >= 10 ** log_z_edges[b]) & (z < 10 ** log_z_edges[b + 1])
            if mask.sum() > 0:
                means[b] = resid[mask].mean()
                stds[b] = resid[mask].std() / max(1, np.sqrt(mask.sum()))
            else:
                means[b] = np.nan

        ax.errorbar(
            z_centres,
            means,
            yerr=stds,
            fmt="o-",
            color=_MODEL_COLORS.get(cfg.name, HLCDM_COLORS["neutral"]),
            alpha=0.8,
            label=cfg.name,
            markersize=4,
        )

    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\mu_{\rm obs} - \mu_{\rm model}$ (binned)")
    ax.set_title("Pantheon+SH0ES Hubble-diagram residuals")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# H(z) comparison with BAO-derived points
# -----------------------------------------------------------------------------

def plot_Hz_curves(
    configs: List[ModelConfig],
    results: List[MCMCResult],
    bundle: ExpansionDataBundle,
    out_path: Path,
) -> None:
    z_grid = np.linspace(0.01, 2.5, 200)
    fig, ax = plt.subplots(figsize=(8, 4.5))

    for cfg, result in zip(configs, results):
        H = _model_H(cfg, result)
        ax.plot(
            z_grid,
            H(z_grid),
            color=_MODEL_COLORS.get(cfg.name, HLCDM_COLORS["neutral"]),
            label=cfg.name,
            lw=1.8,
        )

    # Overlay D_H/r_d points converted to H(z) via H = c / (r_d * (D_H/r_d)).
    # Use the Model-B r_d when drawing the point, so the scatter represents
    # Model B's implied data coordinates.
    cfg_B = next((c for c in configs if c.has_epsilon), configs[0])
    for i, (zi, kind, val, err) in enumerate(
        zip(bundle.bao.z, bundle.bao.kind, bundle.bao.value, bundle.bao.error)
    ):
        if kind != "D_H/r_d":
            continue
        H_val = C_KMS / (cfg_B.r_d * val)
        H_err = H_val * err / val
        ax.errorbar(zi, H_val, yerr=H_err, fmt="ko", alpha=0.7, markersize=4)

    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$H(z)$ [km/s/Mpc]")
    ax.set_title("Expansion history: best-fit comparison")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# Per-dataset χ² bar chart
# -----------------------------------------------------------------------------

def plot_chi2_contributions(
    results: List[MCMCResult],
    out_path: Path,
) -> None:
    labels = ["bao", "sn", "cmb"]
    x = np.arange(len(labels))
    width = 0.8 / max(len(results), 1)
    fig, ax = plt.subplots(figsize=(6.5, 4))

    for i, result in enumerate(results):
        values = [result.best_fit_chi2[k] for k in labels]
        ax.bar(
            x + (i - (len(results) - 1) / 2.0) * width,
            values,
            width,
            color=_MODEL_COLORS.get(result.model_name, HLCDM_COLORS["neutral"]),
            label=result.model_name,
        )

    ax.set_xticks(x)
    ax.set_xticklabels(["BAO", "SN", "CMB θ*"])
    ax.set_ylabel(r"$\chi^2$")
    ax.set_title("Per-dataset χ² at best-fit")
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# Corner plots
# -----------------------------------------------------------------------------

def plot_corner(result: MCMCResult, out_path: Path) -> None:
    try:
        import corner
    except ImportError:
        logger.warning("corner not installed; skipping corner plot")
        return

    fig = corner.corner(
        result.samples,
        labels=[
            {"H0": r"$H_0$", "Om": r"$\Omega_m$", "eps": r"$\varepsilon$"}[p]
            for p in result.param_names
        ],
        truths=[result.best_fit[p] for p in result.param_names],
        color=_MODEL_COLORS.get(result.model_name, HLCDM_COLORS["neutral"]),
        show_titles=True,
        title_kwargs={"fontsize": 10},
    )
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


# -----------------------------------------------------------------------------
# Orchestrator
# -----------------------------------------------------------------------------

def plot_residual_shape(
    result_B_residuals: MCMCResult,
    out_path: Path,
) -> None:
    """ε(z) profile inherited from Planck TT+TE+EE residuals, scaled by best-fit ε_amp."""
    from .cmb_residuals import epsilon_shape, load_combined_residuals

    eps_amp = result_B_residuals.best_fit["eps"]
    z_grid = np.linspace(0.0, 1089.0, 1500)
    shape = epsilon_shape(z_grid)
    eps_z = eps_amp * shape

    sig = load_combined_residuals()
    fig, axes = plt.subplots(2, 1, figsize=(8, 6.5), sharex=False)

    ax = axes[0]
    ax.plot(sig.ell, sig.z_combined, color=HLCDM_COLORS["qtep_purple"], lw=0.8, alpha=0.9)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"signed IVW residual  $z(\ell)$  [$\sigma$]")
    ax.set_title("Planck 2018 TT+TE+EE combined residuals (data − best-fit ΛCDM)")

    ax = axes[1]
    ax.plot(z_grid, eps_z, color=HLCDM_COLORS["qtep_purple"], lw=1.2)
    ax.axhline(0, color="k", lw=0.5, alpha=0.5)
    ax.set_xlabel(r"$z$")
    ax.set_ylabel(r"$\varepsilon(z)$  (best-fit, Model B_residuals)")
    ax.set_title(
        r"$\varepsilon(z) = \varepsilon_{amp} \cdot z_{\rm Planck}(\ell(z))$, "
        f"ε_amp = {eps_amp:.4f}"
    )
    ax.set_xscale("log")
    ax.set_xlim(0.01, 1089)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    logger.info(f"Saved {out_path}")


def generate_all_figures(
    bundle: ExpansionDataBundle,
    configs: List[ModelConfig],
    results: List[MCMCResult],
    figures_dir: Path,
) -> Dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    paths: Dict[str, Path] = {}

    bao_path = figures_dir / "residuals_bao.png"
    plot_bao_residuals(bundle.bao, configs, results, bao_path)
    paths["residuals_bao"] = bao_path

    sn_path = figures_dir / "residuals_sn.png"
    plot_sn_residuals(bundle, configs, results, sn_path)
    paths["residuals_sn"] = sn_path

    hz_path = figures_dir / "Hz_curves.png"
    plot_Hz_curves(configs, results, bundle, hz_path)
    paths["Hz_curves"] = hz_path

    chi2_path = figures_dir / "chi2_contributions.png"
    plot_chi2_contributions(results, chi2_path)
    paths["chi2_contributions"] = chi2_path

    for result in results:
        p = figures_dir / f"corner_{result.model_name}.png"
        plot_corner(result, p)
        paths[f"corner_{result.model_name}"] = p

    # ε(z) shape plot for the residual-shape model (if present).
    res_result = next((r for r in results if r.model_name == "B_residuals"), None)
    if res_result is not None:
        p = figures_dir / "residual_shape.png"
        plot_residual_shape(res_result, p)
        paths["residual_shape"] = p

    return paths
