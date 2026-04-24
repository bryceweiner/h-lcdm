"""
Visualization for the TRGB comparative analysis.

All figures use the HLCDM palette, 300 DPI, PDF output. Conventions mirror
:mod:`pipeline.common.visualization` and the per-pipeline patterns used by
:mod:`pipeline.expansion_enhancement.visualization`.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Optional

import matplotlib.pyplot as plt
import numpy as np

from ..common.visualization import HLCDM_COLORS
from .framework_methodology import FrameworkPrediction
from .freedman_2020_methodology import FreedmanCaseResult

logger = logging.getLogger(__name__)


_CASE_COLORS = {
    "case_a": HLCDM_COLORS["accent"],
    "case_b": HLCDM_COLORS["primary"],
    "framework": HLCDM_COLORS["qtep_purple"],
    "published": HLCDM_COLORS["neutral"],
}


def _apply_pub_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 10,
            "axes.labelsize": 12,
            "axes.titlesize": 13,
            "savefig.dpi": 300,
            "figure.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def _save_pdf(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    logger.info(f"Saved {path}")
    return path


# ---------------------------------------------------------------------------
# Individual plots
# ---------------------------------------------------------------------------


def plot_cmd_edges(result: FreedmanCaseResult, out_path: Path) -> Path:
    """Luminosity function + Sobel response for each anchor field detection."""
    _apply_pub_style()
    n_panels = max(1, len(result.edge_detections_anchor))
    n_cols = min(3, n_panels)
    n_rows = int(np.ceil(n_panels / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3 * n_rows), squeeze=False)
    axes = axes.flatten()
    for ax, (field_id, det) in zip(axes, result.edge_detections_anchor.items()):
        if "mag_centres" in det.diagnostics and "smoothed_LF" in det.diagnostics:
            mag_centres = det.diagnostics["mag_centres"]
            lf = det.diagnostics["smoothed_LF"]
            ax.plot(
                mag_centres,
                lf,
                color=_CASE_COLORS["case_a"],
                lw=1.2,
                label="smoothed LF",
            )
            if "weighted_response" in det.diagnostics:
                resp = det.diagnostics["weighted_response"]
                ax2 = ax.twinx()
                ax2.plot(
                    mag_centres,
                    resp,
                    color=HLCDM_COLORS["accent"],
                    lw=1.0,
                    alpha=0.7,
                    label="Sobel response",
                )
                ax2.axvline(det.I_TRGB, color=HLCDM_COLORS["warning"], ls="--", lw=0.8)
        ax.set_title(
            f"{result.anchor.name} / {field_id}\nI_TRGB = {det.I_TRGB:.3f} ± {det.sigma_I_TRGB:.3f}"
        )
        ax.set_xlabel("I (mag)")
    for ax in axes[len(result.edge_detections_anchor):]:
        ax.axis("off")
    fig.tight_layout()
    return _save_pdf(fig, out_path)


def plot_h0_posteriors(
    case_a: Optional[FreedmanCaseResult],
    case_b: Optional[FreedmanCaseResult],
    framework_a: Optional[FrameworkPrediction],
    framework_b: Optional[FrameworkPrediction],
    out_path: Path,
) -> Path:
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    def _plot_posterior(samples: np.ndarray, label: str, color: str, ls: str = "-"):
        lo, hi = float(np.percentile(samples, 0.5)), float(np.percentile(samples, 99.5))
        grid = np.linspace(lo, hi, 400)
        # Gaussian KDE-like histogram smoothing.
        hist, edges = np.histogram(samples, bins=60, range=(lo, hi), density=True)
        centres = 0.5 * (edges[:-1] + edges[1:])
        ax.plot(centres, hist, color=color, lw=1.6, ls=ls, label=label)

    if case_a is not None:
        samples = case_a.mcmc_result.samples[:, 0]
        _plot_posterior(samples, "Case A reproduced (Freedman 2020)", _CASE_COLORS["case_a"])
        ax.axvline(
            case_a.H0_published,
            color=_CASE_COLORS["case_a"],
            ls=":",
            lw=1.0,
            alpha=0.8,
            label=f"Case A published ({case_a.H0_published})",
        )
    if case_b is not None:
        samples = case_b.mcmc_result.samples[:, 0]
        _plot_posterior(samples, "Case B reproduced (Freedman 2024)", _CASE_COLORS["case_b"])
        ax.axvline(
            case_b.H0_published,
            color=_CASE_COLORS["case_b"],
            ls=":",
            lw=1.0,
            alpha=0.8,
            label=f"Case B published ({case_b.H0_published})",
        )
    if framework_a is not None:
        samples = framework_a.H0_samples
        _plot_posterior(samples, f"{framework_a.label} (framework)", _CASE_COLORS["framework"], ls="--")
    if framework_b is not None:
        samples = framework_b.H0_samples
        _plot_posterior(samples, f"{framework_b.label} (framework)", _CASE_COLORS["framework"], ls="-.")

    ax.set_xlabel(r"$H_0$ [km/s/Mpc]")
    ax.set_ylabel("posterior density")
    ax.set_title("TRGB comparative H₀ posteriors")
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    fig.tight_layout()
    return _save_pdf(fig, out_path)


def plot_predictions_vs_observed(
    case_a: Optional[FreedmanCaseResult],
    case_b: Optional[FreedmanCaseResult],
    framework_a: Optional[FrameworkPrediction],
    framework_b: Optional[FrameworkPrediction],
    out_path: Path,
) -> Path:
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(8, 5))

    rows = []
    if case_a is not None:
        rows.append(("Case A published", case_a.H0_published, case_a.H0_sigma_stat_published, _CASE_COLORS["published"]))
        rows.append((
            "Case A MCMC Pantheon+",
            case_a.mcmc_posterior_H0_pantheon_plus,
            case_a.mcmc_posterior_sigma_pantheon_plus,
            _CASE_COLORS["case_a"],
        ))
    if framework_a is not None:
        rows.append(("Case A framework", framework_a.H0_median, 0.5 * (framework_a.H0_high - framework_a.H0_low), _CASE_COLORS["framework"]))
    if case_b is not None:
        rows.append(("Case B published", case_b.H0_published, case_b.H0_sigma_stat_published, _CASE_COLORS["published"]))
        rows.append((
            "Case B MCMC Pantheon+",
            case_b.mcmc_posterior_H0_pantheon_plus,
            case_b.mcmc_posterior_sigma_pantheon_plus,
            _CASE_COLORS["case_b"],
        ))
    if framework_b is not None:
        rows.append(("Case B framework", framework_b.H0_median, 0.5 * (framework_b.H0_high - framework_b.H0_low), _CASE_COLORS["framework"]))

    y = np.arange(len(rows))[::-1]
    labels = [r[0] for r in rows]
    H0s = np.array([r[1] for r in rows])
    errs = np.array([r[2] for r in rows])
    colors = [r[3] for r in rows]

    ax.errorbar(H0s, y, xerr=errs, fmt="o", ecolor=colors, markerfacecolor="none")
    for i, (yi, h, c) in enumerate(zip(y, H0s, colors)):
        ax.plot(h, yi, "o", color=c, markersize=7)

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel(r"$H_0$ [km/s/Mpc]")
    ax.axvline(73.0, color=HLCDM_COLORS["qtep_purple"], ls="--", lw=0.8, alpha=0.5, label="SH0ES-NGC4258 band")
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Predictions vs. reproductions — unconditional")
    fig.tight_layout()
    return _save_pdf(fig, out_path)


def plot_cross_case_shift(
    case_a: Optional[FreedmanCaseResult],
    case_b: Optional[FreedmanCaseResult],
    framework_a: Optional[FrameworkPrediction],
    framework_b: Optional[FrameworkPrediction],
    out_path: Path,
) -> Path:
    """Observed CCHP shift (LMC anchor → NGC 4258 anchor) vs. framework shift."""
    _apply_pub_style()
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if case_a is not None and case_b is not None:
        a_H0 = case_a.mcmc_posterior_H0_pantheon_plus
        b_H0 = case_b.mcmc_posterior_H0_pantheon_plus
        obs_shift = b_H0 - a_H0
        ax.annotate(
            "",
            xy=(b_H0, 1.2),
            xytext=(a_H0, 1.2),
            arrowprops=dict(arrowstyle="->", color=HLCDM_COLORS["primary"], lw=2),
        )
        ax.text(
            0.5 * (a_H0 + b_H0),
            1.25,
            f"MCMC P+ Δ = {obs_shift:+.2f}",
            ha="center",
            color=HLCDM_COLORS["primary"],
        )

    if framework_a is not None and framework_b is not None:
        fw_shift = framework_b.H0_median - framework_a.H0_median
        ax.annotate(
            "",
            xy=(framework_b.H0_median, 2.2),
            xytext=(framework_a.H0_median, 2.2),
            arrowprops=dict(arrowstyle="->", color=HLCDM_COLORS["qtep_purple"], lw=2),
        )
        ax.text(
            0.5 * (framework_a.H0_median + framework_b.H0_median),
            2.25,
            f"framework Δ = {fw_shift:+.2f}",
            ha="center",
            color=HLCDM_COLORS["qtep_purple"],
        )

    ax.set_xlabel(r"$H_0$ [km/s/Mpc]")
    ax.set_yticks([1.2, 2.2])
    ax.set_yticklabels(["observed", "framework"])
    ax.set_ylim(0.5, 3.0)
    ax.set_title("Cross-case shift: Case A (LMC) → Case B (NGC 4258)")
    fig.tight_layout()
    return _save_pdf(fig, out_path)


def generate_all_figures(
    case_a: Optional[FreedmanCaseResult],
    case_b: Optional[FreedmanCaseResult],
    framework_a: Optional[FrameworkPrediction],
    framework_b: Optional[FrameworkPrediction],
    figures_dir: Path,
) -> Dict[str, Path]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    out: Dict[str, Path] = {}
    if case_a is not None:
        out["cmd_edges_case_a"] = plot_cmd_edges(
            case_a, figures_dir / "trgb_comparative_cmd_edges_case_a.pdf"
        )
    if case_b is not None:
        out["cmd_edges_case_b"] = plot_cmd_edges(
            case_b, figures_dir / "trgb_comparative_cmd_edges_case_b.pdf"
        )
    out["h0_posteriors"] = plot_h0_posteriors(
        case_a, case_b, framework_a, framework_b,
        figures_dir / "trgb_comparative_h0_posteriors.pdf",
    )
    out["predictions_vs_observed"] = plot_predictions_vs_observed(
        case_a, case_b, framework_a, framework_b,
        figures_dir / "trgb_comparative_predictions_vs_observed.pdf",
    )
    out["cross_case_shift"] = plot_cross_case_shift(
        case_a, case_b, framework_a, framework_b,
        figures_dir / "trgb_comparative_cross_case_shift.pdf",
    )
    return out


__all__ = [
    "generate_all_figures",
    "plot_cmd_edges",
    "plot_cross_case_shift",
    "plot_h0_posteriors",
    "plot_predictions_vs_observed",
]
