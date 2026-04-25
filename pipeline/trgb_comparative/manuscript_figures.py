"""Manuscript figures for the H-ΛCDM TRGB comparative analysis.

This module generates four publication-ready figures aimed at communicating
the framework's empirical findings to a reader unfamiliar with the pipeline.
All figures are saved to ``figures/manuscript/`` (a tracked, persistent
location outside ``results/``) as both PDF and PNG.

The figures complement — they do not replace — the diagnostic figures in
``results/trgb_comparative/figures/``, which target verifying that the
analysis ran correctly. The diagnostics are supplementary; these are
the headline communications.

Style conventions
-----------------
* Sans-serif font, label sizes readable at journal column width (≈ 85 mm).
* Colorblind-safe Tol bright qualitative palette for the four photometric
  systems; sequential cividis-like ramp for tension intensity.
* Framework prediction shown consistently across every figure that
  references it: dark teal band for the propagated 1σ envelope, thicker
  dashed central line.
* SH0ES Cepheid (73.04 ± 1.04, Riess+ 2022) and Planck CMB (67.4 ± 0.5,
  Planck 2018 VI) reference values shown in identical styling whenever
  they appear: SH0ES in dusty-rose, Planck in slate-blue.

Figures
-------
* ``fig1_framework_vs_chains``: headline. 12 chain medians grouped by
  case, color-coded by photometric system, against the framework
  prediction band(s) at the corresponding anchor and the SH0ES / Planck
  references.
* ``fig2_cross_anchor_shift``: per-system Δ(LMC → NGC 4258) shifts
  alongside the framework prediction (−1.21 km/s/Mpc), plus annotation
  of the Pantheon+ outlier as SH0ES Cepheid contamination.
* ``fig3_tension_matrix``: 3 × 4 heatmap of the σ-tension between each
  chain and the framework prediction for its anchor.
* ``fig4_prediction_vs_distance``: continuous H_local(d_local) curve
  under the linear-form projection formula, with the LMC and NGC 4258
  reproduction chains overlaid at their respective anchor distances.
"""

from __future__ import annotations

import csv
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from matplotlib.colors import TwoSlopeNorm

from .framework_methodology import FrameworkPrediction
from .projection_formula import (
    compute_gamma_over_H_at_z0,
    holographic_h_ratio,
)
from hlcdm.parameters import HLCDM_PARAMS

logger = logging.getLogger(__name__)


# Reference values (with units km/s/Mpc).
SH0ES_H0: float = 73.04
SH0ES_SIGMA: float = 1.04
PLANCK_H0: float = 67.4
PLANCK_SIGMA: float = 0.5

# Anchor distances (Mpc).
D_LMC_MPC: float = 0.0496
D_N4258_MPC: float = 7.58

# Chain ordering.
CASES: Tuple[str, ...] = ("case_a", "case_b", "case_b_jwst_only")
SYSTEMS: Tuple[str, ...] = ("csp_i", "csp_ii", "supercal", "pantheon_plus")

CASE_LABELS: Dict[str, str] = {
    "case_a": "Case A\n(LMC anchor)",
    "case_b": "Case B\n(24-SN aug.)",
    "case_b_jwst_only": "Case B\n(JWST-only)",
}
SYSTEM_LABELS: Dict[str, str] = {
    "csp_i": "CSP-I",
    "csp_ii": "CSP-II",
    "supercal": "SuperCal",
    "pantheon_plus": "Pantheon+",
}

# Tol bright palette — colorblind-safe, distinct hues.
SYSTEM_COLORS: Dict[str, str] = {
    "csp_i": "#4477AA",        # blue
    "csp_ii": "#66CCEE",        # cyan
    "supercal": "#CCBB44",      # yellow
    "pantheon_plus": "#AA3377", # purple
}

# Reference / framework colors (kept consistent across figures).
FRAMEWORK_COLOR = "#117733"     # dark teal
SH0ES_COLOR = "#EE6677"         # dusty rose
PLANCK_COLOR = "#222255"        # slate

# Marker per case (allows shape-coding even if printed black-and-white).
CASE_MARKERS: Dict[str, str] = {
    "case_a": "o",
    "case_b": "s",
    "case_b_jwst_only": "D",
}


@dataclass(frozen=True)
class ChainRecord:
    """Subset of 12-chain matrix CSV columns needed for the manuscript figures."""

    case: str
    system: str
    H0_median: float
    H0_sigma: float
    framework_H0_median: float
    framework_sigma: float
    tension: float


def _read_chain_matrix(csv_path: Path) -> List[ChainRecord]:
    out: List[ChainRecord] = []
    with csv_path.open() as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            try:
                out.append(
                    ChainRecord(
                        case=row["case"],
                        system=row["system"],
                        H0_median=float(row["H0_median"]),
                        H0_sigma=float(row["H0_sigma"]),
                        framework_H0_median=float(row["framework_H0_median"]),
                        framework_sigma=float(row["framework_sigma"]),
                        tension=float(row["tension_sigma_stat_only"]),
                    )
                )
            except (ValueError, KeyError):
                # Skip placeholder rows for failed/skipped chains.
                continue
    return out


def _apply_manuscript_style() -> None:
    """Sans-serif, journal-column-friendly sizing."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": [
                "DejaVu Sans", "Helvetica", "Arial", "Liberation Sans",
            ],
            "font.size": 9,
            "axes.labelsize": 10,
            "axes.titlesize": 11,
            "axes.linewidth": 0.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "figure.dpi": 200,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def _save_dual(fig, base_path: Path) -> Tuple[Path, Path]:
    base_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path = base_path.with_suffix(".pdf")
    png_path = base_path.with_suffix(".png")
    fig.savefig(pdf_path)
    fig.savefig(png_path, dpi=300)
    plt.close(fig)
    return pdf_path, png_path


def _ref_band(ax, value: float, sigma: float, color: str, label: str) -> None:
    """Horizontal reference band (mean ± 1σ) plus central line."""
    ax.axhspan(value - sigma, value + sigma, color=color, alpha=0.15, lw=0)
    ax.axhline(value, color=color, lw=1.0, ls="--", alpha=0.9, label=label)


def _anchor_for_case(case: str) -> str:
    return "lmc" if case == "case_a" else "ngc4258"


# ---------------------------------------------------------------------------
# Figure 1 — Framework prediction vs all 12 chains
# ---------------------------------------------------------------------------


def _figure1_framework_vs_chains(
    chains: List[ChainRecord],
    framework_a: FrameworkPrediction,
    framework_b: FrameworkPrediction,
    out_base: Path,
) -> Tuple[Path, Path]:
    fig, ax = plt.subplots(figsize=(7.0, 4.5))

    # Reference horizontal bands across the full plot.
    _ref_band(ax, SH0ES_H0, SH0ES_SIGMA, SH0ES_COLOR, f"SH0ES Cepheid {SH0ES_H0:.2f} ± {SH0ES_SIGMA:.2f}")
    _ref_band(ax, PLANCK_H0, PLANCK_SIGMA, PLANCK_COLOR, f"Planck CMB {PLANCK_H0:.2f} ± {PLANCK_SIGMA:.2f}")

    # Per-case x positions; within each case, systems are offset for visibility.
    n_cases = len(CASES)
    n_sys = len(SYSTEMS)
    case_centers = {case: i for i, case in enumerate(CASES)}
    sys_offsets = np.linspace(-0.30, 0.30, n_sys)

    # Framework prediction bands per case (one anchor per case region).
    fw_per_anchor = {"lmc": framework_a, "ngc4258": framework_b}
    half_band_x = 0.45
    for case_id in CASES:
        cx = case_centers[case_id]
        anchor = _anchor_for_case(case_id)
        fw = fw_per_anchor[anchor]
        med = float(fw.H0_median)
        sigma = 0.5 * (float(fw.H0_high) - float(fw.H0_low))
        ax.fill_between(
            [cx - half_band_x, cx + half_band_x],
            [med - sigma] * 2,
            [med + sigma] * 2,
            color=FRAMEWORK_COLOR, alpha=0.20, lw=0,
        )
        ax.hlines(
            med, cx - half_band_x, cx + half_band_x,
            color=FRAMEWORK_COLOR, lw=1.6, ls="-", alpha=0.95,
        )

    # Chain points (color = system; marker = case).
    sys_index = {s: i for i, s in enumerate(SYSTEMS)}
    for rec in chains:
        x = case_centers[rec.case] + sys_offsets[sys_index[rec.system]]
        ax.errorbar(
            x, rec.H0_median, yerr=rec.H0_sigma,
            fmt=CASE_MARKERS[rec.case], color=SYSTEM_COLORS[rec.system],
            markersize=6, markeredgecolor="black", markeredgewidth=0.5,
            elinewidth=1.0, capsize=2.5,
        )

    ax.set_xticks(list(case_centers.values()))
    ax.set_xticklabels([CASE_LABELS[c] for c in CASES])
    ax.set_xlim(-0.7, n_cases - 0.3)
    ax.set_ylabel(r"$H_0$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_ylim(63, 78)

    # Two legends: systems (color) on the left, references (lines) on the right.
    sys_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=SYSTEM_COLORS[s], markeredgecolor="black",
            markersize=6, label=SYSTEM_LABELS[s],
        )
        for s in SYSTEMS
    ]
    fw_handle = mpatches.Patch(
        facecolor=FRAMEWORK_COLOR, alpha=0.30,
        label="H-ΛCDM framework prediction (1σ)",
    )
    sh0es_handle = mpatches.Patch(
        facecolor=SH0ES_COLOR, alpha=0.20,
        label=f"SH0ES Cepheid {SH0ES_H0:.2f} ± {SH0ES_SIGMA:.2f}",
    )
    planck_handle = mpatches.Patch(
        facecolor=PLANCK_COLOR, alpha=0.20,
        label=f"Planck CMB {PLANCK_H0:.2f} ± {PLANCK_SIGMA:.2f}",
    )
    leg1 = ax.legend(
        handles=sys_handles, title="Photometric system",
        loc="upper left", bbox_to_anchor=(0.005, 0.99),
    )
    ax.add_artist(leg1)
    ax.legend(
        handles=[fw_handle, sh0es_handle, planck_handle],
        loc="upper right", bbox_to_anchor=(0.995, 0.99),
    )

    ax.set_title(
        "Framework prediction vs. 12-chain reproduction matrix\n"
        "Case A: LMC anchor (framework ≈ 70.4)   |   "
        "Case B: NGC 4258 anchor (framework ≈ 69.2)",
        loc="left",
    )
    fig.tight_layout()
    return _save_dual(fig, out_base)


# ---------------------------------------------------------------------------
# Figure 2 — Cross-anchor shift comparison
# ---------------------------------------------------------------------------


def _figure2_cross_anchor_shift(
    chains: List[ChainRecord],
    framework_a: FrameworkPrediction,
    framework_b: FrameworkPrediction,
    out_base: Path,
) -> Tuple[Path, Path]:
    """Δ(NGC 4258 anchor − LMC anchor) per photometric system.

    "Δ" is computed as ``case_b_jwst_only minus case_a`` to keep the
    comparison apples-to-apples on the JWST-anchored side. The framework
    prediction is the (case_b NGC 4258) − (case_a LMC) shift evaluated
    by the same projection formula at the two anchors.
    """
    by_case_sys: Dict[Tuple[str, str], ChainRecord] = {
        (r.case, r.system): r for r in chains
    }

    framework_shift = float(framework_b.H0_median - framework_a.H0_median)
    framework_sigma_a = 0.5 * (float(framework_a.H0_high) - float(framework_a.H0_low))
    framework_sigma_b = 0.5 * (float(framework_b.H0_high) - float(framework_b.H0_low))
    framework_shift_sigma = math.sqrt(framework_sigma_a ** 2 + framework_sigma_b ** 2)

    fig, ax = plt.subplots(figsize=(6.5, 4.3))

    ax.axhspan(
        framework_shift - framework_shift_sigma,
        framework_shift + framework_shift_sigma,
        color=FRAMEWORK_COLOR, alpha=0.20, lw=0,
        label=f"Framework Δ = {framework_shift:+.2f} ± {framework_shift_sigma:.2f}",
    )
    ax.axhline(framework_shift, color=FRAMEWORK_COLOR, lw=1.6, ls="--", alpha=0.95)

    x_positions = {s: i for i, s in enumerate(SYSTEMS)}
    pantheon_outlier_x: Optional[float] = None
    pantheon_outlier_y: Optional[float] = None
    for system in SYSTEMS:
        rec_a = by_case_sys.get(("case_a", system))
        rec_b = by_case_sys.get(("case_b_jwst_only", system))
        if rec_a is None or rec_b is None:
            continue
        delta = rec_b.H0_median - rec_a.H0_median
        sigma = math.sqrt(rec_a.H0_sigma ** 2 + rec_b.H0_sigma ** 2)
        x = x_positions[system]
        ax.errorbar(
            x, delta, yerr=sigma,
            fmt="o", color=SYSTEM_COLORS[system],
            markersize=8, markeredgecolor="black", markeredgewidth=0.6,
            elinewidth=1.2, capsize=3,
            label=f"{SYSTEM_LABELS[system]}: Δ = {delta:+.2f} ± {sigma:.2f}",
        )
        if system == "pantheon_plus":
            pantheon_outlier_x, pantheon_outlier_y = x, delta

    if pantheon_outlier_x is not None:
        ax.annotate(
            "SH0ES Cepheid contamination\n(LMC-anchor calibrators)",
            xy=(pantheon_outlier_x, pantheon_outlier_y),
            xytext=(pantheon_outlier_x - 1.4, pantheon_outlier_y - 1.3),
            fontsize=8, ha="left", color="black",
            arrowprops=dict(
                arrowstyle="->", color="black", lw=0.7,
                connectionstyle="arc3,rad=-0.18",
            ),
        )

    ax.axhline(0.0, color="black", lw=0.5, alpha=0.4)
    ax.set_xticks(list(x_positions.values()))
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in SYSTEMS])
    ax.set_xlim(-0.5, len(SYSTEMS) - 0.5)
    ax.set_ylabel(
        r"$\Delta H_{\mathrm{local}}$ [km s$^{-1}$ Mpc$^{-1}$]"
        "\nNGC 4258-anchor (Case B JWST-only) − LMC-anchor (Case A)"
    )
    ax.set_title(
        "Cross-anchor shift: empirical vs. framework prediction",
        loc="left",
    )
    ax.legend(loc="lower left", bbox_to_anchor=(0.02, 0.02))
    fig.tight_layout()
    return _save_dual(fig, out_base)


# ---------------------------------------------------------------------------
# Figure 3 — σ-tension matrix
# ---------------------------------------------------------------------------


def _figure3_tension_matrix(
    chains: List[ChainRecord],
    out_base: Path,
) -> Tuple[Path, Path]:
    by_case_sys = {(r.case, r.system): r for r in chains}
    n_cases, n_sys = len(CASES), len(SYSTEMS)
    grid = np.full((n_cases, n_sys), np.nan)
    for i, case in enumerate(CASES):
        for j, system in enumerate(SYSTEMS):
            rec = by_case_sys.get((case, system))
            if rec is not None:
                grid[i, j] = rec.tension

    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    vmax = max(3.0, float(np.nanmax(grid)) * 1.05)
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
    im = ax.imshow(
        grid, cmap="RdBu_r", norm=norm, aspect="auto",
        origin="upper",
    )

    ax.set_xticks(range(n_sys))
    ax.set_xticklabels([SYSTEM_LABELS[s] for s in SYSTEMS])
    ax.set_yticks(range(n_cases))
    ax.set_yticklabels([CASE_LABELS[c].replace("\n", " ") for c in CASES])
    ax.tick_params(top=False, bottom=True)

    n_below_one_sigma = 0
    for i in range(n_cases):
        for j in range(n_sys):
            v = grid[i, j]
            if math.isnan(v):
                ax.text(j, i, "—", ha="center", va="center", color="grey", fontsize=9)
                continue
            color = "white" if (v >= 1.5 or v <= 0.3) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                    color=color, fontsize=10, fontweight="bold")
            if v < 1.0:
                n_below_one_sigma += 1

    cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    cbar.set_label(
        r"$\tau = |H_0^{\mathrm{MCMC}} - H_0^{\mathrm{framework}}|"
        r" / \sqrt{\sigma^2_{\mathrm{MCMC}} + \sigma^2_{\mathrm{framework}}}$"
    )
    cbar.ax.axhline(1.0, color="black", lw=0.8)

    ax.set_title(
        f"σ-tension matrix — {n_below_one_sigma} of {n_cases * n_sys} chains "
        "agree within 1σ",
        loc="left",
    )
    fig.tight_layout()
    return _save_dual(fig, out_base)


# ---------------------------------------------------------------------------
# Figure 4 — Framework prediction across anchor distances
# ---------------------------------------------------------------------------


def _figure4_prediction_vs_distance(
    chains: List[ChainRecord],
    out_base: Path,
) -> Tuple[Path, Path]:
    """Continuous framework prediction H_local(d_local) under linear form.

    H_CMB band (Planck 2018) is propagated through the formula by
    multiplying the ratio at each d_local; γ/H is the runtime value
    used by the pipeline.
    """
    d_grid = np.geomspace(0.01, 100.0, 400)
    gamma_over_H = compute_gamma_over_H_at_z0()
    d_cmb = float(HLCDM_PARAMS.D_CMB_PLANCK_2018)
    ratios = np.array([
        holographic_h_ratio(
            d_local_mpc=float(d), d_cmb_mpc=d_cmb,
            gamma_over_H=gamma_over_H, emit_warning=False,
        ).ratio
        for d in d_grid
    ])
    central = PLANCK_H0 * ratios
    upper = (PLANCK_H0 + PLANCK_SIGMA) * ratios
    lower = (PLANCK_H0 - PLANCK_SIGMA) * ratios

    fig, ax = plt.subplots(figsize=(7.0, 4.3))
    ax.fill_between(
        d_grid, lower, upper, color=FRAMEWORK_COLOR, alpha=0.20, lw=0,
        label="Framework prediction (Planck H_CMB ± 1σ propagated)",
    )
    ax.plot(
        d_grid, central, color=FRAMEWORK_COLOR, lw=1.6, ls="-",
        label=r"Central: $H_{\mathrm{CMB}} \cdot [1 + (\gamma/H)\ln(d_{\mathrm{CMB}}/d_{\mathrm{local}})]$",
    )

    ax.axvline(D_LMC_MPC, color="black", lw=0.5, ls=":", alpha=0.6)
    ax.axvline(D_N4258_MPC, color="black", lw=0.5, ls=":", alpha=0.6)
    ax.text(
        D_LMC_MPC, 64.0, "LMC\n(0.05 Mpc)",
        ha="center", va="bottom", fontsize=8,
    )
    ax.text(
        D_N4258_MPC, 64.0, "NGC 4258\n(7.58 Mpc)",
        ha="center", va="bottom", fontsize=8,
    )

    # Reproduction chains overlaid at their anchor distances.
    by_case_sys = {(r.case, r.system): r for r in chains}
    # Symmetric small log offsets so the four systems don't pile on top.
    log_offsets = np.array([-0.18, -0.06, 0.06, 0.18])
    for system, dlog in zip(SYSTEMS, log_offsets):
        for case_id, d_anchor in (("case_a", D_LMC_MPC),
                                   ("case_b_jwst_only", D_N4258_MPC)):
            rec = by_case_sys.get((case_id, system))
            if rec is None:
                continue
            x = d_anchor * (10.0 ** dlog)
            ax.errorbar(
                x, rec.H0_median, yerr=rec.H0_sigma,
                fmt=CASE_MARKERS[case_id], color=SYSTEM_COLORS[system],
                markersize=6, markeredgecolor="black", markeredgewidth=0.5,
                elinewidth=1.0, capsize=2.5,
                zorder=3,
            )

    # Reference horizontal lines.
    ax.axhline(SH0ES_H0, color=SH0ES_COLOR, lw=0.9, ls="--", alpha=0.85)
    ax.text(
        d_grid[-1] * 0.65, SH0ES_H0 + 0.15,
        f"SH0ES Cepheid ({SH0ES_H0:.2f})",
        color=SH0ES_COLOR, fontsize=8, ha="left",
    )
    ax.axhline(PLANCK_H0, color=PLANCK_COLOR, lw=0.9, ls=":", alpha=0.85)
    ax.text(
        d_grid[-1] * 0.65, PLANCK_H0 - 0.65,
        f"Planck CMB ({PLANCK_H0:.2f})",
        color=PLANCK_COLOR, fontsize=8, ha="left",
    )

    sys_handles = [
        plt.Line2D(
            [0], [0], marker="o", color="w",
            markerfacecolor=SYSTEM_COLORS[s], markeredgecolor="black",
            markersize=6, label=SYSTEM_LABELS[s],
        )
        for s in SYSTEMS
    ]
    case_handles = [
        plt.Line2D(
            [0], [0], marker=CASE_MARKERS[c], color="black",
            markerfacecolor="white", markersize=6, label=CASE_LABELS[c].replace("\n", " "),
        )
        for c in ("case_a", "case_b_jwst_only")
    ]
    leg1 = ax.legend(
        handles=sys_handles, title="System",
        loc="upper left", bbox_to_anchor=(0.005, 0.99),
    )
    ax.add_artist(leg1)
    leg2 = ax.legend(
        handles=case_handles, title="Anchor",
        loc="upper left", bbox_to_anchor=(0.18, 0.99),
    )
    ax.add_artist(leg2)
    fw_handle = mpatches.Patch(
        facecolor=FRAMEWORK_COLOR, alpha=0.30,
        label="Framework H-ΛCDM (Planck-propagated 1σ)",
    )
    ax.legend(handles=[fw_handle], loc="lower left", bbox_to_anchor=(0.005, 0.005))

    ax.set_xscale("log")
    ax.set_xlim(0.01, 100.0)
    ax.set_ylim(63, 78)
    ax.set_xlabel(r"$d_{\mathrm{local}}$ [Mpc] (log scale)")
    ax.set_ylabel(r"$H_{\mathrm{local}}$ [km s$^{-1}$ Mpc$^{-1}$]")
    ax.set_title(
        "Linear-form framework prediction across anchor distances",
        loc="left",
    )
    fig.tight_layout()
    return _save_dual(fig, out_base)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def generate_manuscript_figures(
    chain_matrix_csv: Path,
    framework_a: FrameworkPrediction,
    framework_b: FrameworkPrediction,
    out_dir: Path,
) -> Dict[str, Tuple[Path, Path]]:
    """Render the four manuscript figures.

    Parameters
    ----------
    chain_matrix_csv:
        Path to the 12-chain reproduction matrix CSV
        (``results/12_chain_matrix.csv`` by default).
    framework_a, framework_b:
        Framework predictions for the LMC and NGC 4258 anchors.
    out_dir:
        Output directory. Each figure is written as both ``.pdf`` and
        ``.png``. Created if missing.
    """
    _apply_manuscript_style()
    chains = _read_chain_matrix(chain_matrix_csv)
    out_dir = Path(out_dir)

    figs: Dict[str, Tuple[Path, Path]] = {}
    figs["fig1_framework_vs_chains"] = _figure1_framework_vs_chains(
        chains, framework_a, framework_b,
        out_dir / "fig1_framework_vs_chains",
    )
    figs["fig2_cross_anchor_shift"] = _figure2_cross_anchor_shift(
        chains, framework_a, framework_b,
        out_dir / "fig2_cross_anchor_shift",
    )
    figs["fig3_tension_matrix"] = _figure3_tension_matrix(
        chains, out_dir / "fig3_tension_matrix",
    )
    figs["fig4_prediction_vs_distance"] = _figure4_prediction_vs_distance(
        chains, out_dir / "fig4_prediction_vs_distance",
    )

    n_below = sum(1 for r in chains if r.tension < 1.0)
    logger.info(
        "Manuscript figures → %s  (%d/%d chains within 1σ of framework)",
        out_dir, n_below, len(chains),
    )
    return figs


__all__ = ["generate_manuscript_figures"]
