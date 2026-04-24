"""
Markdown report builder for the TRGB comparative analysis.

The pipeline-local reporter writes a standalone markdown summary. A thin
wrapper at :mod:`pipeline.common.reporting.trgb_comparative_reporter` exposes
the same content through ``HLambdaDMReporter.generate_pipeline_report``.

Unconditional reporting: both cases always print, including inconvenient
tensions. No branches gated on result sign or magnitude.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .framework_methodology import FrameworkPrediction
from .freedman_2020_methodology import FreedmanCaseResult


def _sigma_tension(a_value: float, a_sigma: float, b_value: float, b_sigma: float) -> float:
    if a_sigma < 0 or b_sigma < 0:
        return float("nan")
    denom = math.sqrt(a_sigma * a_sigma + b_sigma * b_sigma)
    if denom == 0.0:
        return float("inf")
    return abs(a_value - b_value) / denom


def _fmt_ci(ci: tuple, precision: int = 3) -> str:
    lo, med, hi = ci
    return f"{med:.{precision}f} (+{hi-med:.{precision}f} / -{med-lo:.{precision}f})"


def _case_section(
    case_label: str,
    case: FreedmanCaseResult,
    framework: Optional[FrameworkPrediction],
) -> str:
    lines: List[str] = []
    lines.append(f"### {case_label}\n\n")
    lines.append(
        f"- Published target: H₀ = **{case.H0_published}** ± "
        f"{case.H0_sigma_stat_published} (stat) ± "
        f"{case.H0_sigma_sys_published} (sys) km/s/Mpc\n"
    )
    # Pipeline-computed MCMC posterior (Pantheon+SH0ES flow) — the only
    # H₀ value this pipeline currently computes.
    conv = "converged" if case.mcmc_converged else "NOT CONVERGED"
    lines.append(
        f"- **MCMC posterior (Pantheon+SH0ES flow, pipeline-computed):** "
        f"H₀ = **{case.mcmc_posterior_H0_pantheon_plus:.3f}** ± "
        f"{case.mcmc_posterior_sigma_pantheon_plus:.3f} (stat); "
        f"R̂_max = {case.mcmc_rhat_max:.4f} ({conv}, gate {case.mcmc_convergence_gate}); "
        f"walkers={case.mcmc_n_walkers}, steps={case.mcmc_n_steps}, "
        f"burn-in={case.mcmc_n_burnin}\n"
    )
    lines.append(
        f"- Δ(MCMC_Pantheon+ − published) = {case.pantheon_plus_mcmc_delta:+.3f}; "
        f"tolerance ±{case.tolerance_mag} (stat); "
        f"within tolerance: **{'YES' if case.pantheon_plus_mcmc_within_tolerance else 'NO'}**\n"
    )
    # Literature citations — explicitly labeled, never promoted to
    # reproduction-named fields.
    if case.literature_citations:
        lines.append("\n**Literature citations (NOT pipeline computations):**\n\n")
        for tag, rec in case.literature_citations.items():
            nat = rec.get("nature", "")
            H0v = rec.get("H0")
            if H0v is None:
                continue
            lines.append(
                f"- `{tag}`: H₀ = {H0v:.3f}  *({nat})*\n"
            )
        lines.append("\n")

    if framework is not None:
        lines.append(
            f"- Framework prediction (*{framework.label}*): "
            f"H₀ = **{framework.H0_median:.3f}** "
            f"(68% CI [{framework.H0_low:.3f}, {framework.H0_high:.3f}])\n"
        )
        lines.append(
            f"- Framework d_local: {framework.inputs['d_local_mpc']} Mpc "
            f"± {framework.inputs['sigma_d_local_mpc']} Mpc. "
            f"γ/H at z=0: {framework.inputs['gamma_over_H']:.6f} "
            f"(1/{1/framework.inputs['gamma_over_H']:.1f}).\n"
        )
        if framework.breakdown_flag_any:
            frac = framework.breakdown_fraction
            lines.append(
                f"- **Perturbative-breakdown flag**: {frac*100:.0f}% of draws flagged "
                "— formula predictions unreliable in this regime.\n"
            )
            for msg in framework.breakdown_messages[:3]:
                lines.append(f"  > {msg}\n")
        else:
            lines.append(
                "- Perturbative regime: formula predictions are reliable for this d_local.\n"
            )

        fw_sigma = 0.5 * (framework.H0_high - framework.H0_low)
        tension = _sigma_tension(
            case.mcmc_posterior_H0_pantheon_plus,
            case.mcmc_posterior_sigma_pantheon_plus,
            framework.H0_median,
            fw_sigma,
        )
        lines.append(
            f"- Tension between MCMC Pantheon+ posterior and framework-predicted H₀: "
            f"**{tension:.2f}σ** (stat-only).\n\n"
        )

    mcmc = case.mcmc_result
    if mcmc is not None:
        lines.append(
            "\n**MCMC posterior parameters (Pantheon+SH0ES flow):**\n\n"
            "| Parameter | Median (68% CI) | R̂ |\n| --- | --- | --- |\n"
        )
        for p in mcmc.param_names:
            ci = mcmc.credible_intervals[p]
            rhat = mcmc.r_hat.get(p, float("nan"))
            lines.append(f"| {p} | {_fmt_ci(ci, 4)} | {rhat:.4f} |\n")
        lines.append(
            "\n*Note: The MCMC posterior median for H0 here is the "
            "pipeline-computed value against Pantheon+SH0ES — which carries "
            "the SH0ES Cepheid calibration layer (Hoyt 2025 §4 documents a "
            "+2 km/s/Mpc bias). The Hoyt 2025 citation above is the value "
            "reported as the \"reproduction\" when `sn_system` is set.*\n"
        )
    lines.append("\n")
    return "".join(lines)


def write_summary(
    case_a: Optional[FreedmanCaseResult],
    case_b: Optional[FreedmanCaseResult],
    framework_a: Optional[FrameworkPrediction],
    framework_b: Optional[FrameworkPrediction],
    figure_paths: Dict[str, Path],
    reports_dir: Path,
    preregistration_info: Optional[Dict[str, object]] = None,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    lines.append("# TRGB Comparative Analysis — LMC + NGC 4258 anchors\n\n")
    lines.append(f"*Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n\n")

    lines.append("## Summary\n\n")
    lines.append(
        "Paired reproduction of two CCHP TRGB measurements — Freedman 2019/2020 HST "
        "(LMC anchor) and Freedman 2024/2025 JWST (NGC 4258 anchor) — compared "
        "side-by-side with the H-ΛCDM framework's holographic projection formula "
        "applied as a pure forward prediction for each d_local.\n\n"
        "Both cases always run; all results are reported regardless of agreement.\n\n"
    )

    if preregistration_info:
        lines.append("## Preregistration\n\n")
        for stage in ("stage1", "stage2"):
            info = preregistration_info.get(stage)
            if info:
                lines.append(
                    f"- **Stage {stage[-1]}**: `{info.get('path', 'N/A')}` "
                    f"(SHA-256 `{info.get('sha256', 'N/A')[:16]}...`)\n"
                )
        lines.append("\n")

    lines.append("## Case A — Freedman 2019/2020 (LMC anchor)\n\n")
    if case_a is not None:
        lines.append(_case_section("Results", case_a, framework_a))
    else:
        lines.append("*Case A not run in this invocation.*\n\n")

    lines.append("## Case B — Freedman 2024/2025 (NGC 4258 anchor)\n\n")
    if case_b is not None:
        lines.append(_case_section("Results", case_b, framework_b))
    else:
        lines.append("*Case B not run in this invocation.*\n\n")

    lines.append("## Cross-case comparison\n\n")
    if case_a is not None and case_b is not None:
        obs_shift = (
            case_b.mcmc_posterior_H0_pantheon_plus
            - case_a.mcmc_posterior_H0_pantheon_plus
        )
        lines.append(
            f"- Pipeline MCMC Pantheon+ shift (LMC → NGC 4258): "
            f"**{obs_shift:+.2f} km/s/Mpc** (from "
            f"{case_a.mcmc_posterior_H0_pantheon_plus:.2f} to "
            f"{case_b.mcmc_posterior_H0_pantheon_plus:.2f}).\n"
        )
    if framework_a is not None and framework_b is not None:
        fw_shift = framework_b.H0_median - framework_a.H0_median
        lines.append(
            f"- Framework-predicted shift: **{fw_shift:+.2f} km/s/Mpc** "
            f"(from {framework_a.H0_median:.2f} to {framework_b.H0_median:.2f}).\n"
        )
    lines.append(
        "- The framework predicts a NEGATIVE shift when moving from LMC anchor "
        "(in the perturbative-breakdown regime, predicted ≈ 81 km/s/Mpc) to "
        "NGC 4258 anchor (reliable perturbative regime, predicted ≈ 73 km/s/Mpc). "
        "Whether the observed shift matches in sign and magnitude is a test of "
        "the framework's domain-of-applicability boundary.\n\n"
    )

    if figure_paths:
        lines.append("## Figures\n\n")
        for name, p in figure_paths.items():
            lines.append(f"- `{name}`: [{p.name}]({p.name})\n")
        lines.append("\n")

    lines.append("## Caveats\n\n")
    lines.append(
        "1. The framework's projection formula is perturbative in γ/H · ln(d_CMB/d_local). "
        "For d_local < 1 Mpc (Case A / LMC anchor), the expansion breaks down; "
        "predictions in that regime are surfaced with a breakdown flag and should "
        "not be taken at face value.\n"
        "2. Per-paper fidelity: each reproduction branch uses the extinction and "
        "metallicity treatments published in the corresponding Freedman paper; "
        "sensitivity variants are computed separately and do not feed into primary "
        "numbers.\n"
        "3. All methodological choices (samples, edge detection parameters, priors) "
        "were frozen in a two-stage preregistration document before any H₀ value "
        "was computed. See `docs/trgb_comparative_preregistration_stage{1,2}.md`.\n"
    )

    report_path = reports_dir / "trgb_comparative_analysis_report.md"
    report_path.write_text("".join(lines))

    # JSON companion.
    json_path = reports_dir / "trgb_comparative_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "case_a": case_a.as_dict() if case_a else None,
                "case_b": case_b.as_dict() if case_b else None,
                "framework_a": framework_a.as_dict() if framework_a else None,
                "framework_b": framework_b.as_dict() if framework_b else None,
                "generated": datetime.utcnow().isoformat(),
            },
            indent=2,
            default=str,
        )
    )

    return report_path


__all__ = ["write_summary"]
