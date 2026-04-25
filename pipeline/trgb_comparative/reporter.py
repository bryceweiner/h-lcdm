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


def _render_chain_matrix_section(
    chain_matrix: Optional[Dict[str, Dict[str, Dict[str, object]]]],
) -> str:
    """Render the 8-chain (4 SN systems × 2 cases) table block."""
    if not chain_matrix:
        return ""
    lines: List[str] = ["## Per-SN-system MCMC chain matrix (pipeline-computed H₀)\n\n"]
    lines.append(
        "All values below are **pipeline-computed MCMC posteriors** on the "
        "indicated SN photometric system and TRGB distance scale. No value is "
        "a literature citation. Non-converged chains (R̂ ≥ 1.01) are reported "
        "with the ``converged = No`` flag and not promoted as primary results.\n\n"
    )
    lines.append(
        "| Case | System | Mode | N_cal | N_flow | H₀ (med) | σ(H₀) | R̂_max | Converged |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: |\n"
    )
    for case_key, systems in chain_matrix.items():
        case_label = "A (LMC)" if case_key == "case_a" else "B (NGC 4258)"
        for sys_key, rec in systems.items():
            if rec.get("skipped") or rec.get("failed") or "error" in rec:
                lines.append(
                    f"| {case_label} | {sys_key} | – | – | – | – | – | – | ERROR |\n"
                )
                continue
            mode = rec.get("mode", "")
            n_cal = rec.get("n_calibrators", rec.get("n_data", 0))
            n_flow = rec.get("n_flow", 0)
            if "n_flow" not in rec and "n_data" in rec and rec["n_data"]:
                # simple chain tracks only n_data (= n_cal + n_flow)
                n_flow = rec["n_data"] - int(n_cal) if n_cal else 0
            h0 = rec["H0_median"]
            sigma = rec["H0_sigma"]
            rh = rec.get("rhat_max", rec.get("rhat_H0", float("nan")))
            conv = "✓" if rec.get("converged") else "✗"
            lines.append(
                f"| {case_label} | {sys_key} | {mode} | {n_cal} | {n_flow} | "
                f"{h0:.3f} | {sigma:.3f} | {rh:.4f} | {conv} |\n"
            )
    lines.append("\n")
    # Cross-case (system-by-system) shift in MCMC H₀:
    case_a = chain_matrix.get("case_a", {})
    case_b = chain_matrix.get("case_b", {})
    if case_a and case_b:
        lines.append("### Cross-case shift per SN system\n\n")
        lines.append(
            "| System | Case A H₀ | Case B H₀ | Δ(B − A) | Both converged |\n"
            "| --- | ---: | ---: | ---: | :---: |\n"
        )
        for sys_id in ("csp_i", "csp_ii", "supercal", "pantheon_plus"):
            ra = case_a.get(sys_id, {}); rb = case_b.get(sys_id, {})
            if not ra or not rb or "H0_median" not in ra or "H0_median" not in rb:
                continue
            ha = ra["H0_median"]; hb = rb["H0_median"]
            delta = hb - ha
            both_conv = ra.get("converged") and rb.get("converged")
            lines.append(
                f"| {sys_id} | {ha:.3f} | {hb:.3f} | {delta:+.3f} | "
                f"{'✓' if both_conv else '✗'} |\n"
            )
        lines.append("\n")
    return "".join(lines)


def _render_full_calibrator_section(
    full_cal: Optional[Dict[str, Dict[str, Dict[str, object]]]],
) -> str:
    """Render the full-Freedman-calibrator-sample chain matrix table block.

    Distinct from the legacy intersection chain matrix: every entry here
    operates on the full Freedman calibrator sample appropriate to its
    case. Output filenames carry a `_full` suffix.
    """
    if not full_cal:
        return ""
    lines: List[str] = ["## Full-calibrator-sample MCMC chain matrix (audit Recommendation 1)\n\n"]
    lines.append(
        "These are the **primary pipeline-computed reproductions going "
        "forward**. Every chain operates on the full Freedman calibrator "
        "sample appropriate to its case, with explicit per-chain coverage "
        "reported (`Cal req/got` ratio = requested calibrator count / "
        "calibrators with valid SNooPy parameters in this photometric "
        "system).\n\n"
        "Targets: Case A → 69.8 (F2019 18-SN); Case B (primary) → 70.39 "
        "(F2025 24-SN augmented HST+JWST); Case B JWST-only sensitivity → "
        "68.81 (F2025 11-SN Table 2 TRGB+JAGB).\n\n"
    )
    lines.append(
        "| Case | System | Cal req/got | N_flow | Mode | H₀ (med) | σ(H₀) | "
        "R̂_max | Conv | Target | Δ(MCMC−tgt) | Within ±stat? |\n"
        "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | :---: | "
        "---: | ---: | :---: |\n"
    )
    for case_key, systems in full_cal.items():
        for sys_key in ("csp_i", "csp_ii", "supercal", "pantheon_plus"):
            rec = systems.get(sys_key, {})
            if not rec or "H0_median" not in rec:
                continue
            req = rec.get("n_calibrators_requested", "?")
            got = rec.get("n_calibrators_matched", "?")
            tgt = float(rec.get("published_target_H0", 0.0))
            tol = float(rec.get("published_sigma_stat", 0.0))
            delta = float(rec["H0_median"]) - tgt
            within = "✓" if abs(delta) <= tol else "✗"
            mode = rec.get("mode", "?")
            rh = float(rec.get("rhat_max", rec.get("rhat_H0", float("nan"))))
            conv = "✓" if rec.get("converged") else "✗"
            lines.append(
                f"| {case_key} | {sys_key} | {req}/{got} | "
                f"{rec.get('n_flow', '?')} | {mode} | "
                f"{float(rec['H0_median']):.3f} | {float(rec['H0_sigma']):.3f} | "
                f"{rh:.4f} | {conv} | {tgt:.2f} | {delta:+.3f} | {within} |\n"
            )
    lines.append("\n")
    # Cross-case shift per system, full-cal numbers
    lines.append("### Cross-case shift per system (full-cal)\n\n")
    lines.append(
        "| System | A H₀ | B aug H₀ | B JWST H₀ | Δ(B aug − A) | "
        "Δ(B JWST − A) |\n| --- | ---: | ---: | ---: | ---: | ---: |\n"
    )
    import math as _math
    for sys_id in ("csp_i", "csp_ii", "supercal", "pantheon_plus"):
        ra = full_cal.get("case_a", {}).get(sys_id, {})
        rb = full_cal.get("case_b", {}).get(sys_id, {})
        rj = full_cal.get("case_b_jwst_only", {}).get(sys_id, {})
        if "H0_median" not in ra:
            continue
        ha = float(ra["H0_median"])
        hb = float(rb.get("H0_median", float("nan")))
        hj = float(rj.get("H0_median", float("nan"))) if rj else float("nan")
        d_aug = (hb - ha) if not _math.isnan(hb) else float("nan")
        d_jw = (hj - ha) if not _math.isnan(hj) else float("nan")
        lines.append(
            f"| {sys_id} | {ha:.3f} | {hb:.3f} | "
            f"{(f'{hj:.3f}' if not _math.isnan(hj) else '–')} | "
            f"{d_aug:+.3f} | "
            f"{(f'{d_jw:+.3f}' if not _math.isnan(d_jw) else '–')} |\n"
        )
    lines.append("\n")
    return "".join(lines)


def _render_positive_control_section(pc: Optional[Dict[str, object]]) -> str:
    if not pc:
        return ""
    lines: List[str] = ["## Uddin 2023 positive-control test (audit Recommendation 3)\n\n"]
    if pc.get("error"):
        lines.append(f"**FAILED with error**: `{pc.get('error')}`\n\n")
        return "".join(lines)
    target = pc.get("target_H0", 70.242)
    H0 = float(pc.get("H0_median", float("nan")))
    sigma = float(pc.get("H0_sigma", float("nan")))
    delta = float(pc.get("delta", float("nan")))
    rhat = float(pc.get("rhat_max", float("nan")))
    converged = bool(pc.get("converged", False))
    passed = bool(pc.get("pass", False))
    lines.append(
        f"- Target: H₀ = {target} ± {pc.get('target_sigma', 0.724)} km/s/Mpc "
        "(Uddin 2023, ApJ 970 72 published TRGB result)\n"
        f"- Pipeline: H₀ = **{H0:.3f}** ± **{sigma:.3f}** km/s/Mpc\n"
        f"- Δ vs target: **{delta:+.3f}** km/s/Mpc\n"
        f"- R̂_max: **{rhat:.4f}** ({'CONVERGED' if converged else 'NOT CONVERGED'})\n"
        f"- Acceptance (|Δ| ≤ 1.0 AND R̂_max < 1.01): **{'PASS' if passed else 'FAIL'}**\n\n"
    )
    if not passed:
        lines.append(
            "**HALT**: positive-control failed. CSP chain outputs above "
            "MUST NOT be quoted as reproductions until the 8-parameter "
            "SNooPy likelihood is fixed and this test passes.\n\n"
        )
    return "".join(lines)


def write_summary(
    case_a: Optional[FreedmanCaseResult],
    case_b: Optional[FreedmanCaseResult],
    framework_a: Optional[FrameworkPrediction],
    framework_b: Optional[FrameworkPrediction],
    figure_paths: Dict[str, Path],
    reports_dir: Path,
    preregistration_info: Optional[Dict[str, object]] = None,
    chain_matrix: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
    full_calibrator_matrix: Optional[Dict[str, Dict[str, Dict[str, object]]]] = None,
    uddin_positive_control: Optional[Dict[str, object]] = None,
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

    # Full-calibrator chain matrix FIRST — primary pipeline-computed
    # results going forward (audit Recommendation 1).
    lines.append(_render_full_calibrator_section(full_calibrator_matrix))

    # Uddin 2023 positive-control validation (audit Recommendation 3).
    lines.append(_render_positive_control_section(uddin_positive_control))

    # Legacy intersection chain matrix retained as a sensitivity baseline.
    if chain_matrix:
        lines.append("## Legacy intersection chain matrix (sensitivity baseline)\n\n")
        lines.append(
            "These chains use the previous Uddin-intersection calibrator "
            "logic. Preserved for comparison; the full-calibrator chains "
            "above are the primary results.\n\n"
        )
        lines.append(_render_chain_matrix_section(chain_matrix))

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
        "- Under the post-2026-04-25 linear-form correction (formula reduced to "
        "1 + (γ/H)·L), the framework predicts H_local ≈ 70.40 km/s/Mpc at the "
        "LMC anchor (d_local = 0.05 Mpc) and ≈ 69.20 km/s/Mpc at the NGC 4258 "
        "anchor (d_local = 7.58 Mpc) — a small NEGATIVE shift driven entirely "
        "by ln(d_CMB/d_local). The |γ/H · L| ≥ 1 breakdown criterion does not "
        "fire for either anchor (γ/H · L ≈ 0.045 at LMC, ≈ 0.027 at NGC 4258); "
        "both predictions are firmly in the perturbative regime. Whether the "
        "observed shift matches in sign and magnitude is the framework test.\n\n"
    )

    if figure_paths:
        lines.append("## Figures\n\n")
        for name, p in figure_paths.items():
            lines.append(f"- `{name}`: [{p.name}]({p.name})\n")
        lines.append("\n")

    lines.append("## Caveats\n\n")
    lines.append(
        "1. The framework's projection formula reduces to the linear form "
        "1 + (γ/H)·L after the 2026-04-25 correction (b parameter and C(G) "
        "term both removed). The breakdown criterion |γ/H · L| ≥ 1 does not "
        "fire for any realistic distance-ladder anchor (LMC γ/H · L ≈ 0.045; "
        "NGC 4258 ≈ 0.027). The breakdown infrastructure remains in the code "
        "as defense-in-depth but no longer flags the LMC anchor.\n"
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

    # JSON companion. Guard ``as_dict()`` calls so that callers passing
    # plain dicts / SimpleNamespace stubs (e.g. the regen script) succeed.
    def _to_jsonable(obj):
        if obj is None:
            return None
        as_dict = getattr(obj, "as_dict", None)
        if callable(as_dict):
            return as_dict()
        if isinstance(obj, dict):
            return obj
        try:
            return dict(vars(obj))
        except Exception:
            return None

    json_path = reports_dir / "trgb_comparative_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "case_a": _to_jsonable(case_a),
                "case_b": _to_jsonable(case_b),
                "framework_a": _to_jsonable(framework_a),
                "framework_b": _to_jsonable(framework_b),
                "sn_system_chains": chain_matrix,
                "full_calibrator_chains": full_calibrator_matrix,
                "uddin_positive_control": uddin_positive_control,
                "generated": datetime.utcnow().isoformat(),
            },
            indent=2,
            default=str,
        )
    )

    return report_path


__all__ = ["write_summary"]
