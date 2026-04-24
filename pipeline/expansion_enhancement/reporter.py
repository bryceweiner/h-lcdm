"""
Markdown summary writer for the expansion-enhancement test.

Self-contained (no ``HLambdaDMReporter`` subclass) because this analysis is
a single falsifiability test with a fixed output structure — a full Grok-enabled
comprehensive report would overshoot the task. Outputs a single
``expansion_test_summary.md`` plus a small JSON companion.
"""

from __future__ import annotations

import json
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np

from .data_loaders import ExpansionDataBundle
from .likelihood import MODEL_A, ModelConfig
from .mcmc_runner import MCMCResult


# SH0ES local H0 measurement: Riess et al. 2022 ApJ 934 L7 (arXiv:2112.04510).
SH0ES_H0: float = 73.04
SH0ES_SIGMA: float = 1.04

# Framework's BAO-paper predicted ε, for comparison only.
FRAMEWORK_EPS_PREDICTED: float = 0.022


def _aic_bic(chi2: float, n_params: int, n_data: int) -> Dict[str, float]:
    return {
        "aic": chi2 + 2 * n_params,
        "bic": chi2 + n_params * math.log(max(n_data, 2)),
    }


def _sigma_tension_H0(result: MCMCResult) -> float:
    """|Δ| / √(σ_model² + σ_SH0ES²) — conservative two-sided tension."""
    lo, med, hi = result.credible_intervals["H0"]
    sigma_model = 0.5 * (hi - lo)
    return abs(med - SH0ES_H0) / math.sqrt(sigma_model**2 + SH0ES_SIGMA**2)


def _verdict(
    result_A: MCMCResult,
    result_B: MCMCResult,
    n_data: int,
) -> Dict[str, object]:
    """Task-spec falsifiability criteria → supported / disfavored / refined."""
    chi2_A = result_A.best_fit_chi2["total"]
    chi2_B = result_B.best_fit_chi2["total"]
    dchi2 = chi2_B - chi2_A

    eps_lo, eps_med, eps_hi = result_B.credible_intervals["eps"]
    eps_sigma = 0.5 * (eps_hi - eps_lo)
    eps_significance = eps_med / eps_sigma if eps_sigma > 0 else float("inf")

    H0_tension_B = _sigma_tension_H0(result_B)

    # Supported: ε > 0 at >3σ AND Δχ² < -4 (Model B better) AND H0 within 2σ of SH0ES.
    supported = (
        eps_significance > 3.0
        and dchi2 < -4.0
        and H0_tension_B < 2.0
    )
    # Disfavored: ε consistent with 0 within 1σ AND Δχ² ≥ 0.
    disfavored = (eps_significance < 1.0) and (dchi2 >= 0.0)
    refined = not (supported or disfavored)

    return {
        "supported": bool(supported),
        "disfavored": bool(disfavored),
        "refined": bool(refined),
        "delta_chi2": float(dchi2),
        "eps_significance": float(eps_significance),
        "H0_tension_B_sigma": float(H0_tension_B),
    }


def _fmt_ci(ci: tuple, precision: int = 4) -> str:
    lo, med, hi = ci
    return f"{med:.{precision}f} (+{hi-med:.{precision}f} / -{med-lo:.{precision}f})"


def write_summary(
    bundle: ExpansionDataBundle,
    configs: List[ModelConfig],
    results: List[MCMCResult],
    figure_paths: Dict[str, Path],
    reports_dir: Path,
) -> Path:
    reports_dir.mkdir(parents=True, exist_ok=True)

    # Identify the primary Model-A and Model-B_const results for the verdict.
    result_A = next(r for r in results if r.model_name == MODEL_A.name)
    result_B_const = next((r for r in results if r.model_name == "B_const"), None)
    result_B_qtep = next((r for r in results if r.model_name == "B_qtep"), None)
    result_B_res = next((r for r in results if r.model_name == "B_residuals"), None)

    n_data = bundle.n_data

    # AIC/BIC across models.
    model_info: Dict[str, Dict[str, float]] = {}
    for cfg, result in zip(configs, results):
        chi2 = result.best_fit_chi2["total"]
        crit = _aic_bic(chi2, cfg.n_parameters, n_data)
        model_info[cfg.name] = {
            "chi2_total": chi2,
            "chi2_bao": result.best_fit_chi2["bao"],
            "chi2_sn": result.best_fit_chi2["sn"],
            "chi2_cmb": result.best_fit_chi2["cmb"],
            **crit,
        }

    # Verdict based on Model B (constant ε) vs Model A.
    verdict = _verdict(result_A, result_B_const, n_data) if result_B_const else {}

    # Build markdown.
    lines: List[str] = []
    lines.append("# H-ΛCDM Expansion-Enhancement Falsifiability Test\n")
    lines.append(f"*Generated {datetime.utcnow().strftime('%Y-%m-%d %H:%M UTC')}*\n")
    lines.append("\n## Summary\n")
    lines.append(
        "Joint likelihood fit of DESI DR1 BAO, Pantheon+SH0ES supernovae, and "
        "Planck 2018 θ* against two cosmological models:\n\n"
        "- **Model A (ΛCDM)**: H₀, Ω_m free; r_d = 147.5 Mpc fixed; ε = 0.\n"
        "- **Model B (framework)**: H₀, Ω_m, ε free; r_d = 150.71 Mpc fixed; ε applied for z < 1100.\n\n"
        "Two ε(z) prescriptions are tested: constant ε (hard step at z=1100) and a "
        "QTEP-motivated profile tied to γ(z) in `hlcdm/cosmology.py`.\n"
    )

    lines.append("## Data\n")
    lines.append(
        f"- **BAO**: DESI DR1, {bundle.bao.value.size} measurements across 7 redshift bins "
        "(arXiv:2404.03002). Full covariance with intra-bin D_M/D_H correlations.\n"
        f"- **SNe**: Pantheon+SH0ES, {bundle.sn.z.size} SNe "
        f"(z={bundle.sn.z.min():.3f}–{bundle.sn.z.max():.3f}). "
        "Full 1701×1701 stat+sys covariance; absolute magnitude M analytically marginalized.\n"
        f"- **CMB**: Planck 2018 distance prior D_M(z*) = {bundle.cmb.D_M_z_rec:.1f} ± "
        f"{bundle.cmb.sigma_D_M:.1f} Mpc at z* = {bundle.cmb.z_rec}.\n"
        f"- **Total N_data** = {n_data}.\n"
    )

    lines.append("## Best-fit parameters (post-burnin posterior median, 68% CI)\n")
    lines.append("| Model | H₀ [km/s/Mpc] | Ω_m | ε | R̂(max) |\n")
    lines.append("|---|---|---|---|---|\n")
    for cfg, result in zip(configs, results):
        eps_str = _fmt_ci(result.credible_intervals["eps"]) if cfg.has_epsilon else "— (fixed 0)"
        rhat_max = max(result.r_hat.values()) if result.r_hat else float("nan")
        lines.append(
            f"| {cfg.name} "
            f"| {_fmt_ci(result.credible_intervals['H0'], 2)} "
            f"| {_fmt_ci(result.credible_intervals['Om'], 4)} "
            f"| {eps_str} "
            f"| {rhat_max:.4f} |\n"
        )

    lines.append("\n## χ² and information criteria (best-fit point)\n")
    lines.append("| Model | χ²(BAO) | χ²(SN) | χ²(CMB) | χ²(total) | AIC | BIC |\n")
    lines.append("|---|---|---|---|---|---|---|\n")
    for name, info in model_info.items():
        lines.append(
            f"| {name} "
            f"| {info['chi2_bao']:.2f} "
            f"| {info['chi2_sn']:.2f} "
            f"| {info['chi2_cmb']:.2f} "
            f"| {info['chi2_total']:.2f} "
            f"| {info['aic']:.2f} "
            f"| {info['bic']:.2f} |\n"
        )

    lines.append("\n## Model comparison (Model B vs Model A)\n")
    for b_name, result in (
        ("B_const", result_B_const),
        ("B_qtep", result_B_qtep),
        ("B_residuals", result_B_res),
    ):
        if result is None or b_name not in model_info:
            continue
        d_chi2 = model_info[b_name]["chi2_total"] - model_info["A_lcdm"]["chi2_total"]
        d_aic = model_info[b_name]["aic"] - model_info["A_lcdm"]["aic"]
        d_bic = model_info[b_name]["bic"] - model_info["A_lcdm"]["bic"]
        lines.append(
            f"- **Model {b_name} − Model A**: Δχ² = **{d_chi2:+.2f}**, "
            f"ΔAIC = {d_aic:+.2f}, ΔBIC = {d_bic:+.2f}.\n"
        )

    lines.append("\n## Hubble tension\n")
    lines.append("Local measurement: SH0ES H₀ = 73.04 ± 1.04 km/s/Mpc (Riess et al. 2022).\n\n")
    for cfg, result in zip(configs, results):
        tension = _sigma_tension_H0(result)
        lines.append(f"- **{cfg.name}**: H₀ tension with SH0ES = {tension:.2f}σ.\n")

    lines.append("\n## ε against framework prediction\n")
    if result_B_const is not None:
        lo, med, hi = result_B_const.credible_intervals["eps"]
        brackets = lo <= FRAMEWORK_EPS_PREDICTED <= hi
        lines.append(
            f"- **Constant-ε fit**: ε = {_fmt_ci(result_B_const.credible_intervals['eps'])}. "
            f"Framework's BAO-paper prediction ε ≈ {FRAMEWORK_EPS_PREDICTED}. "
            f"68% CI {'brackets' if brackets else 'does NOT bracket'} the predicted value.\n"
        )
    if result_B_qtep is not None:
        lo, med, hi = result_B_qtep.credible_intervals["eps"]
        brackets = lo <= FRAMEWORK_EPS_PREDICTED <= hi
        lines.append(
            f"- **QTEP ε(z) fit** (ε₀ is ε at z=0): ε₀ = {_fmt_ci(result_B_qtep.credible_intervals['eps'])}. "
            f"68% CI {'brackets' if brackets else 'does NOT bracket'} framework prediction.\n"
        )
    if result_B_res is not None:
        lo, med, hi = result_B_res.credible_intervals["eps"]
        lines.append(
            f"- **Residual-shape ε(z) fit** (ε_amp scales the signed Planck residual "
            f"z-score): ε_amp = {_fmt_ci(result_B_res.credible_intervals['eps'], 5)}. "
            "Best-fit ε(z) inherits sign oscillations from Planck TT+TE+EE residuals via "
            "ℓ(z) = π·D_M(z*)/(D_M(z*)−D_M(z)).\n"
        )

    if verdict:
        lines.append("\n## Falsifiability verdict\n")
        if verdict["supported"]:
            verdict_str = "**SUPPORTED** — constant-ε framework is preferred by the combined data."
        elif verdict["disfavored"]:
            verdict_str = "**DISFAVORED** — constant-ε framework is not preferred by the combined data."
        else:
            verdict_str = "**REFINEMENT NEEDED** — neither supported nor cleanly disfavored; see QTEP variant."
        lines.append(verdict_str + "\n\n")
        lines.append("| Criterion | Threshold | Measured | Passes |\n")
        lines.append("|---|---|---|---|\n")
        lines.append(f"| ε significance | > 3σ | {verdict['eps_significance']:.2f}σ | {verdict['eps_significance'] > 3} |\n")
        lines.append(f"| Δχ²(B − A) | < -4 | {verdict['delta_chi2']:+.2f} | {verdict['delta_chi2'] < -4} |\n")
        lines.append(f"| H₀ tension (B) | < 2σ | {verdict['H0_tension_B_sigma']:.2f}σ | {verdict['H0_tension_B_sigma'] < 2} |\n")

    lines.append("\n## Figures\n")
    for name, p in figure_paths.items():
        lines.append(f"- `{name}`: [{p.name}]({p.relative_to(reports_dir.parent) if reports_dir.parent in p.parents else p})\n")

    lines.append("\n## Caveats\n")
    lines.append(
        "1. The constant-ε prescription has a hard step at z = 1100. Only the CMB θ* integral "
        "crosses it; BAO and SN data all sit below z_rec, so their integrands see a uniform "
        "enhancement. No numerical discontinuity arises within any data point's z-domain.\n"
        "2. ε was fit independently of the framework's BAO-paper prediction (0.022). The "
        "reported value is data-driven; comparison to 0.022 is a cross-check, not a prior.\n"
        "3. SN absolute magnitude M is profiled out analytically, so H₀ constraints come "
        "from BAO + CMB via r_d; SNe constrain Ω_m.\n"
        "4. **Monotonicity tension (Model B_residuals).** The signed Planck residual "
        "shape inherits sign oscillations (the TT low-ℓ suppression produces ε(z) < 0 "
        "near z ≈ 0, for example). This is in direct tension with the framework's "
        "'screen cannot reverse' monotonicity commitment, which would require ε(z) ≥ 0 "
        "everywhere. We keep the signed shape by design because the residual pattern is "
        "*data-driven* and the goal is to map the observed diffuse tail. Local sign "
        "violations are treated as fluctuations about the tail's mean, not as "
        "disqualifying. A publication should flag this methodological choice explicitly.\n"
    )

    report_path = reports_dir / "expansion_test_summary.md"
    report_path.write_text("".join(lines))

    # JSON companion for downstream tooling.
    json_companion = reports_dir / "expansion_test_summary.json"
    json_companion.write_text(
        json.dumps(
            {
                "models": {name: info for name, info in model_info.items()},
                "verdict": verdict,
                "credible_intervals": {
                    r.model_name: {p: list(v) for p, v in r.credible_intervals.items()}
                    for r in results
                },
                "r_hat": {r.model_name: r.r_hat for r in results},
                "n_data": n_data,
                "generated": datetime.utcnow().isoformat(),
            },
            indent=2,
        )
    )

    return report_path
