"""TRGB comparative pipeline reporting helper.

Thin wrapper: the heavy lifting lives in
:mod:`pipeline.trgb_comparative.reporter`. This module exposes ``results(...)``
for the central ``HLambdaDMReporter`` dispatch so the pipeline plugs into the
standard machinery.
"""

from __future__ import annotations

from typing import Any, Dict


def _fmt(value, precision: int = 3, default: str = "N/A") -> str:
    if value is None:
        return default
    try:
        return f"{float(value):.{precision}f}"
    except (TypeError, ValueError):
        return str(value)


def results(main_results: Dict[str, Any]) -> str:
    """Render a TRGB-comparative analysis section for the central report."""
    # main.py's run_pipeline_analysis wraps our returned {main, settings}
    # under another "main" key, so unwrap one level if we see a nested
    # ``main`` sub-dict.
    if "main" in main_results and isinstance(main_results["main"], dict):
        inner = main_results["main"]
        if "case_a" in inner or "case_b" in inner or "framework_a" in inner:
            main_results = inner

    case_a = main_results.get("case_a")
    case_b = main_results.get("case_b")
    framework_a = main_results.get("framework_a")
    framework_b = main_results.get("framework_b")

    out = ["### TRGB Comparative Analysis (LMC + NGC 4258)\n\n"]

    out.append(
        "Paired reproduction of Freedman 2019/2020 (LMC anchor) and Freedman "
        "2024/2025 (NGC 4258 anchor) CCHP TRGB H₀ measurements, with H-ΛCDM "
        "framework forward predictions via the holographic projection formula.\n\n"
    )

    def _case_block(label: str, c: Dict[str, Any]) -> str:
        s = [f"#### {label}\n\n"]
        s.append(
            f"- Published: H₀ = {c.get('H0_published')} ± "
            f"{c.get('H0_sigma_stat_published')} (stat) ± "
            f"{c.get('H0_sigma_sys_published')} (sys)\n"
        )
        s.append(
            f"- **MCMC posterior (Pantheon+SH0ES flow, pipeline-computed):** "
            f"H₀ = {_fmt(c.get('mcmc_posterior_H0_pantheon_plus'))} ± "
            f"{_fmt(c.get('mcmc_posterior_sigma_pantheon_plus'))} (stat); "
            f"R̂_max = {_fmt(c.get('mcmc_rhat_max'), 4)} "
            f"({'converged' if c.get('mcmc_converged') else 'NOT CONVERGED'})\n"
        )
        s.append(
            f"- Δ(MCMC_Pantheon+ − published) = "
            f"{_fmt(c.get('pantheon_plus_mcmc_delta'))}; within "
            f"±{c.get('tolerance_mag')}: "
            f"{'YES' if c.get('pantheon_plus_mcmc_within_tolerance') else 'NO'}\n"
        )
        cits = c.get("literature_citations") or {}
        if cits:
            s.append("- Literature citations (NOT pipeline MCMC):\n")
            for tag, rec in cits.items():
                if rec.get("H0") is not None:
                    s.append(f"  - `{tag}`: H₀ = {float(rec['H0']):.3f}  ({rec.get('nature','')})\n")
        return "".join(s)

    out.append("#### Case A — Freedman 2019/2020 (LMC anchor)\n\n")
    if case_a:
        out.append(_case_block("Case A values", case_a))
    else:
        out.append("_Not run in this invocation._\n")
    if framework_a:
        out.append(
            f"- Framework-predicted H₀ (LMC anchor): {_fmt(framework_a.get('H0_median'))} "
            f"[{_fmt(framework_a.get('H0_low'))}, {_fmt(framework_a.get('H0_high'))}] "
            f"(breakdown fraction = {_fmt(framework_a.get('breakdown_fraction'), 2)})\n"
        )
    out.append("\n")

    out.append("#### Case B — Freedman 2024/2025 (NGC 4258 anchor)\n\n")
    if case_b:
        out.append(_case_block("Case B values", case_b))
    else:
        out.append("_Not run in this invocation._\n")
    if framework_b:
        out.append(
            f"- Framework-predicted H₀ (NGC 4258 anchor): {_fmt(framework_b.get('H0_median'))} "
            f"[{_fmt(framework_b.get('H0_low'))}, {_fmt(framework_b.get('H0_high'))}] "
            f"(breakdown fraction = {_fmt(framework_b.get('breakdown_fraction'), 2)})\n"
        )
    out.append("\n")

    # Cross-case comparison — uses MCMC Pantheon+ posteriors ONLY, since
    # those are the only pipeline-computed H₀ values.
    if case_a and case_b and framework_a and framework_b:
        obs_shift = (
            float(case_b.get('mcmc_posterior_H0_pantheon_plus'))
            - float(case_a.get('mcmc_posterior_H0_pantheon_plus'))
        )
        fw_shift = float(framework_b.get('H0_median')) - float(framework_a.get('H0_median'))
        out.append(
            "#### Cross-case shift (LMC → NGC 4258)\n\n"
            f"- Pipeline MCMC Pantheon+ Δ(Case B − Case A): {obs_shift:+.3f} km/s/Mpc\n"
            f"- Framework predicted Δ: {fw_shift:+.3f} km/s/Mpc\n\n"
        )

    report_path = main_results.get("report")
    if report_path:
        out.append(f"Full standalone report: `{report_path}`\n\n")

    return "".join(out)


__all__ = ["results"]
