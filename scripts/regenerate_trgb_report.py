#!/usr/bin/env python3
"""Re-render the TRGB comparative report from the latest cached results.

After a `python main.py --trgb-comparative extended` run, the pipeline
writes ``results/trgb_comparative/json/trgb_comparative_results.json``
with the legacy chain matrix, full-calibrator chain matrix, and
Uddin-positive-control output. This script reloads that JSON and
re-renders the markdown report (and central-reports copy) using the
current `pipeline.trgb_comparative.reporter.write_summary` to surface
the full-calibrator + positive-control sections.

Use case: code/reporter changes made after the CLI started won't
appear in the run-time-rendered report; this script applies the
latest reporter to the latest cached results without re-running MCMC.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from pipeline.trgb_comparative.reporter import write_summary             # noqa: E402


def _stub_freedman_case(d: Dict[str, Any] | None):
    """Reconstruct enough of FreedmanCaseResult for the reporter from JSON.

    The reporter only needs read-only attribute access, so a SimpleNamespace
    wrapper suffices. Numeric fields are coerced to float; mcmc_result
    inner attributes are reconstructed similarly.
    """
    if d is None:
        return None
    inner_mcmc = d.get("mcmc")
    mcmc_obj = None
    if inner_mcmc:
        mcmc_obj = SimpleNamespace(
            param_names=list(inner_mcmc.get("param_names", [])),
            credible_intervals={k: tuple(v) for k, v in
                                inner_mcmc.get("credible_intervals", {}).items()},
            r_hat=inner_mcmc.get("r_hat", {}),
        )
    anchor_d = d.get("anchor")
    anchor_obj = None
    if anchor_d:
        anchor_obj = SimpleNamespace(**anchor_d)
    return SimpleNamespace(
        case=d.get("case", "?"),
        H0_published=float(d.get("H0_published", 0.0)),
        H0_sigma_stat_published=float(d.get("H0_sigma_stat_published", 0.0)),
        H0_sigma_sys_published=float(d.get("H0_sigma_sys_published", 0.0)),
        mcmc_posterior_H0_pantheon_plus=float(d.get(
            "mcmc_posterior_H0_pantheon_plus", float("nan"))),
        mcmc_posterior_sigma_pantheon_plus=float(d.get(
            "mcmc_posterior_sigma_pantheon_plus", float("nan"))),
        mcmc_n_walkers=int(d.get("mcmc_n_walkers", 0)),
        mcmc_n_steps=int(d.get("mcmc_n_steps", 0)),
        mcmc_n_burnin=int(d.get("mcmc_n_burnin", 0)),
        mcmc_rhat_max=float(d.get("mcmc_rhat_max", float("nan"))),
        mcmc_converged=bool(d.get("mcmc_converged", False)),
        mcmc_convergence_gate=float(d.get("mcmc_convergence_gate", 1.01)),
        pantheon_plus_mcmc_delta=float(d.get("pantheon_plus_mcmc_delta", 0.0)),
        pantheon_plus_mcmc_within_tolerance=bool(d.get(
            "pantheon_plus_mcmc_within_tolerance", False)),
        tolerance_mag=float(d.get("tolerance_mag", 0.0)),
        literature_citations=d.get("literature_citations", {}),
        mcmc_result=mcmc_obj,
        anchor=anchor_obj,
    )


def _stub_framework(d: Dict[str, Any] | None):
    if d is None:
        return None
    return SimpleNamespace(
        label=d.get("label", "framework"),
        H0_median=float(d.get("H0_median", float("nan"))),
        H0_low=float(d.get("H0_low", float("nan"))),
        H0_high=float(d.get("H0_high", float("nan"))),
        breakdown_fraction=float(d.get("breakdown_fraction", 0.0)),
        breakdown_flag_any=bool(d.get("breakdown_flag_any", False)),
        breakdown_messages=list(d.get("breakdown_messages", [])),
        inputs=d.get("inputs", {}),
    )


def main() -> int:
    json_path = (ROOT / "results" / "json" /
                 "trgb_comparative_results.json")
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run "
              f"`python main.py --trgb-comparative extended` first.")
        return 1

    print(f"Loading cached results from {json_path}")
    payload = json.loads(json_path.read_text())
    # Pipeline wraps under "results" / "main"; unwrap.
    container = payload.get("results", payload)
    main_block = container.get("main", container)

    case_a = _stub_freedman_case(main_block.get("case_a"))
    case_b = _stub_freedman_case(main_block.get("case_b"))
    framework_a = _stub_framework(main_block.get("framework_a"))
    framework_b = _stub_framework(main_block.get("framework_b"))
    legacy_matrix = main_block.get("sn_system_chains")
    full_cal_matrix = main_block.get("full_calibrator_chains")
    positive_control = main_block.get("uddin_positive_control")
    figure_paths = {k: Path(v)
                    for k, v in main_block.get("figures", {}).items()}

    reports_dir = ROOT / "results" / "trgb_comparative" / "reports"
    print(f"Re-rendering {reports_dir}/trgb_comparative_analysis_report.md "
          "with the current reporter (incl. full-cal + positive-control sections) …")
    report_path = write_summary(
        case_a, case_b, framework_a, framework_b,
        figure_paths, reports_dir,
        chain_matrix=legacy_matrix,
        full_calibrator_matrix=full_cal_matrix,
        uddin_positive_control=positive_control,
    )
    print(f"OK → {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
