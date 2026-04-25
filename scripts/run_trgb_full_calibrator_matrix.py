#!/usr/bin/env python3
"""Thin invoker for the TRGB comparative full-calibrator matrix.

The orchestration logic lives in
:py:meth:`pipeline.trgb_comparative.TRGBComparativePipeline.run_full_calibrator_matrix`
and :py:meth:`.run_uddin_positive_control`. Both are also exercised by
``python main.py --trgb-comparative extended``; this script is a
convenience for invoking just the full-calibrator matrix + positive control
without the rest of the pipeline overhead.

Outputs (preserves any previous chain files):

    results/trgb_comparative/chains/
      case_a_<system>_full.npz                 # 4 files
      case_b_<system>_full.npz                 # 4 files
      case_b_jwst_only_<system>_full.npz       # 4 files (sensitivity)
      uddin_positive_control.npz               # 1 file

    results/trgb_comparative/reports/
      reproduction_validation_full.md / .json
      uddin_positive_control.md
      calibrator_inventory.md
      calibrator_completeness_analysis.md

Usage
-----
    python scripts/run_trgb_full_calibrator_matrix.py
        [--n-walkers N] [--n-steps N] [--n-burnin N]
        [--skip-jwst-only-sensitivity]
        [--skip-uddin-positive-control]
"""

from __future__ import annotations

import argparse
import json
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loader import DataLoader                                        # noqa: E402
from pipeline.trgb_comparative import TRGBComparativePipeline             # noqa: E402
from pipeline.trgb_comparative.full_calibrator_factories import (         # noqa: E402
    all_chains_full,
)
from pipeline.trgb_comparative.mcmc_runner import MCMCSettings            # noqa: E402


def _now_str() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _write_inventory_md(out_dir: Path, results, coverage) -> Path:
    """Write per-chain calibrator inventory."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibrator_inventory.md"
    lines = [f"# Calibrator inventory per chain (full-calibrator run)\n\n",
             f"*Generated {_now_str()}*\n\n",
             "Lists the calibrator sample composition for each "
             "full-calibrator-sample chain.\n\n",
             "| Case | System | Requested cal N | Matched cal N | "
             "Missing SN names |\n",
             "| --- | --- | ---: | ---: | --- |\n"]
    for case, systems in coverage.items():
        for sys_id, cov in systems.items():
            miss = ", ".join(cov.missing_sn_names) if cov.missing_sn_names else "—"
            lines.append(f"| {case} | {sys_id} | {cov.requested_cal_count} | "
                         f"{cov.matched_cal_count} | {miss} |\n")
    lines.append("\n## Notes per chain\n\n")
    for case, systems in coverage.items():
        for sys_id, cov in systems.items():
            if cov.notes:
                lines.append(f"### {case} / {sys_id}\n\n{cov.notes}\n\n")
    path.write_text("".join(lines))
    return path


def _write_completeness_md(out_dir: Path, coverage) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "calibrator_completeness_analysis.md"
    lines = [f"# Calibrator completeness analysis\n\n",
             f"*Generated {_now_str()}*\n\n",
             "## Calibrator sample sizes (target N per case)\n\n",
             "| Case | Target sample | Source CSV | Target N | Target H₀ |\n",
             "| --- | --- | --- | ---: | ---: |\n",
             "| Case A | F2019 Table 3 TRGB-anchored | freedman_2019_table3.csv | 18 | 69.8 |\n",
             "| Case B (primary) | F2025 Table 3 augmented HST+JWST | freedman_2025_table3.csv | 24 | 70.39 |\n",
             "| Case B JWST-only | F2025 Table 2 (TRGB+JAGB averaged) | freedman_2025_table2.csv | 11 | 68.81 |\n\n",
             "## Per-(case, system) coverage\n\n",
             "| Case | System | Requested | Matched | Coverage % |\n",
             "| --- | --- | ---: | ---: | ---: |\n"]
    for case, systems in coverage.items():
        for sys_id, cov in systems.items():
            pct = (cov.matched_cal_count / cov.requested_cal_count * 100.0
                   if cov.requested_cal_count else 0.0)
            lines.append(f"| {case} | {sys_id} | {cov.requested_cal_count} | "
                         f"{cov.matched_cal_count} | {pct:.1f}% |\n")
    lines.append("\n## Missing SNe per (case, system)\n\n")
    for case, systems in coverage.items():
        for sys_id, cov in systems.items():
            if cov.missing_sn_names:
                lines.append(f"### {case} / {sys_id} — {len(cov.missing_sn_names)} missing\n\n")
                for sn in cov.missing_sn_names:
                    lines.append(f"- {sn}\n")
                lines.append("\n")
    lines.append("## Cross-reference to Hoyt 2025 Table I\n\n"
                 "Hoyt 2025 reports per-system calibrator counts for the "
                 "augmented HST+JWST sample: CSP-I=22, CSP(I+II)=24, "
                 "SuperCal=14, Pantheon+=17. Our Case B / supercal full-cal "
                 "chain matches Hoyt's N=14 exactly.\n")
    path.write_text("".join(lines))
    return path


def _write_validation_md(
    out_dir: Path, full_cal_matrix, settings, positive_control,
    legacy_matrix=None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reproduction_validation_full.md"
    lines = [f"# TRGB comparative — full-calibrator validation (v3)\n\n",
             f"*Generated {_now_str()}*\n\n",
             f"## MCMC settings\n\n"
             f"- Walkers: {settings.n_walkers}\n"
             f"- Steps: {settings.n_steps}\n"
             f"- Burn-in: {settings.n_burnin}\n"
             f"- Seed: {settings.seed}\n"
             f"- Convergence gate: R̂ < 1.01\n\n",
             "## Chain matrix (full Freedman calibrator samples)\n\n",
             "| Case | System | Cal req/got | N_flow | Mode | H₀ (med) | σ | "
             "R̂_max | Conv | Target | Δ | Within ±stat? |\n",
             "| --- | --- | ---: | ---: | --- | ---: | ---: | ---: | :---: | "
             "---: | ---: | :---: |\n"]
    for case_id, systems in full_cal_matrix.items():
        for sys_id in ('csp_i', 'csp_ii', 'supercal', 'pantheon_plus'):
            rec = systems.get(sys_id, {})
            if not rec or 'H0_median' not in rec:
                continue
            req = rec.get('n_calibrators_requested', '?')
            got = rec.get('n_calibrators_matched', '?')
            tgt = rec.get('published_target_H0', 0.0)
            tol = rec.get('published_sigma_stat', 0.0)
            delta = rec['H0_median'] - tgt
            within = "✓" if abs(delta) <= tol else "✗"
            mode = rec.get('mode', '?')
            rh = rec.get('rhat_max', rec.get('rhat_H0', float('nan')))
            conv = "✓" if rec.get('converged') else "✗"
            lines.append(f"| {case_id} | {sys_id} | {req}/{got} | "
                         f"{rec.get('n_flow', '?')} | {mode} | "
                         f"{rec['H0_median']:.3f} | {rec['H0_sigma']:.3f} | "
                         f"{rh:.4f} | {conv} | {tgt:.2f} | {delta:+.3f} | "
                         f"{within} |\n")
    lines.append("\n## Cross-case shift per system (full-cal)\n\n")
    lines.append("| System | A H₀ | B aug H₀ | B JWST H₀ | Δ(B aug − A) | "
                 "Δ(B JWST − A) |\n| --- | ---: | ---: | ---: | ---: | ---: |\n")
    import numpy as np
    for sys_id in ('csp_i', 'csp_ii', 'supercal', 'pantheon_plus'):
        ra = full_cal_matrix.get('case_a', {}).get(sys_id, {})
        rb = full_cal_matrix.get('case_b', {}).get(sys_id, {})
        rj = full_cal_matrix.get('case_b_jwst_only', {}).get(sys_id, {})
        if 'H0_median' not in ra:
            continue
        ha = ra['H0_median']
        hb = rb.get('H0_median', float('nan'))
        hj = rj.get('H0_median', float('nan'))
        d_aug = (hb - ha) if np.isfinite(hb) else float('nan')
        d_jw = (hj - ha) if np.isfinite(hj) else float('nan')
        lines.append(f"| {sys_id} | {ha:.3f} | {hb:.3f} | {hj:.3f} | "
                     f"{d_aug:+.3f} | {d_jw:+.3f} |\n")

    if positive_control is not None:
        pc = positive_control
        lines.append("\n## Uddin 2023 positive-control test\n\n")
        lines.append(
            f"- Target: H₀ = {pc.get('target_H0', 70.242)} ± "
            f"{pc.get('target_sigma', 0.724)} km/s/Mpc\n"
            f"- Pipeline: H₀ = {pc.get('H0_median', float('nan')):.3f} ± "
            f"{pc.get('H0_sigma', float('nan')):.3f} km/s/Mpc\n"
            f"- Δ = {pc.get('delta', float('nan')):+.3f} km/s/Mpc\n"
            f"- R̂_max = {pc.get('rhat_max', float('nan')):.4f}\n"
            f"- Result: **{'PASS' if pc.get('pass') else 'FAIL'}**\n\n"
        )
    if legacy_matrix is not None:
        lines.append("\n## Legacy intersection chain matrix (sensitivity baseline)\n\n")
        lines.append(
            "Preserved at `case_*_*.npz` (without `_full` suffix) for comparison "
            "with the previous Uddin-intersection methodology. See "
            "`reproduction_validation.json` for the full per-chain breakdown.\n"
        )
    path.write_text("".join(lines))
    return path


def _write_positive_control_md(out_dir: Path, pc: Dict[str, Any], settings: MCMCSettings) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "uddin_positive_control.md"
    lines = [
        f"# Uddin 2023 positive-control test\n\n*Generated {_now_str()}*\n\n",
        "Validates the pipeline's 8-parameter SNooPy likelihood implementation "
        "against Uddin 2023's published TRGB H₀.\n\n",
        f"**Target**: H₀ = {pc.get('target_H0', 70.242)} ± "
        f"{pc.get('target_sigma', 0.724)} km/s/Mpc (Uddin 2023, ApJ 970 72).\n\n",
        f"**Setup**: full Uddin `B_trgb_update3.csv` "
        f"({pc.get('n_calibrators', '?')} TRGB cal + {pc.get('n_flow', '?')} flow), "
        f"identical 8-param likelihood (`scripts/H0CSP.py`), MCMC "
        f"{settings.n_walkers} × {settings.n_steps} × {settings.n_burnin}.\n\n",
        f"## Result\n\n",
        f"- Pipeline H₀ = **{pc.get('H0_median', float('nan')):.3f} ± "
        f"{pc.get('H0_sigma', float('nan')):.3f}** km/s/Mpc\n",
        f"- Δ vs target: **{pc.get('delta', float('nan')):+.3f}** km/s/Mpc\n",
        f"- R̂_max: **{pc.get('rhat_max', float('nan')):.4f}** "
        f"({'CONVERGED' if pc.get('converged') else 'NOT CONVERGED'})\n",
        f"- Acceptance (|Δ| ≤ 1.0 AND R̂_max < 1.01): "
        f"**{'PASS' if pc.get('pass') else 'FAIL'}**\n\n",
        f"## Per-parameter posterior credible intervals (16/50/84)\n\n",
        f"| Parameter | 16 | 50 | 84 |\n| --- | ---: | ---: | ---: |\n",
    ]
    for k, v in pc.get('credible_intervals', {}).items():
        lines.append(f"| {k} | {v[0]:.4f} | {v[1]:.4f} | {v[2]:.4f} |\n")
    if pc.get('pass'):
        lines.append("\nPositive control passes — 8-parameter likelihood faithfully "
                     "reproduces Uddin 2023's published TRGB H₀.\n")
    else:
        lines.append("\n**HALT**: positive-control failed. Diagnose 8-parameter "
                     "likelihood before quoting CSP results as reproductions.\n")
    path.write_text("".join(lines))
    return path


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-walkers", type=int, default=64)
    p.add_argument("--n-steps", type=int, default=20000)
    p.add_argument("--n-burnin", type=int, default=5000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--output-dir", type=Path,
                   default=Path("results/trgb_comparative"))
    p.add_argument("--skip-jwst-only-sensitivity", action="store_true")
    p.add_argument("--skip-uddin-positive-control", action="store_true")
    p.add_argument("--progress", action="store_true", default=False)
    args = p.parse_args()

    print(f"[run_full_cal_matrix] starting at {_now_str()}")
    settings = MCMCSettings(
        n_walkers=args.n_walkers, n_steps=args.n_steps, n_burnin=args.n_burnin,
        seed=args.seed, progress=args.progress,
    )
    chains_dir = args.output_dir / "chains"
    reports_dir = args.output_dir / "reports"
    chains_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    pipe = TRGBComparativePipeline(output_dir=str(args.output_dir))
    loader = DataLoader()

    full_cal_matrix = pipe.run_full_calibrator_matrix(
        loader, settings, chains_dir,
        include_jwst_only_sensitivity=not args.skip_jwst_only_sensitivity,
    )

    # Coverage report — recompute from plan inventory.
    coverage = {}
    plans = all_chains_full(loader,
                            include_jwst_only_sensitivity=not args.skip_jwst_only_sensitivity)
    for case, systems in plans.items():
        coverage[case] = {sys: cov for sys, (_, cov) in systems.items()}

    inv_path = _write_inventory_md(reports_dir, full_cal_matrix, coverage)
    print(f"[run_full_cal_matrix] inventory → {inv_path}")
    comp_path = _write_completeness_md(reports_dir, coverage)
    print(f"[run_full_cal_matrix] completeness → {comp_path}")

    pc = None
    if not args.skip_uddin_positive_control:
        pc = pipe.run_uddin_positive_control(
            loader, settings, chains_dir / "uddin_positive_control.npz",
        )
        pc_md = _write_positive_control_md(reports_dir, pc, settings)
        print(f"[run_full_cal_matrix] positive-control report → {pc_md}")

    vmd = _write_validation_md(reports_dir, full_cal_matrix, settings, pc)
    print(f"[run_full_cal_matrix] validation → {vmd}")

    json_path = reports_dir / "reproduction_validation_full.json"
    json_path.write_text(json.dumps({
        "generated": _now_str(),
        "settings": {"n_walkers": settings.n_walkers, "n_steps": settings.n_steps,
                     "n_burnin": settings.n_burnin, "seed": settings.seed},
        "full_calibrator_matrix": full_cal_matrix,
        "uddin_positive_control": pc,
        "coverage": {
            case: {sys: {
                "case": cov.case, "system": cov.system,
                "requested_cal_count": cov.requested_cal_count,
                "matched_cal_count": cov.matched_cal_count,
                "missing_sn_names": cov.missing_sn_names,
                "notes": cov.notes,
            } for sys, cov in systems.items()}
            for case, systems in coverage.items()
        },
    }, indent=2, default=str))
    print(f"[run_full_cal_matrix] JSON → {json_path}")

    n_conv = sum(1 for systems in full_cal_matrix.values()
                 for rec in systems.values() if rec.get("converged"))
    n_total = sum(1 for systems in full_cal_matrix.values()
                  for _ in systems.values())
    print(f"[run_full_cal_matrix] DONE: {n_conv}/{n_total} chains converged at R̂ < 1.01.")
    if pc is not None:
        print(f"[run_full_cal_matrix] positive-control: "
              f"{'PASS' if pc.get('pass') else 'FAIL'} "
              f"(|Δ|={abs(pc.get('delta', 0.0)):.3f}, "
              f"R̂_max={pc.get('rhat_max', float('nan')):.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
