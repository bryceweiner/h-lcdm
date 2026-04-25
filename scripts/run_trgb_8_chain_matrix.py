#!/usr/bin/env python3
"""Run the 8-chain (4 SN systems × 2 cases) TRGB comparative MCMC matrix.

Production-quality settings (Uddin 2023 8-parameter chains need more steps
than the default expansion_enhancement MCMCSettings). Each chain is
saved to ``trgb_data/chains/<case>_<system>.npz`` and a consolidated
JSON + markdown report is written under
``results/trgb_comparative/reports/``.

Usage
-----
    python scripts/run_trgb_8_chain_matrix.py
        [--n-walkers N] [--n-steps N] [--n-burnin N]
        [--output-dir PATH] [--strict]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

# Suppress RuntimeWarnings from pandas type coercion during loader.
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from data.loader import DataLoader                                    # noqa: E402
from pipeline.trgb_comparative.mcmc_runner import MCMCSettings        # noqa: E402
from pipeline.trgb_comparative.sn_chain_factories import (            # noqa: E402
    run_all_chains_both_cases,
)


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")


def _write_validation_markdown(
    out_dir: Path, chain_matrix: dict, settings: MCMCSettings, start_time: str,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "reproduction_validation.md"
    lines = []
    lines.append("# TRGB comparative reproduction — per-(case, SN-system) chain validation (v3)\n\n")
    lines.append(f"*Generated {_now()}*\n\n")
    lines.append(
        "This report lists the eight independent MCMC chains produced by the "
        "TRGB comparative pipeline for the paired analysis:\n\n"
        "- 4 SN photometric systems × 2 TRGB geometric anchors (LMC / NGC 4258).\n"
        "- CSP-I and CSP-II chains use the Uddin 2023 8-parameter SNooPy "
        "likelihood on the unified `B_trgb_update3.csv` input (native "
        "Carnegie photometric system). Case B swaps the calibrator block's "
        "μ_TRGB for Freedman 2025 Table 2 NGC 4258-anchored distances, "
        "dropping calibrators whose host is absent from the 2025 sample.\n"
        "- SuperCal and Pantheon+SH0ES chains use a simple 1-parameter "
        "(H₀ only; M_B analytically marginalized) likelihood on pre-"
        "standardized m_B values.\n"
        "- Convergence gate: Gelman-Rubin R̂ < 1.01 on all sampled "
        "parameters.\n\n"
        f"MCMC settings: n_walkers={settings.n_walkers}, "
        f"n_steps={settings.n_steps}, n_burnin={settings.n_burnin}, "
        f"seed={settings.seed}.\n\n"
        f"Run started: {start_time}.\n\n"
    )
    lines.append("## Chain matrix\n\n")
    lines.append(
        "| Case | System | Mode | N_cal | N_flow | H₀ (med) | σ(H₀) | "
        "R̂_max | Converged | Published |\n"
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | :---: | ---: |\n"
    )
    for case_key, systems in chain_matrix.items():
        case_label = "A (LMC)" if case_key == "case_a" else "B (NGC 4258)"
        for sys_key, rec in systems.items():
            if rec.get("skipped") or rec.get("failed") or "error" in rec:
                lines.append(
                    f"| {case_label} | {sys_key} | – | – | – | – | – | – | ERROR | – |\n"
                )
                continue
            mode = rec.get("mode", "")
            n_cal = rec.get("n_calibrators", 0)
            n_flow = rec.get("n_flow", 0)
            if not n_cal and rec.get("n_data"):
                n_cal = rec["n_data"]
            h0 = rec["H0_median"]; sigma = rec["H0_sigma"]
            rh = rec.get("rhat_max", rec.get("rhat_H0", float("nan")))
            conv = "✓" if rec.get("converged") else "✗"
            pub = rec.get("published_target_H0")
            pub_str = f"{pub:.2f}" if pub else "–"
            lines.append(
                f"| {case_label} | {sys_key} | {mode} | {n_cal} | {n_flow} | "
                f"{h0:.3f} | {sigma:.3f} | {rh:.4f} | {conv} | {pub_str} |\n"
            )
    lines.append("\n")

    # Cross-case shift by system:
    lines.append("## Cross-case shift (Case B − Case A) per SN system\n\n")
    lines.append(
        "| System | Case A H₀ | Case B H₀ | Δ(B − A) | Both converged |\n"
        "| --- | ---: | ---: | ---: | :---: |\n"
    )
    for sys_id in ("csp_i", "csp_ii", "supercal", "pantheon_plus"):
        ra = chain_matrix.get("case_a", {}).get(sys_id, {})
        rb = chain_matrix.get("case_b", {}).get(sys_id, {})
        if not ra or not rb or "H0_median" not in ra or "H0_median" not in rb:
            continue
        delta = rb["H0_median"] - ra["H0_median"]
        both_conv = ra.get("converged") and rb.get("converged")
        lines.append(
            f"| {sys_id} | {ra['H0_median']:.3f} | {rb['H0_median']:.3f} | "
            f"{delta:+.3f} | {'✓' if both_conv else '✗'} |\n"
        )
    lines.append("\n")

    # Per-chain details (Uddin 8-param posteriors):
    lines.append("## Per-parameter posterior medians (Uddin 8-param chains)\n\n")
    lines.append(
        "| Case | System | M_B | p1 | p2 | β | α | σ_int | v_pec | H₀ |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    for case_key, systems in chain_matrix.items():
        case_label = "A" if case_key == "case_a" else "B"
        for sys_id in ("csp_i", "csp_ii"):
            rec = systems.get(sys_id, {})
            ci = rec.get("credible_intervals", {})
            if not ci:
                continue
            def _med(name):
                v = ci.get(name)
                if not v:
                    return "–"
                return f"{v[1]:.3f}"
            lines.append(
                f"| {case_label} | {sys_id} | {_med('M_B')} | {_med('p1')} | "
                f"{_med('p2')} | {_med('beta')} | {_med('alpha')} | "
                f"{_med('sigma_int')} | {_med('v_pec')} | {_med('H0')} |\n"
            )
    lines.append("\n")

    lines.append("## Discipline notes\n\n")
    lines.append(
        "- Every H₀ value in this report is a **pipeline-computed MCMC "
        "posterior**. No literature citation is promoted into an MCMC-named "
        "field.\n"
        "- Chains that do not meet the R̂ < 1.01 convergence gate are "
        "reported in the table with the ``converged = ✗`` flag and MUST NOT "
        "be promoted as reproductions. Increase n_steps / n_burnin and "
        "re-run if convergence is desired.\n"
        "- Case B CSP chains drop calibrators whose host is absent from "
        "Freedman 2025 Table 2 (NGC 4258 re-anchoring). Typically 7 cal "
        "remain vs 20 in Case A. The wider posterior width is a direct "
        "consequence, not a bug.\n"
    )
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
    p.add_argument("--progress", action="store_true", default=False)
    args = p.parse_args()

    start = _now()
    settings = MCMCSettings(
        n_walkers=args.n_walkers, n_steps=args.n_steps, n_burnin=args.n_burnin,
        seed=args.seed, progress=args.progress,
    )

    # Chains live in trgb_data/chains/ alongside the rest of the
    # TRGB-pipeline data; reports remain under --output-dir.
    chains_dir = Path("trgb_data/chains")
    reports_dir = args.output_dir / "reports"
    chains_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)

    print(f"[run_trgb_8_chain_matrix] starting at {start}")
    print(f"    settings: walkers={settings.n_walkers}, steps={settings.n_steps}, "
          f"burn-in={settings.n_burnin}, seed={settings.seed}")
    print(f"    chains_dir={chains_dir}, reports_dir={reports_dir}")

    loader = DataLoader()
    chain_matrix = run_all_chains_both_cases(
        loader, settings, chains_dir=chains_dir, log_fn=print,
    )

    json_path = reports_dir / "reproduction_validation.json"
    json_path.write_text(json.dumps(
        {
            "generated": _now(),
            "settings": {
                "n_walkers": settings.n_walkers,
                "n_steps": settings.n_steps,
                "n_burnin": settings.n_burnin,
                "seed": settings.seed,
            },
            "chain_matrix": chain_matrix,
        },
        indent=2, default=str,
    ))
    print(f"[run_trgb_8_chain_matrix] JSON written → {json_path}")

    md_path = _write_validation_markdown(reports_dir, chain_matrix, settings, start)
    print(f"[run_trgb_8_chain_matrix] validation markdown → {md_path}")

    # Summary
    n_conv = 0
    n_total = 0
    for case_key, systems in chain_matrix.items():
        for sys_id, rec in systems.items():
            n_total += 1
            if rec.get("converged"):
                n_conv += 1
    print(f"[run_trgb_8_chain_matrix] DONE: {n_conv} / {n_total} chains converged at R̂ < 1.01.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
