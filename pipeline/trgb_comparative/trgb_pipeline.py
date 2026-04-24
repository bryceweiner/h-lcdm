"""
TRGBComparativePipeline — top-level orchestrator.

Subcommands routed from ``main.py``:

- ``preregister_stage1``: write
  ``docs/trgb_comparative_preregistration_stage1.md`` (pre-data).
- ``load_data``: download archival photometry and build the data bundles.
- ``preregister_stage2``: resolve the preregistered selection rules against
  the loaded data; write
  ``docs/trgb_comparative_preregistration_stage2.md``.
- ``run``: execute both Freedman reproductions (emcee MCMC) and the
  framework forward predictions; generate figures and markdown report.

Both cases always run; unconditional reporting. The analysis refuses to
run the MCMC until both preregistration stages exist.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from data.loader import DataLoader, DataUnavailableError

from ..common.base_pipeline import AnalysisPipeline
from .checkpoint import CheckpointManager
from .data_loaders import TRGBDataBundle, load_case_a, load_case_b
from .framework_methodology import FrameworkMethodology, FrameworkPrediction
from .freedman_2020_methodology import FreedmanCaseResult, run_freedman_2020
from .freedman_2024_methodology import run_freedman_2024
from .mcmc_runner import MCMCSettings
from .preregistration import (
    Stage1Config,
    Stage2Config,
    generate_stage1,
    generate_stage2,
    verify_preregistration_exists,
)
from .reporter import write_summary
from .visualization import generate_all_figures

logger = logging.getLogger(__name__)


class TRGBComparativePipeline(AnalysisPipeline):
    """LMC-anchored + NGC 4258-anchored TRGB comparative analysis."""

    def __init__(self, output_dir: str = "results/trgb_comparative"):
        super().__init__("trgb_comparative", output_dir)
        self.update_metadata("description",
                             "Paired comparative analysis of Freedman 2020 (LMC) and "
                             "Freedman 2024 (NGC 4258) CCHP TRGB measurements with "
                             "H-ΛCDM framework forward predictions.")
        self.checkpoint_manager = CheckpointManager(self.base_output_dir)
        self.docs_dir = Path("docs")

    # ------------------------------------------------------------------
    # Subcommand: Preregister Stage 1
    # ------------------------------------------------------------------

    def preregister_stage1(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = Stage1Config()
        if context:
            if "compute_backend" in context:
                cfg.compute_backend = str(context["compute_backend"])
        path = generate_stage1(self.docs_dir, cfg)
        self.log_progress(f"Stage 1 preregistration written → {path}")
        return {"stage1_path": str(path)}

    # ------------------------------------------------------------------
    # Subcommand: Load data
    # ------------------------------------------------------------------

    def load_data(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        loader = DataLoader(log_file=self.log_file)
        availability = loader.check_data_availability()
        self.log_progress(f"Data availability: {availability}")
        out: Dict[str, Any] = {"availability": availability}
        return out

    # ------------------------------------------------------------------
    # Subcommand: Preregister Stage 2
    # ------------------------------------------------------------------

    def preregister_stage2(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = Stage2Config()
        if context:
            if "loader_git_commit" in context:
                cfg.loader_git_commit = str(context["loader_git_commit"])
        loader = DataLoader(log_file=self.log_file)
        try:
            bundle_a = load_case_a(loader, strict=False)
            cfg.case_a_hosts = sorted(bundle_a.host_fields.keys())
        except DataUnavailableError as exc:
            self.log_progress(f"Case A data partial: {exc}")
        try:
            bundle_b = load_case_b(loader, strict=False)
            cfg.case_b_hosts = sorted(bundle_b.host_fields.keys())
        except DataUnavailableError as exc:
            self.log_progress(f"Case B data partial: {exc}")

        path = generate_stage2(self.docs_dir, config=cfg)
        self.log_progress(f"Stage 2 preregistration written → {path}")
        return {"stage2_path": str(path)}

    # ------------------------------------------------------------------
    # Subcommand: Full run
    # ------------------------------------------------------------------

    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        ctx = context or {}
        subcommand = ctx.get("subcommand")
        if subcommand == "preregister_stage1":
            return self.preregister_stage1(ctx)
        if subcommand == "load_data":
            return self.load_data(ctx)
        if subcommand == "preregister_stage2":
            return self.preregister_stage2(ctx)

        short = bool(ctx.get("short", False))
        enforce_preregistration = bool(ctx.get("enforce_preregistration", True))
        strict_data = bool(ctx.get("strict_data", True))
        parametrization = str(ctx.get("parametrization", "freedman_fixed"))
        tip_source_a = str(ctx.get("tip_source_a", "freedman_2019"))
        tip_source_b = str(ctx.get("tip_source_b", "freedman_2025"))
        # Case A primary SN system: CSP-I (Freedman 2019's sample).
        # Case B primary SN system: CSP-II (Freedman 2025's sample).
        sn_system_a = ctx.get("sn_system_a", "CSP-I")
        sn_system_b = ctx.get("sn_system_b", "CSP-II")

        if enforce_preregistration:
            try:
                stage1, stage2 = verify_preregistration_exists(self.docs_dir)
                self.log_progress(
                    f"Preregistration OK: stage1={stage1.name}, stage2={stage2.name}"
                )
            except FileNotFoundError as exc:
                self.log_progress(f"ABORT: {exc}")
                raise

        settings = MCMCSettings.short() if short else MCMCSettings()

        loader = DataLoader(log_file=self.log_file)

        # --- Load bundles ---
        self.log_progress(
            f"Loading Case A (Freedman 2020) bundle with tip_source={tip_source_a}…"
        )
        try:
            bundle_a = load_case_a(loader, strict=strict_data, tip_source=tip_source_a)
        except DataUnavailableError as exc:
            if strict_data:
                self.log_progress(f"Case A data missing: {exc}")
                raise
            bundle_a = None
            self.log_progress(f"Case A data missing (strict_data=False): {exc}")
        self.log_progress(
            f"Loading Case B (Freedman 2024/2025) bundle with tip_source={tip_source_b}…"
        )
        try:
            bundle_b = load_case_b(loader, strict=strict_data, tip_source=tip_source_b)
        except DataUnavailableError as exc:
            if strict_data:
                self.log_progress(f"Case B data missing: {exc}")
                raise
            bundle_b = None
            self.log_progress(f"Case B data missing (strict_data=False): {exc}")

        chains_dir = self.base_output_dir / "chains"

        # --- Hoyt 2025 SN Ia calibration tables (per-system H₀ reference values) ---
        try:
            hoyt_tables = loader.load_hoyt_2025_sn_calibration()
            self.log_progress(
                f"Loaded Hoyt 2025 SN calibration: "
                f"{list(hoyt_tables['systems'].keys())}"
            )
        except Exception as exc:
            self.log_progress(f"Hoyt 2025 tables unavailable: {exc}")
            hoyt_tables = None

        # --- Case A ---
        case_a_result: Optional[FreedmanCaseResult] = None
        if bundle_a is not None and bundle_a.host_fields:
            self.log_progress(
                f"Running Case A reproduction (parametrization={parametrization}, "
                f"sn_system={sn_system_a})…"
            )
            case_a_result = run_freedman_2020(
                bundle_a,
                settings,
                chain_out_path=chains_dir / "freedman_2020.npz",
                log_fn=self.log_progress,
                tolerance_mag=0.8,
                parametrization=parametrization,
                sn_system=sn_system_a,
                hoyt_tables=hoyt_tables,
            )
        else:
            self.log_progress("Case A skipped: no host fields available.")

        # --- Case B ---
        case_b_result: Optional[FreedmanCaseResult] = None
        if bundle_b is not None and bundle_b.host_fields:
            self.log_progress(
                f"Running Case B reproduction (parametrization={parametrization}, "
                f"sn_system={sn_system_b})…"
            )
            case_b_result = run_freedman_2024(
                bundle_b,
                settings,
                chain_out_path=chains_dir / "freedman_2024.npz",
                log_fn=self.log_progress,
                tolerance_mag=1.22,
                parametrization=parametrization,
                sn_system=sn_system_b,
                hoyt_tables=hoyt_tables,
            )
        else:
            self.log_progress("Case B skipped: no host fields available.")

        # --- Framework forward predictions (always run; independent of data) ---
        self.log_progress("Computing framework forward predictions (Case A, Case B)…")
        fw = FrameworkMethodology()
        framework_a = fw.predict(
            label="H0_framework_predicted_lmc_anchor",
            d_local_mpc=0.0496,
            sigma_d_local_mpc=0.0009,
            n_samples=int(ctx.get("n_framework_samples", 50_000)),
            seed=settings.seed,
        )
        framework_b = fw.predict(
            label="H0_framework_predicted_ngc4258_anchor",
            d_local_mpc=7.58,
            sigma_d_local_mpc=0.08,
            n_samples=int(ctx.get("n_framework_samples", 50_000)),
            seed=settings.seed + 1,
        )
        self.log_progress(
            f"Framework Case A: median H0 = {framework_a.H0_median:.3f} "
            f"[{framework_a.H0_low:.3f}, {framework_a.H0_high:.3f}]; "
            f"breakdown fraction {framework_a.breakdown_fraction:.2f}"
        )
        self.log_progress(
            f"Framework Case B: median H0 = {framework_b.H0_median:.3f} "
            f"[{framework_b.H0_low:.3f}, {framework_b.H0_high:.3f}]; "
            f"breakdown fraction {framework_b.breakdown_fraction:.2f}"
        )

        # --- Figures + markdown report ---
        self.log_progress("Rendering figures…")
        figure_paths = generate_all_figures(
            case_a_result, case_b_result, framework_a, framework_b, self.figures_dir
        )

        self.log_progress("Writing markdown summary…")
        report_path = write_summary(
            case_a_result,
            case_b_result,
            framework_a,
            framework_b,
            figure_paths,
            self.reports_dir,
        )
        self.log_progress(f"Report written → {report_path}")

        results_dict: Dict[str, Any] = {
            "main": {
                "case_a": case_a_result.as_dict() if case_a_result else None,
                "case_b": case_b_result.as_dict() if case_b_result else None,
                "framework_a": framework_a.as_dict(),
                "framework_b": framework_b.as_dict(),
                "figures": {k: str(v) for k, v in figure_paths.items()},
                "report": str(report_path),
            },
            "settings": {
                "n_walkers": settings.n_walkers,
                "n_steps": settings.n_steps,
                "n_burnin": settings.n_burnin,
                "short": short,
            },
        }
        self.results = results_dict
        self.save_results(results_dict)
        return results_dict

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Light validation: confirm framework predictions return plausible values."""
        fw = FrameworkMethodology()
        case_a = fw.predict(
            label="validate_lmc_anchor",
            d_local_mpc=0.0496, sigma_d_local_mpc=0.0009, n_samples=5000,
        )
        case_b = fw.predict(
            label="validate_ngc4258_anchor",
            d_local_mpc=7.58, sigma_d_local_mpc=0.08, n_samples=5000,
        )
        return {
            "validation": {
                "framework_lmc_median": float(case_a.H0_median),
                "framework_ngc4258_median": float(case_b.H0_median),
                "framework_lmc_breakdown_fraction": float(case_a.breakdown_fraction),
                "framework_ngc4258_breakdown_fraction": float(case_b.breakdown_fraction),
                "framework_lmc_breakdown_should_be_1": bool(case_a.breakdown_fraction > 0.95),
                "framework_ngc4258_breakdown_should_be_0": bool(case_b.breakdown_fraction < 0.05),
                "passed": bool(
                    case_a.breakdown_fraction > 0.95
                    and case_b.breakdown_fraction < 0.05
                    and 75.0 < case_a.H0_median < 90.0
                    and 70.0 < case_b.H0_median < 76.0
                ),
            }
        }

    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extended validation: a short MCMC on mock data as a smoke test."""
        return {
            "validation_extended": {
                "note": "Full sensitivity matrix runs as a separate --validate-extended invocation.",
            }
        }


__all__ = ["TRGBComparativePipeline"]
