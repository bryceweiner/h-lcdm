"""
TRGBComparativePipeline — top-level orchestrator.

Subcommands routed from ``main.py``:

- ``preregister_stage1``: write
  ``trgb_data/prereg/trgb_comparative_preregistration_stage1.md``
  (pre-data freeze of methodology).
- ``load_data``: download archival photometry and build the data bundles.
- ``preregister_stage2``: resolve the preregistered selection rules against
  the loaded data; write
  ``trgb_data/prereg/trgb_comparative_preregistration_stage2.md``.
- ``run``: execute both Freedman reproductions (emcee MCMC) and the
  framework forward predictions; generate figures and markdown report.
  ``run`` regenerates both stage documents at the start of the run so a
  full pipeline invocation never depends on stale prereg artefacts.

Both cases always run; unconditional reporting.
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
from .data_acquisition_narrative import write_data_acquisition_narrative
from .full_calibrator_factories import (
    CoverageReport,
    all_chains_full,
)
from .latex_tables import write_latex_data_tables
from .reporter import write_summary
from .manuscript_figures import generate_manuscript_figures
from .twelve_chain_matrix import write_twelve_chain_matrix
from .sn_chain_factories import _run_one as _run_chain_plan
from .sn_chain_factories import run_all_chains_both_cases
from .uddin_csp_chain import (
    build_uddin_inputs_from_loader_dataset,
    run_uddin_csp_chain,
)
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
        # Pre-registration documents land in `trgb_data/prereg/` so they
        # sit alongside the analysis catalog data and stay separate from
        # code documentation in `docs/`.
        self.prereg_dir = Path("trgb_data/prereg")

    # ------------------------------------------------------------------
    # Subcommand: Preregister Stage 1
    # ------------------------------------------------------------------

    def preregister_stage1(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        cfg = Stage1Config()
        if context:
            if "compute_backend" in context:
                cfg.compute_backend = str(context["compute_backend"])
        path = generate_stage1(self.prereg_dir, cfg)
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

        path = generate_stage2(self.prereg_dir, config=cfg)
        self.log_progress(f"Stage 2 preregistration written → {path}")
        return {"stage2_path": str(path)}

    # ------------------------------------------------------------------
    # Full-calibrator chain matrix (audit Recommendation 1) +
    # Uddin 2023 positive-control test (audit Recommendation 3)
    # ------------------------------------------------------------------

    def run_full_calibrator_matrix(
        self,
        loader: DataLoader,
        settings: MCMCSettings,
        chains_dir: Path,
        *,
        include_jwst_only_sensitivity: bool = True,
    ) -> Dict[str, Dict[str, Dict[str, Any]]]:
        """Run the full-Freedman-calibrator-sample chain matrix.

        Produces eight primary chains (Case A 4 systems + Case B primary 4
        systems on the F2025 24-SN augmented sample) and four Case-B JWST-
        only sensitivity chains (F2025 11-SN Table 2 subset). Each chain
        operates on the full Freedman calibrator sample appropriate to its
        case rather than the legacy Uddin-intersection subset.

        Output filenames carry a ``_full`` suffix to preserve the previous
        intersection chains alongside.

        Returns ``{case: {system: result_dict}}`` where ``result_dict``
        carries the chain's H₀ posterior plus calibrator coverage notes
        (``n_calibrators_requested``, ``n_calibrators_matched``,
        ``missing_sn_names``, ``coverage_notes``).
        """
        plans = all_chains_full(
            loader, include_jwst_only_sensitivity=include_jwst_only_sensitivity
        )
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for case, systems in plans.items():
            out[case] = {}
            for system, (plan, cov) in systems.items():
                if plan is None:
                    out[case][system] = {
                        "case": case, "system": system,
                        "error": cov.notes, "skipped": True,
                    }
                    continue
                chain_path = chains_dir / f"{case}_{system}_full.npz"
                self.log_progress(
                    f"[{case}/{system}] full-cal: requested="
                    f"{cov.requested_cal_count} matched={cov.matched_cal_count} "
                    f"→ running chain"
                )
                try:
                    result = _run_chain_plan(
                        plan, settings, chain_out_path=chain_path,
                        log_fn=self.log_progress,
                    )
                    rdict = result.as_dict()
                    rdict["mode"] = plan.mode
                    rdict["n_calibrators_requested"] = cov.requested_cal_count
                    rdict["n_calibrators_matched"] = cov.matched_cal_count
                    rdict["missing_sn_names"] = list(cov.missing_sn_names)
                    rdict["coverage_notes"] = cov.notes
                    out[case][system] = rdict
                except Exception as exc:
                    self.log_progress(
                        f"[{case}/{system}] FAILED — {type(exc).__name__}: {exc}"
                    )
                    out[case][system] = {
                        "case": case, "system": system,
                        "error": f"{type(exc).__name__}: {exc}",
                        "failed": True,
                        "n_calibrators_requested": cov.requested_cal_count,
                        "n_calibrators_matched": cov.matched_cal_count,
                    }
        return out

    def run_uddin_positive_control(
        self, loader: DataLoader, settings: MCMCSettings, chain_path: Path,
    ) -> Dict[str, Any]:
        """Reproduce Uddin 2023 H₀ = 70.242 ± 0.724 as a positive control.

        Validates the 8-parameter SNooPy likelihood implementation. Runs
        the full Uddin `B_trgb_update3.csv` sample (no calibrator
        intersection) under the same likelihood used by the Case A/B CSP
        chains. Acceptance: ``|Δ vs 70.242| ≤ 1.0`` AND ``R̂_max < 1.01``.
        """
        uddin = loader.load_uddin_h0csp_trgb_dataset()
        inputs = build_uddin_inputs_from_loader_dataset(
            uddin, flow_sample_filter='both',
        )
        self.log_progress(
            f"[positive-control] Uddin combined sample: "
            f"n_cal={inputs.n_cal}, n_flow={inputs.n_flow}"
        )
        res = run_uddin_csp_chain(
            inputs, settings,
            case='positive_control',
            system='uddin_2023_full',
            system_label='Uddin 2023 H0CSP positive control (full CSP-I+CSP-II)',
            published_target_H0=70.242, published_sigma_stat=0.724,
            notes=("Uddin 2023 ApJ 970, 72 8-parameter SNooPy MCMC on full "
                   "B_trgb_update3.csv (CSP-I+CSP-II combined). Target H0 = "
                   "70.242 ± 0.724 km/s/Mpc."),
            chain_out_path=chain_path,
            log_fn=self.log_progress,
        )
        delta = res.H0_median - 70.242
        passed = abs(delta) <= 1.0 and res.converged
        self.log_progress(
            f"[positive-control] H0 = {res.H0_median:.3f} ± {res.H0_sigma:.3f}  "
            f"Δ = {delta:+.3f}  R̂_max = {res.rhat_max:.4f}  "
            f"PASS = {passed}"
        )
        return {
            "H0_median": float(res.H0_median),
            "H0_sigma": float(res.H0_sigma),
            "rhat_max": float(res.rhat_max),
            "converged": bool(res.converged),
            "target_H0": 70.242, "target_sigma": 0.724,
            "delta": float(delta),
            "pass": bool(passed),
            "n_calibrators": int(res.n_calibrators),
            "n_flow": int(res.n_flow),
            "n_walkers": int(res.n_walkers),
            "n_steps": int(res.n_steps),
            "n_burnin": int(res.n_burnin),
            "credible_intervals": {k: list(v) for k, v in res.credible_intervals.items()},
        }

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
        # Note: sn_system_a / sn_system_b are no longer used to select
        # between citations — MCMC posteriors are always Pantheon+SH0ES-based
        # at present. CSP-I/II MCMC chains are scheduled for Steps 7/8 of
        # the resolution task.

        # Pre-registration: regenerate both stage documents at the start
        # of the run so the artefacts always reflect the current
        # methodology code. Stage 1 is purely deterministic from the
        # config dataclass; Stage 2 resolves selection criteria against
        # the loaded data bundles. Both land in `trgb_data/prereg/`.
        self.log_progress(
            "Generating pre-registration documents in "
            f"{self.prereg_dir}…"
        )
        stage1_result = self.preregister_stage1(
            {"compute_backend": ctx.get("compute_backend", "auto")}
        )
        stage2_result = self.preregister_stage2(
            {"loader_git_commit": ctx.get("loader_git_commit", "")}
        )
        if enforce_preregistration:
            stage1, stage2 = verify_preregistration_exists(self.prereg_dir)
            self.log_progress(
                f"Pre-registration OK: stage1={stage1.name}, stage2={stage2.name}"
            )

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

        chains_dir = Path("trgb_data/chains")
        chains_dir.mkdir(parents=True, exist_ok=True)

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
                f"Running Case A Pantheon+SH0ES MCMC "
                f"(parametrization={parametrization})…"
            )
            case_a_result = run_freedman_2020(
                bundle_a,
                settings,
                chain_out_path=chains_dir / "freedman_2020_pantheon_plus.npz",
                log_fn=self.log_progress,
                tolerance_mag=0.8,
                parametrization=parametrization,
                hoyt_tables=hoyt_tables,
            )
        else:
            self.log_progress("Case A skipped: no host fields available.")

        # --- Case B ---
        case_b_result: Optional[FreedmanCaseResult] = None
        if bundle_b is not None and bundle_b.host_fields:
            self.log_progress(
                f"Running Case B Pantheon+SH0ES MCMC "
                f"(parametrization={parametrization})…"
            )
            case_b_result = run_freedman_2024(
                bundle_b,
                settings,
                chain_out_path=chains_dir / "freedman_2024_pantheon_plus.npz",
                log_fn=self.log_progress,
                tolerance_mag=1.22,
                parametrization=parametrization,
                hoyt_tables=hoyt_tables,
            )
        else:
            self.log_progress("Case B skipped: no host fields available.")

        # --- Per-SN-system MCMC chains: legacy intersection + full-calibrator runs ---
        # Two chain matrices are produced:
        # 1. Legacy intersection-based chains (Uddin TRGB-cal subset only) — preserved
        #    at `case_*_*.npz` for sensitivity comparison.
        # 2. **Full-calibrator chains** (audit Recommendation 1) — operate on the
        #    full Freedman calibrator sample appropriate to each case (F2019 18 SN
        #    for Case A; F2025 24-SN augmented HST+JWST for Case B; plus an
        #    11-SN F2025 Table 2 JWST-only sensitivity variant). Output filenames
        #    carry a `_full` suffix and these are the primary results going forward.
        self.log_progress("Running legacy per-(case, SN-system) intersection chains (4 systems × 2 cases = 8 chains)…")
        chain_matrix = run_all_chains_both_cases(
            loader, settings, chains_dir=chains_dir, log_fn=self.log_progress,
        )

        self.log_progress("Running FULL-CALIBRATOR per-(case, SN-system) chains (12 chains incl. JWST-only sensitivity)…")
        full_cal_matrix = self.run_full_calibrator_matrix(
            loader, settings, chains_dir,
            include_jwst_only_sensitivity=True,
        )

        self.log_progress("Running Uddin 2023 positive-control test (target H₀ = 70.242)…")
        try:
            positive_control = self.run_uddin_positive_control(
                loader, settings, chains_dir / "uddin_positive_control.npz",
            )
        except Exception as exc:
            self.log_progress(f"[positive-control] FAILED: {exc}")
            positive_control = {
                "error": f"{type(exc).__name__}: {exc}", "pass": False,
            }
        n_converged = 0
        for case_id, systems in chain_matrix.items():
            for sys_id, r in systems.items():
                if r.get("error"):
                    self.log_progress(
                        f"[{case_id}/{sys_id}] FAILED/skipped: {r.get('error')}"
                    )
                elif r.get("converged"):
                    n_converged += 1
                    self.log_progress(
                        f"[{case_id}/{sys_id}] H₀ = {r['H0_median']:.3f} ± "
                        f"{r['H0_sigma']:.3f}  R̂={r.get('rhat_max', r.get('rhat_H0')):.4f}  "
                        "(converged)"
                    )
                else:
                    self.log_progress(
                        f"[{case_id}/{sys_id}] H₀ = {r['H0_median']:.3f} ± "
                        f"{r['H0_sigma']:.3f}  R̂={r.get('rhat_max', r.get('rhat_H0')):.4f}  "
                        "(NOT CONVERGED; reported but not promoted)"
                    )
        self.log_progress(
            f"Chain matrix: {n_converged} / 8 chains converged at R̂ < 1.01."
        )

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
        self.log_progress("Rendering diagnostic figures…")
        figure_paths = generate_all_figures(
            case_a_result, case_b_result, framework_a, framework_b, self.figures_dir
        )

        # Manuscript figures need the 12-chain CSV; that's written below
        # in this method, so we generate manuscript figures after the CSV
        # is in place. Defer the markdown summary until then so it can
        # reference both legacy and manuscript figure sets.

        self.log_progress("Writing publication-ready LaTeX data tables…")
        latex_tables_path = write_latex_data_tables(
            loader, self.reports_dir / "data_tables.tex",
            log_fn=self.log_progress,
        )

        self.log_progress("Writing peer-review data acquisition narrative…")
        data_narrative_path = write_data_acquisition_narrative(
            loader, self.reports_dir / "data_acquisition_narrative.md",
            log_fn=self.log_progress,
        )

        # 12-chain reproduction matrix CSV (per the resolution plan
        # `results/12_chain_matrix.csv`). Always regenerated from the
        # current full_cal_matrix + framework predictions; no special-
        # case code path.
        self.log_progress("Writing 12-chain reproduction matrix CSV…")
        twelve_chain_csv_path = write_twelve_chain_matrix(
            full_cal_matrix, framework_a, framework_b,
            Path("results") / "12_chain_matrix.csv",
            log_fn=self.log_progress,
        )

        # Manuscript figures — communicate the empirical findings to
        # readers unfamiliar with the pipeline. Saved to a tracked,
        # persistent location (`figures/manuscript/`) outside `results/`.
        self.log_progress("Rendering manuscript figures…")
        manuscript_figures = generate_manuscript_figures(
            chain_matrix_csv=twelve_chain_csv_path,
            framework_a=framework_a, framework_b=framework_b,
            out_dir=Path("figures") / "manuscript",
        )
        for name, (pdf_path, png_path) in manuscript_figures.items():
            self.log_progress(f"  {name} → {pdf_path}, {png_path}")

        self.log_progress("Writing markdown summary…")
        report_path = write_summary(
            case_a_result,
            case_b_result,
            framework_a,
            framework_b,
            figure_paths,
            self.reports_dir,
            chain_matrix=chain_matrix,
            full_calibrator_matrix=full_cal_matrix,
            uddin_positive_control=positive_control,
            manuscript_figures=manuscript_figures,
        )
        self.log_progress(f"Report written → {report_path}")

        results_dict: Dict[str, Any] = {
            "main": {
                "case_a": case_a_result.as_dict() if case_a_result else None,
                "case_b": case_b_result.as_dict() if case_b_result else None,
                "framework_a": framework_a.as_dict(),
                "framework_b": framework_b.as_dict(),
                "sn_system_chains": chain_matrix,
                "full_calibrator_chains": full_cal_matrix,
                "uddin_positive_control": positive_control,
                "figures": {k: str(v) for k, v in figure_paths.items()},
                "report": str(report_path),
                "latex_data_tables": str(latex_tables_path),
                "data_acquisition_narrative": str(data_narrative_path),
                "twelve_chain_matrix_csv": str(twelve_chain_csv_path),
                "manuscript_figures": {
                    name: {"pdf": str(p[0]), "png": str(p[1])}
                    for name, p in manuscript_figures.items()
                },
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
                # Post-2026-04-25 C(G)-removal correction: the formula reduced
                # to the linear form 1 + (γ/H)·L, with breakdown criterion
                # |γ/H · L| ≥ 1. Under this form, NEITHER anchor triggers
                # breakdown — γ/H · L ≈ 0.045 at LMC and ≈ 0.027 at NGC 4258,
                # both well below 1. Predicted H₀ medians: LMC ≈ 70.40,
                # NGC 4258 ≈ 69.20.
                "framework_lmc_breakdown_should_be_0": bool(case_a.breakdown_fraction < 0.05),
                "framework_ngc4258_breakdown_should_be_0": bool(case_b.breakdown_fraction < 0.05),
                "passed": bool(
                    case_a.breakdown_fraction < 0.05
                    and case_b.breakdown_fraction < 0.05
                    and 69.0 < case_a.H0_median < 71.5
                    and 68.0 < case_b.H0_median < 70.5
                ),
            }
        }

    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extended validation.

        Runs three independent reproduction checks:

        1. **Legacy intersection 8-chain matrix** (sensitivity baseline) —
           Uddin TRGB-cal subset only.
        2. **Full-calibrator 12-chain matrix** (audit Recommendation 1) —
           full Freedman 2019 / 2025 calibrator samples; primary results
           going forward. Includes Case-B JWST-only sensitivity variants.
        3. **Uddin 2023 positive-control test** (audit Recommendation 3) —
           reproduces Uddin's published H₀ = 70.242 ± 0.724 to validate
           the 8-parameter SNooPy likelihood implementation.

        The validation passes only if (a) every full-calibrator chain
        meets the R̂ < 1.01 gate AND (b) the positive control passes
        (|Δ| ≤ 1.0 km/s/Mpc and converged).
        """
        ctx = context or {}
        short = bool(ctx.get("short", False))
        settings = MCMCSettings.short() if short else MCMCSettings()
        loader = DataLoader(log_file=self.log_file)
        chains_dir = Path("trgb_data/chains")
        chains_dir.mkdir(parents=True, exist_ok=True)

        self.log_progress(
            "[validate_extended] (1/3) legacy intersection 8-chain matrix "
            f"(walkers={settings.n_walkers}, steps={settings.n_steps}, "
            f"burn-in={settings.n_burnin}) …"
        )
        legacy_matrix = run_all_chains_both_cases(
            loader, settings, chains_dir=chains_dir, log_fn=self.log_progress,
        )

        self.log_progress(
            "[validate_extended] (2/3) FULL-CALIBRATOR 12-chain matrix "
            "(F2019 18-SN / F2025 24-SN augmented / F2025 11-SN JWST-only) …"
        )
        full_cal_matrix = self.run_full_calibrator_matrix(
            loader, settings, chains_dir,
            include_jwst_only_sensitivity=True,
        )

        self.log_progress(
            "[validate_extended] (3/3) Uddin 2023 positive-control test "
            "(target H₀ = 70.242 ± 0.724 km/s/Mpc) …"
        )
        try:
            positive_control = self.run_uddin_positive_control(
                loader, settings, chains_dir / "uddin_positive_control.npz",
            )
        except Exception as exc:
            self.log_progress(f"[positive-control] FAILED: {exc}")
            positive_control = {
                "error": f"{type(exc).__name__}: {exc}", "pass": False,
            }

        # Compute aggregate convergence + pass criteria
        def _summarize(matrix: Dict[str, Dict[str, Dict[str, Any]]]) -> Dict[str, Any]:
            n_conv = 0; n_total = 0
            per: Dict[str, Any] = {}
            for case_id, systems in matrix.items():
                for sys_id, rec in systems.items():
                    n_total += 1
                    key = f"{case_id}_{sys_id}"
                    if rec.get("error") or rec.get("skipped") or rec.get("failed"):
                        per[key] = {"error": rec.get("error", "skipped")}
                        continue
                    conv = bool(rec.get("converged", False))
                    if conv:
                        n_conv += 1
                    per[key] = {
                        "H0_median": float(rec.get("H0_median", float("nan"))),
                        "H0_sigma": float(rec.get("H0_sigma", float("nan"))),
                        "rhat_max": float(rec.get("rhat_max",
                                                  rec.get("rhat_H0", float("nan")))),
                        "converged": conv,
                        "n_calibrators_requested": int(rec.get("n_calibrators_requested", 0)),
                        "n_calibrators_matched": int(rec.get(
                            "n_calibrators_matched", rec.get("n_calibrators", 0))),
                        "n_flow": int(rec.get("n_flow", 0)),
                        "missing_sn_names": list(rec.get("missing_sn_names", [])),
                    }
            return {"per_chain": per, "n_converged": n_conv, "n_total": n_total}

        legacy_summary = _summarize(legacy_matrix)
        full_cal_summary = _summarize(full_cal_matrix)

        passed = (
            full_cal_summary["n_converged"] == full_cal_summary["n_total"]
            and bool(positive_control.get("pass", False))
        )

        return {
            "validation_extended": {
                "legacy_intersection_matrix": legacy_matrix,
                "full_calibrator_matrix": full_cal_matrix,
                "uddin_positive_control": positive_control,
                "legacy_summary": legacy_summary,
                "full_calibrator_summary": full_cal_summary,
                "n_converged_full_cal": full_cal_summary["n_converged"],
                "n_total_full_cal": full_cal_summary["n_total"],
                "passed": bool(passed),
                "convergence_gate": 1.01,
                "settings": {
                    "n_walkers": settings.n_walkers,
                    "n_steps": settings.n_steps,
                    "n_burnin": settings.n_burnin,
                    "seed": settings.seed,
                },
            }
        }


__all__ = ["TRGBComparativePipeline"]
