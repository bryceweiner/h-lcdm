"""
Two-stage preregistration for the TRGB comparative analysis.

Stage 1 is generated BEFORE any data loading. It freezes framework sources,
data source URLs/archives, selection-criteria RULES (not resolved values),
methodology choices, MCMC settings, d_local values per case, and tolerance
targets.

Stage 2 is generated AFTER data loading but BEFORE analysis. It freezes
the resolved host lists, the specific numerical values that required data
loading to determine, and SHA-256 checksums of every archive used.

Both documents are committed to the repo at
``docs/trgb_comparative_preregistration_stage{1,2}.md``. The main pipeline
refuses to run the analysis unless both exist.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple


DOCS_DIR_DEFAULT = Path("docs")
STAGE1_FILENAME = "trgb_comparative_preregistration_stage1.md"
STAGE2_FILENAME = "trgb_comparative_preregistration_stage2.md"


# ---------------------------------------------------------------------------
# Stage 1 — pre-data
# ---------------------------------------------------------------------------


@dataclass
class Stage1Config:
    generated: str = ""

    # Framework inputs
    gamma_over_H_source: str = (
        "HLCDMCosmology.gamma_at_redshift(0.0) / HLCDM_PARAMS.get_hubble_at_redshift(0.0) "
        "— runtime computation, no hardcoded constant."
    )
    clustering_coefficient_source: str = (
        "hlcdm.e8.e8_cache.E8Cache.get_clustering_coefficient() "
        "— theoretical value 25/32 = 0.78125."
    )
    d_cmb_source: str = (
        "HLCDM_PARAMS.D_CMB_PLANCK_2018 = 13869.7 Mpc "
        "(Planck 2018 VI, A&A 641 A6, Table 2)."
    )
    perturbative_threshold_mpc: float = 1.0

    # Reproduction configuration
    parametrization: str = (
        "freedman_fixed — only H0 is sampled; M_TRGB, E(B-V), and β are held "
        "at published central values (Freedman's frequentist-profile approach)."
    )
    parametrization_sensitivity: str = (
        "bayesian_sampled — all 4 parameters (H0, M_TRGB, E(B-V), β) sampled "
        "with Gaussian priors; retained as a sensitivity-analysis variant."
    )
    tip_source_case_a: str = (
        "freedman_2019 — per-host μ_TRGB read from data/catalogs/"
        "freedman_2019_table3.csv (Freedman 2019 Table 3 transcription, SHA-256 "
        "verified; 15 unique hosts, 18 SN Ia calibrators)."
    )
    tip_source_case_b: str = (
        "freedman_2025 — per-host μ_TRGB read from data/catalogs/"
        "freedman_2025_table2.csv (Freedman 2025 Table 2 transcription; 10 "
        "unique hosts with weighted TRGB+JAGB distance moduli)."
    )
    tip_source_sensitivity: str = (
        "anand_2022 — Anand 2021 independent EDD reduction μ_TRGB (variant "
        "cross-check against the Freedman-paper primary)."
    )
    per_host_extinction_source: str = (
        "Freedman 2019 Table 1 per-host A_F814W values "
        "(data/catalogs/freedman_2019_table1.csv). Fallback: placeholder "
        "EBV_SFD for hosts not in Freedman Table 1 — sensitivity-only path."
    )
    hubble_flow_z_cuts: str = (
        "0.023 ≤ z_CMB ≤ 0.15 (Freedman 2019 §6.3 Supercal subsample). "
        "Applied to Pantheon+SH0ES non-calibrator SNe; N_flow ≈ 496. "
        "For the Pantheon+-only variant of H₀; CSP/SuperCal variants use "
        "Hoyt 2025 Eq. 15 reference values directly (see sn_system_*)."
    )
    sn_system_case_a: str = (
        "CSP-I (Freedman 2019 primary SN sample). Per-system reproduced H₀ "
        "computed via Hoyt 2025 Eq. 15 applied to our TRGB distances with "
        "the Hoyt 2025 Table 6/7 reference values. Variants: CSP-I, "
        "CSP-II, SuperCal, Pantheon+ all computed and reported."
    )
    sn_system_case_b: str = (
        "CSP-II (Freedman 2025 primary SN sample). Same 4-variant analysis "
        "as Case A. The Pantheon+ variant is expected to be +2 km/s/Mpc "
        "higher than CSP-II per Hoyt 2025 Section 4; this is documented "
        "but not used as the primary number."
    )
    sn_system_amendment: str = (
        "2026-04-24 amendment: primary Hubble-flow SN sample for each case "
        "is CCHP's own CSP-I/II rather than Pantheon+SH0ES. Pantheon+ "
        "calibration is known (Hoyt 2025 §4, 3.1σ significance) to bias H₀ "
        "upward by ≈+2 km/s/Mpc relative to CSP, because Pantheon+ μ_SH0ES "
        "values are Cepheid/SH0ES-anchored. Retaining Pantheon+ as the "
        "fourth variant directly demonstrates this +2 km/s/Mpc shift."
    )

    # Case A
    case_a_anchor: str = "LMC (Pietrzyński 2019 DEB); μ = 18.477 ± 0.026 (stat) ± 0.024 (sys)."
    case_a_d_local_mpc: float = 0.0496
    case_a_d_local_sigma_mpc: float = 0.0009
    case_a_sample_criteria: List[str] = field(
        default_factory=lambda: [
            "Hosts in Freedman 2019 ApJ 882, 34 Table 3 (15 unique TRGB hosts, "
            "18 SN Ia calibrators). Each Table 3 host is included with its "
            "Freedman-published μ_TRGB value.",
            "HST photometry from the Anand 2021 EDD reduction attached where "
            "available (10 hosts: NGC 1316, 1365, 1404, 1448, 3627, 4038, "
            "4424, 4526, 4536, 5643); remaining 5 hosts (M101, NGC 1309, 3021, "
            "3368, 3370, 5584) enter as 'photometry stubs' — published μ only.",
            "NGC 4258 is excluded from Case A (anchor-galaxy contamination; "
            "enforced by scripts/build_trgb_manifests.py CASE_A_EXCLUDED_HOSTS).",
        ]
    )
    case_a_primary_band: str = "F814W"
    case_a_extinction: str = (
        "Freedman 2019 Table 1 per-host A_F814W values (authoritative paper-"
        "tabulated extinction). Freedman's own SFD + CCM89 R_V=3.1 values."
    )
    case_a_metallicity: str = "Freedman 2020 F814W color slope, β = 0.20 (Rizzi 2007), fixed."
    case_a_reproduction_tolerance_mag: float = 0.8

    # Case B
    case_b_anchor: str = "NGC 4258 (Reid 2019 maser); μ = 29.397 ± 0.024 (stat) ± 0.022 (sys)."
    case_b_d_local_mpc: float = 7.58
    case_b_d_local_sigma_mpc: float = 0.08
    case_b_sample_criteria: List[str] = field(
        default_factory=lambda: [
            "Hosts in Freedman 2025 ApJ 985, 203 Table 2 (10 JWST-observed "
            "hosts, 11 SN Ia calibrators). Each is included with its "
            "published weighted μ_bar.",
            "JWST raw NIRCam photometry was NOT downloaded & reduced; public "
            "from MAST (GO-1995, 2875, 3055) but DOLPHOT-level re-reduction "
            "is out of scope. We use Freedman 2025's published μ_TRGB values "
            "directly (faithful 'reproduction of published' posture).",
            "Edge-detection sensitivity variants operate on HST-era photometry "
            "where overlap exists (NGC 1365, 1448, 4038, 4424, 4536, 5643).",
        ]
    )
    case_b_primary_band: str = "F150W (reported in the paper); μ_TRGB values used are the weighted TRGB+JAGB μ_bar"
    case_b_extinction: str = (
        "Placeholder (JWST NIRCam per-host extinction infrastructure "
        "not in pipeline). Sensitivity-only path; primary reproduction uses "
        "Freedman 2025 Table 2 published μ_bar directly and therefore does "
        "not depend on our per-field extinction."
    )
    case_b_metallicity: str = (
        "Inherits M_TRGB_abs = -4.049 common zero point (Freedman 2025 §14.2: "
        "'F19 and F21 share a common TRGB absolute magnitude zero point')."
    )
    case_b_reproduction_tolerance_mag: float = 1.22

    # Edge detection
    edge_detection_primary: str = (
        "Published μ_TRGB (bypass). Edge detection runs on the raw photometry "
        "for sensitivity diagnostics only; the primary path reads μ_TRGB "
        "from the Freedman-paper table (tip_source)."
    )
    edge_detection_sensitivity: Tuple[str, ...] = (
        "Sobel kernel width 1.0",
        "Sobel kernel width 2.0 (Freedman published choice)",
        "Sobel kernel width 3.0",
        "Model-based (Makarov 2006-style broken power-law fit)",
        "Bayesian (Hatt 2017-style posterior over tip location)",
    )

    # MCMC
    mcmc_n_walkers: int = 32
    mcmc_n_steps: int = 10_000
    mcmc_n_burnin: int = 2_000
    mcmc_rhat_gate: float = 1.01
    mcmc_seed: int = 42

    # Prior boxes per parameter. In the freedman_fixed parametrization only
    # the H0 prior is actively sampled; the remaining parameters are held
    # at their prior 'mean' values.
    priors_freedman_2020: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "H0": {"lo": 55.0, "hi": 85.0},
            "M_TRGB": {"lo": -5.0, "hi": -3.5, "mean": -4.049, "sigma": 0.045},
            # Freedman 2019 §4: MT814_RGB = -4.049 ± 0.022 (stat) ± 0.039 (sys)
            "EBV": {"lo": -0.10, "hi": 0.30, "mean": 0.07, "sigma": 0.03},
            "beta": {"lo": -0.2, "hi": 0.6, "mean": 0.20, "sigma": 0.1},
            # β = 0.20 Rizzi 2007, held fixed in freedman_fixed mode.
        }
    )
    priors_freedman_2024: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "H0": {"lo": 55.0, "hi": 85.0},
            # Freedman 2025 §14.2: "F19 and F21 share a common TRGB absolute
            # magnitude zero point, M814W = -4.049 mag".
            "M_TRGB": {"lo": -4.20, "hi": -3.90, "mean": -4.049, "sigma": 0.05},
            "EBV": {"lo": -0.10, "hi": 0.30, "mean": 0.07, "sigma": 0.03},
            "beta": {"lo": -0.2, "hi": 0.6, "mean": 0.20, "sigma": 0.1},
        }
    )

    # Compute backend
    compute_backend: str = "auto"

    # Sensitivity variants
    extinction_sensitivity_variants: Tuple[str, ...] = ("green2019_3d",)
    metallicity_sensitivity_variants: Tuple[str, ...] = ("rizzi2007", "jang_lee_2017")

    def render_markdown(self) -> str:
        ts = self.generated or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        lines: List[str] = []
        lines.append("# TRGB Comparative Analysis — Preregistration Stage 1\n\n")
        lines.append(f"*Generated: {ts}*\n\n")
        lines.append(
            "**Stage 1 is frozen BEFORE any data loading.** Every methodological choice "
            "documented below is committed to the repository in this form.\n\n"
        )

        lines.append("## Framework inputs (runtime-computed, not cached)\n\n")
        lines.append(f"- γ/H at z=0: {self.gamma_over_H_source}\n")
        lines.append(f"- C(G): {self.clustering_coefficient_source}\n")
        lines.append(f"- d_CMB: {self.d_cmb_source}\n")
        lines.append(f"- Perturbative-regime threshold: d_local < {self.perturbative_threshold_mpc} Mpc flags breakdown.\n\n")

        lines.append("## Reproduction configuration\n\n")
        lines.append(f"- Parametrization (primary): {self.parametrization}\n")
        lines.append(f"- Parametrization (sensitivity variant): {self.parametrization_sensitivity}\n")
        lines.append(f"- Case A tip source: {self.tip_source_case_a}\n")
        lines.append(f"- Case B tip source: {self.tip_source_case_b}\n")
        lines.append(f"- Tip source sensitivity variant: {self.tip_source_sensitivity}\n")
        lines.append(f"- Per-host extinction: {self.per_host_extinction_source}\n")
        lines.append(f"- Hubble-flow z cuts: {self.hubble_flow_z_cuts}\n")
        lines.append(f"- SN Ia system (Case A primary): {self.sn_system_case_a}\n")
        lines.append(f"- SN Ia system (Case B primary): {self.sn_system_case_b}\n\n")
        lines.append("### 2026-04-24 Amendment: primary SN sample\n\n")
        lines.append(f"{self.sn_system_amendment}\n\n")

        lines.append("## Case A — Freedman 2019/2020 (LMC anchor)\n\n")
        lines.append(f"- Anchor: {self.case_a_anchor}\n")
        lines.append(
            f"- d_local: {self.case_a_d_local_mpc} Mpc ± {self.case_a_d_local_sigma_mpc} Mpc\n"
        )
        lines.append(f"- Primary band: {self.case_a_primary_band}\n")
        lines.append(f"- Extinction: {self.case_a_extinction}\n")
        lines.append(f"- Metallicity: {self.case_a_metallicity}\n")
        lines.append(
            f"- Reproduction tolerance target: ±{self.case_a_reproduction_tolerance_mag} km/s/Mpc (stat).\n"
        )
        lines.append("- Sample selection criteria:\n")
        for line in self.case_a_sample_criteria:
            lines.append(f"  - {line}\n")
        lines.append("\n")

        lines.append("## Case B — Freedman 2024/2025 (NGC 4258 anchor)\n\n")
        lines.append(f"- Anchor: {self.case_b_anchor}\n")
        lines.append(
            f"- d_local: {self.case_b_d_local_mpc} Mpc ± {self.case_b_d_local_sigma_mpc} Mpc\n"
        )
        lines.append(f"- Primary band: {self.case_b_primary_band}\n")
        lines.append(f"- Extinction: {self.case_b_extinction}\n")
        lines.append(f"- Metallicity: {self.case_b_metallicity}\n")
        lines.append(
            f"- Reproduction tolerance target: ±{self.case_b_reproduction_tolerance_mag} km/s/Mpc (stat).\n"
        )
        lines.append("- Sample selection criteria:\n")
        for line in self.case_b_sample_criteria:
            lines.append(f"  - {line}\n")
        lines.append("\n")

        lines.append("## Edge detection\n\n")
        lines.append(f"- Primary: {self.edge_detection_primary}\n")
        lines.append("- Sensitivity variants:\n")
        for v in self.edge_detection_sensitivity:
            lines.append(f"  - {v}\n")
        lines.append("\n")

        lines.append("## MCMC\n\n")
        lines.append(
            f"- Settings: {self.mcmc_n_walkers} walkers × {self.mcmc_n_steps} steps "
            f"({self.mcmc_n_burnin} burn-in), Gelman-Rubin R̂ gate {self.mcmc_rhat_gate}, "
            f"seed {self.mcmc_seed}.\n\n"
        )
        lines.append("### Prior boxes — Freedman 2020\n\n")
        lines.append("| Parameter | lo | hi | mean | sigma |\n| --- | --- | --- | --- | --- |\n")
        for p, d in self.priors_freedman_2020.items():
            lines.append(
                f"| {p} | {d.get('lo')} | {d.get('hi')} | {d.get('mean','')} | {d.get('sigma','')} |\n"
            )
        lines.append("\n### Prior boxes — Freedman 2024\n\n")
        lines.append("| Parameter | lo | hi | mean | sigma |\n| --- | --- | --- | --- | --- |\n")
        for p, d in self.priors_freedman_2024.items():
            lines.append(
                f"| {p} | {d.get('lo')} | {d.get('hi')} | {d.get('mean','')} | {d.get('sigma','')} |\n"
            )
        lines.append("\n")

        lines.append("## Compute backend\n\n")
        lines.append(f"- Preference: `{self.compute_backend}` (MLX when available; NumPy otherwise).\n\n")

        lines.append("## Sensitivity variants\n\n")
        lines.append(
            "Sensitivity variants run in a SEPARATE analysis stage; they never feed "
            "into the primary reproduction numbers.\n\n"
        )
        for v in self.extinction_sensitivity_variants:
            lines.append(f"- Extinction: {v}\n")
        for v in self.metallicity_sensitivity_variants:
            lines.append(f"- Metallicity: {v}\n")
        lines.append("\n")

        return "".join(lines)


def generate_stage1(
    docs_dir: Path | str = DOCS_DIR_DEFAULT,
    config: Optional[Stage1Config] = None,
) -> Path:
    """Write the Stage 1 preregistration document."""
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    cfg = config or Stage1Config()
    if not cfg.generated:
        cfg.generated = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    path = docs_dir / STAGE1_FILENAME
    path.write_text(cfg.render_markdown())
    return path


# ---------------------------------------------------------------------------
# Stage 2 — post-data
# ---------------------------------------------------------------------------


@dataclass
class Stage2Config:
    generated: str = ""
    case_a_hosts: List[str] = field(default_factory=list)
    case_b_hosts: List[str] = field(default_factory=list)
    archive_sha256: Dict[str, str] = field(default_factory=dict)
    loader_git_commit: str = ""
    completeness_cut_mag: float = 1.0  # ≥1 mag below tip per preregistration
    rgb_star_count_cut: int = 400

    def render_markdown(self, stage1_sha256: str) -> str:
        ts = self.generated or datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
        lines: List[str] = []
        lines.append("# TRGB Comparative Analysis — Preregistration Stage 2\n\n")
        lines.append(f"*Generated: {ts}*\n\n")
        lines.append(f"- Stage 1 document SHA-256: `{stage1_sha256}`\n")
        lines.append(f"- Loader git commit: `{self.loader_git_commit}`\n")
        lines.append(f"- Completeness cut: ≥{self.completeness_cut_mag} mag below tip\n")
        lines.append(f"- RGB star count cut: ≥{self.rgb_star_count_cut}\n\n")

        lines.append("## Case A — resolved SN Ia host list\n\n")
        if self.case_a_hosts:
            for h in self.case_a_hosts:
                lines.append(f"- {h}\n")
        else:
            lines.append("_(no hosts resolved; Stage 1 selection rules returned empty set)_\n")
        lines.append("\n## Case B — resolved SN Ia host list\n\n")
        if self.case_b_hosts:
            for h in self.case_b_hosts:
                lines.append(f"- {h}\n")
        else:
            lines.append("_(no hosts resolved)_\n")
        lines.append("\n## Archive SHA-256 checksums\n\n")
        if self.archive_sha256:
            for name, sha in self.archive_sha256.items():
                lines.append(f"- `{name}`: `{sha}`\n")
        else:
            lines.append("_(no archives present — Stage 2 run before data download)_\n")
        lines.append("\n")
        return "".join(lines)


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def generate_stage2(
    docs_dir: Path | str = DOCS_DIR_DEFAULT,
    stage1_path: Optional[Path] = None,
    config: Optional[Stage2Config] = None,
) -> Path:
    """Write the Stage 2 preregistration document.

    The Stage 1 document SHA-256 is recorded in Stage 2 so downstream can
    verify Stage 1 has not changed between stages.
    """
    docs_dir = Path(docs_dir)
    docs_dir.mkdir(parents=True, exist_ok=True)
    stage1_path = stage1_path or (docs_dir / STAGE1_FILENAME)
    if not stage1_path.exists():
        raise FileNotFoundError(
            f"Stage 1 preregistration not found at {stage1_path}; "
            "run generate_stage1 first."
        )
    stage1_sha = _sha256_of_file(stage1_path)
    cfg = config or Stage2Config()
    if not cfg.generated:
        cfg.generated = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    path = docs_dir / STAGE2_FILENAME
    path.write_text(cfg.render_markdown(stage1_sha))
    return path


def verify_preregistration_exists(docs_dir: Path | str = DOCS_DIR_DEFAULT) -> Tuple[Path, Path]:
    """Raise FileNotFoundError unless both stage documents exist."""
    docs_dir = Path(docs_dir)
    stage1 = docs_dir / STAGE1_FILENAME
    stage2 = docs_dir / STAGE2_FILENAME
    if not stage1.exists():
        raise FileNotFoundError(
            f"Stage 1 preregistration missing at {stage1}. "
            "Run `python main.py --trgb-comparative-preregister-stage1` first."
        )
    if not stage2.exists():
        raise FileNotFoundError(
            f"Stage 2 preregistration missing at {stage2}. "
            "Run `python main.py --trgb-comparative-preregister-stage2` after data loading."
        )
    return stage1, stage2


__all__ = [
    "Stage1Config",
    "Stage2Config",
    "generate_stage1",
    "generate_stage2",
    "verify_preregistration_exists",
]
