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

    # Case A
    case_a_anchor: str = "LMC (Pietrzyński 2019 DEB); μ = 18.477 ± 0.026 (stat) ± 0.024 (sys)."
    case_a_d_local_mpc: float = 0.0496
    case_a_d_local_sigma_mpc: float = 0.0009
    case_a_sample_criteria: List[str] = field(
        default_factory=lambda: [
            "Hosts in Freedman 2019 ApJ 882, 34 Table 1 (N=18 TRGB-calibrated SN Ia hosts).",
            "Plus Anand 2022 catalog hosts meeting ALL of:",
            "  (a) LMC-calibrated TRGB distance available",
            "  (b) HST F814W photometry complete ≥1.0 mag below the tip",
            "  (c) RGB star count ≥ 400 in the CMD selection box",
            "  (d) at least one spec-confirmed SN Ia with a Pantheon+ entry",
        ]
    )
    case_a_primary_band: str = "F814W"
    case_a_extinction: str = "SFD + CCM89 (Schlegel 1998 + Cardelli 1989)"
    case_a_metallicity: str = "Freedman 2020 F814W color slope"
    case_a_reproduction_tolerance_mag: float = 0.8

    # Case B
    case_b_anchor: str = "NGC 4258 (Reid 2019 maser); μ = 29.397 ± 0.024 (stat) ± 0.022 (sys)."
    case_b_d_local_mpc: float = 7.58
    case_b_d_local_sigma_mpc: float = 0.08
    case_b_sample_criteria: List[str] = field(
        default_factory=lambda: [
            "Hosts in CCHP 2024 JWST sample (Freedman et al. 2024, N=10 JWST NIRCam).",
            "Plus HST archival hosts with NGC 4258-calibrated TRGB distances meeting:",
            "  (a) NGC 4258-anchored TRGB distance available",
            "  (b) photometry complete ≥1.0 mag below the tip",
            "  (c) RGB star count ≥ 400",
            "  (d) at least one spec-confirmed SN Ia with a Pantheon+ entry",
        ]
    )
    case_b_primary_band: str = "F150W"
    case_b_extinction: str = "SFD + CCM89 extended to JWST NIR (CCHP 2024 treatment)"
    case_b_metallicity: str = "CCHP 2024 NIR color slope (dM/d(F090W-F150W))"
    case_b_reproduction_tolerance_mag: float = 1.22

    # Edge detection
    edge_detection_primary: str = "Sobel kernel width = 2.0 bins (Freedman published choice)"
    edge_detection_sensitivity: Tuple[str, ...] = (
        "Sobel kernel width 1.0",
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

    # Prior boxes per free parameter (H0, M_TRGB, E(B-V), beta)
    priors_freedman_2020: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "H0": {"lo": 55.0, "hi": 85.0},
            "M_TRGB": {"lo": -5.0, "hi": -3.5, "mean": -4.047, "sigma": 0.045},
            "EBV": {"lo": -0.10, "hi": 0.30, "mean": 0.07, "sigma": 0.03},
            "beta": {"lo": -0.2, "hi": 0.6, "mean": 0.2, "sigma": 0.1},
        }
    )
    priors_freedman_2024: Dict[str, Dict[str, float]] = field(
        default_factory=lambda: {
            "H0": {"lo": 55.0, "hi": 85.0},
            "M_TRGB": {"lo": -5.5, "hi": -4.0, "mean": -4.362, "sigma": 0.05},
            "EBV": {"lo": -0.10, "hi": 0.30, "mean": 0.07, "sigma": 0.03},
            "beta": {"lo": -0.2, "hi": 0.6, "mean": 0.08, "sigma": 0.1},
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
