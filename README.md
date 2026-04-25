# H-ΛCDM Analysis Framework

A research codebase implementing the H-ΛCDM cosmology — Holographic Λ-Cold Dark Matter — and testing its parameter-free predictions against observational data. The framework emerges from holographic entropy bounds, quantum measurement theory, and a runtime-computed information processing rate γ(z); each pipeline tests a distinct prediction against an independent observational dataset.

Two analysis pipelines are currently implemented:

1. **BAO pipeline** — tests the H-ΛCDM enhanced sound horizon `r_s = 150.71 Mpc` (a 2.18% enhancement from quantum anti-viscosity at recombination) against ten independent BAO surveys.
2. **TRGB comparative pipeline** — tests the holographic projection formula `H_local / H_CMB = 1 + (γ/H) · ln(d_CMB / d_local)` as a forward prediction at the LMC and NGC 4258 anchors, against twelve MCMC reproductions of the CCHP (Carnegie–Chicago Hubble Program) Tip of the Red Giant Branch H₀ measurements (Freedman 2019/2020 HST and 2024/2025 JWST).

Both pipelines implement two-stage preregistration before any data is touched, run unconditional reporting (results print whether or not they confirm predictions), and produce publication-ready outputs.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager
- ~25 GB free disk space (raw photometry archives for the TRGB pipeline)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/bryceweiner/h-lcdm.git
cd h-lcdm
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

---

## BAO Pipeline

Tests the H-ΛCDM theoretical prediction for the baryon acoustic oscillation sound horizon against observational data from ten independent BAO surveys.

### Running the BAO Pipeline

#### Basic Run

Run with default settings (all available datasets):

```bash
python main.py --bao
```

#### With Validation

Run with basic statistical validation:

```bash
python main.py --bao validate
```

#### Extended Validation

Run with extended validation (bootstrap, jackknife, Monte Carlo, cross-validation):

```bash
python main.py --bao validate extended
```

#### Select Specific Datasets

```bash
python main.py --bao --bao-datasets boss_dr12 desi_y1 eboss
```

Available datasets:
- `boss_dr12` — BOSS DR12 consensus measurements
- `desi_y1` — DESI Year 1 measurements
- `eboss` — eBOSS DR16 measurements
- `sixdfgs` — 6dF Galaxy Survey measurements
- `wigglez` — WiggleZ Dark Energy Survey measurements
- `sdss_mgs` — SDSS Main Galaxy Sample measurements
- `sdss_dr7` — SDSS DR7 measurements
- `2dfgrs` — 2dF Galaxy Redshift Survey measurements
- `des_y1` — DES Year 1 measurements
- `des_y3` — DES Year 3 measurements

#### Custom Output Directory

```bash
python main.py --bao --output-dir ./my_results
```

#### Quiet Mode

```bash
python main.py --bao --quiet
```

### What the BAO Pipeline Does

1. **Loads BAO Data** from 10 independent surveys.
2. **Calculates Theoretical Predictions** using:
   - Enhanced sound horizon `r_s = 150.71 Mpc` (2.18% enhancement)
   - Parameter-free predictions (`α = −5.7`, `γ = 1.707 × 10⁻¹⁶ s⁻¹`)
3. **Performs Statistical Validation**:
   - χ² goodness-of-fit tests
   - Model comparison (H-ΛCDM vs ΛCDM) using BIC, AIC, and Bayes Factor
   - Bootstrap resampling (50,000 iterations)
   - Jackknife resampling
   - Leave-one-out cross-validation
   - Monte Carlo validation (50,000 simulations)
4. **Generates Reports** as comprehensive markdown.

### BAO Output

Results land in the `./results` directory (or your `--output-dir` path):

- **`results/reports/bao_analysis_report.md`** — comprehensive report (per-dataset validation, BIC/AIC/Bayes-factor comparisons, validation results, DESI Year 3 forward predictions).
- **`results/figures/`** — visualization figures (when `--generate-figures` is set).
- **`execution_summary.json`** — execution metadata and configuration.

### BAO Example Commands

Full analysis with extended validation:
```bash
python main.py --bao validate extended --generate-figures
```

Test on a subset of datasets:
```bash
python main.py --bao validate --bao-datasets boss_dr12 desi_y1 eboss
```

Quick test run:
```bash
python main.py --bao --quiet --output-dir ./test_results
```

---

## TRGB Comparative Pipeline

Paired forward-prediction test of the holographic projection formula
`H_local / H_CMB = 1 + (γ/H) · ln(d_CMB / d_local)` against twelve
MCMC reproductions of the CCHP TRGB H₀ measurements. The formula is
parameter-free (γ/H is computed at runtime from
`HLCDMCosmology.gamma_at_redshift(0.0) / HLCDM_PARAMS.get_hubble_at_redshift(0.0)`;
d_CMB is the Planck 2018 comoving distance to last scattering;
d_local is the geometric anchor distance for each case). The
pipeline implements two-stage preregistration, regenerates both
stage documents at the start of every run, and produces both
diagnostic and manuscript-quality figures.

### Running the TRGB Pipeline

#### Basic Run

```bash
python main.py --trgb-comparative
```

This runs the full analysis (~22 minutes on a single Apple-Silicon machine):
1. Generates pre-registration Stage 1 and Stage 2 documents to `trgb_data/prereg/`.
2. Loads tracked photometry catalogs from `trgb_data/catalogs/` and raw archives from `trgb_data/downloaded_data/`.
3. Runs both Freedman case reproductions (LMC anchor and NGC 4258 anchor) under the published methodologies.
4. Runs the 12-chain reproduction matrix (3 cases × 4 photometric SN systems) via emcee MCMC.
5. Computes framework forward predictions for each anchor via Monte Carlo propagation.
6. Renders manuscript figures to `figures/manuscript/`, diagnostic figures and reports to `results/`.

#### Pre-registration Subcommands

For incremental workflows, the two preregistration stages can be generated independently:

```bash
python main.py --trgb-comparative-preregister-stage1
python main.py --trgb-comparative-load-data
python main.py --trgb-comparative-preregister-stage2
```

The full pipeline regenerates both stage documents at the start of every run, so these standalone flags are optional.

#### Compute Backend

```bash
python main.py --trgb-comparative --compute-backend auto    # default; selects MLX on Apple Silicon
python main.py --trgb-comparative --compute-backend numpy   # force CPU/NumPy reference path
```

### Anchor Cases

| Case | Anchor | d_local (Mpc) | Source |
|------|--------|---------------|--------|
| Case A — Freedman 2019/2020 (HST)  | LMC          | 0.0496 ± 0.0009 | Pietrzyński 2019 DEB        |
| Case B — Freedman 2024/2025 (JWST) | NGC 4258     | 7.58 ± 0.08      | Reid et al. 2019 maser     |
| Case B — JWST-only sensitivity     | NGC 4258     | 7.58 ± 0.08      | F2025 Table 2 11-SN subset |

### Photometric SN Systems

Each case is reproduced against four SN photometric systems independently, yielding a 3 × 4 = 12-chain matrix:

- `csp_i` — Carnegie Supernova Project I
- `csp_ii` — Carnegie Supernova Project II
- `supercal` — SuperCal compilation
- `pantheon_plus` — Pantheon+SH0ES with full covariance

CSP-I/II chains use the Uddin 2023 8-parameter SNooPy likelihood; SuperCal and Pantheon+ chains use a simple 1-parameter likelihood. Convergence is gated at Gelman-Rubin R̂ < 1.01.

### What the TRGB Pipeline Does

1. **Two-stage preregistration** (Stage 1 freezes methodology before any data load; Stage 2 resolves selection rules against loaded data and records SHA-256 checksums of every catalog).
2. **Loads photometry**: Freedman 2019 Table 1 + Table 3, Freedman 2025 Table 2 + Table 3, Hoyt 2025 SN calibration tables, Pantheon+SH0ES with covariance, Pantheon 2018, Anand 2022 EDD catalog, Pietrzyński 2019 LMC distance, Reid 2019 NGC 4258 distance, Uddin 2023 H0CSP samples.
3. **Reproduces both Freedman papers** under their published methodologies (extinction, metallicity, tip-detection treatments matched per paper).
4. **Runs the 12-chain MCMC matrix** (32 walkers × 10 000 steps × 2 000 burn-in by default, 64 × 20 000 × 5 000 in production); R̂ < 1.01 convergence gate.
5. **Runs the Uddin 2023 positive-control test** (target H₀ = 70.242 ± 0.724) — verifies the SNooPy likelihood implementation.
6. **Computes framework forward predictions** via Monte Carlo propagation of (γ/H, d_local, H_CMB) draws through the linear-form projection formula. Reference predictions:
   - LMC anchor: H_local ≈ 70.40 km/s/Mpc
   - NGC 4258 anchor: H_local ≈ 69.20 km/s/Mpc
7. **Generates manuscript figures** (`figures/manuscript/`) plus diagnostic figures, the 12-chain reproduction matrix CSV, and a markdown report (`results/`).

### TRGB Output

Tracked, persistent locations (committed to git):

- **`trgb_data/catalogs/`** — published table transcriptions (5 CSVs).
- **`trgb_data/prereg/`** — Stage 1 and Stage 2 preregistration documents (regenerated each run).
- **`trgb_data/chains/`** — MCMC `.npz` outputs (23 files) for the 12 full-calibrator chains, the 8 legacy intersection chains, the Uddin positive-control chain, and the per-case `freedman_*_pantheon_plus.npz`.
- **`figures/manuscript/`** — four publication-ready figures (PDF + PNG):
  - `fig1_framework_vs_chains` — headline: 12 chain medians grouped by case, color-coded by SN system, against framework prediction bands and SH0ES / Planck reference bands.
  - `fig2_cross_anchor_shift` — per-system Δ(LMC → NGC 4258) shift vs. framework prediction.
  - `fig3_tension_matrix` — 3 × 4 σ-tension heatmap diverging at 1σ.
  - `fig4_prediction_vs_distance` — continuous H_local(d_local) curve with reproduction chains overlaid.

Run-time outputs (regenerated each run; **not** tracked):

- **`trgb_data/downloaded_data/`** — raw HST/JWST photometry archives, Anand 2022 EDD reduction, Uddin 2023 H0CSP samples, LMC Hatt 2018 halo photometry. Re-downloadable from MAST and EDD.
- **`results/12_chain_matrix.csv`** — 12-chain reproduction matrix (case, system, MCMC posterior, framework prediction, σ-tension, Freedman target, within-tolerance flag).
- **`results/trgb_comparative/reports/trgb_comparative_analysis_report.md`** — comprehensive markdown report (per-case reproduction vs. published, framework predictions, cross-case shift, tension analysis, caveats, manuscript-figure index).
- **`results/trgb_comparative/reports/data_tables.tex`** — publication-ready LaTeX tables for all astronomical data used in the analysis.
- **`results/trgb_comparative/reports/data_acquisition_narrative.md`** — peer-review data-acquisition narrative.
- **`results/trgb_comparative/figures/`** — diagnostic figures (CMD edges, posteriors, predictions vs. observed, hubble flow).

### TRGB Reproduction Notebook

A self-contained Jupyter notebook reproduces the full 12-chain matrix + Uddin positive control + framework predictions from raw data:

```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/trgb_comparative_reproduction.ipynb
```

Runs cleanly from a fresh clone (catalogs and chains are tracked; the notebook builds chain plans, runs MCMC, and computes framework predictions end-to-end).

### TRGB Example Commands

Full analysis:
```bash
python main.py --trgb-comparative
```

Pre-registration only (incremental workflow):
```bash
python main.py --trgb-comparative-preregister-stage1
```

Force NumPy backend (for backend-parity verification):
```bash
python main.py --trgb-comparative --compute-backend numpy
```

---

## ML-derived Recommendations (CMB residual γ test)

```bash
python main.py --recommendation 1
```

Uses ACT DR6 TT data minus Planck ΛCDM TT to search for `γ = H/π²` modulation in `ℓ = 500–2000`. Optional validation tiers mirror the other pipelines: `--recommendation 1 validate` or `--recommendation 1 validate extended`. Recommendation IDs are integer-coded; current analysis is `1`.

---

## Help

For complete command-line options:

```bash
python main.py --help
```

## Citation

If you use this codebase, please cite the associated paper:

```bibtex
@article{Weiner2026HLCDMFramework,
  title={H-ΛCDM Analysis Framework},
  author={Weiner, Bryce},
  journal={IPI Letters},
  year={2026},
  status={submitted}
}
```

A separate manuscript covering the TRGB comparative analysis is in preparation. See `CITATION.cff` for complete citation information.

## License

MIT License — see LICENSE file for details.
