# BAO Pipeline - H-ΛCDM Analysis Framework

This repository contains the BAO (Baryon Acoustic Oscillation) analysis pipeline for testing H-ΛCDM theoretical predictions against observational data from multiple BAO surveys.

## Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone <repository-url>
cd h-lcdm-codebase
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

### Running the BAO Pipeline

#### Basic Run

Run the BAO pipeline with default settings (all available datasets):

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

Run analysis on specific BAO datasets:

```bash
python main.py --bao --bao-datasets boss_dr12 desi_y1 eboss
```

Available datasets:
- `boss_dr12` - BOSS DR12 consensus measurements
- `desi_y1` - DESI Year 1 measurements
- `eboss` - eBOSS DR16 measurements
- `sixdfgs` - 6dF Galaxy Survey measurements
- `wigglez` - WiggleZ Dark Energy Survey measurements
- `sdss_mgs` - SDSS Main Galaxy Sample measurements
- `sdss_dr7` - SDSS DR7 measurements
- `2dfgrs` - 2dF Galaxy Redshift Survey measurements
- `des_y1` - DES Year 1 measurements
- `des_y3` - DES Year 3 measurements

#### Custom Output Directory

Specify a custom output directory:

```bash
python main.py --bao --output-dir ./my_results
```

#### Quiet Mode

Run without progress messages:

```bash
python main.py --bao --quiet
```

## Output

The pipeline generates results in the `./results` directory (or your specified output directory):

- **`results/reports/bao_analysis_report.md`** - Comprehensive analysis report with:
  - Dataset-by-dataset validation results
  - Model comparison statistics (BIC, AIC, Bayes Factor)
  - Statistical validation results (bootstrap, jackknife, Monte Carlo, cross-validation)
  - Forward predictions for DESI Year 3

- **`results/figures/`** - Visualization figures (if `--generate-figures` is used)

- **`execution_summary.json`** - Execution metadata and configuration

## What the Pipeline Does

1. **Loads BAO Data**: Retrieves observational data from 10 independent BAO surveys
2. **Calculates Theoretical Predictions**: Computes H-ΛCDM predictions using:
   - Enhanced sound horizon: r_s = 150.71 Mpc (2.18% enhancement)
   - Parameter-free predictions (α = -5.7, γ = 1.707 × 10⁻¹⁶ s⁻¹)
3. **Performs Statistical Validation**:
   - χ² goodness-of-fit tests
   - Model comparison (H-ΛCDM vs ΛCDM) using BIC, AIC, and Bayes Factor
   - Bootstrap resampling (50,000 iterations)
   - Jackknife resampling
   - Leave-one-out cross-validation
   - Monte Carlo validation (50,000 simulations)
4. **Generates Reports**: Creates comprehensive markdown reports with all results

## Example Commands

### Full analysis with extended validation:
```bash
python main.py --bao validate extended --generate-figures
```

### Test on subset of datasets:
```bash
python main.py --bao validate --bao-datasets boss_dr12 desi_y1 eboss
```

### Quick test run:
```bash
python main.py --bao --quiet --output-dir ./test_results
```

### ML-derived recommendations (CMB residual γ test):
```bash
python main.py --recommendation 1
```
- Uses ACT DR6 TT data minus Planck ΛCDM TT to search for γ = H/π² modulation in ℓ = 500–2000.
- Optional validation tiers mirror other pipelines: `--recommendation 1 validate` or `--recommendation 1 validate extended`.
- Recommendation IDs are integer-coded; current analysis is `1`.

## Help

For complete command-line options:

```bash
python main.py --help
```

## Citation

If you use this codebase, please cite the associated paper:

```bibtex
@article{Weiner2025QuantumAntiViscosity,
  title={Quantum Anti-Viscosity at Cosmic Recombination: Parameter-Free Prediction of Baryon Acoustic Oscillations from Holographic Information Theory},
  author={Weiner, Bryce},
  journal={The Astrophysical Journal},
  year={2025},
  status={submitted}
}
```

See `CITATION.cff` for complete citation information.

## License

MIT License - see LICENSE file for details.

