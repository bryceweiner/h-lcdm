# ML Architecture Implementation Plan

## Overview

Implement complete 5-stage ML architecture from ml-design.md for cross-survey cosmological pattern detection. **Critical approach**: Use ALL existing H-ΛCDM data (BAO, FRB, Lyman-alpha, voids, CMB, JWST) + new galaxy catalogs as blind test - ML will detect patterns without knowing which data contains H-ΛCDM signals.

## Data Infrastructure

### 1. Consolidate All Existing Data Sources

**File:** `h-lcdm-codebase/data/loader.py`

Existing data to include in ML training (blind test with known H-ΛCDM signals):

- **CMB data**: ACT DR6, Planck 2018, SPT-3G E-mode (contains phase transitions)
- **BAO data**: BOSS DR12, DESI, eBOSS, 6dFGS, WiggleZ (contains enhanced sound horizon r_s=150.71 Mpc)
- **Void catalogs**: Douglass, Clampitt & Jain, ZOBOV, VIDE (contains E8×E8 alignments)
- **FRB data**: FRB catalog (contains Little Bang timing signatures)
- **Lyman-alpha data**: Lyman-alpha forest (contains phase transitions)
- **JWST data**: Early galaxy catalogs (contains anti-viscosity signatures)

Verify all loaders functional, extract features for ML.

### 2. Deduplicate Void Catalogs

Use existing "voids_deduplicated.pkl" or execute the requisite parts of the existing API in order to generate it from the --void pipeline.

There's no need to write new code to do this since the software package already does it. Follow exisitng patterns.

Extract features from existing known columns.

### 3. Add New Galaxy Catalogs

**File:** `h-lcdm-codebase/data/loader.py`

Add methods using astroquery/astropy:

- `load_sdss_galaxy_catalog()` - SDSS DR16 spectroscopic + photometric via astroquery.sdss
- `load_desi_galaxy_catalog()` - DESI spectroscopic redshifts via astroquery (if available)
- `load_legacy_survey_imaging()` - DESI Legacy Survey imaging via astroquery
- `load_euclid_galaxy_data()` - Euclid survey via astroquery (if available, otherwise skip)

Extract features:

- Morphological: Petrosian radius, concentration, asymmetry, Sérsic index
- Clustering: nearest neighbor distances, correlation functions
- Spectroscopic: redshift, emission/absorption line strengths
- Photometric: u-g, g-r, r-i, i-z colors, absolute magnitudes

### 4. Create Unified Data Manifest

**New file:** `h-lcdm-codebase/data/data_manifest.py`

Track all data sources for ML (BLINDED to H-ΛCDM signal presence):

```python
class DataManifest:
    # Data types: CMB, BAO, void, galaxy, FRB, lyman_alpha, jwst
    # Survey provenance and systematic effects
    # Feature extraction metadata
    # Quality flags
    # H-ΛCDM signal presence - KEPT HIDDEN FROM ML ALGORITHMS
```

### 5. Create Mock Dataset Generator (Validation Only)

**New file:** `h-lcdm-codebase/data/mock_generator.py`

Mock datasets for null hypothesis testing ONLY:

- `generate_mock_cmb_maps()` - Gaussian random fields matching power spectrum
- `generate_mock_bao_measurements()` - Random BAO without H-ΛCDM enhancement
- `generate_mock_void_catalog()` - Random void distributions without E8×E8 alignment
- `generate_mock_galaxy_catalog()` - Match luminosity function, clustering
- `generate_mock_frb_catalog()` - Random FRB timing without information saturation
- `generate_mock_lyman_alpha()` - Random optical depth without phase transitions
- `generate_mock_jwst_catalog()` - Random early galaxies without anti-viscosity

Match statistical properties of real data, NOT hydrodynamic simulations.

### 6. Feature Extractors for All Modalities

**New directory:** `h-lcdm-codebase/data/feature_extractors/`

Separate files for each data type:

- `cmb_features.py`: Harmonic coefficients, power spectrum, bispectrum, topological features
- `bao_features.py`: D_M/r_d ratios, redshift dependence, correlation functions
- `void_features.py`: Size, ellipticity, density contrast, alignment angles, clustering
- `galaxy_features.py`: Morphology, colors, clustering, redshift evolution
- `frb_features.py`: Timing intervals, dispersion measures, redshift distribution
- `lyman_alpha_features.py`: Optical depth evolution, flux statistics, correlation functions
- `jwst_features.py`: High-z galaxy properties, mass estimates, formation signatures

## Stage 1: Self-Supervised Feature Learning

### 7. Contrastive Learning Framework

**New file:** `h-lcdm-codebase/pipeline/ml/ssl_encoder.py`

SimCLR-based contrastive learning:

```python
class ContrastiveLearner:
    # Data augmentation per modality
    # Temperature-scaled InfoNCE loss
    # Momentum encoder for stable training
    # Multi-GPU training support
```

Augmentation strategies:

- CMB: rotations, harmonic space transformations, masking variations
- BAO: redshift binning variations, covariance perturbations
- Voids: scaling, ellipticity perturbations, position jittering
- Galaxies: photometric noise, morphology variations, redshift uncertainties
- FRB: timing noise, dispersion measure variations
- Lyman-alpha: flux noise, continuum fitting variations
- JWST: photometric uncertainties, selection function variations

### 8. Modality-Specific Encoders

**New directory:** `h-lcdm-codebase/pipeline/ml/encoders/`

Seven encoder architectures:

- `cmb_encoder.py`: CNN for harmonic/pixel space, handles masking
- `bao_encoder.py`: MLP for distance measurements, redshift evolution
- `void_encoder.py`: GNN for void network topology + MLP for properties
- `galaxy_encoder.py`: ResNet for imaging + MLP for spectroscopy
- `frb_encoder.py`: Temporal CNN for timing sequences
- `lyman_alpha_encoder.py`: 1D CNN for flux spectra
- `jwst_encoder.py`: CNN for imaging + MLP for photometry

Each encoder outputs fixed-dimension embeddings for multimodal fusion.

### 9. Multimodal Integration

**New file:** `h-lcdm-codebase/pipeline/ml/multimodal_fusion.py`

```python
class MultimodalFusion:
    # Cross-attention between all 7 modalities
    # Late fusion in 512-dimensional shared latent space
    # Projection heads for contrastive learning
    # Handle missing modalities (not all data has all types)
```

## Stage 2: Domain Adaptation

### 10. Survey-Invariant Training

**New file:** `h-lcdm-codebase/pipeline/ml/domain_adapter.py`

```python
class DomainAdaptationTrainer:
    # Maximum Mean Discrepancy (MMD) loss between surveys
    # Domain adversarial training with gradient reversal
    # Survey ID conditioning with adversarial discriminator
    # Train on all surveys simultaneously
```

Handle survey-specific systematics:

- CMB: different beam sizes, foreground removal
- BAO: different selection functions, systematics
- Voids: different void finders (Douglass, ZOBOV, VIDE)
- Galaxies: different photometric systems, depth
- FRB: different telescope sensitivities
- Lyman-alpha: different quasar samples
- JWST: different NIRCam filters

### 11. Cross-Survey Validation

**Add to:** `h-lcdm-codebase/pipeline/ml/ml_pipeline.py`

- Hold-out one survey type at a time for validation
- Measure feature distribution alignment (KL divergence, Wasserstein distance)
- Quantify survey-specific systematic errors
- Ensure learned features are survey-invariant

## Stage 3: Pattern Detection Ensemble

### 12. Anomaly Detection Methods

**New file:** `h-lcdm-codebase/pipeline/ml/anomaly_detectors.py`

Three independent methods operating on SSL encoder output:

```python
class IsolationForestDetector:
    # Sklearn IsolationForest on 512-dim latent space
    # Contamination parameter from expected anomaly rate

class HDBSCANDetector:
    # Density-based clustering
    # Outliers = points not in any cluster
    # Adaptive density thresholds

class VAEDetector:
    # Variational autoencoder
    # Reconstruction loss as anomaly score
    # KL divergence regularization
```

### 13. Ensemble Aggregation

**New file:** `h-lcdm-codebase/pipeline/ml/ensemble.py`

```python
class EnsembleDetector:
    # Weighted average of three anomaly scores
    # Weights learned via cross-validation
    # Consensus voting for high-confidence detections
    # Rank aggregation for final anomaly ranking
```

## Stage 4: Interpretability

### 14. LIME Implementation

**New file:** `h-lcdm-codebase/pipeline/ml/interpretability/lime_explainer.py`

```python
class LIMEExplainer:
    # Local linear approximation around detection
    # Feature importance for individual anomalies
    # Which modality contributed most?
    # Which features within modality?
```

### 15. SHAP Implementation

**New file:** `h-lcdm-codebase/pipeline/ml/interpretability/shap_explainer.py`

```python
class SHAPExplainer:
    # Shapley values for global feature importance
    # Tree SHAP for ensemble methods
    # Deep SHAP for neural encoders
    # Summary plots, dependence plots
```

### 16. Interpretability Reports

**Add to:** `h-lcdm-codebase/pipeline/common/reporting.py`

Generate markdown reports with:

- Top 10 most anomalous objects per modality
- LIME explanations for each detection
- SHAP global feature importance rankings
- Modality contribution breakdown
- Visualization of detected patterns
- Comparison with H-ΛCDM theoretical predictions (AFTER detection, not before)

## Stage 5: Statistical Validation

### 17. Cross-Survey Validation Framework

**New file:** `h-lcdm-codebase/pipeline/ml/validation/cross_survey_validator.py`

```python
class CrossSurveyValidator:
    # Multiple-source cross-validation (Geras & Sutton 2013)
    # Train on N-1 data types, test on held-out type
    # E.g., train on CMB+BAO+voids+galaxies+FRB+lyman, test on JWST
    # Rotate through all combinations
```

### 18. Bootstrap Stability Analysis

**New file:** `h-lcdm-codebase/pipeline/ml/validation/bootstrap_validator.py`

```python
class BootstrapValidator:
    # 1000+ bootstrap resamples
    # Track consistency of detected patterns
    # Only report patterns in ≥95% of bootstraps
    # Confidence intervals on anomaly scores
```

### 19. Null Hypothesis Testing

**New file:** `h-lcdm-codebase/pipeline/ml/validation/null_hypothesis_tester.py`

```python
class NullHypothesisTester:
    # Generate mocks using mock_generator.py
    # Apply identical pipeline to mocks
    # Calculate empirical p-values: P(mock as anomalous as real)
    # Bonferroni or FDR correction for multiple testing
    # Compare detection rates: real vs. mock
```

**Critical validation**: If ML detects H-ΛCDM signals in real data but NOT in mocks (which lack those signals), this validates the detection capability.

### 20. Blind Analysis Protocol

**New file:** `h-lcdm-codebase/pipeline/ml/validation/blind_protocol.py`

```python
class BlindAnalysisProtocol:
    # Pre-register methodology before seeing results
    # Document: model architecture, hyperparameters, validation
    # Blind final detections until all validation complete
    # Generate unblinding report comparing to H-ΛCDM predictions
```

### 21. Data Splitting and Stability

**New file:** `h-lcdm-codebase/pipeline/ml/validation/stability_validator.py`

```python
class StabilityValidator:
    # K-fold cross-validation within each survey
    # Stability across different random seeds
    # Model selection consistency
    # Uncertainty quantification via ensemble variance
```

## Integration and Pipeline Updates

### 22. Update ML Pipeline Main Class

**File:** `h-lcdm-codebase/pipeline/ml/ml_pipeline.py`

Replace current tests with full 5-stage architecture:

```python
class MLPipeline:
    def run_ssl_training()  # Stage 1-2: SSL + Domain Adaptation
    def run_pattern_detection()  # Stage 3: Ensemble Detection
    def run_interpretability()  # Stage 4: LIME + SHAP
    def run_validation()  # Stage 5: Full Validation Suite
    def run_unblinding()  # Generate unblinding report
    
    # Support subcommand modes: train, detect, interpret, validate, blind, unblind
    # Training vs. inference modes
    # Model checkpointing
    # Results aggregation
```

### 23. Add ML Training Script

**New file:** `h-lcdm-codebase/scripts/train_ml_models.py`

Command-line interface:

```bash
python scripts/train_ml_models.py \
    --stage ssl \
    --epochs 100 \
    --batch-size 256 \
    --lr 0.001 \
    --modalities cmb,bao,void,galaxy,frb,lyman_alpha,jwst
```

### 24. Update Main CLI

**File:** `h-lcdm-codebase/main.py`

Add `--ml` argument with subcommands (following existing code pattern):

- `--ml train`: Train SSL encoders + domain adaptation on all data (Stages 1-2)
- `--ml detect`: Run blind pattern detection (Stage 3, requires trained models)
- `--ml interpret`: Generate LIME/SHAP explanations (Stage 4)
- `--ml validate`: Full validation suite - bootstrap, null hypothesis, cross-survey (Stage 5)
- `--ml blind`: Enable blind analysis protocol
- `--ml unblind`: Generate unblinding report comparing detections to H-ΛCDM predictions
- `--ml` (no subcommand): Execute all stages in sequence (train → detect → interpret → validate)

Example usage:

```bash
python main.py --ml train          # Stages 1-2 only
python main.py --ml detect         # Stage 3 only (requires trained models)
python main.py --ml interpret      # Stage 4 only
python main.py --ml validate       # Stage 5 only
python main.py --ml                # Full pipeline all stages
```

## Dependencies and Requirements

### 25. Update Requirements

**File:** `h-lcdm-codebase/requirements.txt`

Add ML dependencies:

```
torch>=2.0.0
torchvision>=0.15.0
scikit-learn>=1.3.0
hdbscan>=0.8.33
shap>=0.43.0
lime>=0.2.0.1
```

### 26. Documentation

**New file:** `h-lcdm-codebase/docs/ml_architecture.md`

Complete guide:

- Architecture overview with diagrams
- Data preparation for each modality
- Training procedures
- Interpretation guidelines
- Validation protocols
- Troubleshooting

## Testing and Validation

### 27. Unit Tests

**New directory:** `h-lcdm-codebase/tests/test_ml/`

Test coverage:

- SSL encoder forward/backward pass
- Domain adaptation loss computation
- Anomaly detector scoring
- LIME/SHAP explanation generation
- Validation metric calculations
- Mock data generation

### 28. Integration Tests

**File:** `h-lcdm-codebase/tests/test_ml/test_full_pipeline.py`

End-to-end tests:

- Train on small subset, validate convergence
- Cross-survey validation workflow
- Bootstrap stability on toy dataset
- Null hypothesis testing with mocks
- Blind analysis protocol execution

## Implementation Notes

**Critical Design Principles:**

1. **Blind Test Approach**: ML doesn't know which data contains H-ΛCDM signals (BAO enhancement, FRB timing, etc.). Success = detecting these signals without being told.
2. **No Mock Data for Training**: Only use mocks for null hypothesis validation testing.
3. **All Real Data**: Use astropy/astroquery exclusively for real astronomical data.
4. **Scientific Rigor**: Follow ml-design.md exactly - self-supervised, domain adaptation, ensemble, interpretability, validation.
5. **Full Interpretability**: LIME + SHAP for peer review acceptance.
6. **Quantified Significance**: p-values from null hypothesis testing against mocks.
7. **Pre-Registration**: Document methodology before unblinding.

**Execution Order:**

1. Data infrastructure: items 1-6 (consolidate existing + add galaxy catalogs + dedup voids)
2. Stage 1 SSL: items 7-9 (contrastive learning, encoders, fusion)
3. Stage 2 Domain: items 10-11 (survey-invariant training)
4. Stage 3 Detection: items 12-13 (ensemble anomaly detection)
5. Stage 4 Interpret: items 14-16 (LIME, SHAP, reporting)
6. Stage 5 Validate: items 17-21 (cross-survey, bootstrap, null hypothesis, blind protocol)
7. Integration: items 22-24 (pipeline updates, training scripts, CLI)
8. Final: items 25-28 (dependencies, docs, tests)

**Success Criteria:**

- ML detects H-ΛCDM signals (BAO r_s enhancement, FRB timing patterns, E8×E8 void alignments, etc.) without prior knowledge
- Detections are stable across bootstrap resampling (≥95%)
- Detections are NOT present in mock datasets (p < 0.05)
- Interpretability shows physical features (not instrumental artifacts)
- Cross-survey consistency validates cosmological signal vs. systematics