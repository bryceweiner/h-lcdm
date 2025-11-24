# ML Architecture for Cosmological Pattern Detection

## Overview

This document describes the complete 5-stage machine learning architecture implemented for detecting H-ΛCDM signatures in cosmological data. The architecture is designed to maintain scientific rigor while enabling discovery of subtle patterns in multi-modal astronomical datasets.

## Architecture Principles

### Scientific Rigor
- **Blind Analysis**: ML algorithms do not know which datasets contain H-ΛCDM signals
- **Null Hypothesis Testing**: Extensive validation against mock datasets without signals
- **Bootstrap Stability**: 1000+ resamples ensure robust detections
- **Cross-Survey Validation**: Patterns must be consistent across different surveys

### Technical Design
- **Multi-Modal**: Joint analysis of CMB, BAO, voids, galaxies, FRB, Lyman-α, JWST data
- **Self-Supervised**: No labeled data required - learns from data structure
- **Domain Invariant**: Features work across different surveys and systematics
- **Interpretable**: LIME and SHAP explanations for all detections

## Stage 1: Self-Supervised Feature Learning

### Contrastive Learning Framework
- **SimCLR-based** contrastive learning on multi-modal cosmological data
- **Modality-specific encoders** for each data type (CMB, BAO, voids, etc.)
- **Multimodal fusion** with cross-attention mechanisms
- **Momentum encoders** for stable training

### Data Augmentation Strategies
- **CMB**: Rotations, harmonic space transformations, masking variations
- **BAO**: Redshift binning variations, covariance perturbations
- **Voids**: Scaling, ellipticity perturbations, position jittering
- **Galaxies**: Photometric noise, morphology variations, redshift uncertainties
- **FRB**: Timing noise, dispersion measure variations
- **Lyman-α**: Flux noise, continuum fitting variations
- **JWST**: Photometric uncertainties, selection function variations

### Encoder Architecture
```python
# Example: CMB Encoder
Conv1d(1, 64) -> BatchNorm -> ReLU -> Dropout
Conv1d(64, 128) -> BatchNorm -> ReLU -> Dropout
AdaptiveAvgPool1d(1) -> Linear(128, 512)
```

## Stage 2: Domain Adaptation

### Survey-Invariant Training
- **Maximum Mean Discrepancy (MMD)** loss between survey distributions
- **Domain adversarial training** with gradient reversal layers
- **Survey embeddings** to learn survey-specific features

### Cross-Survey Validation
- **Leave-one-survey-out** cross-validation
- **Distribution alignment metrics** (KL divergence, Wasserstein distance)
- **Systematics quantification** across surveys

## Stage 3: Ensemble Pattern Detection

### Anomaly Detection Methods
1. **Isolation Forest**: Unsupervised tree-based anomaly detection
2. **HDBSCAN**: Density-based clustering for outlier identification
3. **Variational Autoencoder**: Reconstruction-based anomaly scoring

### Ensemble Aggregation
- **Weighted averaging** of anomaly scores
- **Rank aggregation** using Borda count
- **Consensus voting** for high-confidence detections

### Detection Thresholds
- **Adaptive calibration** based on expected anomaly rate
- **Bootstrap validation** of threshold stability

## Stage 4: Interpretability

### LIME (Local Interpretable Model-agnostic Explanations)
- **Local explanations** for individual anomaly detections
- **Feature importance** with statistical significance
- **Stability assessment** across multiple explanations

### SHAP (SHapley Additive exPlanations)
- **Global feature importance** across all detections
- **Interaction effects** between features
- **Model-agnostic** explanations for ensemble methods

### Explanation Quality Metrics
- **R² score**: How well the explanation approximates the model
- **Feature stability**: Consistency across explanation runs
- **Physical plausibility**: Alignment with known physics

## Stage 5: Statistical Validation

### Bootstrap Stability Analysis (1000+ samples)
- **Detection stability**: Patterns present in ≥95% of bootstraps
- **Score distribution**: Uncertainty quantification
- **Confidence intervals**: Statistical significance bounds

### Null Hypothesis Testing
- **Mock dataset generation**: Statistical properties match real data
- **No H-ΛCDM signals**: Mocks lack theoretical signatures
- **p-value calculation**: Significance of real vs. mock differences

### Blind Analysis Protocol
- **Pre-registration**: Methodology documented before results
- **Unblinding report**: Comparison to H-ΛCDM predictions
- **Confirmation bias mitigation**: Results hidden until validation complete

### Cross-Survey Validation
- **Survey consistency**: Patterns robust across different datasets
- **Systematics control**: Distinguish cosmological signals from artifacts
- **Generalizability**: Features work on unseen survey data

## Implementation Details

### Data Pipeline
```
Raw Data → Feature Extraction → SSL Encoding → Domain Adaptation → Anomaly Detection → Interpretation → Validation
```

### Model Training
```bash
# Train SSL encoders
python scripts/train_ml_models.py --stage ssl --epochs 100 --batch-size 256

# Run domain adaptation
python scripts/train_ml_models.py --stage domain --surveys 5

# Execute full pipeline
python main.py --ml  # Runs all stages
```

### CLI Interface
```bash
# Individual stages
python main.py --ml-train      # Stages 1-2
python main.py --ml-detect     # Stage 3
python main.py --ml-interpret  # Stage 4
python main.py --ml-validate   # Stage 5

# Full pipeline
python main.py --ml           # All stages
```

## Validation Metrics

### Success Criteria
- **Bootstrap stability**: ≥95% of resamples show consistent patterns
- **Null hypothesis**: p < 0.05 for real vs. mock data differences
- **Cross-survey consistency**: Patterns appear in multiple surveys
- **Interpretability**: Explanations align with physical intuition

### Performance Monitoring
- **Training convergence**: Loss curves and validation metrics
- **Detection confidence**: Ensemble agreement scores
- **Explanation quality**: R² scores and stability metrics

## File Structure

```
pipeline/ml/
├── __init__.py
├── ml_pipeline.py           # Main 5-stage pipeline
├── ssl_encoder.py           # Stage 1: Contrastive learning
├── domain_adapter.py        # Stage 2: Domain adaptation
├── anomaly_detectors.py     # Stage 3: Ensemble detection
├── ensemble.py              # Stage 3: Aggregation methods
├── interpretability/        # Stage 4: Explanations
│   ├── lime_explainer.py
│   └── shap_explainer.py
├── validation/              # Stage 5: Validation
│   ├── cross_survey_validator.py
│   ├── bootstrap_validator.py
│   ├── null_hypothesis_tester.py
│   └── blind_protocol.py
└── encoders/                # Modality-specific encoders
    ├── __init__.py
    ├── cmb_encoder.py
    ├── bao_encoder.py
    ├── void_encoder.py
    ├── galaxy_encoder.py
    ├── frb_encoder.py
    ├── lyman_alpha_encoder.py
    └── jwst_encoder.py

data/
├── loader.py                # Data loading (enhanced for ML)
├── mock_generator.py        # Mock data for validation
├── data_manifest.py         # Dataset tracking
└── feature_extractors/      # Feature extraction
    ├── cmb_features.py
    ├── bao_features.py
    ├── void_features.py
    ├── galaxy_features.py
    ├── frb_features.py
    ├── lyman_alpha_features.py
    └── jwst_features.py

scripts/
└── train_ml_models.py       # Training script
```

## Dependencies

### Required
- torch>=2.0.0
- scikit-learn>=1.3.0
- hdbscan>=0.8.33
- shap>=0.43.0
- lime>=0.2.0.1

### Optional
- CUDA support for GPU acceleration
- SHAP visualization libraries

## Usage Examples

### Training
```python
from pipeline.ml.ml_pipeline import MLPipeline

# Initialize pipeline
pipeline = MLPipeline(output_dir="results/ml_analysis")

# Run specific stages
ssl_results = pipeline.run_ssl_training()
domain_results = pipeline.run_domain_adaptation()
detection_results = pipeline.run_pattern_detection()
```

### Validation
```python
from pipeline.ml.validation.bootstrap_validator import BootstrapValidator

validator = BootstrapValidator(n_bootstraps=1000)
stability_results = validator.validate_stability(
    model_factory=lambda: EnsembleDetector(input_dim=512),
    full_dataset=test_data
)
```

### Interpretation
```python
from pipeline.ml.interpretability.lime_explainer import LIMEExplainer

explainer = LIMEExplainer(predict_function=model.predict)
explanation = explainer.explain_instance(anomalous_sample)
```

## Troubleshooting

### Common Issues
1. **Memory errors**: Reduce batch size or use CPU-only training
2. **Convergence issues**: Check learning rate and data preprocessing
3. **SHAP failures**: Ensure background dataset is representative
4. **Validation timeouts**: Reduce bootstrap count for initial testing

### Performance Optimization
- Use GPU acceleration for training
- Implement data parallelism for large datasets
- Cache preprocessed features
- Use smaller models for initial testing

## Future Extensions

### Advanced Architectures
- **Graph neural networks** for void network analysis
- **Transformers** for sequential cosmological data
- **Diffusion models** for generative data augmentation

### Additional Validation
- **Adversarial validation** against confounding variables
- **Causal inference** methods for feature relationships
- **Multi-hypothesis testing** corrections

### Interpretability Enhancements
- **Counterfactual explanations** for "what if" scenarios
- **Feature interaction networks** for complex relationships
- **Uncertainty quantification** in explanations

---

This architecture provides a comprehensive, scientifically rigorous framework for discovering subtle cosmological patterns while maintaining full reproducibility and interpretability.
