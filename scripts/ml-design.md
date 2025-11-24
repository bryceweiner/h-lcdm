# ML Architecture for Cross-Survey Pattern Detection in Cosmological Datasets

Based on current literature, here's a scientifically rigorous ML architecture for cross-survey pattern detection in cosmological datasets:

## Core Architecture: Hybrid Self-Supervised + Interpretable Framework

### Stage 1: Self-Supervised Feature Learning

Self-supervised learning (SSL) trained on unlabeled data from your three surveys is the current gold standard for pattern discovery without introducing human bias.[^1][^2] Specifically:

**Contrastive Learning Approach:**
- Use self-supervised methods to learn informative summary statistics that capture multi-scale structure while being insensitive to prescribed systematic effects[^3]
- Train separate encoders for each data type (galaxy catalogs, void catalogs, CMB maps) using contrastive methods like SimCLR or similar
- Multimodal integration can improve pattern detection significantly - a recent AstroM³ model showed up to 14% improvements by combining multiple data modalities[^4]

### Stage 2: Domain Adaptation for Cross-Survey Consistency

**Critical for scientific acceptability:**

Domain adaptation methods implemented during training help extract only features present in all datasets, aligning data distributions so models work across surveys without being affected by dataset-specific noise or systematic differences.[^5]

**Implementation:**
- Train on combined data from all 3 surveys simultaneously
- Use alignment losses to ensure learned features are survey-invariant
- Test on realistic simulated datasets based on your survey specifications before applying to real data[^6]

### Stage 3: Pattern Detection with Multiple Methods

**Don't rely on a single algorithm:**

Different unsupervised methods produce significantly different but valid outcomes for high-dimensional cosmological data - this is natural and expected.[^7]

**Recommended ensemble:**
1. **Isolation Forests** for anomaly scoring
2. **HDBSCAN** for density-based clustering  
3. **Variational Autoencoders** for reconstruction-based detection
4. Use the penultimate layer of your SSL encoder as latent space, then apply multiple anomaly detection algorithms to this representation[^8]

### Stage 4: Interpretability (Essential for Peer Review)

Recent Physical Review Letters paper on cosmological model selection achieved ~97% accuracy using neural networks with LIME interpretability to identify which features influenced decisions.[^9][^10]

**Apply both:**
- **LIME** (Local Interpretable Model-agnostic Explanations) for understanding individual detections
- **SHAP** (SHapley Additive exPlanations) for global feature importance

Interpretable ML helps diagnose why an object is anomalous, highlighting which features contribute most - this improves trust and enables prioritization of compelling candidates.[^11]

### Stage 5: Statistical Validation (Make It Bulletproof)

**Multi-level validation strategy:**

1. **Cross-survey validation**: Multiple-source cross-validation where each survey serves as test set while training on the other two[^12]

2. **Bootstrap stability**: Run detection pipeline 1000+ times with bootstrap resampling - only report patterns that appear consistently

3. **Null hypothesis testing**: 
   - Generate mock datasets matching your surveys' statistical properties
   - Apply identical pipeline to mocks
   - Calculate p-values: how often do random data produce similar patterns?

4. **Blind analysis protocol**: Pre-register your methodology before looking at final results to avoid a posteriori statistics issues

5. Use data-splitting approaches and stability analysis for validation, alongside theoretical statistical results on model selection consistency and uncertainty quantification[^13]

### Practical Implementation Recommendations

**For each dataset type:**

**CMB data:**
- Be aware that CMB large-scale anomalies are typically detected at <3σ level with cosmic variance dominating - factor this into significance testing[^14]
- Work with harmonic space (spherical harmonics) or pixel-based representations
- Account for survey-specific masking and foreground removal

**Void catalogs:**
- Use void properties (size, ellipticity, density contrast) as features
- Cosmic voids are being actively exploited as probes of ΛCDM with Roman, Euclid, and LSST data[^15]

**Galaxy catalogs:**
- Extract morphological and clustering features
- Consider redshift evolution effects

### Why This Approach is "Scientifically Acceptable"

1. **No training labels** = no bias toward expected patterns
2. **Cross-survey agreement** = real cosmological signal, not instrumental artifacts  
3. **Multiple independent methods** = robust against algorithm-specific false positives
4. **Full interpretability** = reviewers can understand *why* patterns are flagged
5. **Rigorous null testing** = quantified statistical significance
6. **Pre-registered protocol** = addresses multiple testing concerns

### Red Flags to Avoid

- Don't heavily rely only on simulations without proper quantification of statistical and systematic errors - state-of-the-art hydrodynamic simulations may not overlap with reality in high dimensions[^16]
- Avoid single-method detection without validation
- Never report patterns without quantifying significance against null hypothesis
- Don't apply post-hoc corrections after seeing results

## Conclusion

This architecture balances cutting-edge ML capabilities with the statistical rigor required for high-impact cosmology papers. The combination of self-supervision, domain adaptation, ensemble detection, interpretability, and multi-level validation creates a defensible methodology that addresses common reviewer concerns about ML "black boxes" in physics.

---

## References

[^1]: Riggi, S., et al. (2024). Self-supervised contrastive learning of radio data for source detection, classification and peculiar object discovery. *Publications of the Astronomical Society of Australia*, 41.

[^2]: Baron Perez, N., et al. (2025). Classification of radio sources through self-supervised learning. *Astronomy & Astrophysics*, 699, A302.

[^3]: Makinen, T. L., et al. (2023). Data compression and inference in cosmology with self-supervised machine learning. *Monthly Notices of the Royal Astronomical Society*, 527(3), 7459.

[^4]: Rizhko, O., & Bloom, J. (2025). AstroM³: A Self-supervised Multimodal Model for Astronomy. *arXiv preprint arXiv:2506.xxxxx*.

[^5]: Villaescusa-Navarro, F., et al. (2022). Machine Learning and Cosmology. In *Multi-Messenger Astrophysics* (pp. 94). *arXiv:2203.08056*.

[^6]: Singh, P., et al. (2024). Cosmology with galaxy cluster properties using machine learning. *Astronomy & Astrophysics*, 687, A24.

[^7]: Logan, C. J. L., & Fotopoulou, S. (2020). Effectively using unsupervised machine learning in next generation astronomical surveys. *Astronomy and Computing*, 33, 100429.

[^8]: Gupta, R., et al. (2025). A Classifier-Based Approach to Multi-Class Anomaly Detection for Astronomical Transients. *RAS Techniques and Instruments*, rzae054.

[^9]: Ocampo, I., et al. (2025). Enhancing Cosmological Model Selection with Interpretable Machine Learning. *Physical Review Letters*, 134, 041002.

[^10]: Ocampo, I., et al. (2024). Enhancing Cosmological Model Selection with Interpretable Machine Learning. *arXiv:2406.08351v2*.

[^11]: Bhambra, J., et al. (2025). Interpreting anomaly detection of SDSS spectra. *Astronomy & Astrophysics*, in press. *arXiv:2510.05235*.

[^12]: Geras, K., & Sutton, C. (2013). Multiple-source cross-validation. In *International Conference on Machine Learning* (pp. 1292–1300). PMLR.

[^13]: Murdoch, W. J., et al. (2024). Interpretable Machine Learning for Discovery: Statistical Challenges and Opportunities. *Annual Review of Statistics and Its Application*, 11, 97–121.

[^14]: Jung, G., et al. (2024). Revisiting the large-scale CMB anomalies: The impact of the SZ signal from the Local Universe. *Astronomy & Astrophysics*, 692, A123.

[^15]: CosmoVerse Istanbul 2025 Workshop. (2025). Cosmic voids as probes of ΛCDM with upcoming observational programs. Retrieved from https://cosmoversetensions.eu/event/cosmoverseistanbul-2025/

[^16]: Villaescusa-Navarro, F., et al. (2022). Machine Learning and Cosmology. *arXiv:2203.08056*, Section on Hydrodynamic Simulations and Systematic Uncertainties.