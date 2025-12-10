"""
Grok Analysis Client
===================

Client for interacting with the Grok API to generate qualitative analysis 
of ML pipeline results.
"""

import requests
import json
import logging
import time
import os
from typing import Dict, Any, Optional, List

class GrokAnalysisClient:
    """
    Client for generating qualitative analysis using Grok.
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Grok client.
        
        Parameters:
            api_key: API key for x.ai. If not provided, will attempt to load from
                    XAI_API_KEY environment variable.
        """
        # Get API key from parameter, environment variable, or raise error
        if api_key:
            self.api_key = api_key
        else:
            self.api_key = os.getenv('XAI_API_KEY')
            if not self.api_key:
                raise ValueError(
                    "xAI API key not provided. Either pass api_key parameter or "
                    "set XAI_API_KEY environment variable."
                )
        self.api_url = "https://api.x.ai/v1/chat/completions"
        self.logger = logging.getLogger(__name__)
        
    def generate_anomaly_report(self, 
                              anomalies: List[Dict[str, Any]], 
                              context: str = "cosmological data",
                              two_stage: bool = True,
                              three_stage: bool = True) -> str:
        """
        Generate a qualitative report for detected anomalies.
        
        Parameters:
            anomalies: List of anomaly dictionaries
            context: Context string for the analysis
            two_stage: If True, generate high-level narrative first, then detailed analysis
            three_stage: If True, also generate analytical test recommendations (requires two_stage=True)
            
        Returns:
            str: Generated analysis text (narrative + detailed analysis + recommendations if three_stage=True)
        """
        try:
            if two_stage:
                # Stage 1: Generate detailed individual analysis
                detailed_analysis = self._generate_detailed_analysis(anomalies, context)
                
                # Stage 2: Generate high-level narrative from detailed analysis
                narrative = self._generate_narrative_synthesis(detailed_analysis, anomalies, context)
                
                # Stage 3: Generate analytical test recommendations (if requested)
                recommendations = ""
                if three_stage:
                    recommendations = self._generate_analytical_recommendations(
                        detailed_analysis, anomalies, context
                    )
                    recommendations = f"\n\n---\n\n{recommendations}"
                
                # Combine: narrative first, then recommendations, then detailed analysis
                return f"{narrative}{recommendations}\n\n---\n\n### Detailed Anomaly Analysis\n\n{detailed_analysis}"
            else:
                # Single-stage: just detailed analysis
                return self._generate_detailed_analysis(anomalies, context)
            
        except Exception as e:
            self.logger.error(f"Failed to generate Grok analysis after retries: {e}")
            return f"Qualitative analysis unavailable: {e}"

    def generate_custom_report(self, prompt: str, max_retries: int = 3, initial_timeout: int = 120) -> str:
        """
        Generate a custom Grok report using an arbitrary prompt.
        """
        try:
            return self._call_grok_api(prompt, max_retries=max_retries, initial_timeout=initial_timeout)
        except Exception as e:
            self.logger.error(f"Failed to generate custom Grok report: {e}")
            return f"Grok interpretation unavailable: {e}"
    
    def _generate_detailed_analysis(self, 
                                   anomalies: List[Dict[str, Any]], 
                                   context: str) -> str:
        """Generate detailed individual anomaly analysis."""
        # Construct prompt
        prompt = self._construct_anomaly_prompt(anomalies, context)
        
        # Log prompt size for debugging
        prompt_length = len(prompt)
        non_indeterminate_count = len([
            a for a in anomalies 
            if a.get('favored_model', 'INDETERMINATE') != 'INDETERMINATE'
        ])
        self.logger.info(
            f"Grok detailed analysis: prompt_length={prompt_length} chars, "
            f"non_indeterminate_anomalies={non_indeterminate_count}"
        )
        
        # Call API with retries
        response = self._call_grok_api(prompt, max_retries=3)
        
        return response
    
    def _generate_narrative_synthesis(self, 
                                     detailed_analysis: str,
                                     anomalies: List[Dict[str, Any]],
                                     context: str) -> str:
        """Generate high-level academic narrative from detailed analysis."""
        
        # Filter to only H_LCDM_candidate anomalies (exclude LCDM_consistent)
        h_lcdm_anomalies = [
            a for a in anomalies 
            if a.get('favored_model', 'INDETERMINATE') == 'H_LCDM_candidate'
        ]
        
        # Summarize key statistics
        h_lcdm_count = len(h_lcdm_anomalies)
        if h_lcdm_anomalies:
            scores = [a.get('anomaly_score', 0.0) for a in h_lcdm_anomalies]
            max_score = max(scores)
            avg_score = sum(scores) / len(scores)
            score_summary = f"{h_lcdm_count} H-ΛCDM candidate anomalies detected (scores: {max_score:.3f} max, {avg_score:.3f} avg)"
        else:
            score_summary = "No H-ΛCDM candidate anomalies detected"
        
        # Extract redshift regimes and modalities
        redshift_regimes = set()
        modalities = set()
        for a in h_lcdm_anomalies:
            ctx = a.get('context', {}) if isinstance(a.get('context'), dict) else {}
            zreg = ctx.get('redshift_regime', '')
            if zreg and zreg != 'n/a':
                redshift_regimes.add(zreg)
            mods = ctx.get('modalities', [])
            modalities.update(mods)
        
        # Extract key points from detailed analysis (first 6000 chars should contain individual discussions)
        analysis_excerpt = detailed_analysis[:6000] if len(detailed_analysis) > 6000 else detailed_analysis
        if len(detailed_analysis) > 6000:
            analysis_excerpt += "\n\n[... detailed analysis continues with individual anomaly discussions ...]"
        
            # Extract which modalities are actually present
            modality_list = sorted(list(modalities)) if modalities else []
            modality_categories = {
                'cmb': [m for m in modality_list if m.startswith('cmb_')],
                'bao': [m for m in modality_list if m.startswith('bao_')],
                'voids': [m for m in modality_list if m.startswith('void_')],
                'other': [m for m in modality_list if not any(m.startswith(p) for p in ['cmb_', 'bao_', 'void_'])]
            }
            
            narrative_prompt = f"""
You are a data scientist summarizing unsupervised ML anomaly detection results. Your task is to describe what the ML found WITHOUT imposing theoretical bias.

## METHODOLOGY
- 5-stage unsupervised pipeline: SSL → Domain Adaptation → Ensemble Anomaly Detection → Interpretability → Validation
- Anomalies represent distributional deviations in a fused 512-D latent space
- The ML is theory-agnostic: it identifies statistical outliers, not physical mechanisms

## RESULTS SUMMARY
{score_summary}
- Total samples analyzed: {len(anomalies)}
- Score distribution: ≥0.70 flagged as candidates, ≤0.55 as consistent with baseline
- Redshift regimes present: {', '.join(sorted(redshift_regimes)) if redshift_regimes else 'various'}

## MODALITIES ANALYZED ({len(modality_list)} total)
- CMB ({len(modality_categories['cmb'])}): {', '.join(modality_categories['cmb']) if modality_categories['cmb'] else 'none'}
- BAO ({len(modality_categories['bao'])}): {', '.join(modality_categories['bao']) if modality_categories['bao'] else 'none'}
- Voids ({len(modality_categories['voids'])}): {', '.join(modality_categories['voids']) if modality_categories['voids'] else 'none'}
- Other ({len(modality_categories['other'])}): {', '.join(modality_categories['other']) if modality_categories['other'] else 'none'}

## TASK: Write 300-400 words addressing

1. **Primary pattern**: What modality combinations appear in high-score anomalies? Report which data types show joint deviations without pre-assuming physical cause.

2. **Cross-modal structure**: Do anomalies cluster in specific modality combinations, or appear uniformly across data types? Be specific about which modalities correlate.

3. **Redshift distribution**: At what redshifts do anomalies concentrate? Is there clustering, or uniform distribution?

4. **Score distribution**: Is there bimodality, continuous spread, or clustering? What does the gap (if any) between high and low scores suggest?

5. **What baseline-consistent samples show**: Which modalities appear "normal"? This bounds where deviations occur.

6. **Statistical robustness**: Bootstrap stability, cross-survey agreement, false positive assessment.

CRITICAL RULES:
- DO NOT assume CMB TT/TE/EE correlations are the primary signal unless the data shows it
- DO NOT privilege any single modality category—report what the ML found across ALL data types
- If voids, galaxies, FRBs, Lyman-α, JWST, or GW data contribute to anomalies, say so
- Describe patterns agnostically; let the data speak

TONE: Empirical, dispassionate. Report findings, not theory confirmation.
"""
        self.logger.info("Generating high-level narrative synthesis from detailed analysis")
        narrative = self._call_grok_api(narrative_prompt, max_retries=3, initial_timeout=120)
        
        return f"### High-Level Scientific Narrative\n\n{narrative}"
    
    def _generate_analytical_recommendations(self,
                                            detailed_analysis: str,
                                            anomalies: List[Dict[str, Any]],
                                            context: str) -> str:
        """Generate specific analytical test recommendations from detailed analysis."""
        
        # Filter to only H_LCDM_candidate anomalies
        h_lcdm_anomalies = [
            a for a in anomalies 
            if a.get('favored_model', 'INDETERMINATE') == 'H_LCDM_candidate'
        ]
        
        # Extract key information about anomalies
        anomaly_summary = []
        datasets_mentioned = set()
        redshift_ranges = []
        modalities_mentioned = set()
        
        for a in h_lcdm_anomalies[:10]:  # Top 10 for context
            idx = a.get('sample_index', 'N/A')
            score = a.get('anomaly_score', 0.0)
            ctx = a.get('context', {}) if isinstance(a.get('context'), dict) else {}
            zreg = ctx.get('redshift_regime', '')
            redshift = ctx.get('redshift')
            mods = ctx.get('modalities', [])
            
            modalities_mentioned.update(mods)
            if zreg and zreg != 'n/a':
                redshift_ranges.append(zreg)
            if redshift is not None:
                redshift_ranges.append(f"z={redshift:.3f}")
            
            # Extract dataset names from modalities
            for mod in mods:
                if 'cmb_' in mod:
                    datasets_mentioned.add(mod.replace('cmb_', '').replace('_tt', '').replace('_te', '').replace('_ee', ''))
                elif 'bao_' in mod:
                    datasets_mentioned.add(mod.replace('bao_', ''))
                elif 'void_' in mod:
                    datasets_mentioned.add(mod.replace('void_', ''))
            
            anomaly_summary.append(f"Sample {idx}: score={score:.3f}, z={zreg or redshift or 'n/a'}, mods={mods[:3]}")
        
        # Extract key excerpts from detailed analysis (focus on physical interpretations)
        analysis_excerpt = detailed_analysis[:8000] if len(detailed_analysis) > 8000 else detailed_analysis
        if len(detailed_analysis) > 8000:
            analysis_excerpt += "\n\n[... analysis continues ...]"
        
            # Build modality summary from actual anomalies
            modality_counts = {}
            for mod in modalities_mentioned:
                modality_counts[mod] = modality_counts.get(mod, 0) + 1
            modality_summary = ", ".join([f"{k}" for k in sorted(modality_counts.keys())])
        
            recommendations_prompt = f"""
You are a data scientist designing follow-up analyses for ML-detected anomalies in cosmological data.

## CONTEXT: What the ML Found

The unsupervised ML pipeline (SSL + Domain Adaptation + Ensemble Detection) flagged anomalies across multiple cosmological datasets. Your task is to design tests that VERIFY whether these anomalies reflect:
1. Genuine physical deviations from standard cosmology
2. Unmodeled systematics
3. Statistical fluctuations

## ANOMALY SUMMARY

Top anomalies detected in:
{chr(10).join(anomaly_summary[:10])}

Datasets with anomalies: {datasets_mentioned}
Modalities involved: {modality_summary}
Redshift ranges: {', '.join(set(redshift_ranges[:10])) if redshift_ranges else 'various'}

## MODALITY CATEGORIES IN THE DATA

The ML analyzed 23 modalities across multiple physical probes:

**CMB (Cosmic Microwave Background)**
- Temperature (TT): ACT DR6, Planck 2018, SPT-3G, COBE, WMAP
- Polarization (TE, EE): ACT DR6, Planck 2018, SPT-3G, WMAP

**BAO (Baryon Acoustic Oscillations)**
- Galaxy surveys: BOSS DR12, DESI, eBOSS

**Large-Scale Structure**
- Void catalogs: SDSS DR7, SDSS DR16
- Galaxy clustering: combined surveys

**Multi-messenger / High-z**
- Fast Radio Bursts (FRBs): dispersion measure catalog
- Lyman-α forest: high-z intergalactic medium
- JWST: high-z galaxy population
- Gravitational waves: LIGO, Virgo

## TASK: Design Tests Based on ML Findings

For EACH modality category where anomalies were detected, propose a specific follow-up test.

**CRITICAL: Let the data guide the tests.**
- If the ML found correlated anomalies in CMB + BAO, test that correlation specifically
- If voids or galaxy clustering show anomalies, design void/clustering tests
- If high-z probes (JWST, Lyman-α) are flagged, design high-z tests
- If GW or FRB data contribute, design multi-messenger tests

## OUTPUT FORMAT

For each recommendation:

### Recommendation N: [Descriptive Title Based on Actual Anomaly Pattern]
- **Dataset**: [Specific survey/data product where ML found anomalies]
- **Test**: [Methodology to verify whether anomaly is physical or systematic]
- **ΛCDM Null**: [What standard cosmology predicts, with uncertainties]
- **Alternative Hypothesis**: [What a deviation would imply—be agnostic about mechanism]
- **Discriminating Power**: [What makes this test conclusive]
- **Connection to Anomalies**: [Which specific ML-flagged samples this addresses]
- **Feasibility**: [Data availability, computational cost]
- **Potential Confounds**: [Systematics, selection effects, contamination]

## REQUIREMENTS

1. **Cover ALL modality categories** where anomalies appear—not just CMB
2. **Be empirical**: Describe what the ML found, not what theory predicts
3. **Include multi-probe tests**: If anomalies appear in both CMB and BAO (or voids, or galaxies), test their correlation
4. **Include null tests**: Some recommendations should test whether "anomalies" are consistent with noise
5. **Address under-explored probes**: If GW, FRB, JWST, or Lyman-α data show anomalies, these deserve dedicated tests

Generate 6-10 recommendations covering the full range of anomalous modalities.
"""   
        self.logger.info("Generating analytical test recommendations from detailed analysis")
        recommendations = self._call_grok_api(recommendations_prompt, max_retries=3, initial_timeout=120)
        
        return f"### Recommended Analytical Tests\n\n{recommendations}"
    
    def _construct_anomaly_prompt(self, anomalies: List[Dict[str, Any]], context: str) -> str:
        """Construct the prompt for Grok."""
        
        # Filter to only non-indeterminate anomalies (H_LCDM_candidate and LCDM_consistent)
        non_indeterminate = [
            a for a in anomalies 
            if a.get('favored_model', 'INDETERMINATE') != 'INDETERMINATE'
        ]
        
        # Sort by score (descending) for consistent ordering
        non_indeterminate_sorted = sorted(
            non_indeterminate, 
            key=lambda x: x.get('anomaly_score', 0.0), 
            reverse=True
        )
        
        # Format ALL non-indeterminate anomalies for the prompt
        h_lcdm_anomalies = []
        lcdm_anomalies = []
        
        h_lcdm_counter = 1
        lcdm_counter = 1
        
        for anomaly in non_indeterminate_sorted:
            score = anomaly.get('anomaly_score', 'N/A')
            idx = anomaly.get('sample_index', 'N/A')
            favored = anomaly.get('favored_model', 'INDETERMINATE')
            tags = anomaly.get('ontology_tags', [])
            ctx = anomaly.get('context', {}) if isinstance(anomaly.get('context'), dict) else {}
            zreg = ctx.get('redshift_regime', 'n/a')
            redshift = ctx.get('redshift', None)
            mods = ctx.get('modalities', [])
            mods_str = ', '.join(mods[:8]) if mods else 'unknown'  # Show more modalities
            if len(mods) > 8:
                mods_str += f" (+{len(mods)-8} more)"
            z_info = f"z={redshift:.3f}" if redshift is not None else f"z_regime={zreg}"
            
            if favored == 'H_LCDM_candidate':
                anomaly_line = f"  H-{h_lcdm_counter}. Sample {idx} | Score={score:.4f} | {z_info} | Modalities=[{mods_str}] | Tags={tags}"
                h_lcdm_anomalies.append(anomaly_line)
                h_lcdm_counter += 1
            elif favored == 'LCDM_consistent':
                anomaly_line = f"  Λ-{lcdm_counter}. Sample {idx} | Score={score:.4f} | {z_info} | Modalities=[{mods_str}] | Tags={tags}"
                lcdm_anomalies.append(anomaly_line)
                lcdm_counter += 1
        
        # Format anomaly lists
        h_lcdm_text = "\n".join(h_lcdm_anomalies) if h_lcdm_anomalies else "  None"
        lcdm_text = "\n".join(lcdm_anomalies) if lcdm_anomalies else "  None"
        
        # Count model preferences
        model_counts = {}
        for anomaly in anomalies:
            model = anomaly.get('favored_model', 'INDETERMINATE')
            model_counts[model] = model_counts.get(model, 0) + 1
        model_summary = ", ".join([f"{k}: {v}" for k, v in model_counts.items()])

        prompt = f"""
You are analyzing unsupervised ML anomaly detection results in {context}.

## PART I: METHODOLOGICAL CONTEXT

### Pipeline Architecture
The analysis uses a 5-stage unsupervised ML pipeline:
- **Stage 1**: Self-supervised contrastive learning (SimCLR) on multi-modal cosmological data
- **Stage 2**: Domain adaptation via Maximum Mean Discrepancy (MMD) for survey-invariant features
- **Stage 3**: Ensemble anomaly detection (Isolation Forest + HDBSCAN + VAE) on fused 512-dimensional latent space
- **Stage 4**: Interpretability (LIME/SHAP) mapping anomalies to physical observables
- **Stage 5**: Statistical validation (bootstrap stability, null hypothesis testing, cross-survey validation)

### Anomaly Detection Method
Anomalies are detected in a **fused latent space** after SSL and domain adaptation:
- Raw observations encoded into shared 512-D space
- Domain adaptation ensures survey invariance across datasets
- Ensemble methods identify distributional deviations from learned normal patterns
- Anomalies represent **distributional deviations**, not direct physical measurements

### Scoring System
- Scores range 0.0–1.0 (higher = more anomalous)
- ≥ 0.70: Strong deviation from baseline expectations
- ≤ 0.55: Consistent with standard cosmology
- 0.55–0.70: INDETERMINATE (requires further investigation)

---

## PART II: DATA MODALITIES ANALYZED

The ML fuses 23 modalities across distinct physical probes:

### CMB (Cosmic Microwave Background)
| Modality | Survey | Observable |
|----------|--------|------------|
| cmb_act_dr6_tt, _te, _ee | ACT DR6 | Temperature/Polarization power spectra |
| cmb_planck_2018_tt, _te, _ee | Planck 2018 | Temperature/Polarization power spectra |
| cmb_spt3g_tt, _te, _ee | SPT-3G | Temperature/Polarization power spectra |
| cmb_cobe_tt | COBE DMR | Low-ℓ temperature |
| cmb_wmap_tt, _te | WMAP | Temperature/Polarization |

### BAO (Baryon Acoustic Oscillations)
| Modality | Survey | Observable |
|----------|--------|------------|
| bao_boss_dr12 | BOSS DR12 | Galaxy clustering, D_V/r_s |
| bao_desi | DESI Year 1 | Galaxy/quasar BAO |
| bao_eboss | eBOSS DR16 | Extended galaxy BAO |

### Large-Scale Structure
| Modality | Survey | Observable |
|----------|--------|------------|
| void_sdss_dr7 | SDSS DR7 | Void catalog, size distribution |
| void_sdss_dr16 | SDSS DR16 | Void catalog, clustering |
| galaxy | Combined | Galaxy clustering statistics |

### Multi-Messenger / High-z Probes
| Modality | Data | Observable |
|----------|------|------------|
| frb | FRB catalog | Dispersion measures, host galaxies |
| lyman_alpha | Lyman-α forest | IGM at z~2-6 |
| jwst | JWST surveys | High-z (z>8) galaxy population |
| gw_ligo | LIGO | GW event parameters |
| gw_virgo | Virgo | GW event parameters |

---

## PART III: ANOMALY DATA

Total anomalies analyzed: {len(anomalies)}
Model preference distribution: {model_summary}
Non-indeterminate anomalies requiring individual discussion: {len(non_indeterminate_sorted)}

### High-Score Anomalies (scores ≥ 0.7, total: {len(h_lcdm_anomalies)}):
{h_lcdm_text}

### Baseline-Consistent Anomalies (scores ≤ 0.55, total: {len(lcdm_anomalies)}):
{lcdm_text}

---

## PART IV: ANALYSIS REQUIREMENTS

### 1. Individual Anomaly Discussions (REQUIRED for each non-indeterminate anomaly)

**For each high-score anomaly, address:**

a) **Basic identification**: Sample index, score, redshift

b) **Modality analysis**: 
   - Which modalities contribute? List ALL that appear (CMB, BAO, voids, galaxy, FRB, Lyman-α, JWST, GW)
   - Do multiple modalities show joint deviation, or is it isolated?
   - Which surveys agree vs disagree?

c) **Cross-modal patterns**:
   - Are deviations correlated ACROSS different physical probes?
   - Example: CMB + BAO together? Voids + galaxies together? CMB + GW?
   - Report what you see, not what theory predicts

d) **Redshift distribution**:
   - At what z does this anomaly appear?
   - Is it localized or spread across redshift bins?

e) **Alternative explanations**:
   - Could this be unmodeled systematics?
   - Statistical fluctuation probability?
   - Survey-specific artifacts vs cross-survey agreement?

**For each baseline-consistent anomaly, address:**
- Which modalities appear normal
- What constraints this places on deviations

### 2. Collective Pattern Analysis

After individual discussions, synthesize:

a) **Cross-modal clustering**: 
   - How many H-ΛCDM candidates show TT+TE+EE joint elevation?
   - This count is the primary evidence metric for H-ΛCDM

b) **Redshift distribution**:
   - Do anomalies cluster at z ~ 2 (structure formation) or z ~ 1100 (recombination)?
   - Clustering at specific physically-motivated redshifts strengthens interpretation

c) **Modality consensus**:
   - Which survey combinations (ACT+Planck, BOSS+DESI) show consistent anomalies?
   - Cross-survey agreement reduces systematic contamination probability

d) **Score distribution**:
   - Gap between highest H-ΛCDM candidate and LCDM-consistent scores?
   - Bimodal distribution suggests real physical distinction; continuous distribution suggests noise

### 3. Model Comparison Analysis

Explicitly address:

a) **Quantitative consistency check**:
   - Are anomaly amplitudes consistent with α ≈ -5.7?
   - Do BAO anomalies imply r_s ~ 150.7 Mpc or r_s ~ 147.5 Mpc?

b) **QTEP manifestation**:
   - Does the ratio of coherent-to-decoherent features in the anomaly pattern approximate 2.257?
   - This is speculative but worth noting if apparent

c) **γ/H scaling**:
   - Do anomaly strengths scale with expected γ(z)/H(z) across redshift bins?
   - Stronger anomalies at z ~ 1100 vs z ~ 0.5 would support this

### 4. Validation Status and Next Steps

a) **Current validation**:
   - Bootstrap stability: Do H-ΛCDM candidates persist across resampling?
   - Null hypothesis: What is the probability of this many high-score anomalies by chance?
   - Cross-survey: Do ACT and Planck flag the same physical scales?

b) **Recommended follow-up**:
   - Explicit cross-correlation measurement: Compute ρ(ΔC_ℓ^TT, ΔC_ℓ^TE) directly
   - α extraction: Fit α as free parameter to CMB damping tail
   - r_s consistency: Compare CMB-derived vs BAO-derived sound horizon

c) **Priority ranking**:
   - Which anomalies are most robust to systematic variation?
   - Which would be most constraining if confirmed?

---

## PART V: OUTPUT STRUCTURE

Organize response as:

### Section 1: Individual Anomaly Discussions
- Subsection H-1, H-2, ... for each high-score anomaly
- Subsection Λ-1, Λ-2, ... for each baseline-consistent
- Each subsection: ~100-150 words covering points (a)-(e) above

### Section 2: Collective Patterns and Cross-Modal Coherence
- Which modality combinations appear together in high-score anomalies?
- Report ALL cross-modal patterns: CMB×CMB, CMB×BAO, CMB×voids, BAO×voids, galaxy×voids, etc.
- Do FRB, Lyman-α, JWST, or GW data contribute? If so, report those patterns too
- Redshift and survey patterns
- ~200-300 words

### Section 3: Model Comparison
- Do anomalies cluster in ways consistent with deviations from ΛCDM?
- What the baseline-consistent samples constrain
- Be agnostic about mechanism—describe patterns, not theory
- ~150-200 words

### Section 4: Validation and Recommendations
- Current statistical confidence
- Prioritized follow-up analyses across ALL relevant modalities
- ~150 words

---

## WRITING REQUIREMENTS

**Empirical Focus:**
- Report what the ML found, not what theory predicts
- Specify which modalities contribute to each anomaly (ALL of them)
- Do not privilege CMB over other probes unless data warrants it

**Tone:**
- Third person throughout ("The analysis reveals...", "Sample 86 exhibits...")
- Formal, dispassionate, appropriate for high-impact letters journal
- Definitive logical connectors ("this implies", "it follows") where warranted
- Appropriate hedging for speculative connections

**Critical Check:**
- Did you report on ALL modality categories with anomalies (CMB, BAO, voids, galaxy, FRB, Lyman-α, JWST, GW)?
- For multi-modal anomalies: Which combinations appear? Are they correlated across probes?
- If only CMB shows anomalies: Say so explicitly
- If non-CMB probes (voids, GW, FRB, etc.) show anomalies: These deserve equal attention
"""  
        return prompt

    def _call_grok_api(self, prompt: str, max_retries: int = 3, initial_timeout: int = 120) -> str:
        """
        Make the actual API call with retry logic and exponential backoff.
        
        Parameters:
            prompt: The prompt to send to Grok
            max_retries: Maximum number of retry attempts
            initial_timeout: Initial timeout in seconds (increases with retries)
            
        Returns:
            str: Generated analysis text
        """
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}"
        }
        
        payload = {
            "messages": [
                {
                    "role": "system",
                    "content": "You are a data scientist analyzing ML anomaly detection results on multi-modal cosmological data. Report patterns empirically without bias toward any specific theoretical interpretation."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            "model": "grok-4-latest",
            "stream": False,
            "temperature": 0.2
        }
        
        last_exception = None
        for attempt in range(max_retries):
            try:
                # Increase timeout with each retry: 120s, 180s, 240s
                timeout = initial_timeout + (attempt * 60)
                self.logger.info(
                    f"Grok API call attempt {attempt + 1}/{max_retries} "
                    f"(timeout={timeout}s)"
                )
                
                response = requests.post(
                    self.api_url, 
                    headers=headers, 
                    json=payload, 
                    timeout=timeout
                )
                response.raise_for_status()
                
                data = response.json()
                content = data['choices'][0]['message']['content']
                
                self.logger.info(
                    f"Grok API call successful: response_length={len(content)} chars"
                )
                return content
                
            except requests.exceptions.Timeout as e:
                last_exception = e
                self.logger.warning(
                    f"Grok API timeout on attempt {attempt + 1}/{max_retries}: {e}"
                )
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt  # Exponential backoff: 1s, 2s, 4s
                    self.logger.info(f"Retrying after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise requests.exceptions.Timeout(
                        f"Grok API call timed out after {max_retries} attempts "
                        f"(final timeout={timeout}s). The prompt may be too long or "
                        f"the API may be experiencing high load."
                    ) from e
                    
            except requests.exceptions.RequestException as e:
                last_exception = e
                self.logger.warning(
                    f"Grok API error on attempt {attempt + 1}/{max_retries}: {e}"
                )
                status = getattr(getattr(e, "response", None), "status_code", None)
                # Log response body for 4xx diagnostics
                if e.response is not None:
                    try:
                        self.logger.error(f"Grok response body: {e.response.text}")
                    except Exception:
                        pass

                # On explicit auth/argument errors, stop immediately and return a clear message
                if status in (400, 401, 403):
                    return f"Grok interpretation unavailable: {e.response.text if e.response is not None else str(e)}"

                # Retry only for server errors (>=500); on other 4xx, stop immediately
                if attempt < max_retries - 1 and (status is None or status >= 500):
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    break
        
        # Fallback: return a descriptive message instead of raising
        if last_exception:
            return f"Grok interpretation unavailable: {last_exception}"
        return "Grok interpretation unavailable: unknown error"

