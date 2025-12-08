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
        
        narrative_prompt = f"""
You are a senior theoretical physicist synthesizing cosmological anomaly detection results into an accessible academic narrative about early universe physics.

CONTEXT:
{context}

KEY RESULTS SUMMARY:
- {score_summary}
- Redshift regimes: {', '.join(sorted(redshift_regimes)) if redshift_regimes else 'various'}
- Modalities involved: {len(modalities)} different observational probes (CMB, BAO, voids, galaxies, FRBs, GW, JWST, Lyman-α)

DETAILED ANALYSIS EXCERPT (for synthesis):
{analysis_excerpt}

TASK:
Generate a high-level academic narrative that:
1. **Sets the scientific context**: Explain what these unsupervised ML anomaly detections reveal about early universe physics
2. **Draws connections**: Connect the detected anomalies to fundamental physics principles (QTEP ≈ 2.257, γ = H/π², holographic information theory, quantum thermodynamics)
3. **Early universe focus**: Emphasize implications for:
   - Primordial structure formation
   - Information-theoretic constraints on cosmic evolution
   - Quantum-to-classical transitions in cosmology
   - Holographic corrections to standard cosmology
   - The relationship between information entropy and spacetime geometry
4. **Accessible language**: Use clear, precise language that connects technical results to physical intuition
5. **Academic tone**: Formal, third-person, dispassionate style appropriate for a high-impact letters journal
6. **Narrative flow**: Create a coherent story that guides the reader from detection → interpretation → implications → conclusions

STRUCTURE:
- Opening: What the pipeline detected and why it matters for early universe physics
- Middle: How these anomalies connect to fundamental information-theoretic principles and early universe dynamics
- Closing: Implications for our understanding of cosmic evolution and information constraints

CRITICAL REQUIREMENTS:
- NEVER use first person ("I", "we", "our", "my")
- Use phrases like "The analysis reveals...", "These detections indicate...", "The framework suggests..."
- Connect anomalies to early universe physics explicitly
- Make the narrative accessible while maintaining rigor
- Focus on drawing conclusions about information-theoretic constraints on cosmic evolution

Generate a narrative that sets the tone for understanding these results in the context of early universe physics and information theory.
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
        
        recommendations_prompt = f"""
You are a senior theoretical physicist proposing specific analytical tests to reveal the physical features detected by unsupervised ML anomaly detection.

CONTEXT:
{context}

ANOMALY SUMMARY:
{chr(10).join(anomaly_summary[:10])}

KEY DATASETS INVOLVED:
{', '.join(sorted(datasets_mentioned)) if datasets_mentioned else 'Various cosmological surveys'}

REDSHIFT REGIMES:
{', '.join(set(redshift_ranges[:10])) if redshift_ranges else 'Various'}

MODALITIES:
{', '.join(sorted(modalities_mentioned)[:15]) if modalities_mentioned else 'CMB, BAO, voids, galaxies, FRBs, GW, JWST, Lyman-α'}

DETAILED ANALYSIS (for reference):
{analysis_excerpt}

TASK:
Generate specific, actionable analytical test recommendations that would reveal the physical features being detected. For each recommendation, provide:

1. **Specific Dataset**: Which exact dataset/survey to analyze (e.g., "Planck 2018 CMB TT power spectrum", "DESI BAO measurements at z=0.5-1.0", "SDSS DR16 void catalog")
2. **Specific Test**: What exact analytical test to perform (e.g., "Cross-correlation analysis between CMB TT and EE modes", "BAO scale measurement at z=0.8", "Void size distribution power-law fit")
3. **Expected Feature**: What specific feature should be revealed (e.g., "2.18% shift in sound horizon scale", "QTEP ratio ≈ 2.257 in decoherence patterns", "γ = H/π² scaling in void geometry")
4. **Physical Interpretation**: Why this test would reveal H-ΛCDM signatures vs ΛCDM
5. **Feasibility**: Whether this test can be performed with existing data or requires new observations

STRUCTURE:
- Group recommendations by dataset/modality
- Prioritize tests that can be performed immediately with existing data
- Include both direct tests (measuring predicted quantities) and indirect tests (cross-correlations, consistency checks)
- Specify exact redshift ranges, multipole ranges, or other parameter ranges where applicable

CRITICAL REQUIREMENTS:
- NEVER use first person ("I", "we", "our", "my")
- Be SPECIFIC: name exact datasets, exact tests, exact parameter ranges
- Connect each test to the detected anomalies and H-ΛCDM predictions
- Focus on tests that would reveal information-theoretic signatures (QTEP, γ, holographic corrections)
- Prioritize tests with existing data over future observations
- Use formal, third-person academic tone

Generate 5-10 specific analytical test recommendations, organized by dataset/modality, that would reveal the features being detected.
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
You are a senior theoretical physicist analyzing unsupervised ML anomaly detection results in {context}.

CRITICAL METHODOLOGICAL CONTEXT:

1. **Pipeline Architecture**: The analysis uses a 5-stage unsupervised ML pipeline:
   - Stage 1: Self-supervised contrastive learning (SimCLR) on multi-modal cosmological data
   - Stage 2: Domain adaptation via Maximum Mean Discrepancy (MMD) to ensure survey-invariant features
   - Stage 3: Ensemble anomaly detection (Isolation Forest + HDBSCAN + VAE) on fused 512-dimensional latent space
   - Stage 4: Interpretability (LIME/SHAP) to map anomalies back to physical observables
   - Stage 5: Statistical validation (bootstrap stability, null hypothesis testing, cross-survey validation)

2. **Anomaly Detection Method**: Anomalies are detected in a **fused latent space** after SSL and domain adaptation. This means:
   - Raw observations (CMB power spectra, BAO measurements, void catalogs, etc.) are encoded into a shared 512-dimensional latent space
   - Domain adaptation ensures features are survey-invariant (ACT, Planck, SPT-3G for CMB; BOSS, DESI, eBOSS for BAO)
   - Ensemble methods identify samples that deviate from the learned normal distribution in this latent space
   - Anomalies represent **distributional deviations**, not direct physical measurements

3. **Scoring System**: 
   - Anomaly scores range from 0.0 to 1.0 (higher = more anomalous)
   - Scores ≥ 0.7: Classified as H_LCDM_candidate (strong deviation from ΛCDM expectations)
   - Scores ≤ 0.55: Classified as LCDM_consistent (consistent with standard cosmology)
   - Scores 0.55-0.7: Classified as INDETERMINATE (requires further investigation)

4. **Unsupervised Nature**: No labeled data was used. The pipeline learns normal patterns from data structure alone, then identifies deviations.

KEY DEFINITIONS:
- QTEP (Quantum Thermodynamic Entropy Partition): The ratio S_coh / |S_decoh| ≈ ln(2)/(1-ln(2)) ≈ 2.257, representing the fundamental information partition in quantum measurement.
- Gamma (γ): The holographic information processing rate (γ = H/π² in fundamental units) that serves as the affine parameter across all timelike worldlines.

ANOMALY DATA:
Total anomalies analyzed: {len(anomalies)}
Model preference distribution: {model_summary}
Non-indeterminate anomalies (require individual discussion): {len(non_indeterminate_sorted)}

H-ΛCDM CANDIDATE ANOMALIES (scores ≥ 0.7, total: {len(h_lcdm_anomalies)}):
{h_lcdm_text}

ΛCDM-CONSISTENT ANOMALIES (scores ≤ 0.55, total: {len(lcdm_anomalies)}):
{lcdm_text}

CRITICAL INTERPRETATION REQUIREMENTS:

1. **INDIVIDUAL ANOMALY DISCUSSION (REQUIRED)**: 
   You MUST provide individual discussion for EACH non-indeterminate anomaly listed above.
   For each H_LCDM_candidate anomaly, provide:
   - Sample index and score
   - Which modalities contribute most significantly (use the provided modalities list)
   - Redshift regime and its physical implications
   - Physical interpretation as distributional deviation in latent space
   - Why this anomaly favors H-ΛCDM over ΛCDM
   - Potential physical mechanisms (QTEP effects, γ evolution, information-theoretic constraints)
   
   For each LCDM_consistent anomaly, provide:
   - Sample index and score
   - Why this anomaly is consistent with standard ΛCDM expectations
   - Which modalities show standard behavior
   - Implications for the overall analysis

2. **Collective Physical Implications**: After individual discussions, address:
   - Patterns across H_LCDM_candidate anomalies: common modalities, redshift regimes, score distributions
   - What do high scores (≥ 0.7) collectively imply? Consider: (a) new physics signatures consistent with H-ΛCDM predictions, (b) unmodeled systematics, (c) rare astrophysical objects, (d) statistical fluctuations
   - Emphasize that these are **latent-space anomalies** requiring interpretation through LIME/SHAP to map to physical observables
   - Ensemble consensus (agreement across Isolation Forest, HDBSCAN, VAE) strengthens confidence

3. **Model Comparison**: Explain:
   - How the favored_model classification influences interpretation
   - Why H-ΛCDM predicts stronger deviations than ΛCDM for certain redshift regimes/modalities
   - The role of QTEP ≈ 2.257 and γ = H/π² in explaining H_LCDM_candidate anomalies
   - Why LCDM_consistent anomalies support standard cosmology

4. **Validation and Next Steps**: Discuss:
   - The importance of statistical validation (bootstrap stability, null hypothesis testing, cross-survey consistency)
   - Which modalities/redshift regimes should be prioritized for cross-validation
   - Recommended follow-up analyses (supervised classification, targeted observations, etc.)

5. **Reporting Style**: 
   - Use formal, dispassionate tone appropriate for a high-impact letters journal
   - Write in third person or passive voice; NEVER use first person ("I", "we", "our", "my")
   - Use phrases like "The analysis reveals...", "These results indicate...", "The pipeline detects..." instead of "I present...", "We find...", "Our analysis..."
   - Avoid speculative hedging; use definitive logical connectors ("this implies", "it follows", "the framework necessitates")
   - Distinguish between derivation, ansatz, and conjecture
   - Focus on scaling relations, symmetry arguments, and thermodynamic constraints

STRUCTURE YOUR RESPONSE AS:
1. Individual Anomaly Discussions (one subsection per non-indeterminate anomaly)
2. Collective Patterns and Physical Implications
3. Model Comparison Analysis
4. Validation Status and Recommended Next Steps

Be rigorous, precise, and ensure every non-indeterminate anomaly receives individual discussion.
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
                    "content": "You are a senior theoretical physicist analyzing cosmological data."
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
                if attempt < max_retries - 1:
                    wait_time = 2 ** attempt
                    self.logger.info(f"Retrying after {wait_time}s...")
                    time.sleep(wait_time)
                else:
                    raise
        
        # Should not reach here, but handle it anyway
        raise last_exception or Exception("Grok API call failed for unknown reason")

