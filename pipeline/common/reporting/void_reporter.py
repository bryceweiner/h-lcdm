"""Void pipeline reporting helpers."""

from typing import Any, Dict
import numpy as np


def _construct_void_grok_prompt(main_results: Dict[str, Any],
                                 basic_val: Dict[str, Any],
                                 extended_val: Dict[str, Any]) -> str:
    """
    Construct Grok prompt for void analysis interpretation.
    
    Design Principles:
    1. Provide ALL numerical results explicitly
    2. Define physical meaning of clustering coefficient
    3. Specify null hypothesis (ΛCDM) numerically
    4. Request interpretation based ONLY on provided data
    5. Avoid imposing theoretical bias - let data speak
    """
    
    # Extract clustering analysis
    clustering_analysis = main_results.get("clustering_analysis", {}) or {}
    void_data = main_results.get("void_data", {}) or {}
    processing_costs = clustering_analysis.get("processing_costs", {}) or {}
    clustering_comparison = clustering_analysis.get("clustering_comparison", {}) or {}
    model_comparison = clustering_analysis.get("model_comparison", {}) or {}
    
    # Core observables
    observed_cc = clustering_analysis.get("observed_clustering_coefficient", None)
    observed_std = clustering_analysis.get("observed_clustering_std", None)
    n_voids = void_data.get("total_voids", 0)
    n_edges = void_data.get("network_analysis", {}).get("n_edges", 0)
    mean_degree = void_data.get("network_analysis", {}).get("mean_degree", 0)
    linking_length = void_data.get("network_analysis", {}).get("linking_length", 0)
    
    # Theoretical comparison values
    eta_natural = 0.4430  # (1 - ln(2)) / ln(2) ≈ 0.443
    c_e8 = 0.78125  # 25/32, E8×E8 pure substrate
    c_lcdm = 0.0  # ΛCDM predicts isotropic/random (no clustering)
    
    # Comparison statistics
    eta_data = clustering_comparison.get("thermodynamic_efficiency", {}) or {}
    lcdm_data = clustering_comparison.get("lcdm", {}) or {}
    eta_sigma = eta_data.get("sigma", None)
    lcdm_sigma = lcdm_data.get("sigma", None)
    
    # Chi-squared from model comparison
    overall_scores = model_comparison.get("overall_scores", {}) or {}
    hlcdm_chi2 = overall_scores.get("hlcdm_combined", None)
    lcdm_chi2 = overall_scores.get("lcmd_connectivity_only", None)
    best_model = model_comparison.get("best_model", "N/A")
    
    # Processing costs
    baryonic_cost = processing_costs.get("baryonic_precipitation", {}).get("value", None)
    causal_diamond_cost = processing_costs.get("causal_diamond_structure", {}).get("value", None)
    
    # Extract validation results
    bootstrap = extended_val.get("bootstrap", {}) if extended_val else {}
    jackknife = extended_val.get("jackknife", {}) if extended_val else {}
    null_hypothesis = extended_val.get("null_hypothesis", {}) if extended_val else {}
    cross_val = extended_val.get("cross_validation", {}) if extended_val else {}
    bayesian = extended_val.get("bayesian_model_comparison", {}) if extended_val else {}
    
    # Bootstrap results
    bootstrap_mean = bootstrap.get("bootstrap_mean", None)
    bootstrap_std = bootstrap.get("bootstrap_std", None)
    bootstrap_z = bootstrap.get("z_score", None)
    bootstrap_passed = bootstrap.get("passed", None)
    
    # Null hypothesis results
    null_mean = null_hypothesis.get("null_mean", None)
    null_std = null_hypothesis.get("null_std", None)
    null_z = null_hypothesis.get("z_score", None)
    null_p = null_hypothesis.get("p_value", None)
    null_passed = null_hypothesis.get("passed", None)
    
    # Bayesian model comparison
    bayes_best = bayesian.get("best_model", None)
    bayes_factor = None
    if bayesian.get("models"):
        eta_model = bayesian.get("models", {}).get("thermodynamic_efficiency", {})
        bayes_factor = eta_model.get("bayes_factor_vs_lcdm", None)
    
    prompt = f"""
You are analyzing cosmic void network clustering coefficient data from multi-survey astronomical observations.

## CRITICAL: Interpretation Rules

1. **Report only what the data shows** - do not impose theoretical preferences
2. **State numerical results explicitly** before any interpretation
3. **Acknowledge statistical limitations** - uncertainty, sample size, systematics
4. **Distinguish between consistency and confirmation** - matching a prediction within error is consistency, not proof
5. **Consider alternative explanations** - systematic effects, selection biases, survey-specific artifacts

---

## PHYSICAL BACKGROUND

### What is Being Measured

The **clustering coefficient** (C) of a void network measures how interconnected neighboring voids are:
- C = 0: No clustering (voids connected randomly, like a random graph)
- C = 1: Maximum clustering (every void's neighbors are also connected to each other)

The void network is constructed by:
1. Taking void center positions in comoving coordinates
2. Connecting voids within a linking length (based on void size distribution)
3. Computing the network's global clustering coefficient

### Physical Interpretations

**ΛCDM Prediction (C ≈ 0):**
Standard ΛCDM predicts voids form from Gaussian random field initial conditions. The resulting void network should be approximately isotropic with low clustering (C → 0 for large samples).

**H-ΛCDM Prediction (C ≈ η_natural ≈ 0.443):**
H-ΛCDM predicts void clustering reflects the thermodynamic efficiency ratio:
- η_natural = (1 - ln 2) / ln 2 ≈ 0.4430
- This ratio emerges from entropy mechanics as the minimum thermodynamic cost of information processing

**E8×E8 Pure Substrate (C_E8 ≈ 0.781):**
The pure computational capacity without thermodynamic processing:
- C_E8 = 25/32 ≈ 0.78125
- Difference from η_natural represents thermodynamic "cost" of baryonic matter processing

---

## DATA (Analyze ONLY what's provided)

### Network Statistics

- **Total voids analyzed:** {n_voids:,}
- **Network edges:** {n_edges:,}
- **Mean degree:** {mean_degree:.2f}
- **Linking length:** {linking_length:.2f} Mpc

### Primary Observable

- **Observed clustering coefficient:** C_obs = {f"{observed_cc:.4f}" if observed_cc is not None else "N/A"} ± {f"{observed_std:.4f}" if observed_std is not None else "N/A"}

### Comparison to Theoretical Predictions

| Model | Predicted C | Observed - Predicted | Significance |
|-------|-------------|---------------------|--------------|
| ΛCDM (random) | {c_lcdm:.2f} | {f"{observed_cc - c_lcdm:.4f}" if observed_cc is not None else "N/A"} | {f"{lcdm_sigma:.1f}σ" if lcdm_sigma is not None else "N/A"} |
| H-ΛCDM (η_natural) | {eta_natural:.4f} | {f"{observed_cc - eta_natural:.4f}" if observed_cc is not None else "N/A"} | {f"{eta_sigma:.1f}σ" if eta_sigma is not None else "N/A"} |
| E8×E8 substrate | {c_e8:.4f} | {f"{observed_cc - c_e8:.4f}" if observed_cc is not None else "N/A"} | N/A |

### Model Comparison (χ² Analysis)

- **H-ΛCDM combined χ²:** {f"{hlcdm_chi2:.3f}" if hlcdm_chi2 is not None else "N/A"}
- **ΛCDM χ²:** {f"{lcdm_chi2:.3f}" if lcdm_chi2 is not None else "N/A"}
- **Δχ²:** {f"{abs(hlcdm_chi2 - lcdm_chi2):.3f}" if hlcdm_chi2 is not None and lcdm_chi2 is not None else "N/A"}
- **Best-fit model:** {best_model}

### Processing Cost Analysis

- **Baryonic precipitation cost (C_E8 - C_obs):** {f"{baryonic_cost:.4f}" if baryonic_cost is not None else "N/A"}
- **Causal diamond structure cost (C_E8 - η_natural):** {f"{causal_diamond_cost:.4f}" if causal_diamond_cost is not None else "N/A"}

---

## VALIDATION RESULTS

### Bootstrap Validation (10,000 iterations)

- **Status:** {"PASSED" if bootstrap_passed else "FAILED" if bootstrap_passed is not None else "N/A"}
- **Bootstrap mean:** {f"{bootstrap_mean:.4f}" if bootstrap_mean is not None else "N/A"} ± {f"{bootstrap_std:.4f}" if bootstrap_std is not None else "N/A"}
- **z-score (stability):** {f"{bootstrap_z:.2f}σ" if bootstrap_z is not None else "N/A"}

### Null Hypothesis Testing (1,000 random networks)

- **Status:** {"PASSED" if null_passed else "FAILED" if null_passed is not None else "N/A"}
- **Null hypothesis mean:** {f"{null_mean:.4f}" if null_mean is not None else "N/A"} ± {f"{null_std:.4f}" if null_std is not None else "N/A"}
- **z-score vs null:** {f"{null_z:.2f}σ" if null_z is not None else "N/A"}
- **p-value:** {f"{null_p:.4e}" if null_p is not None else "N/A"}

### Bayesian Model Comparison

- **Best model (BIC):** {bayes_best if bayes_best else "N/A"}
- **Bayes factor (η_natural vs ΛCDM):** {f"{bayes_factor:.2e}" if bayes_factor is not None else "N/A"}

---

## YOUR TASK

Based EXCLUSIVELY on the numerical results above, provide:

### 1. Data Summary (100 words)

State the primary observable (C_obs) and its uncertainty. Report the network size and linking length. This is purely descriptive - no interpretation yet.

### 2. Statistical Significance Assessment (150 words)

**Address these questions:**
- How many σ is C_obs from ΛCDM prediction (C = 0)?
- How many σ is C_obs from H-ΛCDM prediction (η_natural = 0.443)?
- What does the null hypothesis p-value indicate?
- Is the bootstrap distribution stable (low z-score)?

**Do NOT:**
- Claim "strong evidence" unless significance exceeds 3σ
- Ignore that consistency is not confirmation

### 3. Model Comparison (150 words)

**Address these questions:**
- Which model has lower χ²? By how much?
- What does the Bayes factor indicate? (>3 = moderate, >10 = strong, >100 = decisive)
- Is there model selection bias from using the same data for fitting and comparison?

**Critical caveat:** Lower χ² indicates better fit but does not prove mechanism.

### 4. Alternative Explanations (100 words)

Consider:
- Could survey selection effects create artificial clustering?
- Are different void catalogs (SDSS DR7, DESI) consistent?
- Could the linking length choice bias the result?
- What systematic uncertainties are not captured in the error bar?

### 5. Scientific Verdict (50 words)

One of:
- "The data **strongly favors** H-ΛCDM over ΛCDM" (requires: Δχ² > 6, significance > 3σ, cross-validation passed)
- "The data **moderately favors** H-ΛCDM over ΛCDM" (requires: Δχ² > 2, significance > 2σ)
- "The data **is consistent with** H-ΛCDM but does not exclude ΛCDM" (if significance < 2σ)
- "The data **is inconsistent with** both predictions" (if neither model fits)

---

## TONE AND STYLE

- Empirical, dispassionate, appropriate for high-impact letters journal
- Third person throughout ("The analysis reveals...", "The observed coefficient...")
- Definitive logical connectors where warranted ("this implies", "it follows")
- Appropriate hedging for statistical limitations ("consistent with", "suggests", "does not exclude")
- NO superlatives ("remarkable", "striking") unless statistically justified
"""
    
    return prompt


def grok_results(main_results: Dict[str, Any],
                 basic_val: Dict[str, Any],
                 extended_val: Dict[str, Any],
                 grok_client) -> str:
    """
    Generate Grok interpretation for void analysis.
    
    Parameters:
        main_results: Pipeline results dictionary
        basic_val: Basic validation results
        extended_val: Extended validation results
        grok_client: GrokAnalysisClient instance (optional)
        
    Returns:
        str: Grok interpretation + raw data
    """
    prompt = _construct_void_grok_prompt(main_results, basic_val, extended_val)
    
    grok_interpretation = ""
    if grok_client:
        try:
            grok_interpretation = grok_client.generate_custom_report(prompt)
        except Exception as e:
            grok_interpretation = f"Grok interpretation unavailable: {e}"
    else:
        grok_interpretation = "Grok interpretation unavailable (no Grok client)"
    
    # Generate raw data tables
    raw_data = _generate_void_raw_data_tables(main_results, basic_val, extended_val)
    
    return f"""## Grok Scientific Interpretation

{grok_interpretation}

---

## Raw Data Tables

{raw_data}
"""


def _generate_void_raw_data_tables(main_results: Dict[str, Any],
                                    basic_val: Dict[str, Any],
                                    extended_val: Dict[str, Any]) -> str:
    """Generate raw data tables for reproducibility."""
    
    clustering_analysis = main_results.get("clustering_analysis", {}) or {}
    void_data = main_results.get("void_data", {}) or {}
    network = void_data.get("network_analysis", {}) or {}
    processing_costs = clustering_analysis.get("processing_costs", {}) or {}
    
    tables = []
    
    # Network statistics table
    tables.append("### Network Statistics\n")
    tables.append("| Parameter | Value |")
    tables.append("|-----------|-------|")
    tables.append(f"| Total voids | {void_data.get('total_voids', 'N/A'):,} |")
    tables.append(f"| Network edges | {network.get('n_edges', 'N/A'):,} |")
    tables.append(f"| Mean degree | {network.get('mean_degree', 0):.2f} |")
    tables.append(f"| Linking length (Mpc) | {network.get('linking_length', 0):.2f} |")
    tables.append(f"| Observed C | {clustering_analysis.get('observed_clustering_coefficient', 'N/A'):.4f} |")
    tables.append(f"| Observed σ | {clustering_analysis.get('observed_clustering_std', 'N/A'):.4f} |")
    tables.append("")
    
    # Survey breakdown
    survey_breakdown = void_data.get("survey_breakdown", {})
    if survey_breakdown:
        tables.append("### Survey Breakdown\n")
        tables.append("| Survey | Voids |")
        tables.append("|--------|-------|")
        for survey, count in survey_breakdown.items():
            tables.append(f"| {survey} | {count:,} |")
        tables.append("")
    
    # Validation summary
    if extended_val:
        tables.append("### Extended Validation Summary\n")
        tables.append("| Test | Status | Key Metric |")
        tables.append("|------|--------|------------|")
        
        bootstrap = extended_val.get("bootstrap", {})
        if bootstrap:
            status = "PASSED" if bootstrap.get("passed") else "FAILED"
            z = bootstrap.get("z_score", "N/A")
            z_str = f"{z:.2f}σ" if isinstance(z, (int, float)) else str(z)
            tables.append(f"| Bootstrap (10k) | {status} | z = {z_str} |")
        
        jackknife = extended_val.get("jackknife", {})
        if jackknife:
            status = "PASSED" if jackknife.get("passed") else "FAILED"
            bias = jackknife.get("jackknife_bias", "N/A")
            bias_str = f"{bias:.6f}" if isinstance(bias, (int, float)) else str(bias)
            tables.append(f"| Jackknife (100) | {status} | bias = {bias_str} |")
        
        null_hyp = extended_val.get("null_hypothesis", {})
        if null_hyp:
            status = "PASSED" if null_hyp.get("passed") else "FAILED"
            p = null_hyp.get("p_value", "N/A")
            p_str = f"{p:.4e}" if isinstance(p, (int, float)) else str(p)
            tables.append(f"| Null Hypothesis (1k) | {status} | p = {p_str} |")
        
        tables.append("")
    
    return "\n".join(tables)


def results(main_results: Dict[str, Any]) -> str:
    """Render void pipeline analysis results."""
    clustering_analysis = main_results.get("clustering_analysis", {}) or {}
    analysis_summary = main_results.get("analysis_summary", {}) or {}
    processing_costs = clustering_analysis.get("processing_costs", {}) if clustering_analysis else {}
    clustering_comparison = clustering_analysis.get("clustering_comparison", {}) if clustering_analysis else {}

    observed_cc = clustering_analysis.get("observed_clustering_coefficient", 0.0) if clustering_analysis else 0.0
    observed_std = clustering_analysis.get("observed_clustering_std", 0.03) if clustering_analysis else 0.03
    eta_data = clustering_comparison.get("thermodynamic_efficiency", {}) if clustering_comparison else {}
    eta_sigma = eta_data.get("sigma", np.inf) if eta_data else np.inf

    results_section = "### Statistical Analysis Results\n\n"
    results_section += f"**Observed Clustering Coefficient:** C_obs = {observed_cc:.4f} ± {observed_std:.4f}\n\n"
    results_section += "**Comparison with H-ΛCDM Thermodynamic Ratio (η_natural = 0.4430):**\n"
    results_section += f"- Statistical significance: {eta_sigma:.2f}σ\n\n"

    model_comparison = clustering_analysis.get("model_comparison", {}) if clustering_analysis else {}
    baryonic_chi2 = model_comparison.get("baryonic_costs", {}).get("chi2_observed_vs_hlcdm", 0.0)
    hlcdm_combined_chi2 = model_comparison.get("overall_scores", {}).get("hlcdm_combined", 0.0)
    lcdm_combined_chi2 = model_comparison.get("overall_scores", {}).get("lcmd_connectivity_only", 0.0)
    results_section += "**Model Comparison (Combined χ²):**\n"
    results_section += f"- H-ΛCDM: χ² = {hlcdm_combined_chi2:.3f}\n"
    results_section += f"- ΛCDM: χ² = {lcdm_combined_chi2:.3f}\n"
    results_section += f"- Δχ² = {abs(hlcdm_combined_chi2 - lcdm_combined_chi2):.3f}\n\n"

    if processing_costs:
        baryonic_cost = processing_costs.get("baryonic_precipitation", {}).get("value", None)
        causal_diamond_cost = processing_costs.get("causal_diamond_structure", {}).get("value", None)
        if baryonic_cost is not None and causal_diamond_cost is not None:
            results_section += "**Processing Cost Analysis:**\n\n"
            results_section += f"- Processing cost to precipitate baryonic matter: ΔC = {baryonic_cost:.4f}\n"
            results_section += f"- Thermodynamic cost of information processing system (without baryonic matter): ΔC = {causal_diamond_cost:.4f}\n\n"

    if analysis_summary:
        total_voids = analysis_summary.get("total_voids_analyzed", 0)
        conclusion = analysis_summary.get("overall_conclusion", "N/A")
        results_section += f"Voids analyzed: {total_voids:,}; conclusion: {conclusion}\n\n"

    return results_section


def summary(main_results: Dict[str, Any]) -> str:
    """Short summary for comprehensive report."""
    formatted = ""
    if "analysis_summary" in main_results:
        summary_data = main_results["analysis_summary"]
        formatted += f"- **Voids analyzed:** {summary_data.get('total_voids_analyzed', 0)}\n"
        formatted += f"- **Conclusion:** {summary_data.get('overall_conclusion', 'N/A')}\n"
    return formatted


def validation(basic_val: Dict[str, Any], extended_val: Dict[str, Any]) -> str:
    """Void-specific validation including bootstrap and jackknife."""
    validation = ""
    if basic_val:
        overall_status = basic_val.get("overall_status", "UNKNOWN")
        validation += "### Basic Validation\n\n"
        validation += f"**Overall Status:** {overall_status}\n\n"

    if extended_val:
        validation += "### Extended Validation\n\n"
        ext_status = extended_val.get("overall_status", "UNKNOWN")
        validation += f"**Overall Status:** {ext_status}\n\n"

        bootstrap = extended_val.get("bootstrap", {})
        if isinstance(bootstrap, dict) and bootstrap.get("test") == "bootstrap_clustering_validation":
            validation += "#### Bootstrap Clustering Validation (10,000 iterations)\n\n"
            validation += f"**Status:** {'✓ PASSED' if bootstrap.get('passed', False) else '✗ FAILED'}\n\n"
            obs_cc = bootstrap.get("observed_clustering_coefficient", "N/A")
            bootstrap_mean = bootstrap.get("bootstrap_mean", "N/A")
            bootstrap_std = bootstrap.get("bootstrap_std", "N/A")
            z_score = bootstrap.get("z_score", "N/A")
            validation += f"- Observed clustering coefficient: {obs_cc:.4f}\n" if isinstance(obs_cc, (int, float)) else f"- Observed clustering coefficient: {obs_cc}\n"
            validation += f"- Bootstrap mean: {bootstrap_mean:.4f} ± {bootstrap_std:.4f}\n" if isinstance(bootstrap_mean, (int, float)) and isinstance(bootstrap_std, (int, float)) else f"- Bootstrap mean: {bootstrap_mean} ± {bootstrap_std}\n"
            validation += f"- z-score (stability): {z_score:.2f}σ\n" if isinstance(z_score, (int, float)) else f"- z-score (stability): {z_score}σ\n"

            comparison = bootstrap.get("comparison_to_fundamental_values", {})
            if comparison:
                validation += "\n**Comparison to Fundamental Values:**\n"
                eta_comp = comparison.get("thermodynamic_efficiency", {})
                lcdm_comp = comparison.get("lcdm", {})
                if eta_comp:
                    eta_val = eta_comp.get("value", "N/A")
                    eta_sig = eta_comp.get("sigma", "N/A")
                    eta_str = f"{eta_val:.4f}" if isinstance(eta_val, (int, float)) else str(eta_val)
                    sig_str = f"{eta_sig:.1f}" if isinstance(eta_sig, (int, float)) else str(eta_sig)
                    validation += f"- Thermodynamic efficiency (η_natural = {eta_str}): {sig_str}σ, "
                    validation += f"{'within 95% CI' if eta_comp.get('within_ci_95', False) else 'outside 95% CI'}\n"
                if lcdm_comp:
                    lcdm_val = lcdm_comp.get("value", "N/A")
                    lcdm_sig = lcdm_comp.get("sigma", "N/A")
                    lcdm_str = f"{lcdm_val:.2f}" if isinstance(lcdm_val, (int, float)) else str(lcdm_val)
                    sig_str = f"{lcdm_sig:.1f}" if isinstance(lcdm_sig, (int, float)) else str(lcdm_sig)
                    validation += f"- ΛCDM (C = {lcdm_str}): {sig_str}σ, "
                    validation += f"{'within 95% CI' if lcdm_comp.get('within_ci_95', False) else 'outside 95% CI'}\n"
            validation += "\n"

        jackknife = extended_val.get("jackknife", {})
        if isinstance(jackknife, dict) and jackknife.get("test") == "jackknife_clustering_validation":
            validation += "#### Jackknife Clustering Validation (100 subsamples)\n\n"
            validation += f"**Status:** {'✓ PASSED' if jackknife.get('passed', False) else '✗ FAILED'}\n\n"
            orig_cc = jackknife.get("original_clustering_coefficient", "N/A")
            validation += f"- Original C: {orig_cc:.4f}\n" if isinstance(orig_cc, (int, float)) else f"- Original C: {orig_cc}\n"
            jk_bias = jackknife.get("jackknife_bias", "N/A")
            jk_std = jackknife.get("jackknife_std", "N/A")
            validation += f"- Jackknife bias: {jk_bias:.6f}\n" if isinstance(jk_bias, (int, float)) else f"- Jackknife bias: {jk_bias}\n"
            validation += f"- Jackknife std error: {jk_std:.6f}\n" if isinstance(jk_std, (int, float)) else f"- Jackknife std error: {jk_std}\n"
            validation += "\n"

    return validation or "No validation results available.\n\n"


def conclusion(main_results: Dict[str, Any], overall_status: str) -> str:
    """Void pipeline conclusion text."""
    clustering_analysis = main_results.get("clustering_analysis", {})
    clustering_comparison = clustering_analysis.get("clustering_comparison", {}) if clustering_analysis else {}
    processing_costs = clustering_analysis.get("processing_costs", {}) if clustering_analysis else {}

    eta_data = clustering_comparison.get("thermodynamic_efficiency", {}) if clustering_comparison else {}
    eta_sigma = eta_data.get("sigma", np.inf) if eta_data else np.inf
    matches_eta = clustering_analysis.get("matches_thermodynamic_efficiency", False) if clustering_analysis else False

    conclusion_text = "### Statistical Analysis Results\n\n"
    observed_cc = clustering_analysis.get("observed_clustering_coefficient", 0.0) if clustering_analysis else 0.0
    observed_std = clustering_analysis.get("observed_clustering_std", 0.03) if clustering_analysis else 0.03
    model_comparison = clustering_analysis.get("model_comparison", {}) if clustering_analysis else {}

    baryonic_chi2 = model_comparison.get("baryonic_costs", {}).get("chi2_observed_vs_hlcdm", 0.0)
    hlcdm_combined_chi2 = model_comparison.get("overall_scores", {}).get("hlcdm_combined", 0.0)
    lcdm_combined_chi2 = model_comparison.get("overall_scores", {}).get("lcmd_connectivity_only", 0.0)

    try:
        from scipy import stats

        p_value_eta = 1.0 - stats.chi2.cdf(baryonic_chi2, df=1) if baryonic_chi2 > 0 else None
        p_value_hlcdm = 1.0 - stats.chi2.cdf(hlcdm_combined_chi2, df=2) if hlcdm_combined_chi2 > 0 else None
        p_value_lcdm = 1.0 - stats.chi2.cdf(lcdm_combined_chi2, df=1) if lcdm_combined_chi2 > 0 else None
    except Exception:
        p_value_eta = None
        p_value_hlcdm = None
        p_value_lcdm = None

    conclusion_text += f"**Observed Clustering Coefficient:** C_obs = {observed_cc:.4f} ± {observed_std:.4f}\n\n"
    conclusion_text += f"**Comparison with H-ΛCDM Thermodynamic Ratio (η_natural = 0.4430):**\n"
    conclusion_text += f"- Difference: {observed_cc - 0.4430:.4f}\n"
    conclusion_text += f"- Statistical significance: {eta_sigma:.2f}σ\n"
    conclusion_text += f"- χ² = {baryonic_chi2:.3f}"
    if p_value_eta is not None:
        conclusion_text += f", p = {p_value_eta:.4f}\n\n"
    else:
        conclusion_text += "\n\n"

    conclusion_text += "**Model Comparison (Combined χ²):**\n"
    conclusion_text += f"- H-ΛCDM: χ² = {hlcdm_combined_chi2:.3f}"
    if p_value_hlcdm is not None:
        conclusion_text += f", p = {p_value_hlcdm:.4f}\n"
    else:
        conclusion_text += "\n"
    conclusion_text += f"- ΛCDM: χ² = {lcdm_combined_chi2:.3f}"
    if p_value_lcdm is not None:
        conclusion_text += f", p = {p_value_lcdm:.4f}\n"
    else:
        conclusion_text += "\n"
    conclusion_text += f"- Δχ² = {abs(hlcdm_combined_chi2 - lcdm_combined_chi2):.3f}\n\n"

    if processing_costs:
        baryonic_cost = processing_costs.get("baryonic_precipitation", {}).get("value", None)
        causal_diamond_cost = processing_costs.get("causal_diamond_structure", {}).get("value", None)
        if baryonic_cost is not None and causal_diamond_cost is not None:
            conclusion_text += "**Processing Cost Analysis:**\n\n"
            conclusion_text += f"- Processing cost to precipitate baryonic matter: ΔC = {baryonic_cost:.4f}\n"
            conclusion_text += f"- Thermodynamic cost of information processing system (without baryonic matter): ΔC = {causal_diamond_cost:.4f}\n\n"

            conclusion_text += (
                "The difference between E8×E8 pure substrate (C_E8 = 25/32 ≈ 0.781, pure computational capacity) "
                "and thermodynamic ratio (η_natural) represents the thermodynamic cost of the information processing "
                "system (causal diamond/light cone structure) without baryonic matter.\n\n"
            )

    model_comp = main_results.get("validation", {}).get("model_comparison", {}) if isinstance(main_results.get("validation", {}), dict) else {}
    if model_comp.get("test") == "clustering_model_comparison":
        best_model = model_comp.get("best_model", "N/A")
        models = model_comp.get("models", {})
        thermodynamic_model = models.get("thermodynamic_efficiency", {})
        delta_bic = thermodynamic_model.get("delta_bic", 0)
        bayes_factor = thermodynamic_model.get("bayes_factor_vs_lcdm", 1.0)
        conclusion_text += (
            f"**Model Comparison:** Bayesian analysis favors {best_model} model. "
            f"Thermodynamic efficiency has ΔBIC = {delta_bic:.1f} and Bayes factor = {bayes_factor:.2e} relative to ΛCDM.\n\n"
        )

    void_data = main_results.get("void_data", {})
    total_voids = void_data.get("total_voids", 0) if void_data else 0
    if isinstance(observed_cc, (int, float)):
        conclusion_text += f"Analyzed {total_voids:,} cosmic voids with observed clustering coefficient C_obs = {observed_cc:.3f}. "
    else:
        conclusion_text += f"Analyzed {total_voids:,} cosmic voids with observed clustering coefficient C_obs = {observed_cc}. "
    conclusion_text += f"Validation status: **{overall_status}**.\n\n"
    return conclusion_text

