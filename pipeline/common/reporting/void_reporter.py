"""Void pipeline reporting helpers."""

from typing import Any, Dict
import numpy as np


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

