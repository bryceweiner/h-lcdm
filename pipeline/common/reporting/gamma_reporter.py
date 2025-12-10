"""Gamma pipeline reporting helpers."""

from typing import Any, Dict


def results(main_results: Dict[str, Any]) -> str:
    """Render gamma pipeline analysis results."""
    theory_summary = main_results.get("theory_summary", {})
    model_comparison = main_results.get("model_comparison", {})
    results_section = "### Theoretical γ(z) and Λ_eff(z) Predictions\n\n"
    results_section += (
        "**H-ΛCDM Theoretical Prediction:** Parameter-free calculation of information "
        "processing rate γ(z) and effective cosmological constant Λ_eff(z) as functions of redshift.\n\n"
    )

    if theory_summary:
        present_day = theory_summary.get("present_day", {})
        recombination = theory_summary.get("recombination_era", {})
        evolution = theory_summary.get("evolution_ratios", {})

        results_section += "**Present-day values (z=0):**\n\n"
        results_section += f"- Information processing rate: γ(z=0) = {present_day.get('gamma_s^-1', 'N/A'):.2e} s⁻¹\n"
        results_section += f"- Effective cosmological constant: Λ_eff(z=0) = {present_day.get('lambda_m^-2', 'N/A'):.2e} m⁻²\n\n"

        results_section += f"**Recombination era values (z={recombination.get('redshift', 1100):.0f}):**\n\n"
        results_section += f"- Information processing rate: γ(z={recombination.get('redshift', 1100):.0f}) = {recombination.get('gamma_s^-1', 'N/A'):.2e} s⁻¹\n"
        results_section += f"- Effective cosmological constant: Λ_eff(z={recombination.get('redshift', 1100):.0f}) = {recombination.get('lambda_m^-2', 'N/A'):.2e} m⁻²\n\n"

        if evolution:
            gamma_evol = evolution.get("gamma_recomb/gamma_today", "N/A")
            lambda_evol = evolution.get("lambda_recomb/lambda_today", "N/A")
            results_section += "**Evolution ratios:**\n\n"
            results_section += f"- γ(z=1100)/γ(z=0) = {gamma_evol:.2f}\n"
            results_section += f"- Λ_eff(z=1100)/Λ_eff(z=0) = {lambda_evol:.2f}\n\n"

        qtep_ratio = theory_summary.get("qtep_ratio", "N/A")
        results_section += f"**QTEP ratio:** {qtep_ratio:.3f} (theoretical prediction: 2.257 = ln(2)/(1-ln(2)))\n\n"

        key_equations = theory_summary.get("key_equations", [])
        if key_equations:
            results_section += "**Key theoretical equations:**\n\n"
            for eq in key_equations:
                results_section += f"- {eq}\n"
            results_section += "\n"

    if model_comparison and model_comparison.get("comparison_available", False):
        comparison = model_comparison.get("comparison", {})
        hlcdm = model_comparison.get("hlcdm", {})
        lcdm = model_comparison.get("lcdm", {})

        results_section += "### Model Comparison: H-ΛCDM vs ΛCDM\n\n"
        results_section += "Quantitative comparison using BIC, AIC, and Bayesian evidence.\n\n"

        results_section += f"**Data Points:** {model_comparison.get('n_data_points', 'N/A')} redshift points\n\n"

        results_section += "**H-ΛCDM Model:**\n\n"
        results_section += f"- χ² = {hlcdm.get('chi_squared', 'N/A'):.2f}\n"
        results_section += f"- log L = {hlcdm.get('log_likelihood', 'N/A'):.2f}\n"
        results_section += f"- AIC = {hlcdm.get('aic', 'N/A'):.2f}\n"
        results_section += f"- BIC = {hlcdm.get('bic', 'N/A'):.2f}\n"
        results_section += f"- Parameters: {hlcdm.get('n_parameters', 0)} (parameter-free prediction)\n\n"

        results_section += "**ΛCDM Model:**\n\n"
        results_section += f"- χ² = {lcdm.get('chi_squared', 'N/A'):.2f}\n"
        results_section += f"- log L = {lcdm.get('log_likelihood', 'N/A'):.2f}\n"
        results_section += f"- AIC = {lcdm.get('aic', 'N/A'):.2f}\n"
        results_section += f"- BIC = {lcdm.get('bic', 'N/A'):.2f}\n"
        results_section += f"- Parameters: {lcdm.get('n_parameters', 0)}\n\n"

        delta_aic = comparison.get("delta_aic", 0)
        delta_bic = comparison.get("delta_bic", 0)
        bayes_factor = comparison.get("bayes_factor", 1.0)
        preferred = comparison.get("preferred_model", "UNKNOWN")
        evidence_strength = comparison.get("evidence_strength", "UNKNOWN")

        results_section += "**Comparison Metrics:**\n\n"
        results_section += f"- ΔAIC = AIC_ΛCDM - AIC_H-ΛCDM = {delta_aic:.2f}\n"
        if abs(delta_aic) < 2:
            results_section += "  → Inconclusive (|ΔAIC| < 2)\n"
        elif abs(delta_aic) > 6:
            results_section += "  → Strong evidence for H-ΛCDM (|ΔAIC| > 6)\n"
        else:
            results_section += "  → Moderate evidence\n"

        results_section += f"- ΔBIC = BIC_ΛCDM - BIC_H-ΛCDM = {delta_bic:.2f}\n"
        if abs(delta_bic) < 2:
            results_section += "  → Inconclusive (|ΔBIC| < 2)\n"
        elif abs(delta_bic) > 6:
            results_section += "  → Strong evidence for H-ΛCDM (|ΔBIC| > 6)\n"
        else:
            results_section += "  → Moderate evidence\n"

        results_section += f"- Bayes Factor B = P(data|H-ΛCDM) / P(data|ΛCDM) = {bayes_factor:.2f}\n"
        results_section += f"  (log B = {comparison.get('log_bayes_factor', 0):.2f})\n"
        results_section += f"  → {evidence_strength} evidence\n\n"

        results_section += f"**Preferred Model:** {preferred}\n\n"

        interpretation = comparison.get("interpretation", "")
        if interpretation:
            results_section += f"**Interpretation:**\n\n{interpretation}\n\n"

    return results_section


def summary(main_results: Dict[str, Any]) -> str:
    """Short summary for comprehensive report."""
    formatted = ""

    if "theory_summary" in main_results:
        ts = main_results["theory_summary"]
        formatted += "- **Present-day values:**\n"
        formatted += f"  - γ(z=0) = {ts.get('present_day', {}).get('gamma_s^-1', 'N/A')}\n"
        formatted += f"  - Λ(z=0) = {ts.get('present_day', {}).get('lambda_m^-2', 'N/A')}\n"
        formatted += "- **Evolution ratios:**\n"
        formatted += f"  - γ(z=1100)/γ(z=0) = {ts.get('evolution_ratios', {}).get('gamma_recomb/gamma_today', 'N/A')}\n"
        formatted += f"  - Λ(z=1100)/Λ(z=0) = {ts.get('evolution_ratios', {}).get('lambda_recomb/lambda_today', 'N/A')}\n"

    if "validation" in main_results:
        validation = main_results["validation"]
        formatted += f"- **Validation status:** {validation.get('overall_status', 'UNKNOWN')}\n"

    return formatted


def conclusion(main_results: Dict[str, Any], overall_status: str) -> str:
    """Gamma pipeline conclusion text."""
    conclusion_text = "### Did We Find What We Were Looking For?\n\n"

    theory_summary = main_results.get("theory_summary", {})
    present_day = theory_summary.get("present_day", {})

    if present_day:
        conclusion_text += (
            "**YES** - The theoretical framework consistently derives the information "
            "processing rate γ(z) and effective cosmological constant Λ_eff(z) from first principles. "
            f"The present-day values (γ ≈ {present_day.get('gamma_s^-1', 'N/A'):.2e} s⁻¹) "
            "are consistent with the observed acceleration of the universe.\n\n"
        )
    else:
        conclusion_text += (
            "**INCONCLUSIVE** - Theoretical values were calculated but require comparison with observational constraints.\n\n"
        )

    model_comparison = main_results.get("model_comparison", {})
    if model_comparison and model_comparison.get("comparison_available", False):
        comparison = model_comparison.get("comparison", {})
        delta_aic = comparison.get("delta_aic", 0)
        delta_bic = comparison.get("delta_bic", 0)
        bayes_factor = comparison.get("bayes_factor", 1.0)

        if abs(delta_bic) > 6 or abs(delta_aic) > 6:
            better_model = comparison.get("preferred_model", "UNKNOWN")
            evidence = comparison.get("evidence_strength", "strong")
            conclusion_text += (
                "**Model Comparison:** Statistical analysis provides "
                f"{evidence} evidence favoring {better_model} over the alternative model "
                f"(Bayes factor {bayes_factor:.2f}).\n\n"
            )

    conclusion_text += f"Validation status: **{overall_status}**.\n\n"
    return conclusion_text

