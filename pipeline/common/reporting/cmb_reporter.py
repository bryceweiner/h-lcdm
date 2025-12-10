"""CMB pipeline reporting helpers."""

from typing import Any, Dict


def results(main_results: Dict[str, Any]) -> str:
    """Render CMB pipeline analysis results."""
    detection_summary = main_results.get("detection_summary", {})
    evidence_strength = detection_summary.get("evidence_strength", "UNKNOWN")

    null_test_result = main_results.get("null_test_result", {})
    null_p_value = null_test_result.get("p_value", 1.0)
    null_rejected = null_test_result.get("null_rejected", False)

    contradiction = evidence_strength in ["STRONG", "VERY_STRONG"] and not null_rejected and null_p_value > 0.05

    results_section = "### Did We Find What We Were Looking For?\n\n"
    if contradiction:
        results_section += (
            f"**MORE ANALYSIS REQUIRED** - Detection methods claim {evidence_strength} evidence for H-ΛCDM signatures, "
            f"but null hypothesis test shows no signal (p = {null_p_value:.3f}). This contradiction requires further investigation.\n\n"
        )
    elif evidence_strength in ["STRONG", "VERY_STRONG"] and null_rejected:
        results_section += (
            f"**YES** - Strong evidence ({evidence_strength}) for H-ΛCDM signatures (phase transitions, non-Gaussianity, "
            f"E8 patterns) in CMB E-mode data, confirmed by null hypothesis rejection.\n\n"
        )
    elif evidence_strength == "MODERATE":
        results_section += "**PARTIAL** - Moderate evidence for H-ΛCDM signatures in CMB data, requiring further investigation.\n\n"
    elif evidence_strength in ["STRONG", "VERY_STRONG"] and not null_rejected:
        results_section += (
            f"**NO** - Detection methods suggest signal but null hypothesis test shows consistency with ΛCDM "
            f"(p = {null_p_value:.3f}). No robust evidence for H-ΛCDM signatures.\n\n"
        )
    else:
        results_section += f"**NO** - Insufficient evidence ({evidence_strength}) for H-ΛCDM signatures in CMB E-mode data.\n\n"

    results_section += (
        "Multiple analysis methods were applied to search for phase transitions, non-Gaussianity, and E8×E8 signatures. "
    )

    analysis_methods = main_results.get("analysis_methods", {})
    if analysis_methods:
        results_section += "**Detailed Test Results:**\n\n"
        for method, result in analysis_methods.items():
            score = result.get("score", 0)
            significance = result.get("significance", 0)
            results_section += f"- **{method.title()}:** Score = {score:.2f}, Significance = {significance:.2f}σ\n"
        results_section += "\n"

    return results_section


def summary(main_results: Dict[str, Any]) -> str:
    """Short summary for comprehensive report."""
    formatted = ""
    if "detection_summary" in main_results:
        summary_data = main_results["detection_summary"]
        formatted += f"- **Evidence strength:** {summary_data.get('evidence_strength', 'N/A')}\n"
        formatted += f"- **Detection score:** {summary_data.get('detection_score', 0):.2f}\n"
        formatted += f"- **Methods used:** {len(main_results.get('analysis_methods', {}))}\n"
    return formatted


def conclusion(main_results: Dict[str, Any], overall_status: str) -> str:
    """CMB pipeline conclusion text."""
    detection_summary = main_results.get("detection_summary", {})
    evidence_strength = detection_summary.get("evidence_strength", "UNKNOWN")

    null_test_result = main_results.get("null_test_result", {})
    null_p_value = null_test_result.get("p_value", 1.0)
    null_rejected = null_test_result.get("null_rejected", False)

    contradiction = evidence_strength in ["STRONG", "VERY_STRONG"] and not null_rejected and null_p_value > 0.05

    conclusion_text = "### Did We Find What We Were Looking For?\n\n"
    if contradiction:
        conclusion_text += (
            f"**MORE ANALYSIS REQUIRED** - Detection methods claim {evidence_strength} evidence for H-ΛCDM signatures, "
            f"but null hypothesis test shows no signal (p = {null_p_value:.3f}). This contradiction requires further investigation.\n\n"
        )
    elif evidence_strength in ["STRONG", "VERY_STRONG"] and null_rejected:
        conclusion_text += (
            f"**YES** - Strong evidence ({evidence_strength}) for H-ΛCDM signatures (phase transitions, non-Gaussianity, "
            f"E8 patterns) in CMB E-mode data, confirmed by null hypothesis rejection.\n\n"
        )
    elif evidence_strength == "MODERATE":
        conclusion_text += "**PARTIAL** - Moderate evidence for H-ΛCDM signatures in CMB data, requiring further investigation.\n\n"
    elif evidence_strength in ["STRONG", "VERY_STRONG"] and not null_rejected:
        conclusion_text += (
            f"**NO** - Detection methods suggest signal but null hypothesis test shows consistency with ΛCDM "
            f"(p = {null_p_value:.3f}). No robust evidence for H-ΛCDM signatures.\n\n"
        )
    else:
        conclusion_text += f"**NO** - Insufficient evidence ({evidence_strength}) for H-ΛCDM signatures in CMB E-mode data.\n\n"

    conclusion_text += (
        "Multiple analysis methods were applied to search for phase transitions, non-Gaussianity, and E8×E8 signatures. "
        f"Validation status: **{overall_status}**.\n\n"
    )
    return conclusion_text

