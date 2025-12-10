"""HLCDM pipeline reporting helpers."""

from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional


def generate_individual_reports(reporter, results: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate individual reports for each HLCDM test."""
    main_results = results.get("main", {})
    test_results = main_results.get("test_results", {})
    tests_run = main_results.get("tests_run", [])

    generated_reports = []
    if test_results:
        for test_name, test_result in test_results.items():
            if isinstance(test_result, dict) and "error" not in test_result:
                report_path = _generate_individual_hlcdm_test_report(reporter.reports_dir, test_name, test_result, metadata)
                if report_path:
                    generated_reports.append(report_path)

    if generated_reports:
        summary = f"Generated {len(generated_reports)} individual HLCDM test reports:\n"
        for report in generated_reports:
            summary += f"  - {report}\n"
        return summary
    return "No HLCDM test reports generated."


def conclusion(main_results: Dict[str, Any], overall_status: str) -> str:
    """HLCDM pipeline conclusion text."""
    synthesis = main_results.get("synthesis", {})
    strength = synthesis.get("strength_category", "UNKNOWN") if synthesis else "UNKNOWN"

    conclusion_text = "### Did We Find What We Were Looking For?\n\n"
    if strength in ["STRONG", "VERY_STRONG"]:
        conclusion_text += f"**YES** - Strong evidence ({strength}) from extension tests supports H-ΛCDM predictions.\n\n"
    elif strength == "MODERATE":
        conclusion_text += f"**PARTIAL** - Moderate evidence from extension tests, promising but requires further validation.\n\n"
    else:
        conclusion_text += f"**NO** - Limited evidence ({strength}) from extension tests for H-ΛCDM predictions.\n\n"

    conclusion_text += (
        "Multiple extension tests (JWST, Lyman-alpha, FRB, E8 Chiral, Temporal Cascade) were evaluated. "
        f"Validation status: **{overall_status}**.\n\n"
    )
    return conclusion_text


def _generate_individual_hlcdm_test_report(
    reports_dir: Path, test_name: str, test_result: Dict[str, Any], metadata: Optional[Dict[str, Any]] = None
) -> Optional[str]:
    """Generate individual comprehensive report for a single HLCDM test."""
    try:
        report_filename = f"{test_name}_analysis_report.md"
        report_path = reports_dir / report_filename

        report_content = _generate_hlcdm_test_header(test_name, metadata)
        report_content += _generate_hlcdm_test_results(test_name, test_result)
        report_content += _generate_hlcdm_test_validation(test_name, test_result)
        report_content += _generate_hlcdm_test_conclusion(test_name, test_result)

        with open(report_path, "w") as f:
            f.write(report_content)

        return str(report_path)
    except Exception as exc:  # noqa: BLE001
        print(f"Error generating individual HLCDM test report for {test_name}: {exc}")
        return None


def _generate_hlcdm_test_header(test_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
    """Generate header for individual HLCDM test report."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    test_descriptions = {
        "jwst": {
            "full_name": "JWST Early Galaxy Formation Analysis",
            "question": "Do JWST observations of early galaxies support H-ΛCDM information saturation limits?",
            "looking_for": "Evidence that halo masses are limited by information processing constraints at high redshift",
            "prediction": "H-ΛCDM predicts maximum halo masses decrease with redshift due to information saturation",
        },
        "lyman_alpha": {
            "full_name": "Lyman-α Forest Phase Transitions",
            "question": "Are there phase transitions in the Lyman-α forest consistent with H-ΛCDM predictions?",
            "looking_for": "Phase transitions at specific redshifts where information processing changes the intergalactic medium",
            "prediction": "H-ΛCDM predicts phase transitions in Lyman-α optical depth at specific redshifts",
        },
        "frb": {
            "full_name": "Fast Radio Burst Timing Analysis",
            "question": "Do FRB timing patterns show signatures of Little Bang information processing?",
            "looking_for": "Temporal delays and dispersion measure patterns consistent with information cascade effects",
            "prediction": "H-ΛCDM predicts FRB timing signatures from Little Bang information processing events",
        },
        "e8_ml": {
            "full_name": "E8×E8 Machine Learning Pattern Recognition",
            "question": "Can machine learning detect E8×E8 heterotic geometric patterns in cosmic data?",
            "looking_for": "Statistical signatures of E8×E8 geometry in large-scale structure and CMB data",
            "prediction": "H-ΛCDM predicts detectable E8×E8 patterns through heterotic string theory connections",
        },
        "e8_chiral": {
            "full_name": "E8×E8 Chiral Symmetry Analysis",
            "question": "Are there chiral symmetry signatures from E8×E8 heterotic string theory?",
            "looking_for": "Chiral asymmetry patterns and broken symmetries consistent with E8×E8 structure",
            "prediction": "H-ΛCDM predicts chiral signatures from E8×E8 heterotic string theory",
        },
        "temporal_cascade": {
            "full_name": "Temporal Cascade Information Processing",
            "question": "Do temporal scales show evidence of information processing hierarchies?",
            "looking_for": "Hierarchical temporal structures and entropy production patterns",
            "prediction": "H-ΛCDM predicts temporal cascades from information processing at different scales",
        },
    }

    desc = test_descriptions.get(
        test_name,
        {
            "full_name": f"{test_name.upper()} Analysis",
            "question": f"Does {test_name} data support H-ΛCDM predictions?",
            "looking_for": f"Evidence for H-ΛCDM signatures in {test_name} data",
            "prediction": f"H-ΛCDM predicts specific signatures in {test_name} observations",
        },
    )

    header = f"""# {desc['full_name']} Analysis Report

**Generated:** {timestamp}

**Test:** {test_name}

## Scientific Question

**What are we analyzing?** {desc['question']}

**What are we looking for?** {desc['looking_for']}

**H-ΛCDM Prediction:** {desc['prediction']}

"""

    if metadata:
        header += "### Analysis Parameters\n\n"
        for key, value in metadata.items():
            header += f"**{key.replace('_', ' ').title()}:** {value}\n"
        header += "\n"

    header += "---\n\n"
    return header


def _generate_hlcdm_test_results(test_name: str, test_result: Dict[str, Any]) -> str:
    """Generate results section for individual HLCDM test."""
    results = "## Analysis Results\n\n"

    if test_name == "jwst":
        halo_masses = test_result.get("halo_masses", {})
        gw_lims = test_result.get("gw_mass_limits", {})
        results += "### JWST Early Galaxy Formation Results\n\n"
        results += f"- Galaxies analyzed: {test_result.get('n_galaxies', 0)}\n"
        results += f"- H-ΛCDM halo mass limit at z=10: {halo_masses.get('z10_limit', 'N/A')}\n"
        results += f"- Information saturation ratio: {gw_lims.get('information_saturation_ratio', 'N/A')}\n\n"
    
    elif test_name == "lyman_alpha":
        phase_transitions = test_result.get("phase_transitions", {})
        results += "### Lyman-α Phase Transition Results\n\n"
        for key, val in phase_transitions.items():
            results += f"- {key.replace('_',' ').title()}: {val}\n"
        results += "\n"
    
    elif test_name == "frb":
        timing = test_result.get("timing_analysis", {})
        results += "### FRB Timing Analysis\n\n"
        results += f"- Sample size: {timing.get('n_frbs', 0)}\n"
        results += f"- Information cascade signature: {timing.get('cascade_signature', 'N/A')}\n\n"
    
    elif test_name in ["e8_ml", "e8_chiral", "temporal_cascade"]:
        summary = test_result.get("summary", {})
        results += f"### {test_name.replace('_',' ').title()} Results\n\n"
        for key, val in summary.items():
            label = key.replace("_", " ").title()
            if isinstance(val, float):
                results += f"- {label}: {val:.4f}\n"
            else:
                results += f"- {label}: {val}\n"
        results += "\n"
    
    else:
        results += "No structured results available for this test.\n\n"

    return results


def _generate_hlcdm_test_validation(test_name: str, test_result: Dict[str, Any]) -> str:
    """Generate validation section for individual HLCDM test."""
    validation = "## Validation\n\n"
    basic_val = test_result.get("validation", {})

    if not basic_val:
        validation += "No validation results available.\n\n"
        return validation

    overall_status = basic_val.get("overall_status", "UNKNOWN")
    validation += f"**Overall Status:** {overall_status}\n\n"

    validation_tests = {k: v for k, v in basic_val.items() if isinstance(v, dict) and "passed" in v}
    if validation_tests:
        validation += "**Validation Tests:**\n\n"
        for test_name_i, test_result_i in validation_tests.items():
            passed = test_result_i.get("passed", False)
            status = "✓ PASSED" if passed else "✗ FAILED"
            validation += f"- **{test_name_i.replace('_', ' ').title()}**: {status}\n"
            if not passed and "error" in test_result_i:
                validation += f"  - Error: {test_result_i['error']}\n"
        validation += "\n"

    null_test = basic_val.get("null_hypothesis_test", {})
    if null_test and isinstance(null_test, dict):
        validation += "### Null Hypothesis Testing\n\n"
        null_hypothesis = null_test.get("null_hypothesis", "N/A")
        alternative = null_test.get("alternative_hypothesis", "N/A")
        rejected = null_test.get("null_hypothesis_rejected", False)
        p_value = null_test.get("p_value", None)

        validation += f"**Null Hypothesis:** {null_hypothesis}\n\n"
        validation += f"**Alternative Hypothesis:** {alternative}\n\n"
        if p_value is not None:
            validation += f"**p-value:** {p_value:.4f}\n\n"
        validation += f"**Result:** {'Null hypothesis rejected' if rejected else 'Null hypothesis not rejected (null result)'}\n\n"
        if "interpretation" in null_test:
            validation += f"**Interpretation:** {null_test['interpretation']}\n\n"

    return validation


def _generate_hlcdm_test_conclusion(test_name: str, test_result: Dict[str, Any]) -> str:
    """Generate conclusion for individual HLCDM test report."""
    conclusion = "## Conclusion\n\n"
    validation = test_result.get("validation", {})
    overall_status = validation.get("overall_status", "UNKNOWN")
    conclusion += f"Validation status: **{overall_status}**.\n\n"
    conclusion += f"Analysis completed for {test_name} test.\n"
    return conclusion

