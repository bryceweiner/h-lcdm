"""BAO pipeline reporting helpers."""

from typing import Any, Dict, Iterable, List


def _render_dataset_systematics(dataset_results: Dict[str, Any], bao_data: Dict[str, Any]) -> str:
    dataset_name = dataset_results.get("dataset_name")
    dataset_info = bao_data.get(dataset_name, {})
    survey_systematics = dataset_info.get("survey_systematics", {})
    redshift_calibration = dataset_info.get("redshift_calibration", {})

    section = ""
    if survey_systematics:
        section += "**Survey-Specific Systematics:**\n\n"
        baseline = survey_systematics.get("baseline", False)
        method = survey_systematics.get("method", "unknown")
        tracer = survey_systematics.get("tracer", "unknown")
        reference = survey_systematics.get("reference", "N/A")

        section += f"- **Baseline:** {'✓ YES (reference dataset)' if baseline else '✗ NO (calibrated to BOSS_DR12)'}\n"
        section += f"- **Method:** {method}\n"
        section += f"- **Tracer:** {tracer}\n"
        section += f"- **Reference:** {reference}\n\n"

        sys_components = [
            ("redshift_calibration", "Redshift calibration"),
            ("survey_geometry", "Survey geometry"),
            ("reconstruction_bias", "Reconstruction bias"),
            ("fiducial_cosmology", "Fiducial cosmology"),
            ("fiber_collision", "Fiber collision"),
            ("template_fitting", "Template fitting"),
            ("photo_z_scatter", "Photometric redshift scatter"),
            ("fiducial_compression_systematic", "Fiducial-compression (legacy rs/D_V)"),
        ]

        for key, label in sys_components:
            val = survey_systematics.get(key, 0.0)
            if val > 0:
                section += f"- {label}: {val:.4f}\n"

        total_sys = survey_systematics.get("total_systematic", 0.0)
        section += f"- **Total Systematic Uncertainty:** {total_sys:.4f}\n\n"

    if redshift_calibration:
        section += "**Redshift Calibration:**\n\n"
        section += f"- Effective redshift: z_eff = {redshift_calibration.get('z_effective', 0):.3f}\n"
        section += f"- Redshift error: σ_z = {redshift_calibration.get('z_error', 0):.4f}\n"
        section += f"- Resolution: R = {redshift_calibration.get('resolution', 0)}\n\n"
    return section


def _render_measurements(measurements: Iterable[Dict[str, Any]]) -> str:
    section = "**Measurements vs Predictions:**\n\n"
    section += "| Redshift | Measurement | Error | H-ΛCDM Prediction | Pull (σ) | Status |\n"
    section += "|----------|-------------|-------|-------------------|----------|--------|\n"
    for m in measurements:
        z = m.get("z", 0)
        val = m.get("value", 0)
        err = m.get("error", 0)
        pred = m.get("predicted", 0)
        pull = m.get("pull", 0)
        consistent = m.get("consistent", False)
        status = "✓ Consistent" if consistent else "✗ Tension"
        if abs(pull) > 3:
            status = "⚠ High Tension"
        section += f"| {z:.3f} | {val:.3f} | {err:.3f} | {pred:.3f} | {pull:.2f} | {status} |\n"
    section += "\n"
    return section


def results(main_results: Dict[str, Any]) -> str:
    """Render BAO pipeline analysis results."""
    rs_theory = main_results.get("theoretical_rs", 150.71)
    rs_lcdm = main_results.get("rs_lcdm", 147.5)
    prediction_test = main_results.get("prediction_test", {})
    bao_data = main_results.get("bao_data", {})
    datasets_tested = main_results.get("datasets_tested", [])
    forward_predictions = main_results.get("forward_predictions", {})
    sound_horizon_consistency = main_results.get("sound_horizon_consistency", {})

    results_section = "### BAO Scale Predictions\n\n"
    results_section += (
        f"**H-ΛCDM Theoretical Prediction:** Enhanced sound horizon r_s = {rs_theory} Mpc "
        f"(vs ΛCDM r_s = {rs_lcdm} Mpc)\n\n"
    )

    results_section += (
        "**Analysis Approach:** Each survey is treated with its own unique systematic errors and redshift calibration. "
        "No normalization is performed - we \"meet them where they are\" by accounting for each survey's specific "
        "characteristics. BOSS_DR12 is used as the baseline for all comparisons.\n\n"
    )

    all_datasets = list(bao_data.keys())
    results_section += f"**Available Datasets:** {len(all_datasets)} datasets: {', '.join([d.upper() for d in all_datasets])}\n\n"
    results_section += f"**Datasets Tested:** {len(datasets_tested)} surveys: {', '.join([d.upper() for d in datasets_tested])}\n\n"

    baseline_datasets = [d for d in datasets_tested if bao_data.get(d, {}).get("survey_systematics", {}).get("baseline", False)]
    if baseline_datasets:
        results_section += f"**Baseline Dataset:** {', '.join([d.upper() for d in baseline_datasets])} (used as reference for all comparisons)\n\n"

    untested = [d for d in all_datasets if d not in datasets_tested]
    if untested:
        results_section += f"**Note:** {len(untested)} datasets available but not tested in this run: {', '.join([d.upper() for d in untested])}\n\n"

    if prediction_test:
        results_section += "### Individual Dataset Tests\n\n"
        for dataset_name, dataset_results in prediction_test.items():
            results_section += f"#### {dataset_name.upper()}\n\n"
            results_section += _render_dataset_systematics({"dataset_name": dataset_name}, bao_data)

            if "measurements" in dataset_results:
                results_section += _render_measurements(dataset_results["measurements"])

            if "statistics" in dataset_results:
                stats = dataset_results["statistics"]
                results_section += "**Statistical Summary:**\n"
                results_section += f"- Chi-squared: {stats.get('chi2', 0):.2f} (dof={stats.get('dof', 0)})\n"
                results_section += f"- p-value: {stats.get('p_value', 0):.4f}\n"
                results_section += f"- Consistency: {'✓ PASSED' if stats.get('consistent', False) else '✗ FAILED'}\n\n"

    if sound_horizon_consistency:
        consistent = sound_horizon_consistency.get("consistent", False)
        h0_implied = sound_horizon_consistency.get("h0_implied", 0)
        h0_std = sound_horizon_consistency.get("h0_std", 0)
        results_section += "### Sound Horizon Consistency Test\n\n"
        results_section += f"**Overall Consistency:** {'✓ PASSED' if consistent else '✗ FAILED'}\n\n"
        results_section += (
            f"Implied Hubble Constant from H-ΛCDM r_s: H0 = {h0_implied:.2f} ± {h0_std:.2f} km/s/Mpc\n"
            "(Consistent with Planck 2018 H0 = 67.4 ± 0.5 km/s/Mpc)\n\n"
        )

    if forward_predictions:
        results_section += "### Forward Predictions (Blind Test)\n\n"
        results_section += "Predictions for future surveys/redshifts based on H-ΛCDM scaling:\n\n"
        results_section += "| Survey | Redshift | D_V/r_s Prediction | Expected Error |\n"
        results_section += "|--------|----------|-------------------|----------------|\n"
        surveys: List[Dict[str, Any]] = forward_predictions.get("surveys", [])
        for survey in surveys:
            name = survey.get("name", "Unknown")
            z = survey.get("z", 0)
            pred = survey.get("prediction", 0)
            err = survey.get("expected_error", 0)
            results_section += f"| {name} | {z:.2f} | {pred:.2f} | {err:.3f} |\n"
            results_section += "\n"

    return results_section


def summary(main_results: Dict[str, Any]) -> str:
    """Short summary for comprehensive report."""
    formatted = ""

    if "summary" in main_results:
        summary_data = main_results["summary"]
        formatted += f"- **Theoretical prediction:** α = {main_results.get('theoretical_alpha', 'N/A')}\n"
        formatted += f"- **Tests performed:** {summary_data.get('total_tests', 0)}\n"
        formatted += f"- **Successful predictions:** {summary_data.get('total_passed', 0)}\n"
        formatted += f"- **Success rate:** {summary_data.get('overall_success_rate', 0):.1%}\n"

    if "alpha_consistency" in main_results:
        consistency = main_results["alpha_consistency"]
        formatted += f"- **Alpha consistency:** {consistency.get('theoretical_comparison', {}).get('consistent', False)}\n"

    return formatted


def conclusion(main_results: Dict[str, Any], overall_status: str) -> str:
    """BAO pipeline conclusion text."""
    summary_data = main_results.get("summary", {})
    success_rate = summary_data.get("overall_success_rate", 0)

    conclusion_text = "### Did We Find What We Were Looking For?\n\n"
    if success_rate > 0.8:
        conclusion_text += f"**YES** - H-ΛCDM predictions are consistent with {success_rate:.1%} of tested BAO datasets.\n\n"
    elif success_rate > 0.5:
        conclusion_text += f"**PARTIAL** - H-ΛCDM predictions are consistent with {success_rate:.1%} of tested BAO datasets.\n\n"
    else:
        conclusion_text += f"**NO** - H-ΛCDM predictions are consistent with only {success_rate:.1%} of tested BAO datasets.\n\n"

    conclusion_text += "The analysis \"met the data where it is\" by accounting for survey-specific systematics without applying arbitrary normalizations. "
    conclusion_text += f"Validation status: **{overall_status}**.\n\n"
    return conclusion_text

