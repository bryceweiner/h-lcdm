"""ML pipeline reporting helpers."""

from typing import Any, Dict, Tuple
import numpy as np


def _ml_sections(actual_results: Dict[str, Any], grok_client) -> Tuple[str, str]:
    """Generate Grok and analysis sections for ML pipeline."""
    main_results = actual_results.get("main", {})
    if not main_results or len(main_results) == 0:
        main_results = {k: v for k, v in actual_results.items() if k not in ["validation", "validation_extended"]}

    pattern = main_results.get("pattern_detection", {})
    grok_sections = ""
    analysis_results_section = "## Analysis Results\n\n"

    actual_main = actual_results.get("results", actual_results)
    if isinstance(actual_main, dict) and "main" in actual_main:
        main_results = actual_main["main"]

    if not main_results or len(main_results) == 0:
        main_results = {k: v for k, v in actual_results.items() if k not in ["validation", "validation_extended"]}

    actual_results = actual_results.get("results", actual_results)
    main_results = actual_results.get("main", main_results)
    if not main_results or len(main_results) == 0:
        main_results = {k: v for k, v in actual_results.items() if k not in ["validation", "validation_extended"]}

    pattern = main_results.get("pattern_detection", {})
    top_anoms = pattern.get("top_anomalies", []) if pattern else []
    sample_context = pattern.get("sample_context", {}) or {}

    robust_indices = []
    bootstrap = main_results.get("validation", {}).get("bootstrap", {}) if isinstance(main_results.get("validation", {}), dict) else {}
    stability = bootstrap.get("stability_analysis", {}) if isinstance(bootstrap, dict) else {}
    robust_patterns = stability.get("robust_patterns", {}) if isinstance(stability, dict) else {}
    if isinstance(robust_patterns, dict):
        robust_indices = robust_patterns.get("robust_anomaly_indices", []) or []

    by_idx = {a.get("sample_index"): a for a in top_anoms if "sample_index" in a}

    anomalies_ge = []
    ensemble_scores = pattern.get("aggregated_results", {}).get("ensemble_scores", []) if pattern else []
    agg_context = pattern.get("aggregated_results", {}) or {} if pattern else {}
    default_modalities = agg_context.get("modalities", [])
    default_ctx = {"redshift_regime": "n/a", "modalities": default_modalities}
    sample_context = agg_context.get("sample_context", {}) or {}

    has_scores = False
    if isinstance(ensemble_scores, (list, tuple)):
        has_scores = bool(ensemble_scores)
    elif isinstance(ensemble_scores, np.ndarray):
        has_scores = ensemble_scores.size > 0

    if has_scores:
        for i, sc in enumerate(ensemble_scores):
            if sc >= 0.5:
                base = by_idx.get(i, {"sample_index": i})
                entry = {
                    **base,
                    "anomaly_score": base.get("anomaly_score", float(sc)),
                    "favored_model": base.get("favored_model", "INDETERMINATE"),
                    "ontology_tags": base.get("ontology_tags", []),
                    "context": base.get("context")
                    or sample_context.get(str(i))
                    or sample_context.get(i, {})
                    or default_ctx,
                }
                if not entry["context"]:
                    entry["context"] = default_ctx
                entry["context"].setdefault("redshift_regime", "n/a")
                entry["context"].setdefault("modalities", default_modalities)
                anomalies_ge.append(entry)
    else:
        anomalies_ge = []
        for a in top_anoms:
            if a.get("anomaly_score", 0) >= 0.5:
                ctx = a.get("context") or sample_context.get(str(a.get("sample_index"))) or sample_context.get(a.get("sample_index"), {}) or default_ctx
                ctx.setdefault("redshift_regime", "n/a")
                ctx.setdefault("modalities", default_modalities)
                enriched = dict(a)
                enriched.setdefault("favored_model", "INDETERMINATE")
                enriched.setdefault("ontology_tags", [])
                enriched["context"] = ctx
                anomalies_ge.append(enriched)

    if robust_indices:
        anomalies_ge = [a for a in anomalies_ge if a.get("sample_index") in robust_indices]

    anomalies_for_grok = anomalies_ge if anomalies_ge else top_anoms
    if grok_client and anomalies_for_grok:
        grok_sections += "### Scientific Interpretation (AI Generated)\n\n"
        grok_analysis = grok_client.generate_anomaly_report(
            anomalies_for_grok,
            context="unsupervised ML pipeline analyzing CMB, BAO, and Void data for H-Lambda-CDM signatures",
            two_stage=True,
            three_stage=True,
        )
        grok_sections += f"{grok_analysis}\n\n"

    pipeline_status = main_results.get("pipeline_completed", False)
    stages = main_results.get("stages_completed", {})
    feature_summary = main_results.get("feature_summary", {})
    ssl = main_results.get("ssl_training", {})
    domain = main_results.get("domain_adaptation", {})
    validation = main_results.get("validation", {})
    interp = main_results.get("interpretability", {})
    mcmc_res = main_results.get("mcmc", {})

    analysis_results_section += "### ML Pipeline Status\n\n"
    analysis_results_section += f"- Pipeline completed: {'✓' if pipeline_status else '✗'}\n"
    if stages:
        completed = [k for k, v in stages.items() if v]
        analysis_results_section += f"- Stages completed: {', '.join(completed)}\n"
    if feature_summary:
        analysis_results_section += f"- Latent samples: {feature_summary.get('n_samples', 'N/A')} × {feature_summary.get('latent_dim', 'N/A')} dims\n"
    analysis_results_section += "\n"

    if ssl:
        analysis_results_section += "### SSL Training\n\n"
        analysis_results_section += f"- Training completed: {'✓' if ssl.get('training_completed') else '✗'}\n"
        analysis_results_section += f"- Final contrastive loss: {ssl.get('final_loss', 'N/A')}\n"
        analysis_results_section += f"- Modalities trained: {', '.join(ssl.get('modalities_trained', []))}\n\n"

    if domain:
        metrics = domain.get("adaptation_metrics", {})
        analysis_results_section += "### Domain Adaptation\n\n"
        analysis_results_section += f"- Adaptation batches: {metrics.get('total_adaptation_steps', domain.get('total_batches', 'N/A'))}\n"
        avg_losses = metrics.get("average_losses", {})
        if avg_losses:
            analysis_results_section += f"- Avg total adaptation loss: {avg_losses.get('avg_total_adaptation', 'N/A')}\n"
        analysis_results_section += "\n"

    if mcmc_res:
        analysis_results_section += "### MCMC Inference\n\n"
        acc = mcmc_res.get("acceptance_rate")
        device = mcmc_res.get("device")
        if acc is not None:
            analysis_results_section += f"- Acceptance rate: {acc:.3f}\n"
        if device:
            analysis_results_section += f"- Device: {device}\n"
        summary = mcmc_res.get("summary", {})
        if summary:
            analysis_results_section += "- Posterior summaries:\n"
            for p, stats in summary.items():
                mean = stats.get("mean", "N/A")
                lo = stats.get("ci16", None)
                hi = stats.get("ci84", None)
                if lo is not None and hi is not None:
                    analysis_results_section += f"  * {p}: {mean} [{lo}, {hi}]\n"
                else:
                    analysis_results_section += f"  * {p}: {mean}\n"
        analysis_results_section += "\n"

    if interp:
        analysis_results_section += "### Interpretability\n\n"
        analysis_results_section += f"- Interpretability completed: {'✓' if interp.get('interpretability_completed') else '✗'}\n"
        lime_count = len(interp.get("lime_explanations", []))
        analysis_results_section += f"- LIME explanations: {lime_count}\n\n"

    if validation:
        analysis_results_section += "### Validation\n\n"
        if isinstance(validation, dict):
            for k, v in validation.items():
                if isinstance(v, (int, float, str, bool)):
                    analysis_results_section += f"- {k.replace('_',' ').title()}: {v}\n"
        analysis_results_section += "\n"

        analysis_results_section += "### Statistical Test Results\n\n"
        bootstrap = validation.get("bootstrap", {}) if isinstance(validation, dict) else {}
        if bootstrap:
            bootstrap_results = bootstrap.get("bootstrap_validation", {}) if isinstance(bootstrap, dict) else {}
            stability = bootstrap.get("stability_analysis", {}) if isinstance(bootstrap, dict) else {}

            if stability:
                analysis_results_section += "**Bootstrap Stability Analysis:**\n\n"
                mean_freq = stability.get("mean_detection_frequency", None)
                std_freq = stability.get("std_detection_frequency", None)
                if mean_freq is not None and std_freq is not None:
                    analysis_results_section += f"- Mean detection frequency: {mean_freq:.4f} ± {std_freq:.4f}\n"
                stable_samples = stability.get("highly_stable_samples", None)
                if stable_samples is not None:
                    analysis_results_section += f"- Highly stable samples (≥95%): {stable_samples}\n"
                unstable_samples = stability.get("unstable_samples", None)
                if unstable_samples is not None:
                    analysis_results_section += f"- Unstable samples (≤5%): {unstable_samples}\n"
                percentiles = stability.get("detection_frequency_percentiles", [])
                if percentiles and len(percentiles) >= 3:
                    analysis_results_section += f"- Detection frequency percentiles: Q25={percentiles[0]:.4f}, Q50={percentiles[1]:.4f}, Q75={percentiles[2]:.4f}\n"
                analysis_results_section += "\n"

            robust_patterns = stability.get("robust_patterns", {}) if isinstance(stability, dict) else {}
            if robust_patterns:
                n_robust = robust_patterns.get("n_robust_anomalies", None)
                if n_robust is not None:
                    analysis_results_section += f"- Robust anomalies (detected in ≥95% of bootstrap samples): {n_robust}\n"
                analysis_results_section += "\n"

        test_results = validation.get("test_results", {}) if isinstance(validation, dict) else {}
        if test_results:
            chi2_results = test_results.get("chi_squared", {}) if isinstance(test_results, dict) else {}
            if chi2_results:
                analysis_results_section += "**Chi-Squared Comparison:**\n\n"
                hlcdm_chi2 = chi2_results.get("hlcdm", None)
                lcdm_chi2 = chi2_results.get("lcdm", None)
                if hlcdm_chi2 is not None:
                    analysis_results_section += f"- H-ΛCDM: χ² = {hlcdm_chi2:.3f}\n"
                if lcdm_chi2 is not None:
                    analysis_results_section += f"- ΛCDM: χ² = {lcdm_chi2:.3f}\n"
                if hlcdm_chi2 is not None and lcdm_chi2 is not None:
                    delta_chi2 = abs(hlcdm_chi2 - lcdm_chi2)
                    analysis_results_section += f"- Δχ² = {delta_chi2:.3f}\n"
                analysis_results_section += "\n"

        mc_results = validation.get("monte_carlo", {}) if isinstance(validation, dict) else {}
        if mc_results:
            analysis_results_section += "**Monte Carlo Simulation Results:**\n\n"
            n_sim = mc_results.get("n_simulations", None)
            if n_sim is not None:
                analysis_results_section += f"- Simulations run: {n_sim}\n"
            if "p_value" in mc_results:
                analysis_results_section += f"- p-value: {mc_results.get('p_value', 0):.4f}\n"
            if "confidence_level" in mc_results:
                analysis_results_section += f"- Confidence level: {mc_results.get('confidence_level', 0):.3f}\n"
            if "null_hypothesis_rejected" in mc_results:
                rejected = mc_results.get("null_hypothesis_rejected", False)
                analysis_results_section += f"- Null hypothesis rejected: {'✓ YES' if rejected else '✗ NO'}\n"
            analysis_results_section += "\n"

        null_test = validation.get("null_hypothesis", {}) if isinstance(validation, dict) else {}
        if null_test:
            analysis_results_section += "**Null Hypothesis Testing:**\n\n"
            test_name = null_test.get("test_name", "Null Hypothesis Test")
            analysis_results_section += f"- Test: {test_name}\n"
            analysis_results_section += f"- Null hypothesis: {null_test.get('null_hypothesis', 'N/A')}\n"
            analysis_results_section += f"- Alternative hypothesis: {null_test.get('alternative_hypothesis', 'N/A')}\n"
            if "p_value" in null_test:
                analysis_results_section += f"- p-value: {null_test.get('p_value', 0):.4f}\n"
            if "bayes_factor" in null_test:
                analysis_results_section += f"- Bayes factor: {null_test.get('bayes_factor', 1.0):.2f}\n"
            if "result" in null_test:
                analysis_results_section += f"- Result: {null_test.get('result', 'N/A')}\n"
            analysis_results_section += "\n"

    if pattern:
        analysis_results_section += "### Pattern Detection\n\n"
        analysis_results_section += f"- Detection completed: {'✓' if pattern.get('detection_completed') else '✗'}\n"
        analysis_results_section += f"- Samples analyzed: {pattern.get('n_samples_analyzed', 'N/A')}\n"
        analysis_results_section += f"- Anomalies with score ≥ 0.5: {len(anomalies_ge)}\n"
        if robust_indices:
            analysis_results_section += f"  (Filtered to bootstrap-robust indices: {len(robust_indices)})\n"

        if anomalies_ge:
            anomalies_ge_sorted = sorted(anomalies_ge, key=lambda x: -float(x.get("anomaly_score", 0)))
            analysis_results_section += f"  - Highest anomaly score: {anomalies_ge_sorted[0].get('anomaly_score', 'N/A')}\n"
            analysis_results_section += f"  - Anomaly leaderboard (all {len(anomalies_ge_sorted)} samples):\n"
            for entry in anomalies_ge_sorted:
                si = entry.get("sample_index", "N/A")
                sc = entry.get("anomaly_score", "N/A")
                fm = entry.get("favored_model", "INDETERMINATE")
                tags = entry.get("ontology_tags", [])
                ctx = entry.get("context", {}) if isinstance(entry.get("context"), dict) else {}
                zreg = ctx.get("redshift_regime", "n/a")
                mods = ctx.get("modalities", [])
                analysis_results_section += f"    * {si}: {sc} | {fm} | tags={tags} | z={zreg} | mods={mods}\n"

            favored_counts = {}
            tag_counts = {}
            for entry in anomalies_ge_sorted:
                favored = entry.get("favored_model", "INDETERMINATE")
                favored_counts[favored] = favored_counts.get(favored, 0) + 1
                for tag in entry.get("ontology_tags", []):
                    tag_counts[tag] = tag_counts.get(tag, 0) + 1
            if favored_counts:
                analysis_results_section += "  - Model preference (favored_model counts):\n"
                for fm, cnt in favored_counts.items():
                    analysis_results_section += f"    * {fm}: {cnt}\n"
            if tag_counts:
                analysis_results_section += "  - Ontology tags (counts):\n"
                for tg, cnt in sorted(tag_counts.items(), key=lambda x: -x[1]):
                    analysis_results_section += f"    * {tg}: {cnt}\n"
        else:
            analysis_results_section += "- No anomalies meet the score ≥ 0.5 threshold under current robustness filters.\n"
        analysis_results_section += "\n"

    return grok_sections, analysis_results_section


def results(actual_results: Dict[str, Any], grok_client) -> str:
    """Generate full ML results section including Grok output."""
    grok_sections, analysis = _ml_sections(actual_results, grok_client)
    return grok_sections + analysis


def validation(actual_results: Dict[str, Any]) -> str:
    """Generate ML validation text (mirrors previous inlined logic)."""
    validation = "## Validation\n\n"
    validation += _ml_validation_section(actual_results)
    return validation


def _ml_validation_section(actual_results: Dict[str, Any]) -> str:
    validation = "### Basic Validation\n\n"
    validation_data = actual_results.get("validation", {})

    overall_status = validation_data.get("overall_status", "UNKNOWN")
    if overall_status == "UNKNOWN":
        bootstrap = validation_data.get("bootstrap", {})
        null_hypothesis = validation_data.get("null_hypothesis", {})
        monte_carlo = validation_data.get("monte_carlo", {})
        if bootstrap or null_hypothesis or monte_carlo:
            overall_status = "PENDING_STATISTICAL_ANALYSIS"

    validation += f"**Overall Status:** {overall_status}\n\n"

    bootstrap = validation_data.get("bootstrap", {})
    if bootstrap:
        validation += "#### Bootstrap Stability Analysis\n\n"
        stability_analysis = bootstrap.get("stability_analysis", {})
        if stability_analysis:
            mean_freq = stability_analysis.get("mean_detection_frequency", 0)
            highly_stable = stability_analysis.get("highly_stable_samples", 0)
            unstable = stability_analysis.get("unstable_samples", 0)
            robust_anomalies = stability_analysis.get("n_robust_anomalies", 0)

            validation += f"**Bootstrap Resampling:** {bootstrap.get('validation_metadata', {}).get('n_bootstraps', 'N/A')} iterations\n\n"
            validation += f"- Mean detection frequency: {mean_freq:.3f}\n"
            validation += f"- Highly stable samples (≥95% frequency): {highly_stable}\n"
            validation += f"- Unstable samples (≤5% frequency): {unstable}\n"
            validation += f"- Robust anomalies (≥95% frequency): {robust_anomalies}\n\n"

            stability_summary = bootstrap.get("stability_summary", {})
            if stability_summary:
                status = stability_summary.get("stability_status", "N/A")
                validation += f"**Stability Status:** {status}\n\n"

    null_hypothesis = validation_data.get("null_hypothesis", {})
    if null_hypothesis:
        validation += "#### Chi-Squared Comparison (H-ΛCDM vs ΛCDM)\n\n"
        real_data_result = null_hypothesis.get("real_data_result", {})
        statistical_analysis = null_hypothesis.get("statistical_analysis", {})

        hlcdm_chi2 = real_data_result.get("hlcdm_chi2")
        lcdm_chi2 = real_data_result.get("lcdm_chi2")
        if hlcdm_chi2 is None:
            hlcdm_chi2 = statistical_analysis.get("hlcdm_chi2")
        if lcdm_chi2 is None:
            lcdm_chi2 = statistical_analysis.get("lcdm_chi2")

        if hlcdm_chi2 is not None and lcdm_chi2 is not None:
            delta_chi2 = abs(hlcdm_chi2 - lcdm_chi2)
            validation += "**Model Comparison:**\n\n"
            validation += f"- H-ΛCDM χ²: {hlcdm_chi2:.2f}\n"
            validation += f"- ΛCDM χ²: {lcdm_chi2:.2f}\n"
            validation += f"- Δχ² (H-ΛCDM - ΛCDM): {delta_chi2:.2f}\n\n"
            if delta_chi2 > 6:
                if hlcdm_chi2 < lcdm_chi2:
                    validation += (
                        f"**Conclusion:** Strong statistical evidence (Δχ² = {delta_chi2:.2f} > 6) favors H-ΛCDM over ΛCDM. "
                        f"The lower χ² value ({hlcdm_chi2:.2f} vs {lcdm_chi2:.2f}) indicates H-ΛCDM provides a better fit to the observed anomaly patterns.\n\n"
                    )
                else:
                    validation += (
                        f"**Conclusion:** Strong statistical evidence (Δχ² = {delta_chi2:.2f} > 6) favors ΛCDM over H-ΛCDM. "
                        f"The lower χ² value ({lcdm_chi2:.2f} vs {hlcdm_chi2:.2f}) indicates ΛCDM provides a better fit.\n\n"
                    )
            elif delta_chi2 > 2:
                validation += (
                    f"**Conclusion:** Moderate statistical evidence (2 < Δχ² = {delta_chi2:.2f} ≤ 6) suggests a preference, but not definitive. "
                    "Further data or analysis may clarify the model comparison.\n\n"
                )
            else:
                validation += (
                    f"**Conclusion:** Weak or no statistical preference (Δχ² = {delta_chi2:.2f} ≤ 2). "
                    "Both models fit the data comparably within statistical uncertainty.\n\n"
                )
        else:
            validation += "Chi-squared values not available in validation results.\n\n"

        significance_test = null_hypothesis.get("significance_test", {})
        if significance_test:
            p_value = significance_test.get("p_value")
            overall_sig = significance_test.get("overall_significance", False)
            validation += "**Null Hypothesis Test:**\n\n"
            if p_value is not None:
                validation += f"- p-value: {p_value:.4f}\n"
            validation += f"- Overall significance: {'Significant' if overall_sig else 'Not significant'}\n\n"

    monte_carlo = validation_data.get("monte_carlo", {})
    if monte_carlo:
        validation += "#### Monte Carlo Simulation Results\n\n"
        n_simulations = monte_carlo.get("n_simulations", "N/A")
        p_value = monte_carlo.get("p_value")
        significance_level = monte_carlo.get("significance_level", 0.05)
        mean_sim_score = monte_carlo.get("mean_sim_score")
        std_sim_score = monte_carlo.get("std_sim_score")

        validation += f"**Simulations:** {n_simulations}\n"
        if p_value is not None:
            validation += f"- p-value: {p_value:.4f}\n"
            validation += f"- Significance level: {significance_level}\n"
            if p_value < significance_level:
                validation += f"- **Result:** Statistically significant (p < {significance_level})\n\n"
            else:
                validation += f"- **Result:** Not statistically significant (p ≥ {significance_level})\n\n"
        if mean_sim_score is not None and std_sim_score is not None:
            validation += f"- Mean anomaly score (simulations): {mean_sim_score:.3f} ± {std_sim_score:.3f}\n\n"

    if not bootstrap and not null_hypothesis and not monte_carlo:
        validation += "No statistical validation results available. Validation tests may not have been completed.\n\n"

    return validation


def summary(main_results: Dict[str, Any]) -> str:
    """Short summary for comprehensive report."""
    formatted = ""

    if "synthesis" in main_results:
        synthesis = main_results["synthesis"]
        formatted += f"- **Evidence strength:** {synthesis.get('strength_category', 'N/A')}\n"
        formatted += f"- **Score:** {synthesis.get('total_score', 0)}/{synthesis.get('max_possible_score', 0)}\n"

    if "test_results" in main_results:
        formatted += f"- **Tests run:** {len(main_results.get('test_results', {}))}\n"

    return formatted


def conclusion(main_results: Dict[str, Any], overall_status: str) -> str:
    """ML pipeline conclusion text."""
    key_findings = main_results.get("key_findings", {})
    detected = key_findings.get("detected_anomalies", 0)
    n_samples = main_results.get("pattern_detection", {}).get("n_samples_analyzed", 0)
    pipeline_completed = main_results.get("pipeline_completed", False)

    if detected >= 5:
        strength = "ELEVATED"
    elif detected > 0:
        strength = "WEAK"
    else:
        strength = "NONE"

    conclusion_text = "### Did We Find What We Were Looking For?\n\n"
    completion_txt = "✓ completed" if pipeline_completed else "⚠ not completed"
    conclusion_text += (
        f"Detected anomalies: {detected} of {n_samples or 'N/A'} analyzed samples; "
        f"evidence tag: {strength}. Pipeline status: {completion_txt}.\n\n"
    )
    conclusion_text += (
        "ML pattern recognition combined ensemble anomaly scores with LIME/SHAP interpretability. "
        "Interpret these findings as distributional deviations across surveys, not single-point detections.\n\n"
    )
    conclusion_text += f"Validation status: **{overall_status}**.\n\n"
    return conclusion_text

