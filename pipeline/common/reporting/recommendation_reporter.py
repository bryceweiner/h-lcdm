"""
Recommendation pipeline reporter.
"""

from typing import Dict, Any, List, Optional
import numpy as np

# Constants from amplitude_consistency module
ALPHA_THEORY = -5.7
ALPHA_THEORY_ERR = 2.0
ALPHA_BAO = -7.2


def results(main_results: Dict[str, Any], grok_client: Optional[Any] = None) -> str:
    """
    Render recommendation pipeline results into markdown with optional Grok interpretation.
    
    Handles new 6-phase H-ΛCDM analysis format and new recommendation format.
    """
    lines: List[str] = []
    grok_sections: List[str] = []
    
    # Check if this is the new format (has cross_modal_coherence) or old format
    if "cross_modal_coherence" in main_results:
        # New format: 6-phase H-ΛCDM analysis
        return _render_hlcdm_analysis(main_results, grok_client)
    elif "model_comparison" in main_results:
        # Legacy MCMC format
        return _render_model_comparison(main_results, grok_client)
    elif "qtep_verification" in main_results or "recommendation_id" in main_results:
        # New recommendation format (Recommendation 6: Cross-Modal Coherence)
        return _render_recommendation_6(main_results, grok_client)
    else:
        # Legacy recommendation format
        return _render_legacy_results(main_results, grok_client)


def _render_hlcdm_analysis(main_results: Dict[str, Any], grok_client: Optional[Any] = None) -> str:
    """Render new 6-phase H-ΛCDM analysis format."""
    lines: List[str] = []
    grok_sections: List[str] = []
    
    lines.append("## H-ΛCDM CMB Cross-Modal Coherence Analysis\n")
    
    # Datasets
    datasets = main_results.get("datasets", [])
    lines.append(f"**Datasets analyzed:** {', '.join(datasets)}\n")
    
    # Phase 2: Cross-Modal Coherence (PRIMARY RESULT)
    lines.append("### Phase 2: Cross-Modal Coherence (Key Discriminant)\n")
    coherence = main_results.get("cross_modal_coherence", {})
    coherence_summary = coherence.get("summary", {})
    
    # NEW: Acoustic peak correlation analysis
    acoustic_peaks = coherence_summary.get('acoustic_peaks', {})
    if acoustic_peaks and acoustic_peaks.get('n_acoustic_peaks_identified', 0) > 0:
        lines.append("**Acoustic Peak Correlation Analysis:**\n")
        lines.append("_Tests whether residuals are more correlated at acoustic peak positions than off-peak._\n")
        lines.append(f"- Acoustic peaks identified: {acoustic_peaks.get('n_acoustic_peaks_identified', 0)}")
        lines.append(f"- Spectrum pairs tested: {acoustic_peaks.get('n_pairs_tested', 0)}")
        lines.append(f"- Mean |ρ| at ALL peak positions: {acoustic_peaks.get('mean_abs_rho_peak_global', 0):.4f}")
        lines.append(f"- Mean |ρ| at off-peak positions: {acoustic_peaks.get('mean_abs_rho_offpeak_global', 0):.4f}")
        
        median_enh = acoustic_peaks.get('median_enhancement_ratio')
        enh_per_pair = acoustic_peaks.get('enhancement_per_pair', [])
        
        if median_enh is not None:
            lines.append(f"- **Median enhancement ratio: {median_enh:.2f}×**")
            lines.append(f"  (Enhancement ratio = |ρ_peaks| / |ρ_offpeak| for each pair)")
        
        if enh_per_pair:
            lines.append(f"- Per-pair enhancements: {[f'{e:.2f}×' for e in enh_per_pair]}")
        
        # Interpretation based on median enhancement
        if median_enh is not None:
            if median_enh > 1.5:
                lines.append("\n  → ✓ **Residuals correlate MORE at acoustic peaks**")
                lines.append("  → Consistent with H-ΛCDM coherent enhancement prediction")
            elif median_enh > 1.1:
                lines.append("\n  → △ Weak enhancement at acoustic peaks")
            else:
                lines.append("\n  → ✗ No enhancement (consistent with ΛCDM uncorrelated noise)")
        lines.append("")
    
    # NEW: ML-targeted test at specific ℓ range
    ml_targeted = coherence_summary.get('ml_targeted', {})
    if ml_targeted and ml_targeted.get('n_pairs_tested', 0) > 0:
        lines.append("**ML-Targeted Correlation Test (ℓ=800-1200):**\n")
        lines.append("_Tests ML Recommendation 1: Correlation at specific multipole range where ML detected anomalies._\n")
        ell_range = ml_targeted.get('ell_range', [800, 1200])
        lines.append(f"- Target ℓ range: {ell_range[0]:.0f}-{ell_range[1]:.0f} (3rd acoustic peak region)")
        lines.append(f"- Spectrum pairs tested: {ml_targeted.get('n_pairs_tested', 0)}")
        lines.append(f"- Mean |ρ| IN target range: {ml_targeted.get('mean_abs_rho_in_range', 0):.4f}")
        lines.append(f"- Mean |ρ| OUTSIDE range: {ml_targeted.get('mean_abs_rho_outside', 0):.4f}")
        
        median_enh_ml = ml_targeted.get('median_enhancement')
        n_met = ml_targeted.get('n_predictions_met', 0)
        n_total = ml_targeted.get('n_pairs_tested', 0)
        ml_threshold = ml_targeted.get('ml_predicted_threshold', 0.3)
        
        if median_enh_ml is not None:
            lines.append(f"- **Median enhancement: {median_enh_ml:.2f}×**")
        
        lines.append(f"- **ML prediction (|ρ| > {ml_threshold}): {n_met}/{n_total} pairs met**")
        
        # Interpretation
        frac_met = ml_targeted.get('fraction_predictions_met', 0.0)
        if frac_met >= 0.5:
            lines.append(f"\n  → ✓ **ML prediction confirmed** ({frac_met*100:.0f}% of pairs)")
            lines.append("  → Residuals ARE more correlated at ℓ=800-1200")
        elif frac_met > 0:
            lines.append(f"\n  → △ Partial support ({frac_met*100:.0f}% of pairs)")
        else:
            lines.append("\n  → ✗ **ML prediction NOT confirmed**")
            lines.append("  → No enhanced correlation at ML-identified range")
        lines.append("")
    
    # Legacy summary
    lines.append("**Legacy Analysis Summary:**\n")
    lines.append(f"- Significant coherence peaks: {coherence_summary.get('n_significant_peaks', 0)}")
    lines.append(f"- Peaks in expected range (ℓ≈800-1200): {coherence_summary.get('n_peaks_in_range', 0)}")
    lines.append(f"- Mean peak correlation: {coherence_summary.get('mean_peak_correlation', 'N/A'):.3f}" if coherence_summary.get('mean_peak_correlation') is not None else "- Mean peak correlation: N/A")
    lines.append(f"- Total pairs tested: {coherence_summary.get('total_pairs_tested', 0)}")
    lines.append("")
    
    # By survey
    by_survey = coherence.get("by_survey", {})
    for survey_name, survey_results in by_survey.items():
        lines.append(f"**{survey_name.upper()}:**\n")
        for pair_name, pair_results in survey_results.items():
            peak = pair_results.get("peak", {})
            lines.append(f"- {pair_name.upper()}:")
            lines.append(f"  - Peak ℓ: {peak.get('peak_ell', 'N/A'):.0f}" if peak.get('peak_ell') is not None else "  - Peak ℓ: N/A")
            lines.append(f"  - Peak correlation: {peak.get('peak_correlation', 'N/A'):.3f}" if peak.get('peak_correlation') is not None else "  - Peak correlation: N/A")
            lines.append(f"  - p-value: {peak.get('peak_p_value', 'N/A'):.4f}" if peak.get('peak_p_value') is not None else "  - p-value: N/A")
            lines.append(f"  - In expected range: {peak.get('in_expected_range', False)}")
        lines.append("")
    
    # Phase 3: Characteristic Scales
    lines.append("### Phase 3: Characteristic Scale Analysis\n")
    scales = main_results.get("characteristic_scales", {})
    scales_summary = scales.get("summary", {})
    
    lines.append(f"- Expected characteristic ℓ: {scales.get('expected_characteristic_ell', 'N/A'):.0f}" if scales.get('expected_characteristic_ell') is not None else "- Expected characteristic ℓ: N/A")
    lines.append(f"- Expected Δℓ: {scales.get('expected_delta_ell', 'N/A'):.0f}" if scales.get('expected_delta_ell') is not None else "- Expected Δℓ: N/A")
    lines.append(f"- Peaks matching expected scale: {scales_summary.get('n_peaks_matching_expected', 0)}/{scales_summary.get('n_total_tests', 0)}")
    lines.append(f"- Mean excess ratio near characteristic scale: {scales_summary.get('mean_excess_ratio', 'N/A'):.3f}" if scales_summary.get('mean_excess_ratio') is not None else "- Mean excess ratio: N/A")
    lines.append("")
    
    # Phase 4: Model Comparison
    lines.append("### Phase 4: H-ΛCDM vs ΛCDM Model Comparison\n")
    model_comp = main_results.get("model_comparison", {})
    
    for survey_name, survey_results in model_comp.items():
        lines.append(f"**{survey_name.upper()}:**\n")
        for spectrum, comp_data in survey_results.items():
            delta_chi2 = comp_data.get("delta_chi2", np.nan)
            lines.append(f"- {spectrum}: Δχ² = {delta_chi2:.2f}" if not np.isnan(delta_chi2) else f"- {spectrum}: Δχ² = N/A")
            if not np.isnan(delta_chi2) and delta_chi2 < 0:
                lines.append(f"  → H-ΛCDM preferred (Δχ² = {delta_chi2:.2f})")
        lines.append("")
    
    # Phase 5: Amplitude Consistency
    lines.append("### Phase 5: Amplitude Consistency\n")
    amplitude = main_results.get("amplitude_consistency", {})
    amp_consistency = amplitude.get("consistency", {})
    
    alpha_mean = amp_consistency.get("mean_alpha", np.nan)
    lines.append(f"- Mean α from CMB: {alpha_mean:.2f} ± {amp_consistency.get('std_alpha', 'N/A'):.2f}" if not np.isnan(alpha_mean) else "- Mean α: N/A")
    lines.append(f"- Theoretical α: {ALPHA_THEORY:.1f} ± {ALPHA_THEORY_ERR:.1f}")
    lines.append(f"- BAO α: {ALPHA_BAO:.1f}")
    lines.append(f"- Within prior [-7.7, -3.7]: {amp_consistency.get('within_prior', False)}")
    lines.append(f"- Consistent across TT/TE/EE: {amp_consistency.get('consistent_across_spectra', False)}")
    lines.append(f"- Consistent with BAO: {amp_consistency.get('consistent_with_bao', False)}")
    lines.append("")
    
    # Phase 6: ML Anomaly Targeting
    lines.append("### Phase 6: ML Anomaly Targeting\n")
    ml_targeting = main_results.get("ml_anomaly_targeting", {})
    ml_summary = ml_targeting.get("summary", {})
    
    lines.append(f"- Samples tested: {ml_summary.get('n_samples_tested', 0)}")
    lines.append(f"- Samples exceeding 2σ: {ml_summary.get('n_exceed_2sigma', 0)}")
    lines.append(f"- Samples with elevated correlation: {ml_summary.get('n_elevated_correlation', 0)}")
    lines.append("")
    
    # Note: Conclusion section is added by base reporter, not here
    # to avoid duplication with the ## Conclusion header
    
    # Extract conclusion data for Grok prompt (but don't render here)
    conclusion_data = main_results.get("conclusion", {})
    conclusion_text = conclusion_data.get('conclusion', 'N/A')
    
    # Grok interpretation
    if grok_client:
        grok_sections.append("### Scientific Interpretation (AI Generated)\n")
        
        # Extract key metrics for critical analysis
        acoustic_enh = acoustic_peaks.get('median_enhancement_ratio', 1.0) if acoustic_peaks else 1.0
        acoustic_sig = acoustic_peaks.get('n_acoustic_peaks_significant', 0) if acoustic_peaks else 0
        acoustic_tot = acoustic_peaks.get('n_acoustic_peaks_identified', 0) if acoustic_peaks else 0
        
        ml_targeted = coherence_summary.get('ml_targeted', {})
        ml_rho_in_range = ml_targeted.get('mean_abs_rho_in_range', 0.0)
        ml_predictions_met = ml_targeted.get('n_predictions_met', 0)
        ml_pairs_tested = ml_targeted.get('n_pairs_tested', 0)
        
        prompt = (
            "You are a senior cosmologist conducting critical peer review of an H-ΛCDM analysis. "
            "Write in third-person, dispassionate academic style. Be RIGOROUS about statistical significance and physical interpretation.\n\n"
            "KEY RESULTS TO ASSESS:\n"
            f"1a. Acoustic peak correlation (global): Enhancement ratio = {acoustic_enh:.2f}× ({acoustic_sig}/{acoustic_tot} peaks significant). "
            "ΛCDM null is ~1.0×. Is this significant?\n"
            f"1b. ML-targeted test (ℓ=800-1200): |ρ| = {ml_rho_in_range:.3f}, ML predicted |ρ| > 0.3. "
            f"ML prediction met: {ml_predictions_met}/{ml_pairs_tested} pairs. "
            "This range is where ML detected cross-modal coherence in anomalies. Is the ML prediction confirmed?\n"
            f"2. Amplitude consistency: CMB-derived α = {alpha_mean:.2f} vs theoretical α = -5.7 ± 2.0. "
            "These differ by >2σ. What does this imply about H-ΛCDM signatures in CMB?\n"
            "3. Model comparison: All Δχ² = 0.00. This is EXPECTED—H-ΛCDM predicts IDENTICAL CMB spectra to ΛCDM. "
            "The signature is in BAO sound horizon (r_s = 150.71 vs 147.09 Mpc), not CMB.\n"
            "4. Overall: Do these tests validate what the ML detected? Does cross-modal coherence discriminate H-ΛCDM from ΛCDM?\n\n"
            "Be precise. Acknowledge null results. Max 4 paragraphs."
        )
        try:
            grok_text = grok_client.generate_custom_report(prompt)
            grok_sections.append(grok_text)
            grok_sections.append("")
        except Exception:
            grok_sections.append("_Grok interpretation unavailable._\n")
    
    return "\n".join(lines + grok_sections)


def _render_model_comparison(main_results: Dict[str, Any], grok_client: Optional[Any] = None) -> str:
    """Render legacy MCMC model comparison format."""
    lines: List[str] = []
    grok_sections: List[str] = []
    
    lines.append("## CMB Model Comparison Results\n")
    
    # Datasets used
    datasets = main_results.get("datasets", [])
    lines.append(f"**Datasets analyzed:** {', '.join(datasets)}\n")
    
    z_eff = main_results.get("z_eff", None)
    if z_eff is not None:
        lines.append(f"**Effective redshift (γ-scaling):** z = {z_eff:.3f}\n")
    
    lines.append("")
    
    # ΛCDM Fit Results
    lines.append("### ΛCDM Model Fit\n")
    lcdm_fit = main_results.get("lcdm_fit", {})
    if lcdm_fit:
        best_fit = lcdm_fit.get("best_fit_params", {})
        lines.append("**Best-fit parameters:**\n")
        for param, value in best_fit.items():
            lines.append(f"- {param}: {value:.6f}")
        
        lines.append(f"\n**Fit statistics:**\n")
        lines.append(f"- χ²: {lcdm_fit.get('chi2', 'N/A'):.2f}")
        lines.append(f"- Reduced χ²: {lcdm_fit.get('reduced_chi2', 'N/A'):.4f}")
        lines.append(f"- MCMC acceptance rate: {lcdm_fit.get('acceptance_rate', 'N/A'):.3f}")
        lines.append("")
    
    # H-ΛCDM Fit Results
    lines.append("### H-ΛCDM Model Fit\n")
    hlcdm_fit = main_results.get("hlcdm_fit", {})
    if hlcdm_fit:
        best_fit = hlcdm_fit.get("best_fit_params", {})
        lines.append("**Best-fit parameters:**\n")
        for param, value in best_fit.items():
            lines.append(f"- {param}: {value:.6f}")
        
        lines.append(f"\n**Fit statistics:**\n")
        lines.append(f"- χ²: {hlcdm_fit.get('chi2', 'N/A'):.2f}")
        lines.append(f"- Reduced χ²: {hlcdm_fit.get('reduced_chi2', 'N/A'):.4f}")
        lines.append(f"- MCMC acceptance rate: {hlcdm_fit.get('acceptance_rate', 'N/A'):.3f}")
        if hlcdm_fit.get("gamma_scaling_z_eff"):
            lines.append(f"- γ-scaling redshift: z = {hlcdm_fit['gamma_scaling_z_eff']:.3f}")
        lines.append("")
    
    # Model Comparison
    lines.append("### Model Comparison\n")
    comparison = main_results.get("model_comparison", {})
    if comparison:
        lines.append("**Information Criteria:**\n")
        lines.append(f"- Δχ²_eff = χ²_ΛCDM - χ²_HLCDM: {comparison.get('delta_chi2_eff', 'N/A'):.2f}")
        lines.append(f"- AIC_ΛCDM: {comparison.get('aic_lcdm', 'N/A'):.2f}")
        lines.append(f"- AIC_HLCDM: {comparison.get('aic_hlcdm', 'N/A'):.2f}")
        lines.append(f"- ΔAIC: {comparison.get('delta_aic', 'N/A'):.2f}")
        lines.append(f"- BIC_ΛCDM: {comparison.get('bic_lcdm', 'N/A'):.2f}")
        lines.append(f"- BIC_HLCDM: {comparison.get('bic_hlcdm', 'N/A'):.2f}")
        lines.append(f"- ΔBIC: {comparison.get('delta_bic', 'N/A'):.2f}")
        
        bayes_factor = comparison.get('bayes_factor')
        if bayes_factor is not None and not np.isnan(bayes_factor):
            lines.append(f"- Bayes factor B_10: {bayes_factor:.2f}")
        
        lines.append(f"\n**Preferred model:** {comparison.get('preferred_model', 'N/A')}")
        lines.append(f"**Interpretation:** {comparison.get('significance', 'N/A')}")
        lines.append("")
    
    # Grok interpretation
    if grok_client:
        grok_sections.append("### Scientific Interpretation (AI Generated)\n")
        prompt = (
            "You are a senior cosmologist drafting a formal journal note. "
            "NEVER use first person; write in third-person, dispassionate academic style. "
            "Provide a concise interpretation of the ΛCDM vs H-ΛCDM model comparison results. "
            "Focus on: (1) whether Δχ²_eff, AIC, BIC, and Bayes factors favor one model; "
            "(2) the physical significance of γ-scaling modifications in H-ΛCDM; "
            "(3) statistical weight of the comparison given the data. "
            "Use the supplied results: "
            f"Δχ²_eff={comparison.get('delta_chi2_eff', 'N/A')}, "
            f"ΔAIC={comparison.get('delta_aic', 'N/A')}, "
            f"ΔBIC={comparison.get('delta_bic', 'N/A')}, "
            f"preferred_model={comparison.get('preferred_model', 'N/A')}. "
            "State conclusions as declarative, evidence-weighted findings."
        )
        try:
            grok_text = grok_client.generate_custom_report(prompt)
            grok_sections.append(grok_text)
            grok_sections.append("")
        except Exception:
            grok_sections.append("_Grok interpretation unavailable._\n")
    
    return "\n".join(lines + grok_sections)


def _render_recommendation_6(main_results: Dict[str, Any], grok_client: Optional[Any] = None) -> str:
    """Render Recommendation 6: Cross-Modal Coherence Test results."""
    lines: List[str] = []
    grok_sections: List[str] = []
    
    lines.append("## Recommendation 6: Cross-Modal Coherence Test (TE-EE)\n")
    
    # Top-level metadata
    recommendation_id = main_results.get("recommendation_id", "N/A")
    test_name = main_results.get("test_name", "N/A")
    physical_test = main_results.get("physical_test", "")
    note = main_results.get("note", "")
    datasets = main_results.get("datasets", [])
    
    lines.append(f"**Recommendation ID**: {recommendation_id}\n")
    lines.append(f"**Test Name**: {test_name}\n")
    if physical_test:
        lines.append(f"**Physical Test**: {physical_test}\n")
    if note:
        lines.append(f"**Note**: {note}\n")
    if datasets:
        lines.append(f"**Datasets**: {', '.join(datasets)}\n")
    lines.append("")
    
    # QTEP verification results
    qtep_verification = main_results.get("qtep_verification", {})
    if qtep_verification:
        lines.append("### Cross-Modal Coherence Results\n")
        
        test_name_qtep = qtep_verification.get("test_name", "")
        physical_test_qtep = qtep_verification.get("physical_test", "")
        note_qtep = qtep_verification.get("note", "")
        ell_range = qtep_verification.get("ell_range_predicted", [800, 1200])
        spectra = qtep_verification.get("spectra", [])
        
        if test_name_qtep:
            lines.append(f"**Test**: {test_name_qtep}\n")
        if physical_test_qtep:
            lines.append(f"**Physical Basis**: {physical_test_qtep}\n")
        if note_qtep:
            lines.append(f"**Note**: {note_qtep}\n")
        if ell_range:
            lines.append(f"**Multipole Range**: ℓ = {ell_range[0]}-{ell_range[1]}\n")
        if spectra:
            lines.append(f"**Spectra**: {', '.join(spectra)}\n")
        lines.append("")
        
        # Combined results
        combined_full = qtep_verification.get("combined_full", {})
        combined_predicted = qtep_verification.get("combined_predicted", {})
        
        if combined_full:
            lines.append("**Full Multipole Range Results:**\n")
            rho_full = combined_full.get("rho_median", combined_full.get("R_median", np.nan))
            rho_std_full = combined_full.get("rho_std", combined_full.get("R_std", np.nan))
            bf_full = combined_full.get("bayes_factor", np.nan)
            interpretation_full = combined_full.get("interpretation", "")
            
            if not np.isnan(rho_full):
                sig_full = abs(rho_full) / rho_std_full if rho_std_full > 0 else 0
                lines.append(f"- ρ(TE,EE) = {rho_full:.4f} ± {rho_std_full:.4f} ({sig_full:.2f}σ from null)\n")
            if not np.isnan(bf_full):
                lines.append(f"- Bayes factor: {bf_full:.2f}\n")
            if interpretation_full:
                lines.append(f"- Interpretation: {interpretation_full}\n")
            lines.append("")
        
        if combined_predicted:
            lines.append("**Predicted Range (ℓ=800-1200) Results:**\n")
            rho_pred = combined_predicted.get("rho_median", combined_predicted.get("R_median", np.nan))
            rho_std_pred = combined_predicted.get("rho_std", combined_predicted.get("R_std", np.nan))
            bf_pred = combined_predicted.get("bayes_factor", np.nan)
            
            if not np.isnan(rho_pred):
                sig_pred = abs(rho_pred) / rho_std_pred if rho_std_pred > 0 else 0
                lines.append(f"- ρ(TE,EE) = {rho_pred:.4f} ± {rho_std_pred:.4f} ({sig_pred:.2f}σ from null)\n")
            if not np.isnan(bf_pred):
                lines.append(f"- Bayes factor: {bf_pred:.2f}\n")
            lines.append("")
        
        # Survey-level results summary
        surveys = qtep_verification.get("surveys", {})
        if surveys:
            lines.append("**Survey-Level Results:**\n")
            for survey_name, survey_data in surveys.items():
                lines.append(f"- **{survey_name.upper()}**:\n")
                
                # Correlation fit results
                fit_full = survey_data.get("correlation_fit_full", survey_data.get("qtep_fit_full", {}))
                if fit_full:
                    rho_mean = fit_full.get("rho_mean", fit_full.get("R_mean", np.nan))
                    rho_std = fit_full.get("rho_std", fit_full.get("R_std", np.nan))
                    if not np.isnan(rho_mean):
                        lines.append(f"  - ρ = {rho_mean:.4f} ± {rho_std:.4f}\n")
                
                # Bayes factor
                bf = survey_data.get("bayes_factor_full", {})
                if bf and not np.isnan(bf.get("bayes_factor", np.nan)):
                    lines.append(f"  - Bayes factor: {bf.get('bayes_factor', np.nan):.2f} ({bf.get('interpretation', 'N/A')})\n")
                
                # Consistency check
                consistency = survey_data.get("hlcdm_consistency_full", {})
                if consistency:
                    rho_fitted = consistency.get("rho_fitted", consistency.get("R_fitted", np.nan))
                    consistent = consistency.get("consistent_with_hlcdm", False)
                    sig = consistency.get("significance_vs_null", np.nan)
                    if not np.isnan(sig):
                        lines.append(f"  - Significance vs null: {sig:.2f}σ\n")
                    lines.append(f"  - Consistent with H-ΛCDM: {consistent}\n")
            lines.append("")
    
    # Grok interpretation
    if grok_client:
        grok_sections.append("### Scientific Interpretation (AI Generated)\n")
        prompt = (
            "You are a senior theoretical physicist analyzing cross-modal coherence test results. "
            "Write in third-person, dispassionate academic style. "
            "Interpret the cross-modal correlation ρ(TE,EE) results:\n"
            f"- Observed ρ = {combined_full.get('rho_median', combined_full.get('R_median', np.nan)):.4f} "
            f"± {combined_full.get('rho_std', combined_full.get('R_std', np.nan)):.4f}\n"
            f"- Bayes factor: {combined_full.get('bayes_factor', np.nan):.2f}\n"
            f"- Interpretation: {combined_full.get('interpretation', 'N/A')}\n\n"
            "H-ΛCDM predicts ρ > 0 (correlated residuals from Lindblad-Zeno mechanism). "
            "ΛCDM predicts ρ ≈ 0 (independent Gaussian noise). "
            "Assess whether the results support H-ΛCDM or are consistent with ΛCDM null. "
            "Be precise about statistical significance and physical interpretation. "
            "Max 3 paragraphs."
        )
        try:
            grok_text = grok_client.generate_custom_report(prompt)
            grok_sections.append(grok_text)
            grok_sections.append("")
        except Exception:
            grok_sections.append("_Grok interpretation unavailable._\n")
    
    return "\n".join(lines + grok_sections)


def _render_legacy_results(main_results: Dict[str, Any], grok_client: Optional[Any] = None) -> str:
    """Render legacy recommendation format (for backward compatibility)."""
    # Skip if this looks like new format (has top-level metadata fields)
    if any(key in main_results for key in ["recommendation_id", "test_name", "qtep_verification", "physical_test"]):
        return "No legacy recommendation results available (new format detected).\n"
    
    recs = main_results.get("recommendations", main_results)
    if not isinstance(recs, dict) or len(recs) == 0:
        return "No recommendation results available.\n"

    lines: List[str] = []
    grok_sections: List[str] = []
    rec_summaries_for_grok = []

    lines.append("## Recommendation Results\n")
    for rec_id, rec in recs.items():
        # Skip top-level metadata fields that aren't recommendation entries
        if rec_id in ["recommendation_id", "test_name", "physical_test", "note", "datasets", "qtep_verification", "validation"]:
            continue
            
        lines.append(f"### {rec_id}\n")
        if not isinstance(rec, dict):
            lines.append(f"- Status: ERROR — Invalid result format (expected dict, got {type(rec).__name__})\n")
            continue
        if rec.get("error"):
            lines.append(f"- Status: ERROR — {rec.get('error')}\n")
            continue

        amp = rec.get("fourier_summary", {}).get("modulation_amplitude_pct")
        gamma_ratio = rec.get("gamma_ratio")
        z_eff = rec.get("z_eff")
        ell_band = rec.get("ell_band", [])
        dataset = rec.get("dataset", "unknown")
        target_power = rec.get("fourier_summary", {}).get("target_power")
        power_ratio = rec.get("fourier_summary", {}).get("power_ratio")
        anomalies = rec.get("anomaly_indices", [])

        lines.append(f"- Dataset: {dataset}")
        if ell_band:
            lines.append(f"- Multipole band: ℓ ∈ [{ell_band[0]}, {ell_band[1]}]")
        if z_eff is not None:
            lines.append(f"- Effective redshift (ML anomalies): {z_eff:.3f}")
        if amp is not None:
            lines.append(f"- Modulation amplitude: {amp:.3f}%")
        if gamma_ratio is not None:
            lines.append(f"- γ_obs / γ_theory: {gamma_ratio:.3f}" if gamma_ratio == gamma_ratio else "- γ_obs / γ_theory: nan")
        if target_power is not None:
            lines.append(f"- Target Fourier power: {target_power:.3e}")
        if power_ratio is not None:
            lines.append(f"- Target/median power ratio: {power_ratio:.3f}")
        if anomalies:
            lines.append(f"- Linked ML anomalies: {', '.join(str(a) for a in anomalies)}")

        rec_summaries_for_grok.append(
            {
                "id": rec_id,
                "dataset": dataset,
                "ell_band": ell_band,
                "z_eff": z_eff,
                "modulation_pct": amp,
                "gamma_ratio": gamma_ratio,
                "power_ratio": power_ratio,
                "anomalies": anomalies,
            }
        )

        lines.append("")  # blank line

    # Grok interpretation block
    if grok_client and rec_summaries_for_grok:
        grok_sections.append("### Scientific Interpretation (AI Generated)\n")
        prompt = (
            "You are a senior cosmologist drafting a formal journal note. "
            "NEVER use first person; write in third-person, dispassionate academic style. "
            "Provide a concise interpretation of whether the detected modulation amplitudes "
            "and γ_obs/γ_theory ratios support or refute H-ΛCDM holographic corrections. "
            "Use the supplied summaries as the only empirical inputs: "
            f"{rec_summaries_for_grok}. Ground the discussion in scaling relations "
            "(γ = H/π², modulation amplitude in percent), statistical weight, and linkage "
            "to ML-flagged anomalies. State conclusions and caveats as declarative, "
            "evidence-weighted findings."
        )
        try:
            grok_text = grok_client.generate_custom_report(prompt)
            grok_sections.append(grok_text)
            grok_sections.append("")
        except Exception:
            grok_sections.append("_Grok interpretation unavailable._\n")

    return "\n".join(lines + grok_sections)


def validation(results: Dict[str, Any]) -> str:
    """Render validation block."""
    val = results.get("validation", {})
    if not val:
        # Check if this is new format but validation wasn't run
        if "cross_modal_coherence" in results:
            lines = ["## Validation\n"]
            lines.append("Validation not yet performed. Run pipeline with 'validate' option.\n")
            lines.append("")
            return "\n".join(lines)
        return ""  # Return empty string if no validation results

    lines = ["## Validation\n"]
    
    # Bootstrap validation (new format)
    bootstrap = val.get("bootstrap", {})
    if bootstrap and not bootstrap.get("error"):
        lines.append("### Bootstrap Analysis\n")
        lines.append("_Bootstrap resampling of cross-modal coherence ρ(TE,EE) to assess stability._\n")
        
        n_bootstrap = bootstrap.get("n_bootstrap", bootstrap.get("n_successful", "N/A"))
        rho_observed = bootstrap.get("rho_observed", np.nan)
        rho_bootstrap_mean = bootstrap.get("rho_bootstrap_mean", np.nan)
        rho_bootstrap_std = bootstrap.get("rho_bootstrap_std", np.nan)
        rho_ci_95 = bootstrap.get("rho_ci_95", [np.nan, np.nan])
        within_ci = bootstrap.get("within_ci_95", False)
        stability_ok = bootstrap.get("stability_ok", False)
        
        lines.append(f"- Bootstrap iterations: {n_bootstrap}")
        if not np.isnan(rho_observed):
            lines.append(f"- Observed ρ(TE,EE): {rho_observed:.4f}")
        if not np.isnan(rho_bootstrap_mean):
            lines.append(f"- Bootstrap mean ρ: {rho_bootstrap_mean:.4f} ± {rho_bootstrap_std:.4f}")
        if not np.isnan(rho_ci_95[0]):
            lines.append(f"- 95% credible interval: [{rho_ci_95[0]:.4f}, {rho_ci_95[1]:.4f}]")
        lines.append(f"- Observed value within 95% CI: {within_ci}")
        lines.append(f"- Stability check: {stability_ok}")
        lines.append(f"- **Status**: {bootstrap.get('interpretation', 'N/A')}")
        lines.append("")
    elif bootstrap.get("error"):
        lines.append("### Bootstrap Analysis\n")
        lines.append(f"**Error**: {bootstrap.get('error')}\n")
        lines.append("")
    
    # Null hypothesis testing (new format)
    null_hypothesis = val.get("null_hypothesis", {})
    if null_hypothesis and not null_hypothesis.get("error"):
        lines.append("### Null Hypothesis Testing\n")
        lines.append("_Tests H₀: ρ(TE,EE) = 0 (ΛCDM) vs H₁: ρ(TE,EE) > 0 (H-ΛCDM)_\n")
        
        n_null = null_hypothesis.get("n_null", "N/A")
        rho_observed = null_hypothesis.get("rho_observed", np.nan)
        rho_null_mean = null_hypothesis.get("rho_null_mean", np.nan)
        rho_null_std = null_hypothesis.get("rho_null_std", np.nan)
        p_value = null_hypothesis.get("p_value", np.nan)
        significance_sigma = null_hypothesis.get("significance_sigma", np.nan)
        null_rejected = null_hypothesis.get("null_rejected", False)
        evidence_strength = null_hypothesis.get("evidence_strength", "N/A")
        
        lines.append(f"- Null simulations: {n_null}")
        if not np.isnan(rho_observed):
            lines.append(f"- Observed ρ(TE,EE): {rho_observed:.4f}")
        if not np.isnan(rho_null_mean):
            lines.append(f"- Null distribution mean: {rho_null_mean:.4f} ± {rho_null_std:.4f}")
        if not np.isnan(p_value):
            lines.append(f"- **p-value**: {p_value:.4f}")
        if not np.isnan(significance_sigma):
            lines.append(f"- **Significance**: {significance_sigma:.2f}σ from null")
        lines.append(f"- Null hypothesis rejected (p < 0.05): {null_rejected}")
        lines.append(f"- Evidence strength: {evidence_strength}")
        lines.append(f"- **Interpretation**: {null_hypothesis.get('interpretation', 'N/A')}")
        lines.append("")
        
        # Legacy format compatibility (for backward compatibility)
        null_lcdm = null_hypothesis.get("lcdm", {})
        if null_lcdm:
            lines.append("**Legacy Format (ΛCDM):**\n")
            lines.append(f"- Iterations: {null_lcdm.get('n_null', 'N/A')}")
            lines.append(f"- Observed χ²: {null_lcdm.get('chi2_obs', 'N/A'):.2f}")
            lines.append(f"- Null χ² mean: {null_lcdm.get('chi2_null_mean', 'N/A'):.2f} ± {null_lcdm.get('chi2_null_std', 'N/A'):.2f}")
            lines.append(f"- p-value: {null_lcdm.get('p_value_chi2', 'N/A'):.4f}")
            lines.append(f"- Interpretation: {null_lcdm.get('interpretation_chi2', 'N/A')}")
            lines.append("")
    elif null_hypothesis.get("error"):
        lines.append("### Null Hypothesis Testing\n")
        lines.append(f"**Error**: {null_hypothesis.get('error')}\n")
        lines.append("")
    
    # Legacy format (for backward compatibility with old results)
    bootstrap_lcdm = bootstrap.get("lcdm", {}) if isinstance(bootstrap, dict) else {}
    bootstrap_hlcdm = bootstrap.get("hlcdm", {}) if isinstance(bootstrap, dict) else {}
    
    if bootstrap_lcdm and not bootstrap.get("rho_observed"):  # Only if new format not present
        lines.append("### Bootstrap Analysis (Legacy Format)\n")
        lines.append("**ΛCDM Bootstrap:**\n")
        lines.append(f"- Iterations: {bootstrap_lcdm.get('n_bootstrap', 'N/A')}")
        lines.append(f"- Successful: {bootstrap_lcdm.get('n_successful', 'N/A')}")
        lines.append(f"- χ² mean: {bootstrap_lcdm.get('chi2_mean', 'N/A'):.2f} ± {bootstrap_lcdm.get('chi2_std', 'N/A'):.2f}")
        lines.append("")
    
    if bootstrap_hlcdm and not bootstrap.get("rho_observed"):  # Only if new format not present
        lines.append("**H-ΛCDM Bootstrap:**\n")
        lines.append(f"- Iterations: {bootstrap_hlcdm.get('n_bootstrap', 'N/A')}")
        lines.append(f"- Successful: {bootstrap_hlcdm.get('n_successful', 'N/A')}")
        lines.append(f"- χ² mean: {bootstrap_hlcdm.get('chi2_mean', 'N/A'):.2f} ± {bootstrap_hlcdm.get('chi2_std', 'N/A'):.2f}")
        lines.append("")
    
    # Overall validation status
    overall_status = val.get("overall_status", "")
    if overall_status:
        lines.append(f"**Overall Validation Status**: {overall_status}\n")
        lines.append("")
    
    return "\n".join(lines)


def conclusion(results: Dict[str, Any], overall_status: str = "") -> str:
    """Render conclusion. Note: Header is added by base reporter, not here."""
    lines = []
    
    # Check if new format
    if "cross_modal_coherence" in results:
        conclusion_data = results.get("conclusion", {})
        conclusion_text = conclusion_data.get("conclusion", "N/A")
        evidence = conclusion_data.get("evidence", [])
        
        lines.append(f"**{conclusion_text}**\n")
        
        if evidence:
            lines.append("**Evidence:**\n")
            for item in evidence:
                lines.append(f"- {item}")
            lines.append("")
        
        lines.append(f"**Coherence:** {conclusion_data.get('coherence_summary', 'N/A')}\n")
        lines.append(f"**Amplitude:** {conclusion_data.get('amplitude_summary', 'N/A')}\n")
        lines.append(f"**ML Validation:** {conclusion_data.get('ml_summary', 'N/A')}\n")
        lines.append("")
        return "\n".join(lines)
    elif "model_comparison" in results:
        comparison = results.get("model_comparison", {})
        preferred = comparison.get("preferred_model", "N/A")
        delta_chi2 = comparison.get("delta_chi2_eff", np.nan)
        
        lines.append(f"**Model comparison results:**\n")
        lines.append(f"- Preferred model: {preferred}")
        lines.append(f"- Δχ²_eff: {delta_chi2:.2f}")
        
        if comparison.get("delta_aic"):
            lines.append(f"- ΔAIC: {comparison['delta_aic']:.2f}")
        if comparison.get("delta_bic"):
            lines.append(f"- ΔBIC: {comparison['delta_bic']:.2f}")
        
        lines.append(f"\n**Interpretation:** {comparison.get('significance', 'N/A')}")
        lines.append("")
        return "\n".join(lines)
    else:
        return "No recommendation conclusions available.\n"
