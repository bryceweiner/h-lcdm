"""CMB Cold Spot QTEP pipeline reporting helpers."""

from typing import Any, Dict
import numpy as np


def _format_survey_results(survey_results: Dict[str, Dict]) -> str:
    """Format survey results for Grok prompt."""
    if not survey_results:
        return "No survey results available"
    
    lines = []
    for survey_name, results in survey_results.items():
        deficit = results.get('deficit', 'N/A')
        uncertainty = results.get('uncertainty', 'N/A')
        lines.append(f"- {survey_name}: δT/T = {deficit} ± {uncertainty}")
    
    return "\n".join(lines)


def _construct_cold_spot_prompt(test_1: Dict[str, Any],
                                test_2: Dict[str, Any],
                                test_3: Dict[str, Any],
                                main_results: Dict[str, Any]) -> str:
    """
    Construct Grok prompt with explicit data grounding to prevent hallucination.
    
    Critical Design Principles:
    1. Provide ALL numerical results explicitly
    2. Define what each test measures physically
    3. Specify ΛCDM null hypothesis numerically
    4. Request interpretation based ONLY on provided data
    5. Require explicit acknowledgment of failures/contradictions
    """
    
    # Format survey results
    survey_results_str = _format_survey_results(main_results.get('survey_results', {}))
    
    prompt = f"""
You are analyzing CMB Cold Spot data to assess if it shows signatures of the QTEP (Quantum-Thermodynamic Entropy Partition) framework.

## CRITICAL: What Each Test Actually Does

**Test 1 - Framework Consistency (NON-CIRCULAR):**
This is NOT a prediction test! It checks if three things are internally consistent:
1. SIGN: Cold Spot (δT < 0) should imply Enhanced Zeno efficiency (δη/η > 0)
2. MAGNITUDE: Is inferred δη/η ≈ 10⁻³? (reasonable CMB fluctuation scale)
3. GLOBAL: Does framework predict 1.4% overall CMB cooling?

**Test 2 - Regime Transitions (NOT discrete peaks!):**
Tests if QTEP-predicted multipoles [18, 40, 91, 207] mark regime boundaries:
- ℓ=18: Sachs-Wolfe → Acoustic transition
- ℓ=40: Acoustic evolution boundary
- ℓ=91: Peak → Damping transition
Uses binomial test: Do these align with detected empirical transitions?
NOTE: Gaussian p=1.0 is EXPECTED (no discrete peaks, just boundaries)

**Test 3 - Cross-Power Spectrum (partially tautological):**
Compares CMB temperature map with QTEP efficiency map at z~1089.
NOTE: This is somewhat circular since QTEP map generated from CMB with δη/η = δT/T
Main test is χ² consistency, not the correlation itself.

---

## QTEP Framework (Zeno Concentration Paradigm)

**Global Effect:** Entire CMB is 1.4% cooler than naive Sachs-Wolfe due to coherent entropy buildup during recombination

**Local Effect:** Cold Spot has ENHANCED Zeno (δη/η > 0), causing ADDITIONAL cooling
- More efficiency → More coherence → Less thermal precipitation → Colder
- Sign: δT < 0 ⟺ δη/η > 0 (colder means more efficient, not less!)

**ΛCDM Alternative:** Cold Spot is a ~3σ Gaussian fluctuation, no special origin

---

## DATA (Analyze ONLY what's provided)

### Test 1: Framework Consistency (Non-Circular)

**Method**: {test_1.get('method', 'Framework consistency test')} — {test_1.get('test_type', 'Sign + Magnitude + Global Zeno consistency')}

**Note**: {test_1.get('note', 'Non-circular: Tests framework consistency, not predictive match')}

**Global Zeno Cooling** (entire CMB):
- Mechanism: Coherent entropy buildup during recombination defers thermal precipitation
- Global suppression: ΔT/T = {test_1.get('global_zeno_suppression', 0):.2%} = {test_1.get('global_zeno_suppression_mK', 0):.1f} mK
- Reference: {test_1.get('reference', 'Weiner (2025) entropy mechanics')}

**Cold Spot as Zeno Concentration**:
- Observed Cold Spot deficit: δT/T = {test_1.get('observed_deficit', 'N/A')}
- Inferred local Zeno enhancement: δη/η = {f"{test_1['local_zeno_enhancement']:.4e}" if test_1.get('local_zeno_enhancement') is not None else 'N/A'}
- Thermal coupling: {f"{test_1['thermal_coupling']:.4f}" if test_1.get('thermal_coupling') is not None else 'N/A'}
- Paradigm: {test_1.get('paradigm', 'Enhanced Zeno → Enhanced cooling')}

**Framework Consistency Checks**:
- Sign check: {test_1.get('sign_check', 'N/A')} (Cold δT<0 should imply Enhanced Zeno δη>0)
- Magnitude check: {test_1.get('magnitude_check', 'N/A')} (Is δη/η ~ 10⁻³ reasonable?)
- Consistency: {test_1.get('consistency_sigma', 'N/A')}σ, p = {test_1.get('p_value', 'N/A')}
- Result: **{test_1.get('result', 'N/A')}**

**Survey consistency**: {test_1.get('cross_survey_agreement', 'N/A')}

---

### Test 2: Angular Power Spectrum Regime Transitions

**Method**: {test_2.get('test_paradigm', 'Testing for QTEP-predicted regime boundaries in power spectrum structure')}

**QTEP Prediction**: Information processing constraints create regime boundaries at multipoles ℓ_n = ℓ₀ × η^n where η ≈ 2.257

**Regime Transition Analysis:**
- QTEP-predicted multipoles: {test_2.get('qtep_multipoles', 'N/A')}
- Detected empirical transitions: {len(test_2.get('detected_transitions', []))} transitions found
- Match rate: {test_2.get('regime_match_fraction', 0):.0%} ({int(test_2.get('regime_match_fraction', 0) * len(test_2.get('qtep_multipoles', [])))} out of {len(test_2.get('qtep_multipoles', []))})
- Statistical significance (binomial test): p = {test_2.get('regime_p_value', 'N/A'):.4f}
- Regime transition result: **{test_2.get('regime_transition_result', 'N/A')}**

**Interpretation**: QTEP multipoles mark WHERE the dominant physics transitions occur (Sachs-Wolfe→Acoustic at ℓ=18, acoustic evolution at ℓ=40, peak→damping at ℓ=91), not discrete amplitude peaks.

**Legacy Discrete Peak Test** (tests wrong prediction):
- Gaussian null p-value: p = {test_2.get('gaussian_p_value', 'N/A')} (No discrete peaks found, as expected)
- Discrete feature score: {test_2.get('discrete_feature_score', 'N/A')}σ (Gaussian fluctuation scale)

**Overall Test Result:** {test_2.get('result', 'N/A')}

---

### Test 3: Angular Cross-Power Spectrum (z~1089)

**Method**: {test_3.get('method', 'Angular cross-power spectrum')} at {test_3.get('epoch', 'recombination')}

**QTEP Prediction**: {test_3.get('prediction_tested', 'C_ℓ^Tη = C_ℓ^TT')}

**Cross-Power Analysis:**
- Multipole range: {test_3.get('multipole_range', 'N/A')}
- Cross-power correlation: r = {test_3.get('cross_power_correlation', 'N/A')}
- Significance: {test_3.get('cross_power_significance', 'N/A')}σ
- QTEP consistency χ²/dof: {test_3.get('qtep_consistency_chi2_per_dof', 'N/A'):.2e}
- Cosmic variance limited: {test_3.get('cosmic_variance_limited', 'N/A')}

**Test Result:** {test_3.get('result', 'N/A')}

**Note**: Test is partially tautological since QTEP map generated from CMB with δη/η = δT/T. The χ² test checks if cross-power spectrum matches this prediction.

---

## CROSS-SURVEY VALIDATION

CMB datasets analyzed: {main_results.get('datasets_analyzed', [])}

**Per-Survey Temperature Deficits:**
{survey_results_str}

**Survey Consistency:**
- χ² test: χ²/dof = {main_results.get('chi_squared_per_dof', 'N/A')}
- Consistency p-value: p = {main_results.get('survey_consistency_p_value', 'N/A')}

---

## YOUR TASK

Based EXCLUSIVELY on the numerical results above, provide:

### 1. Test-by-Test Assessment (100-150 words each)

**Test 1 - Framework Consistency:**
- DO NOT compare "observed vs predicted deficit" - that's circular data!
- FOCUS ON: Are sign check and magnitude check both PASS?
- p-value indicates consistency of inferred δη/η with expected scale
- Result "FRAMEWORK_CONSISTENT" means internal logic holds

**Test 2 - Regime Transitions:**
- DO NOT focus on "Gaussian p=1.0" - that tests wrong prediction!
- FOCUS ON: Regime p-value (binomial test) and match fraction
- p < 0.05 with match fraction >50% = significant detection
- Result "QTEP_REGIME_STRUCTURE" means boundaries detected

**Test 3 - Cross-Power:**
- Note this is partially tautological (QTEP map from CMB)
- FOCUS ON: χ²/dof for consistency with prediction
- High correlation expected, χ² tests if it matches theory

Address alternative explanations and robustness for each.

### 2. Overall Conclusion (200-300 words)

**Primary Question**: Do the data support QTEP or standard ΛCDM?

**Evidence Assessment:**
- Test 1: Does framework hang together? (sign + magnitude checks)
- Test 2: Do regime transitions align with predictions? (binomial p-value)
- Test 3: Is cross-power consistent? (acknowledge tautology)
- Cross-survey: Do different datasets agree? (χ²/dof from table)

**If Tests Disagree**: Explain which metrics matter most and why

**Systematic Concerns**: Large cross-survey χ² indicates calibration issues

### 3. Scientific Verdict

One sentence: "The data [SUPPORTS/CONTRADICTS/IS INCONCLUSIVE FOR] the QTEP hypothesis."

---

## CRITICAL REMINDERS

- Test 1 "predicted deficit" field is LEGACY/CIRCULAR - ignore it!
- Test 2 "Gaussian p=1.0" tests WRONG thing - ignore it!
- Test 3 correlation is expected - focus on χ² consistency
- ACTUAL metrics: sign/magnitude checks, regime p-value, χ²/dof

Be dispassionate. If QTEP fails, say so. If it succeeds, say so. Use only provided numbers.
"""
    
    return prompt


def results(main_results: Dict[str, Any], grok_client) -> str:
    """
    Generate results section using Grok for interpretation.
    
    Parameters:
        main_results: Pipeline results dictionary
        grok_client: GrokAnalysisClient instance (optional)
        
    Returns:
        str: Results section with Grok interpretation + raw data
    """
    # Extract test results
    test_1 = main_results.get("test_1_temperature_deficit", {})
    test_2 = main_results.get("test_2_angular_power_spectrum", {})
    test_3 = main_results.get("test_3_spatial_correlation", {})
    
    # Build Grok prompt
    prompt = _construct_cold_spot_prompt(test_1, test_2, test_3, main_results)
    
    # Generate interpretation
    grok_interpretation = ""
    if grok_client:
        try:
            grok_interpretation = grok_client.generate_custom_report(prompt)
        except Exception as e:
            grok_interpretation = f"Grok interpretation unavailable: {e}"
    else:
        grok_interpretation = "Grok interpretation unavailable (no Grok client)"
    
    # Generate raw data tables
    raw_data_section = _generate_raw_data_tables(test_1, test_2, test_3, main_results)
    
    # Combine interpretation with raw data
    return f"""## Grok Scientific Interpretation

{grok_interpretation}

---

## Raw Data Tables

{raw_data_section}
"""


def _generate_raw_data_tables(test_1: Dict[str, Any],
                              test_2: Dict[str, Any],
                              test_3: Dict[str, Any],
                              main_results: Dict[str, Any]) -> str:
    """Generate raw data tables for reproducibility."""
    
    tables = []
    
    # Test 1 table
    tables.append("### Test 1: Framework Consistency (Non-Circular)\n")
    tables.append("| Parameter | Value |")
    tables.append("|-----------|-------|")
    tables.append(f"| Method | {test_1.get('method', 'N/A')} |")
    tables.append(f"| Test type | {test_1.get('test_type', 'N/A')} |")
    tables.append(f"| | |")
    tables.append(f"| **Global Zeno Cooling** | |")
    tables.append(f"| Global suppression (%) | {test_1.get('global_zeno_suppression', 0):.2%} |")
    tables.append(f"| Global ΔT (mK) | {test_1.get('global_zeno_suppression_mK', 0):.1f} |")
    tables.append(f"| | |")
    tables.append(f"| **Cold Spot Local Enhancement** | |")
    tables.append(f"| Observed deficit (δT/T) | {test_1.get('observed_deficit', 'N/A')} |")
    tables.append(f"| Inferred Zeno boost (δη/η) | {test_1.get('local_zeno_enhancement', 'N/A'):.4e} |")
    tables.append(f"| Thermal coupling | {test_1.get('thermal_coupling', 'N/A'):.4f} |")
    tables.append(f"| | |")
    tables.append(f"| **Consistency Checks** | |")
    tables.append(f"| Sign check | {test_1.get('sign_check', 'N/A')} (δT<0 ⇒ δη>0) |")
    tables.append(f"| Magnitude check | {test_1.get('magnitude_check', 'N/A')} (δη/η~10⁻³) |")
    tables.append(f"| Consistency (σ) | {test_1.get('consistency_sigma', 'N/A')} |")
    tables.append(f"| p-value | {test_1.get('p_value', 'N/A')} |")
    tables.append(f"| Result | **{test_1.get('result', 'N/A')}** |")
    tables.append("")
    
    # Test 2 table
    tables.append("### Test 2: Angular Power Spectrum Regime Transitions\n")
    tables.append("| Parameter | Value |")
    tables.append("|-----------|-------|")
    tables.append(f"| Test paradigm | {test_2.get('test_paradigm', 'Regime transitions')} |")
    tables.append(f"| QTEP-predicted multipoles | {test_2.get('qtep_multipoles', 'N/A')} |")
    tables.append(f"| Empirical transitions detected | {len(test_2.get('detected_transitions', []))} |")
    tables.append(f"| Match fraction | {test_2.get('regime_match_fraction', 0):.0%} ({int(test_2.get('regime_match_fraction', 0) * len(test_2.get('qtep_multipoles', [])))} of {len(test_2.get('qtep_multipoles', []))}) |")
    tables.append(f"| Regime p-value (binomial) | {test_2.get('regime_p_value', 'N/A'):.4f} |")
    tables.append(f"| Regime transition result | **{test_2.get('regime_transition_result', 'N/A')}** |")
    tables.append(f"| Overall result | **{test_2.get('result', 'N/A')}** |")
    tables.append(f"| | |")
    tables.append(f"| *Legacy discrete peak test* | *(tests wrong prediction)* |")
    tables.append(f"| Gaussian p-value | {test_2.get('gaussian_p_value', 'N/A')} (no peaks, expected) |")
    tables.append(f"| Discrete feature score (σ) | {test_2.get('discrete_feature_score', 'N/A')} |")
    tables.append("")
    
    # Test 3 table
    tables.append("### Test 3: Angular Cross-Power Spectrum (z~1089)\n")
    tables.append("| Parameter | Value |")
    tables.append("|-----------|-------|")
    tables.append(f"| Method | {test_3.get('method', 'N/A')} |")
    tables.append(f"| Epoch | {test_3.get('epoch', 'N/A')} |")
    tables.append(f"| Prediction tested | {test_3.get('prediction_tested', 'N/A')} |")
    tables.append(f"| Multipole range | {test_3.get('multipole_range', 'N/A')} |")
    tables.append(f"| Cross-power correlation (r) | {test_3.get('cross_power_correlation', 'N/A')} |")
    tables.append(f"| Significance (σ) | {test_3.get('cross_power_significance', 'N/A')} |")
    chi2_val = test_3.get('qtep_consistency_chi2_per_dof')
    chi2_str = f"{chi2_val:.2e}" if chi2_val is not None else 'N/A'
    tables.append(f"| QTEP consistency χ²/dof | {chi2_str} |")
    tables.append(f"| Cosmic variance limited | {test_3.get('cosmic_variance_limited', 'N/A')} |")
    tables.append(f"| Result | {test_3.get('result', 'N/A')} |")
    tables.append("")
    
    # Cross-survey table
    survey_results = main_results.get('survey_results', {})
    if survey_results:
        tables.append("### Cross-Survey Validation\n")
        tables.append("| Survey | Temperature Deficit (δT/T) | Uncertainty |")
        tables.append("|--------|---------------------------|-------------|")
        for survey_name, results in survey_results.items():
            deficit = results.get('deficit', 'N/A')
            uncertainty = results.get('uncertainty', 'N/A')
            tables.append(f"| {survey_name} | {deficit} | {uncertainty} |")
        tables.append("")
        tables.append(f"| χ²/dof | {main_results.get('chi_squared_per_dof', 'N/A')} |")
        tables.append(f"| Consistency p-value | {main_results.get('survey_consistency_p_value', 'N/A')} |")
    
    return "\n".join(tables)


def summary(main_results: Dict[str, Any]) -> str:
    """Short summary for comprehensive report."""
    formatted = ""
    
    test_1 = main_results.get("test_1_temperature_deficit", {})
    test_2 = main_results.get("test_2_angular_power_spectrum", {})
    test_3 = main_results.get("test_3_spatial_correlation", {})
    
    formatted += f"- **Test 1 (Temperature Deficit):** {test_1.get('result', 'N/A')} (p = {test_1.get('p_value', 'N/A')})\n"
    formatted += f"- **Test 2 (Angular Power Spectrum):** {test_2.get('result', 'N/A')} (Gaussian p = {test_2.get('gaussian_p_value', 'N/A')})\n"
    formatted += f"- **Test 3 (Spatial Correlation):** {test_3.get('result', 'N/A')} (p = {test_3.get('random_location_p_value', 'N/A')})\n"
    formatted += f"- **Datasets analyzed:** {len(main_results.get('datasets_analyzed', []))}\n"
    
    return formatted


def conclusion(main_results: Dict[str, Any], overall_status: str) -> str:
    """CMB Cold Spot pipeline conclusion text."""
    test_1 = main_results.get("test_1_temperature_deficit", {})
    test_2 = main_results.get("test_2_angular_power_spectrum", {})
    test_3 = main_results.get("test_3_spatial_correlation", {})
    
    # Determine overall assessment from test results
    results_list = [
        test_1.get('result', 'UNKNOWN'),
        test_2.get('result', 'UNKNOWN'),
        test_3.get('result', 'UNKNOWN')
    ]
    
    # Count consistent vs inconsistent
    n_consistent = sum(1 for r in results_list if 'CONSISTENT' in r or 'DISCRETE' in r or 'CORRELATED' in r)
    n_inconsistent = sum(1 for r in results_list if 'INCONSISTENT' in r or 'CONTINUOUS' in r or 'UNCORRELATED' in r)
    
    conclusion_text = "### CMB Cold Spot QTEP Analysis Summary\n\n"
    
    if n_consistent >= 2:
        conclusion_text += "**EVIDENCE FOR QTEP ORIGIN** - Multiple tests support the QTEP hypothesis.\n\n"
    elif n_inconsistent >= 2:
        conclusion_text += "**EVIDENCE AGAINST QTEP ORIGIN** - Multiple tests contradict the QTEP hypothesis.\n\n"
    else:
        conclusion_text += "**INCONCLUSIVE** - Mixed results across tests.\n\n"
    
    conclusion_text += f"Test results: {', '.join(results_list)}\n\n"
    conclusion_text += f"Validation status: **{overall_status}**.\n\n"
    conclusion_text += "See Grok Scientific Interpretation section for detailed analysis."
    
    return conclusion_text

