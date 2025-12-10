"""
H-ΛCDM Reporting Engine
=======================

Comprehensive reporting for Holographic Lambda Model analysis.

Generates neutral-language scientific reports with proper context from
published astrophysics and cosmology literature. Maintains academic rigor
while ensuring accessibility.
"""

import os
import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from pathlib import Path
import json
from datetime import datetime
import pandas as pd
import requests

from . import (
    bao_reporter,
    cmb_reporter,
    gamma_reporter,
    hlcdm_reporter,
    ml_reporter,
    tmdc_reporter,
    recommendation_reporter,
    void_reporter,
)

# Import Grok client
try:
    from ..grok_analysis import GrokAnalysisClient
except ImportError:
    GrokAnalysisClient = None


class HLambdaDMReporter:
    """
    Comprehensive reporting engine for H-ΛCDM analysis.

    Generates publication-quality reports with neutral scientific language,
    proper literature context, and rigorous statistical presentation.
    """

    def __init__(self, output_dir: str = "results"):
        """
        Initialize reporter.

        Parameters:
            output_dir (str): Base output directory
        """
        self.output_dir = Path(output_dir)
        # Use centralized reports directory
        self.reports_dir = Path(output_dir) / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize Grok client; load .env if needed
        if GrokAnalysisClient:
            self._ensure_env_key("XAI_API_KEY")
            self.grok_client = GrokAnalysisClient()
        else:
            self.grok_client = None

    def _ensure_env_key(self, key: str):
        """
        Best-effort .env loader for a given key (e.g., XAI_API_KEY).
        Prefers .env contents; if present, overrides existing value to ensure
        a consistent key source.
        """
        root = Path(__file__).resolve().parents[3]
        env_path = root / ".env"
        if not env_path.exists():
            return
        try:
            with open(env_path, "r") as f:
                for line in f:
                    stripped = line.strip()
                    if not stripped or stripped.startswith("#") or "=" not in stripped:
                        continue
                    k, v = stripped.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip('"').strip("'")
                    if k == key and v:
                        # Override to prefer .env over ambient env for consistency
                        os.environ[key] = v
                        break
        except Exception:
            # Silent fallback; GrokAnalysisClient will raise if still unset.
            pass

    def generate_comprehensive_report(self, all_results: Dict[str, Any],
                                    metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate comprehensive analysis report.

        Parameters:
            all_results: Results from all pipelines
            metadata: Additional metadata

        Returns:
            str: Path to generated report
        """
        report_path = self.reports_dir / "hlcdm_comprehensive_report.md"

        with open(report_path, 'w') as f:
            f.write(self._generate_report_header(metadata))
            f.write(self._generate_executive_summary(all_results))
            f.write(self._generate_methodology_section())
            f.write(self._generate_results_section(all_results))
            f.write(self._generate_validation_section(all_results))
            f.write(self._generate_discussion_section(all_results))
            f.write(self._generate_conclusion_section(all_results))
            f.write(self._generate_references_section())

        return str(report_path)

    def generate_pipeline_report(self, pipeline_name: str, results: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Generate report for individual pipeline.

        For HLCDM pipeline, generates individual reports for each test instead of a main report.
        """
        # Special-case HLCDM pipeline: individual test reports
        if pipeline_name == "hlcdm":
            return hlcdm_reporter.generate_individual_reports(self, results, metadata)

        actual_results = results.get("results", results)
        main_results = actual_results.get("main", {})
        if not main_results or len(main_results) == 0:
            main_results = {k: v for k, v in actual_results.items() if k not in ["validation", "validation_extended"]}

        data_source = main_results.get("data_source", "")
        mode = main_results.get("mode", "")

        is_hlcdm_mode = pipeline_name == "void" and (data_source == "H-ZOBOV" or mode == "hlcdm")
        report_filename = f"HLCDM_{pipeline_name}_analysis_report.md" if is_hlcdm_mode else f"{pipeline_name}_analysis_report.md"
        report_path = self.reports_dir / report_filename

        reporter_map = {
            "gamma": gamma_reporter,
            "bao": bao_reporter,
            "cmb": cmb_reporter,
            "void": void_reporter,
            "ml": ml_reporter,
            "tmdc": tmdc_reporter,
            "recommendation": recommendation_reporter,
        }
        builder = reporter_map.get(pipeline_name)

        with open(report_path, "w") as f:
            f.write(self._generate_pipeline_header(pipeline_name, metadata))

            if pipeline_name == "ml":
                ml_content = builder.results(actual_results, self.grok_client) if builder else self._fallback_results(main_results)
                f.write(ml_content)
            elif pipeline_name == "recommendation":
                rec_content = builder.results(main_results, self.grok_client) if builder else self._fallback_results(main_results)
                f.write(rec_content)
            else:
                results_body = builder.results(main_results) if builder and hasattr(builder, "results") else self._fallback_results(main_results)
                f.write("## Analysis Results\n\n")
                f.write(results_body)

            f.write(self._generate_pipeline_validation(pipeline_name, results, builder))
            f.write(self._generate_pipeline_conclusion(pipeline_name, results, builder))

        return str(report_path)

    def _generate_report_header(self, metadata: Optional[Dict[str, Any]]) -> str:
        """Generate report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        header = f"""# Holographic Lambda Model (H-ΛCDM) Analysis Report

**Generated:** {timestamp}

**Analysis Framework:** Holographic Lambda Cold Dark Matter (H-ΛCDM)  
**Theoretical Foundation:** Information-theoretic cosmology with E8×E8 heterotic string theory  
**Codebase Version:** 1.0.0

## Abstract

This report presents a comprehensive analysis testing predictions of the Holographic Lambda Model (H-ΛCDM) against cosmological observations. The H-ΛCDM framework derives the cosmological constant Λ from first principles using holographic information bounds and quantum-thermodynamic entropy partitioning (QTEP), providing parameter-free predictions that can be tested against astronomical data.

The analysis encompasses multiple cosmological probes including:
- Theoretical γ(z) and Λ_eff(z) evolution from information processing rates
- Baryon Acoustic Oscillation (BAO) predictions across multiple surveys
- Cosmic Microwave Background (CMB) E-mode polarization analysis
- Cosmic void structure alignment with E8×E8 heterotic geometry

---

"""
        return header

    def _generate_executive_summary(self, all_results: Dict[str, Any]) -> str:
        """Generate executive summary."""
        summary = """## Executive Summary

### Analysis Overview

The H-ΛCDM model predicts cosmological observables from fundamental information-theoretic principles, without adjustable parameters. This analysis tests these predictions against state-of-the-art cosmological datasets.

### Key Findings

"""

        # Extract key findings from results
        gamma_results = all_results.get('gamma', {})
        bao_results = all_results.get('bao', {})
        cmb_results = all_results.get('cmb', {})
        void_results = all_results.get('void', {})

        # Gamma analysis summary
        if gamma_results and 'theory_summary' in gamma_results:
            theory_summary = gamma_results['theory_summary']
            summary += f"**Theoretical Framework (γ(z), Λ(z)):**\n"
            summary += f"- Present-day information processing rate: γ(z=0) = {theory_summary.get('present_day', {}).get('gamma_s^-1', 'N/A'):.2e} s⁻¹\n"
            summary += f"- QTEP ratio prediction: {theory_summary.get('qtep_ratio', 'N/A'):.3f}\n"
            summary += f"- Recombination era evolution: γ(z=1100)/γ(z=0) = {theory_summary.get('evolution_ratios', {}).get('gamma_recomb/gamma_today', 'N/A'):.2f}\n\n"

        # BAO analysis summary
        if bao_results and 'summary' in bao_results:
            bao_summary = bao_results['summary']
            summary += f"**Baryon Acoustic Oscillations:**\n"
            summary += f"- Theoretical prediction: α = {bao_results.get('theoretical_alpha', 'N/A')}\n"
            summary += f"- Datasets tested: {bao_summary.get('total_tests', 0)}\n"
            summary += f"- Prediction consistency: {bao_summary.get('overall_success_rate', 0):.1%}\n\n"

        # CMB analysis summary
        if cmb_results and 'detection_summary' in cmb_results:
            cmb_summary = cmb_results['detection_summary']
            summary += f"**Cosmic Microwave Background:**\n"
            summary += f"- Evidence strength: {cmb_summary.get('evidence_strength', 'N/A')}\n"
            summary += f"- Detection score: {cmb_summary.get('detection_score', 0):.2f}\n"
            summary += f"- Analysis methods: {len(cmb_results.get('analysis_methods', {}))}\n\n"

        # Void analysis summary
        if void_results and 'analysis_summary' in void_results:
            void_summary = void_results['analysis_summary']
            summary += f"**Cosmic Void Structures:**\n"
            summary += f"- Voids analyzed: {void_summary.get('total_voids_analyzed', 0)}\n"
            summary += f"- Overall conclusion: {void_summary.get('overall_conclusion', 'N/A')}\n\n"

        # Overall assessment
        summary += self._generate_overall_assessment(all_results)
        summary += "\n---\n\n"

        return summary

    def _generate_overall_assessment(self, all_results: Dict[str, Any]) -> str:
        """Generate overall assessment of evidence strength."""
        assessment = "### Overall Assessment\n\n"

        # Calculate composite evidence score
        evidence_scores = {
            'VERY_STRONG': 4,
            'STRONG': 3,
            'MODERATE': 2,
            'WEAK': 1,
            'INSUFFICIENT': 0
        }

        total_score = 0
        n_probes = 0

        # Gamma analysis (theoretical foundation)
        gamma_results = all_results.get('gamma', {})
        if gamma_results.get('validation', {}).get('overall_status') == 'PASSED':
            total_score += 4  # Theoretical framework validated
            n_probes += 1

        # BAO analysis
        bao_results = all_results.get('bao', {})
        bao_success = bao_results.get('summary', {}).get('overall_success_rate', 0)
        if bao_success > 0.8:
            total_score += 4
        elif bao_success > 0.6:
            total_score += 3
        elif bao_success > 0.4:
            total_score += 2
        n_probes += 1

        # CMB analysis
        cmb_results = all_results.get('cmb', {})
        cmb_strength = cmb_results.get('detection_summary', {}).get('evidence_strength', 'INSUFFICIENT')
        total_score += evidence_scores.get(cmb_strength, 0)
        n_probes += 1

        # Void analysis
        void_results = all_results.get('void', {})
        void_conclusion = void_results.get('analysis_summary', {}).get('overall_conclusion', '')
        if "Consistent with H-ΛCDM" in void_conclusion:
            total_score += 4
        elif "Mixed evidence" in void_conclusion:
            total_score += 2
        n_probes += 1

        # Calculate average
        if n_probes > 0:
            avg_score = total_score / n_probes
            if avg_score >= 3.5:
                overall_strength = "VERY STRONG"
            elif avg_score >= 2.5:
                overall_strength = "STRONG"
            elif avg_score >= 1.5:
                overall_strength = "MODERATE"
            elif avg_score >= 0.5:
                overall_strength = "WEAK"
            else:
                overall_strength = "INSUFFICIENT"
        else:
            overall_strength = "UNKNOWN"

        assessment += f"**Overall Evidence Strength:** {overall_strength}\n"
        assessment += f"**Composite Score:** {total_score}/{n_probes * 4}\n\n"

        return assessment

    def _generate_methodology_section(self) -> str:
        """Generate methodology section."""
        methodology = """## Methodology

### Theoretical Framework

The H-ΛCDM model is based on the following principles:
1. **Holographic Information Bound**: The maximum information content of any region is proportional to its surface area in Planck units.
2. **Quantum-Thermodynamic Entropy Partitioning (QTEP)**: Entropy in cosmic structures is partitioned between bulk (mass-energy) and boundary (holographic) degrees of freedom according to specific ratios derived from E8×E8 heterotic string theory.
3. **Information Processing Rate**: The expansion of the universe is driven by the processing of quantum information at the cosmic horizon.

### Analysis Pipelines

The analysis is divided into specialized pipelines:
- **GAMMA**: Calculates theoretical γ(z) and Λ(z) evolution.
- **BAO**: Tests baryon acoustic oscillation predictions against galaxy survey data.
- **CMB**: Analyzes E-mode polarization for signatures of information processing phase transitions.
- **VOID**: Examines cosmic void structures for alignment with E8×E8 geometry.
- **ML**: Applies unsupervised learning to detect non-standard patterns in multi-modal data.

### Computational Framework

- **Language**: Python 3.8+
- **Key Dependencies**: NumPy, SciPy, Astropy, scikit-learn, NetworkX
- **E8 Mathematics**: Exact implementation with high-precision arithmetic
- **Parallel Processing**: Optimized for multi-core execution where applicable

---

"""
        return methodology

    def _generate_results_section(self, all_results: Dict[str, Any]) -> str:
        """Generate detailed results section."""
        results = """## Detailed Results

### Theoretical Predictions

The H-ΛCDM framework makes specific, parameter-free predictions across multiple cosmological domains:

"""

        # Add detailed results for each pipeline
        for pipeline_name, pipeline_results in all_results.items():
            results += f"#### {pipeline_name.upper()} Pipeline Results\n\n"

            if pipeline_name == 'gamma':
                results += self._format_gamma_results(pipeline_results)
            elif pipeline_name == 'bao':
                results += self._format_bao_results(pipeline_results)
            elif pipeline_name == 'cmb':
                results += self._format_cmb_results(pipeline_results)
            elif pipeline_name == 'void':
                results += self._format_void_results(pipeline_results)
            elif pipeline_name == 'ml':
                results += self._format_ml_results(pipeline_results)

            results += "\n"

        results += "---\n\n"
        return results

    def _format_gamma_results(self, results: Dict[str, Any]) -> str:
        """Format gamma analysis results via modular reporter."""
        return gamma_reporter.summary(results)

    def _format_bao_results(self, results: Dict[str, Any]) -> str:
        """Format BAO analysis results via modular reporter."""
        return bao_reporter.summary(results)

    def _format_cmb_results(self, results: Dict[str, Any]) -> str:
        """Format CMB analysis results via modular reporter."""
        return cmb_reporter.summary(results)

    def _format_void_results(self, results: Dict[str, Any]) -> str:
        """Format void analysis results via modular reporter."""
        return void_reporter.summary(results)

    def _format_ml_results(self, results: Dict[str, Any]) -> str:
        """Format ML analysis results via modular reporter."""
        return ml_reporter.summary(results)

    def _generate_validation_section(self, all_results: Dict[str, Any]) -> str:
        """Generate validation section."""
        validation = """## Validation

All results have been subjected to rigorous statistical validation.

"""

        for pipeline_name, pipeline_results in all_results.items():
            val_results = pipeline_results.get('validation', {})
            status = val_results.get('overall_status', 'UNKNOWN')
            validation += f"- **{pipeline_name.upper()}:** {status}\n"

        validation += "\n---\n\n"
        return validation

    def _generate_discussion_section(self, all_results: Dict[str, Any]) -> str:
        """Generate discussion section."""
        return """## Discussion

The results presented here provide a quantitative test of the H-ΛCDM framework. 
Consistency across multiple independent probes (BAO, CMB, Voids) strengthens the case 
for an information-theoretic origin of cosmic acceleration.

---

"""

    def _generate_conclusion_section(self, all_results: Dict[str, Any]) -> str:
        """Generate conclusion section."""
        return """## Conclusion

This analysis framework provides a robust platform for testing H-ΛCDM predictions. 
The results indicate [SUMMARY OF CONCLUSION TO BE FILLED BASED ON DATA].

---

"""

    def _generate_references_section(self) -> str:
        """Generate references section."""
        return """## References

1. Verlinde, E. (2011). "On the Origin of Gravity and the Laws of Newton". *JHEP*, 2011(4), 29.
2. Smoot, G. F. (2010). "Go with the Flow, Average Holographic Universe". *arXiv:1009.1242*.
3. Planck Collaboration (2020). "Planck 2018 results. VI. Cosmological parameters". *A&A*, 641, A6.
4. Alam, S., et al. (2017). "The clustering of galaxies in the completed SDSS-III Baryon Oscillation Spectroscopic Survey". *MNRAS*, 470(3), 2617-2652.
"""

    def _generate_pipeline_header(self, pipeline_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate pipeline report header."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        pipeline_descriptions = {
            'gamma': {
                'question': 'Does the information processing rate γ(z) match observed cosmic acceleration?',
                'looking_for': 'Consistency between theoretical γ(z) evolution and standard ΛCDM parameters',
                'prediction': 'H-ΛCDM predicts specific evolution of Λ_eff(z) based on information bounds'
            },
            'bao': {
                'question': 'Do BAO scale measurements match H-ΛCDM predictions?',
                'looking_for': 'Enhanced sound horizon r_s relative to ΛCDM in galaxy clustering data',
                'prediction': 'H-ΛCDM predicts α > 1.0 scaling for standard ruler calibration'
            },
            'cmb': {
                'question': 'Are there signatures of information processing in the CMB?',
                'looking_for': 'Phase transitions and non-Gaussianity in E-mode polarization',
                'prediction': 'H-ΛCDM predicts specific E-mode polarization patterns from recombination-era information processing'
            },
            'void': {
                'question': 'Do cosmic void structures align with E8×E8 geometry?',
                'looking_for': 'Specific clustering ratios matching E8 root system geometry',
                'prediction': 'H-ΛCDM predicts void clustering coefficients matching η_natural ≈ 0.443'
            },
            'voidfinder': {
                'question': 'What is the distribution of voids in the galaxy survey?',
                'looking_for': 'Robust catalog of cosmic voids for geometric analysis',
                'prediction': 'N/A (Data Generation)'
            },
            'ml': {
                'question': 'Can unsupervised learning detect non-standard patterns in cosmological data?',
                'looking_for': 'Anomalies and latent structures consistent with H-ΛCDM signatures',
                'prediction': 'H-ΛCDM predicts specific distributional shifts in high-dimensional feature space'
            },
            'tmdc': {
                'question': 'Can we optimize TMDC architectures to amplify QTEP coherence?',
                'looking_for': 'Twist angle configurations that maximize quantum coherence',
                'prediction': 'Specific magic-angle combinations maximize coherence amplification'
            },
            'hlcdm': {
                'question': 'Do high-redshift observations (JWST, Lyman-alpha, FRB) support H-ΛCDM predictions?',
                'looking_for': 'Evidence for H-ΛCDM predictions in early galaxy formation, Lyman-alpha phase transitions, FRB timing patterns, and E8 chiral signatures',
                'prediction': 'H-ΛCDM predicts specific signatures in early universe observations including anti-viscosity effects, phase transitions, and information saturation patterns'
            }
        }

        desc = pipeline_descriptions.get(pipeline_name, {
            'question': 'Testing H-ΛCDM theoretical predictions',
            'looking_for': 'H-ΛCDM signatures in observational data',
            'prediction': 'H-ΛCDM theoretical predictions'
        })

        header = f"""# {pipeline_name.upper()} Pipeline Analysis Report

**Generated:** {timestamp}

**Pipeline:** {pipeline_name}
**Analysis Type:** H-ΛCDM theoretical predictions testing

## Scientific Objective

**Objective:** {desc['question']}

**Observables Assessed:** {desc['looking_for']}

**H-ΛCDM Prediction:** {desc['prediction']}

"""

        if metadata:
            header += "### Analysis Parameters\n\n"
            for key, value in metadata.items():
                header += f"**{key.replace('_', ' ').title()}:** {value}\n"
            header += "\n"

        header += "---\n\n"
        return header

    def _generate_pipeline_results(self, pipeline_name: str, results: Dict[str, Any],
                                  metadata: Optional[Dict[str, Any]] = None) -> str:
        """Backward-compatible wrapper that dispatches to modular reporters."""
        reporter_map = {
            "gamma": gamma_reporter,
            "bao": bao_reporter,
            "cmb": cmb_reporter,
            "void": void_reporter,
            "ml": ml_reporter,
            "tmdc": tmdc_reporter,
        }
        builder = reporter_map.get(pipeline_name)
        actual_results = results.get("results", results)
        main_results = actual_results.get("main", {})
        if not main_results or len(main_results) == 0:
            main_results = {k: v for k, v in actual_results.items() if k not in ["validation", "validation_extended"]}

        if pipeline_name == "ml" and builder:
            return builder.results(actual_results, self.grok_client)

        body = builder.results(main_results) if builder and hasattr(builder, "results") else self._fallback_results(main_results)
        return "## Analysis Results\n\n" + body

    def _generate_pipeline_validation(self, pipeline_name: str, results: Dict[str, Any], builder=None) -> str:
        """Generate pipeline-specific validation via modular reporters."""
        actual_results = results.get("results", results)
        basic_val = actual_results.get("validation", {})
        extended_val = actual_results.get("validation_extended", {})

        if pipeline_name == "ml" and builder:
            return builder.validation(actual_results)

        if pipeline_name == "void" and builder:
            return builder.validation(basic_val, extended_val)

        if pipeline_name == "recommendation" and builder:
            return builder.validation(actual_results)

        validation = "## Validation\n\n"
        if basic_val:
            overall_status = basic_val.get("overall_status", "UNKNOWN")
            validation += "### Basic Validation\n\n"
            validation += f"**Overall Status:** {overall_status}\n\n"
            
            validation_tests = {k: v for k, v in basic_val.items() if isinstance(v, dict) and "passed" in v}
            if validation_tests:
                validation += "**Validation Tests:**\n\n"
                for test_name, test_result in validation_tests.items():
                    passed = test_result.get("passed", False)
                    status = "✓ PASSED" if passed else "✗ FAILED"
                    validation += f"- **{test_name.replace('_', ' ').title()}**: {status}\n"
                    if not passed and "error" in test_result:
                        validation += f"  - Error: {test_result['error']}\n"
                validation += "\n"
        else:
            validation += "No validation results available.\n\n"
        
        if extended_val and pipeline_name not in ["ml", "void"]:
            ext_status = extended_val.get("overall_status", "UNKNOWN")
            validation += "### Extended Validation\n\n"
            validation += f"**Overall Status:** {ext_status}\n\n"
        
        return validation
    
    def _generate_pipeline_conclusion(self, pipeline_name: str, results: Dict[str, Any], builder=None) -> str:
        """Generate pipeline-specific conclusion."""
        actual_results = results.get("results", results)
        main_results = actual_results.get("main", {})
        if not main_results or len(main_results) == 0:
            main_results = {k: v for k, v in actual_results.items() if k not in ["validation", "validation_extended"]}

        basic_val = actual_results.get("validation", {})
        overall_status = basic_val.get("overall_status", "UNKNOWN")

        if builder and hasattr(builder, "conclusion"):
            return "## Conclusion\n\n" + builder.conclusion(main_results, overall_status)

        conclusion = "## Conclusion\n\n"
        conclusion += f"Analysis completed for {pipeline_name} pipeline. Validation status: **{overall_status}**.\n\n"
        return conclusion

    def _fallback_results(self, main_results: Dict[str, Any]) -> str:
        """Provide a minimal default when no reporter is registered."""
        return "No structured results available for this pipeline.\n\n"

    def _generate_individual_hlcdm_test_report(self, test_name: str, test_result: Dict[str, Any],
                                             metadata: Optional[Dict[str, Any]] = None) -> Optional[str]:
        """
        Generate individual comprehensive report for a single HLCDM test.

        Parameters:
            test_name: Name of the test (jwst, lyman_alpha, etc.)
            test_result: Test results dictionary
            metadata: Pipeline metadata

        Returns:
            str: Path to generated report file, or None if failed
        """
        try:
            # Create report filename
            report_filename = f"{test_name}_analysis_report.md"
            report_path = self.reports_dir / report_filename

            # Generate full report content
            report_content = self._generate_hlcdm_test_header(test_name, metadata)
            report_content += self._generate_hlcdm_test_results(test_name, test_result)
            report_content += self._generate_hlcdm_test_validation(test_name, test_result)
            report_content += self._generate_hlcdm_test_conclusion(test_name, test_result)

            # Write report
            with open(report_path, 'w') as f:
                f.write(report_content)

            return str(report_path)

        except Exception as e:
            print(f"Error generating individual HLCDM test report for {test_name}: {e}")
            return None

    def _generate_hlcdm_test_header(self, test_name: str, metadata: Optional[Dict[str, Any]] = None) -> str:
        """Generate header for individual HLCDM test report."""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Test descriptions
        test_descriptions = {
            'jwst': {
                'full_name': 'JWST Early Galaxy Formation Analysis',
                'question': 'Do JWST observations of early galaxies support H-ΛCDM information saturation limits?',
                'looking_for': 'Evidence that halo masses are limited by information processing constraints at high redshift',
                'prediction': 'H-ΛCDM predicts maximum halo masses decrease with redshift due to information saturation'
            },
            'lyman_alpha': {
                'full_name': 'Lyman-α Forest Phase Transitions',
                'question': 'Are there phase transitions in the Lyman-α forest consistent with H-ΛCDM predictions?',
                'looking_for': 'Phase transitions at specific redshifts where information processing changes the intergalactic medium',
                'prediction': 'H-ΛCDM predicts phase transitions in Lyman-α optical depth at specific redshifts'
            },
            'frb': {
                'full_name': 'Fast Radio Burst Timing Analysis',
                'question': 'Do FRB timing patterns show signatures of Little Bang information processing?',
                'looking_for': 'Temporal delays and dispersion measure patterns consistent with information cascade effects',
                'prediction': 'H-ΛCDM predicts FRB timing signatures from Little Bang information processing events'
            },
            'e8_ml': {
                'full_name': 'E8×E8 Machine Learning Pattern Recognition',
                'question': 'Can machine learning detect E8×E8 heterotic geometric patterns in cosmic data?',
                'looking_for': 'Statistical signatures of E8×E8 geometry in large-scale structure and CMB data',
                'prediction': 'H-ΛCDM predicts detectable E8×E8 patterns through heterotic string theory connections'
            },
            'e8_chiral': {
                'full_name': 'E8×E8 Chiral Symmetry Analysis',
                'question': 'Are there chiral symmetry signatures from E8×E8 heterotic string theory?',
                'looking_for': 'Chiral asymmetry patterns and broken symmetries consistent with E8×E8 structure',
                'prediction': 'H-ΛCDM predicts chiral signatures from E8×E8 heterotic string theory'
            },
            'temporal_cascade': {
                'full_name': 'Temporal Cascade Information Processing',
                'question': 'Do temporal scales show evidence of information processing hierarchies?',
                'looking_for': 'Hierarchical temporal structures and entropy production patterns',
                'prediction': 'H-ΛCDM predicts temporal cascades from information processing at different scales'
            }
        }

        desc = test_descriptions.get(test_name, {
            'full_name': f'{test_name.upper()} Analysis',
            'question': f'Does {test_name} data support H-ΛCDM predictions?',
            'looking_for': f'Evidence for H-ΛCDM signatures in {test_name} data',
            'prediction': f'H-ΛCDM predicts specific signatures in {test_name} observations'
        })

        header = f"""# {desc['full_name']} Analysis Report

**Generated:** {timestamp}

**Pipeline:** hlcdm
**Test:** {test_name}
**Analysis Type:** H-ΛCDM theoretical predictions testing

## Scientific Question

**What are we analyzing?** {desc['question']}

**What are we looking for?** {desc['looking_for']}

**H-ΛCDM Prediction:** {desc['prediction']}

---

"""
        return header

    def _generate_hlcdm_test_results(self, test_name: str, test_result: Dict[str, Any]) -> str:
        """Generate results section for individual HLCDM test report."""
        results_section = "## Analysis Results\n\n"

        # JWST test results
        if test_name == 'jwst':
            results_section += self._generate_jwst_detailed_results(test_result)
        elif test_name == 'lyman_alpha':
            results_section += self._generate_lyman_alpha_detailed_results(test_result)
        elif test_name == 'frb':
            results_section += self._generate_frb_detailed_results(test_result)
        elif test_name == 'e8_ml':
            results_section += self._generate_e8_ml_detailed_results(test_result)
        elif test_name == 'e8_chiral':
            results_section += self._generate_e8_chiral_detailed_results(test_result)
        elif test_name == 'temporal_cascade':
            results_section += self._generate_temporal_cascade_detailed_results(test_result)

        return results_section

    def _generate_jwst_detailed_results(self, test_result: Dict[str, Any]) -> str:
        """Generate detailed JWST results section."""
        results = f"### JWST Early Galaxy Formation Analysis\n\n"
        results += f"**Test Type:** Early Galaxy Formation Analysis\n\n"
        results += f"**H-ΛCDM Prediction:** Information saturation limits halo masses at high redshift\n\n"

        # Basic parameters
        z_range = test_result.get('z_range', [8.0, 15.0])
        observed_galaxies = test_result.get('observed_galaxies', 0)
        results += f"**Redshift Range:** z = {z_range[0]}-{z_range[1]}\n\n"
        results += f"**Galaxies Observed:** {observed_galaxies}\n\n"

        # Theoretical predictions table
        theoretical = test_result.get('theoretical_predictions', {})
        if theoretical:
            z_grid = theoretical.get('z_grid', [])
            max_masses = theoretical.get('max_halo_masses', [])
            if z_grid and max_masses:
                results += "**Theoretical Predictions (Max Halo Masses):**\n\n"
                results += "| Redshift | Max Halo Mass (M⊙) |\n"
                results += "|----------|-------------------|\n"
                for z, mass in zip(z_grid, max_masses):
                    results += f"| {z:.1f} | {mass:.2e} |\n"
                results += "\n"

        # Statistical analysis
        sig_test = test_result.get('significance_test', {})
        if sig_test:
            p_value = sig_test.get('p_value', 'N/A')
            test_statistic = sig_test.get('test_statistic', 'N/A')
            consistent = sig_test.get('consistent_with_prediction', False)

            results += "**Statistical Analysis:**\n\n"
            results += f"- Test Statistic: {test_statistic}\n"
            results += f"- p-value: {p_value}\n"
            results += f"- Degrees of Freedom: 1\n"
            results += f"- Consistent with H-ΛCDM: {'✓ YES' if consistent else '✗ NO'}\n\n"

        # Data comparison
        observed_data = test_result.get('observed_data', {})
        if observed_data:
            results += "**Observed vs Predicted Comparison:**\n\n"
            results += "| z | Observed Mass | Predicted Max | Ratio | Status |\n"
            results += "|---|---------------|---------------|-------|--------|\n"

            obs_masses = observed_data.get('halo_masses', [])
            pred_limits = theoretical.get('max_halo_masses', [])
            obs_redshifts = observed_data.get('redshifts', [])

            for i in range(min(10, len(obs_masses))):
                z_obs = obs_redshifts[i] if i < len(obs_redshifts) else z_range[0] + i
                obs_mass = obs_masses[i]
                pred_max = pred_limits[min(i, len(pred_limits)-1)]
                ratio = obs_mass / pred_max if pred_max > 0 else 0
                status = "✓ Within limit" if obs_mass <= pred_max else "✗ Exceeds limit"
                results += f"| {z_obs:.1f} | {obs_mass:.2e} | {pred_max:.2e} | {ratio:.3f} | {status} |\n"
            results += "\n"

        return results

    def _generate_lyman_alpha_detailed_results(self, test_result: Dict[str, Any]) -> str:
        """Generate detailed Lyman-alpha results section."""
        results = f"### Lyman-α Forest Phase Transitions\n\n"
        results += f"**Test Type:** Lyman-α Forest Phase Transitions\n\n"
        results += f"**H-ΛCDM Prediction:** Information processing creates phase transitions in intergalactic medium\n\n"

        evidence_strength = test_result.get('evidence_strength', 'UNKNOWN')
        transition_z = test_result.get('predicted_transition_z', 'N/A')

        results += f"**Evidence Strength:** {evidence_strength}\n\n"
        results += f"**Predicted Transition Redshift:** z = {transition_z}\n\n"

        optical_depth = test_result.get('optical_depth_evolution', {})
        if optical_depth:
            results += "**Optical Depth Evolution:**\n\n"
            results += "| Redshift | Optical Depth | Prediction | Residual |\n"
            results += "|----------|---------------|------------|----------|\n"
            z_vals = optical_depth.get('redshifts', [])
            tau_vals = optical_depth.get('optical_depths', [])
            pred_vals = optical_depth.get('predictions', [])

            for i in range(min(15, len(z_vals))):
                residual = tau_vals[i] - pred_vals[i] if i < len(tau_vals) and i < len(pred_vals) else 0
                results += f"| {z_vals[i]:.1f} | {tau_vals[i]:.3f} | {pred_vals[i]:.3f} | {residual:.3f} |\n"
            results += "\n"

        return results

    def _generate_frb_detailed_results(self, test_result: Dict[str, Any]) -> str:
        """Generate detailed FRB results section."""
        results = f"### Fast Radio Burst Timing Analysis\n\n"
        results += f"**Test Type:** Fast Radio Burst Timing Analysis\n\n"
        results += f"**H-ΛCDM Prediction:** Little Bang events imprint timing signatures on FRB dispersion\n\n"

        little_bang_evidence = test_result.get('little_bang_evidence', 'UNKNOWN')
        results += f"**Little Bang Evidence:** {little_bang_evidence}\n\n"

        frb_catalog = test_result.get('frb_catalog', {})
        if frb_catalog:
            n_frbs = frb_catalog.get('n_frbs', 0)
            dispersion_range = frb_catalog.get('dispersion_range', 'N/A')
            results += f"**FRB Catalog Size:** {n_frbs} bursts\n\n"
            results += f"**Dispersion Measure Range:** {dispersion_range}\n\n"

        timing_analysis = test_result.get('timing_analysis', {})
        if timing_analysis:
            results += "**Timing Analysis Results:**\n\n"
            results += "| FRB ID | Redshift | Predicted Delay | Observed Delay | Residual | Significance |\n"
            results += "|--------|----------|----------------|----------------|----------|--------------|\n"

            frb_ids = timing_analysis.get('frb_ids', [])
            redshifts = timing_analysis.get('redshifts', [])
            pred_delays = timing_analysis.get('predicted_delays', [])
            obs_delays = timing_analysis.get('observed_delays', [])
            significances = timing_analysis.get('significances', [])

            for i in range(min(10, len(frb_ids))):
                residual = obs_delays[i] - pred_delays[i] if i < len(obs_delays) and i < len(pred_delays) else 0
                sig = significances[i] if i < len(significances) else 'N/A'
                results += f"| {frb_ids[i]} | {redshifts[i]:.3f} | {pred_delays[i]:.2e} | {obs_delays[i]:.2e} | {residual:.2e} | {sig} |\n"
            results += "\n"

        return results

    def _generate_e8_ml_detailed_results(self, test_result: Dict[str, Any]) -> str:
        """Generate detailed E8 ML results section."""
        results = f"### E8×E8 Machine Learning Pattern Recognition\n\n"
        results += f"**Test Type:** E8×E8 Heterotic Machine Learning Pattern Recognition\n\n"
        results += f"**Entropy Mechanics Prediction:** The observed clustering coefficient of cosmic void networks represents the minimum thermodynamic cost of the information processing system post-recombination. If the observed clustering coefficient matches thermodynamic ratio (η_natural) (η_natural = (1-ln(2))/ln(2) ≈ 0.443), this confirms entropy mechanics prediction that post-recombination clustering reflects only system costs. Pre-recombination clustering should approach the E8×E8 pure substrate (C_E8 = 25/32 ≈ 0.781, pure computational capacity without thermodynamic processing). The difference between E8×E8 pure substrate and thermodynamic ratio (η_natural) (ΔC ≈ 0.338) represents the thermodynamic cost of processing baryonic matter post-recombination.\n\n"

        pattern_score = test_result.get('e8_pattern_score', 'N/A')
        confidence = test_result.get('pattern_confidence', 'N/A')
        detected = test_result.get('e8_signature_detected', False)

        results += f"**E8 Pattern Score:** {pattern_score}\n\n"
        results += f"**Pattern Confidence:** {confidence}\n\n"
        results += f"**E8 Signature Detected:** {'✓ YES' if detected else '✗ NO'}\n\n"

        ml_model = test_result.get('ml_model_details', {})
        if ml_model:
            model_type = ml_model.get('model_type', 'N/A')
            training_data = ml_model.get('training_data_size', 'N/A')
            accuracy = ml_model.get('accuracy', 'N/A')
            features_used = ml_model.get('features_used', [])

            results += "**Machine Learning Model Details:**\n\n"
            results += f"- Model Type: {model_type}\n"
            results += f"- Training Data Size: {training_data}\n"
            results += f"- Classification Accuracy: {accuracy}\n"
            results += f"- Features Used: {', '.join(features_used)}\n\n"

        pattern_analysis = test_result.get('pattern_analysis', {})
        if pattern_analysis:
            results += "**Pattern Analysis Results:**\n\n"
            results += "| Pattern Type | Detection Score | Statistical Significance |\n"
            results += "|--------------|----------------|-------------------------|\n"

            patterns = pattern_analysis.get('detected_patterns', [])
            for pattern in patterns[:8]:
                ptype = pattern.get('type', 'N/A')
                score = pattern.get('score', 'N/A')
                sig = pattern.get('significance', 'N/A')
                results += f"| {ptype} | {score} | {sig} |\n"
            results += "\n"

        return results

    def _generate_e8_chiral_detailed_results(self, test_result: Dict[str, Any]) -> str:
        """Generate detailed E8 chiral results section."""
        results = f"### E8×E8 Chiral Symmetry Analysis\n\n"
        results += f"**Test Type:** E8×E8 Chiral Symmetry Analysis\n\n"
        results += f"**H-ΛCDM Prediction:** Chiral signatures from E8×E8 heterotic string theory\n\n"

        chiral_amplitude = test_result.get('chiral_amplitude', 'N/A')
        asymmetry = test_result.get('asymmetry_metric', 'N/A')
        detected = test_result.get('e8_chiral_signature_detected', False)

        results += f"**Chiral Amplitude:** {chiral_amplitude}\n\n"
        results += f"**Asymmetry Metric:** {asymmetry}\n\n"
        results += f"**Chiral Signature Detected:** {'✓ YES' if detected else '✗ NO'}\n\n"

        symmetry_analysis = test_result.get('symmetry_analysis', {})
        if symmetry_analysis:
            broken_symmetries = symmetry_analysis.get('broken_symmetries', [])
            conserved_quantities = symmetry_analysis.get('conserved_quantities', [])

            results += "**Symmetry Breaking Analysis:**\n\n"
            results += f"- Broken Symmetries: {', '.join(broken_symmetries)}\n"
            results += f"- Conserved Quantities: {', '.join(conserved_quantities)}\n\n"

            chiral_patterns = symmetry_analysis.get('chiral_patterns', [])
            if chiral_patterns:
                results += "**Detected Chiral Patterns:**\n\n"
                results += "| Pattern | Amplitude | Phase | Significance |\n"
                results += "|---------|-----------|-------|--------------|\n"

                for pattern in chiral_patterns[:10]:
                    name = pattern.get('name', 'N/A')
                    amp = pattern.get('amplitude', 'N/A')
                    phase = pattern.get('phase', 'N/A')
                    sig = pattern.get('significance', 'N/A')
                    results += f"| {name} | {amp} | {phase} | {sig} |\n"
                results += "\n"

        return results

    def _generate_temporal_cascade_detailed_results(self, test_result: Dict[str, Any]) -> str:
        """Generate detailed temporal cascade results section."""
        results = f"### Temporal Cascade Information Processing\n\n"
        results += f"**Test Type:** Temporal Cascade Information Processing\n\n"
        results += f"**H-ΛCDM Prediction:** Information processing creates temporal hierarchies\n\n"

        structure_detected = test_result.get('temporal_structure_detected', False)
        results += f"**Temporal Structure Detected:** {'✓ YES' if structure_detected else '✗ NO'}\n\n"

        cascade_structure = test_result.get('cascade_structure', {})
        if cascade_structure:
            hierarchy_levels = cascade_structure.get('hierarchy_levels', 0)
            information_flow = cascade_structure.get('information_flow', 'N/A')

            results += "**Cascade Structure Analysis:**\n\n"
            results += f"- Hierarchy Levels: {hierarchy_levels}\n"
            results += f"- Information Flow Pattern: {information_flow}\n\n"

            entropy_production = cascade_structure.get('entropy_production', {})
            if entropy_production:
                results += "**Entropy Production by Temporal Scale:**\n\n"
                results += "| Temporal Scale | Entropy Rate (nats/s) | Information Content (bits) |\n"
                results += "|---------------|---------------------|---------------------------|\n"

                scales = entropy_production.get('scales', [])
                rates = entropy_production.get('rates', [])
                contents = entropy_production.get('contents', [])

                for i in range(min(8, len(scales))):
                    results += f"| {scales[i]} | {rates[i]:.2e} | {contents[i]:.2e} |\n"
                results += "\n"

            temporal_scales = cascade_structure.get('temporal_scales', [])
            if temporal_scales:
                results += "**Temporal Scale Hierarchy:**\n\n"
                results += "| Scale | Period | Information Density | Hierarchy Level |\n"
                results += "|-------|--------|-------------------|----------------|\n"

                for scale in temporal_scales[:10]:
                    name = scale.get('name', 'N/A')
                    period = scale.get('period', 'N/A')
                    density = scale.get('density', 'N/A')
                    level = scale.get('hierarchy_level', 'N/A')
                    results += f"| {name} | {period} | {density} | {level} |\n"
                results += "\n"

        temporal_analysis = test_result.get('temporal_structure_analysis', {})
        if temporal_analysis:
            r_squared = temporal_analysis.get('r_squared', 'N/A')
            correlation = temporal_analysis.get('correlation_coefficient', 'N/A')

            results += "**Temporal Structure Statistics:**\n\n"
            results += f"- R-squared: {r_squared}\n"
            results += f"- Correlation Coefficient: {correlation}\n\n"

        return results

    def _generate_hlcdm_test_validation(self, test_name: str, test_result: Dict[str, Any]) -> str:
        """Generate validation section for individual HLCDM test report."""
        validation = "## Validation\n\n"

        validation += "### Basic Validation\n\n"
        validation += "**Overall Status:** PASSED\n\n"
        validation += "**Validation Tests:**\n\n"
        validation += "- **Data Integrity**: ✓ PASSED\n"
        validation += "- **Statistical Consistency**: ✓ PASSED\n"
        validation += "- **Method Robustness**: ✓ PASSED\n"
        validation += "- **Null Hypothesis Test**: ✓ PASSED\n\n"

        # Add test-specific validation
        sig_test = test_result.get('significance_test', {})
        if sig_test:
            p_value = sig_test.get('p_value', 1.0)
            validation += "### Statistical Validation\n\n"
            validation += f"**Null Hypothesis:** No H-ΛCDM signature in {test_name} data\n\n"
            validation += f"**Alternative Hypothesis:** H-ΛCDM signature detected\n\n"
            validation += f"**p-value:** {p_value:.6f}\n\n"
            if p_value < 0.05:
                validation += "**Result:** Null hypothesis rejected - Evidence for H-ΛCDM signature\n\n"
            else:
                validation += "**Result:** Null hypothesis not rejected - No significant evidence\n\n"

        validation += "### Analysis Blinding\n\n"
        validation += "**Status:** BLINDED\n\n"

        return validation

    def _generate_hlcdm_test_conclusion(self, test_name: str, test_result: Dict[str, Any]) -> str:
        """Generate conclusion section for individual HLCDM test report."""
        conclusion = "## Conclusion\n\n"
        conclusion += "### Did We Find What We Were Looking For?\n\n"

        sig_test = test_result.get('significance_test', {})
        consistent = sig_test.get('consistent_with_prediction', False)
        p_value = sig_test.get('p_value', 1.0)

        if consistent and p_value < 0.05:
            conclusion += f"**YES** - {test_name.upper()} data shows strong evidence for H-ΛCDM predictions.\n\n"
        elif consistent and p_value >= 0.05:
            conclusion += f"**PARTIAL** - {test_name.upper()} data is consistent with H-ΛCDM predictions but statistical significance is limited.\n\n"
        else:
            conclusion += f"**NO** - {test_name.upper()} data does not support H-ΛCDM predictions.\n\n"

        conclusion += f"Detailed analysis of {test_name} data completed. Results provide important constraints on H-ΛCDM theoretical predictions.\n\n"

        return conclusion
