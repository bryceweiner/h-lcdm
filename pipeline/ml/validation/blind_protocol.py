"""
Blind Analysis Protocol
======================

Implements blind analysis protocol to prevent confirmation bias
in ML pattern detection. Pre-registers methodology before seeing results.
"""

import json
import hashlib
from datetime import datetime
from typing import Dict, Any, List, Optional, Callable
from pathlib import Path
import logging


class BlindAnalysisProtocol:
    """
    Blind analysis protocol for unbiased ML validation.

    Pre-registers methodology before seeing results, documents
    complete analysis pipeline, and generates unblinding report
    comparing detections to H-ΛCDM theoretical predictions.
    """

    def __init__(self, protocol_file: str = "blind_protocol.json",
                 results_dir: str = "blind_results"):
        """
        Initialize blind analysis protocol.

        Parameters:
            protocol_file: File to save/load protocol
            results_dir: Directory for blind results
        """
        self.protocol_file = Path("ml_pipeline") / protocol_file
        self.results_dir = Path("ml_pipeline") / results_dir
        
        # Ensure directories exist
        self.results_dir.mkdir(parents=True, exist_ok=True)
        if not self.protocol_file.parent.exists():
            self.protocol_file.parent.mkdir(parents=True, exist_ok=True)

        self.protocol_registered = False
        self.protocol_hash = None
        self.analysis_complete = False

        self.logger = logging.getLogger(__name__)

    def register_protocol(self, methodology: Dict[str, Any],
                         research_question: str,
                         success_criteria: Dict[str, Any]) -> Dict[str, Any]:
        """
        Register blind analysis protocol before seeing results.

        Parameters:
            methodology: Complete methodology description
            research_question: Research question being addressed
            success_criteria: Pre-registered success criteria

        Returns:
            dict: Registration confirmation
        """
        protocol = {
            'registration_timestamp': datetime.now().isoformat(),
            'research_question': research_question,
            'methodology': methodology,
            'success_criteria': success_criteria,
            'protocol_version': '1.0',
            'blind_status': 'REGISTERED'
        }

        # Create hash of protocol for integrity checking
        protocol_str = json.dumps(protocol, sort_keys=True, default=str)
        self.protocol_hash = hashlib.sha256(protocol_str.encode()).hexdigest()
        protocol['protocol_hash'] = self.protocol_hash

        # Save protocol
        with open(self.protocol_file, 'w') as f:
            json.dump(protocol, f, indent=2, default=str)

        self.protocol_registered = True

        self.logger.info(f"Blind analysis protocol registered with hash: {self.protocol_hash}")

        return {
            'registration_success': True,
            'protocol_hash': self.protocol_hash,
            'registration_timestamp': protocol['registration_timestamp']
        }

    def execute_blind_analysis(self, analysis_function: Callable,
                             analysis_params: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute analysis while maintaining blindness to H-ΛCDM signals.

        Parameters:
            analysis_function: Function that performs the analysis
            analysis_params: Parameters for the analysis

        Returns:
            dict: Blind analysis results
        """
        if not self.protocol_registered:
            raise ValueError("Protocol must be registered before executing blind analysis")

        self.logger.info("Executing blind analysis...")

        try:
            # Execute analysis
            results = analysis_function(**analysis_params)

            # Save blind results (without interpretation)
            blind_results_file = self.results_dir / f"blind_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

            blind_results = {
                'protocol_hash': self.protocol_hash,
                'execution_timestamp': datetime.now().isoformat(),
                'analysis_results': results,
                'blind_status': 'RESULTS_GENERATED',
                'interpretation_status': 'BLINDED'
            }

            with open(blind_results_file, 'w') as f:
                json.dump(blind_results, f, indent=2, default=str)

            self.logger.info(f"Blind analysis results saved to {blind_results_file}")

            return {
                'execution_success': True,
                'results_file': str(blind_results_file),
                'blind_status': 'RESULTS_GENERATED'
            }

        except Exception as e:
            self.logger.error(f"Blind analysis execution failed: {e}")
            return {
                'execution_success': False,
                'error': str(e)
            }

    def generate_unblinding_report(self, h_lcdm_predictions: Dict[str, Any],
                                 interpretation_function: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Generate unblinding report comparing results to H-ΛCDM predictions.

        Parameters:
            h_lcdm_predictions: Theoretical H-ΛCDM predictions
            interpretation_function: Optional function to interpret results

        Returns:
            dict: Unblinding report
        """
        if not self.protocol_registered:
            raise ValueError("No registered protocol found")

        # Load protocol
        try:
            with open(self.protocol_file, 'r') as f:
                protocol = json.load(f)
        except FileNotFoundError:
            raise ValueError("Protocol file not found")

        # Load blind results
        blind_results_files = list(self.results_dir.glob("blind_results_*.json"))
        if not blind_results_files:
            raise ValueError("No blind results found")

        # Load most recent results
        latest_results_file = max(blind_results_files, key=lambda x: x.stat().st_mtime)
        with open(latest_results_file, 'r') as f:
            blind_results = json.load(f)

        # Verify protocol integrity
        if blind_results.get('protocol_hash') != self.protocol_hash:
            raise ValueError("Protocol hash mismatch - results may be compromised")

        # Apply interpretation if provided
        interpretation = {}
        if interpretation_function:
            try:
                interpretation = interpretation_function(
                    blind_results['analysis_results'],
                    h_lcdm_predictions
                )
            except Exception as e:
                self.logger.warning(f"Interpretation failed: {e}")

        # Generate unblinding report
        unblinding_report = {
            'unblinding_timestamp': datetime.now().isoformat(),
            'protocol_hash': self.protocol_hash,
            'protocol_summary': {
                'research_question': protocol.get('research_question'),
                'registration_timestamp': protocol.get('registration_timestamp'),
                'methodology_summary': self._summarize_methodology(protocol.get('methodology', {}))
            },
            'h_lcdm_predictions': h_lcdm_predictions,
            'blind_analysis_results': blind_results['analysis_results'],
            'interpretation': interpretation,
            'comparison_analysis': self._compare_to_predictions(
                blind_results['analysis_results'], h_lcdm_predictions
            ),
            'validation_assessment': self._assess_validation_compliance(
                blind_results['analysis_results'], protocol
            ),
            'unblinding_status': 'COMPLETE'
        }

        # Save unblinding report
        report_file = self.results_dir / f"unblinding_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(unblinding_report, f, indent=2, default=str)

        self.analysis_complete = True

        self.logger.info(f"Unblinding report generated: {report_file}")

        return {
            'unblinding_success': True,
            'report_file': str(report_file),
            'key_findings': self._extract_key_findings(unblinding_report)
        }

    def _summarize_methodology(self, methodology: Dict[str, Any]) -> Dict[str, Any]:
        """Create summary of registered methodology."""
        summary = {}

        # Extract key methodological elements
        if 'ml_pipeline' in methodology:
            pipeline = methodology['ml_pipeline']
            summary['pipeline_stages'] = list(pipeline.keys()) if isinstance(pipeline, dict) else []

        if 'validation_methods' in methodology:
            validation = methodology['validation_methods']
            summary['validation_techniques'] = list(validation.keys()) if isinstance(validation, dict) else []

        if 'data_sources' in methodology:
            data = methodology['data_sources']
            summary['n_data_sources'] = len(data) if isinstance(data, list) else 'unknown'

        return summary

    def _compare_to_predictions(self, analysis_results: Dict[str, Any],
                              h_lcdm_predictions: Dict[str, Any]) -> Dict[str, Any]:
        """
        Compare analysis results to H-ΛCDM predictions.

        Parameters:
            analysis_results: Blind analysis results
            h_lcdm_predictions: Theoretical predictions

        Returns:
            dict: Comparison analysis
        """
        comparison = {
            'comparison_timestamp': datetime.now().isoformat(),
            'predictions_tested': list(h_lcdm_predictions.keys())
        }

        # Extract key results from analysis
        detected_patterns = self._extract_detected_patterns(analysis_results)

        # Compare detections to predictions
        matches = []
        non_matches = []
        unexpected_findings = []

        for prediction_key, prediction in h_lcdm_predictions.items():
            prediction_found = self._check_prediction_match(prediction_key, prediction, detected_patterns)

            if prediction_found['match']:
                matches.append({
                    'prediction': prediction_key,
                    'confidence': prediction_found['confidence'],
                    'evidence': prediction_found['evidence']
                })
            else:
                non_matches.append({
                    'prediction': prediction_key,
                    'reason': prediction_found['reason']
                })

        comparison.update({
            'confirmed_predictions': matches,
            'unconfirmed_predictions': non_matches,
            'unexpected_findings': unexpected_findings,
            'overall_consistency': self._assess_overall_consistency(matches, non_matches)
        })

        return comparison

    def _extract_detected_patterns(self, analysis_results: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract detected patterns from analysis results."""
        patterns = []

        # Look for pattern detections in various result formats
        if 'robust_patterns' in analysis_results:
            robust = analysis_results['robust_patterns']
            if 'robust_anomaly_indices' in robust:
                patterns.append({
                    'type': 'robust_anomalies',
                    'indices': robust['robust_anomaly_indices'],
                    'count': robust['n_robust_anomalies'],
                    'confidence': 'high'
                })

        if 'ensemble_results' in analysis_results:
            ensemble = analysis_results['ensemble_results']
            if 'top_anomalies' in ensemble:
                patterns.extend([{
                    'type': 'top_anomaly',
                    'index': anomaly['sample_index'],
                    'score': anomaly['anomaly_score'],
                    'rank': anomaly['rank']
                } for anomaly in ensemble['top_anomalies']])

        return patterns

    def _check_prediction_match(self, prediction_key: str, prediction: Dict[str, Any],
                              detected_patterns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Check if a prediction matches detected patterns.

        Parameters:
            prediction_key: Key for the prediction
            prediction: Prediction details
            detected_patterns: Detected patterns

        Returns:
            dict: Match assessment
        """
        # This is a simplified matching - in practice would be more sophisticated
        # based on the specific H-ΛCDM predictions being tested

        match_criteria = prediction.get('detection_criteria', {})

        # Example matching logic (would be customized per prediction)
        if prediction_key == 'e8_geometry':
            # Look for geometric patterns in detected anomalies
            geometric_patterns = [p for p in detected_patterns if 'geometry' in str(p)]
            if geometric_patterns:
                return {
                    'match': True,
                    'confidence': 'medium',
                    'evidence': f'Found {len(geometric_patterns)} geometric patterns'
                }

        elif prediction_key == 'enhanced_sound_horizon':
            # Look for BAO-related anomalies
            bao_patterns = [p for p in detected_patterns if 'bao' in str(p).lower()]
            if bao_patterns:
                return {
                    'match': True,
                    'confidence': 'high',
                    'evidence': f'Found {len(bao_patterns)} BAO-related anomalies'
                }

        # Default: no match found
        return {
            'match': False,
            'confidence': 'unknown',
            'reason': 'No matching patterns detected'
        }

    def _assess_overall_consistency(self, matches: List[Dict], non_matches: List[Dict]) -> Dict[str, Any]:
        """Assess overall consistency between predictions and detections."""
        total_predictions = len(matches) + len(non_matches)

        if total_predictions == 0:
            return {'consistency_score': 0, 'assessment': 'no_predictions_tested'}

        consistency_score = len(matches) / total_predictions

        if consistency_score >= 0.8:
            assessment = 'high_consistency'
        elif consistency_score >= 0.5:
            assessment = 'moderate_consistency'
        else:
            assessment = 'low_consistency'

        return {
            'consistency_score': consistency_score,
            'assessment': assessment,
            'confirmed_predictions': len(matches),
            'total_predictions': total_predictions
        }

    def _assess_validation_compliance(self, analysis_results: Dict[str, Any],
                                    protocol: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assess compliance with pre-registered validation methods.

        Parameters:
            analysis_results: Analysis results
            protocol: Registered protocol

        Returns:
            dict: Validation compliance assessment
        """
        compliance = {
            'protocol_compliance': True,
            'validation_methods_used': [],
            'compliance_issues': []
        }

        # Check if required validation methods were used
        required_validations = protocol.get('methodology', {}).get('validation_methods', [])

        if 'bootstrap' in str(analysis_results).lower():
            compliance['validation_methods_used'].append('bootstrap')
        if 'null_hypothesis' in str(analysis_results).lower():
            compliance['validation_methods_used'].append('null_hypothesis')
        if 'cross_survey' in str(analysis_results).lower():
            compliance['validation_methods_used'].append('cross_survey')

        # Check for missing validations
        for required_val in required_validations:
            if required_val not in compliance['validation_methods_used']:
                compliance['compliance_issues'].append(f'Missing required validation: {required_val}')
                compliance['protocol_compliance'] = False

        return compliance

    def _extract_key_findings(self, unblinding_report: Dict[str, Any]) -> Dict[str, Any]:
        """Extract key findings from unblinding report."""
        findings = {}

        comparison = unblinding_report.get('comparison_analysis', {})

        findings['confirmed_predictions'] = len(comparison.get('confirmed_predictions', []))
        findings['unconfirmed_predictions'] = len(comparison.get('unconfirmed_predictions', []))
        findings['consistency_assessment'] = comparison.get('overall_consistency', {}).get('assessment')

        return findings

    def get_protocol_status(self) -> Dict[str, Any]:
        """Get current status of blind analysis protocol."""
        status = {
            'protocol_registered': self.protocol_registered,
            'analysis_complete': self.analysis_complete,
            'protocol_hash': self.protocol_hash
        }

        if self.protocol_registered:
            try:
                with open(self.protocol_file, 'r') as f:
                    protocol = json.load(f)
                status['registration_timestamp'] = protocol.get('registration_timestamp')
                status['research_question'] = protocol.get('research_question')
            except:
                status['protocol_error'] = 'Could not load protocol file'

        return status
