"""
Bootstrap Validation
===================

Statistical validation using bootstrap resampling (1000+ samples)
to assess stability and uncertainty of ML pattern detection.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import logging
import warnings


class BootstrapValidator:
    """
    Bootstrap validation for ML pattern detection stability.

    Performs extensive bootstrap resampling (1000+ samples) to:
    - Assess detection stability
    - Quantify uncertainty
    - Identify robust vs. spurious patterns
    - Only report patterns present in ≥95% of bootstraps
    """

    def __init__(self, n_bootstraps: int = 1000,
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        """
        Initialize bootstrap validator.

        Parameters:
            n_bootstraps: Number of bootstrap resamples (default 1000)
            confidence_level: Confidence level for intervals (default 95%)
            random_state: Random seed
        """
        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        # Set random seed
        np.random.seed(random_state)

    def validate_stability(self, model_factory: Callable,
                          full_dataset: Dict[str, Any],
                          detection_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Validate ML detection stability using bootstrap resampling.

        Parameters:
            model_factory: Function that creates fresh model instances
            full_dataset: Full dataset for bootstrapping
            detection_threshold: Threshold for anomaly detection

        Returns:
            dict: Bootstrap validation results
        """
        self.logger.info(f"Starting bootstrap validation with {self.n_bootstraps} resamples")

        bootstrap_results = []

        # Extract features from full dataset
        features = self._extract_features(full_dataset)

        if features is None or len(features) == 0:
            return {'error': 'No features available for bootstrap validation'}

        self.logger.info(f"Bootstrapping on {len(features)} samples with {features.shape[1]} features")

        for i in range(self.n_bootstraps):
            if (i + 1) % 100 == 0:
                self.logger.info(f"Bootstrap iteration {i + 1}/{self.n_bootstraps}")

            # Create bootstrap sample
            bootstrap_indices = self._generate_bootstrap_sample(len(features))
            bootstrap_features = features[bootstrap_indices]

            # Train model on bootstrap sample
            model = model_factory()
            try:
                bootstrap_result = self._run_single_bootstrap(
                    model, bootstrap_features, features, detection_threshold
                )
                bootstrap_result['bootstrap_index'] = i
                bootstrap_results.append(bootstrap_result)

            except Exception as e:
                self.logger.warning(f"Bootstrap {i} failed: {e}")
                bootstrap_results.append({
                    'bootstrap_index': i,
                    'error': str(e),
                    'success': False
                })

        # Analyze bootstrap results
        analysis_results = self._analyze_bootstrap_results(bootstrap_results)

        return {
            'stability_tests': bootstrap_results,
            'stability_analysis': analysis_results,
            'stability_summary': self._create_bootstrap_summary(analysis_results),
            'validation_metadata': {
                'n_bootstraps': self.n_bootstraps,
                'confidence_level': self.confidence_level,
                'original_dataset_size': len(features),
                'detection_threshold': detection_threshold
            }
        }

    def _extract_features(self, dataset: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract feature matrix from dataset.

        Parameters:
            dataset: Dataset dictionary

        Returns:
            np.ndarray: Feature matrix or None if unavailable
        """
        # Try different possible keys for features
        possible_keys = ['features', 'X', 'data', 'feature_matrix']

        for key in possible_keys:
            if key in dataset:
                features = dataset[key]
                if isinstance(features, np.ndarray):
                    return features
                elif hasattr(features, 'values'):  # pandas DataFrame
                    return features.values
                elif hasattr(features, '__array__'):
                    return np.array(features)

        # If no direct features, try to construct from modalities
        if 'modalities' in dataset:
            # This would require more complex feature extraction
            # For now, return None
            pass

        return None

    def _generate_bootstrap_sample(self, n_samples: int) -> np.ndarray:
        """
        Generate bootstrap sample indices.

        Parameters:
            n_samples: Size of original dataset

        Returns:
            np.ndarray: Bootstrap sample indices
        """
        return np.random.choice(n_samples, size=n_samples, replace=True)

    def _run_single_bootstrap(self, model, bootstrap_features: np.ndarray,
                            full_features: np.ndarray,
                            detection_threshold: float) -> Dict[str, Any]:
        """
        Run single bootstrap iteration.

        Parameters:
            model: Fresh model instance
            bootstrap_features: Bootstrap sample features
            full_features: Full dataset features for testing
            detection_threshold: Detection threshold

        Returns:
            dict: Bootstrap iteration results
        """
        try:
            # Train model on bootstrap sample
            if hasattr(model, 'fit'):
                model.fit(bootstrap_features)

            # Test on full dataset
            if hasattr(model, 'predict'):
                predictions = model.predict(full_features)

            # Handle different prediction formats
            if isinstance(predictions, dict):
                if 'ensemble_scores' in predictions:
                    anomaly_scores = predictions['ensemble_scores']
                elif 'individual_scores' in predictions:
                    # Use mean of individual scores
                    scores_list = list(predictions['individual_scores'].values())
                    anomaly_scores = np.mean(scores_list, axis=0)
                else:
                    raise ValueError("Unknown prediction format")
            else:
                anomaly_scores = np.array(predictions)

            # Compute detection statistics
            detections = anomaly_scores > detection_threshold

            bootstrap_result = {
                'success': True,
                'anomaly_scores': anomaly_scores.tolist(),
                'detections': detections.tolist(),
                'n_detected': int(np.sum(detections)),
                'detection_rate': float(np.mean(detections)),
                'mean_score': float(np.mean(anomaly_scores)),
                'std_score': float(np.std(anomaly_scores)),
                'max_score': float(np.max(anomaly_scores)),
                'score_percentiles': np.percentile(anomaly_scores, [25, 50, 75, 90, 95, 99]).tolist()
            }

            # Detect top anomalies (most anomalous samples)
            top_k = min(10, len(anomaly_scores))
            top_indices = np.argsort(-anomaly_scores)[:top_k]
            bootstrap_result['top_anomaly_indices'] = top_indices.tolist()
            bootstrap_result['top_anomaly_scores'] = anomaly_scores[top_indices].tolist()

            return bootstrap_result

        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

    def _analyze_bootstrap_results(self, bootstrap_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze results across all bootstrap iterations.

        Parameters:
            bootstrap_results: List of bootstrap iteration results

        Returns:
            dict: Analysis of bootstrap stability
        """
        successful_bootstraps = [r for r in bootstrap_results if r.get('success', False)]

        if not successful_bootstraps:
            return {'error': 'No successful bootstrap iterations'}

        analysis = {
            'n_successful_bootstraps': len(successful_bootstraps),
            'success_rate': len(successful_bootstraps) / len(bootstrap_results)
        }

        # Analyze detection stability
        detection_rates = [r['detection_rate'] for r in successful_bootstraps]
        analysis['detection_rate_stats'] = self._compute_bootstrap_statistics(detection_rates)

        # Analyze anomaly score stability
        mean_scores = [r['mean_score'] for r in successful_bootstraps]
        analysis['mean_score_stats'] = self._compute_bootstrap_statistics(mean_scores)

        # Analyze individual sample stability
        analysis['sample_stability'] = self._analyze_sample_stability(successful_bootstraps)

        # Identify robust patterns (present in ≥95% of bootstraps)
        analysis['robust_patterns'] = self._identify_robust_patterns(successful_bootstraps)

        # Compute confidence intervals
        analysis['confidence_intervals'] = self._compute_confidence_intervals(
            successful_bootstraps, self.confidence_level
        )

        return analysis

    def _compute_bootstrap_statistics(self, values: List[float]) -> Dict[str, float]:
        """
        Compute bootstrap statistics for a metric.

        Parameters:
            values: List of metric values across bootstraps

        Returns:
            dict: Statistics
        """
        values = np.array(values)

        # Basic statistics
        stats = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'percentile_5': float(np.percentile(values, 5)),
            'percentile_95': float(np.percentile(values, 95))
        }

        # Coefficient of variation
        if stats['mean'] != 0:
            stats['coefficient_of_variation'] = stats['std'] / abs(stats['mean'])
        else:
            stats['coefficient_of_variation'] = float('inf')

        return stats

    def _analyze_sample_stability(self, bootstrap_results: List[Dict]) -> Dict[str, Any]:
        """
        Analyze stability of individual sample classifications.

        Parameters:
            bootstrap_results: Successful bootstrap results

        Returns:
            dict: Sample stability analysis
        """
        if not bootstrap_results:
            return {}

        n_samples = len(bootstrap_results[0]['detections'])

        # Collect detection frequencies for each sample
        detection_frequencies = np.zeros(n_samples)

        for result in bootstrap_results:
            detections = np.array(result['detections'])
            detection_frequencies += detections.astype(int)

        # Normalize to frequency
        detection_frequencies = detection_frequencies / len(bootstrap_results)

        # Analyze stability distribution
        stability_stats = {
            'mean_detection_frequency': float(np.mean(detection_frequencies)),
            'std_detection_frequency': float(np.std(detection_frequencies)),
            'highly_stable_samples': int(np.sum(detection_frequencies >= 0.95)),  # Detected in ≥95% bootstraps
            'unstable_samples': int(np.sum(detection_frequencies <= 0.05)),  # Rarely detected
            'detection_frequency_percentiles': np.percentile(detection_frequencies, [25, 50, 75, 90, 95]).tolist()
        }

        # Identify most stable anomalies
        highly_stable_mask = detection_frequencies >= 0.95
        stable_indices = np.where(highly_stable_mask)[0]

        if len(stable_indices) > 0:
            # Get typical scores for stable samples
            stable_scores = []
            for result in bootstrap_results:
                scores = np.array(result['anomaly_scores'])
                stable_scores.extend(scores[stable_indices])

            stability_stats['stable_sample_mean_score'] = float(np.mean(stable_scores))
            stability_stats['stable_sample_std_score'] = float(np.std(stable_scores))

        return stability_stats

    def _identify_robust_patterns(self, bootstrap_results: List[Dict]) -> Dict[str, Any]:
        """
        Identify patterns that are robust across bootstraps.

        Parameters:
            bootstrap_results: Successful bootstrap results

        Returns:
            dict: Robust pattern analysis
        """
        if not bootstrap_results:
            return {}

        n_samples = len(bootstrap_results[0]['detections'])
        robust_threshold = 0.95  # Present in ≥95% of bootstraps

        # Count how often each sample is detected as anomalous
        detection_counts = np.zeros(n_samples)

        for result in bootstrap_results:
            detections = np.array(result['detections'])
            detection_counts += detections.astype(int)

        detection_frequencies = detection_counts / len(bootstrap_results)

        # Samples that are consistently detected
        robust_anomalies = np.where(detection_frequencies >= robust_threshold)[0]

        robust_patterns = {
            'n_robust_anomalies': len(robust_anomalies),
            'robust_anomaly_indices': robust_anomalies.tolist(),
            'robust_threshold': robust_threshold,
            'robust_anomaly_rates': detection_frequencies[robust_anomalies].tolist()
        }

        # Analyze if robust anomalies have distinctive score distributions
        if len(robust_anomalies) > 0:
            robust_scores = []
            for result in bootstrap_results:
                scores = np.array(result['anomaly_scores'])
                robust_scores.extend(scores[robust_anomalies])

            robust_patterns['robust_scores_stats'] = {
                'mean': float(np.mean(robust_scores)),
                'std': float(np.std(robust_scores)),
                'min': float(np.min(robust_scores)),
                'max': float(np.max(robust_scores))
            }

        return robust_patterns

    def _compute_confidence_intervals(self, bootstrap_results: List[Dict],
                                    confidence_level: float) -> Dict[str, Any]:
        """
        Compute confidence intervals from bootstrap distribution.

        Parameters:
            bootstrap_results: Bootstrap results
            confidence_level: Confidence level (e.g., 0.95)

        Returns:
            dict: Confidence intervals
        """
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)

        ci_results = {}

        # Detection rate confidence interval
        detection_rates = [r['detection_rate'] for r in bootstrap_results if 'detection_rate' in r]
        if detection_rates:
            ci_results['detection_rate'] = {
                'mean': float(np.mean(detection_rates)),
                'ci_lower': float(np.percentile(detection_rates, lower_percentile)),
                'ci_upper': float(np.percentile(detection_rates, upper_percentile)),
                'ci_width': float(np.percentile(detection_rates, upper_percentile) - np.percentile(detection_rates, lower_percentile))
            }

        # Mean score confidence interval
        mean_scores = [r['mean_score'] for r in bootstrap_results if 'mean_score' in r]
        if mean_scores:
            ci_results['mean_score'] = {
                'mean': float(np.mean(mean_scores)),
                'ci_lower': float(np.percentile(mean_scores, lower_percentile)),
                'ci_upper': float(np.percentile(mean_scores, upper_percentile)),
                'ci_width': float(np.percentile(mean_scores, upper_percentile) - np.percentile(mean_scores, lower_percentile))
            }

        return ci_results

    def _create_bootstrap_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create human-readable bootstrap validation summary.

        Parameters:
            analysis: Bootstrap analysis results

        Returns:
            dict: Summary
        """
        summary = {
            'validation_type': 'Bootstrap Stability Analysis',
            'n_bootstraps': self.n_bootstraps,
            'confidence_level': self.confidence_level
        }

        # Overall stability assessment
        if analysis.get('success_rate', 0) < 0.8:
            summary['stability_status'] = 'UNSTABLE'
            summary['issues'] = ['Low bootstrap success rate']
        elif analysis.get('sample_stability', {}).get('highly_stable_samples', 0) == 0:
            summary['stability_status'] = 'NO_ROBUST_PATTERNS'
            summary['issues'] = ['No patterns stable across ≥95% of bootstraps']
        else:
            summary['stability_status'] = 'STABLE'

        # Detection consistency
        detection_stats = analysis.get('detection_rate_stats', {})
        if detection_stats:
            cv = detection_stats.get('coefficient_of_variation', float('inf'))
            if cv < 0.2:
                summary['detection_consistency'] = 'HIGH'
            elif cv < 0.5:
                summary['detection_consistency'] = 'MODERATE'
            else:
                summary['detection_consistency'] = 'LOW'

        # Robust patterns
        robust_patterns = analysis.get('robust_patterns', {})
        summary['n_robust_anomalies'] = robust_patterns.get('n_robust_anomalies', 0)

        # Scientific implications
        implications = []

        if summary['stability_status'] == 'STABLE':
            implications.append("Detection results are stable across bootstrap resampling")

        if summary.get('n_robust_anomalies', 0) > 0:
            implications.append(f"Found {summary['n_robust_anomalies']} robust anomalous patterns present in ≥95% of bootstraps")

        if summary['stability_status'] == 'NO_ROBUST_PATTERNS':
            implications.append("No patterns are sufficiently stable - may indicate noise or methodological issues")

        summary['scientific_implications'] = implications

        return summary
