"""
Bootstrap Validation
===================

Statistical validation using bootstrap resampling (1000+ samples)
to assess stability and uncertainty of ML pattern detection.
Supports multi-modal datasets.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple, Union
from sklearn.utils import resample
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import logging
import warnings

# Suppress sklearn deprecation warnings for internal API changes
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.utils.deprecation')


class BootstrapValidator:
    """
    Bootstrap validation for ML pattern detection stability.
    """

    def __init__(self, n_bootstraps: int = 1000,
                 confidence_level: float = 0.95,
                 random_state: int = 42):
        self.n_bootstraps = n_bootstraps
        self.confidence_level = confidence_level
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)
        np.random.seed(random_state)

    def validate_stability(self, model_factory: Callable,
                          full_dataset: Dict[str, Any],
                          detection_threshold: float = 0.5) -> Dict[str, Any]:
        """
        Validate ML detection stability using bootstrap resampling.
        Handles both numpy arrays and multi-modal dictionaries.
        """
        self.logger.info(f"Starting bootstrap validation with {self.n_bootstraps} resamples")
        bootstrap_results = []

        # Determine dataset size
        n_samples = self._get_dataset_size(full_dataset)
        if n_samples == 0:
            return {'error': 'Empty dataset'}

        self.logger.info(f"Bootstrapping on {n_samples} samples")

        n_failures = 0
        for i in range(self.n_bootstraps):
            if (i + 1) % 100 == 0:
                self.logger.info(f"Bootstrap iteration {i + 1}/{self.n_bootstraps}")

            bootstrap_indices = self._generate_bootstrap_sample(n_samples)
            bootstrap_data = self._subset_dataset(full_dataset, bootstrap_indices)

            model = model_factory()
            try:
                bootstrap_result = self._run_single_bootstrap(
                    model, bootstrap_data, full_dataset, detection_threshold
                )
                bootstrap_result['bootstrap_index'] = i
                bootstrap_results.append(bootstrap_result)
                
                if not bootstrap_result.get('success', False):
                    n_failures += 1

            except (TypeError, ValueError) as e:
                self.logger.error(f"Critical error in bootstrap {i}: {e}")
                raise e
            except Exception as e:
                self.logger.warning(f"Bootstrap {i} failed: {e}")
                n_failures += 1
                bootstrap_results.append({'bootstrap_index': i, 'error': str(e), 'success': False})

        if n_failures > self.n_bootstraps * 0.1:
            raise RuntimeError(f"Bootstrap validation failed: {n_failures}/{self.n_bootstraps} iterations failed. This exceeds the 10% tolerance threshold.")

        analysis_results = self._analyze_bootstrap_results(bootstrap_results)

        return {
            'stability_tests': bootstrap_results,
            'stability_analysis': analysis_results,
            'stability_summary': self._create_bootstrap_summary(analysis_results),
            'validation_metadata': {
                'n_bootstraps': self.n_bootstraps,
                'confidence_level': self.confidence_level,
                'original_dataset_size': n_samples,
                'detection_threshold': detection_threshold
            }
        }

    def _get_dataset_size(self, dataset: Union[Dict[str, Any], np.ndarray]) -> int:
        """Get number of samples in dataset."""
        if isinstance(dataset, np.ndarray):
            return len(dataset)
        elif isinstance(dataset, dict):
            # Assume all modalities have same length, check first non-empty
            for key, val in dataset.items():
                if isinstance(val, (np.ndarray, list)):
                    return len(val)
                elif hasattr(val, 'shape'):
                    return val.shape[0]
        return 0

    def _subset_dataset(self, dataset: Union[Dict[str, Any], np.ndarray], indices: np.ndarray) -> Any:
        """Subset dataset by indices."""
        if isinstance(dataset, np.ndarray):
            return dataset[indices]
        elif isinstance(dataset, dict):
            subset = {}
            for key, val in dataset.items():
                if isinstance(val, np.ndarray):
                    subset[key] = val[indices]
                elif hasattr(val, 'iloc'): # DataFrame
                    subset[key] = val.iloc[indices]
                else:
                    subset[key] = val # Pass through metadata
            return subset
        return dataset

    def _generate_bootstrap_sample(self, n_samples: int) -> np.ndarray:
        return np.random.choice(n_samples, size=n_samples, replace=True)

    def _run_single_bootstrap(self, model, bootstrap_data: Any,
                            full_data: Any,
                            detection_threshold: float) -> Dict[str, Any]:
        try:
            # Train model
            if hasattr(model, 'fit'):
                model.fit(bootstrap_data)

            # Test on full dataset
            if hasattr(model, 'predict'):
                predictions = model.predict(full_data)

            # Handle different prediction formats
            if isinstance(predictions, dict):
                if 'ensemble_scores' in predictions:
                    anomaly_scores = predictions['ensemble_scores']
                elif 'individual_scores' in predictions:
                    scores_list = list(predictions['individual_scores'].values())
                    anomaly_scores = np.mean(scores_list, axis=0)
                else:
                    raise ValueError("Unknown prediction format")
            else:
                anomaly_scores = np.array(predictions)

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

            top_k = min(10, len(anomaly_scores))
            top_indices = np.argsort(-anomaly_scores)[:top_k]
            bootstrap_result['top_anomaly_indices'] = top_indices.tolist()
            bootstrap_result['top_anomaly_scores'] = anomaly_scores[top_indices].tolist()

            return bootstrap_result

        except (TypeError, ValueError) as e:
            # Critical errors indicating code bugs should propagate
            raise e
        except Exception as e:
            return {'success': False, 'error': str(e)}

    def _analyze_bootstrap_results(self, bootstrap_results: List[Dict]) -> Dict[str, Any]:
        successful_bootstraps = [r for r in bootstrap_results if r.get('success', False)]

        if not successful_bootstraps:
            return {'error': 'No successful bootstrap iterations'}

        analysis = {
            'n_successful_bootstraps': len(successful_bootstraps),
            'success_rate': len(successful_bootstraps) / len(bootstrap_results)
        }

        detection_rates = [r['detection_rate'] for r in successful_bootstraps]
        analysis['detection_rate_stats'] = self._compute_bootstrap_statistics(detection_rates)

        mean_scores = [r['mean_score'] for r in successful_bootstraps]
        analysis['mean_score_stats'] = self._compute_bootstrap_statistics(mean_scores)

        analysis['sample_stability'] = self._analyze_sample_stability(successful_bootstraps)
        analysis['robust_patterns'] = self._identify_robust_patterns(successful_bootstraps)
        analysis['confidence_intervals'] = self._compute_confidence_intervals(
            successful_bootstraps, self.confidence_level
        )

        return analysis

    def _compute_bootstrap_statistics(self, values: List[float]) -> Dict[str, float]:
        values = np.array(values)
        stats = {
            'mean': float(np.mean(values)),
            'std': float(np.std(values)),
            'median': float(np.median(values)),
            'min': float(np.min(values)),
            'max': float(np.max(values)),
            'percentile_5': float(np.percentile(values, 5)),
            'percentile_95': float(np.percentile(values, 95))
        }
        if stats['mean'] != 0:
            stats['coefficient_of_variation'] = stats['std'] / abs(stats['mean'])
        else:
            stats['coefficient_of_variation'] = float('inf')
        return stats

    def _analyze_sample_stability(self, bootstrap_results: List[Dict]) -> Dict[str, Any]:
        if not bootstrap_results:
            return {}
        n_samples = len(bootstrap_results[0]['detections'])
        detection_frequencies = np.zeros(n_samples)
        for result in bootstrap_results:
            detections = np.array(result['detections'])
            detection_frequencies += detections.astype(int)
        detection_frequencies = detection_frequencies / len(bootstrap_results)

        stability_stats = {
            'mean_detection_frequency': float(np.mean(detection_frequencies)),
            'std_detection_frequency': float(np.std(detection_frequencies)),
            'highly_stable_samples': int(np.sum(detection_frequencies >= 0.95)),
            'unstable_samples': int(np.sum(detection_frequencies <= 0.05)),
            'detection_frequency_percentiles': np.percentile(detection_frequencies, [25, 50, 75, 90, 95]).tolist()
        }

        highly_stable_mask = detection_frequencies >= 0.95
        stable_indices = np.where(highly_stable_mask)[0]
        if len(stable_indices) > 0:
            stable_scores = []
            for result in bootstrap_results:
                scores = np.array(result['anomaly_scores'])
                stable_scores.extend(scores[stable_indices])
            stability_stats['stable_sample_mean_score'] = float(np.mean(stable_scores))
            stability_stats['stable_sample_std_score'] = float(np.std(stable_scores))
        return stability_stats

    def _identify_robust_patterns(self, bootstrap_results: List[Dict]) -> Dict[str, Any]:
        if not bootstrap_results:
            return {}
        n_samples = len(bootstrap_results[0]['detections'])
        robust_threshold = 0.95
        detection_counts = np.zeros(n_samples)
        for result in bootstrap_results:
            detections = np.array(result['detections'])
            detection_counts += detections.astype(int)
        detection_frequencies = detection_counts / len(bootstrap_results)
        robust_anomalies = np.where(detection_frequencies >= robust_threshold)[0]

        robust_patterns = {
            'n_robust_anomalies': len(robust_anomalies),
            'robust_anomaly_indices': robust_anomalies.tolist(),
            'robust_threshold': robust_threshold,
            'robust_anomaly_rates': detection_frequencies[robust_anomalies].tolist()
        }

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
        alpha = 1 - confidence_level
        lower_percentile = 100 * (alpha / 2)
        upper_percentile = 100 * (1 - alpha / 2)
        ci_results = {}

        detection_rates = [r['detection_rate'] for r in bootstrap_results if 'detection_rate' in r]
        if detection_rates:
            ci_results['detection_rate'] = {
                'mean': float(np.mean(detection_rates)),
                'ci_lower': float(np.percentile(detection_rates, lower_percentile)),
                'ci_upper': float(np.percentile(detection_rates, upper_percentile)),
                'ci_width': float(np.percentile(detection_rates, upper_percentile) - np.percentile(detection_rates, lower_percentile))
            }

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
        summary = {
            'validation_type': 'Bootstrap Stability Analysis',
            'n_bootstraps': self.n_bootstraps,
            'confidence_level': self.confidence_level
        }
        if analysis.get('success_rate', 0) < 0.8:
            summary['stability_status'] = 'UNSTABLE'
            summary['issues'] = ['Low bootstrap success rate']
        elif analysis.get('sample_stability', {}).get('highly_stable_samples', 0) == 0:
            summary['stability_status'] = 'NO_ROBUST_PATTERNS'
            summary['issues'] = ['No patterns stable across ≥95% of bootstraps']
        else:
            summary['stability_status'] = 'STABLE'

        detection_stats = analysis.get('detection_rate_stats', {})
        if detection_stats:
            cv = detection_stats.get('coefficient_of_variation', float('inf'))
            if cv < 0.2:
                summary['detection_consistency'] = 'HIGH'
            elif cv < 0.5:
                summary['detection_consistency'] = 'MODERATE'
            else:
                summary['detection_consistency'] = 'LOW'

        robust_patterns = analysis.get('robust_patterns', {})
        summary['n_robust_anomalies'] = robust_patterns.get('n_robust_anomalies', 0)

        implications = []
        if summary['stability_status'] == 'STABLE':
            implications.append("Detection results are stable across bootstrap resampling")
        if summary.get('n_robust_anomalies', 0) > 0:
            implications.append(f"Found {summary['n_robust_anomalies']} robust anomalous patterns present in ≥95% of bootstraps")
        if summary['stability_status'] == 'NO_ROBUST_PATTERNS':
            implications.append("No patterns are sufficiently stable - may indicate noise or methodological issues")
        summary['scientific_implications'] = implications
        return summary
