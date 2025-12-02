"""
Stability Validation
===================

Validates ML model stability across different random seeds,
data splits, and hyperparameter choices.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import logging
import warnings

# Suppress sklearn deprecation warnings for internal API changes
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.utils.deprecation')


class StabilityValidator:
    """
    Comprehensive stability validation for ML models.

    Tests model stability across:
    - Multiple random seeds
    - Different data splits
    - Hyperparameter variations
    - Uncertainty quantification via ensemble variance
    """

    def __init__(self, n_stability_tests: int = 10,
                 random_seeds: Optional[List[int]] = None,
                 k_folds: int = 5):
        """
        Initialize stability validator.

        Parameters:
            n_stability_tests: Number of stability tests to run
            random_seeds: List of random seeds to test
            k_folds: Number of K-fold splits for cross-validation
        """
        self.n_stability_tests = n_stability_tests
        self.random_seeds = random_seeds or list(range(42, 42 + n_stability_tests))
        self.k_folds = k_folds
        self.logger = logging.getLogger(__name__)

    def validate_model_stability(self, model_factory: Callable,
                               dataset: Dict[str, Any],
                               hyperparameter_ranges: Optional[Dict[str, List]] = None) -> Dict[str, Any]:
        """
        Validate model stability across multiple conditions.

        Parameters:
            model_factory: Function that creates model instances
            dataset: Dataset for validation
            hyperparameter_ranges: Ranges of hyperparameters to test

        Returns:
            dict: Stability validation results
        """
        self.logger.info(f"Starting stability validation with {self.n_stability_tests} tests")

        stability_results = []

        for i, seed in enumerate(self.random_seeds):
            self.logger.info(f"Stability test {i + 1}/{self.n_stability_tests} (seed: {seed})")

            # Test with different random seeds
            seed_result = self._test_random_seed_stability(
                model_factory, dataset, seed
            )
            seed_result['test_type'] = 'random_seed'
            seed_result['seed'] = seed
            stability_results.append(seed_result)

        # Test with different data splits
        split_results = self._test_data_split_stability(model_factory, dataset)
        stability_results.extend(split_results)

        # Test hyperparameter stability if ranges provided
        if hyperparameter_ranges:
            hyperparam_results = self._test_hyperparameter_stability(
                model_factory, dataset, hyperparameter_ranges
            )
            stability_results.extend(hyperparam_results)

        # Analyze overall stability
        stability_analysis = self._analyze_stability_results(stability_results)

        return {
            'stability_tests': stability_results,
            'stability_analysis': stability_analysis,
            'stability_summary': self._create_stability_summary(stability_analysis),
            'validation_metadata': {
                'n_stability_tests': self.n_stability_tests,
                'k_folds': self.k_folds,
                'random_seeds_tested': self.random_seeds,
                'hyperparameters_tested': list(hyperparameter_ranges.keys()) if hyperparameter_ranges else []
            }
        }

    def _test_random_seed_stability(self, model_factory: Callable,
                                  dataset: Dict[str, Any],
                                  seed: int) -> Dict[str, Any]:
        """
        Test model stability with a specific random seed.

        Parameters:
            model_factory: Model factory function
            dataset: Dataset to test on
            seed: Random seed

        Returns:
            dict: Seed-specific test results
        """
        # Set random seed
        np.random.seed(seed)

        try:
            # Create and train model
            model = model_factory()

            # Extract features
            features = self._extract_features(dataset)
            if features is None:
                return {'success': False, 'error': 'No features available'}

            # Train model
            if hasattr(model, 'fit'):
                model.fit(features)

            # Get predictions
            predictions = model.predict(features)

            # Analyze prediction stability
            if isinstance(predictions, dict) and 'ensemble_scores' in predictions:
                scores = predictions['ensemble_scores']
            else:
                scores = np.array(predictions)

            # Compute stability metrics
            result = {
                'success': True,
                'random_seed': seed,
                'n_samples': len(features),
                'mean_prediction': float(np.mean(scores)),
                'std_prediction': float(np.std(scores)),
                'prediction_percentiles': np.percentile(scores, [25, 50, 75, 90, 95]).tolist(),
                'n_anomalies_detected': int(np.sum(scores > 0.5)),
                'detection_rate': float(np.mean(scores > 0.5))
            }

            # Top anomaly stability
            if len(scores) > 10:
                top_indices = np.argsort(-scores)[:10]
                result['top_anomaly_indices'] = top_indices.tolist()
                result['top_anomaly_scores'] = scores[top_indices].tolist()

            return result

        except Exception as e:
            return {
                'success': False,
                'random_seed': seed,
                'error': str(e)
            }

    def _test_data_split_stability(self, model_factory: Callable,
                                 dataset: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Test model stability across different data splits.

        Parameters:
            model_factory: Model factory function
            dataset: Dataset to split

        Returns:
            list: Results for each data split
        """
        features = self._extract_features(dataset)
        if features is None:
            return [{'success': False, 'error': 'No features available'}]

        kf = KFold(n_splits=self.k_folds, shuffle=True, random_state=42)
        split_results = []

        for fold_idx, (train_idx, test_idx) in enumerate(kf.split(features)):
            try:
                # Split data
                train_features = features[train_idx]
                test_features = features[test_idx]

                # Train on train split
                model = model_factory()
                if hasattr(model, 'fit'):
                    model.fit(train_features)

                # Test on test split
                test_predictions = model.predict(test_features)

                if isinstance(test_predictions, dict) and 'ensemble_scores' in test_predictions:
                    scores = test_predictions['ensemble_scores']
                else:
                    scores = np.array(test_predictions)

                split_result = {
                    'success': True,
                    'test_type': 'data_split',
                    'fold_index': fold_idx,
                    'train_size': len(train_features),
                    'test_size': len(test_features),
                    'mean_prediction': float(np.mean(scores)),
                    'std_prediction': float(np.std(scores)),
                    'n_anomalies_detected': int(np.sum(scores > 0.5)),
                    'detection_rate': float(np.mean(scores > 0.5))
                }

                split_results.append(split_result)

            except Exception as e:
                split_results.append({
                    'success': False,
                    'test_type': 'data_split',
                    'fold_index': fold_idx,
                    'error': str(e)
                })

        return split_results

    def _test_hyperparameter_stability(self, model_factory: Callable,
                                     dataset: Dict[str, Any],
                                     hyperparameter_ranges: Dict[str, List]) -> List[Dict[str, Any]]:
        """
        Test model stability across hyperparameter choices.

        Parameters:
            model_factory: Model factory function
            dataset: Dataset to test on
            hyperparameter_ranges: Ranges of hyperparameters to test

        Returns:
            list: Results for different hyperparameter combinations
        """
        features = self._extract_features(dataset)
        if features is None:
            return [{'success': False, 'error': 'No features available'}]

        hyperparam_results = []

        # Generate hyperparameter combinations (simplified - test extremes)
        if 'contamination' in hyperparameter_ranges:
            for contamination in hyperparameter_ranges['contamination'][:3]:  # Test 3 values
                try:
                    # Create model with specific hyperparameters
                    model = model_factory(contamination=contamination)

                    if hasattr(model, 'fit'):
                        model.fit(features)

                    predictions = model.predict(features)

                    if isinstance(predictions, dict) and 'ensemble_scores' in predictions:
                        scores = predictions['ensemble_scores']
                    else:
                        scores = np.array(predictions)

                    result = {
                        'success': True,
                        'test_type': 'hyperparameter',
                        'hyperparameters': {'contamination': contamination},
                        'mean_prediction': float(np.mean(scores)),
                        'std_prediction': float(np.std(scores)),
                        'n_anomalies_detected': int(np.sum(scores > 0.5)),
                        'detection_rate': float(np.mean(scores > 0.5))
                    }

                    hyperparam_results.append(result)

                except Exception as e:
                    hyperparam_results.append({
                        'success': False,
                        'test_type': 'hyperparameter',
                        'hyperparameters': {'contamination': contamination},
                        'error': str(e)
                    })

        return hyperparam_results

    def _extract_features(self, dataset: Dict[str, Any]) -> Optional[np.ndarray]:
        """Extract features from dataset."""
        possible_keys = ['features', 'X', 'data', 'feature_matrix']

        for key in possible_keys:
            if key in dataset:
                features = dataset[key]
                if isinstance(features, np.ndarray):
                    return features
                elif hasattr(features, 'values'):
                    return features.values

        return None

    def _analyze_stability_results(self, stability_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze stability across all tests.

        Parameters:
            stability_results: Results from all stability tests

        Returns:
            dict: Stability analysis
        """
        successful_tests = [r for r in stability_results if r.get('success', False)]

        analysis = {
            'total_tests': len(stability_results),
            'successful_tests': len(successful_tests),
            'success_rate': len(successful_tests) / len(stability_results) if stability_results else 0
        }

        if not successful_tests:
            return analysis

        # Analyze prediction stability
        detection_rates = [r.get('detection_rate', 0) for r in successful_tests]
        mean_predictions = [r.get('mean_prediction', 0) for r in successful_tests]

        analysis['prediction_stability'] = {
            'detection_rate_mean': float(np.mean(detection_rates)),
            'detection_rate_std': float(np.std(detection_rates)),
            'detection_rate_cv': float(np.std(detection_rates) / (np.mean(detection_rates) + 1e-10)),
            'mean_prediction_std': float(np.std(mean_predictions)),
            'prediction_range': float(np.max(mean_predictions) - np.min(mean_predictions))
        }

        # Analyze by test type
        test_types = {}
        for result in successful_tests:
            test_type = result.get('test_type', 'unknown')
            if test_type not in test_types:
                test_types[test_type] = []
            test_types[test_type].append(result)

        analysis['stability_by_type'] = {}
        for test_type, type_results in test_types.items():
            type_detection_rates = [r.get('detection_rate', 0) for r in type_results]
            analysis['stability_by_type'][test_type] = {
                'n_tests': len(type_results),
                'detection_rate_std': float(np.std(type_detection_rates)),
                'detection_rate_range': float(np.max(type_detection_rates) - np.min(type_detection_rates))
            }

        # Overall stability assessment
        detection_rate_cv = analysis['prediction_stability']['detection_rate_cv']

        if detection_rate_cv < 0.1:
            stability_assessment = 'highly_stable'
        elif detection_rate_cv < 0.25:
            stability_assessment = 'stable'
        elif detection_rate_cv < 0.5:
            stability_assessment = 'moderately_stable'
        else:
            stability_assessment = 'unstable'

        analysis['overall_stability_assessment'] = stability_assessment

        return analysis

    def _create_stability_summary(self, analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create human-readable stability summary.

        Parameters:
            analysis: Stability analysis results

        Returns:
            dict: Stability summary
        """
        summary = {
            'validation_type': 'Model Stability Analysis',
            'total_tests': analysis['total_tests'],
            'success_rate': analysis['success_rate']
        }

        stability_assessment = analysis.get('overall_stability_assessment', 'unknown')

        if stability_assessment == 'highly_stable':
            summary['stability_status'] = 'EXCELLENT'
            summary['conclusion'] = 'Model predictions are highly stable across all tested conditions'
        elif stability_assessment == 'stable':
            summary['stability_status'] = 'GOOD'
            summary['conclusion'] = 'Model predictions are stable across tested conditions'
        elif stability_assessment == 'moderately_stable':
            summary['stability_status'] = 'FAIR'
            summary['conclusion'] = 'Model predictions show moderate stability, some variability observed'
        else:
            summary['stability_status'] = 'CONCERNING'
            summary['conclusion'] = 'Model predictions are unstable, significant variability detected'

        # Key metrics
        stability_metrics = analysis.get('prediction_stability', {})
        summary['key_metrics'] = {
            'detection_rate_coefficient_of_variation': stability_metrics.get('detection_rate_cv', 0),
            'prediction_standard_deviation': stability_metrics.get('mean_prediction_std', 0)
        }

        # Scientific implications
        implications = []

        if summary['stability_status'] in ['EXCELLENT', 'GOOD']:
            implications.append("Stable model predictions increase confidence in detected patterns")
        else:
            implications.append("Unstable predictions suggest model may be sensitive to random initialization or data variations")

        if analysis.get('success_rate', 0) < 0.8:
            implications.append("Low test success rate may indicate methodological issues")

        summary['scientific_implications'] = implications

        return summary
