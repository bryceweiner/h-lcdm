"""
Cross-Survey Validation
=======================

Validates ML pattern detection across different cosmological surveys
to ensure detected patterns are cosmological rather than survey-specific.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Callable, Tuple
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
import logging
import warnings

# Suppress sklearn deprecation warnings for internal API changes
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.utils.deprecation')


class CrossSurveyValidator:
    """
    Cross-survey validation for cosmological pattern detection.

    Implements multiple-source cross-validation (Geras & Sutton 2013)
    where models are trained on subsets of surveys and tested on held-out surveys.
    """

    def __init__(self, model_factory: Callable,
                 n_splits: int = 5,
                 random_state: int = 42):
        """
        Initialize cross-survey validator.

        Parameters:
            model_factory: Function that creates fresh model instances
            n_splits: Number of cross-validation splits
            random_state: Random seed for reproducibility
        """
        self.model_factory = model_factory
        self.n_splits = n_splits
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

    def validate_across_surveys(self, survey_datasets: Dict[str, Dict[str, Any]],
                               target_patterns: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Perform cross-survey validation.

        Parameters:
            survey_datasets: Dict of survey_name -> dataset_dict
            target_patterns: Known patterns to look for (blinded in real analysis)

        Returns:
            dict: Cross-survey validation results
        """
        survey_names = list(survey_datasets.keys())
        n_surveys = len(survey_names)

        if n_surveys < 2:
            return {'error': 'Need at least 2 surveys for cross-validation'}

        # Generate train/test splits
        cv_splits = self._generate_cv_splits(survey_names)

        validation_results = []

        for split_idx, (train_surveys, test_surveys) in enumerate(cv_splits):
            self.logger.info(f"Cross-validation split {split_idx + 1}/{len(cv_splits)}")
            self.logger.info(f"Train surveys: {train_surveys}")
            self.logger.info(f"Test surveys: {test_surveys}")

            # Train model on training surveys
            model = self.model_factory()
            train_result = self._train_on_surveys(model, train_surveys, survey_datasets)

            # Test on held-out surveys
            test_result = self._test_on_surveys(model, test_surveys, survey_datasets)

            # Analyze consistency across surveys
            consistency_result = self._analyze_survey_consistency(
                train_result, test_result, train_surveys, test_surveys
            )

            split_result = {
                'split_index': split_idx,
                'train_surveys': train_surveys,
                'test_surveys': test_surveys,
                'train_result': train_result,
                'test_result': test_result,
                'consistency_analysis': consistency_result
            }

            validation_results.append(split_result)

        # Aggregate results across all splits
        aggregated_results = self._aggregate_cv_results(validation_results)

        return {
            'individual_splits': validation_results,
            'aggregated_results': aggregated_results,
            'validation_summary': self._create_validation_summary(aggregated_results)
        }

    def _generate_cv_splits(self, survey_names: List[str]) -> List[Tuple[List[str], List[str]]]:
        """
        Generate cross-validation splits for surveys.

        Uses leave-one-out when possible, otherwise stratified splits.
        """
        splits = []

        if len(survey_names) <= 5:
            # Leave-one-out cross-validation
            for i in range(len(survey_names)):
                test_surveys = [survey_names[i]]
                train_surveys = [s for s in survey_names if s not in test_surveys]
                splits.append((train_surveys, test_surveys))
        else:
            # K-fold cross-validation
            kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
            survey_indices = np.arange(len(survey_names))

            for train_idx, test_idx in kf.split(survey_indices):
                train_surveys = [survey_names[i] for i in train_idx]
                test_surveys = [survey_names[i] for i in test_idx]
                splits.append((train_surveys, test_surveys))

        return splits

    def _train_on_surveys(self, model, train_surveys: List[str],
                         survey_datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Train model on specified surveys.

        Parameters:
            model: ML model to train
            train_surveys: Surveys to train on
            survey_datasets: All survey datasets

        Returns:
            dict: Training results
        """
        # Combine training data from multiple surveys
        train_features = []
        train_labels = []  # For supervised validation (if available)

        for survey in train_surveys:
            if survey in survey_datasets:
                survey_data = survey_datasets[survey]

                # Extract features (this would be handled by feature extractors in practice)
                if 'features' in survey_data:
                    train_features.append(survey_data['features'])

                # Labels for validation (normally not available in blind analysis)
                if 'labels' in survey_data:
                    train_labels.append(survey_data['labels'])

        # Combine features
        if train_features:
            combined_features = np.vstack(train_features)
        else:
            return {'error': 'No training features available'}

        # Train model
        try:
            if hasattr(model, 'fit'):
                model.fit(combined_features)

            training_result = {
                'n_train_samples': len(combined_features),
                'n_train_surveys': len(train_surveys),
                'feature_dim': combined_features.shape[1],
                'training_success': True
            }

        except Exception as e:
            training_result = {
                'error': str(e),
                'training_success': False
            }

        return training_result

    def _test_on_surveys(self, model, test_surveys: List[str],
                        survey_datasets: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test trained model on held-out surveys.

        Parameters:
            model: Trained ML model
            test_surveys: Surveys to test on
            survey_datasets: All survey datasets

        Returns:
            dict: Testing results
        """
        test_results = {}

        for survey in test_surveys:
            if survey in survey_datasets:
                survey_data = survey_datasets[survey]

                if 'features' in survey_data:
                    features = survey_data['features']

                    try:
                        # Get predictions
                        if hasattr(model, 'predict'):
                            predictions = model.predict(features)

                        survey_result = {
                            'n_test_samples': len(features),
                            'predictions_shape': predictions.shape if hasattr(predictions, 'shape') else len(predictions),
                            'test_success': True
                        }

                        # Compute survey-specific metrics
                        if isinstance(predictions, dict) and 'ensemble_scores' in predictions:
                            scores = predictions['ensemble_scores']
                            survey_result.update({
                                'mean_anomaly_score': float(np.mean(scores)),
                                'std_anomaly_score': float(np.std(scores)),
                                'n_detected_anomalies': int(np.sum(scores > 0.5)),
                                'detection_rate': float(np.mean(scores > 0.5))
                            })

                    except Exception as e:
                        survey_result = {
                            'error': str(e),
                            'test_success': False
                        }

                    test_results[survey] = survey_result

        return test_results

    def _analyze_survey_consistency(self, train_result: Dict, test_result: Dict,
                                  train_surveys: List[str], test_surveys: List[str]) -> Dict[str, Any]:
        """
        Analyze consistency of results across training and test surveys.

        Parameters:
            train_result: Training results
            test_result: Testing results
            train_surveys: Surveys used for training
            test_surveys: Surveys used for testing

        Returns:
            dict: Consistency analysis
        """
        consistency_metrics = {}

        # Check if training was successful
        if not train_result.get('training_success', False):
            consistency_metrics['training_failed'] = True
            return consistency_metrics

        # Analyze detection consistency across test surveys
        test_detection_rates = []
        test_mean_scores = []

        for survey, survey_result in test_result.items():
            if survey_result.get('test_success', False):
                detection_rate = survey_result.get('detection_rate', 0)
                mean_score = survey_result.get('mean_anomaly_score', 0)

                test_detection_rates.append(detection_rate)
                test_mean_scores.append(mean_score)

        if test_detection_rates:
            consistency_metrics.update({
                'mean_detection_rate': float(np.mean(test_detection_rates)),
                'std_detection_rate': float(np.std(test_detection_rates)),
                'detection_rate_consistency': float(np.std(test_detection_rates) / (np.mean(test_detection_rates) + 1e-10)),
                'n_consistent_surveys': len(test_detection_rates)
            })

            # Test for significant variation (would indicate survey systematics)
            if len(test_detection_rates) >= 3:
                # Simple F-test for variance significance
                mean_rate = np.mean(test_detection_rates)
                variance = np.var(test_detection_rates, ddof=1)
                expected_variance = mean_rate * (1 - mean_rate) / len(test_detection_rates)  # Binomial variance

                f_statistic = variance / (expected_variance + 1e-10)
                consistency_metrics['detection_variance_f_test'] = float(f_statistic)
                consistency_metrics['excess_variance_significant'] = f_statistic > 2.0  # Rough threshold

        return consistency_metrics

    def _aggregate_cv_results(self, cv_results: List[Dict]) -> Dict[str, Any]:
        """Aggregate results across all cross-validation splits."""
        aggregated = {
            'n_splits': len(cv_results),
            'successful_splits': sum(1 for r in cv_results if r['train_result'].get('training_success', False)),
            'survey_consistency_scores': [],
            'detection_rates': [],
            'mean_scores': []
        }

        for result in cv_results:
            consistency = result['consistency_analysis']

            if 'detection_rate_consistency' in consistency:
                aggregated['survey_consistency_scores'].append(consistency['detection_rate_consistency'])

            if 'mean_detection_rate' in consistency:
                aggregated['detection_rates'].append(consistency['mean_detection_rate'])

            # Collect all survey-specific results
            for survey, survey_result in result['test_result'].items():
                if survey_result.get('test_success', False):
                    if 'mean_anomaly_score' in survey_result:
                        aggregated['mean_scores'].append(survey_result['mean_anomaly_score'])

        # Compute summary statistics
        if aggregated['survey_consistency_scores']:
            aggregated['mean_consistency'] = float(np.mean(aggregated['survey_consistency_scores']))
            aggregated['std_consistency'] = float(np.std(aggregated['survey_consistency_scores']))

        if aggregated['detection_rates']:
            aggregated['overall_detection_rate'] = float(np.mean(aggregated['detection_rates']))
            aggregated['detection_rate_std'] = float(np.std(aggregated['detection_rates']))

        return aggregated

    def _create_validation_summary(self, aggregated_results: Dict) -> Dict[str, Any]:
        """Create human-readable validation summary."""
        summary = {
            'validation_type': 'Cross-Survey Validation',
            'total_splits': aggregated_results['n_splits'],
            'successful_splits': aggregated_results['successful_splits']
        }

        # Overall assessment
        if aggregated_results['successful_splits'] == aggregated_results['n_splits']:
            summary['overall_status'] = 'SUCCESS'
        elif aggregated_results['successful_splits'] > 0:
            summary['overall_status'] = 'PARTIAL_SUCCESS'
        else:
            summary['overall_status'] = 'FAILED'

        # Consistency assessment
        if 'mean_consistency' in aggregated_results:
            consistency_score = aggregated_results['mean_consistency']
            if consistency_score < 0.2:
                summary['consistency_status'] = 'HIGH_CONSISTENCY'
            elif consistency_score < 0.5:
                summary['consistency_status'] = 'MODERATE_CONSISTENCY'
            else:
                summary['consistency_status'] = 'LOW_CONSISTENCY'

            summary['consistency_score'] = consistency_score

        # Detection assessment
        if 'overall_detection_rate' in aggregated_results:
            detection_rate = aggregated_results['overall_detection_rate']
            summary['overall_detection_rate'] = detection_rate

            # Flag if detection rates are suspiciously uniform (might indicate overfitting)
            if 'detection_rate_std' in aggregated_results:
                detection_std = aggregated_results['detection_rate_std']
                summary['detection_uniformity'] = detection_std / (detection_rate + 1e-10)

        # Scientific implications
        implications = []

        if summary.get('consistency_status') == 'HIGH_CONSISTENCY':
            implications.append("High consistency across surveys suggests cosmological signal rather than survey artifacts")

        if summary.get('consistency_status') == 'LOW_CONSISTENCY':
            implications.append("Low consistency may indicate survey-specific systematics or insufficient training data")

        if summary.get('detection_uniformity', 1.0) < 0.1:
            implications.append("Very uniform detection rates across surveys may indicate methodological issues")

        summary['scientific_implications'] = implications

        return summary
