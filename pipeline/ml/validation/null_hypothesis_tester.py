"""
Null Hypothesis Testing
=======================

Statistical validation using null hypothesis testing with mock datasets.
Tests whether ML detections in real data are statistically significant
compared to detections in mock data without H-ΛCDM signals.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Callable, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score
import logging
import warnings

# Suppress sklearn deprecation warnings for internal API changes
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.utils.deprecation')

try:
    from statsmodels.stats.multitest import multipletests
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    warnings.warn("statsmodels not available. Multiple testing correction will be limited.")


class NullHypothesisTester:
    """
    Null hypothesis testing for ML pattern detection significance.

    Generates mock datasets matching real data statistics but lacking
    H-ΛCDM signals, then tests if real data detections are significant.

    Critical validation: If ML detects H-ΛCDM signals in real data but
    NOT in mocks, this validates the detection capability.
    """

    def __init__(self, mock_generator: Any,
                 n_null_tests: int = 100,
                 significance_level: float = 0.05,
                 random_state: int = 42):
        """
        Initialize null hypothesis tester.

        Parameters:
            mock_generator: Mock dataset generator instance
            n_null_tests: Number of null hypothesis tests (mock datasets)
            significance_level: Statistical significance level
            random_state: Random seed
        """
        self.mock_generator = mock_generator
        self.n_null_tests = n_null_tests
        self.significance_level = significance_level
        self.random_state = random_state
        self.logger = logging.getLogger(__name__)

        np.random.seed(random_state)

    def test_null_hypothesis(self, model_factory: Callable,
                           real_dataset: Dict[str, Any],
                           modality: str = 'combined') -> Dict[str, Any]:
        """
        Test null hypothesis using mock datasets.

        Parameters:
            model_factory: Function that creates fresh model instances
            real_dataset: Real cosmological dataset
            modality: Which modality to test ('combined' or specific modality)

        Returns:
            dict: Null hypothesis testing results
        """
        self.logger.info(f"Starting null hypothesis testing with {self.n_null_tests} mock datasets")

        # Test real data first
        real_model = model_factory()
        real_result = self._test_on_dataset(real_model, real_dataset, is_real=True)

        if not real_result.get('success', False):
            return {'error': 'Failed to test on real dataset'}

        self.logger.info(f"Real data: {real_result['n_detected']} detections, rate: {real_result['detection_rate']:.3f}")

        # Test on mock datasets
        mock_results = []

        for i in range(self.n_null_tests):
            if (i + 1) % 20 == 0:
                self.logger.info(f"Null test {i + 1}/{self.n_null_tests}")

            # Generate mock dataset
            mock_data = self.mock_generator.generate_validation_dataset(modality)

            # Test model on mock data
            mock_model = model_factory()
            mock_result = self._test_on_dataset(mock_model, mock_data, is_real=False)

            if mock_result.get('success', False):
                mock_results.append(mock_result)
            else:
                self.logger.warning(f"Mock test {i} failed")

        # Statistical analysis
        statistical_analysis = self._analyze_null_tests(real_result, mock_results)

        return {
            'real_data_result': real_result,
            'mock_results': mock_results,
            'statistical_analysis': statistical_analysis,
            'significance_test': self._test_statistical_significance(real_result, mock_results),
            'validation_summary': self._create_validation_summary(statistical_analysis)
        }

    def _test_on_dataset(self, model, dataset: Dict[str, Any], is_real: bool = False) -> Dict[str, Any]:
        """
        Test model on a single dataset.

        Parameters:
            model: ML model instance
            dataset: Dataset to test on
            is_real: Whether this is real cosmological data

        Returns:
            dict: Test results
        """
        try:
            # Extract features from dataset
            features = self._extract_features_from_dataset(dataset)

            if features is None:
                return {'success': False, 'error': 'No features available'}

            # Train model (if needed)
            if hasattr(model, 'fit'):
                model.fit(features)

            # Get predictions
            if hasattr(model, 'predict'):
                predictions = model.predict(features)

            # Handle different prediction formats
            if isinstance(predictions, dict):
                if 'ensemble_scores' in predictions:
                    anomaly_scores = predictions['ensemble_scores']
                else:
                    # Fallback to first available scores
                    scores_list = list(predictions.values())
                    if scores_list and hasattr(scores_list[0], '__len__'):
                        anomaly_scores = np.mean(scores_list, axis=0)
                    else:
                        anomaly_scores = np.array(scores_list)
            else:
                anomaly_scores = np.array(predictions)

            # Apply detection threshold
            detection_threshold = 0.5  # Could be parameterized
            detections = anomaly_scores > detection_threshold

            result = {
                'success': True,
                'n_samples': len(features),
                'n_detected': int(np.sum(detections)),
                'detection_rate': float(np.mean(detections)),
                'mean_score': float(np.mean(anomaly_scores)),
                'std_score': float(np.std(anomaly_scores)),
                'max_score': float(np.max(anomaly_scores)),
                'is_real_data': is_real,
                'anomaly_scores': anomaly_scores.tolist(),
                'detections': detections.tolist()
            }

            # Additional statistics for real vs mock comparison
            if len(anomaly_scores) > 10:
                result['score_percentiles'] = np.percentile(anomaly_scores, [25, 50, 75, 90, 95, 99]).tolist()

            return result

        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'is_real_data': is_real
            }

    def _extract_features_from_dataset(self, dataset: Dict[str, Any]) -> Optional[np.ndarray]:
        """
        Extract feature matrix from dataset dictionary.

        Parameters:
            dataset: Dataset dictionary (may contain different formats)

        Returns:
            np.ndarray: Feature matrix or None
        """
        # Try different possible keys
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

        # For mock datasets, try to construct features
        if 'modalities' in dataset:
            # This would require modality-specific feature extraction
            # For now, generate random features matching expected size
            n_samples = 1000  # Default
            n_features = 128  # Default
            self.logger.warning("Using random features for mock dataset - implement proper feature extraction")
            return np.random.normal(0, 1, (n_samples, n_features))

        return None

    def _analyze_null_tests(self, real_result: Dict[str, Any],
                           mock_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze results of null hypothesis tests.

        Parameters:
            real_result: Results on real data
            mock_results: Results on mock datasets

        Returns:
            dict: Statistical analysis
        """
        successful_mocks = [r for r in mock_results if r.get('success', False)]

        analysis = {
            'n_mock_tests': len(mock_results),
            'n_successful_mocks': len(successful_mocks),
            'success_rate': len(successful_mocks) / len(mock_results) if mock_results else 0
        }

        if not successful_mocks:
            analysis['error'] = 'No successful mock tests'
            return analysis

        # Compare detection rates
        real_detection_rate = real_result.get('detection_rate', 0)
        mock_detection_rates = [r.get('detection_rate', 0) for r in successful_mocks]

        analysis['detection_rate_comparison'] = {
            'real_detection_rate': real_detection_rate,
            'mock_mean_detection_rate': float(np.mean(mock_detection_rates)),
            'mock_std_detection_rate': float(np.std(mock_detection_rates)),
            'mock_min_detection_rate': float(np.min(mock_detection_rates)),
            'mock_max_detection_rate': float(np.max(mock_detection_rates)),
            'detection_rate_difference': real_detection_rate - np.mean(mock_detection_rates)
        }

        # Compare anomaly scores
        real_mean_score = real_result.get('mean_score', 0)
        mock_mean_scores = [r.get('mean_score', 0) for r in successful_mocks]

        analysis['score_comparison'] = {
            'real_mean_score': real_mean_score,
            'mock_mean_score': float(np.mean(mock_mean_scores)),
            'mock_score_std': float(np.std(mock_mean_scores)),
            'score_difference': real_mean_score - np.mean(mock_mean_scores)
        }

        # Distribution analysis
        analysis['distribution_analysis'] = self._analyze_score_distributions(real_result, successful_mocks)

        return analysis

    def _analyze_score_distributions(self, real_result: Dict[str, Any],
                                   mock_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Analyze differences in anomaly score distributions.

        Parameters:
            real_result: Real data results
            mock_results: Mock data results

        Returns:
            dict: Distribution analysis
        """
        real_scores = np.array(real_result.get('anomaly_scores', []))

        if len(real_scores) == 0:
            return {'error': 'No real scores available'}

        # Collect mock scores
        mock_score_samples = []
        for mock_result in mock_results:
            mock_scores = mock_result.get('anomaly_scores', [])
            if mock_scores:
                mock_score_samples.extend(mock_scores)

        if not mock_score_samples:
            return {'error': 'No mock scores available'}

        mock_scores = np.array(mock_score_samples)

        # Statistical tests
        try:
            # Kolmogorov-Smirnov test for distribution difference
            ks_stat, ks_p_value = stats.ks_2samp(real_scores, mock_scores)

            # Mann-Whitney U test for location difference
            mw_stat, mw_p_value = stats.mannwhitneyu(real_scores, mock_scores, alternative='two-sided')

            distribution_analysis = {
                'ks_test_statistic': float(ks_stat),
                'ks_test_p_value': float(ks_p_value),
                'mann_whitney_statistic': float(mw_stat),
                'mann_whitney_p_value': float(mw_p_value),
                'distributions_different': ks_p_value < self.significance_level,
                'score_distributions_different': mw_p_value < self.significance_level
            }

        except Exception as e:
            distribution_analysis = {
                'error': str(e),
                'distributions_different': None,
                'score_distributions_different': None
            }

        # Percentile comparisons
        if 'score_percentiles' in real_result:
            real_percentiles = real_result['score_percentiles']
            mock_percentiles_all = []

            for mock_result in mock_results:
                if 'score_percentiles' in mock_result:
                    mock_percentiles_all.append(mock_result['score_percentiles'])

            if mock_percentiles_all:
                mock_percentiles_mean = np.mean(mock_percentiles_all, axis=0)

                distribution_analysis['percentile_comparison'] = {
                    'real_percentiles': real_percentiles,
                    'mock_mean_percentiles': mock_percentiles_mean.tolist(),
                    'percentile_differences': (np.array(real_percentiles) - mock_percentiles_mean).tolist()
                }

        return distribution_analysis

    def _test_statistical_significance(self, real_result: Dict[str, Any],
                                     mock_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Test statistical significance of real vs mock results.

        Parameters:
            real_result: Results on real data
            mock_results: Results on mock datasets

        Returns:
            dict: Significance test results
        """
        successful_mocks = [r for r in mock_results if r.get('success', False)]

        if not successful_mocks:
            return {'error': 'No successful mock tests for significance testing'}

        # Test 1: Is detection rate significantly higher than mocks?
        real_detection_rate = real_result.get('detection_rate', 0)
        mock_detection_rates = np.array([r.get('detection_rate', 0) for r in successful_mocks])

        # One-sample t-test (is real detection rate > mock mean?)
        t_stat_detection, p_value_detection = stats.ttest_1samp(
            mock_detection_rates, real_detection_rate, alternative='less'
        )

        # Test 2: Is mean anomaly score significantly higher than mocks?
        real_mean_score = real_result.get('mean_score', 0)
        mock_mean_scores = np.array([r.get('mean_score', 0) for r in successful_mocks])

        t_stat_score, p_value_score = stats.ttest_1samp(
            mock_mean_scores, real_mean_score, alternative='greater'
        )

        # Multiple testing correction (Bonferroni)
        p_values = [p_value_detection, p_value_score]
        if STATSMODELS_AVAILABLE:
            _, p_adjusted, _, _ = multipletests(p_values, method='bonferroni')
        else:
            # Simple Bonferroni correction as fallback
            p_adjusted = [min(1.0, p * len(p_values)) for p in p_values]

        significance_test = {
            'detection_rate_test': {
                't_statistic': float(t_stat_detection),
                'p_value': float(p_value_detection),
                'p_value_adjusted': float(p_adjusted[0]),
                'significant': p_adjusted[0] < self.significance_level,
                'effect_size': (real_detection_rate - np.mean(mock_detection_rates)) / np.std(mock_detection_rates)
            },
            'anomaly_score_test': {
                't_statistic': float(t_stat_score),
                'p_value': float(p_value_score),
                'p_value_adjusted': float(p_adjusted[1]),
                'significant': p_adjusted[1] < self.significance_level,
                'effect_size': (real_mean_score - np.mean(mock_mean_scores)) / np.std(mock_mean_scores)
            },
            'overall_significance': p_adjusted[0] < self.significance_level or p_adjusted[1] < self.significance_level,
            'significance_level': self.significance_level,
            'multiple_testing_correction': 'bonferroni'
        }

        return significance_test

    def _create_validation_summary(self, statistical_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Create human-readable validation summary.

        Parameters:
            statistical_analysis: Statistical analysis results

        Returns:
            dict: Validation summary
        """
        summary = {
            'validation_type': 'Null Hypothesis Testing',
            'n_mock_datasets': self.n_null_tests,
            'significance_level': self.significance_level
        }

        # Overall validation status
        significance_test = statistical_analysis.get('significance_test', {})

        if significance_test.get('overall_significance', False):
            summary['validation_status'] = 'SIGNIFICANT_DETECTION'
            summary['conclusion'] = 'ML detections are statistically significant compared to null hypothesis'
        else:
            summary['validation_status'] = 'NOT_SIGNIFICANT'
            summary['conclusion'] = 'ML detections are not statistically significant compared to null hypothesis'

        # Effect sizes
        detection_effect = significance_test.get('detection_rate_test', {}).get('effect_size', 0)
        score_effect = significance_test.get('anomaly_score_test', {}).get('effect_size', 0)

        summary['effect_sizes'] = {
            'detection_rate': float(detection_effect),
            'anomaly_score': float(score_effect)
        }

        # Scientific implications
        implications = []

        if summary['validation_status'] == 'SIGNIFICANT_DETECTION':
            implications.append("ML successfully detects patterns in real cosmological data that are absent in mocks")
            implications.append("This validates the ML methodology for cosmological pattern detection")

            if abs(detection_effect) > 1.0:
                implications.append("Large effect size suggests strong cosmological signal")

        else:
            implications.append("ML detections are consistent with null hypothesis (mock data)")
            implications.append("Either no real signal present, or methodological issues with detection")

            if statistical_analysis.get('success_rate', 0) < 0.8:
                implications.append("Low mock test success rate may indicate methodological issues")

        summary['scientific_implications'] = implications

        return summary
