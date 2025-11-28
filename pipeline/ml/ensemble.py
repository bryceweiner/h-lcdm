"""
Ensemble Aggregation
===================

Aggregates anomaly scores from multiple detection methods
with sophisticated ranking and consensus algorithms.
"""

import numpy as np
from typing import Dict, Any, List, Optional, Tuple
from scipy import stats
from sklearn.metrics import roc_auc_score


class EnsembleAggregator:
    """
    Advanced ensemble aggregation for anomaly detection scores.

    Implements multiple aggregation strategies and provides
    robust anomaly ranking and consensus detection.
    """

    def __init__(self, methods: List[str],
                 aggregation_method: str = 'weighted_average',
                 consensus_threshold: float = 0.5):
        """
        Initialize ensemble aggregator.

        Parameters:
            methods: List of detection method names
            aggregation_method: 'weighted_average', 'rank_aggregation', 'consensus'
            consensus_threshold: Threshold for consensus detection
        """
        self.methods = methods
        self.aggregation_method = aggregation_method
        self.consensus_threshold = consensus_threshold

        # Learned weights (initialized equally)
        self.weights = np.ones(len(methods)) / len(methods)

        # Performance tracking
        self.method_performance = {}

    def aggregate_scores(self, individual_scores: Dict[str, np.ndarray],
                        return_details: bool = False) -> Dict[str, Any]:
        """
        Aggregate anomaly scores from multiple methods.

        Parameters:
            individual_scores: Dict of method_name -> anomaly_scores
            return_details: Whether to return detailed aggregation info

        Returns:
            dict: Aggregated results
        """
        # Ensure all methods have scores
        n_samples = len(next(iter(individual_scores.values())))
        score_matrix = np.zeros((len(self.methods), n_samples))

        for i, method in enumerate(self.methods):
            if method in individual_scores:
                score_matrix[i] = individual_scores[method]
            else:
                # Use zeros if method missing
                score_matrix[i] = np.zeros(n_samples)

        # Apply aggregation method
        if self.aggregation_method == 'weighted_average':
            ensemble_scores = self._weighted_average_aggregation(score_matrix)
        elif self.aggregation_method == 'rank_aggregation':
            ensemble_scores = self._rank_aggregation(score_matrix)
        elif self.aggregation_method == 'consensus':
            ensemble_scores = self._consensus_aggregation(score_matrix)
        else:
            raise ValueError(f"Unknown aggregation method: {self.aggregation_method}")

        # Generate final predictions
        predictions = ensemble_scores > self.consensus_threshold

        # Rank anomalies (higher score = more anomalous)
        anomaly_ranks = np.argsort(-ensemble_scores)  # Sort descending
        anomaly_ranking = np.empty_like(anomaly_ranks)
        anomaly_ranking[anomaly_ranks] = np.arange(len(anomaly_ranks))

        result = {
            'ensemble_scores': ensemble_scores,
            'predictions': predictions,
            'anomaly_ranks': anomaly_ranking,
            'top_anomalies': self._get_top_anomalies(ensemble_scores, n_top=10),
            'aggregation_method': self.aggregation_method,
            'method_weights': dict(zip(self.methods, self.weights))
        }

        if return_details:
            result.update({
                'individual_scores': individual_scores,
                'score_matrix': score_matrix,
                'method_correlations': self._compute_method_correlations(score_matrix),
                'ensemble_statistics': self._compute_ensemble_statistics(ensemble_scores)
            })

        return result

    def _weighted_average_aggregation(self, score_matrix: np.ndarray) -> np.ndarray:
        """Weighted average of anomaly scores."""
        return np.average(score_matrix, axis=0, weights=self.weights)

    def _rank_aggregation(self, score_matrix: np.ndarray) -> np.ndarray:
        """
        Rank aggregation using Borda count method.

        Converts scores to ranks and aggregates rankings.
        """
        n_methods, n_samples = score_matrix.shape
        rank_matrix = np.zeros_like(score_matrix)

        # Convert scores to ranks for each method (higher score = higher rank)
        for i in range(n_methods):
            # Rank with higher scores getting higher ranks (1 = lowest score, N = highest score)
            ranks = stats.rankdata(score_matrix[i])
            rank_matrix[i] = ranks

        # Aggregate ranks (Borda count)
        borda_scores = np.sum(rank_matrix * self.weights[:, np.newaxis], axis=0)

        # Normalize to 0-1 range
        borda_scores = (borda_scores - np.min(borda_scores)) / (np.max(borda_scores) - np.min(borda_scores) + 1e-10)

        return borda_scores

    def _consensus_aggregation(self, score_matrix: np.ndarray) -> np.ndarray:
        """
        Consensus-based aggregation.

        Uses fraction of methods agreeing on anomaly status.
        """
        # Convert to binary predictions
        binary_predictions = (score_matrix > self.consensus_threshold).astype(int)

        # Count agreements
        consensus_scores = np.mean(binary_predictions, axis=0)

        return consensus_scores

    def _get_top_anomalies(self, ensemble_scores: np.ndarray, n_top: int = 10) -> List[Dict[str, Any]]:
        """Get top N most anomalous samples."""
        # Get indices of top anomalies
        top_indices = np.argsort(-ensemble_scores)[:n_top]

        top_anomalies = []
        for rank, idx in enumerate(top_indices):
            top_anomalies.append({
                'sample_index': int(idx),
                'anomaly_score': float(ensemble_scores[idx]),
                'rank': rank + 1
            })

        return top_anomalies

    def _compute_method_correlations(self, score_matrix: np.ndarray) -> Dict[str, float]:
        """Compute correlations between detection methods."""
        correlations = {}
        n_methods = len(self.methods)

        for i in range(n_methods):
            for j in range(i+1, n_methods):
                corr = np.corrcoef(score_matrix[i], score_matrix[j])[0, 1]
                key = f"{self.methods[i]}_{self.methods[j]}"
                correlations[key] = float(corr)

        return correlations

    def _compute_ensemble_statistics(self, ensemble_scores: np.ndarray) -> Dict[str, float]:
        """Compute statistics of ensemble scores."""
        return {
            'mean_score': float(np.mean(ensemble_scores)),
            'std_score': float(np.std(ensemble_scores)),
            'median_score': float(np.median(ensemble_scores)),
            'score_range': float(np.max(ensemble_scores) - np.min(ensemble_scores)),
            'skewness': float(stats.skew(ensemble_scores)),
            'kurtosis': float(stats.kurtosis(ensemble_scores))
        }

    def learn_optimal_weights(self, validation_scores: Dict[str, np.ndarray],
                             ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Learn optimal weights for ensemble methods.

        Parameters:
            validation_scores: Method scores on validation set
            ground_truth: Known anomaly labels (if available)

        Returns:
            dict: Weight learning results
        """
        if ground_truth is None:
            # Use ensemble agreement as proxy for ground truth
            # Methods that agree are likely detecting real anomalies
            score_matrix = np.array([validation_scores[method] for method in self.methods])
            ground_truth = (np.mean(score_matrix > self.consensus_threshold, axis=0) > 0.5).astype(int)

        # Optimize weights to maximize AUC or accuracy
        best_weights = self.weights.copy()
        best_score = 0

        # Simple grid search (in practice, use more sophisticated optimization)
        weight_candidates = np.array(np.meshgrid(*[np.linspace(0.1, 1.0, 5) for _ in self.methods])).T.reshape(-1, len(self.methods))
        weight_candidates = weight_candidates / weight_candidates.sum(axis=1, keepdims=True)  # Normalize

        for weights in weight_candidates:
            ensemble_scores = np.average(score_matrix, axis=0, weights=weights)

            try:
                auc_score = roc_auc_score(ground_truth, ensemble_scores)
                if auc_score > best_score:
                    best_score = auc_score
                    best_weights = weights
            except:
                # Fallback to simple accuracy
                predictions = (ensemble_scores > self.consensus_threshold).astype(int)
                accuracy = np.mean(predictions == ground_truth)
                if accuracy > best_score:
                    best_score = accuracy
                    best_weights = weights

        self.weights = best_weights

        return {
            'optimal_weights': dict(zip(self.methods, best_weights)),
            'validation_score': best_score,
            'weight_optimization_method': 'grid_search'
        }

    def evaluate_ensemble_performance(self, test_scores: Dict[str, np.ndarray],
                                    ground_truth: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Evaluate ensemble performance on test data.

        Parameters:
            test_scores: Method scores on test set
            ground_truth: Known anomaly labels

        Returns:
            dict: Performance metrics
        """
        ensemble_result = self.aggregate_scores(test_scores, return_details=True)

        performance = {
            'ensemble_statistics': ensemble_result['ensemble_statistics'],
            'n_detected_anomalies': int(np.sum(ensemble_result['predictions'])),
            'detection_rate': float(np.mean(ensemble_result['predictions']))
        }

        # Individual method performance
        individual_performance = {}
        for method in self.methods:
            if method in test_scores:
                scores = test_scores[method]
                individual_performance[method] = {
                    'mean_score': float(np.mean(scores)),
                    'std_score': float(np.std(scores)),
                    'detection_rate': float(np.mean(scores > self.consensus_threshold))
                }

        performance['individual_performance'] = individual_performance

        # Method agreement analysis
        if len(self.methods) > 1:
            score_matrix = np.array([test_scores[method] for method in self.methods if method in test_scores])
            binary_predictions = (score_matrix > self.consensus_threshold).astype(int)
            agreement_matrix = np.corrcoef(binary_predictions)

            performance['method_agreement'] = {
                'agreement_matrix': agreement_matrix.tolist(),
                'mean_agreement': float(np.mean(agreement_matrix[np.triu_indices_from(agreement_matrix, k=1)]))
            }

        return performance

    def calibrate_thresholds(self, calibration_scores: Dict[str, np.ndarray],
                           expected_anomaly_rate: float = 0.1) -> Dict[str, Any]:
        """
        Calibrate detection thresholds based on expected anomaly rate.

        Parameters:
            calibration_scores: Scores on calibration set
            expected_anomaly_rate: Expected fraction of anomalies

        Returns:
            dict: Calibrated thresholds
        """
        ensemble_result = self.aggregate_scores(calibration_scores)
        scores = ensemble_result['ensemble_scores']

        # Set threshold to achieve expected anomaly rate
        sorted_scores = np.sort(scores)[::-1]  # Descending
        threshold_idx = int(len(sorted_scores) * expected_anomaly_rate)
        calibrated_threshold = sorted_scores[threshold_idx] if threshold_idx < len(sorted_scores) else sorted_scores[-1]

        # Update consensus threshold
        old_threshold = self.consensus_threshold
        self.consensus_threshold = calibrated_threshold

        return {
            'old_threshold': old_threshold,
            'new_threshold': calibrated_threshold,
            'expected_anomaly_rate': expected_anomaly_rate,
            'actual_anomaly_rate': float(np.mean(scores > calibrated_threshold))
        }
