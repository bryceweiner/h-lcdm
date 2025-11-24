"""
LIME Explainer
==============

Local Interpretable Model-agnostic Explanations for anomaly detection.
Provides local explanations for individual anomaly predictions.
"""

import numpy as np
import logging
from typing import Dict, Any, List, Optional, Callable, Tuple
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from scipy.linalg import lstsq
import warnings


class LIMEExplainer:
    """
    LIME-based interpretability for anomaly detection models.

    Generates local explanations by perturbing input features and
    fitting interpretable models to understand feature contributions.
    """

    def __init__(self, predict_function: Callable,
                 feature_names: Optional[List[str]] = None,
                 n_samples: int = 1000,
                 kernel_width: Optional[float] = None):
        """
        Initialize LIME explainer.

        Parameters:
            predict_function: Function that takes features and returns anomaly scores
            feature_names: Names of input features
            n_samples: Number of perturbed samples for explanation
            kernel_width: Kernel width for weighting (auto if None)
        """
        self.predict_function = predict_function
        self.feature_names = feature_names or [f'feature_{i}' for i in range(100)]  # Default
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.logger = logging.getLogger(__name__)

    def explain_instance(self, instance: np.ndarray,
                        n_features: int = 10,
                        model_regressor: Optional[Any] = None) -> Dict[str, Any]:
        """
        Explain a single instance prediction.

        Parameters:
            instance: Input instance to explain (1D array)
            n_features: Number of top features to include in explanation
            model_regressor: Sklearn regressor for explanation (default: Ridge)

        Returns:
            dict: Explanation results
        """
        # Ensure instance is 2D
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)

        # Generate perturbed samples around the instance
        perturbed_samples, binary_perturbations = self._generate_perturbations(instance.flatten(), self.n_samples)

        # Get predictions for perturbed samples
        predictions = self.predict_function(perturbed_samples)

        # Compute distances and weights
        distances = np.array([self._euclidean_distance(instance.flatten(), sample) for sample in perturbed_samples])
        weights = self._kernel_function(distances)

        # Fit interpretable model with matrix conditioning
        if model_regressor is None:
            model_regressor = Ridge(alpha=1.0)

        # Prepare design matrix with robust conditioning
        X_design = binary_perturbations.astype(float)

        # Ensure matrix has full column rank by adding small perturbations if needed
        X_design = self._ensure_full_rank(X_design)

        # Center and scale the design matrix to improve conditioning
        X_mean = np.mean(X_design, axis=0)
        X_std = np.std(X_design, axis=0)

        # Handle constant columns properly - don't scale them
        X_scaled = np.zeros_like(X_design, dtype=float)
        for col in range(X_design.shape[1]):
            if X_std[col] > 1e-10:  # Non-constant column
                X_scaled[:, col] = (X_design[:, col] - X_mean[col]) / X_std[col]
            else:  # Constant column - center only
                X_scaled[:, col] = X_design[:, col] - X_mean[col]

        # Use robust linear regression with fallback to numpy lstsq
        try:
            # Try sklearn Ridge with strong regularization first
            alpha = 10.0  # Strong regularization by default
            model_regressor = Ridge(alpha=alpha)

            # Use scaled design matrix
            model_regressor.fit(X_scaled, predictions, sample_weight=weights)

        except (RuntimeWarning, np.linalg.LinAlgError, ValueError) as e:
            # Fallback to numpy's lstsq with Tikhonov regularization
            self.logger.warning(f"sklearn Ridge failed ({e}), using numpy lstsq fallback")

            # Use scipy's robust lstsq with regularization
            try:
                # Add regularization by augmenting the matrix
                lambda_reg = 1.0
                n_features = X_scaled.shape[1]

                # Create augmented matrix for regularization
                X_aug = np.vstack([X_scaled, np.sqrt(lambda_reg) * np.eye(n_features)])
                y_aug = np.concatenate([predictions.flatten(), np.zeros(n_features)])

                # Use scipy's lstsq (more robust than numpy's)
                coeffs, residuals, rank, s = lstsq(X_aug, y_aug)

                intercept = 0.0  # Simplified intercept

                # Create a mock regressor object
                class RobustRegressor:
                    def __init__(self, coef, intercept):
                        self.coef_ = coef
                        self.intercept_ = intercept

                    def predict(self, X):
                        return X @ self.coef_ + self.intercept_

                model_regressor = RobustRegressor(coeffs, intercept)

            except (np.linalg.LinAlgError, ValueError) as e:
                # Final fallback: use zeros
                self.logger.error(f"All regression methods failed: {e}, using zero coefficients")
                class ZeroRegressor:
                    def __init__(self):
                        self.coef_ = np.zeros(X_scaled.shape[1])
                        self.intercept_ = 0.0

                    def predict(self, X):
                        return np.zeros(X.shape[0])

                model_regressor = ZeroRegressor()

        # Get feature importances (scale back to original space)
        raw_coefficients = model_regressor.coef_
        # Coefficients in scaled space need to be transformed back
        coefficients = raw_coefficients / X_std  # Scale back to original feature space

        # Get top features
        feature_importances = self._get_top_features(coefficients, n_features)

        # Compute explanation quality metrics using scaled matrix
        explained_predictions = model_regressor.predict(X_scaled)
        r2 = r2_score(predictions, explained_predictions, sample_weight=weights)

        explanation = {
            'instance': instance.flatten(),
            'predicted_score': float(self.predict_function(instance)[0]),
            'feature_importances': feature_importances,
            'intercept': float(model_regressor.intercept_),
            'r2_score': float(r2),
            'n_samples_used': self.n_samples,
            'top_features': self._format_top_features(feature_importances, n_features)
        }

        return explanation

    def _generate_perturbations(self, instance: np.ndarray, n_samples: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate perturbed samples around instance with guaranteed full-rank binary matrix.

        Parameters:
            instance: Original instance
            n_samples: Number of samples to generate

        Returns:
            tuple: (perturbed_samples, binary_perturbations)
        """
        n_features = len(instance)
        perturbed_samples = []
        binary_perturbations = []

        # Include original instance
        perturbed_samples.append(instance.copy())
        binary_perturbations.append(np.ones(n_features, dtype=int))

        # Ensure we have enough samples to cover all features
        min_samples_needed = max(n_samples - 1, n_features + 1)

        # Generate perturbed samples with guaranteed feature coverage
        samples_generated = 0
        max_attempts = min_samples_needed * 3  # Prevent infinite loops
        attempts = 0

        while samples_generated < (n_samples - 1) and attempts < max_attempts:
            # Strategy 1: For first n_features samples, ensure each feature is perturbed at least once
            if samples_generated < n_features:
                # Force perturbation of the (samples_generated)th feature
                perturbation_mask = np.ones(n_features, dtype=int)
                feature_to_perturb = samples_generated % n_features

                # Randomly decide whether to actually perturb or not (but ensure coverage)
                if np.random.random() < 0.8:  # 80% chance to actually perturb
                    perturbation_mask[feature_to_perturb] = 0  # 0 means "perturbed"
                # Else keep as 1 (not perturbed) but still count as coverage attempt

            else:
                # Strategy 2: Random perturbations for remaining samples
                perturbation_mask = np.random.choice([0, 1], size=n_features, p=[0.4, 0.6])

            # Create perturbed sample
            perturbed = instance.copy()

            # For features marked as perturbed (mask = 0), sample from normal distribution
            perturbed_features = perturbation_mask == 0
            if np.any(perturbed_features):
                # Use feature-specific noise based on typical ranges
                noise_scale = 0.1 * np.abs(instance[perturbed_features]) + 1e-6
                noise = np.random.normal(0, noise_scale)
                perturbed[perturbed_features] += noise
            else:
                # If no features are perturbed, add small random noise to avoid identical samples
                noise = np.random.normal(0, 1e-6, size=len(instance))
                perturbed += noise

            perturbed_samples.append(perturbed)
            binary_perturbations.append(perturbation_mask.astype(int))

            samples_generated += 1
            attempts += 1

        # Ensure we have the right number of samples
        perturbed_samples = perturbed_samples[:n_samples]
        binary_perturbations = binary_perturbations[:n_samples]

        perturbed_samples = np.array(perturbed_samples)
        binary_perturbations = np.array(binary_perturbations)

        # Final check: ensure binary matrix has full column rank
        binary_matrix = binary_perturbations[1:]  # Exclude original instance row
        if binary_matrix.shape[0] > 0:
            # Check for constant columns and fix them
            for col_idx in range(binary_matrix.shape[1]):
                col_values = binary_matrix[:, col_idx]
                if np.all(col_values == col_values[0]):  # Constant column
                    # Flip a random subset to ensure variation
                    flip_indices = np.random.choice(len(col_values), size=max(1, len(col_values)//4), replace=False)
                    binary_matrix[flip_indices, col_idx] = 1 - binary_matrix[flip_indices, col_idx]

        return perturbed_samples, binary_perturbations

    def _ensure_full_rank(self, X: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """
        Ensure matrix has full column rank by handling degenerate columns.

        Parameters:
            X: Input matrix
            tol: Tolerance for rank determination

        Returns:
            np.ndarray: Matrix with full column rank
        """
        X_clean = X.copy()

        # Check for problematic columns (constant, inf, nan)
        for col_idx in range(X.shape[1]):
            col_values = X[:, col_idx]

            # Check for invalid values
            if not np.isfinite(col_values).all():
                self.logger.warning(f"Column {col_idx} contains non-finite values, replacing with zeros")
                X_clean[:, col_idx] = 0.0
                continue

            # Check for constant columns
            col_std = np.std(col_values)
            if col_std < tol:  # Essentially constant
                # Replace with small random noise around mean
                col_mean = np.mean(col_values)
                noise = np.random.normal(0, max(tol, abs(col_mean) * 0.01), size=len(col_values))
                X_clean[:, col_idx] = col_mean + noise

        return X_clean

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        """Compute Euclidean distance between two vectors."""
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _kernel_function(self, distances: np.ndarray) -> np.ndarray:
        """
        Compute kernel weights based on distances.

        Uses exponential kernel: exp(-d^2 / (2 * sigma^2))
        """
        if self.kernel_width is None:
            # Auto kernel width based on median distance
            self.kernel_width = np.median(distances) if len(distances) > 0 else 1.0

        return np.exp(-distances ** 2 / (2 * self.kernel_width ** 2))

    def _get_top_features(self, coefficients: np.ndarray, n_features: int) -> List[Tuple[int, float]]:
        """
        Get top N most important features by absolute coefficient value.

        Returns:
            list: [(feature_index, coefficient), ...] sorted by importance
        """
        abs_coeffs = np.abs(coefficients)
        top_indices = np.argsort(abs_coeffs)[::-1][:n_features]

        top_features = [(int(idx), float(coefficients[idx])) for idx in top_indices]
        return top_features

    def _format_top_features(self, feature_importances: List[Tuple[int, float]],
                           n_features: int) -> List[Dict[str, Any]]:
        """
        Format top features for human-readable output.

        Parameters:
            feature_importances: List of (index, coefficient) tuples
            n_features: Number of top features to format

        Returns:
            list: Formatted feature explanations
        """
        formatted = []

        for idx, coeff in feature_importances[:n_features]:
            feature_name = self.feature_names[idx] if idx < len(self.feature_names) else f'feature_{idx}'

            explanation = {
                'feature_name': feature_name,
                'feature_index': idx,
                'coefficient': coeff,
                'importance': abs(coeff),
                'direction': 'increases' if coeff > 0 else 'decreases',
                'contribution': 'anomaly score' if coeff > 0 else 'normal score'
            }

            formatted.append(explanation)

        return formatted

    def explain_anomalies(self, features: np.ndarray,
                         anomaly_scores: np.ndarray,
                         n_anomalies: int = 5,
                         n_features_per_explanation: int = 5) -> Dict[str, Any]:
        """
        Explain top anomalies in a dataset.

        Parameters:
            features: Feature matrix (n_samples, n_features)
            anomaly_scores: Anomaly scores for each sample
            n_anomalies: Number of top anomalies to explain
            n_features_per_explanation: Features per explanation

        Returns:
            dict: Explanations for top anomalies
        """
        # Get indices of top anomalies
        top_anomaly_indices = np.argsort(-anomaly_scores)[:n_anomalies]

        explanations = []
        for rank, idx in enumerate(top_anomaly_indices):
            instance_features = features[idx]
            try:
                explanation = self.explain_instance(
                    instance_features,
                    n_features=n_features_per_explanation
                )

                explanation['sample_index'] = int(idx)
                explanation['anomaly_rank'] = rank + 1
                explanations.append(explanation)
            except Exception as e:
                # Handle individual explanation failures gracefully
                error_explanation = {
                    'error': f'Explanation failed for sample {idx}: {str(e)}',
                    'sample_index': int(idx),
                    'anomaly_rank': rank + 1
                }
                explanations.append(error_explanation)

        # Summary statistics (only for successful explanations)
        successful_explanations = [exp for exp in explanations if 'error' not in exp]
        summary = {
            'n_anomalies_explained': len(explanations),
            'n_successful_explanations': len(successful_explanations),
            'mean_explanation_r2': float(np.mean([exp['r2_score'] for exp in successful_explanations])) if successful_explanations else 0.0,
            'common_important_features': self._find_common_features(successful_explanations)
        }

        return {
            'explanations': explanations,
            'summary': summary
        }

    def _find_common_features(self, explanations: List[Dict]) -> List[Dict[str, Any]]:
        """
        Find features that are commonly important across explanations.

        Parameters:
            explanations: List of explanation dictionaries

        Returns:
            list: Common important features
        """
        feature_counts = {}

        for explanation in explanations:
            for feature_info in explanation['top_features']:
                feature_name = feature_info['feature_name']
                if feature_name not in feature_counts:
                    feature_counts[feature_name] = {
                        'count': 0,
                        'total_importance': 0,
                        'directions': []
                    }

                feature_counts[feature_name]['count'] += 1
                feature_counts[feature_name]['total_importance'] += feature_info['importance']
                feature_counts[feature_name]['directions'].append(feature_info['direction'])

        # Filter to features that appear in multiple explanations
        common_features = []
        for feature_name, stats in feature_counts.items():
            if stats['count'] >= 2:  # Appears in at least 2 explanations
                most_common_direction = max(set(stats['directions']), key=stats['directions'].count)

                common_features.append({
                    'feature_name': feature_name,
                    'frequency': stats['count'],
                    'average_importance': stats['total_importance'] / stats['count'],
                    'dominant_direction': most_common_direction
                })

        # Sort by frequency and importance
        common_features.sort(key=lambda x: (x['frequency'], x['average_importance']), reverse=True)

        return common_features[:10]  # Top 10 common features

    def get_explanation_stability(self, instance: np.ndarray,
                                n_runs: int = 5) -> Dict[str, Any]:
        """
        Assess stability of explanations by running multiple times.

        Parameters:
            instance: Instance to explain
            n_runs: Number of explanation runs

        Returns:
            dict: Stability metrics
        """
        explanations = []

        # Generate multiple explanations with different random seeds
        original_seed = np.random.get_state()
        for i in range(n_runs):
            np.random.seed(i + 1000)  # Different seeds
            explanation = self.explain_instance(instance)
            explanations.append(explanation)

        np.random.set_state(original_seed)  # Restore original seed

        # Compute stability metrics
        r2_scores = [exp['r2_score'] for exp in explanations]
        intercepts = [exp['intercept'] for exp in explanations]

        # Feature importance stability
        feature_stability = self._compute_feature_stability(explanations)

        stability_metrics = {
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'intercept_mean': float(np.mean(intercepts)),
            'intercept_std': float(np.std(intercepts)),
            'feature_stability': feature_stability,
            'n_runs': n_runs
        }

        return stability_metrics

    def _compute_feature_stability(self, explanations: List[Dict]) -> Dict[str, Any]:
        """Compute stability of feature importances across runs."""
        if not explanations:
            return {}

        # Collect all feature importances
        feature_importances = {}
        for exp in explanations:
            for feature_info in exp['top_features']:
                feature_name = feature_info['feature_name']
                if feature_name not in feature_importances:
                    feature_importances[feature_name] = []
                feature_importances[feature_name].append(feature_info['importance'])

        # Compute stability statistics
        stability_stats = {}
        for feature_name, importances in feature_importances.items():
            if len(importances) >= 2:
                stability_stats[feature_name] = {
                    'mean_importance': float(np.mean(importances)),
                    'importance_std': float(np.std(importances)),
                    'cv_importance': float(np.std(importances) / (np.mean(importances) + 1e-10)),
                    'n_occurrences': len(importances)
                }

        return stability_stats
