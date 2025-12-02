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

# Suppress sklearn deprecation warnings for internal API changes
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.utils.deprecation')
# Suppress sklearn RuntimeWarnings from numerical operations
warnings.filterwarnings('ignore', category=RuntimeWarning, module='sklearn.linear_model')


class LIMEExplainer:
    """
    LIME-based interpretability for anomaly detection models.
    """

    def __init__(self, predict_function: Callable,
                 feature_names: Optional[List[str]] = None,
                 n_samples: int = 1000,
                 kernel_width: Optional[float] = None):
        self.predict_function = predict_function
        self.feature_names = feature_names or [f'feature_{i}' for i in range(100)]
        self.n_samples = n_samples
        self.kernel_width = kernel_width
        self.logger = logging.getLogger(__name__)

    def explain_instance(self, instance: np.ndarray,
                        n_features: int = 10,
                        model_regressor: Optional[Any] = None) -> Dict[str, Any]:
        """Explain a single instance prediction."""
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
        X_design = self._ensure_full_rank(X_design)

        X_mean = np.mean(X_design, axis=0)
        X_std = np.std(X_design, axis=0)

        X_scaled = np.zeros_like(X_design, dtype=float)
        for col in range(X_design.shape[1]):
            if X_std[col] > 1e-10:
                X_scaled[:, col] = (X_design[:, col] - X_mean[col]) / X_std[col]
            else:
                X_scaled[:, col] = X_design[:, col] - X_mean[col]

        X_scaled = np.nan_to_num(X_scaled, nan=0.0, posinf=1e10, neginf=-1e10)
        predictions = np.nan_to_num(predictions.flatten(), nan=0.0, posinf=1e10, neginf=-1e10)
        weights = np.nan_to_num(weights, nan=0.0, posinf=1e10, neginf=-1e10)
        
        weights = np.maximum(weights, 1e-10)
        weights = weights / np.sum(weights)
        
        X_scaled = np.clip(X_scaled, -1e6, 1e6)
        predictions = np.clip(predictions, -1e6, 1e6)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
            try:
                alpha = 100.0
                model_regressor = Ridge(alpha=alpha, solver='svd')
                model_regressor.fit(X_scaled, predictions, sample_weight=weights)

                model_regressor.coef_ = np.nan_to_num(
                    model_regressor.coef_, nan=0.0, posinf=1e6, neginf=-1e6
                )
                model_regressor.coef_ = np.clip(model_regressor.coef_, -1e6, 1e6)
                model_regressor.intercept_ = np.clip(
                    float(np.nan_to_num(model_regressor.intercept_, nan=0.0, posinf=1e6, neginf=-1e6)),
                    -1e6, 1e6
                )

            except (np.linalg.LinAlgError, ValueError) as e:
                self.logger.warning(f"sklearn Ridge failed ({e}), using numpy lstsq fallback")
                try:
                    lambda_reg = 1.0
                    n_features_dim = X_scaled.shape[1]
                    X_aug = np.vstack([X_scaled, np.sqrt(lambda_reg) * np.eye(n_features_dim)])
                    y_aug = np.concatenate([predictions.flatten(), np.zeros(n_features_dim)])
                    
                    coeffs, residuals, rank, s = lstsq(X_aug, y_aug)
                    coeffs = np.nan_to_num(coeffs, nan=0.0, posinf=1e6, neginf=-1e6)
                    coeffs = np.clip(coeffs, -1e6, 1e6)
                    intercept = 0.0

                    class RobustRegressor:
                        def __init__(self, coef, intercept):
                            self.coef_ = coef
                            self.intercept_ = intercept
                        def predict(self, X):
                            X_safe = np.clip(X, -1e6, 1e6)
                            result = X_safe @ self.coef_ + self.intercept_
                            return np.clip(result, -1e6, 1e6)

                    model_regressor = RobustRegressor(coeffs, intercept)

                except (np.linalg.LinAlgError, ValueError) as e2:
                    self.logger.error(f"All regression methods failed: {e2}, using zero coefficients")
                    class ZeroRegressor:
                        def __init__(self):
                            self.coef_ = np.zeros(X_scaled.shape[1])
                            self.intercept_ = 0.0
                        def predict(self, X):
                            return np.zeros(X.shape[0])
                    model_regressor = ZeroRegressor()

        raw_coefficients = model_regressor.coef_
        X_std_safe = np.where(X_std > 1e-10, X_std, 1.0)
        coefficients = raw_coefficients / X_std_safe
        coefficients = np.clip(coefficients, -1e6, 1e6)

        feature_importances = self._get_top_features(coefficients, n_features)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning)
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
        Generate perturbed samples around instance.
        Uses robust perturbation scales suitable for log-scaled or diverse features.
        """
        n_features = len(instance)
        perturbed_samples = []
        binary_perturbations = []

        perturbed_samples.append(instance.copy())
        binary_perturbations.append(np.ones(n_features, dtype=int))

        min_samples_needed = max(n_samples - 1, n_features + 1)
        samples_generated = 0
        max_attempts = min_samples_needed * 3
        attempts = 0

        while samples_generated < (n_samples - 1) and attempts < max_attempts:
            if samples_generated < n_features:
                perturbation_mask = np.ones(n_features, dtype=int)
                feature_to_perturb = samples_generated % n_features
                if np.random.random() < 0.8:
                    perturbation_mask[feature_to_perturb] = 0
            else:
                perturbation_mask = np.random.choice([0, 1], size=n_features, p=[0.4, 0.6])

            perturbed = instance.copy()
            perturbed_features = perturbation_mask == 0
            
            if np.any(perturbed_features):
                # Use conservative perturbation scale:
                # For log-scaled values (e.g. CMB power), perturbations should be small in log space (e.g. 0.01-0.1)
                # For generic features, scale by small fraction of value.
                # We use 0.05 * abs(value) + 1e-6 as a robust baseline.
                # This prevents massive unphysical jumps (e.g. factor of 15 in linear space for log features).
                noise_scale = 0.05 * np.abs(instance[perturbed_features]) + 1e-6
                noise = np.random.normal(0, noise_scale)
                perturbed[perturbed_features] += noise
            else:
                noise = np.random.normal(0, 1e-6, size=len(instance))
                perturbed += noise

            perturbed_samples.append(perturbed)
            binary_perturbations.append(perturbation_mask.astype(int))

            samples_generated += 1
            attempts += 1

        perturbed_samples = perturbed_samples[:n_samples]
        binary_perturbations = binary_perturbations[:n_samples]

        perturbed_samples = np.array(perturbed_samples)
        binary_perturbations = np.array(binary_perturbations)

        binary_matrix = binary_perturbations[1:]
        if binary_matrix.shape[0] > 0:
            for col_idx in range(binary_matrix.shape[1]):
                col_values = binary_matrix[:, col_idx]
                if np.all(col_values == col_values[0]):
                    flip_indices = np.random.choice(len(col_values), size=max(1, len(col_values)//4), replace=False)
                    binary_matrix[flip_indices, col_idx] = 1 - binary_matrix[flip_indices, col_idx]

        return perturbed_samples, binary_perturbations

    def _ensure_full_rank(self, X: np.ndarray, tol: float = 1e-8) -> np.ndarray:
        """Ensure matrix has full column rank."""
        X_clean = X.copy()
        for col_idx in range(X.shape[1]):
            col_values = X[:, col_idx]
            if not np.isfinite(col_values).all():
                X_clean[:, col_idx] = 0.0
                continue
            col_std = np.std(col_values)
            if col_std < tol:
                col_mean = np.mean(col_values)
                noise = np.random.normal(0, max(tol, abs(col_mean) * 0.01), size=len(col_values))
                X_clean[:, col_idx] = col_mean + noise
        return X_clean

    def _euclidean_distance(self, x1: np.ndarray, x2: np.ndarray) -> float:
        return np.sqrt(np.sum((x1 - x2) ** 2))

    def _kernel_function(self, distances: np.ndarray) -> np.ndarray:
        if self.kernel_width is None:
            self.kernel_width = np.median(distances) if len(distances) > 0 else 1.0
        return np.exp(-distances ** 2 / (2 * self.kernel_width ** 2))

    def _get_top_features(self, coefficients: np.ndarray, n_features: int) -> List[Tuple[int, float]]:
        abs_coeffs = np.abs(coefficients)
        top_indices = np.argsort(abs_coeffs)[::-1][:n_features]
        top_features = [(int(idx), float(coefficients[idx])) for idx in top_indices]
        return top_features

    def _format_top_features(self, feature_importances: List[Tuple[int, float]],
                           n_features: int) -> List[Dict[str, Any]]:
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
                error_explanation = {
                    'error': f'Explanation failed for sample {idx}: {str(e)}',
                    'sample_index': int(idx),
                    'anomaly_rank': rank + 1
                }
                explanations.append(error_explanation)
        
        successful_explanations = [exp for exp in explanations if 'error' not in exp]
        summary = {
            'n_anomalies_explained': len(explanations),
            'n_successful_explanations': len(successful_explanations),
            'mean_explanation_r2': float(np.mean([exp['r2_score'] for exp in successful_explanations])) if successful_explanations else 0.0,
            'common_important_features': self._find_common_features(successful_explanations)
        }
        return {'explanations': explanations, 'summary': summary}

    def _find_common_features(self, explanations: List[Dict]) -> List[Dict[str, Any]]:
        feature_counts = {}
        for explanation in explanations:
            for feature_info in explanation['top_features']:
                feature_name = feature_info['feature_name']
                if feature_name not in feature_counts:
                    feature_counts[feature_name] = {'count': 0, 'total_importance': 0, 'directions': []}
                feature_counts[feature_name]['count'] += 1
                feature_counts[feature_name]['total_importance'] += feature_info['importance']
                feature_counts[feature_name]['directions'].append(feature_info['direction'])

        common_features = []
        for feature_name, stats in feature_counts.items():
            if stats['count'] >= 2:
                most_common_direction = max(set(stats['directions']), key=stats['directions'].count)
                common_features.append({
                    'feature_name': feature_name,
                    'frequency': stats['count'],
                    'average_importance': stats['total_importance'] / stats['count'],
                    'dominant_direction': most_common_direction
                })
        common_features.sort(key=lambda x: (x['frequency'], x['average_importance']), reverse=True)
        return common_features[:10]

    def get_explanation_stability(self, instance: np.ndarray, n_runs: int = 5) -> Dict[str, Any]:
        explanations = []
        original_seed = np.random.get_state()
        for i in range(n_runs):
            np.random.seed(i + 1000)
            explanation = self.explain_instance(instance)
            explanations.append(explanation)
        np.random.set_state(original_seed)

        r2_scores = [exp['r2_score'] for exp in explanations]
        intercepts = [exp['intercept'] for exp in explanations]
        feature_stability = self._compute_feature_stability(explanations)

        return {
            'r2_mean': float(np.mean(r2_scores)),
            'r2_std': float(np.std(r2_scores)),
            'intercept_mean': float(np.mean(intercepts)),
            'intercept_std': float(np.std(intercepts)),
            'feature_stability': feature_stability,
            'n_runs': n_runs
        }

    def _compute_feature_stability(self, explanations: List[Dict]) -> Dict[str, Any]:
        if not explanations:
            return {}
        feature_importances = {}
        for exp in explanations:
            for feature_info in exp['top_features']:
                feature_name = feature_info['feature_name']
                if feature_name not in feature_importances:
                    feature_importances[feature_name] = []
                feature_importances[feature_name].append(feature_info['importance'])

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
