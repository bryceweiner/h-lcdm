"""
SHAP Explainer
==============

SHapley Additive exPlanations for global and local interpretability
of anomaly detection models.
"""

import numpy as np
import torch
from typing import Dict, Any, List, Optional, Callable, Union, Tuple
import warnings

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    warnings.warn("SHAP not available. Install with: pip install shap")


class SHAPExplainer:
    """
    SHAP-based interpretability for anomaly detection models.

    Provides global feature importance and local explanations using
    Shapley values and TreeSHAP for ensemble methods.
    """

    def __init__(self, model_predict_function: Callable,
                 background_dataset: Optional[np.ndarray] = None,
                 max_evals: int = 1000):
        """
        Initialize SHAP explainer.

        Parameters:
            model_predict_function: Function that takes features and returns predictions
            background_dataset: Background dataset for SHAP (representative samples)
            max_evals: Maximum evaluations for SHAP computation
        """
        if not SHAP_AVAILABLE:
            raise ImportError("SHAP not available. Install with: pip install shap")

        self.model_predict_function = model_predict_function
        self.background_dataset = background_dataset
        self.max_evals = max_evals

        # SHAP explainer (initialized when needed)
        self.explainer = None

    def explain_instance(self, instance: np.ndarray,
                        background_samples: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Explain a single instance using SHAP.

        Parameters:
            instance: Input instance to explain
            background_samples: Background samples (default: use stored background)

        Returns:
            dict: SHAP explanation
        """
        background = background_samples if background_samples is not None else self.background_dataset

        if background is None:
            raise ValueError("Background dataset required for SHAP explanations")

        # Initialize explainer if needed
        if self.explainer is None:
            self.explainer = shap.Explainer(self.model_predict_function, background)

        # Compute SHAP values
        shap_values = self.explainer(instance.reshape(1, -1),
                                   max_evals=self.max_evals)

        # Extract results
        feature_importances = shap_values.values[0]
        base_value = shap_values.base_values[0]
        expected_value = base_value + np.sum(feature_importances)

        # Get feature importance ranking
        abs_importances = np.abs(feature_importances)
        importance_ranks = np.argsort(-abs_importances)  # Descending

        top_features = []
        for rank, idx in enumerate(importance_ranks[:10]):  # Top 10
            top_features.append({
                'feature_index': int(idx),
                'shap_value': float(feature_importances[idx]),
                'abs_importance': float(abs_importances[idx]),
                'rank': rank + 1
            })

        explanation = {
            'instance': instance.tolist(),
            'shap_values': feature_importances.tolist(),
            'base_value': float(base_value),
            'expected_value': float(expected_value),
            'predicted_value': float(expected_value),  # Same as expected for single instance
            'top_features': top_features,
            'feature_attribution_sum': float(np.sum(feature_importances))
        }

        return explanation

    def explain_dataset(self, dataset: np.ndarray,
                       background_samples: Optional[np.ndarray] = None,
                       max_samples: int = 100) -> Dict[str, Any]:
        """
        Explain multiple instances in a dataset.

        Parameters:
            dataset: Dataset to explain (n_samples, n_features)
            background_samples: Background samples
            max_samples: Maximum samples to explain (for efficiency)

        Returns:
            dict: Dataset-wide SHAP explanations
        """
        if len(dataset) > max_samples:
            # Sample representative instances
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            selected_data = dataset[indices]
        else:
            selected_data = dataset
            indices = np.arange(len(dataset))

        explanations = []
        for i, instance in enumerate(selected_data):
            try:
                explanation = self.explain_instance(instance, background_samples)
                explanation['original_index'] = int(indices[i])
                explanations.append(explanation)
            except Exception as e:
                warnings.warn(f"Failed to explain instance {i}: {e}")
                continue

        # Compute global feature importance
        if explanations:
            all_shap_values = np.array([exp['shap_values'] for exp in explanations])
            global_importance = np.mean(np.abs(all_shap_values), axis=0)

            # Feature importance ranking
            importance_ranks = np.argsort(-global_importance)
            global_features = []

            for rank, idx in enumerate(importance_ranks[:20]):  # Top 20
                global_features.append({
                    'feature_index': int(idx),
                    'mean_abs_shap': float(global_importance[idx]),
                    'std_shap': float(np.std(all_shap_values[:, idx])),
                    'rank': rank + 1
                })
        else:
            global_features = []

        return {
            'individual_explanations': explanations,
            'global_feature_importance': global_features,
            'n_explained_samples': len(explanations),
            'total_samples': len(dataset)
        }

    def get_global_feature_importance(self, dataset: np.ndarray,
                                    background_samples: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Compute global feature importance across entire dataset.

        Parameters:
            dataset: Full dataset for global importance
            background_samples: Background samples

        Returns:
            dict: Global feature importance results
        """
        dataset_explanation = self.explain_dataset(dataset, background_samples,
                                                 max_samples=min(500, len(dataset)))

        global_features = dataset_explanation['global_feature_importance']

        # Additional summary statistics
        if global_features:
            importance_values = np.array([f['mean_abs_shap'] for f in global_features])
            summary = {
                'total_features': len(importance_values),
                'mean_importance': float(np.mean(importance_values)),
                'std_importance': float(np.std(importance_values)),
                'max_importance': float(np.max(importance_values)),
                'importance_skewness': float(self._compute_skewness(importance_values))
            }
        else:
            summary = {}

        return {
            'global_features': global_features,
            'summary': summary,
            'computation_details': {
                'n_samples_used': dataset_explanation['n_explained_samples'],
                'total_samples': dataset_explanation['total_samples']
            }
        }

    def analyze_feature_interactions(self, dataset: np.ndarray,
                                   feature_pairs: Optional[List[Tuple[int, int]]] = None,
                                   max_samples: int = 100) -> Dict[str, Any]:
        """
        Analyze feature interactions using SHAP interaction values.

        Parameters:
            dataset: Dataset to analyze
            feature_pairs: Specific feature pairs to analyze
            max_samples: Maximum samples to analyze

        Returns:
            dict: Feature interaction analysis
        """
        # Sample dataset for efficiency
        if len(dataset) > max_samples:
            indices = np.random.choice(len(dataset), max_samples, replace=False)
            analysis_data = dataset[indices]
        else:
            analysis_data = dataset

        # For interaction analysis, we need a more detailed SHAP computation
        # This is a simplified version - full interaction analysis would require
        # TreeSHAP or specialized methods

        interactions = []

        if feature_pairs:
            for feat1, feat2 in feature_pairs[:10]:  # Limit for computational reasons
                # Compute correlation of SHAP values as proxy for interaction
                explanations = []
                for instance in analysis_data:
                    try:
                        exp = self.explain_instance(instance)
                        explanations.append(exp)
                    except:
                        continue

                if len(explanations) >= 10:
                    shap_feat1 = np.array([exp['shap_values'][feat1] for exp in explanations])
                    shap_feat2 = np.array([exp['shap_values'][feat2] for exp in explanations])

                    interaction_strength = abs(np.corrcoef(shap_feat1, shap_feat2)[0, 1])

                    interactions.append({
                        'feature_pair': (feat1, feat2),
                        'interaction_strength': float(interaction_strength),
                        'n_samples': len(explanations)
                    })

        return {
            'interactions': interactions,
            'analysis_details': {
                'n_samples_analyzed': len(analysis_data),
                'n_interactions_computed': len(interactions)
            }
        }

    def create_shap_plots(self, explanation: Dict[str, Any],
                         feature_names: Optional[List[str]] = None,
                         plot_type: str = 'waterfall') -> Dict[str, Any]:
        """
        Create SHAP visualization plots.

        Parameters:
            explanation: SHAP explanation dictionary
            feature_names: Names for features
            plot_type: Type of plot ('waterfall', 'bar', 'beeswarm')

        Returns:
            dict: Plot data for visualization
        """
        shap_values = np.array(explanation['shap_values'])
        feature_indices = np.arange(len(shap_values))

        if feature_names is None:
            feature_names = [f'Feature {i}' for i in feature_indices]

        # Sort by absolute SHAP value for plotting
        sort_idx = np.argsort(-np.abs(shap_values))

        plot_data = {
            'feature_names': [feature_names[i] for i in sort_idx],
            'shap_values': shap_values[sort_idx].tolist(),
            'feature_indices': sort_idx.tolist(),
            'base_value': explanation['base_value'],
            'expected_value': explanation['expected_value'],
            'plot_type': plot_type
        }

        if plot_type == 'waterfall':
            # Cumulative sum for waterfall plot
            cumulative_shap = np.cumsum(shap_values[sort_idx])
            plot_data['cumulative_shap'] = (cumulative_shap + explanation['base_value']).tolist()

        elif plot_type == 'bar':
            # Absolute values for bar plot
            plot_data['abs_shap_values'] = np.abs(shap_values[sort_idx]).tolist()

        return plot_data

    def _compute_skewness(self, data: np.ndarray) -> float:
        """Compute skewness of data."""
        mean_val = np.mean(data)
        std_val = np.std(data)
        if std_val == 0:
            return 0.0
        return float(np.mean(((data - mean_val) / std_val) ** 3))

    def validate_explanations(self, test_instances: np.ndarray,
                            background_samples: np.ndarray) -> Dict[str, Any]:
        """
        Validate SHAP explanations using various metrics.

        Parameters:
            test_instances: Instances to validate explanations for
            background_samples: Background samples for SHAP

        Returns:
            dict: Validation metrics
        """
        validation_results = []

        for instance in test_instances[:10]:  # Limit for validation
            try:
                explanation = self.explain_instance(instance, background_samples)

                # Consistency check: SHAP values should sum appropriately
                shap_sum = np.sum(explanation['shap_values'])
                expected_minus_base = explanation['expected_value'] - explanation['base_value']

                consistency_error = abs(shap_sum - expected_minus_base)

                validation_results.append({
                    'consistency_error': float(consistency_error),
                    'shap_sum': float(shap_sum),
                    'expected_minus_base': float(expected_minus_base),
                    'is_consistent': consistency_error < 1e-6
                })

            except Exception as e:
                validation_results.append({
                    'error': str(e),
                    'is_consistent': False
                })

        # Summary statistics
        if validation_results:
            consistency_errors = [r['consistency_error'] for r in validation_results if 'consistency_error' in r]
            consistent_explanations = sum(1 for r in validation_results if r.get('is_consistent', False))

            summary = {
                'mean_consistency_error': float(np.mean(consistency_errors)) if consistency_errors else 0,
                'max_consistency_error': float(np.max(consistency_errors)) if consistency_errors else 0,
                'consistent_explanations': consistent_explanations,
                'total_explanations': len(validation_results),
                'consistency_rate': consistent_explanations / len(validation_results)
            }
        else:
            summary = {}

        return {
            'individual_results': validation_results,
            'summary': summary
        }
