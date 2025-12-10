"""
Modality Attribution
====================

Post-hoc modality attribution for multimodal anomaly detection.
Implements:
- Integrated gradients from latent back to input modalities
- Attention-based attribution using fusion attention weights
- Per-modality anomaly contributions before fusion
- Single-modality anomaly detection runs
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
import warnings


class ModalityAttributor:
    """
    Compute modality-specific contributions to anomaly detections.
    
    Provides three attribution methods:
    1. Integrated gradients: Gradient-based attribution from latent to input
    2. Attention weights: Use fusion attention weights as attribution
    3. Single-modality residuals: Compare single-modality vs full anomaly scores
    """

    def __init__(self, ssl_learner: Any,
                 fusion_module: Optional[Any] = None,
                 anomaly_detector: Optional[Any] = None,
                 device: str = 'cpu'):
        """
        Initialize modality attributor.

        Parameters:
            ssl_learner: Trained SSL encoder with modality-specific encoders
            fusion_module: Multimodal fusion module (optional, for attention-based attribution)
            anomaly_detector: Anomaly detection model (optional, for single-modality runs)
            device: Device to run computations on
        """
        self.ssl_learner = ssl_learner
        self.fusion_module = fusion_module
        self.anomaly_detector = anomaly_detector
        self.device = device
        self.logger = logging.getLogger(__name__)

    def compute_modality_contributions(self,
                                     sample_data: Dict[str, torch.Tensor],
                                     anomaly_score_full: float,
                                     n_steps: int = 50) -> Dict[str, Any]:
        """
        Compute modality contributions using multiple methods.

        Parameters:
            sample_data: Input data dictionary {modality: tensor}
            anomaly_score_full: Full anomaly score for this sample
            n_steps: Number of steps for integrated gradients

        Returns:
            dict: Modality attribution results
        """
        results = {}

        # Method 1: Single-modality anomaly scores
        single_modality_scores = self._compute_single_modality_scores(sample_data)
        results['single_modality_scores'] = single_modality_scores

        # Method 2: Modality residuals (contribution before fusion)
        modality_residuals = self._compute_modality_residuals(
            sample_data, anomaly_score_full
        )
        results['modality_residuals'] = modality_residuals

        # Method 3: Attention-based attribution (if fusion module available)
        if self.fusion_module is not None:
            attention_attribution = self._compute_attention_attribution(sample_data)
            results['attention_attribution'] = attention_attribution

        # Method 4: Integrated gradients (if gradients available)
        try:
            integrated_grads = self._compute_integrated_gradients(
                sample_data, n_steps=n_steps
            )
            results['integrated_gradients'] = integrated_grads
        except Exception as e:
            self.logger.warning(f"Integrated gradients failed: {e}")
            results['integrated_gradients'] = {'error': str(e)}

        # Summary: Normalized contributions
        results['summary'] = self._summarize_contributions(results)

        return results

    def _compute_single_modality_scores(self,
                                       sample_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Run anomaly detection on each modality independently.

        Parameters:
            sample_data: Input data dictionary

        Returns:
            dict: Anomaly scores for each modality
        """
        if self.anomaly_detector is None:
            return {}

        single_scores = {}
        self.ssl_learner.eval()

        with torch.no_grad():
            for modality, data_tensor in sample_data.items():
                if modality not in self.ssl_learner.encoders:
                    continue

                try:
                    # Encode single modality
                    encoder = self.ssl_learner.encoders[modality]
                    latent_single = encoder(data_tensor)  # (batch, latent_dim)

                    # Convert to numpy for anomaly detector
                    latent_np = latent_single.cpu().numpy()

                    # Get anomaly score
                    if hasattr(self.anomaly_detector, 'predict'):
                        predictions = self.anomaly_detector.predict(latent_np)
                        if isinstance(predictions, dict):
                            score = predictions.get('ensemble_scores', [0.0])
                            if isinstance(score, np.ndarray):
                                score = score[0] if len(score) > 0 else 0.0
                        else:
                            score = float(predictions[0]) if len(predictions) > 0 else 0.0
                    else:
                        score = 0.0

                    single_scores[modality] = float(score)

                except Exception as e:
                    self.logger.warning(f"Failed to compute single-modality score for {modality}: {e}")
                    single_scores[modality] = 0.0

        return single_scores

    def _compute_modality_residuals(self,
                                   sample_data: Dict[str, torch.Tensor],
                                   anomaly_score_full: float) -> Dict[str, float]:
        """
        Compute modality-specific anomaly contributions before fusion.

        Parameters:
            sample_data: Input data dictionary
            anomaly_score_full: Full anomaly score

        Returns:
            dict: Residual contributions per modality
        """
        if self.anomaly_detector is None:
            return {}

        residuals = {}
        self.ssl_learner.eval()

        with torch.no_grad():
            # Get full encoding (all modalities)
            full_encodings = self.ssl_learner.encode(sample_data)
            if not full_encodings:
                return {}

            # Encode each modality separately
            for modality, data_tensor in sample_data.items():
                if modality not in self.ssl_learner.encoders:
                    continue

                try:
                    # Encode single modality
                    encoder = self.ssl_learner.encoders[modality]
                    latent_single = encoder(data_tensor)

                    # Convert to numpy
                    latent_np = latent_single.cpu().numpy()

                    # Get anomaly score for single modality
                    if hasattr(self.anomaly_detector, 'predict'):
                        predictions = self.anomaly_detector.predict(latent_np)
                        if isinstance(predictions, dict):
                            score_single = predictions.get('ensemble_scores', [0.0])
                            if isinstance(score_single, np.ndarray):
                                score_single = score_single[0] if len(score_single) > 0 else 0.0
                        else:
                            score_single = float(predictions[0]) if len(predictions) > 0 else 0.0
                    else:
                        score_single = 0.0

                    # Compute residual contribution
                    # Ratio of single-modality score to full score
                    if abs(anomaly_score_full) > 1e-10:
                        contribution_ratio = float(score_single) / float(anomaly_score_full)
                    else:
                        contribution_ratio = 0.0

                    residuals[modality] = {
                        'single_score': float(score_single),
                        'contribution_ratio': contribution_ratio,
                        'residual': float(score_single) - float(anomaly_score_full)
                    }

                except Exception as e:
                    self.logger.warning(f"Failed to compute residual for {modality}: {e}")
                    residuals[modality] = {
                        'single_score': 0.0,
                        'contribution_ratio': 0.0,
                        'residual': 0.0
                    }

        return residuals

    def _compute_attention_attribution(self,
                                       sample_data: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute modality attribution using fusion attention weights.

        Parameters:
            sample_data: Input data dictionary

        Returns:
            dict: Attention-based attribution scores
        """
        if self.fusion_module is None:
            return {}

        self.ssl_learner.eval()
        self.fusion_module.eval()

        with torch.no_grad():
            # Encode all modalities
            modality_encodings = self.ssl_learner.encode(sample_data)

            if not modality_encodings:
                return {}

            # Get attention weights from fusion
            fusion_result = self.fusion_module(
                modality_encodings,
                return_attention_weights=True
            )

            attention_weights = {}
            if 'self_attention_weights' in fusion_result:
                # Extract attention weights (shape: batch, n_modalities, n_modalities)
                attn = fusion_result['self_attention_weights']
                if isinstance(attn, torch.Tensor):
                    # Average over attention heads and batch dimension
                    if attn.dim() == 4:  # (batch, heads, n_mod, n_mod)
                        attn = attn.mean(dim=1)  # Average over heads
                    if attn.dim() == 3:  # (batch, n_mod, n_mod)
                        attn = attn.mean(dim=0)  # Average over batch

                    # Sum over query positions to get key importance
                    modality_importance = attn.sum(dim=0).cpu().numpy()

                    # Map to modality names
                    modality_list = list(modality_encodings.keys())
                    for idx, modality in enumerate(modality_list):
                        if idx < len(modality_importance):
                            attention_weights[modality] = float(modality_importance[idx])

            # Also use learned modality weights if available
            if 'modality_weights' in fusion_result:
                weights = fusion_result['modality_weights']
                if isinstance(weights, torch.Tensor):
                    weights_np = weights.cpu().numpy()
                    modality_list = list(modality_encodings.keys())
                    for idx, modality in enumerate(modality_list):
                        if idx < len(weights_np):
                            if modality not in attention_weights:
                                attention_weights[modality] = 0.0
                            attention_weights[modality] += float(weights_np[idx])

        return attention_weights

    def _compute_integrated_gradients(self,
                                     sample_data: Dict[str, torch.Tensor],
                                     n_steps: int = 50) -> Dict[str, np.ndarray]:
        """
        Compute integrated gradients from latent representation to input modalities.

        Parameters:
            sample_data: Input data dictionary
            n_steps: Number of integration steps

        Returns:
            dict: Integrated gradients per modality
        """
        self.ssl_learner.eval()

        # Create baseline (zeros or mean)
        baseline_data = {}
        for modality, data_tensor in sample_data.items():
            # Use zeros as baseline
            baseline_data[modality] = torch.zeros_like(data_tensor)

        integrated_grads = {}

        # Enable gradients for input
        sample_data_grad = {}
        for modality, data_tensor in sample_data.items():
            sample_data_grad[modality] = data_tensor.clone().requires_grad_(True)

        # Forward pass to get latent representation
        encodings = self.ssl_learner.encode_with_grad(sample_data_grad)

        # Create a simple function that maps encodings to anomaly score
        # We'll use the norm of the encoding as a proxy for anomaly score
        # (in practice, this would go through the anomaly detector)
        def anomaly_proxy(encodings_dict: Dict[str, torch.Tensor]) -> torch.Tensor:
            # Concatenate or average encodings
            if len(encodings_dict) == 0:
                return torch.tensor(0.0, device=self.device)
            encodings_list = list(encodings_dict.values())
            combined = torch.stack(encodings_list, dim=0).mean(dim=0)
            # Use L2 norm as proxy
            return combined.norm(dim=-1).mean()

        # Compute integrated gradients for each modality
        for modality, data_tensor in sample_data_grad.items():
            try:
                # Create path from baseline to input
                alphas = torch.linspace(0, 1, n_steps, device=self.device)

                gradients_sum = None

                for alpha in alphas:
                    # Interpolate between baseline and input
                    interpolated = {
                        mod: baseline_data[mod] * (1 - alpha) + sample_data_grad[mod] * alpha
                        if mod == modality else sample_data_grad[mod]
                        for mod in sample_data_grad.keys()
                    }

                    # Forward pass
                    encodings_interp = self.ssl_learner.encode_with_grad(interpolated)
                    score = anomaly_proxy(encodings_interp)

                    # Backward pass
                    if data_tensor.grad is not None:
                        data_tensor.grad.zero_()

                    score.backward(retain_graph=True)

                    # Accumulate gradients
                    if gradients_sum is None:
                        gradients_sum = data_tensor.grad.clone()
                    else:
                        gradients_sum += data_tensor.grad

                # Average and multiply by (input - baseline)
                integrated_grad = (gradients_sum / n_steps) * (data_tensor - baseline_data[modality])
                integrated_grads[modality] = integrated_grad.detach().cpu().numpy()

            except Exception as e:
                self.logger.warning(f"Integrated gradients failed for {modality}: {e}")
                integrated_grads[modality] = np.zeros_like(data_tensor.cpu().numpy())

        return integrated_grads

    def _summarize_contributions(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """
        Summarize modality contributions across all methods.

        Parameters:
            results: Attribution results dictionary

        Returns:
            dict: Summary statistics
        """
        summary = {}

        # Single modality scores
        if 'single_modality_scores' in results:
            single_scores = results['single_modality_scores']
            if single_scores:
                scores_array = np.array(list(single_scores.values()))
                summary['single_modality'] = {
                    'mean': float(np.mean(scores_array)),
                    'std': float(np.std(scores_array)),
                    'max_modality': max(single_scores.items(), key=lambda x: x[1])[0] if single_scores else None,
                    'min_modality': min(single_scores.items(), key=lambda x: x[1])[0] if single_scores else None
                }

        # Modality residuals
        if 'modality_residuals' in results:
            residuals = results['modality_residuals']
            if residuals:
                contribution_ratios = [r.get('contribution_ratio', 0.0) for r in residuals.values()]
                summary['residuals'] = {
                    'mean_contribution_ratio': float(np.mean(contribution_ratios)),
                    'max_contributor': max(residuals.items(), key=lambda x: x[1].get('contribution_ratio', 0.0))[0] if residuals else None
                }

        # Attention attribution
        if 'attention_attribution' in results and results['attention_attribution']:
            attn = results['attention_attribution']
            if attn:
                attn_values = list(attn.values())
                summary['attention'] = {
                    'mean': float(np.mean(attn_values)),
                    'max_modality': max(attn.items(), key=lambda x: x[1])[0] if attn else None
                }

        return summary

    def analyze_batch(self,
                     batch_data: Dict[str, torch.Tensor],
                     anomaly_scores: np.ndarray,
                     n_samples: Optional[int] = None) -> Dict[str, Any]:
        """
        Analyze modality contributions for a batch of samples.

        Parameters:
            batch_data: Batch data dictionary {modality: tensor}
            anomaly_scores: Anomaly scores for batch (n_samples,)
            n_samples: Number of samples to analyze (None = all)

        Returns:
            dict: Batch attribution results
        """
        n_samples = n_samples or len(anomaly_scores)
        n_samples = min(n_samples, len(anomaly_scores))

        batch_results = []
        for i in range(n_samples):
            # Extract single sample
            sample_data = {}
            for modality, data_tensor in batch_data.items():
                if data_tensor.shape[0] > i:
                    sample_data[modality] = data_tensor[i:i+1]  # Keep batch dimension
                else:
                    # Use last sample if not enough samples
                    sample_data[modality] = data_tensor[-1:]

            anomaly_score = float(anomaly_scores[i])

            # Compute contributions
            contributions = self.compute_modality_contributions(
                sample_data, anomaly_score
            )
            contributions['sample_index'] = i
            contributions['anomaly_score'] = anomaly_score
            batch_results.append(contributions)

        # Aggregate results
        aggregated = self._aggregate_batch_results(batch_results)

        return {
            'individual_results': batch_results,
            'aggregated': aggregated,
            'n_samples': n_samples
        }

    def _aggregate_batch_results(self, batch_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Aggregate attribution results across batch.

        Parameters:
            batch_results: List of individual attribution results

        Returns:
            dict: Aggregated statistics
        """
        if not batch_results:
            return {}

        aggregated = {}

        # Aggregate single modality scores
        all_single_scores = {}
        for result in batch_results:
            if 'single_modality_scores' in result:
                for modality, score in result['single_modality_scores'].items():
                    if modality not in all_single_scores:
                        all_single_scores[modality] = []
                    all_single_scores[modality].append(score)

        if all_single_scores:
            aggregated['single_modality_means'] = {
                mod: float(np.mean(scores))
                for mod, scores in all_single_scores.items()
            }
            aggregated['single_modality_stds'] = {
                mod: float(np.std(scores))
                for mod, scores in all_single_scores.items()
            }

        # Aggregate residuals
        all_contribution_ratios = {}
        for result in batch_results:
            if 'modality_residuals' in result:
                for modality, residual_data in result['modality_residuals'].items():
                    if modality not in all_contribution_ratios:
                        all_contribution_ratios[modality] = []
                    all_contribution_ratios[modality].append(
                        residual_data.get('contribution_ratio', 0.0)
                    )

        if all_contribution_ratios:
            aggregated['mean_contribution_ratios'] = {
                mod: float(np.mean(ratios))
                for mod, ratios in all_contribution_ratios.items()
            }

        return aggregated

