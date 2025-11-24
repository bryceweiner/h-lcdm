"""
Domain Adaptation
================

Survey-invariant training to ensure learned features work across
different cosmological surveys and systematic effects.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
from sklearn.metrics.pairwise import rbf_kernel


class DomainAdaptationTrainer:
    """
    Domain adaptation for survey-invariant feature learning.

    Implements techniques to ensure learned representations are
    invariant to survey-specific systematics while preserving
    cosmological signal.
    """

    def __init__(self, base_model: nn.Module,
                 n_surveys: int = 10,
                 latent_dim: int = 512,
                 adaptation_method: str = 'mmd'):
        """
        Initialize domain adaptation trainer.

        Parameters:
            base_model: Base SSL model to adapt
            n_surveys: Number of surveys to handle
            latent_dim: Dimension of latent space
            adaptation_method: 'mmd', 'adv', or 'both'
        """
        self.base_model = base_model
        self.n_surveys = n_surveys
        self.latent_dim = latent_dim
        self.adaptation_method = adaptation_method

        # Domain discriminators
        if adaptation_method in ['adv', 'both']:
            self.domain_discriminators = self._build_domain_discriminators()

        # MMD kernel parameters
        if adaptation_method in ['mmd', 'both']:
            self.mmd_kernel_bandwidth = 1.0
            self.mmd_lambda = 1.0

        # Survey embeddings (learnable)
        self.survey_embeddings = nn.Embedding(n_surveys, latent_dim)

        # Adaptation loss tracking
        self.adaptation_losses = []

    def _build_domain_discriminators(self) -> nn.ModuleDict:
        """Build domain discriminators for adversarial training."""
        discriminators = {}

        # One discriminator per modality
        for modality in self.base_model.encoders.keys():
            discriminator = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid()
            )
            discriminators[modality] = discriminator

        return nn.ModuleDict(discriminators)

    def compute_mmd_loss(self, source_features: torch.Tensor,
                        target_features: torch.Tensor) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy (MMD) loss.

        MMD measures distribution difference between domains.

        Parameters:
            source_features: Features from source domain
            target_features: Features from target domain

        Returns:
            torch.Tensor: MMD loss
        """
        # Convert to numpy for kernel computation
        source_np = source_features.detach().cpu().numpy()
        target_np = target_features.detach().cpu().numpy()

        # Compute RBF kernel matrices
        source_kernel = rbf_kernel(source_np, source_np, gamma=1.0/self.mmd_kernel_bandwidth)
        target_kernel = rbf_kernel(target_np, target_np, gamma=1.0/self.mmd_kernel_bandwidth)
        cross_kernel = rbf_kernel(source_np, target_np, gamma=1.0/self.mmd_kernel_bandwidth)

        # MMD statistic
        n_source = source_np.shape[0]
        n_target = target_np.shape[0]

        mmd = (np.sum(source_kernel) / (n_source ** 2) +
               np.sum(target_kernel) / (n_target ** 2) -
               2 * np.sum(cross_kernel) / (n_source * n_target))

        return torch.tensor(mmd, device=source_features.device, requires_grad=True)

    def compute_adversarial_loss(self, features: torch.Tensor,
                                domain_labels: torch.Tensor,
                                modality: str) -> torch.Tensor:
        """
        Compute adversarial domain adaptation loss.

        Parameters:
            features: Encoded features
            domain_labels: Domain/survey labels (0 or 1)
            modality: Modality name

        Returns:
            torch.Tensor: Adversarial loss
        """
        discriminator = self.domain_discriminators[modality]
        domain_predictions = discriminator(features).squeeze()

        # Binary cross-entropy loss
        loss = F.binary_cross_entropy(domain_predictions, domain_labels.float())

        return loss

    def adapt_domains(self, batch: Dict[str, torch.Tensor],
                     survey_ids: torch.Tensor,
                     adaptation_method: Optional[str] = None) -> Dict[str, torch.Tensor]:
        """
        Perform domain adaptation on batch.

        Parameters:
            batch: Batch of data from different surveys
            survey_ids: Survey identifiers for each sample
            adaptation_method: Override default method

        Returns:
            dict: Adaptation losses
        """
        method = adaptation_method or self.adaptation_method
        losses = {}

        # Get base model encodings
        with torch.no_grad():  # Don't update base model during adaptation
            encodings = self.base_model.encode(batch)

        total_loss = 0

        # Apply adaptation per modality
        for modality, features in encodings.items():
            if method in ['mmd', 'both']:
                # MMD adaptation: minimize distance between survey distributions
                mmd_loss = self._compute_batch_mmd_loss(features, survey_ids)
                losses[f'{modality}_mmd'] = mmd_loss
                total_loss += self.mmd_lambda * mmd_loss

            if method in ['adv', 'both']:
                # Adversarial adaptation: confuse domain discriminator
                adv_loss = self._compute_adversarial_loss(features, survey_ids, modality)
                losses[f'{modality}_adv'] = adv_loss
                total_loss += adv_loss

        losses['total_adaptation'] = total_loss
        self.adaptation_losses.append(losses)

        return losses

    def _compute_batch_mmd_loss(self, features: torch.Tensor,
                               survey_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss across all survey pairs in batch.

        Parameters:
            features: Encoded features
            survey_ids: Survey IDs

        Returns:
            torch.Tensor: Average MMD loss
        """
        unique_surveys = torch.unique(survey_ids)
        mmd_losses = []

        for i, survey1 in enumerate(unique_surveys):
            for survey2 in unique_surveys[i+1:]:
                # Get features for each survey
                mask1 = survey_ids == survey1
                mask2 = survey_ids == survey2

                if mask1.sum() > 1 and mask2.sum() > 1:  # Need at least 2 samples
                    features1 = features[mask1]
                    features2 = features[mask2]

                    mmd_loss = self.compute_mmd_loss(features1, features2)
                    mmd_losses.append(mmd_loss)

        if mmd_losses:
            return torch.stack(mmd_losses).mean()
        else:
            return torch.tensor(0.0, device=features.device)

    def _compute_adversarial_loss(self, features: torch.Tensor,
                                 survey_ids: torch.Tensor,
                                 modality: str) -> torch.Tensor:
        """
        Compute adversarial loss for domain confusion.

        Parameters:
            features: Encoded features
            survey_ids: Survey IDs
            modality: Modality name

        Returns:
            torch.Tensor: Adversarial loss
        """
        # Create domain labels (randomly assign some as "source" vs "target")
        # In practice, this would be based on actual survey groupings
        n_samples = features.shape[0]
        domain_labels = torch.randint(0, 2, (n_samples,), device=features.device)

        return self.compute_adversarial_loss(features, domain_labels, modality)

    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """
        Get adaptation training metrics.

        Returns:
            dict: Adaptation metrics
        """
        if not self.adaptation_losses:
            return {}

        # Average losses over training
        avg_losses = {}
        for key in self.adaptation_losses[0].keys():
            values = [loss_dict.get(key, 0) for loss_dict in self.adaptation_losses]
            avg_losses[f'avg_{key}'] = np.mean(values)

        # Convergence metrics
        recent_losses = self.adaptation_losses[-10:]  # Last 10 batches
        recent_avg = np.mean([loss_dict.get('total_adaptation', 0) for loss_dict in recent_losses])

        return {
            'average_losses': avg_losses,
            'recent_adaptation_loss': recent_avg,
            'convergence_indicator': recent_avg < 0.1,  # Arbitrary threshold
            'total_adaptation_steps': len(self.adaptation_losses)
        }

    def save_adaptation_state(self, path: str):
        """Save adaptation state."""
        state = {
            'survey_embeddings': self.survey_embeddings.state_dict(),
            'adaptation_losses': self.adaptation_losses,
            'n_surveys': self.n_surveys,
            'latent_dim': self.latent_dim,
            'adaptation_method': self.adaptation_method
        }

        if hasattr(self, 'domain_discriminators'):
            state['domain_discriminators'] = self.domain_discriminators.state_dict()

        torch.save(state, path)

    def load_adaptation_state(self, path: str):
        """Load adaptation state."""
        state = torch.load(path)

        self.survey_embeddings.load_state_dict(state['survey_embeddings'])
        self.adaptation_losses = state['adaptation_losses']

        if 'domain_discriminators' in state and hasattr(self, 'domain_discriminators'):
            self.domain_discriminators.load_state_dict(state['domain_discriminators'])


class SurveyInvariantValidator:
    """
    Validate that learned features are survey-invariant.
    """

    def __init__(self, latent_dim: int = 512):
        self.latent_dim = latent_dim

    def compute_survey_invariance_metrics(self,
                                        features_by_survey: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Compute metrics quantifying survey invariance.

        Parameters:
            features_by_survey: Features grouped by survey

        Returns:
            dict: Invariance metrics
        """
        metrics = {}

        if len(features_by_survey) < 2:
            return {'insufficient_surveys': True}

        # Distribution distances between surveys
        survey_names = list(features_by_survey.keys())
        distribution_distances = []

        for i, survey1 in enumerate(survey_names):
            for survey2 in survey_names[i+1:]:
                features1 = features_by_survey[survey1]
                features2 = features_by_survey[survey2]

                # Compute various distance metrics
                # 1. Mean difference
                mean_diff = torch.mean(torch.abs(
                    torch.mean(features1, dim=0) - torch.mean(features2, dim=0)
                ))

                # 2. Distribution KL divergence approximation
                # (Simplified - would need more sophisticated implementation)

                distribution_distances.append({
                    'survey_pair': f'{survey1}_{survey2}',
                    'mean_difference': mean_diff.item()
                })

        # Overall invariance score (lower is better)
        mean_differences = [d['mean_difference'] for d in distribution_distances]
        metrics['average_distribution_distance'] = np.mean(mean_differences)
        metrics['max_distribution_distance'] = np.max(mean_differences)
        metrics['survey_invariance_score'] = 1.0 / (1.0 + np.mean(mean_differences))

        return metrics
