"""
Domain Adaptation
================

Survey-invariant training to ensure learned features work across
different cosmological surveys and systematic effects.
Implements rigorous statistical alignment methods.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Any, Optional, Tuple
import numpy as np
import warnings

# Suppress sklearn deprecation warnings for internal API changes
warnings.filterwarnings('ignore', category=FutureWarning, module='sklearn.utils.deprecation')


class MultiscaleGaussianKernel(nn.Module):
    """
    Multiscale Gaussian Kernel for MMD.
    
    Captures distribution differences at multiple scales, essential for
    high-dimensional cosmological data where single-scale RBF fails.
    """
    def __init__(self, bandwidths: List[float]):
        super().__init__()
        self.bandwidths = bandwidths

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Compute squared Euclidean distance
        xx = torch.matmul(x, x.t())
        yy = torch.matmul(y, y.t())
        xy = torch.matmul(x, y.t())
        
        # Distance matrix: ||x-y||^2 = ||x||^2 + ||y||^2 - 2<x,y>
        dist = xx.unsqueeze(1) + yy.unsqueeze(0) - 2 * xy
        
        # Sum of Gaussians at different scales
        val = 0
        for bandwidth in self.bandwidths:
            val += torch.exp(-0.5 * dist / (bandwidth**2))
            
        return val


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
                 adaptation_method: str = 'mmd',
                 device: Optional[torch.device] = None):
        """
        Initialize domain adaptation trainer.
        
        Parameters:
            base_model: Base SSL model to adapt
            n_surveys: Number of surveys to handle
            latent_dim: Dimension of latent space
            adaptation_method: 'mmd', 'adv', or 'both'
            device: Device to run on (defaults to base_model device)
        """
        self.base_model = base_model
        
        # Determine device from base_model if not provided
        if device is None:
            # Get device from base_model's encoders
            if hasattr(base_model, 'encoders') and base_model.encoders:
                self.device = next(iter(base_model.encoders.values())).parameters().__next__().device
            elif hasattr(base_model, 'device'):
                self.device = base_model.device
            else:
                self.device = torch.device('cpu')
        else:
            self.device = device
        
        self.n_surveys = n_surveys
        self.latent_dim = latent_dim
        self.adaptation_method = adaptation_method

        # Domain discriminators
        if adaptation_method in ['adv', 'both']:
            self.domain_discriminators = self._build_domain_discriminators()
            self.domain_discriminators = self.domain_discriminators.to(self.device)

        # MMD kernel parameters optimized for cosmological data scales
        if adaptation_method in ['mmd', 'both']:
            # Multiscale bandwidths centered around latent_dim
            base_scale = np.sqrt(latent_dim)
            bandwidths = [base_scale * f for f in [0.1, 0.5, 1.0, 2.0, 10.0]]
            self.kernel = MultiscaleGaussianKernel(bandwidths).to(self.device)
            self.mmd_lambda = 1.0

        # Survey embeddings (learnable)
        self.survey_embeddings = nn.Embedding(n_surveys, latent_dim).to(self.device)

        # Adaptation loss tracking
        self.adaptation_losses = []

    def _build_domain_discriminators(self) -> nn.ModuleDict:
        """Build domain discriminators for adversarial training."""
        discriminators = {}

        # One discriminator per modality
        for modality in self.base_model.encoders.keys():
            discriminator = nn.Sequential(
                nn.Linear(self.latent_dim, 128),
                nn.LayerNorm(128),
                nn.ReLU(),
                nn.Linear(128, 64),
                nn.ReLU(),
                nn.Linear(64, self.n_surveys)  # Predict survey ID, not just binary
            )
            discriminators[modality] = discriminator

        return nn.ModuleDict(discriminators)

    def compute_mmd_loss(self, source_features: torch.Tensor,
                        target_features: torch.Tensor) -> torch.Tensor:
        """
        Compute Maximum Mean Discrepancy (MMD) loss using Multiscale Kernel.
        
        Parameters:
            source_features: Features from source domain
            target_features: Features from target domain
            
        Returns:
            torch.Tensor: MMD loss
        """
        if len(source_features) == 0 or len(target_features) == 0:
             return torch.tensor(0.0, device=self.device)

        # Compute kernel matrices
        xx = self.kernel(source_features, source_features)
        yy = self.kernel(target_features, target_features)
        xy = self.kernel(source_features, target_features)
        
        # Unbiased MMD estimate
        # E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        m = source_features.shape[0]
        n = target_features.shape[0]
        
        # Exclude diagonal for unbiased estimator of E[k(x,x')]
        mmd = (xx.sum() - torch.trace(xx)) / (m * (m - 1)) + \
              (yy.sum() - torch.trace(yy)) / (n * (n - 1)) - \
              2 * xy.sum() / (m * n)
              
        return mmd

    def compute_adversarial_loss(self, features: torch.Tensor,
                                survey_ids: torch.Tensor,
                                modality: str) -> torch.Tensor:
        """
        Compute adversarial domain adaptation loss.
        
        Parameters:
            features: Encoded features
            survey_ids: Actual survey IDs
            modality: Modality name
            
        Returns:
            torch.Tensor: Adversarial loss
        """
        discriminator = self.domain_discriminators[modality]
        survey_predictions = discriminator(features)
        
        # Cross-entropy loss: Can the discriminator guess the survey?
        # We want to MAXIMIZE this entropy (confuse discriminator),
        # but standard GRL (Gradient Reversal Layer) isn't implemented here.
        # Instead, we return the loss that the DISCRIMINATOR minimizes.
        # The Encoder should maximize this (or minimize negative).
        
        # NOTE: In a proper GAN setup, we alternate. Here we return the loss
        # for the discriminator. The encoder needs to minimize -loss (or maximize entropy).
        
        loss = F.cross_entropy(survey_predictions, survey_ids.long())
        
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
        # NOTE: We need gradients to flow back to encoder for adaptation!
        # Removing torch.no_grad() if we want to train the encoder to be invariant.
        # If we only train the adapter, then no_grad is fine, but that defeats the purpose.
        # Assuming we are in a training loop where optimizer steps on encoder.
        
        encodings = self.base_model.encode(batch)
        
        total_loss = 0

        # Apply adaptation per modality
        for modality, features in encodings.items():
            # Ensure features require grad
            if not features.requires_grad:
                features.requires_grad_(True)

            if method in ['mmd', 'both']:
                # MMD adaptation: minimize distance between survey distributions
                mmd_loss = self._compute_batch_mmd_loss(features, survey_ids)
                losses[f'{modality}_mmd'] = mmd_loss
                total_loss += self.mmd_lambda * mmd_loss

            if method in ['adv', 'both']:
                # Adversarial adaptation
                # If we are updating the ENCODER, we want to MAXIMIZE the discriminator error
                # (make features indistinguishable).
                # Simple implementation: minimize -entropy (maximize uniform distribution)
                
                discriminator = self.domain_discriminators[modality]
                preds = discriminator(features)
                
                # Target uniform distribution (confusion)
                n_classes = preds.shape[1]
                uniform_target = torch.full_like(preds, 1.0/n_classes)
                
                # KL Divergence from Uniform (minimize this to confuse)
                confusion_loss = F.kl_div(F.log_softmax(preds, dim=1), uniform_target, reduction='batchmean')
                
                losses[f'{modality}_confusion'] = confusion_loss
                total_loss += confusion_loss

        losses['total_adaptation'] = total_loss
        self.adaptation_losses.append(losses)

        return losses

    def _compute_batch_mmd_loss(self, features: torch.Tensor,
                               survey_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss across all survey pairs in batch.
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
            return torch.tensor(0.0, device=features.device, requires_grad=True)
            
    def get_adaptation_metrics(self) -> Dict[str, Any]:
        """Get adaptation training metrics."""
        if not self.adaptation_losses:
            return {}

        # Average losses over training
        avg_losses = {}
        # Safely handle tensors in dict
        loss_keys = self.adaptation_losses[0].keys()
        
        for key in loss_keys:
            values = []
            for loss_dict in self.adaptation_losses:
                val = loss_dict.get(key, 0)
                if isinstance(val, torch.Tensor):
                    val = val.item()
                values.append(val)
            avg_losses[f'avg_{key}'] = np.mean(values)

        return {
            'average_losses': avg_losses,
            'total_adaptation_steps': len(self.adaptation_losses)
        }

    def save_adaptation_state(self, path: str):
        """Save adaptation state."""
        state = {
            'survey_embeddings': self.survey_embeddings.state_dict(),
            # Don't save losses if they contain tensors
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

                distribution_distances.append(mean_diff.item())

        if not distribution_distances:
            return {}
            
        # Overall invariance score (lower is better)
        metrics['average_distribution_distance'] = np.mean(distribution_distances)
        metrics['max_distribution_distance'] = np.max(distribution_distances)
        metrics['survey_invariance_score'] = 1.0 / (1.0 + np.mean(distribution_distances))

        return metrics
