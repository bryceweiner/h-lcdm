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
        # Log input shapes for debugging
        import logging
        logger = logging.getLogger(__name__)
        logger.debug(f"Kernel forward: x.shape={x.shape}, y.shape={y.shape}, x.dim()={x.dim()}, y.dim()={y.dim()}")
        
        # Ensure 2D tensors
        if x.dim() > 2:
            logger.warning(f"Kernel received 3D tensor x: {x.shape}, flattening")
            x = x.flatten(start_dim=1)
        if y.dim() > 2:
            logger.warning(f"Kernel received 3D tensor y: {y.shape}, flattening")
            y = y.flatten(start_dim=1)
        
        # Ensure exactly 2D
        if x.dim() != 2:
            logger.error(f"After flattening, x is still not 2D: {x.shape}")
            x = x.view(x.shape[0], -1)
        if y.dim() != 2:
            logger.error(f"After flattening, y is still not 2D: {y.shape}")
            y = y.view(y.shape[0], -1)
        
        # Ensure matching feature dimensions
        if x.shape[1] != y.shape[1]:
            logger.warning(f"Feature dimension mismatch: x.shape[1]={x.shape[1]}, y.shape[1]={y.shape[1]}, truncating to min")
            min_dim = min(x.shape[1], y.shape[1])
            x = x[:, :min_dim]
            y = y[:, :min_dim]
        
        logger.debug(f"Kernel forward after normalization: x.shape={x.shape}, y.shape={y.shape}")
        
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
        
        # Infer latent_dim from base_model if not provided
        if latent_dim is None or latent_dim <= 0:
            if hasattr(base_model, 'latent_dim'):
                self.latent_dim = base_model.latent_dim
            else:
                self.latent_dim = 512  # Default fallback
        else:
            self.latent_dim = latent_dim
        
        self.n_surveys = n_surveys
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
        # Cache a small sample per (modality_type, survey_id) to enable cross-survey MMD
        # Key format: "modality_type_survey_id" (e.g., "cmb_tt_0" for CMB TT from survey 0)
        self.survey_feature_cache: Dict[str, torch.Tensor] = {}

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
            source_features: Features from source domain (must be 2D: batch, features)
            target_features: Features from target domain (must be 2D: batch, features)
            
        Returns:
            torch.Tensor: MMD loss
        """
        import logging
        logger = logging.getLogger(__name__)
        
        logger.debug(f"compute_mmd_loss ENTRY: source.shape={source_features.shape}, target.shape={target_features.shape}")
        logger.debug(f"compute_mmd_loss ENTRY: source.dim()={source_features.dim()}, target.dim()={target_features.dim()}")
        
        if len(source_features) == 0 or len(target_features) == 0:
             return torch.tensor(0.0, device=self.device)

        # Ensure 2D tensors: flatten if needed
        if source_features.dim() > 2:
            logger.error(f"compute_mmd_loss: source_features is {source_features.dim()}D: {source_features.shape}, flattening")
            source_features = source_features.flatten(start_dim=1)
        if target_features.dim() > 2:
            logger.error(f"compute_mmd_loss: target_features is {target_features.dim()}D: {target_features.shape}, flattening")
            target_features = target_features.flatten(start_dim=1)
        
        # Force 2D if still not 2D
        if source_features.dim() != 2:
            logger.error(f"compute_mmd_loss: source_features still not 2D after flatten: {source_features.shape}, forcing view")
            source_features = source_features.view(source_features.shape[0], -1)
        if target_features.dim() != 2:
            logger.error(f"compute_mmd_loss: target_features still not 2D after flatten: {target_features.shape}, forcing view")
            target_features = target_features.view(target_features.shape[0], -1)
        
        logger.debug(f"compute_mmd_loss AFTER FLATTEN: source.shape={source_features.shape}, target.shape={target_features.shape}")
        
        # Ensure consistent feature dimensions: pad/truncate to latent_dim
        source_feat_dim = source_features.shape[1]
        target_feat_dim = target_features.shape[1]
        
        logger.debug(f"compute_mmd_loss: source_feat_dim={source_feat_dim}, target_feat_dim={target_feat_dim}, latent_dim={self.latent_dim}")
        
        if source_feat_dim < self.latent_dim:
            pad = torch.zeros((source_features.shape[0], self.latent_dim - source_feat_dim), 
                             device=source_features.device, dtype=source_features.dtype)
            source_features = torch.cat([source_features, pad], dim=1)
        elif source_feat_dim > self.latent_dim:
            source_features = source_features[:, :self.latent_dim]
        
        if target_feat_dim < self.latent_dim:
            pad = torch.zeros((target_features.shape[0], self.latent_dim - target_feat_dim), 
                             device=target_features.device, dtype=target_features.dtype)
            target_features = torch.cat([target_features, pad], dim=1)
        elif target_feat_dim > self.latent_dim:
            target_features = target_features[:, :self.latent_dim]

        logger.debug(f"compute_mmd_loss BEFORE KERNEL: source.shape={source_features.shape}, target.shape={target_features.shape}")
        logger.debug(f"compute_mmd_loss BEFORE KERNEL: source.dim()={source_features.dim()}, target.dim()={target_features.dim()}")

        # Compute kernel matrices
        try:
        xx = self.kernel(source_features, source_features)
        yy = self.kernel(target_features, target_features)
        xy = self.kernel(source_features, target_features)
        except Exception as e:
            logger.error(f"compute_mmd_loss KERNEL ERROR: {e}")
            logger.error(f"  source_features.shape={source_features.shape}, dim={source_features.dim()}")
            logger.error(f"  target_features.shape={target_features.shape}, dim={target_features.dim()}")
            raise
        
        # Unbiased MMD estimate
        # E[k(x,x')] + E[k(y,y')] - 2E[k(x,y)]
        m = source_features.shape[0]
        n = target_features.shape[0]
        
        # Handle edge cases
        if m < 2 or n < 2:
            # Fallback for small sample sizes
            if m == 1 and n == 1:
                return torch.tensor(0.0, device=source_features.device, requires_grad=True)
            elif m == 1:
                return (yy.sum() - torch.trace(yy)) / (n * (n - 1))
            elif n == 1:
                return (xx.sum() - torch.trace(xx)) / (m * (m - 1))
        
        # Ensure kernel matrices are 2D for trace operation
        if xx.dim() > 2:
            xx = xx.squeeze()
        if yy.dim() > 2:
            yy = yy.squeeze()
        
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
        
        if hasattr(self.base_model, 'encode_with_grad'):
            encodings = self.base_model.encode_with_grad(batch)
        else:
        encodings = self.base_model.encode(batch)
        
        total_loss = 0

        # Apply adaptation per modality
        import logging
        logger = logging.getLogger(__name__)
        
        for modality, features in encodings.items():
            logger.debug(f"Processing modality {modality}: features.shape={features.shape}, features.dim()={features.dim()}")
            
            # Ensure features require grad
            if not features.requires_grad:
                features.requires_grad_(True)

            # Flatten to 2D and align dims across surveys to latent_dim
            if features.dim() > 2:
                logger.warning(f"Modality {modality}: features is {features.dim()}D with shape {features.shape}, flattening")
                features = features.flatten(start_dim=1)
            elif features.dim() == 1:
                logger.warning(f"Modality {modality}: features is 1D with shape {features.shape}, unsqueezing")
                features = features.unsqueeze(0)
            
            # Ensure exactly 2D
            if features.dim() != 2:
                logger.error(f"Modality {modality}: After normalization, features is still not 2D: {features.shape}")
                features = features.view(features.shape[0], -1)
            
            # Pad or truncate to latent_dim for consistent kernel shapes
            feat_dim = features.shape[1]
            logger.debug(f"Modality {modality}: feat_dim={feat_dim}, latent_dim={self.latent_dim}")
            
            if feat_dim < self.latent_dim:
                pad = torch.zeros((features.shape[0], self.latent_dim - feat_dim), device=features.device, dtype=features.dtype)
                features = torch.cat([features, pad], dim=1)
                logger.debug(f"Modality {modality}: Padded from {feat_dim} to {self.latent_dim}")
            elif feat_dim > self.latent_dim:
                features = features[:, :self.latent_dim]
                logger.debug(f"Modality {modality}: Truncated from {feat_dim} to {self.latent_dim}")
            
            logger.debug(f"Modality {modality}: Final features.shape={features.shape}")

            if method in ['mmd', 'both']:
                # MMD adaptation: minimize distance between survey distributions
                # Extract modality type (e.g., "cmb_tt" from "cmb_act_dr6_tt")
                modality_type = self._extract_modality_type(modality)
                unique_ids = torch.unique(survey_ids)
                current_sid = int(unique_ids[0].item()) if len(unique_ids) == 1 else None
                
                # Always cache features for this modality/survey combination
                if current_sid is not None:
                    cache_key = f"{modality_type}_{current_sid}"
                    # Store normalized features (already normalized above)
                    self.survey_feature_cache[cache_key] = features.detach().cpu()[:32]
                
                # Augment with cached features from OTHER surveys for the SAME modality type
                augmented_features = features
                augmented_ids = survey_ids
                
                if current_sid is not None and self.survey_feature_cache:
                    other_feats = []
                    other_ids = []
                    # Look for cached features from OTHER surveys for THIS modality type
                    for cache_key_str, feats in self.survey_feature_cache.items():
                        if feats is None or feats.numel() == 0:
                            continue
                        # Parse cache key: "modality_type_survey_id"
                        if isinstance(cache_key_str, str) and '_' in cache_key_str:
                            parts = cache_key_str.rsplit('_', 1)
                            if len(parts) == 2:
                                cached_mod_type, cached_sid_str = parts
                                if cached_mod_type == modality_type:
                                    try:
                                        cached_sid = int(cached_sid_str)
                                        # Only use features from different surveys
                                        if cached_sid != current_sid:
                                            feats_dev = feats.to(self.device)
                                            # Ensure cached features are 2D and normalized to latent_dim
                                            if feats_dev.dim() > 2:
                                                feats_dev = feats_dev.flatten(start_dim=1)
                                            elif feats_dev.dim() == 1:
                                                feats_dev = feats_dev.unsqueeze(0)
                                            # Ensure exactly 2D: (batch, features)
                                            if feats_dev.dim() != 2:
                                                continue
                                            cached_feat_dim = feats_dev.shape[1]
                                            if cached_feat_dim < self.latent_dim:
                                                pad = torch.zeros((feats_dev.shape[0], self.latent_dim - cached_feat_dim), device=feats_dev.device, dtype=feats_dev.dtype)
                                                feats_dev = torch.cat([feats_dev, pad], dim=1)
                                            elif cached_feat_dim > self.latent_dim:
                                                feats_dev = feats_dev[:, :self.latent_dim]
                                            # Final check: ensure 2D with correct feature dim
                                            if feats_dev.dim() == 2 and feats_dev.shape[1] == self.latent_dim:
                                                other_feats.append(feats_dev)
                                                other_ids.append(torch.full((feats_dev.shape[0],), cached_sid, dtype=torch.long, device=self.device))
                                    except (ValueError, IndexError, RuntimeError) as e:
                                        continue
                    
                    if other_feats:
                        cached_feat_cat = torch.cat(other_feats, dim=0)
                        cached_ids_cat = torch.cat(other_ids, dim=0)
                        # Ensure features is also properly normalized before concatenation
                        if features.dim() != 2 or features.shape[1] != self.latent_dim:
                            # Re-normalize features to be safe
                            if features.dim() > 2:
                                features = features.flatten(start_dim=1)
                            feat_dim = features.shape[1]
                            if feat_dim < self.latent_dim:
                                pad = torch.zeros((features.shape[0], self.latent_dim - feat_dim), device=features.device, dtype=features.dtype)
                                features = torch.cat([features, pad], dim=1)
                            elif feat_dim > self.latent_dim:
                                features = features[:, :self.latent_dim]
                        augmented_features = torch.cat([features, cached_feat_cat], dim=0)
                        augmented_ids = torch.cat([survey_ids, cached_ids_cat], dim=0)
                
                # Final safety check: ensure augmented_features is 2D with correct dimensions
                if augmented_features.dim() != 2:
                    augmented_features = augmented_features.flatten(start_dim=1)
                if augmented_features.shape[1] != self.latent_dim:
                    feat_dim = augmented_features.shape[1]
                    if feat_dim < self.latent_dim:
                        pad = torch.zeros((augmented_features.shape[0], self.latent_dim - feat_dim), device=augmented_features.device, dtype=augmented_features.dtype)
                        augmented_features = torch.cat([augmented_features, pad], dim=1)
                    else:
                        augmented_features = augmented_features[:, :self.latent_dim]

                mmd_loss = self._compute_batch_mmd_loss(augmented_features, augmented_ids)
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
        
        # Convert tensors to floats for logging/storage to avoid "tensor(0., ...)" strings in JSON
        loggable_losses = {}
        for k, v in losses.items():
            loggable_losses[k] = v.item() if isinstance(v, torch.Tensor) else v
            
        self.adaptation_losses.append(loggable_losses)

        # Return loggable (JSON-safe) losses so callers don't propagate tensors
        return loggable_losses

    def _extract_modality_type(self, modality: str) -> str:
        """
        Extract modality type from full modality name.
        
        Examples:
            "cmb_act_dr6_tt" -> "cmb_tt"
            "cmb_planck_2018_te" -> "cmb_te"
            "bao_boss_dr12" -> "bao"
            "galaxy" -> "galaxy"
        """
        if modality.startswith('cmb_'):
            # Extract polarization type (tt, te, ee) from CMB modalities
            for pol_type in ['tt', 'te', 'ee']:
                if modality.endswith(f'_{pol_type}'):
                    return f'cmb_{pol_type}'
            return 'cmb'  # Fallback
        elif modality.startswith('bao_'):
            return 'bao'
        elif modality.startswith('void_'):
            return 'void'
        elif modality.startswith('gw_'):
            return 'gw'
        else:
            return modality  # galaxy, frb, lyman_alpha, jwst

    def _compute_batch_mmd_loss(self, features: torch.Tensor,
                               survey_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute MMD loss across all survey pairs in batch.
        """
        # Ensure features are properly normalized before MMD computation
        if features.dim() > 2:
            features = features.flatten(start_dim=1)
        elif features.dim() == 1:
            features = features.unsqueeze(0)
        
        # Normalize feature dimensions to latent_dim
        feat_dim = features.shape[1]
        if feat_dim < self.latent_dim:
            pad = torch.zeros((features.shape[0], self.latent_dim - feat_dim), 
                             device=features.device, dtype=features.dtype)
            features = torch.cat([features, pad], dim=1)
        elif feat_dim > self.latent_dim:
            features = features[:, :self.latent_dim]
        
        unique_surveys = torch.unique(survey_ids)
        mmd_losses = []

        for i, survey1 in enumerate(unique_surveys):
            for survey2 in unique_surveys[i+1:]:
                # Get features for each survey
                mask1 = survey_ids == survey1
                mask2 = survey_ids == survey2

                # Need at least 1 sample per survey for MMD
                if mask1.sum() > 0 and mask2.sum() > 0:
                    features1 = features[mask1]
                    features2 = features[mask2]
                    
                    # Final safety check: ensure both are 2D
                    if features1.dim() != 2 or features2.dim() != 2:
                        continue
                    if features1.shape[1] != self.latent_dim or features2.shape[1] != self.latent_dim:
                        continue

                    mmd_loss = self.compute_mmd_loss(features1, features2)
                    mmd_losses.append(mmd_loss)

        if mmd_losses:
            return torch.stack(mmd_losses).mean()
        else:
            # If no pairs found, return a small non-zero value to ensure gradients flow
            return torch.tensor(1e-6, device=features.device, requires_grad=True)
            
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
