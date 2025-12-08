"""
Self-Supervised Learning Encoder
===============================

Contrastive learning framework for cosmological data.
Implements SimCLR-based learning on multi-modal cosmological datasets
without requiring labeled data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
import logging

# Import modality-specific encoders
from pipeline.ml.encoders import (
    CMBEncoder, BAOEncoder, VoidEncoder, GalaxyEncoder,
    FRBEncoder, LymanAlphaEncoder, JWSTEncoder, GWEncoder
)


class ContrastiveLearner:
    """
    Self-supervised contrastive learning for cosmological data.

    Implements SimCLR-style learning on multi-modal datasets:
    - CMB power spectra
    - BAO measurements
    - Void catalogs
    - Galaxy catalogs
    - FRB data
    - Lyman-alpha spectra
    - JWST catalogs
    """

    def __init__(self, encoder_dims: Dict[str, int],
                 latent_dim: int = 512,
                 temperature: float = 0.5,
                 learning_rate: float = 1e-3,
                 device: str = 'auto'):
        """
        Initialize contrastive learner.

        Parameters:
            encoder_dims: Input dimensions for each modality
            latent_dim: Dimension of shared latent space
            temperature: Temperature parameter for NT-Xent loss
            learning_rate: Learning rate for optimization
            device: Device to run on ('cpu', 'cuda', 'auto')
        """
        # Initialize logger FIRST
        self.logger = logging.getLogger(__name__)
        
        self.encoder_dims = encoder_dims
        self.latent_dim = latent_dim
        self.temperature = temperature
        self.learning_rate = learning_rate

        # Set device
        if device == 'auto':
            # Priority: MPS (Apple Silicon) > CUDA > CPU
            if torch.backends.mps.is_available():
                self.device = torch.device('mps')
            elif torch.cuda.is_available():
                self.device = torch.device('cuda')
            else:
                self.device = torch.device('cpu')
        else:
            self.device = torch.device(device)

        # Initialize encoders for each modality
        self.encoders = self._build_encoders()
        self.projector = self._build_projector()

        # Move to device
        self.encoders = {mod: enc.to(self.device) for mod, enc in self.encoders.items()}
        self.projector = self.projector.to(self.device)

        # Optimizer
        encoder_params = []
        for encoder in self.encoders.values():
            encoder_params.extend(list(encoder.parameters()))
        encoder_params.extend(list(self.projector.parameters()))

        self.optimizer = torch.optim.Adam(encoder_params, lr=learning_rate)

        # Momentum encoder for stable training
        self.momentum_encoders = self._build_momentum_encoders()
        self.momentum_projector = self._build_projector()
        self.momentum_encoders = {mod: enc.to(self.device) for mod, enc in self.momentum_encoders.items()}
        self.momentum_projector = self.momentum_projector.to(self.device)

        # Momentum update parameter
        self.momentum = 0.996

    def eval(self):
        """Set encoders/projectors to eval mode for inference."""
        for encoder in self.encoders.values():
            encoder.eval()
        for encoder in self.momentum_encoders.values():
            encoder.eval()
        self.projector.eval()
        self.momentum_projector.eval()

    def _build_encoders(self) -> Dict[str, nn.Module]:
        """Build modality-specific encoders."""
        encoders = {}

        # CMB encoder (harmonic space) - handle both 'cmb' and CMB sub-modalities
        cmb_modalities = [mod for mod in self.encoder_dims.keys() if mod.startswith('cmb')]
        if cmb_modalities:
            # Use the first CMB modality's dimension (they should all be the same)
            cmb_dim = self.encoder_dims[cmb_modalities[0]]
            for mod in cmb_modalities:
                encoders[mod] = CMBEncoder(cmb_dim, self.latent_dim)

        # BAO encoder (distance measurements) - handle both 'bao' and BAO sub-modalities
        bao_modalities = [mod for mod in self.encoder_dims.keys() if mod.startswith('bao')]
        if bao_modalities:
            # Use the first BAO modality's dimension (they should all be the same)
            bao_dim = self.encoder_dims[bao_modalities[0]]
            for mod in bao_modalities:
                encoders[mod] = BAOEncoder(bao_dim, self.latent_dim)

        # Void encoder (catalog features) - handle both 'void' and void sub-modalities
        void_modalities = [mod for mod in self.encoder_dims.keys() if mod.startswith('void')]
        if void_modalities:
            # Use the first void modality's dimension (they should all be the same)
            void_dim = self.encoder_dims[void_modalities[0]]
            for mod in void_modalities:
                encoders[mod] = VoidEncoder(void_dim, self.latent_dim)

        # Galaxy encoder (photometric + morphological)
        if 'galaxy' in self.encoder_dims:
            encoders['galaxy'] = GalaxyEncoder(self.encoder_dims['galaxy'], self.latent_dim)

        # FRB encoder (timing sequences)
        if 'frb' in self.encoder_dims:
            encoders['frb'] = FRBEncoder(self.encoder_dims['frb'], self.latent_dim)

        # Lyman-alpha encoder (spectra)
        if 'lyman_alpha' in self.encoder_dims:
            encoders['lyman_alpha'] = LymanAlphaEncoder(self.encoder_dims['lyman_alpha'], self.latent_dim)

        # JWST encoder (high-z imaging)
        if 'jwst' in self.encoder_dims:
            encoders['jwst'] = JWSTEncoder(self.encoder_dims['jwst'], self.latent_dim)

        # GW encoder (gravitational wave events) - handle both 'gw' and GW sub-modalities
        gw_modalities = [mod for mod in self.encoder_dims.keys() if mod.startswith('gw')]
        if gw_modalities:
            # Use the first GW modality's dimension (they should all be the same)
            gw_dim = self.encoder_dims[gw_modalities[0]]
            for mod in gw_modalities:
                encoders[mod] = GWEncoder(gw_dim, self.latent_dim)

        return encoders

    def _build_momentum_encoders(self) -> Dict[str, nn.Module]:
        """Build momentum encoders for stable training."""
        momentum_encoders = {}
        for modality, encoder in self.encoders.items():
            momentum_encoders[modality] = self._copy_encoder(encoder)
        return momentum_encoders

    def _copy_encoder(self, encoder: nn.Module) -> nn.Module:
        """Create a copy of encoder for momentum updates."""
        # Use type(encoder) to instantiate the correct class when possible.
        # Fall back to deepcopy if that fails for any reason.
        try:
            if isinstance(encoder, CMBEncoder):
                # CMBEncoder signature: (input_dim, latent_dim, n_filters, kernel_size, ...)
                # Use existing input_dim / latent_dim and default values for the rest.
                new_encoder = type(encoder)(encoder.input_dim, encoder.latent_dim)
            else:
                # Generic case: assume (input_dim, latent_dim) constructor or compatible signature.
                if hasattr(encoder, "input_dim") and hasattr(encoder, "latent_dim"):
                    new_encoder = type(encoder)(encoder.input_dim, encoder.latent_dim)  # type: ignore[arg-type]
                else:
                    raise AttributeError("Encoder missing input_dim/latent_dim attributes")

            new_encoder.load_state_dict(encoder.state_dict())
            for param in new_encoder.parameters():
                param.requires_grad = False
            return new_encoder
        except Exception as e:
            self.logger.error(f"Failed to copy encoder {type(encoder)}: {e}")
            # Fallback to deepcopy if manual instantiation fails
            import copy
            new_encoder = copy.deepcopy(encoder)
            for param in new_encoder.parameters():
                param.requires_grad = False
            return new_encoder

    def _build_projector(self) -> nn.Module:
        """Build projection head for contrastive learning."""
        return nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim)
        )

    def _momentum_update(self):
        """Update momentum encoders."""
        for modality in self.encoders.keys():
            # Update encoder parameters
            for param_q, param_k in zip(self.encoders[modality].parameters(),
                                      self.momentum_encoders[modality].parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

            # Update projector parameters
            for param_q, param_k in zip(self.projector.parameters(),
                                      self.momentum_projector.parameters()):
                param_k.data = param_k.data * self.momentum + param_q.data * (1 - self.momentum)

    def forward(self, batch: Dict[str, torch.Tensor], use_momentum: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through encoders.

        Parameters:
            batch: Batch of data for each modality
            use_momentum: Whether to use momentum encoders

        Returns:
            dict: Encoded representations
        """
        encodings = {}

        encoders = self.momentum_encoders if use_momentum else self.encoders

        for modality, data in batch.items():
            if modality in encoders:
                # Encode
                z = encoders[modality](data)

                # Project to contrastive space
                if use_momentum:
                    z_proj = self.momentum_projector(z)
                else:
                    z_proj = self.projector(z)

                encodings[modality] = {
                    'encoded': z,
                    'projected': z_proj
                }

        return encodings

    def compute_contrastive_loss(self, encodings: Dict[str, Dict[str, torch.Tensor]]) -> torch.Tensor:
        """
        Compute NT-Xent loss across all modalities.

        Parameters:
            encodings: Encoded representations from forward pass

        Returns:
            torch.Tensor: Contrastive loss
        """
        total_loss = torch.tensor(0.0, device=self.device)
        n_modalities = len(encodings)

        if n_modalities < 2:
            return total_loss

        # Compute loss for each modality pair
        modalities = list(encodings.keys())
        for i, mod1 in enumerate(modalities):
            for j, mod2 in enumerate(modalities):
                if i != j:
                    # Get projected representations
                    z1 = encodings[mod1]['projected']
                    z2 = encodings[mod2]['projected']

                    # NT-Xent loss
                    loss = self._nt_xent_loss(z1, z2)
                    total_loss += loss

        return total_loss / (n_modalities * (n_modalities - 1))

    def _nt_xent_loss(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Compute NT-Xent loss between two sets of representations.

        Parameters:
            z1: Representations from first modality
            z2: Representations from second modality

        Returns:
            torch.Tensor: NT-Xent loss
        """
        # Normalize representations
        z1 = F.normalize(z1, dim=1)
        z2 = F.normalize(z2, dim=1)

        # Compute similarity matrix
        sim_matrix = torch.matmul(z1, z2.T) / self.temperature

        # Labels are diagonal (matching pairs)
        batch_size = z1.shape[0]
        labels = torch.arange(batch_size, device=self.device)

        # Compute loss
        loss1 = F.cross_entropy(sim_matrix, labels)
        loss2 = F.cross_entropy(sim_matrix.T, labels)

        return (loss1 + loss2) / 2

    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """
        Perform one training step.

        Parameters:
            batch: Batch of augmented data

        Returns:
            dict: Training metrics
        """
        self.optimizer.zero_grad()

        # Forward pass with online encoders
        online_encodings = self.forward(batch, use_momentum=False)

        # Forward pass with momentum encoders
        momentum_encodings = self.forward(batch, use_momentum=True)

        # Compute contrastive loss
        loss = self.compute_contrastive_loss(online_encodings)

        # Backward pass
        loss.backward()
        self.optimizer.step()

        # Update momentum encoders
        self._momentum_update()

        return {
            'loss': loss.item(),
            'n_modalities': len(online_encodings)
        }

    def encode(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode data to latent space (inference mode).

        Parameters:
            data: Input data for each modality

        Returns:
            dict: Encoded representations
        """
        with torch.no_grad():
            encodings = self.forward(data, use_momentum=False)
            return {mod: enc['encoded'] for mod, enc in encodings.items()}

    def encode_with_grad(self, data: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Encode data to latent space with gradients enabled (for domain adaptation).
        """
        encodings = self.forward(data, use_momentum=False)
        # Return only the encoded tensor (not the projected dict) so downstream
        # domain adaptation sees a 2D tensor per modality.
        return {mod: enc['encoded'] for mod, enc in encodings.items() if isinstance(enc, dict) and 'encoded' in enc}

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        checkpoint = {
            'encoder_dims': self.encoder_dims,
            'latent_dim': self.latent_dim,
            'temperature': self.temperature,
            'learning_rate': self.learning_rate,
            'encoders': {mod: enc.state_dict() for mod, enc in self.encoders.items()},
            'projector': self.projector.state_dict(),
            'momentum_encoders': {mod: enc.state_dict() for mod, enc in self.momentum_encoders.items()},
            'momentum_projector': self.momentum_projector.state_dict(),
            'optimizer': self.optimizer.state_dict()
        }
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=self.device)

        # Load encoders
        for mod in self.encoders.keys():
            if mod in checkpoint['encoders']:
                self.encoders[mod].load_state_dict(checkpoint['encoders'][mod])
                self.momentum_encoders[mod].load_state_dict(checkpoint['momentum_encoders'][mod])

        # Load projectors
        self.projector.load_state_dict(checkpoint['projector'])
        self.momentum_projector.load_state_dict(checkpoint['momentum_projector'])

        # Load optimizer
        self.optimizer.load_state_dict(checkpoint['optimizer'])
