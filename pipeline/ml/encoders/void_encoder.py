"""
Void Encoder
===========

Encoder for cosmic void catalogs.
Implements spherical-to-Cartesian coordinate transformation to respect
spatial topology of the universe.
"""

import torch
import torch.nn as nn
import numpy as np


class VoidEncoder(nn.Module):
    """
    Encoder for void properties with geometric awareness.
    
    Assumes input features start with [RA, Dec, Redshift, Radius, ...]
    Performs internal coordinate transformation to Cartesian space.
    """

    def __init__(self, input_dim: int, latent_dim: int = 512):
        super().__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Removed self.input_norm because we handle normalization internally
        # after splitting coordinates from intrinsic features.
        
        # Spatial path: (x, y, z) -> hidden
        # Processes Cartesian coordinates derived from (RA, Dec, z)
        self.spatial_net = nn.Sequential(
            nn.Linear(3, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )
        
        # Intrinsic properties path: (radius, density, etc.) -> hidden
        # input_dim - 3 spatial coords
        intrinsic_dim = max(1, input_dim - 3)
        
        # Optional batch norm for intrinsic features
        self.intrinsic_norm = nn.BatchNorm1d(intrinsic_dim)
        
        self.intrinsic_net = nn.Sequential(
            nn.Linear(intrinsic_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.LayerNorm(128)
        )
        
        # Combined path
        self.combined_net = nn.Sequential(
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, latent_dim),
            nn.LayerNorm(latent_dim)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with coordinate transform.
        
        Expected input format: [RA, Dec, Redshift, ...]
        RA, Dec in degrees.
        """
        # Split first
        if x.shape[1] < 3:
            # Fallback for empty/malformed inputs
            return self.combined_net(torch.cat([
                torch.zeros(x.shape[0], 128, device=x.device),
                torch.zeros(x.shape[0], 128, device=x.device)
            ], dim=1))
            
        # Extract coordinates (assuming first 3 columns)
        # We use RAW values for coords before normalization
        # Angles must be preserved for trig functions
        ra = x[:, 0] * (np.pi / 180.0)  # deg -> rad
        dec = x[:, 1] * (np.pi / 180.0) # deg -> rad
        z = x[:, 2]
        
        # Approximate distance (using redshift as proxy for comoving distance in this embedding)
        dist = z 
        
        # Spherical to Cartesian
        cart_x = dist * torch.cos(dec) * torch.cos(ra)
        cart_y = dist * torch.cos(dec) * torch.sin(ra)
        cart_z = dist * torch.sin(dec)
        
        spatial_features = torch.stack([cart_x, cart_y, cart_z], dim=1)
        
        # Process spatial features
        spatial_out = self.spatial_net(spatial_features)
        
        # Process intrinsic features
        intrinsic_features = x[:, 3:]
        
        if intrinsic_features.shape[1] > 0:
            # Apply normalization to intrinsic features ONLY
            # This handles the scale differences (radius vs density etc)
            # safely without messing up the coordinates.
            if x.shape[0] > 1: # BatchNorm requires batch > 1
                 intrinsic_features = self.intrinsic_norm(intrinsic_features)
                 
            intrinsic_out = self.intrinsic_net(intrinsic_features)
        else:
            # Handle case where only coordinates exist
            dummy_input = torch.zeros(x.shape[0], 1, device=x.device)
            # Use first layer of intrinsic net if it matches dummy dimension? 
            # No, intrinsic_net expects intrinsic_dim.
            # Just create zero embedding of correct size.
            intrinsic_out = torch.zeros(x.shape[0], 128, device=x.device)
        
        # Combine
        combined = torch.cat([spatial_out, intrinsic_out], dim=1)
        return self.combined_net(combined)
