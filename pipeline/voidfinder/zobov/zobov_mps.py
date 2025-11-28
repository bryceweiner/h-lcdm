"""
H-ZOBOV MPS Acceleration Utilities
===================================

MPS-accelerated operations for H-ZOBOV algorithm using Apple Silicon GPU.
"""

import numpy as np
from typing import Optional, Tuple
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class ZOBOVMPSError(Exception):
    """Error in MPS operations."""
    pass


class ZOBOVMPS:
    """
    MPS-accelerated operations for H-ZOBOV.
    
    Provides batched operations for large datasets using Apple Silicon GPU.
    """
    
    def __init__(self, device: Optional[torch.device] = None):
        """
        Initialize MPS accelerator.
        
        Parameters:
            device: Torch device (auto-detected if None)
            
        Raises:
            ZOBOVMPSError: If MPS is not available
        """
        if not TORCH_AVAILABLE:
            raise ZOBOVMPSError("PyTorch not available - cannot use MPS acceleration")
        
        if device is None:
            if not torch.backends.mps.is_available():
                raise ZOBOVMPSError("MPS not available - Apple Silicon required")
            device = torch.device("mps")
        
        self.device = device
        logger.info(f"MPS accelerator initialized on device: {device}")
    
    def batched_pairwise_distances(self, positions: np.ndarray,
                                  batch_size: int = 50000) -> np.ndarray:
        """
        Compute pairwise distances in batches using MPS.
        
        Parameters:
            positions: Array of shape (n_points, 3)
            batch_size: Batch size for processing
            
        Returns:
            Pairwise distance matrix (n_points, n_points)
        """
        n_points = len(positions)
        
        if n_points > batch_size:
            logger.info(f"Computing pairwise distances in batches (batch_size={batch_size})...")
            # Batched computation for large datasets: memory-efficient at cost of additional computation
            distances = np.zeros((n_points, n_points))
            
            for i in range(0, n_points, batch_size):
                end_i = min(i + batch_size, n_points)
                batch_i = positions[i:end_i]
                
                batch_i_tensor = torch.from_numpy(batch_i).float().to(self.device)
                
                for j in range(0, n_points, batch_size):
                    end_j = min(j + batch_size, n_points)
                    batch_j = positions[j:end_j]
                    
                    batch_j_tensor = torch.from_numpy(batch_j).float().to(self.device)
                    
                    # Compute distances
                    batch_distances = torch.cdist(batch_i_tensor, batch_j_tensor)
                    distances[i:end_i, j:end_j] = batch_distances.cpu().numpy()
            
            return distances
        else:
            # Small dataset: compute all pairwise distances simultaneously
            pos_tensor = torch.from_numpy(positions).float().to(self.device)
            distances = torch.cdist(pos_tensor, pos_tensor)
            return distances.cpu().numpy()
    
    def batched_k_nearest_neighbors(self, positions: np.ndarray,
                                   k: int,
                                   batch_size: int = 50000) -> Tuple[np.ndarray, np.ndarray]:
        """
        Find k nearest neighbors in batches using MPS.
        
        Parameters:
            positions: Array of shape (n_points, 3)
            k: Number of neighbors
            batch_size: Batch size for processing
            
        Returns:
            Tuple of (distances, indices) arrays, each of shape (n_points, k)
        """
        n_points = len(positions)
        k = min(k, n_points - 1)  # Constrain k to available neighbors
        
        all_distances = []
        all_indices = []
        
        pos_tensor = torch.from_numpy(positions).float().to(self.device)
        
        for i in range(0, n_points, batch_size):
            end_i = min(i + batch_size, n_points)
            batch_positions = pos_tensor[i:end_i]
            
            # Compute distances to all points
            distances = torch.cdist(batch_positions, pos_tensor)
            
            # Get k+1 nearest (including self)
            k_nearest_distances, k_nearest_indices = torch.topk(distances, k + 1, dim=1, largest=False)
            
            # Remove self (first column)
            k_nearest_distances = k_nearest_distances[:, 1:]
            k_nearest_indices = k_nearest_indices[:, 1:]
            
            all_distances.append(k_nearest_distances.cpu().numpy())
            all_indices.append(k_nearest_indices.cpu().numpy())
        
        distances_array = np.vstack(all_distances)
        indices_array = np.vstack(all_indices)
        
        return distances_array, indices_array
    
    def batched_density_gradient(self, densities: np.ndarray,
                                positions: np.ndarray,
                                batch_size: int = 50000) -> np.ndarray:
        """
        Compute density gradient direction for each particle using MPS.
        
        Parameters:
            densities: Density values (n_particles,)
            positions: Particle positions (n_particles, 3)
            batch_size: Batch size for processing
            
        Returns:
            Gradient directions (n_particles, 3)
        """
        n_particles = len(positions)
        gradients = np.zeros((n_particles, 3))
        
        pos_tensor = torch.from_numpy(positions).float().to(self.device)
        dens_tensor = torch.from_numpy(densities).float().to(self.device)
        
        for i in range(0, n_particles, batch_size):
            end_i = min(i + batch_size, n_particles)
            batch_positions = pos_tensor[i:end_i]
            
            # Compute distances to all particles
            distances = torch.cdist(batch_positions, pos_tensor)
            
            # Find nearest neighbors (excluding self)
            k = min(10, n_particles - 1)
            _, nearest_indices = torch.topk(distances, k + 1, dim=1, largest=False)
            nearest_indices = nearest_indices[:, 1:]  # Remove self
            
            # Compute gradient as weighted average of neighbor directions
            for j, idx in enumerate(range(i, end_i)):
                neighbors = nearest_indices[j]
                neighbor_positions = pos_tensor[neighbors]
                neighbor_densities = dens_tensor[neighbors]
                
                # Weight by inverse distance and density difference
                neighbor_distances = distances[j, neighbors]
                weights = (neighbor_densities - dens_tensor[idx]) / (neighbor_distances + 1e-10)
                weights = torch.clamp(weights, min=0)  # Only consider decreasing density
                
                if torch.sum(weights) > 0:
                    # Normalize weights
                    weights = weights / torch.sum(weights)
                    
                    # Compute weighted direction
                    directions = neighbor_positions - pos_tensor[idx]
                    gradient = torch.sum(weights.unsqueeze(1) * directions, dim=0)
                    gradients[idx] = gradient.cpu().numpy()
        
        # Normalize gradients
        norms = np.linalg.norm(gradients, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        gradients = gradients / norms
        
        return gradients

