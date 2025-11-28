"""
Watershed Zone Finding
======================

Implements the watershed algorithm for identifying zones around density minima.

This is stage 3 of the ZOBOV algorithm: particles are assigned to zones
by following the gradient to the lowest-density neighbor until reaching
a local density minimum.
"""

import numpy as np
from typing import Dict, Any, Optional, Tuple
from tqdm import tqdm
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

logger = logging.getLogger(__name__)


class WatershedError(Exception):
    """Error in watershed zone finding."""
    pass


class WatershedZoneFinder:
    """
    Watershed algorithm for zone identification around density minima.
    
    Algorithm:
    1. Identify all local density minima
    2. For each particle, follow gradient to lowest-density neighbor
    3. Assign particle to zone of the minimum it reaches
    4. Iterate until all particles assigned
    """
    
    def __init__(self, use_mps: bool = True, device=None):
        """
        Initialize watershed zone finder.
        
        Parameters:
            use_mps: Whether to use MPS acceleration
            device: Torch device (auto-detected if None and use_mps=True)
        """
        self.use_mps = use_mps and TORCH_AVAILABLE
        
        if self.use_mps:
            if device is None:
                if not torch.backends.mps.is_available():
                    raise WatershedError("MPS requested but not available")
                device = torch.device("mps")
            self.device = device
            logger.info("Using MPS acceleration for watershed")
        else:
            self.device = None
            logger.info("Using CPU for watershed")
    
    def find_local_minima(self, densities: np.ndarray, 
                         positions: np.ndarray,
                         k_neighbors: int = 10) -> np.ndarray:
        """
        Identify local density minima.
        
        A particle is a local minimum if its density is lower than all k nearest neighbors.
        
        Parameters:
            densities: Density values for each particle
            positions: Particle positions (n_particles, 3)
            k_neighbors: Number of nearest neighbors to check
            
        Returns:
            Boolean array indicating which particles are local minima
        """
        from scipy.spatial import cKDTree
        
        n_particles = len(densities)
        
        # Constrain k_neighbors to available neighbors (maximum n_particles - 1)
        max_k = min(k_neighbors, n_particles - 1)
        if max_k < 1:
            # Insufficient particles for neighbor comparison; treat all as minima
            logger.warning(f"Insufficient particles ({n_particles}) for neighbor comparison; treating all as minima")
            return np.ones(n_particles, dtype=bool)
        
        logger.info(f"Finding local density minima (k={max_k}, n_particles={n_particles})...")
        
        tree = cKDTree(positions)
        is_minimum = np.zeros(n_particles, dtype=bool)
        
        for i in range(n_particles):
            # Query k+1 nearest neighbors (includes self) up to available particles
            k_request = min(max_k + 1, n_particles)
            result = tree.query(positions[i], k=k_request)
            
            # Handle scalar return when k_request=1 (isolated particle)
            if k_request == 1:
                is_minimum[i] = True
                continue
            
            distances, indices = result
            indices = np.atleast_1d(indices)
            
            # Validate indices are within array bounds
            valid_indices = indices[indices < n_particles]
            
            # Compare density to neighbors (exclude self, which is first index)
            if len(valid_indices) > 1:
                neighbor_indices = valid_indices[1:]
                neighbor_densities = densities[neighbor_indices]
                is_minimum[i] = densities[i] < np.min(neighbor_densities)
            else:
                # Isolated particle: no neighbors found, treat as minimum
                is_minimum[i] = True
        
        n_minima = np.sum(is_minimum)
        logger.info(f"✓ Found {n_minima:,} local density minima ({100*n_minima/n_particles:.2f}% of particles)")
        
        return is_minimum
    
    def find_lowest_density_neighbor(self, densities: np.ndarray,
                                    positions: np.ndarray,
                                    particle_idx: int,
                                    k_neighbors: int = 20) -> int:
        """
        Find the lowest-density neighbor of a particle.
        
        Parameters:
            densities: Density values
            positions: Particle positions
            particle_idx: Index of particle to check
            k_neighbors: Number of neighbors to consider
            
        Returns:
            Index of lowest-density neighbor (or self if isolated)
        """
        from scipy.spatial import cKDTree
        
        n_particles = len(densities)
        
        # Constrain k_neighbors to available neighbors
        max_k = min(k_neighbors, n_particles - 1)
        if max_k < 1:
            return particle_idx
        
        tree = cKDTree(positions)
        k_request = min(max_k + 1, n_particles)
        result = tree.query(positions[particle_idx], k=k_request)
        
        # Handle scalar return when k_request=1
        if k_request == 1:
            return particle_idx
        
        distances, indices = result
        indices = np.atleast_1d(indices)
        
        # Validate indices and exclude self (first index)
        valid_indices = indices[indices < n_particles]
        neighbor_indices = valid_indices[1:] if len(valid_indices) > 1 else []
        
        if len(neighbor_indices) == 0:
            return particle_idx
        
        # Identify neighbor with minimum density
        neighbor_densities = densities[neighbor_indices]
        min_idx = np.argmin(neighbor_densities)
        return neighbor_indices[min_idx]
    
    def assign_zones(self, densities: np.ndarray,
                    positions: np.ndarray,
                    local_minima: np.ndarray,
                    max_iterations: int = 1000) -> Dict[str, Any]:
        """
        Assign all particles to zones by following density gradient.
        
        Parameters:
            densities: Density values for each particle
            positions: Particle positions (n_particles, 3)
            local_minima: Boolean array indicating local minima
            max_iterations: Maximum iterations per particle
            
        Returns:
            Dictionary with:
            - 'zone_ids': Zone assignment for each particle
            - 'zone_minima': Indices of zone minima
            - 'n_zones': Number of zones found
            - 'iterations': Number of iterations used
        """
        n_particles = len(densities)
        zone_ids = np.full(n_particles, -1, dtype=np.int32)
        
        # Initialize zones from local minima
        zone_counter = 0
        zone_minima = []
        for i in range(n_particles):
            if local_minima[i]:
                zone_ids[i] = zone_counter
                zone_minima.append(i)
                zone_counter += 1
        
        # Edge case: no local minima detected (e.g., insufficient particles for neighbor comparison)
        # Create single zone from lowest-density particle to ensure algorithm can proceed
        if len(zone_minima) == 0:
            logger.warning(f"No local minima detected with {n_particles} particles; creating single zone from lowest-density particle")
            lowest_density_idx = np.argmin(densities)
            zone_ids[lowest_density_idx] = 0
            zone_minima = [lowest_density_idx]
            zone_counter = 1
        
        logger.info(f"Assigning {n_particles:,} particles to {zone_counter:,} zones...")
        
        # Assign unassigned particles by following density gradient
        unassigned = np.where(zone_ids == -1)[0]
        total_iterations = 0
        
        if len(unassigned) > 0:
            with tqdm(total=len(unassigned), desc="Assigning zones", unit="particle") as pbar:
                for particle_idx in unassigned:
                    current_idx = particle_idx
                    iterations = 0
                    
                    # Follow density gradient until reaching a local minimum
                    while zone_ids[current_idx] == -1 and iterations < max_iterations:
                        next_idx = self.find_lowest_density_neighbor(
                            densities, positions, current_idx
                        )
                        
                        # Assign to zone if minimum reached
                        if local_minima[next_idx] or next_idx in zone_minima:
                            zone_ids[particle_idx] = zone_ids[next_idx]
                            break
                        
                        current_idx = next_idx
                        iterations += 1
                    
                    # Fallback: assign to nearest minimum if gradient following failed
                    if zone_ids[particle_idx] == -1:
                        from scipy.spatial import cKDTree
                        if len(zone_minima) > 0:
                            min_positions = positions[zone_minima]
                            tree = cKDTree(min_positions)
                            result = tree.query(positions[particle_idx], k=1)
                            nearest_min_idx = result[1] if isinstance(result, tuple) else result
                            # Handle scalar return from cKDTree.query
                            if np.isscalar(nearest_min_idx):
                                nearest_min_idx = int(nearest_min_idx)
                            else:
                                nearest_min_idx = int(nearest_min_idx[0]) if len(nearest_min_idx) > 0 else 0
                            zone_ids[particle_idx] = zone_ids[zone_minima[nearest_min_idx]]
                        else:
                            # Safety fallback (should not occur due to check above)
                            zone_ids[particle_idx] = 0
                    
                    total_iterations += iterations
                    pbar.update(1)
        
        n_zones = len(zone_minima)
        avg_iterations = total_iterations / len(unassigned) if len(unassigned) > 0 else 0.0
        logger.info(f"✓ Zone assignment complete: {n_zones:,} zones, "
                   f"avg {avg_iterations:.1f} iterations per particle")
        
        return {
            'zone_ids': zone_ids,
            'zone_minima': np.array(zone_minima),
            'n_zones': n_zones,
            'iterations': total_iterations
        }
    
    def process(self, densities: np.ndarray,
               positions: np.ndarray,
               k_neighbors: int = 10) -> Dict[str, Any]:
        """
        Complete watershed zone finding process.
        
        Parameters:
            densities: Density values for each particle
            positions: Particle positions (n_particles, 3)
            k_neighbors: Number of neighbors for minimum detection
            
        Returns:
            Dictionary with zone assignments and metadata
        """
        # Find local minima
        local_minima = self.find_local_minima(densities, positions, k_neighbors)
        
        # Assign zones
        zone_data = self.assign_zones(densities, positions, local_minima)
        
        return {
            **zone_data,
            'local_minima': local_minima,
            'densities': densities,
            'positions': positions
        }

