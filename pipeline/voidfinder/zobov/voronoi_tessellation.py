"""
Voronoi Tessellation and DTFE Density Estimation
================================================

Computes Voronoi tessellation for galaxy distribution and estimates density
field using Delaunay Tessellation Field Estimator (DTFE) method.

MPS acceleration is used for distance calculations in large datasets.
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional, Dict, Any
from scipy.spatial import Voronoi, cKDTree
import logging

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from .zobov_parameters import HZOBOVParameters

logger = logging.getLogger(__name__)


class VoronoiTessellationError(Exception):
    """Error in Voronoi tessellation."""
    pass


class VoronoiTessellation:
    """
    Voronoi tessellation and DTFE density estimation.
    
    Implements the first stage of ZOBOV algorithm:
    1. Compute Voronoi cells for all galaxies
    2. Calculate cell volumes (inverse density)
    3. Estimate density field using DTFE
    """
    
    def __init__(self, parameters: HZOBOVParameters, use_mps: bool = True):
        """
        Initialize Voronoi tessellation.
        
        Parameters:
            parameters: H-ZOBOV parameters
            use_mps: Whether to use MPS acceleration (requires Apple Silicon)
        """
        self.parameters = parameters
        self.use_mps = use_mps and TORCH_AVAILABLE
        
        if self.use_mps:
            if not torch.backends.mps.is_available():
                raise VoronoiTessellationError("MPS requested but not available - Apple Silicon required")
            self.device = torch.device("mps")
            logger.info("Using MPS acceleration for Voronoi tessellation")
        else:
            self.device = None
            logger.info("Using CPU for Voronoi tessellation")
    
    def compute_voronoi_cells(self, positions: np.ndarray) -> Voronoi:
        """
        Compute Voronoi tessellation for galaxy positions.
        
        Parameters:
            positions: Array of shape (n_galaxies, 3) with x, y, z coordinates in Mpc
            
        Returns:
            scipy.spatial.Voronoi object
            
        Raises:
            VoronoiTessellationError: If tessellation fails
        """
        if len(positions) < 4:
            raise VoronoiTessellationError(f"Need at least 4 points for Voronoi tessellation, got {len(positions)}")
        
        try:
            logger.info(f"Computing Voronoi tessellation for {len(positions):,} galaxies...")
            voronoi = Voronoi(positions)
            logger.info(f"✓ Voronoi tessellation complete: {len(voronoi.points)} points, {len(voronoi.regions)} regions")
            return voronoi
            
        except Exception as e:
            raise VoronoiTessellationError(f"Voronoi tessellation failed: {e}") from e
    
    def compute_cell_volumes(self, voronoi: Voronoi, 
                            positions: np.ndarray) -> np.ndarray:
        """
        Compute Voronoi cell volumes for each galaxy.
        
        For infinite regions (edge cells), uses convex hull approximation.
        
        Parameters:
            voronoi: scipy.spatial.Voronoi object
            positions: Original galaxy positions (n_galaxies, 3)
            
        Returns:
            Array of cell volumes in Mpc³ (same length as positions)
            
        Raises:
            VoronoiTessellationError: If volume calculation fails
        """
        try:
            n_points = len(positions)
            volumes = np.zeros(n_points)
            
            logger.info(f"Computing Voronoi cell volumes for {n_points:,} cells...")
            
            # Compute volumes for finite regions
            for i, region_idx in enumerate(voronoi.point_region):
                region = voronoi.regions[region_idx]
                
                if -1 in region:
                    # Infinite region - use convex hull approximation
                    # Volume estimated as distance to nearest neighbor cubed
                    if self.use_mps:
                        volumes[i] = self._estimate_infinite_volume_mps(positions, i)
                    else:
                        volumes[i] = self._estimate_infinite_volume_cpu(positions, i)
                else:
                    # Finite region - compute convex hull volume
                    vertices = voronoi.vertices[region]
                    volumes[i] = self._compute_convex_hull_volume(vertices)
            
            # Handle any zero or negative volumes
            min_volume = np.min(volumes[volumes > 0])
            volumes[volumes <= 0] = min_volume * 0.1  # Small but finite
            
            logger.info(f"✓ Cell volumes computed: min={np.min(volumes):.2e}, max={np.max(volumes):.2e} Mpc³")
            
            return volumes
            
        except Exception as e:
            raise VoronoiTessellationError(f"Cell volume calculation failed: {e}") from e
    
    def compute_density_field(self, volumes: np.ndarray) -> np.ndarray:
        """
        Compute density field using DTFE: ρ_i = 1 / V_i.
        
        Parameters:
            volumes: Voronoi cell volumes in Mpc³
            
        Returns:
            Density values in galaxies/Mpc³
        """
        densities = 1.0 / volumes
        logger.info(f"✓ Density field computed: min={np.min(densities):.2e}, max={np.max(densities):.2e} galaxies/Mpc³")
        return densities
    
    def _estimate_infinite_volume_cpu(self, positions: np.ndarray, idx: int) -> float:
        """Estimate volume for infinite Voronoi region using CPU."""
        # Find distance to nearest neighbor
        tree = cKDTree(positions)
        distances, _ = tree.query(positions[idx], k=2)  # k=2 to get self + nearest neighbor
        if len(distances) > 1:
            nearest_dist = distances[1]
        else:
            nearest_dist = np.mean(np.linalg.norm(positions - positions[idx], axis=1))
        
        # Estimate volume as sphere with radius = nearest neighbor distance
        volume = (4.0 / 3.0) * np.pi * nearest_dist**3
        return volume
    
    def _estimate_infinite_volume_mps(self, positions: np.ndarray, idx: int) -> float:
        """Estimate volume for infinite Voronoi region using MPS."""
        # Convert to torch tensors
        pos_tensor = torch.from_numpy(positions).float().to(self.device)
        pos_idx = pos_tensor[idx:idx+1]
        
        # Compute distances to all points
        distances = torch.norm(pos_tensor - pos_idx, dim=1)
        
        # Get nearest neighbor (excluding self)
        distances_sorted, _ = torch.sort(distances)
        nearest_dist = distances_sorted[1].item() if len(distances_sorted) > 1 else distances_sorted[0].item()
        
        # Estimate volume
        volume = (4.0 / 3.0) * np.pi * nearest_dist**3
        return volume
    
    def _compute_convex_hull_volume(self, vertices: np.ndarray) -> float:
        """
        Compute volume of convex hull defined by vertices.
        
        Uses scipy.spatial.ConvexHull for 3D volume calculation.
        """
        from scipy.spatial import ConvexHull
        
        if len(vertices) < 4:
            # Not enough points for 3D volume - use bounding box approximation
            ranges = np.max(vertices, axis=0) - np.min(vertices, axis=0)
            return np.prod(ranges)
        
        try:
            hull = ConvexHull(vertices)
            return hull.volume
        except Exception:
            # Approximate volume using bounding box if convex hull computation fails
            ranges = np.max(vertices, axis=0) - np.min(vertices, axis=0)
            return np.prod(ranges)
    
    def process(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """
        Complete Voronoi tessellation and density estimation.
        
        Parameters:
            galaxy_catalog: DataFrame with x, y, z columns (comoving Mpc)
            
        Returns:
            Dictionary with:
            - 'voronoi': Voronoi object
            - 'volumes': Cell volumes in Mpc³
            - 'densities': Density values in galaxies/Mpc³
            - 'positions': Galaxy positions array
            
        Raises:
            VoronoiTessellationError: If processing fails
        """
        # Extract positions
        if not all(col in galaxy_catalog.columns for col in ['x', 'y', 'z']):
            raise VoronoiTessellationError("Galaxy catalog must have x, y, z columns")
        
        positions = galaxy_catalog[['x', 'y', 'z']].values
        
        # Compute Voronoi tessellation
        voronoi = self.compute_voronoi_cells(positions)
        
        # Compute cell volumes
        volumes = self.compute_cell_volumes(voronoi, positions)
        
        # Compute density field
        densities = self.compute_density_field(volumes)
        
        return {
            'voronoi': voronoi,
            'volumes': volumes,
            'densities': densities,
            'positions': positions,
            'n_galaxies': len(positions)
        }

