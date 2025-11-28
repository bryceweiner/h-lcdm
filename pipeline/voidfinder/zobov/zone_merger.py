"""
Zone Merging and Void Formation
================================

Implements zone merging to form voids from watershed zones.
H-ZOBOV enhancement: uses redshift-dependent Λ(z) for significance thresholds.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List, Tuple
from scipy.spatial import cKDTree
from tqdm import tqdm
import logging

from .zobov_parameters import HZOBOVParameters
from .hlcdm_integration import get_lambda_at_redshift

logger = logging.getLogger(__name__)


class ZoneMergerError(Exception):
    """Error in zone merging."""
    pass


class ZoneMerger:
    """
    Zone merging algorithm for void formation.
    
    Zones are merged based on density ratios. H-ZOBOV uses Λ(z)-dependent
    significance thresholds when enabled.
    """
    
    def __init__(self, parameters: HZOBOVParameters):
        """
        Initialize zone merger.
        
        Parameters:
            parameters: H-ZOBOV parameters
        """
        self.parameters = parameters
        self.use_hlcdm_lambda = parameters.use_hlcdm_lambda
    
    def find_adjacent_zones(self, positions: np.ndarray,
                          zone_ids: np.ndarray,
                          k_neighbors: int = 20) -> List[Tuple[int, int]]:
        """
        Find pairs of adjacent zones.
        
        Zones are adjacent if they have particles that are neighbors.
        
        Parameters:
            positions: Particle positions
            zone_ids: Zone assignment for each particle
            k_neighbors: Number of neighbors to check
            
        Returns:
            List of (zone1, zone2) tuples for adjacent zones
        """
        from scipy.spatial import cKDTree
        
        logger.info("Finding adjacent zones...")
        
        tree = cKDTree(positions)
        n_particles = len(positions)
        n_zones = len(np.unique(zone_ids))
        
        # Constrain k_neighbors to available neighbors
        max_k = min(k_neighbors, n_particles - 1)
        if max_k < 1:
            logger.warning(f"Insufficient particles ({n_particles}) for adjacency detection")
            return []
        
        # Build set of adjacent zone pairs
        adjacent_pairs = set()
        
        for i in range(n_particles):
            zone_i = zone_ids[i]
            
            # Query up to n_particles neighbors (all available)
            k_request = min(max_k + 1, n_particles)
            result = tree.query(positions[i], k=k_request)
            
            # Skip if only self is returned
            if k_request == 1:
                continue
            
            distances, indices = result
            indices = np.atleast_1d(indices)
            
            # Validate indices are within array bounds
            valid_indices = indices[indices < n_particles]
            
            # Check neighbors (exclude self, which is first index)
            if len(valid_indices) > 1:
                neighbor_indices = valid_indices[1:]
                for neighbor_idx in neighbor_indices:
                    if neighbor_idx < n_particles:
                        zone_j = zone_ids[neighbor_idx]
                        if zone_i != zone_j:
                            # Store pairs with consistent ordering (min, max)
                            pair = (min(zone_i, zone_j), max(zone_i, zone_j))
                            adjacent_pairs.add(pair)
        
        adjacent_list = list(adjacent_pairs)
        logger.info(f"✓ Found {len(adjacent_list):,} adjacent zone pairs")
        
        return adjacent_list
    
    def compute_zone_statistics(self, densities: np.ndarray,
                              zone_ids: np.ndarray,
                              volumes: np.ndarray) -> Dict[int, Dict[str, float]]:
        """
        Compute statistics for each zone.
        
        Parameters:
            densities: Density values
            zone_ids: Zone assignments
            volumes: Voronoi cell volumes
            
        Returns:
            Dictionary mapping zone_id to statistics dict
        """
        unique_zones = np.unique(zone_ids)
        zone_stats = {}
        
        for zone_id in unique_zones:
            mask = zone_ids == zone_id
            zone_densities = densities[mask]
            zone_volumes = volumes[mask]
            
            zone_stats[zone_id] = {
                'min_density': np.min(zone_densities),
                'mean_density': np.mean(zone_densities),
                'max_density': np.max(zone_densities),
                'total_volume': np.sum(zone_volumes),
                'n_particles': np.sum(mask)
            }
        
        return zone_stats
    
    def compute_significance_threshold(self, redshift: float) -> float:
        """
        Compute significance threshold using Λ(z) if H-ΛCDM is enabled.
        
        Parameters:
            redshift: Effective redshift for the void
            
        Returns:
            Significance threshold (density ratio)
        """
        if self.use_hlcdm_lambda:
            try:
                # Retrieve redshift-dependent cosmological constant Λ(z)
                lambda_z = get_lambda_at_redshift(redshift)
                
                # Convert Λ(z) to density contrast threshold
                # Higher Λ(z) corresponds to lower density contrast threshold
                # Note: This is a simplified scaling relation; full cosmological dependence
                # requires detailed modeling of void formation in H-ΛCDM
                from hlcdm.parameters import HLCDM_PARAMS
                lambda_0 = HLCDM_PARAMS.LAMBDA_OBS
                
                # Scale threshold inversely with Λ(z)/Λ₀ ratio
                lambda_ratio = lambda_z / lambda_0 if lambda_0 > 0 else 1.0
                base_threshold = 0.2
                threshold = base_threshold / lambda_ratio
                
                # Constrain threshold to physically reasonable range
                return max(0.1, min(1.0, threshold))
                
            except Exception as e:
                logger.warning(f"Failed to compute Λ(z) threshold at z={redshift}: {e}, using default")
                return 0.2
        else:
            # Standard ZOBOV parameter-free mode or user-specified threshold
            if self.parameters.significance_ratio is not None:
                return self.parameters.significance_ratio
            return 0.2
    
    def merge_zones(self, zone_ids: np.ndarray,
                   densities: np.ndarray,
                   volumes: np.ndarray,
                   positions: np.ndarray,
                   zone_stats: Dict[int, Dict[str, float]],
                   redshifts: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Merge zones to form voids.
        
        Zones are merged if density ratio r = ρ_link / ρ_min < threshold.
        H-ZOBOV uses redshift-dependent thresholds when enabled.
        
        Parameters:
            zone_ids: Zone assignments
            densities: Density values
            volumes: Cell volumes
            positions: Particle positions
            zone_stats: Zone statistics dictionary
            redshifts: Redshift for each particle (for Λ(z) calculation)
            
        Returns:
            Dictionary with merged void assignments
        """
        logger.info("Merging zones to form voids...")
        
        # Identify adjacent zone pairs for merging evaluation
        adjacent_pairs = self.find_adjacent_zones(positions, zone_ids)
        
        # Compute effective redshift for each zone (mean redshift of constituent particles)
        if redshifts is None:
            # Estimate redshifts from comoving distances (approximate)
            distances = np.linalg.norm(positions, axis=1)
            from astropy.cosmology import Planck18 as cosmo
            from astropy import units as u
            redshifts = np.array([cosmo.redshift_at_value(d * u.Mpc).value 
                                 for d in distances])
        
        zone_redshifts = {}
        unique_zones = np.unique(zone_ids)
        for zone_id in unique_zones:
            mask = zone_ids == zone_id
            zone_redshifts[zone_id] = np.mean(redshifts[mask]) if np.any(mask) else 0.0
        
        # Initialize void assignments: each zone is initially its own void
        void_ids = zone_ids.copy()
        void_counter = len(unique_zones)
        zone_to_void = {zone_id: zone_id for zone_id in unique_zones}
        
        # Evaluate adjacent zones for merging
        merged_count = 0
        
        with tqdm(total=len(adjacent_pairs), desc="Merging zones", unit="pair") as pbar:
            for zone1, zone2 in adjacent_pairs:
                stats1 = zone_stats[zone1]
                stats2 = zone_stats[zone2]
                
                # Identify zone with minimum density (void center)
                if stats1['min_density'] < stats2['min_density']:
                    min_zone, link_zone = zone1, zone2
                    min_density = stats1['min_density']
                    link_density = stats2['mean_density']
                else:
                    min_zone, link_zone = zone2, zone1
                    min_density = stats2['min_density']
                    link_density = stats1['mean_density']
                
                # Compute density ratio r = ρ_link / ρ_min
                density_ratio = link_density / min_density if min_density > 0 else np.inf
                
                # Retrieve redshift-dependent significance threshold (H-ZOBOV)
                effective_z = zone_redshifts[min_zone]
                threshold = self.compute_significance_threshold(effective_z)
                
                # Merge zones if density ratio below threshold
                if density_ratio < threshold:
                    void1 = zone_to_void[min_zone]
                    void2 = zone_to_void[link_zone]
                    
                    if void1 != void2:
                        # Merge voids: reassign all particles from source void to target void
                        target_void = min(void1, void2)
                        source_void = max(void1, void2)
                        
                        mask = void_ids == source_void
                        void_ids[mask] = target_void
                        
                        # Update zone-to-void mapping
                        for zone_id in zone_to_void:
                            if zone_to_void[zone_id] == source_void:
                                zone_to_void[zone_id] = target_void
                        
                        merged_count += 1
                
                pbar.update(1)
        
        n_voids = len(np.unique(void_ids))
        logger.info(f"✓ Zone merging complete: {merged_count:,} merges, {n_voids:,} voids")
        
        return {
            'void_ids': void_ids,
            'n_voids': n_voids,
            'n_merges': merged_count,
            'zone_to_void': zone_to_void
        }
    
    def process(self, zone_data: Dict[str, Any],
               densities: np.ndarray,
               volumes: np.ndarray,
               positions: np.ndarray,
               redshifts: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Complete zone merging process.
        
        Parameters:
            zone_data: Output from watershed zone finder
            densities: Density values
            volumes: Cell volumes
            positions: Particle positions
            redshifts: Redshift for each particle
            
        Returns:
            Dictionary with void assignments and metadata
        """
        zone_ids = zone_data['zone_ids']
        
        # Compute zone statistics
        zone_stats = self.compute_zone_statistics(densities, zone_ids, volumes)
        
        # Merge zones
        merge_data = self.merge_zones(
            zone_ids, densities, volumes, positions,
            zone_stats, redshifts
        )
        
        return {
            **merge_data,
            'zone_stats': zone_stats,
            'densities': densities,
            'volumes': volumes,
            'positions': positions
        }

