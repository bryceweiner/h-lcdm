"""
H-ZOBOV Core Algorithm
======================

Core implementation of the H-ZOBOV void-finding algorithm integrating
all stages: Voronoi tessellation, density estimation, watershed zones,
and zone merging with H-ΛCDM Lambda(z) integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
import logging

from .zobov_parameters import HZOBOVParameters
from .voronoi_tessellation import VoronoiTessellation, VoronoiTessellationError
from .watershed import WatershedZoneFinder, WatershedError
from .zone_merger import ZoneMerger, ZoneMergerError

logger = logging.getLogger(__name__)


class ZOBOVCoreError(Exception):
    """Error in ZOBOV core algorithm."""
    pass


class ZOBOVCore:
    """
    Core H-ZOBOV algorithm implementation.
    
    Integrates all stages:
    1. Voronoi tessellation and DTFE density estimation
    2. Watershed zone finding
    3. Zone merging with Λ(z)-dependent thresholds
    4. Void catalog generation
    """
    
    def __init__(self, parameters: HZOBOVParameters):
        """
        Initialize ZOBOV core.
        
        Parameters:
            parameters: H-ZOBOV parameters
        """
        self.parameters = parameters
        parameters.validate()
        
        # Initialize components
        self.voronoi = VoronoiTessellation(parameters, use_mps=True)
        self.watershed = WatershedZoneFinder(use_mps=True)
        self.merger = ZoneMerger(parameters)
    
    def run_stage_voronoi(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """
        Run Voronoi tessellation stage.
        
        Parameters:
            galaxy_catalog: DataFrame with x, y, z columns
            
        Returns:
            Dictionary with Voronoi results
        """
        try:
            logger.info("=" * 80)
            logger.info("STAGE 1: Voronoi Tessellation and DTFE")
            logger.info("=" * 80)
            
            result = self.voronoi.process(galaxy_catalog)
            
            logger.info(f"✓ Stage 1 complete: {result['n_galaxies']:,} galaxies processed")
            return result
            
        except VoronoiTessellationError as e:
            raise ZOBOVCoreError(f"Voronoi tessellation failed: {e}") from e
    
    def run_stage_watershed(self, voronoi_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Run watershed zone finding stage.
        
        Parameters:
            voronoi_data: Output from Voronoi stage
            
        Returns:
            Dictionary with watershed results
        """
        try:
            logger.info("=" * 80)
            logger.info("STAGE 2: Watershed Zone Finding")
            logger.info("=" * 80)
            
            densities = voronoi_data['densities']
            positions = voronoi_data['positions']
            
            result = self.watershed.process(densities, positions)
            
            logger.info(f"✓ Stage 2 complete: {result['n_zones']:,} zones found")
            return result
            
        except WatershedError as e:
            raise ZOBOVCoreError(f"Watershed zone finding failed: {e}") from e
    
    def run_stage_merging(self, voronoi_data: Dict[str, Any],
                         watershed_data: Dict[str, Any],
                         galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """
        Run zone merging stage.
        
        Parameters:
            voronoi_data: Output from Voronoi stage
            watershed_data: Output from watershed stage
            galaxy_catalog: Original galaxy catalog (for redshifts)
            
        Returns:
            Dictionary with merging results
        """
        try:
            logger.info("=" * 80)
            logger.info("STAGE 3: Zone Merging with H-ΛCDM Lambda(z)")
            logger.info("=" * 80)
            
            densities = voronoi_data['densities']
            volumes = voronoi_data['volumes']
            positions = voronoi_data['positions']
            
            # Extract redshifts if available
            redshifts = None
            if 'redshift' in galaxy_catalog.columns:
                redshifts = galaxy_catalog['redshift'].values
            elif 'z' in galaxy_catalog.columns:
                # Check if z is redshift or comoving coordinate
                z_values = galaxy_catalog['z'].values
                if np.all((z_values > 0) & (z_values < 10)):
                    redshifts = z_values
            
            result = self.merger.process(
                watershed_data, densities, volumes, positions, redshifts
            )
            
            logger.info(f"✓ Stage 3 complete: {result['n_voids']:,} voids formed")
            return result
            
        except ZoneMergerError as e:
            raise ZOBOVCoreError(f"Zone merging failed: {e}") from e
    
    def generate_void_catalog(self, voronoi_data: Dict[str, Any],
                            watershed_data: Dict[str, Any],
                            merge_data: Dict[str, Any],
                            galaxy_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Generate final void catalog from all stages.
        
        Parameters:
            voronoi_data: Voronoi stage results
            watershed_data: Watershed stage results
            merge_data: Merging stage results
            galaxy_catalog: Original galaxy catalog
            
        Returns:
            DataFrame with void catalog
        """
        logger.info("=" * 80)
        logger.info("STAGE 4: Void Catalog Generation")
        logger.info("=" * 80)
        
        void_ids = merge_data['void_ids']
        positions = voronoi_data['positions']
        volumes = voronoi_data['volumes']
        densities = voronoi_data['densities']
        
        unique_voids = np.unique(void_ids)
        n_voids = len(unique_voids)
        
        logger.info(f"Generating catalog for {n_voids:,} voids...")
        
        void_records = []
        
        # Extract redshifts if available
        redshifts = None
        if 'redshift' in galaxy_catalog.columns:
            redshifts = galaxy_catalog['redshift'].values
        elif 'z' in galaxy_catalog.columns:
            z_values = galaxy_catalog['z'].values
            if np.all((z_values > 0) & (z_values < 10)):
                redshifts = z_values
        
        for void_id in unique_voids:
            mask = void_ids == void_id
            void_particles = positions[mask]
            void_volumes = volumes[mask]
            void_densities = densities[mask]
            
            # Compute void properties
            center = np.mean(void_particles, axis=0)
            total_volume = np.sum(void_volumes)
            
            # Effective radius (sphere with same volume)
            effective_radius = (3.0 * total_volume / (4.0 * np.pi))**(1.0/3.0)
            
            # Mean density
            mean_density = np.mean(void_densities)
            
            # Effective redshift
            effective_z = 0.0
            if redshifts is not None:
                effective_z = np.mean(redshifts[mask])
            
            # Compute Lambda(z) if H-ΛCDM enabled
            lambda_z = None
            if self.parameters.use_hlcdm_lambda:
                try:
                    from .hlcdm_integration import get_lambda_at_redshift
                    lambda_z = get_lambda_at_redshift(effective_z)
                except Exception as e:
                    logger.warning(f"Failed to compute Λ(z) for void {void_id}: {e}")
            
            void_records.append({
                'void_id': int(void_id),
                'x': float(center[0]),
                'y': float(center[1]),
                'z': float(center[2]),
                'radius_mpc': float(effective_radius),
                'volume_mpc3': float(total_volume),
                'mean_density': float(mean_density),
                'n_particles': int(np.sum(mask)),
                'redshift': float(effective_z),
                'lambda_z': float(lambda_z) if lambda_z is not None else None,
            })
        
        void_catalog = pd.DataFrame(void_records)
        
        # Filter by minimum volume if specified
        if self.parameters.min_void_volume is not None:
            before = len(void_catalog)
            void_catalog = void_catalog[void_catalog['volume_mpc3'] >= self.parameters.min_void_volume]
            logger.info(f"Filtered voids: {before} -> {len(void_catalog)} (min_volume={self.parameters.min_void_volume} Mpc³)")
        
        logger.info(f"✓ Stage 4 complete: {len(void_catalog):,} voids in catalog")
        
        return void_catalog
    
    def process(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """
        Run complete H-ZOBOV algorithm.
        
        Parameters:
            galaxy_catalog: DataFrame with x, y, z columns (and optionally redshift)
            
        Returns:
            Dictionary with:
            - 'void_catalog': DataFrame with void catalog
            - 'voronoi_data': Voronoi stage results
            - 'watershed_data': Watershed stage results
            - 'merge_data': Merging stage results
            - 'n_voids': Number of voids found
        """
        logger.info("=" * 80)
        logger.info("H-ZOBOV: Starting Complete Algorithm")
        logger.info("=" * 80)
        
        # Stage 1: Voronoi tessellation
        voronoi_data = self.run_stage_voronoi(galaxy_catalog)
        
        # Stage 2: Watershed zones
        watershed_data = self.run_stage_watershed(voronoi_data)
        
        # Stage 3: Zone merging
        merge_data = self.run_stage_merging(voronoi_data, watershed_data, galaxy_catalog)
        
        # Stage 4: Void catalog
        void_catalog = self.generate_void_catalog(
            voronoi_data, watershed_data, merge_data, galaxy_catalog
        )
        
        logger.info("=" * 80)
        logger.info(f"H-ZOBOV Complete: {len(void_catalog):,} voids found")
        logger.info("=" * 80)
        
        return {
            'void_catalog': void_catalog,
            'voronoi_data': voronoi_data,
            'watershed_data': watershed_data,
            'merge_data': merge_data,
            'n_voids': len(void_catalog)
        }

