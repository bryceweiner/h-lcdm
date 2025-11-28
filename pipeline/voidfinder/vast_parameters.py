"""
VAST VoidFinder Parameter Configuration
======================================

Catalog-specific parameterization for VAST VoidFinder.
Each catalog requires different parameters based on its systematics.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from astropy.cosmology import Planck18 as cosmo
from astropy import units as u
import logging

logger = logging.getLogger(__name__)


class VASTParameterConfig:
    """
    Configures VAST VoidFinder parameters for specific galaxy catalogs.
    
    Calculates appropriate parameters based on catalog systematics:
    - Survey geometry and boundaries
    - Redshift/distance ranges
    - Galaxy density
    - Expected void sizes
    """
    
    def __init__(self, catalog_name: str):
        """
        Initialize parameter config for a catalog.
        
        Parameters:
            catalog_name: Name of the catalog (e.g., 'sdss_dr16')
        """
        self.catalog_name = catalog_name
        self.configs = self._get_default_configs()
    
    def _get_default_configs(self) -> Dict[str, Dict[str, Any]]:
        """Get default parameter configurations for known catalogs."""
        return {
            'sdss_dr16': {
                'survey_name': 'SDSS_DR16',
                'mask_type': 'xyz',  # Use xyz mode to avoid needing angular mask
                'min_maximal_radius': 10.0,  # Mpc - typical minimum void size
                'hole_grid_edge_length': 5.0,  # Mpc - grid cell size
                'pts_per_unit_volume': 0.01,  # Points per Mpc^3 for grid
                'max_hole_mask_overlap': 0.1,  # Maximum overlap with survey edge
                'check_only_empty_cells': True,
                'verbose': 1,
                'num_cpus': None,  # Use all available CPUs
            },
            'sdss_dr7': {
                'survey_name': 'SDSS_DR7',
                'mask_type': 'xyz',
                'min_maximal_radius': 10.0,
                'hole_grid_edge_length': 5.0,
                'pts_per_unit_volume': 0.01,
                'max_hole_mask_overlap': 0.1,
                'check_only_empty_cells': True,
                'verbose': 1,
            }
        }
    
    def compute_parameters(self, galaxy_catalog: pd.DataFrame,
                          z_min: float, z_max: float,
                          grid_size: Optional[float] = None) -> Dict[str, Any]:
        """
        Compute VAST parameters from catalog properties.
        
        Parameters:
            galaxy_catalog: DataFrame with x, y, z columns (comoving Mpc)
            z_min: Minimum redshift
            z_max: Maximum redshift
            grid_size: Grid edge length in Mpc (if None, uses default from config)
            
        Returns:
            dict: VAST parameters ready to pass to find_voids()
        """
        # Start with catalog-specific defaults
        params = self.configs.get(self.catalog_name, {}).copy()
        
        # Ensure we have required columns
        if not all(col in galaxy_catalog.columns for col in ['x', 'y', 'z']):
            raise ValueError("Galaxy catalog must have x, y, z columns")
        
        # Compute distance limits from redshift range
        dist_min = cosmo.comoving_distance(z_min).value  # Mpc
        dist_max = cosmo.comoving_distance(z_max).value  # Mpc
        params['dist_limits'] = [dist_min, dist_max]
        
        logger.info(f"Distance limits: {dist_min:.1f} - {dist_max:.1f} Mpc (z={z_min:.3f}-{z_max:.3f})")
        
        # Compute xyz limits from actual galaxy distribution
        # Add padding to avoid edge effects
        padding = 50.0  # Mpc padding
        x_min, x_max = galaxy_catalog['x'].min() - padding, galaxy_catalog['x'].max() + padding
        y_min, y_max = galaxy_catalog['y'].min() - padding, galaxy_catalog['y'].max() + padding
        z_min_xyz, z_max_xyz = galaxy_catalog['z'].min() - padding, galaxy_catalog['z'].max() + padding
        
        # VAST expects shape (2, 3): [[xmin, ymin, zmin], [xmax, ymax, zmax]]
        params['xyz_limits'] = np.array([
            [x_min, y_min, z_min_xyz],
            [x_max, y_max, z_max_xyz]
        ])
        
        logger.info(f"XYZ limits: x=[{x_min:.1f}, {x_max:.1f}], y=[{y_min:.1f}, {y_max:.1f}], z=[{z_min_xyz:.1f}, {z_max_xyz:.1f}] Mpc")
        
        # Compute optimal grid resolution based on galaxy density
        # Higher density = smaller grid cells needed
        n_galaxies = len(galaxy_catalog)
        
        # Estimate expected void count for validation
        # Use correct volume calc for shape (2, 3)
        if isinstance(params['xyz_limits'], np.ndarray):
            lims = params['xyz_limits']
            volume = (lims[1, 0] - lims[0, 0]) * (lims[1, 1] - lims[0, 1]) * (lims[1, 2] - lims[0, 2])
        else:
            # Fallback if not numpy array (shouldn't happen with current code)
            volume = 1.0
            
        density = n_galaxies / volume if volume > 0 else 0
        
        logger.info(f"Galaxy density: {density:.6f} galaxies/Mpc^3 ({n_galaxies:,} galaxies in {volume:.1f} Mpc^3)")
        
        # Set grid edge length (use provided grid_size or default from config)
        if grid_size is not None:
            params['hole_grid_edge_length'] = grid_size
        elif 'hole_grid_edge_length' not in params:
            # Fallback to default if not in config
            params['hole_grid_edge_length'] = 50.0
        
        logger.info(f"Grid edge length: {params['hole_grid_edge_length']:.1f} Mpc")
        
        # Set grid origin to align with galaxy distribution
        # This helps with numerical precision
        # Convert to Python floats (VAST may not handle numpy types well)
        params['grid_origin'] = [float(x_min), float(y_min), float(z_min_xyz)]
        
        # Adjust galaxy_map_grid_edge_length if not set
        # This is the secondary grid for neighbor finding
        # Should be smaller than hole_grid for efficiency
        if 'galaxy_map_grid_edge_length' not in params or params['galaxy_map_grid_edge_length'] is None:
            params['galaxy_map_grid_edge_length'] = params['hole_grid_edge_length'] * 0.5
        
        # Set survey name if not already set
        if 'survey_name' not in params:
            params['survey_name'] = self.catalog_name.upper()
        
        return params
    
    def get_survey_mask(self, galaxy_catalog: pd.DataFrame,
                       z_min: float, z_max: float) -> Optional[Any]:
        """
        Generate survey mask for VAST if needed.
        
        Parameters:
            galaxy_catalog: DataFrame with ra, dec, z columns
            z_min: Minimum redshift
            z_max: Maximum redshift
            
        Returns:
            Mask object or None if not needed
        """
        # For SDSS DR16, we might want to create a mask based on:
        # - Survey footprint (RA/Dec bounds)
        # - Redshift completeness
        # - Magnitude limits
        
        # For now, return None - VAST can work without explicit mask
        # The dist_limits and xyz_limits provide sufficient boundaries
        return None
    
    @staticmethod
    def estimate_void_count(n_galaxies: int, volume: float) -> int:
        """
        Estimate expected number of voids based on galaxy count and volume.
        
        Based on SDSS DR7 results:
        - SDSS DR7 found ~1000-2000 voids with ~600k galaxies
        - Typical void-to-galaxy ratio is ~1:300 to 1:600
        
        Parameters:
            n_galaxies: Number of galaxies
            volume: Survey volume in Mpc^3 (should be actual survey volume, not padded)
            
        Returns:
            Estimated number of voids
        """
        # Conservative estimate: ~1 void per 300-500 galaxies
        # SDSS DR7: ~600k galaxies -> ~1000-2000 voids = 1:300 to 1:600 ratio
        void_galaxy_ratio = 1.0 / 400.0  # Conservative middle estimate
        estimated_voids = int(n_galaxies * void_galaxy_ratio)
        
        # Cap at reasonable maximum based on volume
        # Typical void spacing is ~50-100 Mpc, so void density is ~0.0001-0.0005 voids/Mpc^3
        # But we use a more conservative estimate
        if volume > 0:
            # Use actual survey volume (not padded) - assume padding is ~5% of volume
            actual_volume = volume * 0.95
            volume_based = int(actual_volume * 0.0002)  # ~0.0002 voids/Mpc^3
            # Use the smaller estimate to be conservative
            estimated_voids = min(estimated_voids, volume_based) if volume_based > 0 else estimated_voids
        
        return max(estimated_voids, 1)  # At least 1 void expected
