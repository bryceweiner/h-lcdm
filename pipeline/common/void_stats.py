"""
Void Statistics and Metrics
===========================

Common statistical functions for cosmic void analysis, providing a single source of truth
for linking lengths, density estimations, and volume calculations.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Union, Tuple, List
import logging

logger = logging.getLogger(__name__)

def calculate_effective_volume(
    positions: np.ndarray, 
    method: str = 'convex_hull'
) -> float:
    """
    Calculate the effective volume occupied by a set of points.
    
    Parameters:
        positions: Array of shape (N, 3) containing x, y, z coordinates.
        method: Method to estimate volume ('bounding_box', 'convex_hull').
        
    Returns:
        float: Volume in same units cubed as input positions.
    """
    if len(positions) < 4:
        return 0.0
        
    if method == 'convex_hull':
        try:
            from scipy.spatial import ConvexHull
            hull = ConvexHull(positions)
            return hull.volume
        except ImportError:
            logger.warning("scipy.spatial.ConvexHull not available, falling back to bounding_box")
            return calculate_effective_volume(positions, method='bounding_box')
        except Exception as e:
            # ConvexHull can fail for coplanar points etc.
            logger.debug(f"ConvexHull failed ({e}), falling back to bounding_box")
            return calculate_effective_volume(positions, method='bounding_box')
            
    elif method == 'bounding_box':
        min_coords = np.min(positions, axis=0)
        max_coords = np.max(positions, axis=0)
        # Add small buffer to avoid zero volume for planar distributions
        ranges = max_coords - min_coords
        # If any dimension is 0 (e.g. 2D data), assume unit thickness or use min non-zero range
        if np.any(ranges == 0):
            non_zero = ranges[ranges > 0]
            thickness = np.min(non_zero) * 0.1 if len(non_zero) > 0 else 1.0
            ranges[ranges == 0] = thickness
            
        return np.prod(ranges)
    
    else:
        raise ValueError(f"Unknown volume estimation method: {method}")

def calculate_mean_separation(
    catalog: Union[pd.DataFrame, np.ndarray],
    volume: Optional[float] = None
) -> float:
    """
    Calculate the mean separation between objects (n^-1/3).
    
    This provides a robust, parameter-free scale for the system that is
    independent of void radius definitions.
    
    Parameters:
        catalog: DataFrame with 'x','y','z' or array of positions.
        volume: Optional pre-calculated volume. If None, estimated from positions.
        
    Returns:
        float: Mean separation distance.
    """
    if isinstance(catalog, pd.DataFrame):
        # Check for x, y, z columns
        if all(col in catalog.columns for col in ['x', 'y', 'z']):
            positions = catalog[['x', 'y', 'z']].values
        else:
            # Fallback or error? For robustness, return 0.0 or raise
            # If this is part of a pipeline, we usually want to propagate errors cleanly
            # But if called directly, raising is better.
            # Let's try to extract if possible, otherwise raise
            raise ValueError("Catalog must have x, y, z columns for mean separation calculation")
    else:
        positions = catalog
        
    n_objects = len(positions)
    if n_objects == 0:
        return 0.0
        
    if volume is None:
        volume = calculate_effective_volume(positions)
        
    if volume is None or np.isnan(volume) or np.isinf(volume) or volume <= 0:
        logger.warning(f"Invalid volume ({volume}) for mean separation calculation, returning 0.0")
        return 0.0
        
    number_density = n_objects / volume
    if number_density <= 0:
        logger.warning(f"Invalid number density ({number_density}) for mean separation calculation, returning 0.0")
        return 0.0
    
    mean_sep = number_density ** (-1/3)
    if np.isnan(mean_sep) or np.isinf(mean_sep):
        logger.warning(f"Mean separation calculation produced {mean_sep}, returning 0.0")
        return 0.0
    
    return mean_sep

def calculate_robust_linking_length(
    catalog: Union[pd.DataFrame, np.ndarray],
    method: str = 'robust',
    density_factor: float = 1.5,
    radius_factor: float = 3.0,
    radius_col: str = 'radius_mpc'
) -> Tuple[float, Dict[str, Any]]:
    """
    Calculate a statistically valid linking length for network construction.
    
    This acts as the single source of truth for how void networks are built.
    
    Methods:
    - 'density': Uses mean particle separation (n^-1/3) * density_factor. 
                 Robust to radius errors. Good for uniform samples.
    - 'radius': Uses mean void radius * radius_factor. 
                Standard in some literature (e.g. 3*Reff). Sensitive to radius definitions.
    - 'robust': (Default) Calculates both. Uses 'radius' unless it produces an 
                unphysically large value relative to the density scale (indicating unit errors),
                in which case it falls back to 'density'.
    
    Parameters:
        catalog: Input catalog (DataFrame or positions array).
        method: 'density', 'radius', or 'robust'.
        density_factor: Multiplier for mean separation (default: 1.5).
                        1.5 corresponds to percolation threshold for random spheres.
        radius_factor: Multiplier for mean radius (default: 3.0).
        radius_col: Column name for radius if using radius/robust method.
        
    Returns:
        tuple: (linking_length, metadata_dict)
    """
    metadata = {}
    positions = None
    
    # Extract positions and radii
    if isinstance(catalog, pd.DataFrame):
        if all(col in catalog.columns for col in ['x', 'y', 'z']):
            positions = catalog[['x', 'y', 'z']].values
        
        # Handle radius column aliases
        if radius_col not in catalog.columns:
            for alias in ['radius_Mpc', 'radius_eff', 'reff', 'radius']:
                if alias in catalog.columns:
                    radius_col = alias
                    break
                    
        has_radius = radius_col in catalog.columns
        valid_radii = catalog[radius_col].dropna().values if has_radius else np.array([])
    else:
        positions = catalog
        has_radius = False
        valid_radii = np.array([])

    if positions is None:
        raise ValueError("Could not extract positions from catalog")

    # 1. Calculate Density Scale (Baseline)
    mean_sep = calculate_mean_separation(positions)
    density_linking_length = mean_sep * density_factor if mean_sep > 0 else 0.0
    
    metadata['mean_separation'] = mean_sep
    metadata['density_linking_length'] = density_linking_length
    
    # 2. Calculate Radius Scale (If available)
    radius_linking_length = None
    if len(valid_radii) > 0:
        mean_radius = np.mean(valid_radii)
        radius_linking_length = mean_radius * radius_factor
        metadata['mean_radius'] = mean_radius
        metadata['radius_linking_length'] = radius_linking_length
    
    # 3. Determine Final Length
    final_length = 0.0
    selected_method = method
    
    if method == 'density':
        final_length = density_linking_length
        
    elif method == 'radius':
        if radius_linking_length is not None:
            final_length = radius_linking_length
        else:
            logger.warning("Radius method requested but no radii found. Falling back to density.")
            final_length = density_linking_length
            selected_method = 'density_fallback'
            
    elif method == 'robust':
        # Logic: 
        # If radius scale is wildly different from density scale (>10x or <0.1x),
        # it suggests a unit mismatch (e.g. Mpc vs Mpc/h vs pixel units) or bad definition.
        # In that case, trust the positions (density) over the radii.
        
        # Handle case where density linking length is invalid (0.0 or NaN)
        if density_linking_length <= 0 or np.isnan(density_linking_length) or np.isinf(density_linking_length):
            if radius_linking_length is not None and radius_linking_length > 0:
                logger.warning(f"Density linking length is invalid ({density_linking_length}), using radius-based linking length ({radius_linking_length:.2f})")
                final_length = radius_linking_length
                selected_method = 'radius_fallback'
            else:
                raise ValueError(f"Cannot calculate linking length: density scale is invalid ({density_linking_length}) and no valid radius scale available")
        
        elif radius_linking_length is not None:
            ratio = radius_linking_length / density_linking_length
            
            # Check for inflated radii (likely the current user issue)
            if ratio > 5.0:  # Relaxed from 10x to 5x to catch the 135 Mpc issue sooner
                logger.warning(f"Radius-based linking length ({radius_linking_length:.2f}) is >5x mean separation scale ({density_linking_length:.2f}). Likely radius unit error. Using density scale.")
                final_length = density_linking_length
                selected_method = 'density_robust_override'
                metadata['override_reason'] = 'radius_too_large'
            
            # Check for tiny radii (e.g. pixels interpreted as Mpc?)
            elif ratio < 0.1:
                logger.warning(f"Radius-based linking length ({radius_linking_length:.2f}) is <0.1x mean separation scale. Using density scale.")
                final_length = density_linking_length
                selected_method = 'density_robust_override'
                metadata['override_reason'] = 'radius_too_small'
                
            else:
                # Within reasonable bounds, prefer the physical radius definition
                final_length = radius_linking_length
                selected_method = 'radius'
        else:
            final_length = density_linking_length
            selected_method = 'density_fallback'
            
    metadata['final_method'] = selected_method
    metadata['ratio_radius_to_density'] = (radius_linking_length / density_linking_length) if (radius_linking_length and density_linking_length > 0) else None
    
    return final_length, metadata

