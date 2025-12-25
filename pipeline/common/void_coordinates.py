"""
Void Coordinate Conversion Utilities
====================================

Single source of truth for converting between spherical (ra, dec, z) and 
Cartesian (x, y, z) coordinate systems for cosmic voids.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Any
import logging
import warnings

logger = logging.getLogger(__name__)

def extract_cartesian_coordinates(catalog: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Extract Cartesian coordinates from catalog if available.
    
    Checks for x, y, z columns first, then falls back to x_mpc, y_mpc, z_mpc.
    Returns None if coordinates are missing or invalid, allowing fallback to conversion.
    
    Parameters:
        catalog: DataFrame with potential x, y, z or x_mpc, y_mpc, z_mpc columns
        
    Returns:
        Array of shape (N, 3) with x, y, z coordinates, or None if not available or invalid
    """
    # Try x, y, z columns first
    if all(col in catalog.columns for col in ['x', 'y', 'z']):
        coords = catalog[['x', 'y', 'z']].values
        valid_mask = np.isfinite(coords).all(axis=1)
        n_valid = valid_mask.sum()
        
        # If we have mostly valid coordinates (>= 10%), return them
        # The caller will filter out invalid rows
        if n_valid >= len(coords) * 0.1:
            return coords
    
    # Try x_mpc, y_mpc, z_mpc columns (used by DESI catalogs)
    if all(col in catalog.columns for col in ['x_mpc', 'y_mpc', 'z_mpc']):
        coords = catalog[['x_mpc', 'y_mpc', 'z_mpc']].values
        valid_mask = np.isfinite(coords).all(axis=1)
        n_valid = valid_mask.sum()
        
        # If we have mostly valid coordinates (>= 10%), return them
        # The caller will filter out invalid rows
        if n_valid >= len(coords) * 0.1:
            return coords
    
    return None

def convert_spherical_to_cartesian(
    catalog: pd.DataFrame,
    cosmology: Optional[Any] = None
) -> np.ndarray:
    """
    Convert spherical coordinates (ra, dec, redshift) to Cartesian (x, y, z) in Mpc.
    
    Parameters:
        catalog: DataFrame with ra/ra_deg, dec/dec_deg, redshift columns
        cosmology: Astropy cosmology object (default: Planck18)
        
    Returns:
        Array of shape (N, 3) with x, y, z coordinates in Mpc
    """
    # Handle column name variations
    ra_col = 'ra' if 'ra' in catalog.columns else ('ra_deg' if 'ra_deg' in catalog.columns else None)
    dec_col = 'dec' if 'dec' in catalog.columns else ('dec_deg' if 'dec_deg' in catalog.columns else None)
    
    if ra_col is None or dec_col is None or 'redshift' not in catalog.columns:
        missing_cols = []
        if ra_col is None:
            missing_cols.extend(['ra', 'ra_deg'])
        if dec_col is None:
            missing_cols.extend(['dec', 'dec_deg'])
        if 'redshift' not in catalog.columns:
            missing_cols.append('redshift')
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        if cosmology is None:
            from astropy.cosmology import Planck18 as cosmology
        
        # Filter out rows with invalid input data BEFORE conversion
        ra_vals = catalog[ra_col].values
        dec_vals = catalog[dec_col].values
        z_vals = catalog['redshift'].values
        
        # Create mask for valid rows
        valid_mask = (
            np.isfinite(ra_vals) & 
            np.isfinite(dec_vals) & 
            np.isfinite(z_vals) &
            (z_vals >= 0) & 
            (z_vals <= 10)
        )
        
        n_invalid_input = (~valid_mask).sum()
        if n_invalid_input > 0:
            logger.warning(f"Filtering out {n_invalid_input} rows with invalid input coordinates (NaN/inf or z out of range)")
            # Filter catalog to valid rows only
            catalog = catalog[valid_mask].copy()
            ra_vals = catalog[ra_col].values
            dec_vals = catalog[dec_col].values
            z_vals = catalog['redshift'].values
        
        if len(catalog) == 0:
            raise ValueError("No valid rows remaining after filtering invalid coordinates")
        
        # Convert to SkyCoord and then to Cartesian
        coords = SkyCoord(
            ra=ra_vals * u.deg,
            dec=dec_vals * u.deg,
            distance=cosmology.comoving_distance(z_vals)
        )
        
        # Get Cartesian coordinates in Mpc
        cart_coords = coords.cartesian
        
        # Extract values
        positions = np.column_stack([
            cart_coords.x.value,
            cart_coords.y.value,
            cart_coords.z.value
        ])
        
        # Final check - this should not happen if input was valid, but check anyway
        if not np.isfinite(positions).all():
            n_invalid = (~np.isfinite(positions).all(axis=1)).sum()
            invalid_inputs = catalog[~np.isfinite(positions).all(axis=1)][[ra_col, dec_col, 'redshift']].head(10)
            raise ValueError(
                f"CRITICAL ERROR: Coordinate conversion produced {n_invalid} rows with NaN/inf values despite valid input. "
                f"First invalid inputs:\n{invalid_inputs}\n"
                f"This indicates a problem with the cosmology calculation."
            )
        
        return positions
        
    except ImportError:
        # Fallback: simplified approximation (not physically accurate)
        warnings.warn("astropy not available, using simplified coordinate transformation")
        logger.warning("Using simplified coordinate transformation (astropy not available)")
        
        ra_rad = np.radians(catalog[ra_col].values)
        dec_rad = np.radians(catalog[dec_col].values)
        # Approximate comoving distance (rough approximation)
        r_comov = catalog['redshift'].values * 3000.0  # Mpc/h approximation
        
        return np.column_stack([
            r_comov * np.cos(dec_rad) * np.cos(ra_rad),
            r_comov * np.cos(dec_rad) * np.sin(ra_rad),
            r_comov * np.sin(dec_rad)
        ])

def convert_spherical_to_cartesian_chunked(
    catalog: pd.DataFrame,
    chunk_size: int = 50000,
    show_progress: bool = True,
    cosmology: Optional[Any] = None,
    preserve_redshift: bool = True
) -> pd.DataFrame:
    """
    Convert spherical coordinates to Cartesian with chunking for large catalogs.
    
    This function handles large datasets efficiently by processing in chunks
    and optionally showing progress. It preserves the redshift column before
    overwriting 'z' with the comoving z-coordinate.
    
    Parameters:
        catalog: DataFrame with ra/ra_deg, dec/dec_deg, z/redshift columns
        chunk_size: Number of galaxies to process per chunk (default: 50000)
        show_progress: Whether to show progress bar (default: True)
        cosmology: Astropy cosmology object (default: Planck18)
        preserve_redshift: If True, preserve 'z' as 'redshift' before overwriting (default: True)
        
    Returns:
        DataFrame with added x, y, z columns (comoving coordinates in Mpc)
    """
    df = catalog.copy()
    
    # Check if already has comoving coordinates
    if all(col in df.columns for col in ['x', 'y', 'z']):
        return df
    
    # Handle column name variations
    ra_col = 'ra' if 'ra' in df.columns else ('ra_deg' if 'ra_deg' in df.columns else None)
    dec_col = 'dec' if 'dec' in df.columns else ('dec_deg' if 'dec_deg' in df.columns else None)
    z_col = 'z' if 'z' in df.columns else ('redshift' if 'redshift' in df.columns else None)
    
    if ra_col is None or dec_col is None or z_col is None:
        missing_cols = []
        if ra_col is None:
            missing_cols.extend(['ra', 'ra_deg'])
        if dec_col is None:
            missing_cols.extend(['dec', 'dec_deg'])
        if z_col is None:
            missing_cols.extend(['z', 'redshift'])
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    try:
        from astropy import units as u
        from astropy.coordinates import SkyCoord
        if cosmology is None:
            from astropy.cosmology import Planck18 as cosmology
        
        n_galaxies = len(df)
        
        # Preserve redshift column before overwriting 'z'
        if preserve_redshift and z_col == 'z' and 'redshift' not in df.columns:
            df['redshift'] = df['z'].copy()
        
        # Process in chunks for large catalogs
        if n_galaxies > 10000:
            if show_progress:
                logger.info(f"Converting {n_galaxies:,} galaxies to comoving coordinates...")
                try:
                    from tqdm import tqdm
                    # Create logger writer for progress bar
                    class LoggerWriter:
                        """File-like object that writes to logger."""
                        def __init__(self, logger, level=logging.INFO):
                            self.logger = logger
                            self.level = level
                            self.buf = ''
                        
                        def write(self, s):
                            self.buf = s.strip('\r\n\t ')
                            if self.buf:
                                self.logger.log(self.level, self.buf)
                        
                        def flush(self):
                            pass
                    
                    coord_logger_writer = LoggerWriter(logger)
                    iterator = tqdm(range(0, n_galaxies, chunk_size), 
                                  desc="Converting coordinates", 
                                  unit="chunk", 
                                  file=coord_logger_writer, 
                                  leave=False)
                except ImportError:
                    iterator = range(0, n_galaxies, chunk_size)
            else:
                iterator = range(0, n_galaxies, chunk_size)
            
            coords_list = []
            for i in iterator:
                end_idx = min(i + chunk_size, n_galaxies)
                chunk_df = df.iloc[i:end_idx]
                
                chunk_coords = SkyCoord(
                    ra=chunk_df[ra_col].values * u.deg,
                    dec=chunk_df[dec_col].values * u.deg,
                    distance=cosmology.comoving_distance(chunk_df[z_col].values)
                )
                
                chunk_cart = chunk_coords.cartesian
                coords_list.append({
                    'x': chunk_cart.x.value,
                    'y': chunk_cart.y.value,
                    'z': chunk_cart.z.value
                })
            
            # Combine results
            df['x'] = np.concatenate([c['x'] for c in coords_list])
            df['y'] = np.concatenate([c['y'] for c in coords_list])
            df['z'] = np.concatenate([c['z'] for c in coords_list])  # Comoving z coordinate
        else:
            # Small catalog - process all at once
            coords = SkyCoord(
                ra=df[ra_col].values * u.deg,
                dec=df[dec_col].values * u.deg,
                distance=cosmology.comoving_distance(df[z_col].values)
            )
            
            cart = coords.cartesian
            df['x'] = cart.x.value
            df['y'] = cart.y.value
            df['z'] = cart.z.value  # Comoving z coordinate
        
        return df
        
    except ImportError:
        # Fallback: use non-chunked version
        warnings.warn("astropy not available, using simplified coordinate transformation")
        logger.warning("Using simplified coordinate transformation (astropy not available)")
        
        # Use the basic conversion function
        positions = convert_spherical_to_cartesian(df, cosmology=cosmology)
        df['x'] = positions[:, 0]
        df['y'] = positions[:, 1]
        df['z'] = positions[:, 2]
        
        return df

def get_cartesian_positions(catalog: pd.DataFrame) -> Tuple[np.ndarray, bool]:
    """
    Get Cartesian positions from catalog, converting if necessary.
    
    Parameters:
        catalog: DataFrame with either (x, y, z) or (ra, dec, z) columns
        
    Returns:
        tuple: (positions_array, was_converted)
            positions_array: Array of shape (N, 3) with x, y, z coordinates
            was_converted: True if conversion from spherical was performed
    """
    # Try to extract existing Cartesian coordinates first
    positions = extract_cartesian_coordinates(catalog)
    
    if positions is not None:
        return positions, False
    
    # Convert from spherical if needed
    positions = convert_spherical_to_cartesian(catalog)
    return positions, True

