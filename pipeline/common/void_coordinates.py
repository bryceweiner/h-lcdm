"""
Void Coordinate Conversion Utilities
====================================

Single source of truth for converting between spherical (ra, dec, z) and 
Cartesian (x, y, z) coordinate systems for cosmic voids.
"""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, Any, Dict
import logging
import warnings

logger = logging.getLogger(__name__)

def extract_cartesian_coordinates(catalog: pd.DataFrame) -> Optional[np.ndarray]:
    """
    Extract Cartesian coordinates from catalog if available.
    
    Merges x, y, z and x_mpc, y_mpc, z_mpc columns, using whichever is valid for each row.
    Returns None if coordinates are missing or invalid, allowing fallback to conversion.
    
    Parameters:
        catalog: DataFrame with potential x, y, z or x_mpc, y_mpc, z_mpc columns
        
    Returns:
        Array of shape (N, 3) with x, y, z coordinates, or None if not available or invalid
    """
    has_xyz = all(col in catalog.columns for col in ['x', 'y', 'z'])
    has_xyz_mpc = all(col in catalog.columns for col in ['x_mpc', 'y_mpc', 'z_mpc'])
    
    if not has_xyz and not has_xyz_mpc:
        return None
    
    # Initialize coords array
    coords = np.full((len(catalog), 3), np.nan)
    
    # First, fill in x_mpc, y_mpc, z_mpc if available (DESI voids)
    if has_xyz_mpc:
        coords_mpc = catalog[['x_mpc', 'y_mpc', 'z_mpc']].values
        valid_mask_mpc = np.isfinite(coords_mpc).all(axis=1)
        coords[valid_mask_mpc] = coords_mpc[valid_mask_mpc]
    
    # Then, fill in x, y, z for rows that are still NaN (SDSS voids)
    if has_xyz:
        coords_xyz = catalog[['x', 'y', 'z']].values
        valid_mask_xyz = np.isfinite(coords_xyz).all(axis=1)
        # Only fill rows that don't already have valid coords from x_mpc, y_mpc, z_mpc
        missing_mask = ~np.isfinite(coords).all(axis=1)
        fill_mask = valid_mask_xyz & missing_mask
        coords[fill_mask] = coords_xyz[fill_mask]
    
    # Check if we have any valid coordinates
    valid_mask = np.isfinite(coords).all(axis=1)
    n_valid = valid_mask.sum()
    
    if n_valid == 0:
        return None
    
    return coords

def convert_spherical_to_cartesian(
    catalog: pd.DataFrame,
    cosmology: Optional[Any] = None
) -> Tuple[np.ndarray, pd.DataFrame]:
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
        
        return positions, catalog
        
    except ImportError:
        # Fallback: simplified approximation (not physically accurate)
        warnings.warn("astropy not available, using simplified coordinate transformation")
        logger.warning("Using simplified coordinate transformation (astropy not available)")
        
        ra_rad = np.radians(catalog[ra_col].values)
        dec_rad = np.radians(catalog[dec_col].values)
        # Approximate comoving distance (rough approximation)
        r_comov = catalog['redshift'].values * 3000.0  # Mpc/h approximation
        
        positions = np.column_stack([
            r_comov * np.cos(dec_rad) * np.cos(ra_rad),
            r_comov * np.cos(dec_rad) * np.sin(ra_rad),
            r_comov * np.sin(dec_rad)
        ])
        return positions, catalog

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
        positions, df = convert_spherical_to_cartesian(df, cosmology=cosmology)
        df['x'] = positions[:, 0]
        df['y'] = positions[:, 1]
        df['z'] = positions[:, 2]
        
        return df

def fill_missing_coordinates(catalog: pd.DataFrame) -> pd.DataFrame:
    """
    Fill in missing redshift or comoving coordinates using cosmological conversions.

    This ensures complete coordinate information for all voids, converting between
    redshift and comoving distance as needed.

    Parameters:
        catalog: DataFrame with partial coordinate information

    Returns:
        DataFrame: Catalog with complete coordinate information
    """
    from astropy.cosmology import Planck18
    import astropy.units as u

    catalog = catalog.copy()

    # Check what coordinate information we have
    has_cartesian = all(col in catalog.columns for col in ['x', 'y', 'z']) or \
                   all(col in catalog.columns for col in ['x_mpc', 'y_mpc', 'z_mpc'])
    has_spherical = all(col in catalog.columns for col in ['ra_deg', 'dec_deg', 'redshift'])
    has_redshift = 'redshift' in catalog.columns

    # Case 1: Have Cartesian coordinates but missing/invalid redshift for some rows
    if has_cartesian and has_redshift:
        # Only compute redshifts for rows that have Cartesian coords but missing/invalid redshift
        needs_redshift = catalog['redshift'].isna() | (catalog['redshift'] <= 0)
        n_needs = needs_redshift.sum()

        if n_needs > 0:
            logger.info(f"  Computing redshift for {n_needs} voids with Cartesian coordinates but missing redshift...")

            # Get Cartesian coordinates
            coords = extract_cartesian_coordinates(catalog)
            if coords is not None:
                # Calculate comoving distances
                distances = np.sqrt(coords[:, 0]**2 + coords[:, 1]**2 + coords[:, 2]**2)

                # Only compute redshifts for rows that need them
                try:
                    import scipy.optimize as opt

                    new_redshifts = catalog['redshift'].copy()
                    computed_count = 0

                    for i in range(len(catalog)):
                        if needs_redshift.iloc[i] and np.isfinite(distances[i]) and distances[i] > 0:
                            try:
                                def distance_diff(z):
                                    return abs(Planck18.comoving_distance(z).value - distances[i])

                                result = opt.minimize_scalar(distance_diff, bounds=(0.001, 2.0), method='bounded')
                                if result.success:
                                    new_redshifts.iloc[i] = result.x
                                    computed_count += 1
                            except:
                                pass  # Keep NaN

                    catalog['redshift'] = new_redshifts
                    logger.info(f"  ✓ Computed {computed_count} redshifts from comoving distances")

                except Exception as e:
                    logger.warning(f"  ⚠ Failed to compute redshifts from distances: {e}")

    # Case 2: Have redshift but missing Cartesian coordinates
    if has_spherical and not has_cartesian:
        logger.info("  Computing comoving coordinates from redshift...")

        # Convert spherical to Cartesian
        try:
            positions, filtered_catalog = convert_spherical_to_cartesian(catalog)

            # convert_spherical_to_cartesian may have filtered the catalog,
            # so we need to use the filtered_catalog it returned
            catalog = filtered_catalog
            catalog['x'] = positions[:, 0]
            catalog['y'] = positions[:, 1]
            catalog['z'] = positions[:, 2]

            logger.info(f"  ✓ Computed Cartesian coordinates for {len(positions)} voids")

        except Exception as e:
            logger.warning(f"  ⚠ Failed to compute Cartesian coordinates from redshift: {e}")

    return catalog


def get_cartesian_positions(catalog: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, bool]:
    """
    Get Cartesian positions from catalog, filling in missing coordinate data as needed.
    
    Parameters:
        catalog: DataFrame with partial coordinate information
        
    Returns:
        tuple: (positions_array, filtered_catalog, was_converted)
            positions_array: Array of shape (N, 3) with x, y, z coordinates
            filtered_catalog: Catalog filtered to match valid positions
            was_converted: True if conversion was performed
    """
    # First, fill in any missing coordinate information
    catalog = fill_missing_coordinates(catalog)  # THIS NOW SAVES THE RESULT

    # Try to extract existing Cartesian coordinates first
    positions = extract_cartesian_coordinates(catalog)
    
    if positions is not None:
        # Check if we have enough valid coordinates
        valid_mask = np.isfinite(positions).all(axis=1)
        n_valid = valid_mask.sum()

        # If most coordinates are valid, use them
        if n_valid >= len(positions) * 0.5:
            # Filter catalog to match valid positions
            filtered_catalog = catalog[valid_mask].copy()
            return positions[valid_mask], filtered_catalog, False
    
    # As fallback, convert from spherical (shouldn't happen after filling)
    logger.warning("  Falling back to spherical conversion (shouldn't happen after coordinate filling)")
    positions, catalog = convert_spherical_to_cartesian(catalog)
    # convert_spherical_to_cartesian already filters invalid rows, so catalog is already filtered
    # But we need to return the filtered catalog - for now, assume it matches
    return positions, catalog, True


def validate_quijote_cosmology() -> Dict[str, Any]:
    """
    Verify Quijote fiducial cosmology compatible with Planck18.

    Returns validation metrics and percent differences.

    Quijote fiducial parameters (Villaescusa-Navarro et al. 2020):
    - Ωm = 0.3175
    - Ωb = 0.049
    - h = 0.6711
    - ns = 0.9624
    - σ8 = 0.834

    Planck18 parameters:
    - Ωm = 0.315
    - Ωb = 0.0493
    - h = 0.674
    - ns = 0.965
    - σ8 = 0.811
    """
    quijote_cosmo = {
        'Omega_m': 0.3175,
        'Omega_b': 0.049,
        'h': 0.6711,
        'n_s': 0.9624,
        'sigma_8': 0.834
    }

    planck18_cosmo = {
        'Omega_m': 0.315,
        'Omega_b': 0.0493,
        'h': 0.674,
        'n_s': 0.965,
        'sigma_8': 0.811
    }

    # Calculate percent differences
    differences = {}
    max_diff_percent = 0.0

    for param in quijote_cosmo:
        quijote_val = quijote_cosmo[param]
        planck_val = planck18_cosmo[param]
        diff_percent = abs(quijote_val - planck_val) / planck_val * 100
        differences[param] = {
            'quijote': quijote_val,
            'planck18': planck_val,
            'diff_percent': diff_percent
        }
        max_diff_percent = max(max_diff_percent, diff_percent)

    # Assessment
    compatible = max_diff_percent < 5.0  # All parameters within 5%

    assessment = "compatible" if compatible else "incompatible"

    logger.info(f"Cosmology compatibility check: {assessment} (max difference: {max_diff_percent:.1f}%)")
    for param, vals in differences.items():
        logger.info(f"  {param}: Quijote {vals['quijote']:.4f} vs Planck18 {vals['planck18']:.4f} "
                   f"(diff: {vals['diff_percent']:.1f}%)")

    return {
        'compatible': compatible,
        'max_difference_percent': max_diff_percent,
        'assessment': assessment,
        'parameter_differences': differences
    }

