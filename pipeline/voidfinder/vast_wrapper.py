"""
VAST VoidFinder Wrapper
=======================

Wrapper for VAST VoidFinder algorithm (Hoyle & Vogeley 2002).

Algorithm: VoidFinder - Grid-based sphere-growing void finder
- Imposes cubic grid over galaxy distribution
- Grows spheres from empty grid cells until bounded by galaxies
- Combines overlapping spheres into discrete voids
- Identifies maximal spheres (largest sphere in each void)

VAST handles all multiprocessing internally via Cython. We pass num_cpus parameter
to control parallelization but do NOT implement our own parallelization.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import logging
import platform
import time as time_module

from .vast_parameters import VASTParameterConfig

# Try to import VAST
try:
    import vast
    import vast.voidfinder
    VAST_AVAILABLE = True
except ImportError:
    VAST_AVAILABLE = False

# Note: VAST VoidFinder uses its own internal CPU parallelization (Cython + Multiprocessing).
# We pass num_cpus parameter to VAST and let it handle all parallelization.
# We do NOT implement our own parallelization - VAST handles everything internally.
# We do not import torch here as VAST doesn't use GPU acceleration.

logger = logging.getLogger(__name__)


class VASTVoidFinderWrapper:
    """
    Wrapper for VAST VoidFinder algorithm.
    
    Provides interface to VAST void finding.
    """
    
    def __init__(self, use_acceleration: bool = False):
        """
        Initialize VAST wrapper.
        
        Parameters:
            use_acceleration: Deprecated/Ignored (VAST uses CPU-only parallelization)
        """
        if not VAST_AVAILABLE:
            raise ImportError(
                "VAST not available. Install with: "
                "pip install git+https://github.com/DESI-UR/VAST.git"
            )
        
        if use_acceleration:
            logger.info("Note: VAST VoidFinder uses internal CPU parallelization (Cython + Multiprocessing). "
                       "GPU acceleration not applicable - all parallelization is handled by VAST via num_cpus parameter.")

    
    def find_voids(self, galaxy_catalog: pd.DataFrame,
                   min_radius: float = 5.0,
                   max_radius: float = 100.0,
                   survey_name: str = "SDSS_DR16",
                   out_directory: Optional[Path] = None,
                   z_min: Optional[float] = None,
                   z_max: Optional[float] = None,
                   catalog_name: Optional[str] = None,
                   grid_size: Optional[float] = None,
                   num_cpus: Optional[int] = None,
                   save_after: Optional[int] = 10000,
                   use_start_checkpoint: bool = False,
                   **kwargs) -> pd.DataFrame:
        """
        Find voids in galaxy catalog using VAST VoidFinder.
        
        Parameters:
            galaxy_catalog: DataFrame with columns: x, y, z (comoving Mpc)
            min_radius: Minimum void radius (Mpc) - maps to min_maximal_radius
            max_radius: Maximum void radius (Mpc) - used for filtering results
            survey_name: Name of the survey (required by VAST for output filenames)
            out_directory: Directory to save VAST output (optional, uses temp if None)
            z_min, z_max: Redshift range for distance calculations
            catalog_name: Name of the catalog config to use (default: survey_name)
            num_cpus: Number of CPUs for parallelization (None = all physical cores, VAST default)
            save_after: Save checkpoint after every N cells processed (default: 10000)
            use_start_checkpoint: Resume from VAST checkpoint file if found
            **kwargs: Additional VAST parameters
            
        Returns:
            DataFrame with void catalog
        """
        if not VAST_AVAILABLE:
            raise RuntimeError("VAST not available")
        
        logger.setLevel(logging.DEBUG)  # Enable debug logging
        logger.debug("=" * 80)
        logger.debug("VAST VoidFinder: Starting find_voids()")
        logger.debug("=" * 80)
        logger.info(f"Finding voids in catalog with {len(galaxy_catalog):,} galaxies")
        logger.info(f"  Min radius: {min_radius:.1f} Mpc")
        logger.info(f"  Max radius: {max_radius:.1f} Mpc")
        logger.debug(f"  Input catalog columns: {list(galaxy_catalog.columns)}")
        logger.debug(f"  Input catalog shape: {galaxy_catalog.shape}")
        
        # VAST workflow requires:
        # 1. Galaxy table with ra, dec, z (or Rgal for comoving distance)
        # 2. filter_galaxies() to separate wall/field galaxies
        # 3. find_voids() on wall_coords_xyz only
        
        # Ensure we have required columns for VAST filter_galaxies
        # It needs: ra, dec, z (or Rgal), and optionally rabsmag for magnitude cut
        from astropy.table import Table
        
        # Build galaxy table for VAST
        galaxy_table_dict = {}
        
        # Required: ra, dec, z
        if 'ra' in galaxy_catalog.columns:
            galaxy_table_dict['ra'] = galaxy_catalog['ra'].values
        elif 'ra_deg' in galaxy_catalog.columns:
            galaxy_table_dict['ra'] = galaxy_catalog['ra_deg'].values
        else:
            raise ValueError("Missing 'ra' or 'ra_deg' column")
        
        if 'dec' in galaxy_catalog.columns:
            galaxy_table_dict['dec'] = galaxy_catalog['dec'].values
        elif 'dec_deg' in galaxy_catalog.columns:
            galaxy_table_dict['dec'] = galaxy_catalog['dec_deg'].values
        else:
            raise ValueError("Missing 'dec' or 'dec_deg' column")
        
        # CRITICAL: After _convert_to_comoving, 'z' is comoving coordinate, not redshift!
        # Always prefer 'redshift' column if available
        if 'redshift' in galaxy_catalog.columns:
            galaxy_table_dict['z'] = galaxy_catalog['redshift'].values
        elif 'z' in galaxy_catalog.columns:
            # Check if z values look like redshifts (0 < z < 10) vs comoving coords (can be negative/large)
            z_values = galaxy_catalog['z'].values
            if np.all((z_values > 0) & (z_values < 10)):
                # Looks like redshift
                galaxy_table_dict['z'] = z_values
            else:
                raise ValueError("'z' column appears to be comoving coordinates, not redshift. Need 'redshift' column.")
        else:
            raise ValueError("Missing 'z' or 'redshift' column")
        
        # For comoving distance metric, VAST needs 'Rgal' column
        # Calculate from redshift if we have x,y,z but need Rgal
        if 'Rgal' not in galaxy_catalog.columns:
            # Calculate comoving distance from redshift
            from astropy.cosmology import Planck18 as cosmo
            z_values = galaxy_table_dict['z']
            galaxy_table_dict['Rgal'] = cosmo.comoving_distance(z_values).value  # Mpc
        
        # Optional: absolute magnitude for magnitude cut
        if 'abs_mag' in galaxy_catalog.columns:
            galaxy_table_dict['rabsmag'] = galaxy_catalog['abs_mag'].values
        elif 'abs_mag_r' in galaxy_catalog.columns:
            galaxy_table_dict['rabsmag'] = galaxy_catalog['abs_mag_r'].values
        
        # Create astropy Table
        galaxy_table = Table(galaxy_table_dict)
        
        # Set up output directory (use temp directory if not provided)
        import tempfile
        if out_directory is None:
            temp_dir = tempfile.mkdtemp(prefix='vast_voidfinder_')
            out_directory = Path(temp_dir)
            cleanup_temp = True
        else:
            out_directory = Path(out_directory)
            out_directory.mkdir(parents=True, exist_ok=True)
            cleanup_temp = False
        
        logger.info(f"  Using output directory: {out_directory}")
        
        # Compute catalog-specific parameters
        # Infer z_min/z_max from catalog if not provided
        if z_min is None or z_max is None:
            if 'z' in galaxy_catalog.columns:
                z_min = galaxy_catalog['z'].min() if z_min is None else z_min
                z_max = galaxy_catalog['z'].max() if z_max is None else z_max
            elif 'redshift' in galaxy_catalog.columns:
                z_min = galaxy_catalog['redshift'].min() if z_min is None else z_min
                z_max = galaxy_catalog['redshift'].max() if z_max is None else z_max
            else:
                # Fallback: use reasonable defaults
                z_min = z_min or 0.01
                z_max = z_max or 0.2
                logger.warning(f"No redshift info found, using defaults: z={z_min}-{z_max}")
        
        # Get catalog-specific parameter configuration
        # Use catalog_name if provided, otherwise fallback to survey_name
        # (survey_name might be a chunk name like 'SDSS_DR16_chunk_0', so we want the base name)
        config_name = (catalog_name or survey_name).lower()
        
        # Handle case where config_name includes chunk info
        if 'chunk' in config_name and catalog_name is None:
            # Try to extract base name
            if 'sdss_dr16' in config_name:
                config_name = 'sdss_dr16'
            elif 'sdss_dr7' in config_name:
                config_name = 'sdss_dr7'
        
        # CRITICAL: Filter galaxies using VAST's filter_galaxies function
        # This separates wall (non-isolated) from field (isolated) galaxies
        # VoidFinder only works on wall galaxies!
        logger.debug("=" * 80)
        logger.debug("STEP 1: Filtering galaxies")
        logger.debug("=" * 80)
        logger.info("Filtering galaxies (removing isolated galaxies)...")
        logger.debug(f"  Galaxy table rows: {len(galaxy_table)}")
        logger.debug(f"  Galaxy table columns: {galaxy_table.colnames}")
        logger.debug(f"  Survey name: {survey_name}")
        logger.debug(f"  Output directory: {out_directory}")
        
        from vast.voidfinder import filter_galaxies as vast_filter_galaxies
        
        # Compute dist_limits from redshift range
        from astropy.cosmology import Planck18 as cosmo
        dist_min = cosmo.comoving_distance(z_min).value  # Mpc
        dist_max = cosmo.comoving_distance(z_max).value  # Mpc
        dist_limits = [dist_min, dist_max]
        logger.debug(f"  Distance limits: {dist_limits} Mpc (z={z_min}-{z_max})")
        
        # Filter galaxies: removes isolated galaxies, separates wall/field
        # Note: filter_galaxies converts ra,dec,z to xyz internally
        logger.debug("  Calling vast_filter_galaxies()...")
        filter_start_time = time_module.time()
        
        try:
            wall_coords_xyz, field_coords_xyz = vast_filter_galaxies(
                galaxy_table,
                survey_name,
                str(out_directory),
                dist_limits=dist_limits,
                dist_metric='comoving',
                rm_isolated=True,  # Remove isolated galaxies (critical!)
                mag_cut=False,  # We already did absolute magnitude cut earlier
                write_table=False,  # Don't write intermediate files
                verbose=2  # Maximum verbosity for debugging
            )
            filter_elapsed = time_module.time() - filter_start_time
            logger.debug(f"  ✓ filter_galaxies() completed in {filter_elapsed:.2f} seconds")
        except Exception as e:
            filter_elapsed = time_module.time() - filter_start_time
            logger.error(f"  ✗ filter_galaxies() failed after {filter_elapsed:.2f} seconds: {e}")
            import traceback
            logger.error(f"  Traceback: {traceback.format_exc()}")
            raise
        
        # Check if wall_coords_xyz is numpy array or astropy Table
        if hasattr(wall_coords_xyz, 'shape'):
            logger.debug(f"  Wall coords shape: {wall_coords_xyz.shape}")
        elif hasattr(wall_coords_xyz, '__len__'):
            logger.debug(f"  Wall coords length: {len(wall_coords_xyz)}")
        else:
            logger.debug(f"  Wall coords type: {type(wall_coords_xyz)}")
        
        if hasattr(field_coords_xyz, 'shape'):
            logger.debug(f"  Field coords shape: {field_coords_xyz.shape}")
        elif hasattr(field_coords_xyz, '__len__'):
            logger.debug(f"  Field coords length: {len(field_coords_xyz)}")
        else:
            logger.debug(f"  Field coords type: {type(field_coords_xyz)}")
        logger.info(f"  Wall galaxies (non-isolated): {len(wall_coords_xyz):,}")
        logger.info(f"  Field galaxies (isolated, removed): {len(field_coords_xyz):,}")
        
        if len(wall_coords_xyz) == 0:
            raise RuntimeError("No wall galaxies found after filtering - cannot find voids")
        
        # Compute grid_origin from wall galaxies (as per VAST example)
        # This is the minimum x,y,z coordinate used as grid origin
        logger.debug("  Computing grid origin from wall galaxies...")
        coords_min = np.min(wall_coords_xyz, axis=0)
        coords_max = np.max(wall_coords_xyz, axis=0)
        logger.debug(f"  Wall galaxy coordinate range:")
        logger.debug(f"    X: [{coords_min[0]:.1f}, {coords_max[0]:.1f}] Mpc")
        logger.debug(f"    Y: [{coords_min[1]:.1f}, {coords_max[1]:.1f}] Mpc")
        logger.debug(f"    Z: [{coords_min[2]:.1f}, {coords_max[2]:.1f}] Mpc")
        logger.info(f"  Grid origin: [{coords_min[0]:.1f}, {coords_min[1]:.1f}, {coords_min[2]:.1f}] Mpc")
        
        # Get parameter config for other VAST settings
        param_config = VASTParameterConfig(config_name)
        
        # Compute parameters using wall galaxies only
        wall_df = pd.DataFrame(wall_coords_xyz, columns=['x', 'y', 'z'])
        vast_params = param_config.compute_parameters(wall_df, z_min, z_max, grid_size=grid_size)
        
        
        # Override min_maximal_radius with user-specified min_radius
        vast_params['min_maximal_radius'] = min_radius
        
        # Store dist_limits for find_voids
        vast_params['dist_limits'] = dist_limits
        
        # Estimate expected void count for validation
        if 'xyz_limits' in vast_params:
            lims = vast_params['xyz_limits']
            # Handle both list of lists and numpy array
            if isinstance(lims, np.ndarray):
                # Shape (2, 3)
                volume = (lims[1, 0] - lims[0, 0]) * (lims[1, 1] - lims[0, 1]) * (lims[1, 2] - lims[0, 2])
            else:
                # Assume shape (2, 3) list
                volume = (lims[1][0] - lims[0][0]) * (lims[1][1] - lims[0][1]) * (lims[1][2] - lims[0][2])
                
            estimated_voids = param_config.estimate_void_count(len(galaxy_catalog), volume)
            logger.info(f"  Estimated void count: ~{estimated_voids:,} (based on {len(galaxy_catalog):,} galaxies)")
        
        # Prepare VAST parameters (merge with user kwargs, user kwargs take precedence)
        # Remove survey_name from vast_params since we pass it positionally
        vast_kwargs = {
            **{k: v for k, v in vast_params.items() if k != 'survey_name'},
            **kwargs,  # User kwargs override computed parameters
            'min_maximal_radius': min_radius,  # Ensure min_radius is used
        }
        
        # Also remove survey_name from kwargs if present
        vast_kwargs.pop('survey_name', None)
        
        # Explicitly set parallelization and checkpoint parameters
        # These override defaults from vast_params
        if num_cpus is not None:
            vast_kwargs['num_cpus'] = num_cpus
            if num_cpus == 1:
                logger.info(f"  Running in single-threaded mode (num_cpus=1)")
            else:
                logger.info(f"  Running in multi-threaded mode (num_cpus={num_cpus})")
        else:
            # Use VAST default (all physical cores)
            vast_kwargs['num_cpus'] = None
            logger.info(f"  Using VAST default parallelization (all physical cores)")
        
        # Set checkpoint parameters
        if save_after is not None:
            vast_kwargs['save_after'] = save_after
            logger.info(f"  Checkpointing enabled: saving every {save_after:,} cells")
        else:
            vast_kwargs['save_after'] = None
        
        vast_kwargs['use_start_checkpoint'] = use_start_checkpoint
        if use_start_checkpoint:
            logger.info(f"  Will attempt to resume from VAST checkpoint file if found")
        
        logger.debug("=" * 80)
        logger.debug("STEP 2: Preparing VAST find_voids() parameters")
        logger.debug("=" * 80)
        logger.debug(f"  Final vast_kwargs keys: {list(vast_kwargs.keys())}")
        for key, value in vast_kwargs.items():
            if isinstance(value, (int, float, str, bool)) or value is None:
                logger.debug(f"    {key} = {value}")
            elif isinstance(value, np.ndarray):
                logger.debug(f"    {key} = np.ndarray(shape={value.shape}, dtype={value.dtype})")
            elif isinstance(value, (list, tuple)):
                logger.debug(f"    {key} = {type(value).__name__}(len={len(value)})")
            else:
                logger.debug(f"    {key} = {type(value).__name__}")
        
        logger.info(f"  VAST parameters: min_maximal_radius={vast_kwargs['min_maximal_radius']:.1f} Mpc, "
                   f"grid_edge_length={vast_kwargs['hole_grid_edge_length']:.1f} Mpc")
        
        try:
            # Call VAST VoidFinder with correct API
            # CRITICAL: Use wall_coords_xyz only (not all galaxies!)
            logger.debug("=" * 80)
            logger.debug("STEP 3: Calling VAST find_voids()")
            logger.debug("=" * 80)
            logger.info("Calling VAST VoidFinder on wall galaxies...")
            # Handle both numpy arrays and other types
            if hasattr(wall_coords_xyz, 'shape'):
                logger.debug(f"  Wall coords shape: {wall_coords_xyz.shape}")
                logger.debug(f"  Wall coords dtype: {wall_coords_xyz.dtype}")
            elif hasattr(wall_coords_xyz, '__len__'):
                logger.debug(f"  Wall coords length: {len(wall_coords_xyz)}")
                logger.debug(f"  Wall coords type: {type(wall_coords_xyz)}")
            else:
                logger.debug(f"  Wall coords type: {type(wall_coords_xyz)}")
            logger.debug(f"  Survey name: {survey_name}")
            logger.debug(f"  Output directory: {out_directory}")
            
            from vast.voidfinder import find_voids as vast_find_voids
            
            # Set grid_origin explicitly (as per VAST example)
            vast_kwargs['grid_origin'] = coords_min
            logger.debug(f"  grid_origin set to: {coords_min}")
            
            # Use xyz mask type (we computed xyz_limits from galaxy distribution)
            # Note: VAST example uses ra_dec_z with angular mask, but xyz mode works
            # without needing to generate an angular mask, which is simpler for our use case
            vast_kwargs['mask_type'] = 'xyz'  # Use xyz mode (we computed xyz_limits)
            logger.debug(f"  mask_type set to: 'xyz'")
            
            # Ensure dist_limits is passed (required for xyz mode validation)
            vast_kwargs['dist_limits'] = dist_limits
            logger.debug(f"  dist_limits set to: {dist_limits}")
            
            # Set verbose to maximum for debugging
            vast_kwargs['verbose'] = 2  # Maximum verbosity
            vast_kwargs['print_after'] = 1.0  # Print progress every second
            
            logger.debug("  About to call vast_find_voids()...")
            logger.debug(f"  Function signature: vast_find_voids(wall_coords_xyz, '{survey_name}', '{out_directory}', **vast_kwargs)")
            find_voids_start_time = time_module.time()
            
            vast_find_voids(
                wall_coords_xyz,  # CRITICAL: Use filtered wall galaxies only!
                survey_name,
                str(out_directory),
                **vast_kwargs
            )
            
            find_voids_elapsed = time_module.time() - find_voids_start_time
            logger.debug(f"  ✓ vast_find_voids() completed in {find_voids_elapsed:.2f} seconds")
            
            # VAST saves results to files in out_directory
            # Load the results
            logger.debug("=" * 80)
            logger.debug("STEP 4: Loading VAST results")
            logger.debug("=" * 80)
            load_start_time = time_module.time()
            voids = self._load_vast_results(out_directory, max_radius)
            load_elapsed = time_module.time() - load_start_time
            logger.debug(f"  ✓ Loaded results in {load_elapsed:.2f} seconds")
            
            if cleanup_temp:
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
                
        except Exception as e:
            logger.error(f"VAST VoidFinder failed: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            if cleanup_temp and out_directory.exists():
                import shutil
                shutil.rmtree(temp_dir, ignore_errors=True)
            
            # Don't use fallback - raise the error so user knows VAST failed
            raise RuntimeError(f"VAST VoidFinder execution failed: {e}") from e
        
        return voids
    
    def _load_vast_results(self, out_directory: Path, max_radius: float = 100.0) -> pd.DataFrame:
        """
        Load void results from VAST output directory.
        
        Parameters:
            out_directory: Directory where VAST saved results
            max_radius: Maximum radius to filter voids
            
        Returns:
            DataFrame with void catalog
        """
        import glob
        
        # VAST typically saves results as FITS files or pickle files
        # Look for common output file patterns
        result_files = []
        
        logger.info(f"Loading VAST results from {out_directory}")
        
        # List all files for debugging
        all_files = list(out_directory.glob("*"))
        logger.info(f"Files in output directory ({len(all_files)} total): {[f.name for f in all_files[:10]]}")
        
        void_df = pd.DataFrame()
        
        # Priority 1: Check for FITS files (VAST's primary output format)
        # VAST typically names files like: survey_name_voids.fits, survey_name_maximals.fits, etc.
        fits_files = sorted(out_directory.glob("*.fits")) + sorted(out_directory.glob("*.fit"))
        if fits_files:
            logger.info(f"Found {len(fits_files)} FITS files")
            from astropy.table import Table
            # Try to find the main void catalog file (usually contains 'void' or 'maximal' in name)
            void_fits = [f for f in fits_files if 'void' in f.name.lower() or 'maximal' in f.name.lower()]
            if not void_fits:
                void_fits = fits_files  # Use all if no specific match
            
            for fits_file in void_fits:
                try:
                    logger.info(f"  Reading FITS file: {fits_file.name}")
                    vast_table = Table.read(fits_file)
                    void_df = vast_table.to_pandas()
                    logger.info(f"  ✓ Loaded {len(void_df)} voids from {fits_file.name}")
                    logger.info(f"  Columns: {list(void_df.columns)[:10]}")
                    if len(void_df) > 0:
                        break  # Use first successful read with data
                except Exception as e:
                    logger.warning(f"  Failed to read {fits_file.name}: {e}")
                    continue
        
        # Priority 2: Check for pickle files
        if void_df.empty:
            pkl_files = sorted(out_directory.glob("*.pkl"))
            if pkl_files:
                logger.info(f"Found {len(pkl_files)} pickle files")
                for pkl_file in pkl_files:
                    try:
                        logger.info(f"  Reading pickle file: {pkl_file.name}")
                        void_df = pd.read_pickle(pkl_file)
                        logger.info(f"  ✓ Loaded {len(void_df)} voids from {pkl_file.name}")
                        logger.info(f"  Columns: {list(void_df.columns)[:10]}")
                        if len(void_df) > 0:
                            break
                    except Exception as e:
                        logger.warning(f"  Failed to read {pkl_file.name}: {e}")
                        continue
        
        # Priority 3: Check for HDF5 files
        if void_df.empty:
            h5_files = sorted(out_directory.glob("*.h5")) + sorted(out_directory.glob("*.hdf5"))
            if h5_files:
                logger.info(f"Found {len(h5_files)} HDF5 files")
                try:
                    h5_file = h5_files[0]
                    logger.info(f"  Reading HDF5 file: {h5_file.name}")
                    void_df = pd.read_hdf(h5_file)
                    logger.info(f"  ✓ Loaded {len(void_df)} voids from HDF5")
                except Exception as e:
                    logger.warning(f"  Failed to read HDF5: {e}")
        
        # Priority 4: Check for text/csv files
        if void_df.empty:
            txt_files = sorted(out_directory.glob("*.txt")) + sorted(out_directory.glob("*.csv"))
            if txt_files:
                logger.info(f"Found {len(txt_files)} text files")
                for txt_file in txt_files:
                    try:
                        logger.info(f"  Reading text file: {txt_file.name}")
                        # Try different separators
                        for sep in [r'\s+', ',', '\t']:
                            try:
                                void_df = pd.read_csv(txt_file, sep=sep, comment='#')
                                if len(void_df) > 0 and len(void_df.columns) > 2:
                                    logger.info(f"  ✓ Loaded {len(void_df)} voids from {txt_file.name} (sep={sep})")
                                    break
                            except:
                                continue
                        if not void_df.empty:
                            break
                    except Exception as e:
                        logger.warning(f"  Failed to read {txt_file.name}: {e}")
                        continue
        
        if void_df.empty:
            logger.error(f"✗ No valid void data found in {out_directory}")
            logger.error(f"  All files: {[f.name for f in all_files]}")
            logger.error(f"  This may indicate VAST did not find any voids or failed to save results")
            return pd.DataFrame()
        
        if void_df.empty:
            logger.warning("No voids found in VAST output")
            return pd.DataFrame()
        
        # Standardize column names
        column_mapping = {
            'x': 'x',
            'y': 'y',
            'z': 'z',
            'X': 'x',
            'Y': 'y',
            'Z': 'z',
            'radius': 'radius_mpc',
            'Radius': 'radius_mpc',
            'R': 'radius_mpc',
            'r': 'radius_mpc',
            'Reff': 'radius_eff',
            'reff': 'radius_eff',
            'radius_eff': 'radius_eff',
            'ra': 'ra_deg',
            'RA': 'ra_deg',
            'dec': 'dec_deg',
            'Dec': 'dec_deg',
            'DEC': 'dec_deg',
            'redshift': 'redshift',
            'z': 'redshift',
            'Z': 'redshift',
        }
        
        for old_col, new_col in column_mapping.items():
            if old_col in void_df.columns and new_col not in void_df.columns:
                void_df = void_df.rename(columns={old_col: new_col})
        
        # Ensure required columns exist
        if 'radius_mpc' not in void_df.columns and 'radius_eff' in void_df.columns:
            void_df['radius_mpc'] = void_df['radius_eff']
        elif 'radius_mpc' not in void_df.columns:
            # Try to find radius column with different name
            radius_cols = [c for c in void_df.columns if 'radius' in c.lower() or 'r' == c.lower()]
            if radius_cols:
                void_df['radius_mpc'] = void_df[radius_cols[0]]
            else:
                logger.warning("No radius column found, using default")
                void_df['radius_mpc'] = 10.0  # Default radius
        
        # Filter by max_radius if specified
        if max_radius and 'radius_mpc' in void_df.columns:
            before_filter = len(void_df)
            void_df = void_df[void_df['radius_mpc'] <= max_radius].copy()
            if len(void_df) < before_filter:
                logger.info(f"Filtered {before_filter - len(void_df)} voids exceeding max_radius {max_radius} Mpc")
        
        # Calculate volume if not present
        if 'volume_mpc3' not in void_df.columns:
            void_df['volume_mpc3'] = (4/3) * np.pi * void_df['radius_mpc']**3
        
        # Add metadata columns
        void_df['survey'] = 'SDSS_DR16'
        void_df['algorithm'] = 'VAST_VOIDFINDER'
        
        # Add edge_flag if not present
        if 'edge_flag' not in void_df.columns:
            void_df['edge_flag'] = 0
        
        # Add void IDs if not present
        if 'void_id' not in void_df.columns:
            void_df['void_id'] = range(len(void_df))
        
        logger.info(f"✓ Found {len(void_df):,} voids")
        
        return void_df

