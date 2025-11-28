"""
VoidFinder Pipeline
===================

Pipeline for generating void catalogs from galaxy surveys using VAST VoidFinder.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from pathlib import Path
import time
import logging
from tqdm import tqdm

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

from astropy import units as u
from astropy.coordinates import SkyCoord
from astropy.cosmology import Planck18 as cosmo

from ..common.base_pipeline import AnalysisPipeline
from .checkpoint import CheckpointManager
from .vast_wrapper import VASTVoidFinderWrapper
from .galaxy_catalogs.catalog_registry import CatalogRegistry

logger = logging.getLogger(__name__)


class VoidFinderPipeline(AnalysisPipeline):
    """
    VAST VoidFinder pipeline for generating void catalogs from galaxy surveys.
    
    Implements the VoidFinder algorithm (Hoyle & Vogeley 2002) via VAST.
    
    Algorithm:
    - Imposes a cubic grid over the galaxy distribution
    - Grows spheres (holes) from empty grid cells until bounded by galaxies
    - Combines overlapping spheres into discrete voids
    - Identifies maximal spheres (largest sphere in each void)
    
    Downloads galaxy catalogs, filters to volume-limited samples, applies
    VAST VoidFinder algorithm, and stores results for consumption by void analysis pipeline.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize VoidFinder pipeline.
        
        Parameters:
            output_dir (str): Output directory
        """
        super().__init__("voidfinder", output_dir)
        
        # Initialize checkpoint manager
        checkpoint_dir = self.processed_data_dir / "checkpoints"
        self.checkpoint_manager = CheckpointManager(checkpoint_dir, "voidfinder")
        
        # Initialize catalog registry
        self.catalog_registry = CatalogRegistry
        
        # Initialize VAST wrapper
        try:
            self.voidfinder = VASTVoidFinderWrapper(use_acceleration=True)
        except ImportError:
            logger.warning("VAST not available - install with: pip install git+https://github.com/DESI-UR/VAST.git")
            self.voidfinder = None
        
        # Available catalogs
        self.available_catalogs = self.catalog_registry.list_available()
        
        self.update_metadata('description', 'VAST VoidFinder pipeline for generating void catalogs from galaxy surveys')
        self.update_metadata('available_catalogs', self.available_catalogs)
    
    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        # Enable debug logging for troubleshooting
        logger.setLevel(logging.DEBUG)
        # Also enable debug for vast_wrapper logger
        vast_logger = logging.getLogger('pipeline.voidfinder.vast_wrapper')
        vast_logger.setLevel(logging.DEBUG)
        """
        Execute VoidFinder pipeline.
        
        Parameters:
            context (dict, optional): Analysis parameters
                - algorithm: Algorithm to use ('vast' or 'zobov', default: 'vast')
                - catalog: Catalog name (default: 'sdss_dr16')
                - z_min: Minimum redshift (default: 0.0)
                - z_max: Maximum redshift (default: 1.0)
                - mag_limit: Magnitude limit (default: 22.0)
                - abs_mag_limit: Absolute magnitude limit for volume-limited sample (default: -20.0)
                - min_radius: Minimum void radius in Mpc (default: 5.0)
                - max_radius: Maximum void radius in Mpc (default: 100.0)
                - chunk_size: Galaxy chunk size for processing (default: 100000)
                - num_cpus: Number of CPUs for VAST parallelization (default: None = all physical cores)
                - save_after: Save VAST checkpoint after every N cells (default: None = disabled)
                - use_start_checkpoint: Resume from VAST checkpoint if found (default: False)
                - force_redownload: Force re-download catalog (default: False)
                - output_name: MANDATORY for ZOBOV - base name for output files
        
        Returns:
            dict: Pipeline results
        """
        try:
            # Check if H-ZOBOV algorithm is requested
            algorithm = context.get('algorithm', 'vast') if context else 'vast'
            if algorithm == 'zobov':
                # Route to H-ZOBOV pipeline
                self.log_progress("Routing to H-ZOBOV pipeline...")
                from .zobov.zobov_pipeline import HZOBOVPipeline
                hzobov = HZOBOVPipeline(str(self.base_output_dir))
                return hzobov.run(context)
            
            # Continue with VAST VoidFinder
            self.log_progress("Starting VoidFinder pipeline...")
            
            # Parse context parameters
            catalog_name = context.get('catalog', 'sdss_dr16') if context else 'sdss_dr16'
            z_min = context.get('z_min', 0.0) if context else 0.0
            z_max = context.get('z_max', 1.0) if context else 1.0
            mag_limit = context.get('mag_limit', 22.0) if context else 22.0
            min_radius = context.get('min_radius', 5.0) if context else 5.0
            max_radius = context.get('max_radius', 100.0) if context else 100.0
            abs_mag_limit = context.get('abs_mag_limit', -20.0) if context else -20.0
            chunk_size = context.get('chunk_size', 100000) if context else 100000
            force_redownload = context.get('force_redownload', False) if context else False
            num_cpus = context.get('num_cpus', None) if context else None  # None = use all physical cores (VAST default)
            save_after = context.get('save_after', 10000) if context else 10000  # Default: save every 10k cells
            use_start_checkpoint = context.get('use_start_checkpoint', False) if context else False
            grid_size = context.get('grid_size', 50.0) if context else 50.0  # Default 50 Mpc
            z_bin_size = context.get('z_bin_size', None) if context else None  # None = no binning
            
            # Validate catalog name
            if catalog_name not in self.available_catalogs:
                available = ', '.join(self.available_catalogs)
                error_msg = f"Unknown catalog '{catalog_name}'. Available: {available}"
                self.log_progress(f"✗ {error_msg}")
                raise ValueError(error_msg)
            
            if self.voidfinder is None:
                error_msg = "VAST VoidFinder not available - cannot run pipeline"
                self.log_progress(f"✗ {error_msg}")
                raise RuntimeError(error_msg)
            
            self.log_progress(f"Processing catalog: {catalog_name}")
            self.log_progress(f"  Grid size: {grid_size} Mpc")
            if z_bin_size:
                self.log_progress(f"  Redshift binning: {z_bin_size} per bin")
            
            # Step 1: Download/load galaxy catalog
            self.log_progress("Step 1: Downloading/loading galaxy catalog...")
            galaxy_catalog = self._download_catalog(
                catalog_name, z_min, z_max, mag_limit, force_redownload
            )
            
            if galaxy_catalog is None or len(galaxy_catalog) == 0:
                error_msg = "Failed to load galaxy catalog"
                self.log_progress(f"✗ {error_msg}")
                return {'error': error_msg}
            
            self.log_progress(f"✓ Loaded {len(galaxy_catalog):,} galaxies")
            
            # Step 1.5: Filter Volume Limited Sample (Scientific Rigor)
            self.log_progress(f"Step 1.5: Applying volume-limited sample cut (Mr < {abs_mag_limit})...")
            galaxy_catalog = self._filter_volume_limited(galaxy_catalog, abs_mag_limit)
            self.log_progress(f"✓ Remaining galaxies after Mr cut: {len(galaxy_catalog):,}")
            
            # Step 2: Convert to comoving coordinates
            self.log_progress("Step 2: Converting to comoving coordinates...")
            galaxy_coords = self._convert_to_comoving(galaxy_catalog)
            
            # Step 3: Process voids (with redshift binning if requested)
            if z_bin_size:
                self.log_progress(f"Step 3: Finding voids in redshift bins (bin size: {z_bin_size})...")
                void_catalog = self._find_voids_redshift_binned(
                    galaxy_coords, min_radius, max_radius, chunk_size, z_min, z_max, 
                    z_bin_size, catalog_name, grid_size,
                    num_cpus=num_cpus, save_after=save_after, use_start_checkpoint=use_start_checkpoint
                )
            else:
                self.log_progress("Step 3: Finding voids (with checkpointing)...")
                void_catalog = self._find_voids_chunked(
                    galaxy_coords, min_radius, max_radius, chunk_size, z_min, z_max, catalog_name,
                    grid_size=grid_size,
                    num_cpus=num_cpus, save_after=save_after, use_start_checkpoint=use_start_checkpoint
                )
            
            if void_catalog is None or len(void_catalog) == 0:
                error_msg = "No voids found"
                self.log_progress(f"✗ {error_msg}")
                return {'error': error_msg}
            
            self.log_progress(f"✓ Found {len(void_catalog):,} voids")
            
            # Step 4: Save void catalog
            self.log_progress("Step 4: Saving void catalog...")
            output_file = self._save_void_catalog(catalog_name, void_catalog)
            
            # Generate comprehensive statistics and results
            catalog_stats = self._compute_catalog_statistics(galaxy_catalog, z_min, z_max, mag_limit)
            void_stats = self._compute_void_statistics(void_catalog)
            vast_params = self._get_vast_parameters(galaxy_coords, z_min, z_max, min_radius, max_radius)
            processing_info = self._get_processing_info(galaxy_catalog, void_catalog, chunk_size)
            
            # Generate results
            results = {
                'catalog_name': catalog_name,
                'n_galaxies': len(galaxy_catalog),
                'n_voids': len(void_catalog),
                'void_catalog_file': str(output_file),
                'parameters': {
                    'z_min': z_min,
                    'z_max': z_max,
                    'mag_limit': mag_limit,
                    'min_radius': min_radius,
                    'max_radius': max_radius,
                    'chunk_size': chunk_size
                },
                'catalog_statistics': catalog_stats,
                'void_statistics': void_stats,
                'vast_parameters': vast_params,
                'processing_info': processing_info,
                'summary': self._generate_summary(void_catalog, catalog_stats, void_stats, vast_params)
            }
            
            self.log_progress("✓ VoidFinder pipeline complete")
            
            # Save results
            self.save_results(results)
            
            # Generate comprehensive report
            self._generate_comprehensive_report(results)
            
            return results
            
        except Exception as e:
            error_msg = f"Fatal error in VoidFinder pipeline: {type(e).__name__}: {str(e)}"
            self.log_progress(f"✗ {error_msg}")
            import traceback
            self.log_progress(f"Traceback: {traceback.format_exc()}")
            return {
                'error': error_msg,
                'exception_type': type(e).__name__,
                'traceback': traceback.format_exc()
            }
    
    def _download_catalog(self, catalog_name: str, z_min: float, z_max: float,
                        mag_limit: float, force_redownload: bool) -> Optional[pd.DataFrame]:
        """Download or load galaxy catalog."""
        try:
            catalog_provider = self.catalog_registry.get(
                catalog_name,
                self.downloaded_data_dir,
                self.processed_data_dir
            )
            
            # Try loading from cache first
            if not force_redownload:
                cached = catalog_provider.load(use_cache=True)
                if cached is not None:
                    # Apply redshift and magnitude cuts
                    mask = (cached['z'] >= z_min) & (cached['z'] <= z_max)
                    if 'magnitude' in cached.columns:
                        mask &= (cached['magnitude'] <= mag_limit)
                    elif 'r_mag' in cached.columns:
                        mask &= (cached['r_mag'] <= mag_limit)
                    
                    filtered = cached[mask].copy()
                    
                    # If cache has data in this range, return it
                    if len(filtered) > 0:
                        return filtered
                    else:
                        self.log_progress(f"Cache exists but contains no galaxies in range z={z_min}-{z_max}. Downloading...")
                        # Cache exists but doesn't cover this range -> fall through to download
            
            # Download with checkpointing
            return catalog_provider.download(
                checkpoint_manager=self.checkpoint_manager,
                z_min=z_min,
                z_max=z_max,
                mag_limit=mag_limit,
                force_redownload=force_redownload
            )
            
        except Exception as e:
            logger.error(f"Failed to download catalog: {e}")
            raise
    
    def _filter_volume_limited(self, galaxy_catalog: pd.DataFrame, abs_mag_limit: float) -> pd.DataFrame:
        """
        Filter catalog to create a volume-limited sample based on absolute magnitude.
        
        Formula: M = m - 5 * log10(d_L) - 25
        (Ignoring K-correction for now as per 'simple void search' scope, 
         but using luminosity distance from Planck18 cosmology)
        
        Parameters:
            galaxy_catalog: DataFrame with 'z' and magnitude columns
            abs_mag_limit: Absolute magnitude limit (e.g., -20.0)
            
        Returns:
            Filtered DataFrame
        """
        df = galaxy_catalog.copy()
        
        # Identify magnitude column
        mag_cols = ['magnitude', 'r_mag', 'petroMag_r']
        mag_col = None
        for col in mag_cols:
            if col in df.columns:
                mag_col = col
                break
        
        if not mag_col:
            self.log_progress("⚠ No magnitude column found, skipping volume-limited cut.")
            return df
            
        if 'z' not in df.columns and 'redshift' not in df.columns:
            self.log_progress("⚠ No redshift column found, skipping volume-limited cut.")
            return df
            
        z_col = 'z' if 'z' in df.columns else 'redshift'
        
        # Calculate Luminosity Distance
        # For efficiency on large arrays, use interpolation or numpy operations if possible,
        # but astropy is robust.
        self.log_progress("Calculating absolute magnitudes...")
        
        # Filter invalid redshifts first
        df = df[df[z_col] > 0.001].copy()
        
        # Calculate distance modulus: mu = 5 * log10(d_L_Mpc) + 25
        # d_L in Mpc
        d_L = cosmo.luminosity_distance(df[z_col].values).value
        dist_mod = 5.0 * np.log10(d_L) + 25.0
        
        # Calculate Absolute Magnitude M = m - mu
        # (Simple approximation without K-correction or evolution correction)
        df['abs_mag'] = df[mag_col] - dist_mod
        
        # Filter
        original_count = len(df)
        df_filtered = df[df['abs_mag'] <= abs_mag_limit].copy()
        filtered_count = len(df_filtered)
        
        self.log_progress(f"  Absolute Magnitude cut (M < {abs_mag_limit}):")
        self.log_progress(f"  Removed {original_count - filtered_count:,} galaxies ({(original_count - filtered_count)/original_count:.1%})")
        
        return df_filtered

    def _convert_to_comoving(self, galaxy_catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Convert galaxy catalog to comoving coordinates (x, y, z in Mpc).
        
        Uses modular coordinate conversion function for consistency and maintainability.
        
        Parameters:
            galaxy_catalog: DataFrame with ra, dec, z columns
            
        Returns:
            DataFrame with added x, y, z columns
        """
        from pipeline.common.void_coordinates import convert_spherical_to_cartesian_chunked
        
        # Use modular chunked conversion function
        return convert_spherical_to_cartesian_chunked(
            galaxy_catalog,
            chunk_size=50000,
            show_progress=True,
            cosmology=cosmo,
            preserve_redshift=True
        )
    
    def _find_voids_chunked(self, galaxy_coords: pd.DataFrame,
                           min_radius: float, max_radius: float,
                           chunk_size: int, z_min: float, z_max: float, 
                           catalog_name: str = 'sdss_dr16',
                           grid_size: float = 50.0,
                           num_cpus: Optional[int] = None,
                           save_after: Optional[int] = None,
                           use_start_checkpoint: bool = False) -> pd.DataFrame:
        """
        Find voids in chunks with checkpointing.
        
        Parameters:
            galaxy_coords: DataFrame with x, y, z columns
            min_radius: Minimum void radius
            max_radius: Maximum void radius
            chunk_size: Number of galaxies per chunk
            z_min, z_max: Redshift range
            catalog_name: Name of catalog (for config lookup)
            num_cpus: Number of CPUs for VAST (None = all physical cores)
            save_after: Save checkpoint after N cells (None = disabled)
            use_start_checkpoint: Resume from VAST checkpoint if found
            
        Returns:
            Combined void catalog DataFrame
        """
        n_galaxies = len(galaxy_coords)
        n_chunks = (n_galaxies + chunk_size - 1) // chunk_size
        
        self.log_progress(f"Processing {n_galaxies:,} galaxies in {n_chunks} chunks...")
        
        # Check for existing checkpoint
        checkpoint = self.checkpoint_manager.load_checkpoint('voidfinding')
        resume_from_chunk = 0
        existing_voids = []
        
        if checkpoint and checkpoint.get('parameters', {}).get('min_radius') == min_radius:
            resume_from_chunk = checkpoint.get('chunk_id', 0)
            self.log_progress(f"Resuming from chunk {resume_from_chunk}")
            
            # Load existing voids if available
            void_chunk_dir = self.processed_data_dir / "void_chunks"
            if void_chunk_dir.exists():
                for chunk_file in sorted(void_chunk_dir.glob("chunk_*.pkl")):
                    chunk_num = int(chunk_file.stem.split('_')[1])
                    if chunk_num < resume_from_chunk:
                        existing_voids.append(pd.read_pickle(chunk_file))
        
        # Create chunk directory
        void_chunk_dir = self.processed_data_dir / "void_chunks"
        void_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        all_voids = existing_voids.copy()
        
        # Create progress bar for void finding
        # Use logger writer so progress goes to logs
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
        
        logger_writer = LoggerWriter(logger)
        
        # Process chunks sequentially (VAST handles all CPU parallelization internally)
        # We chunk data for memory/performance reasons, but VAST's num_cpus parameter
        # controls all multiprocessing - we do NOT implement our own parallelization
        with tqdm(
            total=n_chunks,
            initial=resume_from_chunk,
            desc="Finding voids",
            unit="chunk",
            file=logger_writer,
            ncols=100,
            mininterval=1.0
        ) as pbar:
            for chunk_id in range(resume_from_chunk, n_chunks):
                start_idx = chunk_id * chunk_size
                end_idx = min((chunk_id + 1) * chunk_size, n_galaxies)
                
                # Update progress bar description
                total_voids_so_far = sum(len(v) for v in all_voids)
                pbar.set_description(
                    f"Finding voids [chunk {chunk_id + 1}/{n_chunks}, "
                    f"galaxies {start_idx:,}-{end_idx:,}]"
                )
                
                chunk_galaxies = galaxy_coords.iloc[start_idx:end_idx].copy()
                
                try:
                    # Find voids in this chunk
                    # Create a temporary output directory for this chunk
                    chunk_output_dir = void_chunk_dir / f"vast_output_chunk_{chunk_id:04d}"
                    chunk_output_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Dynamically compute redshift limits for this chunk to constrain the grid volume
                    # This is critical for performance: scanning the full z=0-1 volume for every chunk is infeasible
                    # NOTE: After _convert_to_comoving, 'z' is comoving coordinate, not redshift!
                    # Use 'redshift' column if available, otherwise check if 'z' is actually redshift
                    if 'redshift' in chunk_galaxies.columns:
                        chunk_z_min = chunk_galaxies['redshift'].min()
                        chunk_z_max = chunk_galaxies['redshift'].max()
                    elif 'z' in chunk_galaxies.columns:
                        # Check if z values are reasonable redshifts (0 < z < 10) vs comoving coords (can be negative/large)
                        z_values = chunk_galaxies['z'].values
                        if np.all((z_values > 0) & (z_values < 10)):
                            # Looks like redshift
                            chunk_z_min = z_values.min()
                            chunk_z_max = z_values.max()
                        else:
                            # Looks like comoving coordinates - use global z_min/z_max
                            logger.warning(f"  Chunk z column appears to be comoving coordinates, using global redshift range")
                            chunk_z_min = z_min
                            chunk_z_max = z_max
                    else:
                        # Fallback to global range
                        chunk_z_min = z_min
                        chunk_z_max = z_max
                    
                    # Add small buffer to redshift limits to avoid edge effects
                    z_buffer = 0.05
                    # Ensure we don't go beyond global limits
                    effective_z_min = max(z_min, chunk_z_min - z_buffer)
                    effective_z_max = min(z_max, chunk_z_max + z_buffer)
                    
                    self.log_progress(f"  Chunk redshift range: {chunk_z_min:.3f}-{chunk_z_max:.3f} "
                                    f"(Grid: {effective_z_min:.3f}-{effective_z_max:.3f})")

                    # Check for VAST checkpoint file in output directory
                    chunk_survey_name = f"{catalog_name}_chunk_{chunk_id:04d}"
                    checkpoint_file = chunk_output_dir / f"{chunk_survey_name}VoidFinderCheckpoint.h5"
                    chunk_use_checkpoint = use_start_checkpoint and checkpoint_file.exists()
                    
                    if chunk_use_checkpoint:
                        self.log_progress(f"  Found VAST checkpoint: {checkpoint_file.name}")
                    
                    chunk_voids = self.voidfinder.find_voids(
                        chunk_galaxies,
                        min_radius=min_radius,
                        max_radius=max_radius,
                        survey_name=chunk_survey_name,
                        out_directory=chunk_output_dir,
                        z_min=effective_z_min,
                        z_max=effective_z_max,
                        catalog_name=catalog_name,  # Explicitly pass catalog config name
                        grid_size=grid_size,  # Pass grid size parameter
                        num_cpus=num_cpus,  # None = use all physical cores (VAST default)
                        save_after=save_after,  # None = disable checkpointing
                        use_start_checkpoint=chunk_use_checkpoint  # Resume from VAST checkpoint if exists
                    )
                    
                    if len(chunk_voids) > 0:
                        # Save chunk checkpoint
                        chunk_file = void_chunk_dir / f"chunk_{chunk_id:04d}.pkl"
                        chunk_voids.to_pickle(chunk_file)
                        
                        all_voids.append(chunk_voids)
                        
                        # Update checkpoint
                        total_voids = sum(len(v) for v in all_voids)
                        self.checkpoint_manager.update_voidfinding_progress(
                            chunk_id + 1, n_chunks, total_voids,
                            {'min_radius': min_radius, 'max_radius': max_radius}
                        )
                        
                        # Update progress bar
                        pbar.update(1)
                        pbar.set_postfix({
                            'voids': f'{total_voids:,}',
                            'chunk_voids': len(chunk_voids)
                        })
                        
                        logger.info(f"✓ Chunk {chunk_id + 1}/{n_chunks}: {len(chunk_voids)} voids "
                                  f"(total: {total_voids:,})")
                    else:
                        pbar.update(1)
                        logger.info(f"✓ Chunk {chunk_id + 1}/{n_chunks}: No voids found")
                        
                except Exception as e:
                    logger.error(f"Error processing chunk {chunk_id + 1}: {e}")
                    pbar.update(1)  # Still update progress even on error
                    # Continue with other chunks
                    continue
        
        # Combine all voids
        if len(all_voids) == 0:
            return pd.DataFrame()
        
        logger.info("Combining void chunks...")
        with tqdm(total=len(all_voids), desc="Combining voids", unit="chunk", 
                 file=logger_writer, leave=False) as combine_pbar:
            combined = pd.concat(all_voids, ignore_index=True)
            combine_pbar.update(len(all_voids))
        
        # Remove duplicates (voids that overlap significantly)
        logger.info("Removing duplicate voids...")
        combined = self._remove_duplicate_voids(combined)
        
        # Clear checkpoint on success
        self.checkpoint_manager.clear_checkpoint('voidfinding')
        
        # Clean up chunk files
        self.log_progress("Cleaning up chunk files...")
        for chunk_file in void_chunk_dir.glob("chunk_*.pkl"):
            chunk_file.unlink()
        
        return combined
    
    def _find_voids_redshift_binned(self, galaxy_coords: pd.DataFrame,
                                    min_radius: float, max_radius: float,
                                    chunk_size: int, z_min: float, z_max: float,
                                    z_bin_size: float, catalog_name: str = 'sdss_dr16',
                                    grid_size: float = 50.0,
                                    num_cpus: Optional[int] = None,
                                    save_after: Optional[int] = None,
                                    use_start_checkpoint: bool = False) -> pd.DataFrame:
        """
        Find voids by processing in redshift bins.
        
        This splits the redshift range into bins and processes each bin separately,
        which dramatically reduces the volume per bin and speeds up processing.
        
        Parameters:
            galaxy_coords: DataFrame with x, y, z, redshift columns
            min_radius: Minimum void radius
            max_radius: Maximum void radius
            chunk_size: Number of galaxies per chunk (within each redshift bin)
            z_min, z_max: Global redshift range
            z_bin_size: Size of each redshift bin (e.g., 0.05 for bins of 0.05 width)
            catalog_name: Name of catalog (for config lookup)
            grid_size: Grid edge length in Mpc (default: 50.0)
            num_cpus: Number of CPUs for VAST (None = all physical cores)
            save_after: Save checkpoint after N cells (None = disabled)
            use_start_checkpoint: Resume from VAST checkpoint if found
            
        Returns:
            Combined void catalog DataFrame from all redshift bins
        """
        # Create redshift bins
        z_bins = []
        current_z = z_min
        while current_z < z_max:
            bin_z_max = min(current_z + z_bin_size, z_max)
            z_bins.append((current_z, bin_z_max))
            current_z = bin_z_max
        
        self.log_progress(f"Processing {len(z_bins)} redshift bins: {z_bins}")
        
        all_voids = []
        
        # Process each redshift bin
        for bin_idx, (bin_z_min, bin_z_max) in enumerate(z_bins):
            self.log_progress(f"Redshift bin {bin_idx + 1}/{len(z_bins)}: z={bin_z_min:.3f}-{bin_z_max:.3f}")
            
            # Filter galaxies in this redshift bin
            if 'redshift' in galaxy_coords.columns:
                bin_mask = (galaxy_coords['redshift'] >= bin_z_min) & (galaxy_coords['redshift'] < bin_z_max)
            elif 'z' in galaxy_coords.columns:
                # Check if z is redshift or comoving coordinate
                z_values = galaxy_coords['z'].values
                if np.all((z_values > 0) & (z_values < 10)):
                    # Looks like redshift
                    bin_mask = (galaxy_coords['z'] >= bin_z_min) & (galaxy_coords['z'] < bin_z_max)
                else:
                    # Comoving coordinates - need to convert back to redshift
                    # This is complex, so skip binning for comoving coords
                    logger.warning("Cannot bin by redshift: 'z' column appears to be comoving coordinates")
                    # Fall back to processing all at once
                    return self._find_voids_chunked(
                        galaxy_coords, min_radius, max_radius, chunk_size, z_min, z_max,
                        catalog_name, grid_size, num_cpus, save_after, use_start_checkpoint
                    )
            else:
                logger.warning("Cannot bin by redshift: no redshift column found")
                # Fall back to processing all at once
                return self._find_voids_chunked(
                    galaxy_coords, min_radius, max_radius, chunk_size, z_min, z_max,
                    catalog_name, grid_size, num_cpus, save_after, use_start_checkpoint
                )
            
            bin_galaxies = galaxy_coords[bin_mask].copy()
            
            if len(bin_galaxies) == 0:
                self.log_progress(f"  No galaxies in bin z={bin_z_min:.3f}-{bin_z_max:.3f}, skipping")
                continue
            
            self.log_progress(f"  Bin contains {len(bin_galaxies):,} galaxies")
            
            # Process this redshift bin (with its own chunking)
            bin_voids = self._find_voids_chunked(
                bin_galaxies, min_radius, max_radius, chunk_size, bin_z_min, bin_z_max,
                catalog_name, grid_size, num_cpus, save_after, use_start_checkpoint
            )
            
            if len(bin_voids) > 0:
                all_voids.append(bin_voids)
                self.log_progress(f"  ✓ Bin {bin_idx + 1}: Found {len(bin_voids):,} voids")
            else:
                self.log_progress(f"  ✓ Bin {bin_idx + 1}: No voids found")
        
        # Combine all bins
        if len(all_voids) == 0:
            return pd.DataFrame()
        
        self.log_progress(f"Combining {len(all_voids)} redshift bins...")
        combined = pd.concat(all_voids, ignore_index=True)
        
        # Remove duplicates across bins (voids near bin boundaries might overlap)
        combined = self._remove_duplicate_voids(combined)
        
        self.log_progress(f"✓ Total voids from all bins: {len(combined):,}")
        
        return combined
    
    def _remove_duplicate_voids(self, void_catalog: pd.DataFrame,
                               min_separation: float = 5.0) -> pd.DataFrame:
        """
        Remove duplicate voids that are too close together.
        
        Parameters:
            void_catalog: Void catalog DataFrame
            min_separation: Minimum separation in Mpc
            
        Returns:
            Deduplicated catalog
        """
        if len(void_catalog) == 0:
            return void_catalog
        
        from pipeline.common.void_distances import compute_pairwise_distances
        
        positions = void_catalog[['x', 'y', 'z']].values
        
        # Use modular distance computation with automatic chunking
        n_voids = len(void_catalog)
        if n_voids > 1000:
            logger.info(f"Computing distances for {n_voids:,} voids (this may take a while)...")
        
        distances = compute_pairwise_distances(positions)
        
        # Mark duplicates with progress bar for large catalogs
        keep_mask = np.ones(len(void_catalog), dtype=bool)
        if n_voids > 1000:
            iterator = tqdm(range(len(void_catalog)), desc="Removing duplicates", 
                          unit="void", file=logger_writer, leave=False)
        else:
            iterator = range(len(void_catalog))
        
        for i in iterator:
            if keep_mask[i]:
                # Find close neighbors
                close = np.where((distances[i, :] < min_separation) & 
                                (distances[i, :] > 0) & 
                                (np.arange(len(void_catalog)) > i))[0]
                keep_mask[close] = False
        
        filtered = void_catalog[keep_mask].copy()
        
        if len(filtered) < len(void_catalog):
            self.log_progress(f"Removed {len(void_catalog) - len(filtered)} duplicate voids")
        
        return filtered
    
    def _save_void_catalog(self, catalog_name: str, void_catalog: pd.DataFrame) -> Path:
        """
        Save void catalog to processed_data directory.
        
        Parameters:
            catalog_name: Name of source catalog
            void_catalog: Void catalog DataFrame
            
        Returns:
            Path to saved file
        """
        output_file = self.processed_data_dir / f"voidfinder_{catalog_name}.pkl"
        
        void_catalog.to_pickle(output_file)
        
        self.log_progress(f"✓ Saved void catalog: {output_file}")
        self.log_progress(f"  {len(void_catalog):,} voids")
        
        return output_file
    
    def _compute_catalog_statistics(self, galaxy_catalog: pd.DataFrame,
                                    z_min: float, z_max: float, mag_limit: float) -> Dict[str, Any]:
        """Compute comprehensive statistics for the galaxy catalog."""
        stats = {
            'n_galaxies': len(galaxy_catalog),
            'redshift_range': {'min': z_min, 'max': z_max},
            'magnitude_limit': mag_limit
        }
        
        # Redshift statistics
        if 'z' in galaxy_catalog.columns:
            z_values = galaxy_catalog['z'].values
            stats['redshift_statistics'] = {
                'min': float(np.min(z_values)),
                'max': float(np.max(z_values)),
                'mean': float(np.mean(z_values)),
                'median': float(np.median(z_values)),
                'std': float(np.std(z_values)),
                'percentiles': {
                    'p5': float(np.percentile(z_values, 5)),
                    'p25': float(np.percentile(z_values, 25)),
                    'p75': float(np.percentile(z_values, 75)),
                    'p95': float(np.percentile(z_values, 95))
                }
            }
        
        # Magnitude statistics
        mag_cols = ['magnitude', 'r_mag', 'petroMag_r']
        mag_col = None
        for col in mag_cols:
            if col in galaxy_catalog.columns:
                mag_col = col
                break
        
        if mag_col:
            mag_values = galaxy_catalog[mag_col].values
            stats['magnitude_statistics'] = {
                'min': float(np.min(mag_values)),
                'max': float(np.max(mag_values)),
                'mean': float(np.mean(mag_values)),
                'median': float(np.median(mag_values)),
                'std': float(np.std(mag_values))
            }
        
        # Spatial statistics (if comoving coordinates available)
        if all(col in galaxy_catalog.columns for col in ['x', 'y', 'z']):
            from pipeline.common.void_stats import calculate_effective_volume, calculate_mean_separation
            
            x, y, z = galaxy_catalog[['x', 'y', 'z']].values.T
            positions = galaxy_catalog[['x', 'y', 'z']].values
            
            # Compute volume using modular function (bounding box method for consistency)
            volume = calculate_effective_volume(positions, method='bounding_box')
            
            # Compute mean separation for density understanding
            mean_sep = calculate_mean_separation(positions)
            
            stats['spatial_statistics'] = {
                'x_range': [float(np.min(x)), float(np.max(x))],
                'y_range': [float(np.min(y)), float(np.max(y))],
                'z_range': [float(np.min(z)), float(np.max(z))],
                'volume_mpc3': volume,
                'galaxy_density_mpc3': len(galaxy_catalog) / volume if volume > 0 else 0.0,
                'mean_separation_mpc': mean_sep,
                'comoving_distance_range': {
                    'min': float(np.min(np.sqrt(x**2 + y**2 + z**2))),
                    'max': float(np.max(np.sqrt(x**2 + y**2 + z**2)))
                }
            }
        
        return stats
    
    def _compute_void_statistics(self, void_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Compute comprehensive statistics for the void catalog."""
        if len(void_catalog) == 0:
            return {'n_voids': 0, 'empty': True}
        
        stats = {
            'n_voids': len(void_catalog),
            'empty': False
        }
        
        # Radius statistics
        if 'radius_mpc' in void_catalog.columns:
            radii = void_catalog['radius_mpc'].values
            stats['radius_statistics'] = {
                'min': float(np.min(radii)),
                'max': float(np.max(radii)),
                'mean': float(np.mean(radii)),
                'median': float(np.median(radii)),
                'std': float(np.std(radii)),
                'percentiles': {
                    'p5': float(np.percentile(radii, 5)),
                    'p25': float(np.percentile(radii, 25)),
                    'p75': float(np.percentile(radii, 75)),
                    'p95': float(np.percentile(radii, 95))
                }
            }
        
        # Volume statistics
        if 'volume_mpc3' in void_catalog.columns:
            volumes = void_catalog['volume_mpc3'].values
            stats['volume_statistics'] = {
                'min': float(np.min(volumes)),
                'max': float(np.max(volumes)),
                'mean': float(np.mean(volumes)),
                'median': float(np.median(volumes)),
                'std': float(np.std(volumes)),
                'total_volume_mpc3': float(np.sum(volumes))
            }
        
        # Spatial distribution
        if all(col in void_catalog.columns for col in ['x', 'y', 'z']):
            x, y, z = void_catalog[['x', 'y', 'z']].values.T
            
            stats['spatial_distribution'] = {
                'x_range': [float(np.min(x)), float(np.max(x))],
                'y_range': [float(np.min(y)), float(np.max(y))],
                'z_range': [float(np.min(z)), float(np.max(z))],
                'mean_position': [float(np.mean(x)), float(np.mean(y)), float(np.mean(z))],
                'std_position': [float(np.std(x)), float(np.std(y)), float(np.std(z))]
            }
            
            # Compute void density (will be computed from catalog stats if available)
            stats['void_density_mpc3'] = None  # Will be computed in summary if catalog stats available
        
        # Redshift distribution (if available)
        if 'redshift' in void_catalog.columns:
            redshifts = void_catalog['redshift'].values
            stats['redshift_statistics'] = {
                'min': float(np.min(redshifts)),
                'max': float(np.max(redshifts)),
                'mean': float(np.mean(redshifts)),
                'median': float(np.median(redshifts)),
                'std': float(np.std(redshifts))
            }
        
        # Edge flag statistics
        if 'edge_flag' in void_catalog.columns:
            edge_flags = void_catalog['edge_flag'].values
            stats['edge_statistics'] = {
                'n_edge_voids': int(np.sum(edge_flags)),
                'n_interior_voids': int(len(edge_flags) - np.sum(edge_flags)),
                'edge_fraction': float(np.mean(edge_flags))
            }
        
        return stats
    
    def _get_vast_parameters(self, galaxy_coords: pd.DataFrame,
                            z_min: float, z_max: float,
                            min_radius: float, max_radius: float) -> Dict[str, Any]:
        """Get VAST parameters that were computed and used."""
        from .vast_parameters import VASTParameterConfig
        
        param_config = VASTParameterConfig('sdss_dr16')
        vast_params = param_config.compute_parameters(galaxy_coords, z_min, z_max)
        
        # Add user-specified parameters
        vast_params['min_maximal_radius'] = min_radius
        vast_params['max_radius_filter'] = max_radius
        
        # Estimate expected void count
        if 'xyz_limits' in vast_params:
            lims = vast_params['xyz_limits']
            # Handle both list of lists and numpy array
            if hasattr(lims, 'shape') and lims.shape == (2, 3):
                volume = (lims[1, 0] - lims[0, 0]) * (lims[1, 1] - lims[0, 1]) * (lims[1, 2] - lims[0, 2])
            elif isinstance(lims, list) and len(lims) == 2 and len(lims[0]) == 3:
                volume = (lims[1][0] - lims[0][0]) * (lims[1][1] - lims[0][1]) * (lims[1][2] - lims[0][2])
            else:
                # Fallback for old shape or different format
                try:
                    volume = np.prod([lim[1] - lim[0] for lim in lims])
                except:
                    volume = 1.0
                    
            estimated_voids = param_config.estimate_void_count(len(galaxy_coords), volume)
            vast_params['estimated_void_count'] = estimated_voids
        
        return vast_params
    
    def _get_processing_info(self, galaxy_catalog: pd.DataFrame,
                            void_catalog: pd.DataFrame,
                            chunk_size: int) -> Dict[str, Any]:
        """Get information about processing."""
        n_galaxies = len(galaxy_catalog)
        n_chunks = (n_galaxies + chunk_size - 1) // chunk_size
        
        info = {
            'chunk_size': chunk_size,
            'n_chunks': n_chunks,
            'galaxies_per_chunk_avg': n_galaxies / n_chunks if n_chunks > 0 else 0,
            'checkpointing_used': True,
            'voids_per_chunk_avg': len(void_catalog) / n_chunks if n_chunks > 0 and len(void_catalog) > 0 else 0
        }
        
        # Check checkpoint status
        checkpoint = self.checkpoint_manager.load_checkpoint('voidfinding')
        if checkpoint:
            info['checkpoint_info'] = {
                'resumed': checkpoint.get('chunk_id', 0) > 0,
                'last_chunk': checkpoint.get('chunk_id', 0),
                'total_chunks': checkpoint.get('total_chunks', n_chunks)
            }
        
        return info
    
    def _generate_summary(self, void_catalog: pd.DataFrame,
                         catalog_stats: Dict[str, Any],
                         void_stats: Dict[str, Any],
                         vast_params: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive summary statistics."""
        summary = {
            'n_voids': len(void_catalog),
            'n_galaxies': catalog_stats.get('n_galaxies', 0),
            'void_finding_efficiency': len(void_catalog) / catalog_stats.get('n_galaxies', 1) if catalog_stats.get('n_galaxies', 0) > 0 else 0.0
        }
        
        # Add radius summary
        if 'radius_statistics' in void_stats:
            summary['mean_void_radius_mpc'] = void_stats['radius_statistics']['mean']
            summary['median_void_radius_mpc'] = void_stats['radius_statistics']['median']
        
        # Add comparison with expected
        if 'estimated_void_count' in vast_params:
            estimated = vast_params['estimated_void_count']
            actual = len(void_catalog)
            ratio = actual / estimated if estimated > 0 else 0.0
            summary['void_count_comparison'] = {
                'estimated': estimated,
                'actual': actual,
                'ratio': ratio,
                'within_expectations': 0.1 <= ratio <= 10.0
            }
        
        # Compute void density if catalog stats available
        if 'spatial_statistics' in catalog_stats:
            survey_volume = catalog_stats['spatial_statistics'].get('volume_mpc3', 1.0)
            if survey_volume > 0:
                summary['void_density_mpc3'] = len(void_catalog) / survey_volume
                if 'void_statistics' in void_stats:
                    void_stats['void_density_mpc3'] = summary['void_density_mpc3']
        
        return summary
    
    def validate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform basic validation of void catalog.
        
        Parameters:
            context (dict, optional): Validation parameters
            
        Returns:
            dict: Validation results
        """
        catalog_name = context.get('catalog', 'sdss_dr16') if context else 'sdss_dr16'
        
        void_file = self.processed_data_dir / f"voidfinder_{catalog_name}.pkl"
        
        if not void_file.exists():
            return {
                'valid': False,
                'error': f'Void catalog not found: {void_file}'
            }
        
        try:
            void_catalog = pd.read_pickle(void_file)
            
            # Basic validation checks
            required_cols = ['x', 'y', 'z', 'radius_mpc']
            missing = [col for col in required_cols if col not in void_catalog.columns]
            
            if missing:
                return {
                    'valid': False,
                    'error': f'Missing required columns: {missing}'
                }
            
            # Check for reasonable values
            issues = []
            
            if void_catalog['radius_mpc'].min() < 0:
                issues.append('Negative radii found')
            
            if void_catalog['radius_mpc'].max() > 1000:
                issues.append('Unusually large radii found')
            
            if len(void_catalog) == 0:
                issues.append('Empty catalog')
            
            return {
                'valid': len(issues) == 0,
                'n_voids': len(void_catalog),
                'issues': issues if issues else None
            }
            
        except Exception as e:
            return {
                'valid': False,
                'error': f'Failed to load catalog: {e}'
            }
    
    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform extended validation (Monte Carlo, bootstrap, etc.).
        
        Parameters:
            context (dict, optional): Extended validation parameters
            
        Returns:
            dict: Extended validation results
        """
        # For now, return basic extended validation
        # Can be expanded with statistical tests
        basic_validation = self.validate(context)
        
        return {
            'basic_validation': basic_validation,
            'extended_tests': {
                'note': 'Extended validation not yet implemented'
            }
        }
    
    def _generate_comprehensive_report(self, results: Dict[str, Any]) -> str:
        """
        Generate comprehensive markdown report with all statistics and parameters.
        
        Parameters:
            results: Pipeline results dictionary
            
        Returns:
            str: Path to generated report file
        """
        from datetime import datetime
        
        report_path = self.output_dir / "reports" / "voidfinder_analysis_report.md"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(report_path, 'w') as f:
            # Header
            f.write("# VAST VoidFinder Pipeline Analysis Report\n\n")
            f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Pipeline:** voidfinder\n")
            f.write(f"**Catalog:** {results.get('catalog_name', 'unknown')}\n\n")
            f.write("---\n\n")
            
            # Executive Summary
            f.write("## Executive Summary\n\n")
            summary = results.get('summary', {})
            f.write(f"- **Galaxies Processed:** {summary.get('n_galaxies', 0):,}\n")
            f.write(f"- **Voids Found:** {summary.get('n_voids', 0):,}\n")
            if 'void_count_comparison' in summary:
                comp = summary['void_count_comparison']
                f.write(f"- **Expected Voids:** {comp.get('estimated', 0):,}\n")
                f.write(f"- **Void Finding Efficiency:** {comp.get('ratio', 0):.2%}\n")
                status = "✓" if comp.get('within_expectations', False) else "⚠"
                f.write(f"- **Within Expectations:** {status}\n")
            f.write("\n")
            
            # Catalog Statistics
            f.write("## Galaxy Catalog Statistics\n\n")
            cat_stats = results.get('catalog_statistics', {})
            if cat_stats:
                f.write(f"### Basic Properties\n\n")
                f.write(f"- **Total Galaxies:** {cat_stats.get('n_galaxies', 0):,}\n")
                f.write(f"- **Redshift Range:** {cat_stats.get('redshift_range', {}).get('min', 0):.3f} - {cat_stats.get('redshift_range', {}).get('max', 0):.3f}\n")
                f.write(f"- **Magnitude Limit:** {cat_stats.get('magnitude_limit', 0):.2f}\n\n")
                
                if 'redshift_statistics' in cat_stats:
                    z_stats = cat_stats['redshift_statistics']
                    f.write(f"### Redshift Distribution\n\n")
                    f.write(f"- **Mean:** {z_stats.get('mean', 0):.3f}\n")
                    f.write(f"- **Median:** {z_stats.get('median', 0):.3f}\n")
                    f.write(f"- **Std Dev:** {z_stats.get('std', 0):.3f}\n")
                    f.write(f"- **Range:** {z_stats.get('min', 0):.3f} - {z_stats.get('max', 0):.3f}\n")
                    f.write(f"- **Percentiles:** P5={z_stats.get('percentiles', {}).get('p5', 0):.3f}, "
                           f"P25={z_stats.get('percentiles', {}).get('p25', 0):.3f}, "
                           f"P75={z_stats.get('percentiles', {}).get('p75', 0):.3f}, "
                           f"P95={z_stats.get('percentiles', {}).get('p95', 0):.3f}\n\n")
                
                if 'spatial_statistics' in cat_stats:
                    sp_stats = cat_stats['spatial_statistics']
                    f.write(f"### Spatial Properties\n\n")
                    f.write(f"- **Volume:** {sp_stats.get('volume_mpc3', 0):,.1f} Mpc³\n")
                    f.write(f"- **Galaxy Density:** {sp_stats.get('galaxy_density_mpc3', 0):.6f} galaxies/Mpc³\n")
                    f.write(f"- **Comoving Distance Range:** {sp_stats.get('comoving_distance_range', {}).get('min', 0):.1f} - "
                           f"{sp_stats.get('comoving_distance_range', {}).get('max', 0):.1f} Mpc\n")
                    f.write(f"- **XYZ Ranges:**\n")
                    f.write(f"  - X: {sp_stats.get('x_range', [0, 0])[0]:.1f} - {sp_stats.get('x_range', [0, 0])[1]:.1f} Mpc\n")
                    f.write(f"  - Y: {sp_stats.get('y_range', [0, 0])[0]:.1f} - {sp_stats.get('y_range', [0, 0])[1]:.1f} Mpc\n")
                    f.write(f"  - Z: {sp_stats.get('z_range', [0, 0])[0]:.1f} - {sp_stats.get('z_range', [0, 0])[1]:.1f} Mpc\n\n")
            
            # VAST Parameters
            f.write("## VAST VoidFinder Parameters\n\n")
            vast_params = results.get('vast_parameters', {})
            if vast_params:
                f.write(f"### Algorithm Configuration\n\n")
                f.write(f"- **Survey Name:** {vast_params.get('survey_name', 'N/A')}\n")
                f.write(f"- **Mask Type:** {vast_params.get('mask_type', 'N/A')}\n")
                f.write(f"- **Min Maximal Radius:** {vast_params.get('min_maximal_radius', 0):.1f} Mpc\n")
                f.write(f"- **Max Radius Filter:** {vast_params.get('max_radius_filter', 0):.1f} Mpc\n")
                f.write(f"- **Hole Grid Edge Length:** {vast_params.get('hole_grid_edge_length', 0):.1f} Mpc\n")
                f.write(f"- **Galaxy Map Grid Edge Length:** {vast_params.get('galaxy_map_grid_edge_length', 0):.1f} Mpc\n")
                f.write(f"- **Points per Unit Volume:** {vast_params.get('pts_per_unit_volume', 0):.4f}\n")
                f.write(f"- **Max Hole Mask Overlap:** {vast_params.get('max_hole_mask_overlap', 0):.2f}\n")
                f.write(f"- **Check Only Empty Cells:** {vast_params.get('check_only_empty_cells', False)}\n\n")
                
                f.write(f"### Distance and Spatial Limits\n\n")
                if 'dist_limits' in vast_params:
                    dist_lims = vast_params['dist_limits']
                    f.write(f"- **Distance Limits:** {dist_lims[0]:.1f} - {dist_lims[1]:.1f} Mpc\n")
                if 'xyz_limits' in vast_params:
                    xyz_lims = vast_params['xyz_limits']
                    f.write(f"- **XYZ Limits:**\n")
                    f.write(f"  - X: {xyz_lims[0][0]:.1f} - {xyz_lims[0][1]:.1f} Mpc\n")
                    f.write(f"  - Y: {xyz_lims[1][0]:.1f} - {xyz_lims[1][1]:.1f} Mpc\n")
                    f.write(f"  - Z: {xyz_lims[2][0]:.1f} - {xyz_lims[2][1]:.1f} Mpc\n")
                if 'grid_origin' in vast_params:
                    origin = vast_params['grid_origin']
                    f.write(f"- **Grid Origin:** [{origin[0]:.1f}, {origin[1]:.1f}, {origin[2]:.1f}] Mpc\n")
                f.write("\n")
            
            # Void Statistics
            f.write("## Void Catalog Statistics\n\n")
            void_stats = results.get('void_statistics', {})
            if void_stats and not void_stats.get('empty', False):
                f.write(f"### Basic Properties\n\n")
                f.write(f"- **Total Voids:** {void_stats.get('n_voids', 0):,}\n\n")
                
                if 'radius_statistics' in void_stats:
                    r_stats = void_stats['radius_statistics']
                    f.write(f"### Radius Distribution\n\n")
                    f.write(f"- **Mean:** {r_stats.get('mean', 0):.2f} Mpc\n")
                    f.write(f"- **Median:** {r_stats.get('median', 0):.2f} Mpc\n")
                    f.write(f"- **Std Dev:** {r_stats.get('std', 0):.2f} Mpc\n")
                    f.write(f"- **Range:** {r_stats.get('min', 0):.2f} - {r_stats.get('max', 0):.2f} Mpc\n")
                    f.write(f"- **Percentiles:** P5={r_stats.get('percentiles', {}).get('p5', 0):.2f}, "
                           f"P25={r_stats.get('percentiles', {}).get('p25', 0):.2f}, "
                           f"P75={r_stats.get('percentiles', {}).get('p75', 0):.2f}, "
                           f"P95={r_stats.get('percentiles', {}).get('p95', 0):.2f} Mpc\n\n")
                
                if 'volume_statistics' in void_stats:
                    v_stats = void_stats['volume_statistics']
                    f.write(f"### Volume Distribution\n\n")
                    f.write(f"- **Mean:** {v_stats.get('mean', 0):,.0f} Mpc³\n")
                    f.write(f"- **Median:** {v_stats.get('median', 0):,.0f} Mpc³\n")
                    f.write(f"- **Total Volume:** {v_stats.get('total_volume_mpc3', 0):,.0f} Mpc³\n")
                    f.write(f"- **Range:** {v_stats.get('min', 0):,.0f} - {v_stats.get('max', 0):,.0f} Mpc³\n\n")
                
                if 'spatial_distribution' in void_stats:
                    sp_dist = void_stats['spatial_distribution']
                    f.write(f"### Spatial Distribution\n\n")
                    f.write(f"- **Mean Position:** [{sp_dist.get('mean_position', [0, 0, 0])[0]:.1f}, "
                           f"{sp_dist.get('mean_position', [0, 0, 0])[1]:.1f}, "
                           f"{sp_dist.get('mean_position', [0, 0, 0])[2]:.1f}] Mpc\n")
                    f.write(f"- **Std Position:** [{sp_dist.get('std_position', [0, 0, 0])[0]:.1f}, "
                           f"{sp_dist.get('std_position', [0, 0, 0])[1]:.1f}, "
                           f"{sp_dist.get('std_position', [0, 0, 0])[2]:.1f}] Mpc\n\n")
                
                if 'edge_statistics' in void_stats:
                    e_stats = void_stats['edge_statistics']
                    f.write(f"### Edge Effects\n\n")
                    f.write(f"- **Interior Voids:** {e_stats.get('n_interior_voids', 0):,}\n")
                    f.write(f"- **Edge Voids:** {e_stats.get('n_edge_voids', 0):,}\n")
                    f.write(f"- **Edge Fraction:** {e_stats.get('edge_fraction', 0):.2%}\n\n")
            else:
                f.write("*No voids found in catalog.*\n\n")
            
            # Processing Information
            f.write("## Processing Information\n\n")
            proc_info = results.get('processing_info', {})
            if proc_info:
                f.write(f"- **Chunk Size:** {proc_info.get('chunk_size', 0):,} galaxies\n")
                f.write(f"- **Number of Chunks:** {proc_info.get('n_chunks', 0)}\n")
                f.write(f"- **Average Galaxies per Chunk:** {proc_info.get('galaxies_per_chunk_avg', 0):,.0f}\n")
                f.write(f"- **Checkpointing Used:** {proc_info.get('checkpointing_used', False)}\n")
                if 'checkpoint_info' in proc_info:
                    cp_info = proc_info['checkpoint_info']
                    f.write(f"- **Resumed from Checkpoint:** {cp_info.get('resumed', False)}\n")
                    if cp_info.get('resumed', False):
                        f.write(f"- **Resumed from Chunk:** {cp_info.get('last_chunk', 0)}\n")
                f.write("\n")
            
            # Output Files
            f.write("## Output Files\n\n")
            f.write(f"- **Void Catalog:** `{results.get('void_catalog_file', 'N/A')}`\n\n")
            
            # Conclusion
            f.write("## Conclusion\n\n")
            if summary.get('n_voids', 0) > 0:
                f.write(f"Successfully identified {summary.get('n_voids', 0):,} cosmic voids from "
                       f"{summary.get('n_galaxies', 0):,} galaxies using VAST VoidFinder algorithm.\n\n")
                if 'void_count_comparison' in summary:
                    comp = summary['void_count_comparison']
                    if comp.get('within_expectations', False):
                        f.write("Void count is within expected range based on catalog properties.\n\n")
                    else:
                        f.write(f"⚠ Void count ({comp.get('actual', 0):,}) differs from expected "
                               f"({comp.get('estimated', 0):,}). This may indicate:\n")
                        f.write("- Catalog-specific systematics requiring parameter adjustment\n")
                        f.write("- Survey completeness issues\n")
                        f.write("- Algorithm parameter tuning needed\n\n")
            else:
                f.write("⚠ No voids were identified. This may indicate:\n")
                f.write("- VAST parameter tuning required for this catalog\n")
                f.write("- Survey completeness or data quality issues\n")
                f.write("- Algorithm configuration needs adjustment\n\n")
        
        self.log_progress(f"✓ Comprehensive report generated: {report_path}")
        return str(report_path)

