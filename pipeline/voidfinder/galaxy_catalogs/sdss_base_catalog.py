"""
SDSS Base Catalog Provider
==========================

Base class for SDSS data release catalog providers (DR7, DR16, etc.)
Contains all common functionality for downloading SDSS galaxy catalogs.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import time
import logging
from tqdm import tqdm

try:
    from astroquery.sdss import SDSS
    ASTROQUERY_AVAILABLE = True
except ImportError:
    ASTROQUERY_AVAILABLE = False

from .base_catalog import BaseGalaxyCatalog

logger = logging.getLogger(__name__)


class SDSSBaseCatalog(BaseGalaxyCatalog):
    """
    Base class for SDSS catalog providers.
    
    Provides common functionality for downloading SDSS galaxy catalogs
    with parameter-specific caching, checkpointing, and batch processing.
    """
    
    def __init__(self, downloaded_data_dir: Path, processed_data_dir: Path):
        """Initialize SDSS base catalog provider."""
        super().__init__(downloaded_data_dir, processed_data_dir)
        # Cache base directory (subclasses set catalog-specific subdirectory)
        self.cache_base_dir = None  # Set by subclasses
    
    @property
    def required_columns(self) -> list:
        """Return required column names."""
        return ['ra', 'dec', 'z', 'objid']
    
    def _get_cache_filename(self, z_min: float, z_max: float, mag_limit: float) -> str:
        """
        Generate parameter-specific cache filename.
        
        Format: {catalog_name}-zmin_{z_min}-zmax_{z_max}-mag_{mag_limit}.pkl
        Decimals represented as underscores.
        
        Parameters:
            z_min: Minimum redshift
            z_max: Maximum redshift
            mag_limit: Magnitude limit
            
        Returns:
            Cache filename string
        """
        z_min_str = str(z_min).replace('.', '_')
        z_max_str = str(z_max).replace('.', '_')
        mag_str = str(mag_limit).replace('.', '_')
        
        return f"{self.catalog_name}-zmin_{z_min_str}-zmax_{z_max_str}-mag_{mag_str}.pkl"
    
    def _get_cache_file(self, z_min: float, z_max: float, mag_limit: float) -> Path:
        """
        Get cache file path for specific parameters.
        
        Parameters:
            z_min: Minimum redshift
            z_max: Maximum redshift
            mag_limit: Magnitude limit
            
        Returns:
            Path to cache file
        """
        if self.cache_base_dir is None:
            raise ValueError("cache_base_dir must be set by subclass")
        filename = self._get_cache_filename(z_min, z_max, mag_limit)
        return self.cache_base_dir / filename
    
    def _build_sql_query(self, z_min: float, z_max: float, mag_limit: float, 
                        last_objid: int = 0) -> str:
        """
        Build SQL query for SDSS catalog download.
        
        Subclasses should override to customize query (e.g., clean flag, data_release).
        
        Parameters:
            z_min: Minimum redshift
            z_max: Maximum redshift
            mag_limit: r-band magnitude limit
            last_objid: Last object ID for pagination
            
        Returns:
            SQL query string
        """
        return f"""
        SELECT TOP {self.batch_size}
            p.objid,
            p.ra,
            p.dec,
            s.z as z,
            s.zErr as z_err,
            p.petroMag_r as r_mag,
            p.petroMagErr_r as r_mag_err,
            p.petroMag_u as u_mag,
            p.petroMag_g as g_mag,
            p.petroMag_i as i_mag,
            p.petroMag_z as z_mag,
            p.petroRad_r as petrosian_radius,
            p.extinction_r as extinction_r
        FROM PhotoObj AS p
        JOIN SpecObj AS s ON s.bestObjID = p.objID
        WHERE s.z BETWEEN {z_min} AND {z_max}
            AND s.zWarning = 0
            AND p.petroMag_r < {mag_limit}
            AND (p.type = 3 OR p.type = 6)
            AND p.objid > {last_objid}
        ORDER BY p.objid
        """
    
    def _execute_query(self, query: str, max_retries: int = 1, 
                      retry_delay: int = 5) -> Optional[pd.DataFrame]:
        """
        Execute SDSS SQL query with optional retry logic.
        
        Subclasses can override to add data_release parameter or custom retry logic.
        
        Parameters:
            query: SQL query string
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            
        Returns:
            DataFrame with results, or None if no results
        """
        if not ASTROQUERY_AVAILABLE:
            raise ImportError("astroquery not available - required for SDSS data")
        
        for attempt in range(max_retries):
            try:
                result = SDSS.query_sql(query, timeout=600)
                if result is not None:
                    return result.to_pandas()
                return None
            except Exception as e:
                error_str = str(e).lower()
                if attempt < max_retries - 1:
                    # Check for retryable errors
                    if any(keyword in error_str for keyword in ['inconsistent', 'parse', 'column', 'timeout']):
                        logger.warning(f"Query error (attempt {attempt + 1}/{max_retries}): {e}")
                        logger.info(f"Retrying in {retry_delay} seconds...")
                        time.sleep(retry_delay)
                        retry_delay *= 2  # Exponential backoff
                        continue
                # Non-retryable or last attempt
                raise
        
        return None
    
    def _check_cache(self, cache_file: Path, z_min: float, z_max: float, 
                    mag_limit: float, force_redownload: bool) -> Optional[pd.DataFrame]:
        """
        Check if parameter-specific cache exists and is valid.
        
        Parameters:
            cache_file: Path to cache file
            z_min: Minimum redshift
            z_max: Maximum redshift
            mag_limit: Magnitude limit
            force_redownload: Whether to force re-download
            
        Returns:
            Cached DataFrame if valid, None otherwise
        """
        if force_redownload or not cache_file.exists():
            return None
        
        logger.info(f"Checking {self.catalog_name.upper()} catalog cache: {cache_file.name}")
        logger.info(f"  Parameters: z={z_min}-{z_max}, mag<{mag_limit}")
        
        try:
            cached_df = pd.read_pickle(cache_file)
            
            # Verify cache covers requested range
            if len(cached_df) > 0 and 'z' in cached_df.columns:
                cache_z_min = cached_df['z'].min()
                cache_z_max = cached_df['z'].max()
                
                # Check if cached catalog covers the FULL requested range
                if z_min >= (cache_z_min - 0.01) and z_max <= (cache_z_max + 0.01):
                    # Cache covers the range, filter to exact range
                    filtered_df = cached_df[(cached_df['z'] >= z_min) & (cached_df['z'] <= z_max)].copy()
                    logger.info(f"Cache covers requested redshift range ({z_min}-{z_max}).")
                    logger.info(f"  Cached range: {cache_z_min:.3f}-{cache_z_max:.3f} ({len(cached_df):,} galaxies)")
                    logger.info(f"  Filtered to requested range: {len(filtered_df):,} galaxies")
                    if len(filtered_df) > 0:
                        return filtered_df
                    else:
                        logger.warning(f"Cache exists but filtering to z={z_min}-{z_max} returned 0 galaxies. Redownloading.")
                else:
                    logger.info(f"Cached catalog z-range ({cache_z_min:.3f}-{cache_z_max:.3f}) does not cover requested range ({z_min}-{z_max}). Redownloading.")
            else:
                logger.info("Cached catalog is empty or invalid. Redownloading.")
        except Exception as e:
            logger.warning(f"Error checking cache: {e}. Redownloading.")
        
        return None
    
    def _load_checkpoint(self, checkpoint_manager, z_min: float, z_max: float, 
                         mag_limit: float, cache_file: Path):
        """
        Load checkpoint and validate parameters.
        
        Returns:
            None if cache should be used (checkpoint says completed and cache exists)
            tuple: (resume_from_batch, last_objid, total_batches_estimate, existing_batches) otherwise
        """
        resume_from_batch = 0
        last_objid = 0
        total_batches_estimate = self._get_initial_batch_estimate()
        existing_batches = []
        
        if not checkpoint_manager:
            return resume_from_batch, last_objid, total_batches_estimate, existing_batches
        
        checkpoint = checkpoint_manager.load_checkpoint('download')
        if checkpoint and checkpoint.get('catalog_name') == self.catalog_name:
            # Check if checkpoint parameters match current query
            checkpoint_z_min = checkpoint.get('z_min')
            checkpoint_z_max = checkpoint.get('z_max')
            checkpoint_mag_limit = checkpoint.get('mag_limit')
            
            # If redshift range or magnitude limit changed, CLEAR checkpoint and start fresh
            if (checkpoint_z_min is not None and abs(checkpoint_z_min - z_min) > 0.001) or \
               (checkpoint_z_max is not None and abs(checkpoint_z_max - z_max) > 0.001) or \
               (checkpoint_mag_limit is not None and abs(checkpoint_mag_limit - mag_limit) > 0.1):
                logger.info(f"Checkpoint parameters don't match current query:")
                logger.info(f"  Checkpoint: z={checkpoint_z_min}-{checkpoint_z_max}, mag={checkpoint_mag_limit}")
                logger.info(f"  Requested: z={z_min}-{z_max}, mag={mag_limit}")
                logger.info(f"  Clearing checkpoint and starting fresh download")
                # Clear the checkpoint to prevent confusion
                checkpoint_manager.save_checkpoint('download', {
                    'stage': 'download',
                    'catalog_name': self.catalog_name,
                    'batch_id': 0,
                    'last_objid': 0,
                    'completed': False,
                    'z_min': z_min,
                    'z_max': z_max,
                    'mag_limit': mag_limit
                })
                return resume_from_batch, last_objid, total_batches_estimate, existing_batches
            
            resume_from_batch = checkpoint.get('batch_id', 0)
            last_objid = checkpoint.get('last_objid', 0)
            total_batches_estimate = checkpoint.get('total_batches', self._get_initial_batch_estimate())
            
            # If we've already completed, check if cache exists for these exact parameters
            if checkpoint.get('completed', False):
                logger.info("Download already completed according to checkpoint")
                if cache_file.exists():
                    try:
                        cached_df = pd.read_pickle(cache_file)
                        logger.info(f"Loading completed catalog from cache: {len(cached_df):,} galaxies")
                        logger.info(f"  Cache parameters match: z={z_min}-{z_max}, mag<{mag_limit}")
                        # Return a flag to indicate cache should be used
                        return None, None, None, None  # Signal to use cache
                    except Exception as e:
                        logger.warning(f"Error loading cached catalog: {e}. Redownloading.")
                else:
                    logger.warning(f"Checkpoint says completed but cache file {cache_file.name} not found. Redownloading.")
            
            logger.info(f"Resuming download from batch {resume_from_batch}, last objid: {last_objid}")
            
            # Load existing batches (parameter-specific batch directory)
            batch_dir = cache_file.parent / f"batches_{cache_file.stem}"
            if batch_dir.exists():
                batch_files = sorted(batch_dir.glob("batch_*.pkl"))
                if len(batch_files) > 0:
                    logger.info(f"Found {len(batch_files)} existing batch files in {batch_dir.name}")
                    for batch_file in batch_files:
                        try:
                            batch_num = int(batch_file.stem.split('_')[1])
                            if batch_num < resume_from_batch:
                                batch_df = pd.read_pickle(batch_file)
                                existing_batches.append(batch_df)
                                # Update last_objid from loaded batches
                                if len(batch_df) > 0 and 'objid' in batch_df.columns:
                                    last_objid = max(last_objid, batch_df['objid'].max())
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Could not parse batch number from {batch_file.name}: {e}")
                else:
                    logger.info(f"No batch files found in {batch_dir.name}, starting fresh")
                    # Reset resume state if no batches found
                    resume_from_batch = 0
                    last_objid = 0
                    # Update checkpoint to reflect reset
                    checkpoint_manager.save_checkpoint('download', {
                        'stage': 'download',
                        'catalog_name': self.catalog_name,
                        'batch_id': 0,
                        'last_objid': 0,
                        'completed': False,
                        'z_min': z_min,
                        'z_max': z_max,
                        'mag_limit': mag_limit
                    })
            else:
                logger.info(f"Batch directory {batch_dir.name} does not exist, starting fresh")
                # Reset resume state if batch directory doesn't exist
                resume_from_batch = 0
                last_objid = 0
                # Update checkpoint to reflect reset
                checkpoint_manager.save_checkpoint('download', {
                    'stage': 'download',
                    'catalog_name': self.catalog_name,
                    'batch_id': 0,
                    'last_objid': 0,
                    'completed': False,
                    'z_min': z_min,
                    'z_max': z_max,
                    'mag_limit': mag_limit
                })
        
        return resume_from_batch, last_objid, total_batches_estimate, existing_batches
    
    def _get_initial_batch_estimate(self) -> int:
        """Get initial estimate for number of batches. Subclasses override."""
        return 60
    
    def _create_logger_writer(self):
        """Create logger writer for progress bars."""
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
        
        return LoggerWriter(logger)
    
    def _download_batches(self, cache_file: Path, checkpoint_manager, z_min: float, 
                          z_max: float, mag_limit: float, resume_from_batch: int,
                          last_objid: int, total_batches_estimate: int, 
                          existing_batches: list) -> list:
        """
        Download catalog in batches with checkpointing.
        
        Parameters:
            cache_file: Path to cache file
            checkpoint_manager: Checkpoint manager
            z_min: Minimum redshift
            z_max: Maximum redshift
            mag_limit: Magnitude limit
            resume_from_batch: Batch ID to resume from
            last_objid: Last object ID for pagination
            total_batches_estimate: Estimated total batches
            existing_batches: List of already-loaded batches
            
        Returns:
            List of batch DataFrames
        """
        # Create parameter-specific batch directory
        batch_dir = cache_file.parent / f"batches_{cache_file.stem}"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Query SDSS in batches
        all_batches = existing_batches.copy()
        batch_id = resume_from_batch
        
        # Update total_batches_estimate if we've already exceeded it
        if batch_id >= total_batches_estimate:
            total_batches_estimate = batch_id + 20  # Add buffer for remaining batches
        
        logger_writer = self._create_logger_writer()
        pbar = None
        
        try:
            # Initialize progress bar
            pbar = tqdm(
                total=total_batches_estimate,
                initial=resume_from_batch,
                desc="Downloading galaxies",
                unit="batch",
                file=logger_writer,
                ncols=100,
                mininterval=1.0
            )
            
            while True:
                # Update progress bar description
                total_rows = sum(len(b) for b in all_batches)
                pbar.set_description(f"Downloading batch {batch_id + 1} (total: {total_rows:,} galaxies)")
                
                # Build and execute query
                query = self._build_sql_query(z_min, z_max, mag_limit, last_objid)
                result_df = self._execute_query(query, max_retries=self._get_max_retries())
                
                # Check for empty result
                if result_df is None or len(result_df) == 0:
                    logger.info(f"No more data at batch {batch_id + 1} - download complete")
                    total_rows = sum(len(b) for b in all_batches)
                    total_batches_estimate = batch_id + 1
                    self._save_completion_checkpoint(checkpoint_manager, batch_id + 1, 
                                                    total_batches_estimate, total_rows, 
                                                    last_objid, z_min, z_max, mag_limit)
                    break
                
                # Get the maximum objid from this batch
                if 'objid' in result_df.columns:
                    batch_max_objid = result_df['objid'].max()
                else:
                    batch_max_objid = last_objid + len(result_df)
                
                # Save batch checkpoint
                batch_file = batch_dir / f"batch_{batch_id:04d}.pkl"
                result_df.to_pickle(batch_file)
                
                all_batches.append(result_df)
                
                # Update last_objid for next query
                last_objid = batch_max_objid
                
                # Update checkpoint with current progress
                total_rows = sum(len(b) for b in all_batches)
                
                # Dynamically update total_batches_estimate
                if batch_id + 1 >= total_batches_estimate:
                    total_batches_estimate = batch_id + 20
                    logger.info(f"Updating total batches estimate to {total_batches_estimate}")
                
                if checkpoint_manager:
                    checkpoint_manager.update_download_progress(
                        batch_id + 1, total_batches_estimate, total_rows, 
                        self.catalog_name, last_objid
                    )
                    # Save query parameters to checkpoint
                    checkpoint_data = checkpoint_manager.load_checkpoint('download')
                    if checkpoint_data:
                        checkpoint_data['z_min'] = z_min
                        checkpoint_data['z_max'] = z_max
                        checkpoint_data['mag_limit'] = mag_limit
                        checkpoint_manager.save_checkpoint('download', checkpoint_data)
                
                # Update progress bar
                if batch_id + 1 >= pbar.total:
                    pbar.total = total_batches_estimate
                
                pbar.update(1)
                pbar.set_postfix({
                    'galaxies': f'{total_rows:,}',
                    'batch_size': len(result_df)
                })
                
                logger.info(f"✓ Batch {batch_id + 1}: {len(result_df)} galaxies (total: {total_rows:,})")
                
                # If we got fewer rows than batch_size, we're done
                if len(result_df) < self.batch_size:
                    logger.info(f"Reached end of catalog (got {len(result_df):,} rows, less than batch size {self.batch_size:,})")
                    total_batches_estimate = batch_id + 1
                    pbar.total = total_batches_estimate
                    self._save_completion_checkpoint(checkpoint_manager, batch_id + 1,
                                                    total_batches_estimate, total_rows,
                                                    last_objid, z_min, z_max, mag_limit)
                    break
                
                batch_id += 1
                time.sleep(1)  # Rate limiting
        
        except Exception as e:
            logger.error(f"Error downloading batch {batch_id + 1}: {e}")
            if len(all_batches) > 0:
                logger.warning("Continuing with partial download")
            else:
                raise
        finally:
            if pbar is not None:
                pbar.close()
        
        return all_batches
    
    def _save_completion_checkpoint(self, checkpoint_manager, batch_id: int,
                                   total_batches: int, total_rows: int,
                                   last_objid: int, z_min: float, z_max: float,
                                   mag_limit: float):
        """Save completion checkpoint."""
        if checkpoint_manager:
            checkpoint_data = {
                'stage': 'download',
                'catalog_name': self.catalog_name,
                'batch_id': batch_id,
                'total_batches': total_batches,
                'rows_downloaded': total_rows,
                'last_objid': last_objid,
                'progress': 1.0,
                'completed': True,
                'z_min': z_min,
                'z_max': z_max,
                'mag_limit': mag_limit
            }
            checkpoint_manager.save_checkpoint('download', checkpoint_data)
    
    def _combine_and_save_batches(self, all_batches: list, cache_file: Path,
                                  checkpoint_manager, z_min: float, z_max: float,
                                  mag_limit: float) -> pd.DataFrame:
        """
        Combine batches and save final catalog.
        
        Parameters:
            all_batches: List of batch DataFrames
            cache_file: Path to cache file
            checkpoint_manager: Checkpoint manager
            z_min: Minimum redshift
            z_max: Maximum redshift
            mag_limit: Magnitude limit
            
        Returns:
            Combined DataFrame
        """
        if len(all_batches) == 0:
            # Check if we have any existing batches that weren't loaded
            batch_dir = cache_file.parent / f"batches_{cache_file.stem}"
            if batch_dir.exists():
                batch_files = list(batch_dir.glob("batch_*.pkl"))
                if len(batch_files) > 0:
                    logger.warning(f"Found {len(batch_files)} batch files but none were loaded. This may indicate a checkpoint mismatch.")
                    logger.info("Clearing checkpoint and batch files, starting fresh download...")
                    # Clear checkpoint
                    if checkpoint_manager:
                        checkpoint_manager.clear_checkpoint('download')
                    # Remove batch files
                    for batch_file in batch_files:
                        batch_file.unlink()
                    logger.info("Cleared old batch files. Please re-run the command.")
            raise RuntimeError("No data downloaded - checkpoint may be from a different parameter set. Clear checkpoint and retry.")
        
        logger.info(f"Combining {len(all_batches)} batches...")
        logger_writer = self._create_logger_writer()
        
        with tqdm(total=len(all_batches), desc="Combining batches", unit="batch", 
                 file=logger_writer, leave=False) as combine_pbar:
            combined = pd.concat(all_batches, ignore_index=True)
            combine_pbar.update(len(all_batches))
        
        # Remove duplicates by objid
        original_len = len(combined)
        combined = combined.drop_duplicates(subset=['objid'], keep='first')
        if len(combined) < original_len:
            logger.info(f"Removed {original_len - len(combined)} duplicate entries")
        
        # Standardize column names
        combined = combined.rename(columns={
            'ra': 'ra',
            'dec': 'dec',
            'z': 'z',
            'objid': 'objid'
        })
        
        # Add magnitude column if not present
        if 'magnitude' not in combined.columns and 'r_mag' in combined.columns:
            combined['magnitude'] = combined['r_mag']
        
        # Save final catalog to parameter-specific cache file
        logger.info(f"Saving catalog: {len(combined):,} galaxies")
        logger.info(f"  Cache file: {cache_file.name}")
        combined.to_pickle(cache_file)
        
        # Mark download as completed in checkpoint
        if checkpoint_manager:
            final_total_rows = len(combined)
            checkpoint_data = {
                'stage': 'download',
                'catalog_name': self.catalog_name,
                'batch_id': len(all_batches),
                'total_batches': len(all_batches),
                'rows_downloaded': final_total_rows,
                'last_objid': combined['objid'].max() if 'objid' in combined.columns else 0,
                'progress': 1.0,
                'completed': True,
                'z_min': z_min,
                'z_max': z_max,
                'mag_limit': mag_limit
            }
            checkpoint_manager.save_checkpoint('download', checkpoint_data)
        
        # Clean up batch files
        logger.info("Cleaning up batch files...")
        batch_dir = cache_file.parent / f"batches_{cache_file.stem}"
        if batch_dir.exists():
            for batch_file in batch_dir.glob("batch_*.pkl"):
                batch_file.unlink()
        
        logger.info(f"✓ {self.catalog_name.upper()} catalog downloaded: {len(combined):,} galaxies")
        
        return combined
    
    def download(self, checkpoint_manager=None, z_min: float = 0.0, 
                z_max: float = 1.0, mag_limit: float = 22.0,
                force_redownload: bool = False) -> pd.DataFrame:
        """
        Download SDSS catalog with parameter-specific caching.
        
        Parameters:
            checkpoint_manager: Optional CheckpointManager for resuming
            z_min: Minimum redshift
            z_max: Maximum redshift
            mag_limit: r-band magnitude limit
            force_redownload: Force re-download even if cached
            
        Returns:
            DataFrame with galaxy catalog
        """
        if not ASTROQUERY_AVAILABLE:
            raise ImportError("astroquery not available - required for SDSS data")
        
        # Get parameter-specific cache file
        cache_file = self._get_cache_file(z_min, z_max, mag_limit)
        
        # Check cache first
        cached_df = self._check_cache(cache_file, z_min, z_max, mag_limit, force_redownload)
        if cached_df is not None:
            return cached_df
        
        logger.info(f"Downloading {self.catalog_name.upper()} full spectroscopic galaxy catalog...")
        logger.info(f"  Redshift range: z = {z_min:.3f} - {z_max:.3f}")
        logger.info(f"  Magnitude limit: r < {mag_limit:.1f}")
        
        # Load checkpoint and validate parameters
        checkpoint_result = self._load_checkpoint(checkpoint_manager, z_min, z_max, mag_limit, cache_file)
        
        # Check if checkpoint indicates cache should be used (returns None)
        if checkpoint_result is None:
            # Checkpoint says completed and cache exists
            return pd.read_pickle(cache_file)
        
        resume_from_batch, last_objid, total_batches_estimate, existing_batches = checkpoint_result
        
        # Download batches
        try:
            all_batches = self._download_batches(
                cache_file, checkpoint_manager, z_min, z_max, mag_limit,
                resume_from_batch, last_objid, total_batches_estimate, existing_batches
            )
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # Save partial results if we have any
            if len(existing_batches) > 0:
                logger.warning("Saving partial download for resume")
                combined = pd.concat(existing_batches, ignore_index=True)
                combined.to_pickle(cache_file)
            raise
        
        # Combine and save
        return self._combine_and_save_batches(all_batches, cache_file, checkpoint_manager,
                                              z_min, z_max, mag_limit)
    
    def load(self, use_cache: bool = True, z_min: float = 0.0, 
             z_max: float = 1.0, mag_limit: float = 22.0) -> Optional[pd.DataFrame]:
        """
        Load catalog from parameter-specific cache.
        
        Parameters:
            use_cache: Whether to use cached data
            z_min: Minimum redshift (required for parameter-specific cache)
            z_max: Maximum redshift (required for parameter-specific cache)
            mag_limit: Magnitude limit (required for parameter-specific cache)
            
        Returns:
            DataFrame or None if not cached
        """
        if not use_cache:
            return None
        
        cache_file = self._get_cache_file(z_min, z_max, mag_limit)
        if not cache_file.exists():
            return None
        
        try:
            logger.info(f"Loading {self.catalog_name.upper()} catalog from cache: {cache_file.name}")
            logger.info(f"  Parameters: z={z_min}-{z_max}, mag<{mag_limit}")
            df = pd.read_pickle(cache_file)
            logger.info(f"Loaded {len(df):,} galaxies")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None
    
    def _get_max_retries(self) -> int:
        """Get maximum number of retries for queries. Subclasses can override."""
        return 1

