"""
SDSS DR7 Galaxy Catalog Provider
=================================

Downloads and manages SDSS DR7 full spectroscopic galaxy catalog.
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


class SDSSDR7Catalog(BaseGalaxyCatalog):
    """
    SDSS DR7 full spectroscopic galaxy catalog provider.
    
    Downloads ~600k galaxies in batches with checkpointing support.
    """
    
    def __init__(self, downloaded_data_dir: Path, processed_data_dir: Path):
        """Initialize SDSS DR7 catalog provider."""
        super().__init__(downloaded_data_dir, processed_data_dir)
        self.cache_file = self.downloaded_data_dir / "sdss_dr7" / "galaxies_full.pkl"
        self.cache_file.parent.mkdir(parents=True, exist_ok=True)
        
        # Batch size for queries (SDSS DR7 may have stricter limits than DR16)
        # Using smaller batch size for DR7 to avoid server errors
        self.batch_size = 25000  # Smaller batch size for DR7 stability
    
    @property
    def catalog_name(self) -> str:
        """Return catalog identifier."""
        return "sdss_dr7"
    
    @property
    def required_columns(self) -> list:
        """Return required column names."""
        return ['ra', 'dec', 'z', 'objid']
    
    def download(self, checkpoint_manager=None, z_min: float = 0.0, 
                z_max: float = 1.0, mag_limit: float = 22.0,
                force_redownload: bool = False) -> pd.DataFrame:
        """
        Download SDSS DR7 full spectroscopic galaxy catalog.
        
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
        
        # Check cache first
        if not force_redownload and self.cache_file.exists():
            logger.info(f"Loading SDSS DR7 catalog from cache: {self.cache_file}")
            try:
                cached_df = pd.read_pickle(self.cache_file)
                
                # Verify cache covers requested range
                if len(cached_df) > 0 and 'z' in cached_df.columns:
                    cache_z_min = cached_df['z'].min()
                    cache_z_max = cached_df['z'].max()
                    
                    # Check coverage with small buffer for float comparison
                    # If requested range is effectively within cached range, use cache
                    if z_min >= (cache_z_min - 0.01) and z_max <= (cache_z_max + 0.01):
                        logger.info(f"Cache covers requested redshift range ({z_min}-{z_max}).")
                        return cached_df
                    else:
                        logger.info(f"Cached catalog z-range ({cache_z_min:.3f}-{cache_z_max:.3f}) does not cover requested range ({z_min}-{z_max}). Redownloading.")
                else:
                    logger.info("Cached catalog is empty or invalid. Redownloading.")
            except Exception as e:
                logger.warning(f"Error checking cache: {e}. Redownloading.")
        
        logger.info("Downloading SDSS DR7 full spectroscopic galaxy catalog...")
        logger.info(f"  Redshift range: z = {z_min:.3f} - {z_max:.3f}")
        logger.info(f"  Magnitude limit: r < {mag_limit:.1f}")
        
        # Check for existing checkpoint
        resume_from_batch = 0
        last_objid = 0  # Track last objid to ensure no duplicates
        existing_batches = []
        total_batches_estimate = 15  # Initial estimate for DR7 (~600k galaxies vs ~3M for DR16)
        
        if checkpoint_manager:
            checkpoint = checkpoint_manager.load_checkpoint('download')
            if checkpoint and checkpoint.get('catalog_name') == self.catalog_name:
                # Check if checkpoint parameters match current query
                checkpoint_z_min = checkpoint.get('z_min')
                checkpoint_z_max = checkpoint.get('z_max')
                checkpoint_mag_limit = checkpoint.get('mag_limit')
                
                # If redshift range or magnitude limit changed, don't resume from checkpoint
                if (checkpoint_z_min is not None and abs(checkpoint_z_min - z_min) > 0.001) or \
                   (checkpoint_z_max is not None and abs(checkpoint_z_max - z_max) > 0.001) or \
                   (checkpoint_mag_limit is not None and abs(checkpoint_mag_limit - mag_limit) > 0.1):
                    logger.info(f"Checkpoint parameters don't match current query (z={z_min}-{z_max}, mag={mag_limit}), starting fresh")
                    resume_from_batch = 0
                    last_objid = 0
                    total_batches_estimate = 15
                else:
                    resume_from_batch = checkpoint.get('batch_id', 0)
                    last_objid = checkpoint.get('last_objid', 0)
                    total_batches_estimate = checkpoint.get('total_batches', 15)
                    
                    # If we've already completed, don't resume
                    if checkpoint.get('completed', False):
                        logger.info("Download already completed according to checkpoint")
                        if self.cache_file.exists():
                            logger.info(f"Loading completed catalog from cache: {self.cache_file}")
                            return pd.read_pickle(self.cache_file)
                    
                    logger.info(f"Resuming download from batch {resume_from_batch}, last objid: {last_objid}")
                
                # Load existing batches
                batch_dir = self.cache_file.parent / "batches"
                if batch_dir.exists():
                    for batch_file in sorted(batch_dir.glob("batch_*.pkl")):
                        batch_num = int(batch_file.stem.split('_')[1])
                        if batch_num < resume_from_batch:
                            batch_df = pd.read_pickle(batch_file)
                            existing_batches.append(batch_df)
                            # Update last_objid from loaded batches
                            if len(batch_df) > 0 and 'objid' in batch_df.columns:
                                last_objid = max(last_objid, batch_df['objid'].max())
        
        # Create batch directory
        batch_dir = self.cache_file.parent / "batches"
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Query SDSS in batches
        all_batches = existing_batches.copy()
        batch_id = resume_from_batch
        
        # Update total_batches_estimate if we've already exceeded it
        if batch_id >= total_batches_estimate:
            total_batches_estimate = batch_id + 10  # Add buffer for remaining batches
        
        # Create progress bar for batch downloading
        # Configure tqdm to write to logger instead of stdout
        # We'll use a custom file-like object that writes to logger
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
        
        pbar = None
        try:
            # Initialize progress bar (will update as we discover total)
            # Use logger_writer so progress goes to logs
            pbar = tqdm(
                total=total_batches_estimate,
                initial=resume_from_batch,
                desc="Downloading galaxies",
                unit="batch",
                file=logger_writer,
                ncols=100,
                mininterval=1.0  # Update at least once per second
            )
            
            while True:
                # Update progress bar description with current batch info
                total_rows = sum(len(b) for b in all_batches)
                pbar.set_description(f"Downloading batch {batch_id + 1} (total: {total_rows:,} galaxies)")
                
                # SQL query for SDSS DR7 spectroscopic galaxies
                # Use OFFSET/FETCH for proper pagination, starting from last_objid
                # This ensures we don't download duplicates when resuming
                query = f"""
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
                
                try:
                    # Key difference: add data_release=7 parameter for DR7
                    # DR7 may be slower or have different limits, so use retry logic
                    max_retries = 3
                    retry_delay = 5  # seconds
                    result = None
                    
                    for attempt in range(max_retries):
                        try:
                            result = SDSS.query_sql(query, data_release=7, timeout=600)
                            # If we got a result, break out of retry loop
                            if result is not None:
                                break
                        except Exception as e:
                            # Check if it's a parsing error (likely server error page or "No objects found")
                            error_str = str(e).lower()
                            if 'inconsistent' in error_str or 'parse' in error_str or 'column' in error_str or \
                               'no such file' in error_str or 'no objects' in error_str:
                                # Check if the error message contains "No objects have been found"
                                if 'no objects' in error_str or ('no such file' in error_str and 'no objects' in str(e)):
                                    # This means the query returned no results - we've reached the end
                                    logger.info(f"No more objects found at batch {batch_id + 1} (objid > {last_objid}) - download complete")
                                    result = None  # Signal completion
                                    break
                                
                                if attempt < max_retries - 1:
                                    logger.warning(f"DR7 query returned error (attempt {attempt + 1}/{max_retries}): {e}")
                                    logger.info(f"Retrying in {retry_delay} seconds...")
                                    time.sleep(retry_delay)
                                    retry_delay *= 2  # Exponential backoff
                                    continue
                                else:
                                    # Last attempt failed, raise the error
                                    raise
                            else:
                                # Different error, don't retry
                                raise
                    
                    # Check for empty result - this means we've reached the end
                    if result is None or len(result) == 0:
                        logger.info(f"No more data at batch {batch_id + 1} - download complete")
                        # Mark as completed since we got empty result
                        total_rows = sum(len(b) for b in all_batches)
                        total_batches_estimate = batch_id + 1  # Final batch count
                        if checkpoint_manager:
                            checkpoint_data = {
                                'stage': 'download',
                                'catalog_name': self.catalog_name,
                                'batch_id': batch_id + 1,
                                'total_batches': total_batches_estimate,
                                'rows_downloaded': total_rows,
                                'last_objid': last_objid,
                                'progress': 1.0,
                                'completed': True,
                                'z_min': z_min,
                                'z_max': z_max,
                                'mag_limit': mag_limit
                            }
                            checkpoint_manager.save_checkpoint('download', checkpoint_data)
                        break
                    
                    batch_df = result.to_pandas()
                    
                    # Check for empty DataFrame - also means we're done
                    if len(batch_df) == 0:
                        logger.info(f"Empty batch {batch_id + 1} - download complete")
                        # Mark as completed
                        total_rows = sum(len(b) for b in all_batches)
                        total_batches_estimate = batch_id + 1
                        if checkpoint_manager:
                            checkpoint_data = {
                                'stage': 'download',
                                'catalog_name': self.catalog_name,
                                'batch_id': batch_id + 1,
                                'total_batches': total_batches_estimate,
                                'rows_downloaded': total_rows,
                                'last_objid': last_objid,
                                'progress': 1.0,
                                'completed': True,
                                'z_min': z_min,
                                'z_max': z_max,
                                'mag_limit': mag_limit
                            }
                            checkpoint_manager.save_checkpoint('download', checkpoint_data)
                        break
                    
                    # Get the maximum objid from this batch to track progress
                    if 'objid' in batch_df.columns:
                        batch_max_objid = batch_df['objid'].max()
                    else:
                        # Fallback: estimate based on batch_id if objid missing
                        batch_max_objid = last_objid + len(batch_df)
                    
                    # Save batch checkpoint
                    batch_file = batch_dir / f"batch_{batch_id:04d}.pkl"
                    batch_df.to_pickle(batch_file)
                    
                    all_batches.append(batch_df)
                    
                    # Update last_objid for next query
                    last_objid = batch_max_objid
                    
                    # Update checkpoint with current progress
                    total_rows = sum(len(b) for b in all_batches)
                    
                    # Dynamically update total_batches_estimate if we're approaching it
                    if batch_id + 1 >= total_batches_estimate:
                        # We've exceeded our estimate, add more buffer
                        total_batches_estimate = batch_id + 10
                        logger.info(f"Updating total batches estimate to {total_batches_estimate}")
                    
                    if checkpoint_manager:
                        checkpoint_manager.update_download_progress(
                            batch_id + 1, total_batches_estimate, total_rows, self.catalog_name, last_objid
                        )
                        # Save query parameters to checkpoint for validation on resume
                        checkpoint_data = checkpoint_manager.load_checkpoint('download')
                        if checkpoint_data:
                            checkpoint_data['z_min'] = z_min
                            checkpoint_data['z_max'] = z_max
                            checkpoint_data['mag_limit'] = mag_limit
                            checkpoint_manager.save_checkpoint('download', checkpoint_data)
                    
                    # Update progress bar total if needed
                    if batch_id + 1 >= pbar.total:
                        pbar.total = total_batches_estimate
                    
                    # Update progress bar
                    pbar.update(1)
                    pbar.set_postfix({
                        'galaxies': f'{total_rows:,}',
                        'batch_size': len(batch_df)
                    })
                    
                    logger.info(f"✓ Batch {batch_id + 1}: {len(batch_df)} galaxies (total: {total_rows:,})")
                    
                    # If we got fewer rows than batch_size (50k), we're done
                    # This means we've reached the end of the catalog
                    if len(batch_df) < self.batch_size:
                        logger.info(f"Reached end of catalog (got {len(batch_df):,} rows, less than batch size {self.batch_size:,})")
                        total_batches_estimate = batch_id + 1  # Update total to actual
                        pbar.total = total_batches_estimate
                        
                        # Mark as completed in checkpoint
                        if checkpoint_manager:
                            checkpoint_data = {
                                'stage': 'download',
                                'catalog_name': self.catalog_name,
                                'batch_id': batch_id + 1,
                                'total_batches': total_batches_estimate,
                                'rows_downloaded': total_rows,
                                'last_objid': last_objid,
                                'progress': 1.0,
                                'completed': True,
                                'z_min': z_min,
                                'z_max': z_max,
                                'mag_limit': mag_limit
                            }
                            checkpoint_manager.save_checkpoint('download', checkpoint_data)
                        break
                    
                    batch_id += 1
                    
                    # Rate limiting: small delay between queries
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error downloading batch {batch_id + 1}: {e}")
                    # If we have some data, continue with what we have
                    if len(all_batches) > 0:
                        logger.warning("Continuing with partial download")
                        break
                    else:
                        raise
        
        except Exception as e:
            logger.error(f"Download failed: {e}")
            # Save partial results if we have any
            if len(all_batches) > 0:
                logger.warning("Saving partial download for resume")
                combined = pd.concat(all_batches, ignore_index=True)
                combined.to_pickle(self.cache_file)
            raise
        finally:
            # Close progress bar
            if pbar is not None:
                pbar.close()
        
        # Combine all batches
        if len(all_batches) == 0:
            raise RuntimeError("No data downloaded")
        
        logger.info(f"Combining {len(all_batches)} batches...")
        # Use logger writer for combine progress bar (LoggerWriter already defined above)
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
        
        # Save final catalog
        logger.info(f"Saving catalog: {len(combined):,} galaxies")
        combined.to_pickle(self.cache_file)
        
        # Mark download as completed in checkpoint
        if checkpoint_manager:
            final_total_rows = len(combined)
            checkpoint_data = {
                'stage': 'download',
                'catalog_name': self.catalog_name,
                'batch_id': len(all_batches),
                'total_batches': len(all_batches),
                'rows_downloaded': final_total_rows,
                'last_objid': last_objid if 'objid' in combined.columns else 0,
                'progress': 1.0,
                'completed': True
            }
            checkpoint_manager.save_checkpoint('download', checkpoint_data)
        
        # Clean up batch files
        logger.info("Cleaning up batch files...")
        for batch_file in batch_dir.glob("batch_*.pkl"):
            batch_file.unlink()
        
        logger.info(f"✓ SDSS DR7 catalog downloaded: {len(combined):,} galaxies")
        
        return combined
    
    def load(self, use_cache: bool = True) -> Optional[pd.DataFrame]:
        """
        Load catalog from cache.
        
        Parameters:
            use_cache: Whether to use cached data
            
        Returns:
            DataFrame or None if not cached
        """
        if not use_cache or not self.cache_file.exists():
            return None
        
        try:
            logger.info(f"Loading SDSS DR7 catalog from cache: {self.cache_file}")
            df = pd.read_pickle(self.cache_file)
            logger.info(f"Loaded {len(df):,} galaxies")
            return df
        except Exception as e:
            logger.warning(f"Failed to load cache: {e}")
            return None

