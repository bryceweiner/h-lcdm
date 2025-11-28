"""
SDSS DR7 Galaxy Catalog Provider
=================================

Downloads and manages SDSS DR7 full spectroscopic galaxy catalog.
"""

import pandas as pd
from typing import Optional
from pathlib import Path
import logging

from .sdss_base_catalog import SDSSBaseCatalog

logger = logging.getLogger(__name__)


class SDSSDR7Catalog(SDSSBaseCatalog):
    """
    SDSS DR7 full spectroscopic galaxy catalog provider.
    
    Downloads ~600k galaxies in batches with checkpointing support.
    Inherits common functionality from SDSSBaseCatalog.
    """
    
    def __init__(self, downloaded_data_dir: Path, processed_data_dir: Path):
        """Initialize SDSS DR7 catalog provider."""
        super().__init__(downloaded_data_dir, processed_data_dir)
        self.cache_base_dir = self.downloaded_data_dir / "sdss_dr7"
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Batch size for queries (SDSS DR7 may have stricter limits than DR16)
        self.batch_size = 25000  # Smaller batch size for DR7 stability
    
    @property
    def catalog_name(self) -> str:
        """Return catalog identifier."""
        return "sdss_dr7"
    
    def _get_initial_batch_estimate(self) -> int:
        """Get initial estimate for number of batches (DR7 has ~600k galaxies)."""
        return 15
    
    def _get_max_retries(self) -> int:
        """Get maximum number of retries for DR7 queries (may have parsing errors)."""
        return 3
    
    def _build_sql_query(self, z_min: float, z_max: float, mag_limit: float, 
                        last_objid: int = 0) -> str:
        """
        Build SQL query for SDSS DR7 catalog download.
        
        DR7-specific: No clean=1 filter (clean=0 is valid in DR7).
        
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
    
    def _execute_query(self, query: str, max_retries: int = 3, 
                      retry_delay: int = 5) -> Optional[pd.DataFrame]:
        """
        Execute SDSS DR7 SQL query with retry logic for parsing errors.
        
        DR7-specific: Handles "No objects have been found" and parsing errors.
        
        Parameters:
            query: SQL query string
            max_retries: Maximum number of retry attempts
            retry_delay: Initial delay between retries (seconds)
            
        Returns:
            DataFrame with results, or None if no results
        """
        import time
        
        for attempt in range(max_retries):
            try:
                from astroquery.sdss import SDSS
                result = SDSS.query_sql(query, data_release=7, timeout=600)
                
                # If we got a result, break out of retry loop
                if result is not None:
                    return result.to_pandas()
            except Exception as e:
                # Check if it's a parsing error or "No objects found"
                error_str = str(e).lower()
                if 'inconsistent' in error_str or 'parse' in error_str or 'column' in error_str or \
                   'no such file' in error_str or 'no objects' in error_str:
                    # Check if the error message contains "No objects have been found"
                    if 'no objects' in error_str or ('no such file' in error_str and 'no objects' in str(e)):
                        # This means the query returned no results - we've reached the end
                        logger.info(f"No more objects found (objid > last_objid) - download complete")
                        return None  # Signal completion
                    
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
        
        return None
