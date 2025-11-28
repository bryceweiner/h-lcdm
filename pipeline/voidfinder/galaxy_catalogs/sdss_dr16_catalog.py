"""
SDSS DR16 Galaxy Catalog Provider
=================================

Downloads and manages SDSS DR16 full spectroscopic galaxy catalog.
"""

import pandas as pd
from typing import Optional
from pathlib import Path
import logging

from .sdss_base_catalog import SDSSBaseCatalog

logger = logging.getLogger(__name__)


class SDSSDR16Catalog(SDSSBaseCatalog):
    """
    SDSS DR16 full spectroscopic galaxy catalog provider.
    
    Downloads ~3M galaxies in batches with checkpointing support.
    Inherits common functionality from SDSSBaseCatalog.
    """
    
    def __init__(self, downloaded_data_dir: Path, processed_data_dir: Path):
        """Initialize SDSS DR16 catalog provider."""
        super().__init__(downloaded_data_dir, processed_data_dir)
        self.cache_base_dir = self.downloaded_data_dir / "sdss_dr16"
        self.cache_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Batch size for queries (SDSS CasJobs limit is typically 500k rows)
        self.batch_size = 50000  # Conservative batch size
    
    @property
    def catalog_name(self) -> str:
        """Return catalog identifier."""
        return "sdss_dr16"
    
    def _get_initial_batch_estimate(self) -> int:
        """Get initial estimate for number of batches (DR16 has ~3M galaxies)."""
        return 60
    
    def _build_sql_query(self, z_min: float, z_max: float, mag_limit: float, 
                        last_objid: int = 0) -> str:
        """
        Build SQL query for SDSS DR16 catalog download.
        
        DR16-specific: Uses clean=1 filter (standard for DR16).
        
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
                    AND p.clean = 1
                    AND (p.type = 3 OR p.type = 6)
                    AND p.objid > {last_objid}
                ORDER BY p.objid
                """
                
    def _execute_query(self, query: str, max_retries: int = 1, 
                      retry_delay: int = 5) -> Optional[pd.DataFrame]:
        """
        Execute SDSS DR16 SQL query.
        
        DR16 uses default data_release (16) and doesn't need special retry logic.
        
        Parameters:
            query: SQL query string
            max_retries: Maximum number of retry attempts (unused for DR16)
            retry_delay: Initial delay between retries (unused for DR16)
            
        Returns:
            DataFrame with results, or None if no results
        """
        from astroquery.sdss import SDSS
        result = SDSS.query_sql(query, timeout=600)
        
        if result is not None:
            return result.to_pandas()
            return None
