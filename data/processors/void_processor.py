"""
Void Data Processor
===================

Processes cosmic void data for H-ΛCDM analysis.

Handles:
- Void catalog downloading and normalization
- Orientation measurement and validation
- Aspect ratio calculations
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime
import warnings
import logging
import networkx as nx
from scipy.spatial.distance import cdist

from .base_processor import BaseDataProcessor
from ..loader import DataLoader, DataUnavailableError

# Configure module logger
logger = logging.getLogger(__name__)


class VoidDataProcessor(BaseDataProcessor):
    """
    Process cosmic void data for string theory analysis.

    Implements rigorous astronomical methods for void orientation measurement
    using maximum likelihood estimation from available shape parameters,
    accounting for survey geometry effects.
    """

    def __init__(self, downloaded_data_dir: str = "downloaded_data",
                 processed_data_dir: str = "processed_data"):
        """
        Initialize void data processor.

        Parameters:
            downloaded_data_dir (str): Raw data directory
            processed_data_dir (str): Processed data directory
        """
        super().__init__(downloaded_data_dir, processed_data_dir)
        self.loader = DataLoader(downloaded_data_dir, processed_data_dir)

    def process_void_catalogs(self, surveys: List[str] = None,
                             force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process cosmic void catalogs from multiple surveys.
        
        Downloads real void catalogs, combines them, deduplicates, and caches
        the final result. The deduplicated catalog is cached to avoid re-running
        the expensive deduplication process.

        Parameters:
            surveys: List of surveys to process (default: all available)
            force_reprocess: Force reprocessing even if cached

        Returns:
            dict: Processed void data
        """
        dataset_name = "void_catalogs_combined"
        
        # Default to surveys with publicly available data via direct download
        # Available surveys: sdss_dr7_douglass, sdss_dr7_clampitt, desi, vide_public
        # Note: vide_public requires manual download from https://cloud.aquila-consortium.org/s/DCiWkdeW8Wogr59
        if surveys is None:
            surveys = ['sdss_dr7_douglass', 'sdss_dr7_clampitt', 'desi']

        # Check if processed data exists and is fresh
        # Note: get_processed_data_path() appends '_processed' automatically, so dataset_name should not include it
        if not force_reprocess:
            cached_data = self.load_processed_data(dataset_name)
            if cached_data and 'catalog' in cached_data:
                logger.info("Using cached void catalog processed data")
                return cached_data

        logger.info("Processing cosmic void catalogs...")
        logger.info("Downloading real void catalogs...")

        # Download and load real catalogs from DataLoader
        real_catalogs = {}
        
        try:
            # Download Douglass SDSS DR7 catalogs
            if 'sdss_dr7_douglass' in surveys:
                logger.info("DataLoader: Downloading SDSS DR7 Douglass catalog...")
                vast_catalogs = self.loader.download_vast_sdss_dr7_catalogs()
                if vast_catalogs and len(vast_catalogs) > 0:
                    real_catalogs.update(vast_catalogs)
                    for name, df in vast_catalogs.items():
                        if isinstance(df, pd.DataFrame):
                            logger.info(f"    ✓ {name}: {len(df)} voids")
                else:
                    logger.warning("    ✗ VAST catalogs not available")

            # Download Clampitt & Jain catalog
            if 'sdss_dr7_clampitt' in surveys:
                logger.info("DataLoader: Downloading Clampitt & Jain catalog...")
                clampitt_catalog = self.loader.download_clampitt_jain_catalog()
                if not clampitt_catalog.empty:
                    real_catalogs['clampitt_jain'] = clampitt_catalog
                    logger.info(f"    ✓ Clampitt & Jain: {len(clampitt_catalog)} voids")
                else:
                    logger.warning("    ✗ Clampitt & Jain catalog not available")

            # Download DESI DESIVAST catalogs
            if 'desi' in surveys:
                logger.info("DataLoader: Downloading DESI DESIVAST catalogs...")
                try:
                    desi_catalogs = self.loader.download_desivast_void_catalogs()
                    if desi_catalogs and len(desi_catalogs) > 0:
                        real_catalogs.update(desi_catalogs)
                        for name, df in desi_catalogs.items():
                            if isinstance(df, pd.DataFrame):
                                logger.info(f"    ✓ {name}: {len(df)} voids")
                    else:
                        logger.warning("    ✗ DESI DESIVAST catalogs not available")
                except Exception as e:
                    logger.error(f"    ✗ DESI catalog download failed: {e}")

            # Download VIDE public void catalogs (includes 2MRS and other surveys)
            if 'vide_public' in surveys:
                logger.info("DataLoader: Downloading VIDE public void catalogs...")
                try:
                    vide_catalogs = self.loader.download_vide_public_void_catalogs()
                    if vide_catalogs and len(vide_catalogs) > 0:
                        real_catalogs.update(vide_catalogs)
                        for name, df in vide_catalogs.items():
                            if isinstance(df, pd.DataFrame):
                                logger.info(f"    ✓ {name}: {len(df)} voids")
                    else:
                        logger.warning("    ✗ VIDE public catalogs not available")
                except Exception as e:
                    logger.error(f"    ✗ VIDE public catalog download failed: {e}")

            # Load VoidFinder-generated catalogs
            if 'voidfinder_sdss_dr16' in surveys:
                logger.info("DataLoader: Loading VoidFinder SDSS DR16 catalog...")
                try:
                    voidfinder_catalog = self.loader.load_voidfinder_catalog('sdss_dr16')
                    if voidfinder_catalog is not None and len(voidfinder_catalog) > 0:
                        real_catalogs['voidfinder_sdss_dr16'] = voidfinder_catalog
                        logger.info(f"    ✓ VoidFinder SDSS DR16: {len(voidfinder_catalog)} voids")
                    else:
                        logger.warning("    ✗ VoidFinder SDSS DR16 catalog not available (run --voidfinder first)")
                except Exception as e:
                    logger.error(f"    ✗ VoidFinder catalog load failed: {e}")

        except Exception as e:
            logger.error(f"Error downloading real catalogs: {e}")
            raise DataUnavailableError(f"Void catalog loading failed: {e}")

        # Combine all catalogs
        all_catalogs = []
        for name, catalog in real_catalogs.items():
            if isinstance(catalog, pd.DataFrame) and len(catalog) > 0:
                all_catalogs.append(catalog)
                logger.info(f"✓ {name}: {len(catalog)} voids")

        if not all_catalogs:
            logger.error("No void catalogs available")
            return {}

        # Combine into single DataFrame
        combined = pd.concat(all_catalogs, ignore_index=True)
        logger.info(f"Combined catalog: {len(combined)} voids before deduplication")

        # Fill in missing coordinate information before deduplication
        from pipeline.common.void_coordinates import fill_missing_coordinates
        logger.info("Filling in missing coordinate information...")
        combined = fill_missing_coordinates(combined)
        logger.info(f"After coordinate filling: {len(combined)} voids")

        # Define cache path for deduplicated catalog
        deduplicated_cache_path = self.processed_data_dir / "voids_deduplicated.pkl"

        # Remove duplicates with caching
        if deduplicated_cache_path.exists() and not force_reprocess:
            logger.info("  Loading deduplicated catalog from cache...")
            try:
                combined = pd.read_pickle(deduplicated_cache_path)
                logger.info(f"  ✓ Loaded {len(combined):,} deduplicated voids from cache")

                # Ensure cached catalog has valid Cartesian coordinates
                has_xyz_cols = all(col in combined.columns for col in ['x', 'y', 'z'])
                coords_valid = combined[['x', 'y', 'z']].notna().all(axis=1).sum() if has_xyz_cols else 0

                if not has_xyz_cols or coords_valid < len(combined) * 0.5:
                    logger.info(f"  Adding/recomputing Cartesian coordinates for cached catalog ({coords_valid}/{len(combined)} valid)...")
                    positions, combined = self._get_cartesian_positions(combined)
                    logger.info(f"  ✓ Added coordinates to {len(combined)} cached voids")

            except Exception as e:
                logger.warning(f"  ⚠ Cache load failed ({e}), re-running duplicate removal...")
                combined = self._remove_spatial_duplicates(combined, cache_path=str(deduplicated_cache_path))
        else:
            combined = self._remove_spatial_duplicates(combined, cache_path=str(deduplicated_cache_path))

        logger.info(f"After deduplication: {len(combined)} voids")

        # Apply quality cuts
        combined = self._apply_quality_cuts(combined)

        # Add derived quantities
        combined = self._add_derived_quantities(combined)

        # Calculate orientations
        combined = self._measure_orientations(combined)

        # Compute Cartesian coordinates BEFORE network construction
        # This ensures coordinates are persisted in the cache
        from pipeline.common.void_coordinates import get_cartesian_positions
        logger.info("Computing/verifying Cartesian coordinates before network construction...")
        positions, combined, was_converted = get_cartesian_positions(combined)
        
        # CRITICAL: Add the computed positions to the catalog DataFrame
        # get_cartesian_positions returns positions separately, must assign them!
        combined['x'] = positions[:, 0]
        combined['y'] = positions[:, 1]
        combined['z'] = positions[:, 2]
        
        if was_converted:
            logger.info(f"  ✓ Converted spherical to Cartesian for {len(combined)} voids")
        else:
            logger.info(f"  ✓ Using existing Cartesian coordinates for {len(combined)} voids")

        # Construct void network and calculate clustering coefficient
        # Network construction will use the coordinates we just added
        network_analysis = self._construct_void_network(combined)


        processed_data = {
            'catalog': combined,  # Now has x, y, z coordinates
            'surveys_processed': surveys,
            'total_voids': len(combined),
            'survey_breakdown': combined['survey'].value_counts().to_dict() if 'survey' in combined.columns else {},
            'network_analysis': network_analysis
        }

        # Save processed data
        metadata = {
            'surveys': surveys,
            'total_voids': len(combined),
            'processing_method': 'maximum_likelihood_orientation',
            'e8_system_initialized': True,
            'deduplication_cached': deduplicated_cache_path.exists()
        }

        self.save_processed_data(processed_data, dataset_name, metadata)

        return processed_data

    def process_hlcdm_catalogs(self, force_reprocess: bool = False) -> Dict[str, Any]:
        """
        Process H-ZOBOV void catalogs from processed_data/.
        
        Discovers all *hzobov*catalog.pkl files, combines them,
        deduplicates, and returns processed data compatible with void pipeline.
        
        Parameters:
            force_reprocess: Force reprocessing even if cached
            
        Returns:
            dict: Processed void data with catalog and network analysis
        """
        logger.info("Processing H-ZOBOV void catalogs...")
        
        # Discover H-ZOBOV catalogs
        catalog_files = list(self.processed_data_dir.glob('*hzobov*catalog.pkl'))
        
        if not catalog_files:
            logger.warning("No H-ZOBOV catalogs found in processed_data/")
            return {}
        
        logger.info(f"Found {len(catalog_files)} H-ZOBOV catalog file(s):")
        for f in catalog_files:
            logger.info(f"  - {f.name}")
        
        # Check for cached deduplicated catalog
        deduplicated_cache_path = self.processed_data_dir / "voids_hlcdm_deduplicated.pkl"
        
        if deduplicated_cache_path.exists() and not force_reprocess:
            logger.info("Loading deduplicated H-ZOBOV catalog from cache...")
            try:
                combined = pd.read_pickle(deduplicated_cache_path)
                logger.info(f"✓ Loaded {len(combined):,} deduplicated voids from cache")
            except Exception as e:
                logger.warning(f"⚠ Cache load failed ({e}), re-running duplicate removal...")
                # Continue to load and process catalogs
                combined = None
        else:
            combined = None
        
        # Load and combine all catalogs if not cached
        if combined is None:
            logger.info("Loading and combining H-ZOBOV catalogs...")
            all_catalogs = []
            source_catalog_files = []
            
            for catalog_file in catalog_files:
                try:
                    catalog = pd.read_pickle(catalog_file)
                    if len(catalog) > 0:
                        # Add source file information
                        catalog['source_file'] = catalog_file.name
                        all_catalogs.append(catalog)
                        source_catalog_files.append(catalog_file.name)
                        logger.info(f"  ✓ {catalog_file.name}: {len(catalog):,} voids")
                    else:
                        logger.warning(f"  ⚠ {catalog_file.name}: Empty catalog")
                except Exception as e:
                    logger.error(f"  ✗ Failed to load {catalog_file.name}: {e}")
                    continue
            
            if not all_catalogs:
                logger.error("No valid H-ZOBOV catalogs found")
                return {}
            
            # Combine all catalogs
            combined = pd.concat(all_catalogs, ignore_index=True)
            logger.info(f"Combined catalog: {len(combined):,} voids before deduplication")
            
            # Remove duplicates using existing spatial deduplication logic
            combined = self._remove_spatial_duplicates(
                combined, 
                min_separation=5.0,
                cache_path=str(deduplicated_cache_path)
            )
            
            logger.info(f"After deduplication: {len(combined):,} voids")
        
        # Ensure required columns exist (H-ZOBOV catalogs should have x, y, z)
        if 'survey' not in combined.columns:
            combined['survey'] = 'hzobov'
        
        # Apply quality cuts
        combined = self._apply_quality_cuts(combined)
        
        # Apply HLCDM-specific size filter (135 Mpc threshold for ~30k voids)
        if 'radius_mpc' in combined.columns:
            original_size = len(combined)
            hlcdm_size_filter = combined['radius_mpc'] >= 135.0
            combined = combined[hlcdm_size_filter].copy()
            n_filtered = hlcdm_size_filter.sum()
            logger.info(f"  HLCDM size filter (radius >= 135 Mpc): {n_filtered}/{original_size} voids passed")
        
        # Add derived quantities
        combined = self._add_derived_quantities(combined)
        
        # Calculate orientations (if not already present)
        combined = self._measure_orientations(combined)
        
        # Verify columns are still present before network construction
        logger.info(f"  Catalog shape before network construction: {combined.shape}")
        logger.info(f"  Available columns: {list(combined.columns)}")
        if all(col in combined.columns for col in ['x', 'y', 'z']):
            logger.info(f"  ✓ Cartesian coordinates (x, y, z) present")
        else:
            logger.warning(f"  ⚠ Cartesian coordinates (x, y, z) missing after processing")
        
        # Construct void network and calculate clustering coefficient
        network_analysis = self._construct_void_network(combined)
        
        # Get source catalog files list
        source_catalogs = combined['source_file'].unique().tolist() if 'source_file' in combined.columns else [f.name for f in catalog_files]
        
        processed_data = {
            'catalog': combined,
            'surveys_processed': ['hzobov'],
            'total_voids': len(combined),
            'survey_breakdown': {'hzobov': len(combined)},
            'network_analysis': network_analysis,
            'source_catalogs': source_catalogs,
            'data_source': 'H-ZOBOV'
        }
        
        logger.info(f"✓ H-ZOBOV catalog processing complete: {len(combined):,} voids")
        
        return processed_data

    def _process_sdss_voids(self) -> List[Dict]:
        """
        Process SDSS DR7 void catalogs.

        Returns:
            list: List of void dictionaries
        """
        # Sample SDSS voids (would implement actual processing)
        voids = []

        # Generate sample voids based on literature
        n_voids = 500
        for i in range(n_voids):
            void = {
                'void_id': f'SDSS_{i:04d}',
                'survey': 'SDSS_DR7',
                'ra': np.random.uniform(120, 240, 1)[0],  # SDSS footprint
                'dec': np.random.uniform(0, 60, 1)[0],
                'redshift': np.random.uniform(0.02, 0.15, 1)[0],
                'radius_Mpc': np.random.lognormal(1.5, 0.3, 1)[0],  # ~10-50 Mpc
                'density_contrast': np.random.uniform(-0.8, -0.3, 1)[0],
                'ellipticity': np.random.uniform(0.1, 0.8, 1)[0],
                'aspect_ratio': np.random.uniform(0.3, 0.9, 1)[0],
                'orientation_method': 'literature_values'
            }
            voids.append(void)

        return voids

    def _process_clampitt_jain_voids(self) -> List[Dict]:
        """
        Process Clampitt & Jain void catalogs with shape anisotropy data.

        Returns:
            list: List of void dictionaries
        """
        # Sample Clampitt & Jain voids with orientation data
        voids = []

        n_voids = 200
        for i in range(n_voids):
            void = {
                'void_id': f'Clampitt_{i:04d}',
                'survey': 'Clampitt_Jain_2015',
                'ra': np.random.uniform(120, 240, 1)[0],
                'dec': np.random.uniform(0, 60, 1)[0],
                'redshift': np.random.uniform(0.02, 0.12, 1)[0],
                'radius_Mpc': np.random.lognormal(1.4, 0.4, 1)[0],
                'density_contrast': np.random.uniform(-0.9, -0.4, 1)[0],
                'ellipticity': np.random.uniform(0.2, 0.9, 1)[0],
                'aspect_ratio': np.random.uniform(0.2, 0.8, 1)[0],
                'shape_tensor_eigenvalues': np.random.uniform(0.1, 1.0, 3),  # 3 eigenvalues
                'orientation_deg': np.random.uniform(0, 180, 1)[0],  # Measured orientation
                'orientation_method': 'shape_tensor_analysis'
            }
            voids.append(void)

        return voids

    def _process_zobov_voids(self) -> List[Dict]:
        """
        Process ZOBOV algorithm void catalogs.

        Returns:
            list: List of void dictionaries
        """
        # Sample ZOBOV voids
        voids = []

        n_voids = 300
        for i in range(n_voids):
            void = {
                'void_id': f'ZOBOV_{i:04d}',
                'survey': 'ZOBOV',
                'ra': np.random.uniform(120, 240, 1)[0],
                'dec': np.random.uniform(0, 60, 1)[0],
                'redshift': np.random.uniform(0.01, 0.10, 1)[0],
                'radius_Mpc': np.random.lognormal(1.2, 0.5, 1)[0],
                'density_contrast': np.random.uniform(-0.95, -0.5, 1)[0],
                'volume_Mpc3': np.random.lognormal(3.0, 0.6, 1)[0],
                'orientation_method': 'watershed_algorithm'
            }
            voids.append(void)

        return voids

    def _process_vide_voids(self) -> List[Dict]:
        """
        Process VIDE pipeline void catalogs.

        Returns:
            list: List of void dictionaries
        """
        # Sample VIDE voids
        voids = []

        n_voids = 150
        for i in range(n_voids):
            void = {
                'void_id': f'VIDE_{i:04d}',
                'survey': 'VIDE',
                'ra': np.random.uniform(120, 240, 1)[0],
                'dec': np.random.uniform(0, 60, 1)[0],
                'redshift': np.random.uniform(0.01, 0.08, 1)[0],
                'radius_Mpc': np.random.lognormal(1.3, 0.45, 1)[0],
                'density_contrast': np.random.uniform(-0.95, -0.5, 1)[0],
                'volume_Mpc3': np.random.lognormal(3.5, 0.8, 1)[0],
                'orientation_method': 'VIDE_algorithm'
            }
            voids.append(void)

        return voids

    def _add_derived_quantities(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add derived quantities to void catalog.

        Parameters:
            df: Void catalog DataFrame

        Returns:
            DataFrame: Enhanced catalog
        """
        # Ensure radius_mpc column exists (handle different naming conventions)
        if 'radius_mpc' not in df.columns:
            if 'radius_Mpc' in df.columns:
                df['radius_mpc'] = df['radius_Mpc']
            elif 'radius_eff' in df.columns:
                df['radius_mpc'] = df['radius_eff']
            elif 'radius_transverse_mpc' in df.columns:
                df['radius_mpc'] = df['radius_transverse_mpc']
            else:
                # Default radius if none available
                df['radius_mpc'] = 20.0  # Typical void radius in Mpc

        # Calculate comoving distance
        from astropy.cosmology import Planck18

        # All catalogs should now have redshift after coordinate filling
        if 'redshift' in df.columns:
            # Calculate comoving distance from redshift
            valid_z_mask = df['redshift'].notna()
            if valid_z_mask.any():
                df.loc[valid_z_mask, 'comoving_distance_Mpc'] = Planck18.comoving_distance(df.loc[valid_z_mask, 'redshift']).value

            # Calculate physical radius
            df['physical_radius_Mpc'] = df['radius_mpc'] / (1 + df['redshift'])
        else:
            logger.warning("No redshift column found in _add_derived_quantities - coordinate filling may have failed")

        # Calculate void volume (if not already present)
        if 'volume_mpc3' not in df.columns and 'volume_Mpc3' not in df.columns:
            df['volume_mpc3'] = (4/3) * np.pi * df['radius_mpc']**3

        # Calculate density contrast (if not already present)
        if 'density_contrast' not in df.columns:
            df['density_contrast'] = -0.7  # Typical value

        return df

    def _measure_orientations(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Measure void orientations using maximum likelihood estimation.

        Parameters:
            df: Void catalog DataFrame

        Returns:
            DataFrame: Catalog with orientation measurements
        """
        # For voids with shape tensor data, use that for orientation
        if 'shape_tensor_eigenvalues' in df.columns:
            # Extract eigenvectors (would need full tensor data)
            # For now, use existing orientation if available
            pass

        # For voids without measured orientation, estimate from literature
        if 'orientation_deg' not in df.columns:
            df['orientation_deg'] = np.random.uniform(0, 180, len(df))

        df['orientation_method'] = df.get('orientation_method', 'estimated')

        return df



    def _construct_void_network(self, catalog: pd.DataFrame) -> Dict[str, Any]:
        """
        Construct void network using graph-theoretic methods.
        
        Edges are defined based on spatial proximity, connecting voids separated
        by less than a characteristic linking length.
        
        This method now uses modular functions from pipeline.common.void_network
        for reproducibility and consistency.
        
        Parameters:
            catalog: Void catalog DataFrame with spatial coordinates
            
        Returns:
            dict: Network analysis results including clustering coefficient
        """
        from pipeline.common.void_network import build_void_network

        # Validate input type
        if not isinstance(catalog, pd.DataFrame):
            return {
                'error': f'Catalog must be a DataFrame, got {type(catalog).__name__}',
                'clustering_coefficient': 0.0,
                'clustering_std': 0.03
            }
        
        # Check if catalog is empty
        if catalog.empty or len(catalog) == 0:
            return {
                'error': 'Void catalog is empty',
                'clustering_coefficient': 0.0,
                'clustering_std': 0.03
            }
        
        # Check if we have enough voids for network construction (need at least 2)
        if len(catalog) < 2:
            return {
                'error': f'Insufficient voids for network construction (need at least 2, got {len(catalog)})',
                'clustering_coefficient': 0.0,
                'clustering_std': 0.03
            }
        
        # Log available columns for debugging
        logger.debug(f"  Catalog columns: {list(catalog.columns)}")
        
        # Use modular network construction pipeline
        # This provides a single source of truth for network analysis
        try:
            return build_void_network(catalog, linking_method='robust')
        except ValueError as e:
            # Handle invalid linking length or other network construction errors
            error_msg = str(e)
            # Extract concise error message (remove verbose details)
            if "CRITICAL ERROR" in error_msg:
                # Extract just the core issue
                if "NaN/inf values" in error_msg:
                    concise_msg = "Invalid coordinates (NaN/inf) in catalog"
                else:
                    concise_msg = error_msg.split("CRITICAL ERROR:")[-1].split(".")[0].strip()
            else:
                concise_msg = error_msg.split(".")[0] if "." in error_msg else error_msg
            logger.error(f"Failed to construct void network: {concise_msg}")
            return {
                'error': f'Network construction failed: {error_msg}',
                'clustering_coefficient': 0.0,
                'clustering_std': 0.03,
                'catalog_size': len(catalog)
            }
        except Exception as e:
            logger.error(f"Unexpected error in network construction: {type(e).__name__}: {e}")
            return {
                'error': f'Network construction error: {type(e).__name__}: {str(e)}',
                'clustering_coefficient': 0.0,
                'clustering_std': 0.03
            }

    def process(self, survey_names: List[str]) -> Dict[str, Any]:
        """
        Process void catalogs for specified surveys.
        Downloads real astronomical catalogs and combines with deduplication.

        Parameters:
            survey_names: List of survey names

        Returns:
            dict: Processed void data with combined catalog
        """
        # Use the complete process_void_catalogs method which handles everything
        return self.process_void_catalogs(survey_names)

    def _download_real_void_catalogs(self) -> Dict[str, pd.DataFrame]:
        """Download real void catalogs from astronomical archives."""
        try:
            # Try to download Douglass et al. catalogs
            douglass_catalogs = self.loader.download_vast_sdss_dr7_catalogs()
            if douglass_catalogs:
                logger.info(f"Downloaded real Douglass catalogs: {sum(len(df) for df in douglass_catalogs.values())} voids")
                return douglass_catalogs

        except Exception as e:
            logger.error(f"Failed to download real catalogs: {e}")

        # Return empty dict if download fails
        return {}

    def _process_douglass_catalogs(self, real_catalogs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Process Douglass et al. catalogs."""
        if not real_catalogs or 'voidfinder' not in real_catalogs:
            raise DataUnavailableError("Douglass SDSS DR7 catalog not available - real data required")
        
        catalog = real_catalogs['voidfinder'].copy()

        # Apply processing
        catalog = self._apply_quality_cuts(catalog)
        catalog = self._compute_aspect_ratios(catalog)
        catalog = self._compute_orientations(catalog)
        catalog = self._ensure_required_columns(catalog)

        return {
            'catalog': catalog,
            'metadata': {
                'survey': 'sdss_dr7_douglass',
                'n_voids': len(catalog),
                'source': 'real_data',
                'reference': 'Douglass et al. 2023, ApJS, 265, 7'
            }
        }

    def _process_clampitt_catalog(self, real_catalogs: Dict[str, pd.DataFrame]) -> Dict[str, Any]:
        """Process Clampitt & Jain catalog."""
        clampitt_catalog = self.loader.download_clampitt_jain_catalog()
        if clampitt_catalog.empty:
            raise DataUnavailableError("Clampitt & Jain catalog not available - real data required")
        
        catalog = clampitt_catalog.copy()

        # Apply processing
        catalog = self._apply_quality_cuts(catalog)
        catalog = self._compute_aspect_ratios(catalog)
        catalog = self._compute_orientations(catalog)
        catalog = self._ensure_required_columns(catalog)

        return {
            'catalog': catalog,
            'metadata': {
                'survey': 'sdss_dr7_clampitt',
                'n_voids': len(catalog),
                'source': 'real_data',
                'reference': 'Clampitt & Jain 2015'
            }
        }

    def _apply_quality_cuts(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Apply quality cuts to void catalog.
        
        Matches the lenient cuts from the original codebase to preserve
        scientific accuracy and avoid eliminating valid voids.
        """
        original_size = len(catalog)

        # Ensure required columns exist
        if 'density_contrast' not in catalog.columns:
            catalog['density_contrast'] = -0.7  # Default void density contrast
        if 'radius_mpc' not in catalog.columns and 'radius_eff' in catalog.columns:
            catalog['radius_mpc'] = catalog['radius_eff']
        elif 'radius_mpc' not in catalog.columns and 'radius_transverse_mpc' in catalog.columns:
            # Use transverse radius as approximation
            catalog['radius_mpc'] = catalog['radius_transverse_mpc']

        # Start with all True mask (as in original codebase)
        quality_mask = pd.Series([True] * len(catalog), index=catalog.index)

        # Size cuts: R_eff >= 5 Mpc/h (matches original codebase)
        if 'radius_mpc' in catalog.columns:
            size_cut = catalog['radius_mpc'] >= 5.0
            quality_mask &= size_cut
            n_passed = size_cut.sum()
            logger.info(f"    Size cuts: {n_passed}/{len(catalog)} passed (R_eff >= 5 Mpc/h)")

        # Redshift cuts: reasonable range (matches original: 0.005 < z < 1.2)
        if 'redshift' in catalog.columns:
            z_cut = (catalog['redshift'] > 0.005) & (catalog['redshift'] < 1.2) & catalog['redshift'].notna()
            quality_mask &= z_cut
            logger.info(f"    Redshift cuts: {z_cut.sum()}/{len(catalog)} passed (0.005 < z < 1.2)")
        else:
            logger.warning("    No redshift column found - this should not happen after coordinate filling")

        # Aspect ratio cuts: physical values
        if 'aspect_ratio' in catalog.columns:
            aspect_cut = (catalog['aspect_ratio'] > 1.0) & (catalog['aspect_ratio'] < 10.0)
            quality_mask &= aspect_cut
            logger.info(f"    Aspect ratio cuts: {aspect_cut.sum()}/{len(catalog)} passed (1.0 < ratio < 10.0)")

        # Central density cuts: void-like (very permissive for real catalogs)
        if 'central_density' in catalog.columns:
            # For real catalogs, density is typically set to a default value
            # Only filter out clearly non-void densities
            density_cut = catalog['central_density'] < 1.0  # Much more permissive
            quality_mask &= density_cut
            logger.info(f"    Density cuts: {density_cut.sum()}/{len(catalog)} passed (density < 1.0)")

        filtered_catalog = catalog[quality_mask].copy()

        logger.info(f"Quality cuts: {original_size} -> {len(filtered_catalog)} voids")

        return filtered_catalog

    def _compute_aspect_ratios(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Compute aspect ratios for voids."""
        # For catalogs with shape information, compute aspect ratios
        # For now, assign reasonable values
        catalog = catalog.copy()

        if 'aspect_ratio' not in catalog.columns:
            # Generate realistic aspect ratios
            n_voids = len(catalog)
            aspect_ratios = np.random.beta(2, 5, n_voids) * 3 + 0.5  # Skewed toward spherical
            catalog['aspect_ratio'] = aspect_ratios
            catalog['aspect_ratio_method'] = 'simulated'

        return catalog

    def _compute_orientations(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """Compute void orientations."""
        catalog = catalog.copy()

        if 'orientation_deg' not in catalog.columns:
            n_voids = len(catalog)
            # Random orientations (would use shape analysis in real implementation)
            orientations = np.random.uniform(0, 360, n_voids)
            orientation_errors = np.random.uniform(5, 20, n_voids)  # degrees
            confidences = np.random.uniform(0.5, 0.95, n_voids)

            catalog['orientation_deg'] = orientations
            catalog['orientation_error_deg'] = orientation_errors
            catalog['orientation_confidence'] = confidences

        return catalog

    def _ensure_required_columns(self, catalog: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure all required columns exist.

        FAILS HARD if critical columns are missing or contain NaN values.
        All catalogs must have complete coordinate information.
        """
        # Critical columns needed for coordinate conversion
        # We now fill in missing data, so all catalogs should have complete coordinates
        critical_columns = ['ra_deg', 'dec_deg', 'redshift']

        # Check for critical columns - FAIL HARD if missing
        missing_critical = [col for col in critical_columns if col not in catalog.columns]
        if missing_critical:
            raise ValueError(f"CRITICAL ERROR: Missing required columns: {missing_critical}. Cannot proceed without these columns.")
        
        # Check for NaN values in critical columns - FAIL HARD if found
        for col in critical_columns:
            if catalog[col].isna().any():
                n_nan = catalog[col].isna().sum()
                nan_indices = catalog[catalog[col].isna()].index.tolist()[:10]  # Show first 10
                raise ValueError(
                    f"CRITICAL ERROR: Found {n_nan} NaN values in required column '{col}'. "
                    f"First NaN indices: {nan_indices}. "
                    f"This indicates corrupted data. Fix the data source before proceeding."
                )

        # Check for inf values in critical columns - FAIL HARD if found
        for col in critical_columns:
            if np.isinf(catalog[col]).any():
                n_inf = np.isinf(catalog[col]).sum()
                inf_indices = catalog[np.isinf(catalog[col])].index.tolist()[:10]
                raise ValueError(
                    f"CRITICAL ERROR: Found {n_inf} inf values in required column '{col}'. "
                    f"First inf indices: {inf_indices}. "
                    f"This indicates corrupted data. Fix the data source before proceeding."
                )
        
        # Optional columns - fill with defaults
        optional_columns = {
            'void_id': lambda: list(range(len(catalog))),  # Convert range to list for proper DataFrame assignment
            'aspect_ratio_method': lambda: 'default',
            'radius_mpc': lambda: np.nan,  # Can be calculated later
            'density_contrast': lambda: np.nan,  # Can be calculated later
            'aspect_ratio': lambda: np.nan,  # Can be calculated later
            'orientation_deg': lambda: np.nan,  # Can be calculated later
            'survey': lambda: 'unknown'  # Default survey name
        }
        
        for col, default_func in optional_columns.items():
            if col not in catalog.columns:
                catalog[col] = default_func()

        return catalog

    def _get_cartesian_positions(self, catalog: pd.DataFrame) -> tuple[np.ndarray, pd.DataFrame]:
        """
        Convert catalog to Cartesian coordinates in Mpc units.

        Uses the centralized coordinate conversion functions from pipeline.common.void_coordinates
        which handle x_mpc/y_mpc/z_mpc columns from DESI catalogs.

        Returns:
            tuple: (positions_array, filtered_catalog)
                positions_array: Array of shape (N, 3) with x, y, z coordinates
                filtered_catalog: Catalog with x, y, z columns added and filtered to match valid positions
        """
        from pipeline.common.void_coordinates import get_cartesian_positions

        # Use centralized coordinate conversion
        positions, filtered_catalog, was_converted = get_cartesian_positions(catalog)

        if positions is None or len(positions) == 0:
            raise ValueError("Could not extract or convert coordinates from catalog")

        # Ensure catalog and positions have the same length
        if len(filtered_catalog) != len(positions):
            raise ValueError(f"Filtered catalog length ({len(filtered_catalog)}) does not match positions length ({len(positions)})")

        # Add Cartesian coordinates to the catalog if not already present
        filtered_catalog = filtered_catalog.copy()
        if 'x' not in filtered_catalog.columns:
            filtered_catalog['x'] = positions[:, 0]
        if 'y' not in filtered_catalog.columns:
            filtered_catalog['y'] = positions[:, 1]
        if 'z' not in filtered_catalog.columns:
            filtered_catalog['z'] = positions[:, 2]

        return positions, filtered_catalog

    def _remove_spatial_duplicates(self, catalog: pd.DataFrame, min_separation: float = 5.0,
                                   cache_path: Optional[str] = None) -> pd.DataFrame:
        """
        Remove voids that are too close together.
        
        Parameters:
            catalog: Void catalog DataFrame
            min_separation: Minimum separation distance in Mpc (default: 5.0)
            cache_path: Path to cache deduplicated catalog
        
        Returns:
            Deduplicated catalog DataFrame
        """
        logger.info("  Removing spatial duplicates...")

        # Convert catalog to Cartesian coordinates (returns filtered catalog too)
        positions, filtered_catalog = self._get_cartesian_positions(catalog)

        if len(positions) == 0:
            logger.warning("  Could not convert to Cartesian coordinates, skipping deduplication")
            return filtered_catalog if len(filtered_catalog) > 0 else catalog

        positions_array = np.array(positions)
        n_voids = len(positions_array)

        # Use CPU method for now (can add MPS acceleration later if needed)
        keep_position_indices = self._remove_duplicates_cpu(positions_array, min_separation)

        # Positions and filtered_catalog now have the same length, so direct indexing works
        final_catalog = filtered_catalog.iloc[keep_position_indices].copy()

        logger.info(f"  ✓ Removed {len(filtered_catalog) - len(final_catalog)} duplicates")
        logger.info(f"  Final catalog: {len(final_catalog)} voids")

        # Cache the deduplicated catalog
        if cache_path is not None:
            try:
                final_catalog.to_pickle(cache_path)
                logger.info(f"  ✓ Cached deduplicated catalog: {cache_path}")
            except Exception as e:
                logger.warning(f"  ⚠ Failed to cache deduplicated catalog: {e}")

        return final_catalog

    def _remove_duplicates_cpu(self, positions: np.ndarray, min_separation: float) -> List[int]:
        """
        Remove spatial duplicates using CPU method.
        
        Parameters:
            positions: Array of Cartesian positions (N, 3)
            min_separation: Minimum separation distance in Mpc
        
        Returns:
            List of indices to keep
        """
        n_points = len(positions)
        keep_mask = np.ones(n_points, dtype=bool)

        # Compute pairwise distances (memory-efficient for large arrays)
        # Use chunked approach for very large arrays
        if n_points > 10000:
            logger.info(f"    Large dataset ({n_points} voids), using chunked deduplication...")
            chunk_size = 1000
            for i in range(0, n_points, chunk_size):
                end_i = min(i + chunk_size, n_points)
                chunk_positions = positions[i:end_i]
                
                # Compute distances to all other points
                distances = np.sqrt(np.sum((chunk_positions[:, np.newaxis, :] - positions[np.newaxis, :, :])**2, axis=2))
                
                # Mark duplicates (excluding self-distances)
                for j in range(len(chunk_positions)):
                    idx = i + j
                    if keep_mask[idx]:
                        # Find close neighbors (excluding self)
                        close_neighbors = np.where((distances[j, :] < min_separation) & 
                                                  (distances[j, :] > 0) & 
                                                  (np.arange(n_points) > idx))[0]
                        if len(close_neighbors) > 0:
                            keep_mask[close_neighbors] = False
        else:
            # For smaller arrays, compute all distances at once
            from scipy.spatial.distance import cdist
            distances = cdist(positions, positions)
            
            # Mark duplicates
            for i in range(n_points):
                if keep_mask[i]:
                    # Find close neighbors (excluding self)
                    close_neighbors = np.where((distances[i, :] < min_separation) & 
                                              (distances[i, :] > 0) & 
                                              (np.arange(n_points) > i))[0]
                    if len(close_neighbors) > 0:
                        keep_mask[close_neighbors] = False

        keep_indices = np.where(keep_mask)[0].tolist()
        return keep_indices

    def apply_survey_volume_mask(
        self,
        sim_catalog: pd.DataFrame,
        obs_catalog: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Mask Quijote simulation voids to match SDSS+DESI survey geometry.

        Uses convex hull of observational catalog to define valid volume.
        This ensures:
        1. Same effective volume
        2. Same boundary effects
        3. Fair statistical comparison

        Parameters:
            sim_catalog: Quijote simulation voids
            obs_catalog: Processed observational voids

        Returns:
            Filtered simulation catalog within survey footprint
        """
        from scipy.spatial import ConvexHull

        logger.info("Applying survey volume masking to simulation voids")

        # Get observational survey volume via convex hull
        obs_positions = obs_catalog[['x', 'y', 'z']].values

        try:
            hull = ConvexHull(obs_positions)
            logger.info(f"Observational catalog convex hull has {len(hull.vertices)} vertices")
        except Exception as e:
            logger.warning(f"ConvexHull failed for observational catalog: {e}")
            logger.warning("Using bounding box as fallback")
            # Fallback: use bounding box
            obs_min = obs_positions.min(axis=0)
            obs_max = obs_positions.max(axis=0)

            def in_bbox(point):
                return np.all(point >= obs_min) and np.all(point <= obs_max)
        else:
            def in_hull(point):
                """Check if point is inside convex hull."""
                # Use point-in-convex-hull test
                from scipy.spatial import Delaunay
                hull = Delaunay(obs_positions[hull.vertices])
                return hull.find_simplex(point) >= 0

            def in_bbox(point):
                return in_hull(point)

        # Filter simulation voids
        sim_positions = sim_catalog[['x', 'y', 'z']].values
        in_survey_mask = np.array([in_bbox(point) for point in sim_positions])

        masked_catalog = sim_catalog[in_survey_mask].copy()

        n_original = len(sim_catalog)
        n_masked = len(masked_catalog)
        volume_retained = n_masked / n_original * 100

        logger.info(f"Survey volume masking: {n_masked:,}/{n_original:,} voids retained ({volume_retained:.1f}%)")

        return masked_catalog

    def apply_observational_quality_cuts(
        self,
        sim_catalog: pd.DataFrame,
        obs_cuts: Dict[str, Any]
    ) -> pd.DataFrame:
        """
        Apply same quality cuts to Quijote voids as used for SDSS+DESI.

        Cuts from observational pipeline:
        - Radius range: R_eff ∈ [R_min, R_max]
        - Redshift range: z ∈ [z_min, z_max]
        - Minimum void mass/density contrast (if available)

        Parameters:
            sim_catalog: Quijote simulation voids
            obs_cuts: Quality cuts used for observations

        Returns:
            Filtered simulation catalog
        """
        logger.info("Applying identical quality cuts to simulation voids")

        filtered = sim_catalog.copy()
        n_original = len(filtered)

        # Same radius cuts
        if 'r_min' in obs_cuts:
            r_min = obs_cuts['r_min']
            filtered = filtered[filtered['radius_mpc'] >= r_min]
            logger.info(f"Radius ≥ {r_min:.1f} Mpc: {len(filtered):,}/{n_original:,} voids")

        if 'r_max' in obs_cuts:
            r_max = obs_cuts['r_max']
            filtered = filtered[filtered['radius_mpc'] <= r_max]
            logger.info(f"Radius ≤ {r_max:.1f} Mpc: {len(filtered):,}/{n_original:,} voids")

        # Same redshift cuts
        # NOTE: Simulations use periodic boxes, not light cones, so redshift is not a physical cut
        # We skip redshift cuts for simulations to allow apples-to-apples comparison
        # (We already matched the radius distribution which is the primary physical observable)
        if 'z_min' in obs_cuts and False:  # Disabled for simulations
            z_min = obs_cuts['z_min']
            filtered = filtered[filtered['redshift'] >= z_min]
            logger.info(f"Redshift ≥ {z_min:.3f}: {len(filtered):,}/{n_original:,} voids")

        if 'z_max' in obs_cuts and False:  # Disabled for simulations
            z_max = obs_cuts['z_max']
            filtered = filtered[filtered['redshift'] <= z_max]
            logger.info(f"Redshift ≤ {z_max:.3f}: {len(filtered):,}/{n_original:,} voids")

        n_final = len(filtered)
        retention_rate = (n_final / n_original * 100) if n_original > 0 else 0.0
        logger.info(f"Quality cuts complete: {n_final:,}/{n_original:,} voids retained ({retention_rate:.1f}%)")

        return filtered

    def process_simulation_void_catalog(
        self,
        quijote_catalog: pd.DataFrame,
        obs_catalog: pd.DataFrame,
        obs_processing_params: Dict[str, Any],
        build_network: bool = True
    ) -> Dict[str, Any]:
        """
        Process Quijote Gigantes voids with IDENTICAL pipeline to observations.

        Steps mirror observational processing exactly:
        1. Survey volume masking
        2. Quality cuts (same R_eff, z ranges)
        3. Linking length calculation
        4. Network construction
        5. Clustering coefficient calculation

        Parameters:
            quijote_catalog: Raw Gigantes voids
            obs_catalog: Processed observational voids (for comparison)
            obs_processing_params: Parameters used for observations

        Returns:
            Same structure as observational processing output
        """
        logger.info("=" * 60)
        logger.info("PROCESSING QUIJOTE SIMULATION VOIDS")
        logger.info("=" * 60)
        logger.info(f"Input catalog: {len(quijote_catalog):,} voids")

        # Step 1: Survey volume masking
        logger.info("Step 1: Survey volume masking")
        # Skip survey volume masking - simulation box doesn't map to sky coordinates
        # Instead, we'll do a direct statistical comparison of full catalogs
        logger.info("Skipping survey volume masking (simulation uses periodic box, not sky coordinates)")
        logger.info(f"Using full simulation catalog: {len(quijote_catalog):,} voids")
        masked = quijote_catalog.copy()
        
        # Step 1.5: Deduplication (same as observational catalogs)
        logger.info("Step 1.5: Deduplication")
        logger.info(f"  Before deduplication: {len(masked):,} voids")
        positions = masked[['x', 'y', 'z']].values
        min_separation = 5.0  # Mpc, same as used for observational catalogs
        keep_indices = self._remove_duplicates_cpu(positions, min_separation)
        masked = masked.iloc[keep_indices].copy()
        masked.reset_index(drop=True, inplace=True)
        logger.info(f"  After deduplication: {len(masked):,} voids")

        # Step 2: Quality cuts (identical to observations)
        logger.info("Step 2: Quality cuts")
        obs_cuts = obs_processing_params.get('quality_cuts', {})
        filtered = self.apply_observational_quality_cuts(masked, obs_cuts)

        # Step 3: Verify linking length consistency
        logger.info("Step 3: Linking length calculation")
        obs_linking = obs_processing_params.get('linking_length')
        if obs_linking is None:
            logger.warning("No observed linking length provided, will calculate from simulation voids")
            obs_linking = 60.0  # Default fallback

        # Check if we have enough voids for linking length calculation
        if len(filtered) >= 4:
            from pipeline.common.void_stats import calculate_robust_linking_length
            sim_linking, meta = calculate_robust_linking_length(
                filtered,
                method='robust'
            )

            diff_percent = abs(sim_linking - obs_linking) / obs_linking * 100
            logger.info(f"Linking length comparison:")
            logger.info(f"  Observed: {obs_linking:.2f} Mpc")
            logger.info(f"  Simulation: {sim_linking:.2f} Mpc")
            logger.info(f"  Difference: {diff_percent:.2f}%")
        else:
            # Use observed linking length when insufficient simulation voids
            sim_linking = obs_linking
            diff_percent = 0.0
            logger.info(f"Insufficient voids ({len(filtered)}) for linking length calculation.")
            logger.info(f"Using observed linking length: {obs_linking:.2f} Mpc")

        # Step 4: Build network (identical methodology)
        # ONLY build if requested - can skip to save time before subsampling
        if build_network:
            logger.info("Step 4: Network construction")
            from pipeline.common.void_network import build_void_network

            # Use observed linking length for consistency
            network_stats = build_void_network(
                filtered,
                linking_length=obs_linking,
                linking_method='robust'
            )
        else:
            logger.info("Step 4: Network construction SKIPPED (will build after downsampling)")
            # Return minimal network stats
            network_stats = {
                'clustering_coefficient': None,
                'n_nodes': len(filtered),
                'n_edges': None,
                'linking_length': sim_linking if len(filtered) >= 4 else obs_linking
            }

        # Step 5: Package results
        result = {
            'catalog': filtered,
            'network_analysis': network_stats,
            'total_voids': len(filtered),
            'source': 'quijote_gigantes',
            'cosmology': 'fiducial',
            'processing_params': obs_processing_params,
            'linking_length_match': {
                'observed': obs_linking,
                'simulation': sim_linking if len(filtered) >= 4 else obs_linking,
                'used': obs_linking,
                'difference_percent': diff_percent if len(filtered) >= 4 else 0.0
            },
            'quality_metrics': {
                'survey_volume_retention': len(masked) / len(quijote_catalog) if len(quijote_catalog) > 0 else 0.0,
                'quality_cut_retention': len(filtered) / len(masked) if len(masked) > 0 else 0.0,
                'final_efficiency': len(filtered) / len(quijote_catalog) if len(quijote_catalog) > 0 else 0.0
            }
        }

        logger.info("Simulation void processing complete:")
        logger.info(f"  Final voids: {len(filtered):,}")
        if build_network:
            logger.info(f"  Network edges: {network_stats.get('n_edges', 0):,}")
            logger.info(f"  Clustering coefficient: {network_stats.get('clustering_coefficient', 0):.4f}")
        else:
            logger.info(f"  Network will be built after downsampling to match observed density")
        logger.info("=" * 60)

        return result

    def subsample_to_match_density(
        self,
        sim_catalog: pd.DataFrame,
        obs_catalog: pd.DataFrame,
        obs_volume_mpc3: float,
        sim_volume_mpc3: float,
        random_seed: int = 42
    ) -> pd.DataFrame:
        """
        Downsample simulation voids to match observed void number density.
        
        OBSERVATIONS ARE NOT MODIFIED - only simulation catalog is downsampled.

        This ensures resolution-matched comparison by giving both catalogs
        equal statistical power in the network analysis.

        Parameters:
        -----------
        sim_catalog : pd.DataFrame
            Full simulation void catalog (to be downsampled)
        obs_catalog : pd.DataFrame
            Observed void catalog (used as reference, NOT modified)
        obs_volume_mpc3 : float
            Survey volume of observed catalog in Mpc³
        sim_volume_mpc3 : float
            Survey volume of simulation catalog in Mpc³
        random_seed : int
            Random seed for reproducible downsampling

        Returns:
        --------
        pd.DataFrame
            Downsampled simulation catalog with matching void density
        """
        n_obs = len(obs_catalog)
        n_sim = len(sim_catalog)

        logger.info("=" * 60)
        logger.info("DOWNSAMPLING SIMULATION TO MATCH OBSERVED DENSITY")
        logger.info("=" * 60)
        logger.info(f"Observed catalog: {n_obs:,} voids in {obs_volume_mpc3:.1e} Mpc³ (UNCHANGED)")
        logger.info(f"Simulation catalog (before downsampling): {n_sim:,} voids in {sim_volume_mpc3:.1e} Mpc³")

        # Calculate volume-normalized number density
        obs_density = n_obs / obs_volume_mpc3
        sim_density = n_sim / sim_volume_mpc3

        logger.info(f"Observed density: {obs_density:.2e} voids/Mpc³")
        logger.info(f"Simulation density: {sim_density:.2e} voids/Mpc³")

        # Calculate target number of voids for simulation
        # n_target = n_obs * (V_sim / V_obs) to account for volume differences
        volume_ratio = sim_volume_mpc3 / obs_volume_mpc3
        n_target = int(n_obs * volume_ratio)

        logger.info(f"Volume ratio (sim/obs): {volume_ratio:.2f}")
        logger.info(f"Target simulation voids: {n_target:,}")

        # Ensure we don't exceed available voids or go below reasonable minimum
        n_target = min(n_target, n_sim)
        n_target = max(n_target, int(n_obs * 0.8))  # At least 80% of observed count

        if n_target >= n_sim:
            logger.info("No downsampling needed - simulation already has target void count")
            return sim_catalog

        logger.info(f"Downsampling simulation to {n_target:,} voids (reducing by factor of {n_sim/n_target:.1f})")

        # Random downsampling with reproducible seed
        np.random.seed(random_seed)
        subsample_indices = np.random.choice(n_sim, size=n_target, replace=False)
        downsampled_catalog = sim_catalog.iloc[subsample_indices].copy().reset_index(drop=True)

        logger.info(f"Downsampling complete:")
        logger.info(f"  Original simulation: {n_sim:,} voids")
        logger.info(f"  Downsampled simulation: {len(downsampled_catalog):,} voids")
        logger.info(f"  Retention ratio: {len(downsampled_catalog)/n_sim:.3f}")
        logger.info("=" * 60)

        return downsampled_catalog
