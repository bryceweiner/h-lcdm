"""
H-ZOBOV Pipeline
=================

Main pipeline class for H-ZOBOV void-finding algorithm with H-ΛCDM integration.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from pathlib import Path
import logging
import json
from tqdm import tqdm

from ...common.base_pipeline import AnalysisPipeline
from .zobov_parameters import HZOBOVParameters
from .zobov_core import ZOBOVCore, ZOBOVCoreError
from .zobov_checkpoint import HZOBOVCheckpointManager, HZOBOV_STAGES
from .hlcdm_integration import get_lambda_at_redshift

logger = logging.getLogger(__name__)


class HZOBOVPipelineError(Exception):
    """Error in H-ZOBOV pipeline."""
    pass


class HZOBOVPipeline(AnalysisPipeline):
    """
    H-ZOBOV void-finding pipeline with H-ΛCDM Lambda(z) integration.
    
    Implements the complete ZOBOV algorithm (Neyrinck 2008) with:
    - Voronoi tessellation and DTFE density estimation
    - Watershed zone finding
    - Zone merging with Λ(z)-dependent significance thresholds
    - Apple Silicon MPS acceleration
    - Stage-based checkpointing
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize H-ZOBOV pipeline.
        
        Parameters:
            output_dir: Output directory for results
        """
        super().__init__("hzobov", output_dir)
        
        # Configure all H-ZOBOV module loggers to write to pipeline log file
        self._configure_zobov_loggers()
        
        # Initialize checkpoint manager (will be set when output_name is known)
        self.checkpoint_manager = None
        
        # Pipeline state
        self.parameters = None
        self.zobov_core = None
        
        self.update_metadata('description', 'H-ZOBOV: Holographic ZOBOV void-finding with H-ΛCDM Lambda(z)')
        self.update_metadata('algorithm', 'ZOBOV (Neyrinck 2008)')
        self.update_metadata('hlcdm_integration', True)
    
    def _configure_zobov_loggers(self):
        """Configure all H-ZOBOV module loggers to write to the pipeline log file."""
        import logging
        
        # Get or create the file handler for the pipeline log file
        pipeline_logger = logging.getLogger(f"pipeline.{self.name}")
        file_handler = None
        for handler in pipeline_logger.handlers:
            if isinstance(handler, logging.FileHandler):
                file_handler = handler
                break
        
        if file_handler is None:
            # Create file handler if it doesn't exist
            file_handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
            file_handler.setLevel(logging.INFO)
            file_formatter = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
            file_handler.setFormatter(file_formatter)
            pipeline_logger.addHandler(file_handler)
        
        # Create formatter for loggers
        formatter = logging.Formatter('[%(asctime)s] %(name)s: %(message)s', datefmt='%H:%M:%S')
        
        # Configure all H-ZOBOV module loggers
        zobov_logger_names = [
            'pipeline.voidfinder.zobov.zobov_core',
            'pipeline.voidfinder.zobov.zobov_pipeline',
            'pipeline.voidfinder.zobov.voronoi_tessellation',
            'pipeline.voidfinder.zobov.watershed',
            'pipeline.voidfinder.zobov.zone_merger',
            'pipeline.voidfinder.zobov.zobov_mps',
            'pipeline.voidfinder.zobov.hlcdm_integration',
            'pipeline.voidfinder.zobov.zobov_parameters',
            'pipeline.voidfinder.zobov.zobov_checkpoint',
            'data.processors.void_processor',  # Also configure void processor logger
            'data.loader',  # Configure data loader logger
        ]
        
        log_file_str = str(self.log_file)
        for logger_name in zobov_logger_names:
            module_logger = logging.getLogger(logger_name)
            module_logger.setLevel(logging.INFO)
            
            # Verify logger does not already have this file handler
            has_handler = any(
                isinstance(h, logging.FileHandler) and 
                hasattr(h, 'baseFilename') and 
                h.baseFilename == log_file_str
                for h in module_logger.handlers
            )
            
            if not has_handler:
                # Create a new handler with the same file
                handler = logging.FileHandler(self.log_file, mode='a', encoding='utf-8')
                handler.setLevel(logging.INFO)
                handler.setFormatter(formatter)
                module_logger.addHandler(handler)
                # Allow propagation so messages also go to parent loggers
                # but set level to avoid duplicate messages at root
                module_logger.propagate = True
    
    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute H-ZOBOV pipeline.
        
        Parameters:
            context: Analysis parameters including:
                - output_name: MANDATORY base name for output files
                - catalog: Catalog name (default: 'sdss_dr16')
                - z_min: Minimum redshift (default: 0.0)
                - z_max: Maximum redshift (default: 1.0)
                - use_hlcdm_lambda: Use H-ΛCDM Lambda(z) (default: True)
                - batch_size: MPS batch size (default: 50000)
                - significance_ratio: Zone merging threshold (optional)
                - min_void_volume: Minimum void volume filter (optional)
                - use_start_checkpoint: Resume from checkpoint (default: False)
        
        Returns:
            dict: Pipeline results
        """
        try:
            self.log_progress("Starting H-ZOBOV pipeline...")
            
            # Parse context parameters
            if context is None:
                context = {}
            
            # MANDATORY: output_name
            output_name = context.get('output_name')
            if output_name is None:
                raise HZOBOVPipelineError("output_name is MANDATORY for H-ZOBOV pipeline")
            
            catalog_name = context.get('catalog', 'sdss_dr16')
            z_min = context.get('z_min', 0.0)
            z_max = context.get('z_max', 1.0)
            use_hlcdm_lambda = context.get('use_hlcdm_lambda', True)
            batch_size = context.get('batch_size', 50000)
            significance_ratio = context.get('significance_ratio', None)
            min_void_volume = context.get('min_void_volume', None)
            use_start_checkpoint = context.get('use_start_checkpoint', False)
            z_bin_size = context.get('z_bin_size', 0.05)  # Default 0.05 for H-ZOBOV
            
            # Create parameters
            self.parameters = HZOBOVParameters(
                output_name=output_name,
                z_min=z_min,
                z_max=z_max,
                use_hlcdm_lambda=use_hlcdm_lambda,
                batch_size=batch_size,
                significance_ratio=significance_ratio,
                min_void_volume=min_void_volume,
                z_bin_size=z_bin_size
            )
            self.parameters.validate()
            
            # Initialize checkpoint manager
            checkpoint_dir = self.processed_data_dir / "checkpoints"
            self.checkpoint_manager = HZOBOVCheckpointManager(checkpoint_dir, output_name)
            
            # Handle existing output files (DELETE as per requirements)
            # Cleanup occurs after parameters are initialized
            
            # Load galaxy catalog (reuse voidfinder pipeline logic)
            self.log_progress("Loading galaxy catalog...")
            galaxy_catalog = self._load_galaxy_catalog(catalog_name, z_min, z_max, context)
            
            if galaxy_catalog is None or len(galaxy_catalog) == 0:
                if galaxy_catalog is None:
                    error_msg = f"Failed to load galaxy catalog '{catalog_name}' (catalog download returned None)"
                else:
                    error_msg = f"No galaxies found in catalog '{catalog_name}' for redshift range z={z_min:.6f} - {z_max:.6f}"
                    error_msg += f" (catalog loaded but empty after filtering)"
                self.log_progress(f"✗ {error_msg}")
                raise HZOBOVPipelineError(error_msg)
            
            self.log_progress(f"✓ Loaded {len(galaxy_catalog):,} galaxies")
            
            # Convert to comoving coordinates if needed
            if not all(col in galaxy_catalog.columns for col in ['x', 'y', 'z']):
                self.log_progress("Converting to comoving coordinates...")
                galaxy_catalog = self._convert_to_comoving(galaxy_catalog)
            
            # Handle existing output files (DELETE as per requirements)
            self._cleanup_existing_outputs()
            
            # Initialize ZOBOV core
            self.zobov_core = ZOBOVCore(self.parameters)
            
            # Run H-ZOBOV algorithm with redshift binning (default for H-ΛCDM)
            # Auto-disable binning if redshift range is too small
            if self.parameters.z_bin_size is not None:
                redshift_range = self.parameters.z_max - self.parameters.z_min
                if self.parameters.z_bin_size >= redshift_range:
                    self.log_progress(f"Redshift range ({redshift_range:.3f}) smaller than bin size ({self.parameters.z_bin_size}), disabling binning")
                    self.parameters.z_bin_size = None
            
            if self.parameters.z_bin_size is not None:
                results = self._run_with_redshift_binning(galaxy_catalog, use_start_checkpoint)
            else:
                # Process all at once (for small redshift ranges)
                self.log_progress("Processing without redshift binning (small redshift range)")
                results = self._run_with_checkpointing(galaxy_catalog, use_start_checkpoint)
            
            # Save catalog immediately after void finding
            if results['void_catalog'] is not None and len(results['void_catalog']) > 0:
                self.log_progress("Saving void catalog...")
                self._save_void_catalog(results['void_catalog'])
                self.log_progress(f"✓ Void catalog saved: {len(results['void_catalog']):,} voids")
            
            # Save results (JSON and report)
            self.log_progress("Saving results...")
            self._save_results(results, galaxy_catalog)
            
            # Generate report
            self.log_progress("Generating report...")
            self._generate_report(results, galaxy_catalog)
            
            self.log_progress("✓ H-ZOBOV pipeline complete")
            
            return results
            
        except Exception as e:
            error_msg = f"Fatal error in H-ZOBOV pipeline: {type(e).__name__}: {str(e)}"
            self.log_progress(f"✗ {error_msg}")
            import traceback
            self.log_progress(f"Traceback: {traceback.format_exc()}")
            raise HZOBOVPipelineError(error_msg) from e
    
    def _cleanup_existing_outputs(self):
        """Delete existing output files if they exist."""
        base_name = self.parameters.get_output_filename_base(include_params=True)
        
        # Files to check and delete
        extensions = ['.json', '.pkl', '.fits', '.log', '.md']
        dirs_to_check = [
            self.json_dir,
            self.processed_data_dir,
            self.logs_dir,
            self.reports_dir
        ]
        
        deleted_count = 0
        for directory in dirs_to_check:
            for ext in extensions:
                pattern = f"{base_name}*{ext}"
                for file_path in directory.glob(pattern):
                    try:
                        file_path.unlink()
                        deleted_count += 1
                        logger.info(f"Deleted existing file: {file_path}")
                    except Exception as e:
                        logger.warning(f"Failed to delete {file_path}: {e}")
        
        if deleted_count > 0:
            self.log_progress(f"Deleted {deleted_count} existing output files")
    
    def _load_galaxy_catalog(self, catalog_name: str, z_min: float, z_max: float,
                            context: Dict[str, Any]) -> Optional[pd.DataFrame]:
        """Load galaxy catalog (reuse voidfinder pipeline logic)."""
        from ..voidfinder_pipeline import VoidFinderPipeline
        
        # Use voidfinder pipeline's catalog loading
        voidfinder = VoidFinderPipeline(self.base_output_dir)
        
        # Download/load catalog
        try:
            logger.info(f"Downloading catalog with redshift range: z={z_min:.6f} - {z_max:.6f}")
            galaxy_catalog = voidfinder._download_catalog(
                catalog_name, z_min, z_max,
                context.get('mag_limit', 22.0),
                context.get('force_redownload', False)
            )
        except Exception as e:
            logger.error(f"Error downloading catalog: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return None
        
        if galaxy_catalog is None:
            logger.warning("Catalog download returned None")
            return None
        
        if len(galaxy_catalog) == 0:
            logger.warning(f"Catalog is empty after download (redshift range z={z_min:.6f} - {z_max:.6f} may be too narrow or contain no galaxies)")
            logger.warning(f"  Redshift range width: {z_max - z_min:.6f} (comoving distance ~{(z_max - z_min) * 3000:.1f} Mpc)")
            logger.warning(f"  This redshift range may be too narrow for SDSS DR16. Try a wider range (e.g., z_min=0.01, z_max=0.1)")
            return None
        
        logger.info(f"Catalog loaded: {len(galaxy_catalog):,} galaxies in redshift range z={z_min:.6f} - {z_max:.6f}")
        
        # Apply volume-limited sample cut if specified
        abs_mag_limit = context.get('abs_mag_limit', None)
        if abs_mag_limit is not None:
            try:
                before_count = len(galaxy_catalog)
                galaxy_catalog = voidfinder._filter_volume_limited(galaxy_catalog, abs_mag_limit)
                if galaxy_catalog is None:
                    logger.warning(f"Volume-limited filter returned None (abs_mag_limit={abs_mag_limit})")
                    return None
                after_count = len(galaxy_catalog)
                logger.info(f"Volume-limited filter: {before_count:,} -> {after_count:,} galaxies (abs_mag_limit={abs_mag_limit})")
                if after_count == 0:
                    logger.warning("Catalog is empty after volume-limited filter")
                    return None
            except Exception as e:
                logger.error(f"Error applying volume-limited filter: {e}")
                import traceback
                logger.error(traceback.format_exc())
                return None
        
        return galaxy_catalog
    
    def _convert_to_comoving(self, galaxy_catalog: pd.DataFrame) -> pd.DataFrame:
        """Convert to comoving coordinates (reuse voidfinder pipeline logic)."""
        from ..voidfinder_pipeline import VoidFinderPipeline
        
        voidfinder = VoidFinderPipeline(self.base_output_dir)
        return voidfinder._convert_to_comoving(galaxy_catalog)
    
    def _run_with_redshift_binning(self, galaxy_catalog: pd.DataFrame,
                                   use_start_checkpoint: bool) -> Dict[str, Any]:
        """
        Run H-ZOBOV algorithm with redshift binning (default for H-ΛCDM).
        
        Processes each redshift bin separately to ensure consistent Λ(z) within bins,
        then combines results with deduplication at bin boundaries.
        """
        z_bin_size = self.parameters.z_bin_size
        z_min = self.parameters.z_min
        z_max = self.parameters.z_max
        
        # Create redshift bins
        z_bins = []
        current_z = z_min
        while current_z < z_max:
            bin_z_max = min(current_z + z_bin_size, z_max)
            z_bins.append((current_z, bin_z_max))
            current_z = bin_z_max
        
        self.log_progress(f"Processing {len(z_bins)} redshift bins: {z_bins}")
        self.log_progress(f"Bin size: {z_bin_size} (default for H-ZOBOV to ensure consistent Λ(z))")
        
        all_void_catalogs = []
        all_voronoi_data = []
        all_watershed_data = []
        all_merge_data = []
        
        # Process each redshift bin
        for bin_idx, (bin_z_min, bin_z_max) in enumerate(z_bins):
            self.log_progress("=" * 80)
            self.log_progress(f"Redshift bin {bin_idx + 1}/{len(z_bins)}: z={bin_z_min:.3f}-{bin_z_max:.3f}")
            self.log_progress("=" * 80)
            
            # Select galaxies within this redshift bin
            if 'redshift' in galaxy_catalog.columns:
                bin_mask = (galaxy_catalog['redshift'] >= bin_z_min) & (galaxy_catalog['redshift'] < bin_z_max)
            elif 'z' in galaxy_catalog.columns:
                # Check if z is redshift or comoving coordinate
                z_values = galaxy_catalog['z'].values
                if np.all((z_values > 0) & (z_values < 10)):
                    # Looks like redshift
                    bin_mask = (galaxy_catalog['z'] >= bin_z_min) & (galaxy_catalog['z'] < bin_z_max)
                else:
                    # Comoving coordinates - need redshift column
                    self.log_progress("⚠ Cannot bin by redshift: 'z' column appears to be comoving coordinates")
                    # Fall back to processing all at once
                    return self._run_with_checkpointing(galaxy_catalog, use_start_checkpoint)
            else:
                self.log_progress("⚠ Cannot bin by redshift: no redshift column found")
                # Fall back to processing all at once
                return self._run_with_checkpointing(galaxy_catalog, use_start_checkpoint)
            
            bin_galaxies = galaxy_catalog[bin_mask].copy()
            
            if len(bin_galaxies) == 0:
                self.log_progress(f"  No galaxies in bin z={bin_z_min:.3f}-{bin_z_max:.3f}, skipping")
                continue
            
            self.log_progress(f"  Bin contains {len(bin_galaxies):,} galaxies")
            
            # Configure parameters for this bin (enables redshift-dependent Λ(z) calculations)
            bin_parameters = HZOBOVParameters(
                output_name=f"{self.parameters.output_name}_bin_{bin_idx}",
                z_min=bin_z_min,
                z_max=bin_z_max,
                use_hlcdm_lambda=self.parameters.use_hlcdm_lambda,
                batch_size=self.parameters.batch_size,
                significance_ratio=self.parameters.significance_ratio,
                min_void_volume=self.parameters.min_void_volume,
                z_bin_size=None  # No sub-binning within bin
            )
            
            # Initialize ZOBOV core algorithm for this bin
            bin_zobov_core = ZOBOVCore(bin_parameters)
            
            # Execute H-ZOBOV algorithm on this bin
            try:
                bin_results = bin_zobov_core.process(bin_galaxies)
                
                if bin_results['void_catalog'] is not None and len(bin_results['void_catalog']) > 0:
                    # Add bin metadata
                    bin_results['void_catalog']['bin_z_min'] = bin_z_min
                    bin_results['void_catalog']['bin_z_max'] = bin_z_max
                    bin_results['void_catalog']['bin_idx'] = bin_idx
                    
                    all_void_catalogs.append(bin_results['void_catalog'])
                    all_voronoi_data.append(bin_results.get('voronoi_data'))
                    all_watershed_data.append(bin_results.get('watershed_data'))
                    all_merge_data.append(bin_results.get('merge_data'))
                    
                    self.log_progress(f"  ✓ Bin {bin_idx + 1}: Found {len(bin_results['void_catalog']):,} voids")
                else:
                    self.log_progress(f"  ✓ Bin {bin_idx + 1}: No voids found")
                    
            except Exception as e:
                error_msg = f"Error processing redshift bin {bin_idx + 1} (z={bin_z_min:.3f}-{bin_z_max:.3f}): {e}"
                self.log_progress(f"  ✗ {error_msg}")
                import traceback
                self.log_progress(f"  Traceback: {traceback.format_exc()}")
                raise HZOBOVPipelineError(error_msg) from e
        
        # Combine all bins
        if len(all_void_catalogs) == 0:
            self.log_progress("⚠ No voids found in any redshift bin")
            return {
                'void_catalog': pd.DataFrame(),
                'voronoi_data': None,
                'watershed_data': None,
                'merge_data': None,
                'n_voids': 0
            }
        
        self.log_progress(f"Combining {len(all_void_catalogs)} redshift bins...")
        combined_catalog = pd.concat(all_void_catalogs, ignore_index=True)
        
        # Remove duplicates at bin boundaries (voids that overlap significantly)
        self.log_progress("Removing duplicate voids at bin boundaries...")
        combined_catalog = self._remove_duplicate_voids(combined_catalog)
        
        self.log_progress(f"✓ Total voids from all bins: {len(combined_catalog):,}")
        
        return {
            'void_catalog': combined_catalog,
            'voronoi_data': all_voronoi_data[0] if all_voronoi_data else None,
            'watershed_data': all_watershed_data[0] if all_watershed_data else None,
            'merge_data': all_merge_data[0] if all_merge_data else None,
            'n_voids': len(combined_catalog),
            'n_bins_processed': len(all_void_catalogs)
        }
    
    def _remove_duplicate_voids(self, void_catalog: pd.DataFrame,
                               min_separation: float = 5.0) -> pd.DataFrame:
        """
        Remove duplicate voids that are too close together (at bin boundaries).
        
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
        
        # Compute pairwise distances using modular function with automatic chunking
        distances = compute_pairwise_distances(positions)
        
        # Mark duplicates
        keep_mask = np.ones(len(void_catalog), dtype=bool)
        
        for i in range(len(void_catalog)):
            if keep_mask[i]:
                # Find close neighbors (excluding self)
                close = np.where((distances[i, :] < min_separation) & 
                                (distances[i, :] > 0) & 
                                (np.arange(len(void_catalog)) > i))[0]
                
                # Keep the void with larger volume (or radius if volume not available)
                for j in close:
                    if keep_mask[j]:
                        if 'volume_mpc3' in void_catalog.columns:
                            if void_catalog.iloc[i]['volume_mpc3'] >= void_catalog.iloc[j]['volume_mpc3']:
                                keep_mask[j] = False
                            else:
                                keep_mask[i] = False
                                break
                        elif 'radius_mpc' in void_catalog.columns:
                            if void_catalog.iloc[i]['radius_mpc'] >= void_catalog.iloc[j]['radius_mpc']:
                                keep_mask[j] = False
                            else:
                                keep_mask[i] = False
                                break
                        else:
                            # Default: keep first one
                            keep_mask[j] = False
        
        filtered = void_catalog[keep_mask].copy()
        
        if len(filtered) < len(void_catalog):
            self.log_progress(f"Removed {len(void_catalog) - len(filtered)} duplicate voids at bin boundaries")
        
        return filtered
    
    def _run_with_checkpointing(self, galaxy_catalog: pd.DataFrame,
                               use_start_checkpoint: bool) -> Dict[str, Any]:
        """Run H-ZOBOV algorithm with stage-based checkpointing (no binning)."""
        
        # Determine starting stage
        start_stage = None
        if use_start_checkpoint:
            start_stage = self.checkpoint_manager.get_checkpoint_stage()
            if start_stage:
                self.log_progress(f"Resuming from checkpoint: stage '{start_stage}'")
        
        # Stage 1: Voronoi tessellation
        voronoi_data = None
        if start_stage is None or start_stage == 'voronoi':
            checkpoint = self.checkpoint_manager.load_checkpoint('voronoi') if use_start_checkpoint else None
            if checkpoint:
                self.log_progress("Loading Voronoi checkpoint...")
                # Reconstruct from checkpoint (simplified - would need full serialization)
                raise NotImplementedError("Checkpoint loading not fully implemented")
            else:
                voronoi_data = self.zobov_core.run_stage_voronoi(galaxy_catalog)
                self.checkpoint_manager.save_checkpoint('voronoi', {
                    'n_galaxies': voronoi_data['n_galaxies'],
                    'volumes': voronoi_data['volumes'].tolist(),
                    'densities': voronoi_data['densities'].tolist()
                })
        
        # Stage 2: Watershed zones
        watershed_data = None
        if start_stage in [None, 'voronoi'] or start_stage == 'watershed':
            if voronoi_data is None:
                raise HZOBOVPipelineError("Cannot run watershed without Voronoi data")
            watershed_data = self.zobov_core.run_stage_watershed(voronoi_data)
            self.checkpoint_manager.save_checkpoint('watershed', {
                'n_zones': watershed_data['n_zones'],
                'zone_ids': watershed_data['zone_ids'].tolist()
            })
        
        # Stage 3: Zone merging
        merge_data = None
        if start_stage in [None, 'voronoi', 'watershed'] or start_stage == 'merging':
            if voronoi_data is None or watershed_data is None:
                raise HZOBOVPipelineError("Cannot run merging without previous stages")
            merge_data = self.zobov_core.run_stage_merging(
                voronoi_data, watershed_data, galaxy_catalog
            )
            self.checkpoint_manager.save_checkpoint('merging', {
                'n_voids': merge_data['n_voids'],
                'void_ids': merge_data['void_ids'].tolist()
            })
        
        # Stage 4: Void catalog
        void_catalog = None
        if start_stage in [None, 'voronoi', 'watershed', 'merging'] or start_stage == 'catalog':
            if voronoi_data is None or watershed_data is None or merge_data is None:
                raise HZOBOVPipelineError("Cannot generate catalog without previous stages")
            void_catalog = self.zobov_core.generate_void_catalog(
                voronoi_data, watershed_data, merge_data, galaxy_catalog
            )
        
        # Clear checkpoints on success
        self.checkpoint_manager.clear_all_checkpoints()
        
        return {
            'void_catalog': void_catalog,
            'voronoi_data': voronoi_data,
            'watershed_data': watershed_data,
            'merge_data': merge_data,
            'n_voids': len(void_catalog) if void_catalog is not None else 0
        }
    
    def _save_void_catalog(self, void_catalog: pd.DataFrame):
        """Save void catalog to disk immediately after void finding."""
        base_name = self.parameters.get_output_filename_base(include_params=True)
        catalog_path = self.processed_data_dir / f"{base_name}_catalog.pkl"
        void_catalog.to_pickle(catalog_path)
        logger.info(f"Void catalog saved: {catalog_path}")
    
    def _save_results(self, results: Dict[str, Any], galaxy_catalog: pd.DataFrame):
        """Save results to JSON (catalog already saved separately)."""
        base_name = self.parameters.get_output_filename_base(include_params=True)
        
        # Save JSON results
        json_path = self.json_dir / f"{base_name}.json"
        json_data = {
            'pipeline': 'hzobov',
            'output_name': self.parameters.output_name,
            'parameters': self.parameters.to_dict(),
            'n_galaxies': len(galaxy_catalog),
            'n_voids': results['n_voids'],
            'void_catalog_file': str(self.processed_data_dir / f"{base_name}_catalog.pkl"),
            'results': {
                'n_voids': results['n_voids'],
                'statistics': self._compute_statistics(results['void_catalog'])
            }
        }
        
        def json_serializer(obj):
            """Custom JSON serializer for numpy types and other non-serializable objects."""
            if isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                val = float(obj)
                if np.isnan(val) or np.isinf(val):
                    return None
                return val
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
                return str(obj)
            elif pd.isna(obj):
                return None
            elif isinstance(obj, bool):
                return bool(obj)
            elif isinstance(obj, type(None)):
                return None
            # Fallback to string representation for unknown types
            try:
                return str(obj)
            except Exception:
                return None
        
        # Recursively sanitize the data structure before serialization
        def sanitize_for_json(obj):
            """Recursively sanitize data structure for JSON serialization."""
            if isinstance(obj, dict):
                return {str(k): sanitize_for_json(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [sanitize_for_json(item) for item in obj]
            elif isinstance(obj, (np.integer, np.int8, np.int16, np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float16, np.float32, np.float64)):
                val = float(obj)
                if np.isnan(val) or np.isinf(val):
                    return None
                return val
            elif isinstance(obj, np.ndarray):
                return sanitize_for_json(obj.tolist())
            elif pd.isna(obj):
                return None
            elif isinstance(obj, bool):
                return bool(obj)
            else:
                return obj
        
        json_data_sanitized = sanitize_for_json(json_data)
        
        with open(json_path, 'w') as f:
            json.dump(json_data_sanitized, f, indent=2, default=json_serializer, ensure_ascii=False)
        
        self.log_progress(f"✓ Results saved: {json_path}")
    
    def _compute_statistics(self, void_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Compute publication-standard statistics."""
        if void_catalog is None or len(void_catalog) == 0:
            return {'n_voids': 0, 'empty': True}
        
        def safe_float(value):
            """Convert value to float, handling NaN and inf."""
            if isinstance(value, (np.integer, np.floating)):
                value = float(value)
            elif not isinstance(value, (int, float)):
                value = float(value)
            
            # Handle NaN and inf
            if np.isnan(value):
                return None
            elif np.isinf(value):
                return None
            return value
        
        stats = {
            'n_voids': len(void_catalog),
            'empty': False
        }
        
        # Radius statistics
        if 'radius_mpc' in void_catalog.columns:
            radii = void_catalog['radius_mpc'].dropna().values
            if len(radii) > 0:
                stats['radius_statistics'] = {
                    'min': safe_float(np.min(radii)),
                    'max': safe_float(np.max(radii)),
                    'mean': safe_float(np.mean(radii)),
                    'median': safe_float(np.median(radii)),
                    'std': safe_float(np.std(radii)),
                    'percentiles': {
                        'p5': safe_float(np.percentile(radii, 5)),
                        'p25': safe_float(np.percentile(radii, 25)),
                        'p75': safe_float(np.percentile(radii, 75)),
                        'p95': safe_float(np.percentile(radii, 95))
                    }
                }
        
        # Volume statistics
        if 'volume_mpc3' in void_catalog.columns:
            volumes = void_catalog['volume_mpc3'].dropna().values
            if len(volumes) > 0:
                stats['volume_statistics'] = {
                    'min': safe_float(np.min(volumes)),
                    'max': safe_float(np.max(volumes)),
                    'mean': safe_float(np.mean(volumes)),
                    'median': safe_float(np.median(volumes)),
                    'std': safe_float(np.std(volumes)),
                    'total_volume_mpc3': safe_float(np.sum(volumes))
                }
        
        # Lambda(z) statistics (H-ZOBOV specific)
        if 'lambda_z' in void_catalog.columns:
            lambda_vals = void_catalog['lambda_z'].dropna().values
            if len(lambda_vals) > 0:
                stats['lambda_statistics'] = {
                    'mean': safe_float(np.mean(lambda_vals)),
                    'median': safe_float(np.median(lambda_vals)),
                    'std': safe_float(np.std(lambda_vals)),
                    'min': safe_float(np.min(lambda_vals)),
                    'max': safe_float(np.max(lambda_vals))
                }
        
        return stats
    
    def _generate_report(self, results: Dict[str, Any], galaxy_catalog: pd.DataFrame):
        """Generate comprehensive markdown report."""
        base_name = self.parameters.get_output_filename_base(include_params=True)
        report_path = self.reports_dir / f"{base_name}_report.md"
        
        stats = self._compute_statistics(results['void_catalog'])
        
        with open(report_path, 'w') as f:
            f.write("# H-ZOBOV Void Catalog Analysis Report\n\n")
            f.write(f"**Output Name:** {self.parameters.output_name}\n\n")
            f.write(f"**Algorithm:** H-ZOBOV (ZOBOV with H-ΛCDM Lambda(z) integration)\n\n")
            f.write("---\n\n")
            
            f.write("## Parameters\n\n")
            params = self.parameters.to_dict()
            for key, value in params.items():
                # Format None values more clearly
                if value is None:
                    if key == 'significance_ratio':
                        display_value = "Not specified (parameter-free mode)"
                    elif key == 'min_void_volume':
                        display_value = "Not specified (no volume filter)"
                    elif key == 'mps_device':
                        display_value = "Auto-detect"
                    elif key == 'z_bin_size':
                        display_value = "Not specified (processing all at once)"
                    else:
                        display_value = "Not specified"
                else:
                    display_value = value
                f.write(f"- **{key}:** {display_value}\n")
            f.write("\n")
            
            # Add binning information if used
            if self.parameters.z_bin_size is not None:
                n_bins = int(np.ceil((self.parameters.z_max - self.parameters.z_min) / self.parameters.z_bin_size))
                f.write(f"**Redshift Binning:** Enabled (bin size: {self.parameters.z_bin_size}, ~{n_bins} bins)\n")
                f.write(f"*Note: Redshift binning is default for H-ZOBOV to ensure consistent Λ(z) within each bin.*\n\n")
            
            f.write("## Input Data Summary\n\n")
            f.write(f"- **Galaxies:** {len(galaxy_catalog):,}\n")
            f.write(f"- **Redshift Range:** {self.parameters.z_min} - {self.parameters.z_max}\n")
            f.write("\n")
            
            f.write("## Void Catalog Statistics\n\n")
            f.write(f"- **Total Voids:** {stats['n_voids']:,}\n\n")
            
            if 'radius_statistics' in stats:
                r_stats = stats['radius_statistics']
                f.write("### Radius Distribution\n\n")
                f.write(f"- **Mean:** {r_stats['mean']:.2f} Mpc\n")
                f.write(f"- **Median:** {r_stats['median']:.2f} Mpc\n")
                f.write(f"- **Range:** {r_stats['min']:.2f} - {r_stats['max']:.2f} Mpc\n\n")
            
            if 'lambda_statistics' in stats:
                l_stats = stats['lambda_statistics']
                f.write("### H-ΛCDM Lambda(z) Statistics\n\n")
                f.write(f"- **Mean Λ(z):** {l_stats['mean']:.2e} m⁻²\n")
                f.write(f"- **Range:** {l_stats['min']:.2e} - {l_stats['max']:.2e} m⁻²\n\n")
        
        self.log_progress(f"✓ Report generated: {report_path}")
    
    def validate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Basic validation of void catalog."""
        # Implementation for validation
        return {'valid': True, 'note': 'Validation not yet implemented'}
    
    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extended validation."""
        return {'valid': True, 'note': 'Extended validation not yet implemented'}

