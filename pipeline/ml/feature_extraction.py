"""
Feature Extraction Module
==========================

Handles feature extraction and augmentation for all cosmological modalities.
Implements physically motivated scaling and error models.
Enforces deterministic column ordering for neural network stability.
"""

import numpy as np
import pandas as pd
import torch
from typing import Dict, Any, Optional, List, Tuple
from scipy.interpolate import interp1d
import logging


class FeatureExtractor:
    """
    Handles feature extraction and augmentation for ML pipeline.
    
    Separated from main pipeline for better organization and testability.
    """
    
    def __init__(self, logger: logging.Logger, device: torch.device):
        """
        Initialize feature extractor.
        
        Parameters:
            logger: Logger instance
            device: PyTorch device
        """
        self.logger = logger
        self.device = device
    
    def extract_features_from_data(self, data: Dict[str, Any], encoder_dims: Dict[str, int]) -> Tuple[Dict[str, np.ndarray], Dict[str, List[Dict[str, Any]]]]:
        """
        Extract features from actual cosmological data and capture metadata for physics residuals.
        
        Parameters:
            data: Dictionary of loaded cosmological data
            encoder_dims: Expected dimensions for each modality
            
        Returns:
            Tuple containing:
                - Dictionary mapping modality names to feature arrays
                - Dictionary mapping modality names to list of metadata dicts (one per sample)
        """
        features = {}
        metadata = {}
        
        for modality, dim in encoder_dims.items():
            if modality not in data or data[modality] is None:
                features[modality] = np.array([])
                metadata[modality] = []
                continue
            
            modality_data = data[modality]
            
            try:
                if modality.startswith('cmb_'):
                    # CMB power spectrum data
                    if isinstance(modality_data, dict):
                        ell = modality_data.get('ell', np.array([]))
                        C_ell = modality_data.get('C_ell', np.array([]))
                        C_ell_err = modality_data.get('C_ell_err', None)
                        
                        if len(C_ell) > 0:
                            # Interpolate/resample to expected dimension
                            feats, metas = self._process_cmb_spectrum(ell, C_ell, C_ell_err, dim)
                            features[modality] = feats
                            metadata[modality] = metas
                        else:
                            features[modality] = np.array([])
                            metadata[modality] = []
                    else:
                        features[modality] = np.array([])
                        metadata[modality] = []
                
                elif modality.startswith('bao_'):
                    # BAO measurement data
                    if isinstance(modality_data, pd.DataFrame):
                        # DataFrame path not fully optimized for metadata yet, converting to dict logic
                        # Or implementing generic DF handler
                        feats, metas = self._extract_bao_features(modality_data, dim)
                        features[modality] = feats
                        metadata[modality] = metas
                    elif isinstance(modality_data, dict):
                        feats, metas = self._extract_bao_features_from_dict(modality_data, dim)
                        features[modality] = feats
                        metadata[modality] = metas
                    else:
                        features[modality] = np.array([])
                        metadata[modality] = []
                
                elif modality.startswith('void_'):
                    # Void catalog data
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        feats, metas = self._extract_void_features(modality_data, dim)
                        features[modality] = feats
                        metadata[modality] = metas
                    else:
                        features[modality] = np.array([])
                        metadata[modality] = []
                
                elif modality == 'galaxy':
                    # Galaxy catalog data
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        feats, metas = self._extract_galaxy_features(modality_data, dim)
                        features[modality] = feats
                        metadata[modality] = metas
                    else:
                        features[modality] = np.array([])
                        metadata[modality] = []
                
                elif modality == 'frb':
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        feats, metas = self._extract_frb_features(modality_data, dim)
                        features[modality] = feats
                        metadata[modality] = metas
                    else:
                        features[modality] = np.array([])
                        metadata[modality] = []
                
                elif modality == 'lyman_alpha':
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        feats, metas = self._extract_lyman_alpha_features(modality_data, dim)
                        features[modality] = feats
                        metadata[modality] = metas
                    else:
                        features[modality] = np.array([])
                        metadata[modality] = []
                
                elif modality == 'jwst':
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        feats, metas = self._extract_jwst_features(modality_data, dim)
                        features[modality] = feats
                        metadata[modality] = metas
                    else:
                        features[modality] = np.array([])
                        metadata[modality] = []
                
                elif modality.startswith('gw_'):
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        feats, metas = self._extract_gw_features(modality_data, dim)
                        features[modality] = feats
                        metadata[modality] = metas
                    else:
                        features[modality] = np.array([])
                        metadata[modality] = []
                
                else:
                    features[modality] = np.array([])
                    metadata[modality] = []
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract features for {modality}: {e}")
                features[modality] = np.array([])
                metadata[modality] = []
        
        return features, metadata

    def _extract_dataframe_features(self, df: pd.DataFrame, target_dim: int, 
                                  priority_cols: List[str] = []) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Generic extraction with deterministic column ordering and downsampling.
        Returns features and metadata list.
        """
        if df.empty:
            return np.zeros((1, target_dim)), [{}]

        # Apply deterministic downsampling to keep sizes manageable but statistically significant
        # Target ~50-100 samples per survey
        target_samples = 100
        if len(df) > target_samples:
            # Use deterministic seed based on dataframe size to ensure consistency
            seed = len(df) % 10000
            rng = np.random.RandomState(seed)
            # Stratified sampling could be added here, but random shuffle is a good baseline
            indices = rng.choice(len(df), target_samples, replace=False)
            df_subset = df.iloc[indices].copy()
        else:
            df_subset = df
            
        # Identify available priority columns
        present_priority = [c for c in priority_cols if c in df_subset.columns]
        
        # Get other numeric columns sorted alphabetically for determinism
        numeric_cols = df_subset.select_dtypes(include=[np.number]).columns.tolist()
        other_cols = sorted([c for c in numeric_cols if c not in present_priority])
        
        # Final column order
        ordered_cols = present_priority + other_cols
        
        if not ordered_cols:
            return np.zeros((len(df_subset), target_dim)), [{} for _ in range(len(df_subset))]
            
        features = df_subset[ordered_cols].values
        
        # Capture metadata for each row
        # Prioritize z, ra, dec, mass, etc.
        metadata_list = []
        for _, row in df_subset.iterrows():
            meta = {
                'z': row.get('z', row.get('redshift', row.get('Z', 0.0))),
                'ra': row.get('ra', row.get('RA', row.get('right_ascension', 0.0))),
                'dec': row.get('dec', row.get('DEC', row.get('declination', 0.0))),
                'original_data': row.to_dict()
            }
            metadata_list.append(meta)
        
        # If we have too few samples, augment by jittering
        if len(features) < 10 and len(features) > 0:
            n_needed = 50
            n_copies = max(2, n_needed // len(features))
            
            aug_features_list = [features]
            aug_meta_list = list(metadata_list)
            
            for _ in range(n_copies):
                # Add small relative noise (1%) using deterministic seed per iteration
                # Create deterministic seed based on feature sum and iteration index
                iteration_seed = int(np.abs(features.sum()) * 1000) + _
                rng = np.random.RandomState(iteration_seed % 2**32)
                noise = rng.normal(0, 0.01, size=features.shape) * (np.abs(features) + 1e-6)
                jittered = features + noise
                aug_features_list.append(jittered)
                
                # Duplicate metadata with flag
                for meta in metadata_list:
                    new_meta = meta.copy()
                    new_meta['is_augmented'] = True
                    aug_meta_list.append(new_meta)
            
            features = np.vstack(aug_features_list)
            metadata_list = aug_meta_list

        # Pad or truncate
        features = self._pad_features(features, target_dim)
            
        # Clean but DO NOT Normalize
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return features, metadata_list
        
    def _pad_features(self, features: np.ndarray, target_dim: int) -> np.ndarray:
        """Pad or truncate features to target dimension."""
        if features.shape[1] < target_dim:
            padding = np.zeros((features.shape[0], target_dim - features.shape[1]))
            return np.hstack([features, padding])
        elif features.shape[1] > target_dim:
            return features[:, :target_dim]
        return features

    def _extract_void_features(self, void_df: pd.DataFrame, target_dim: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extract void features enforcing [RA, Dec, Z, ...] order.
        """
        # Common names for coordinates
        ra_names = ['ra', 'RA', 'Ra', 'right_ascension', 'ra_deg']
        dec_names = ['dec', 'DEC', 'Dec', 'declination', 'dec_deg']
        z_names = ['z', 'Z', 'redshift', 'Redshift']
        
        # Find which names exist
        ra_col = next((c for c in ra_names if c in void_df.columns), None)
        dec_col = next((c for c in dec_names if c in void_df.columns), None)
        z_col = next((c for c in z_names if c in void_df.columns), None)
        
        priority_cols = []
        if ra_col and dec_col and z_col:
            priority_cols = [ra_col, dec_col, z_col]
            
        return self._extract_dataframe_features(void_df, target_dim, priority_cols)

    def _extract_galaxy_features(self, galaxy_df: pd.DataFrame, target_dim: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extract galaxy features with specific astrophysical columns.
        """
        # Map columns to standard set
        col_map = {
            'ra': ['ra', 'RA', 'right_ascension'],
            'dec': ['dec', 'DEC', 'declination'],
            'z': ['z', 'redshift', 'Z'],
            'mass': ['mass', 'stellar_mass', 'logMstar'],
            'sfr': ['sfr', 'SFR', 'logSFR'],
            'metallicity': ['metallicity', 'Z_gas', 'OH_gas'],
            'magnitude': ['mag', 'magnitude', 'petroMag_r']
        }
        
        # Identify present columns
        present_map = {}
        for std, variants in col_map.items():
            for v in variants:
                if v in galaxy_df.columns:
                    present_map[std] = v
                    break
        
        # Construct priority list
        priority_cols = [present_map.get(c) for c in ['ra', 'dec', 'z'] if c in present_map]
        
        # Use generic extractor but prioritize physics columns
        return self._extract_dataframe_features(galaxy_df, target_dim, priority_cols)

    def _extract_bao_features(self, bao_df: pd.DataFrame, target_dim: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """Extract features from BAO DataFrame."""
        return self._extract_dataframe_features(bao_df, target_dim, priority_cols=['z', 'redshift'])

    def _extract_frb_features(self, frb_df: pd.DataFrame, target_dim: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        return self._extract_dataframe_features(frb_df, target_dim)

    def _extract_lyman_alpha_features(self, lyman_df: pd.DataFrame, target_dim: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        return self._extract_dataframe_features(lyman_df, target_dim)

    def _extract_jwst_features(self, jwst_df: pd.DataFrame, target_dim: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        return self._extract_dataframe_features(jwst_df, target_dim)

    def _extract_gw_features(self, gw_df: pd.DataFrame, target_dim: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        return self._extract_dataframe_features(gw_df, target_dim)

    def _augment_single_summary(self, row: np.ndarray, target_dim: int, n_aug: int = 100) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Augment single summary statistic into N samples.
        row: [z, val, err]
        """
        z, val, err = row[0], row[1], row[2]
        
        # Generate N samples
        rng = np.random.RandomState(42 + int(z*100))
        
        # Jitter redshift slightly (observational uncertainty)
        z_samples = rng.normal(z, 0.001, n_aug)
        
        # Sample values from Gaussian
        val_samples = rng.normal(val, err if err > 0 else abs(val)*0.1, n_aug)
        
        # Constant error column
        err_samples = np.full(n_aug, err)
        
        # Stack columns
        features = np.column_stack([z_samples, val_samples, err_samples])
        
        # Create metadata
        metadata_list = []
        for i in range(n_aug):
            metadata_list.append({
                'z': z_samples[i],
                'value': val_samples[i],
                'error': err,
                'is_augmented': True,
                'source_z': z,
                'source_val': val
            })
        
        return self._pad_features(features, target_dim), metadata_list

    def _extract_bao_features_from_dict(self, bao_dict: Dict[str, Any], target_dim: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Extract features from BAO dict format (consensus measurements).
        Returns (n_bins, dim) array and metadata list.
        """
        # Case 1: 'measurements' list (standard from loader)
        if 'measurements' in bao_dict:
            rows = []
            metadata_list = []
            for m in bao_dict['measurements']:
                # Extract z, value, error
                # Handle potential variations in keys
                z = m.get('z', 0.0)
                val = m.get('value', 0.0)
                err = m.get('error', 0.0)
                rows.append([z, val, err])
                metadata_list.append({
                    'z': z,
                    'value': val,
                    'error': err,
                    'type': 'BAO_D_M' # Assuming transverse unless specified
                })
            
            features = np.array(rows)
            
            if len(features) == 0:
                return np.zeros((1, target_dim)), [{}]
                
            # If few points (e.g. < 10), augment each measurement to create a statistical distribution
            if len(features) < 10:
                all_aug_feats = []
                all_aug_meta = []
                # Target ~50-100 total samples, split across the available measurements
                n_aug_per_row = max(10, 100 // len(features))
                
                for i in range(len(features)):
                    # _augment_single_summary returns padded features, we need unpadded for vstack if we want to be careful, 
                    # but it pads to target_dim which is consistent.
                    aug_feats, aug_meta = self._augment_single_summary(features[i], target_dim, n_aug=n_aug_per_row)
                    all_aug_feats.append(aug_feats)
                    all_aug_meta.extend(aug_meta)
                
                # Also include the original raw measurements
                raw_padded = self._pad_features(features, target_dim)
                all_aug_feats.append(raw_padded)
                all_aug_meta.extend(metadata_list)
                
                return np.vstack(all_aug_feats), all_aug_meta
                
            return self._pad_features(features, target_dim), metadata_list

        # Case 2: Direct arrays (legacy or specialized format)
        arrays = []
        for key in ['z', 'D_M_over_r_d', 'D_H_over_r_d', 'error']:
            if key in bao_dict and isinstance(bao_dict[key], np.ndarray):
                arrays.append(bao_dict[key])
                
        if arrays:
            # Assuming first array is z
            features = np.column_stack(arrays)
            # Create basic metadata
            metadata_list = []
            for i in range(len(features)):
                row = features[i]
                metadata_list.append({
                    'z': row[0] if len(row) > 0 else 0.0,
                    'value': row[1] if len(row) > 1 else 0.0,
                    'error': row[-1] if len(row) > 2 else 0.0
                })
            return self._pad_features(features, target_dim), metadata_list
            
        return np.zeros((1, target_dim)), [{}]

    def _process_cmb_spectrum(self, ell: np.ndarray, C_ell: np.ndarray, 
                             C_ell_err: Optional[np.ndarray], target_dim: int) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
        """
        Process CMB power spectrum by chunking into low/mid/high-ell segments.
        Treats each chunk as an independent example.
        Returns (n_samples, dim) and metadata list.
        """
        if len(C_ell) == 0:
            return np.zeros((1, target_dim)), [{}]
        
        C_ell = np.nan_to_num(C_ell, nan=0.0, posinf=1e10, neginf=-1e10)
        ell = np.nan_to_num(ell, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if len(ell) == 0:
            return np.zeros((1, target_dim)), [{}]

        # Define chunks (Low, Mid, High)
        # Overlapping windows to increase sample count slightly and capture boundaries
        chunks = [
            (2, 500),      # Large scale (Low-l)
            (400, 1500),   # Intermediate (Mid-l)
            (1200, 2500),  # Small scale (High-l)
            (2000, 3000)   # Very small scale / Damping tail
        ]
        
        samples = []
        metadata_list = []
        
        # Pre-calculate error or variance
        if C_ell_err is None:
            # Cosmic variance approximation if no error provided
            with np.errstate(divide='ignore', invalid='ignore'):
                sigma_cv = np.sqrt(2.0 / (2.0 * ell + 1.0)) * C_ell
            C_ell_err = np.nan_to_num(sigma_cv, nan=1e-9)

        for l_min, l_max in chunks:
            mask = (ell >= l_min) & (ell < l_max)
            if not np.any(mask):
                continue
                
            chunk_ell = ell[mask]
            chunk_Cell = C_ell[mask]
            chunk_err = C_ell_err[mask]
            
            if len(chunk_ell) < 10:
                continue
                
            # Scale logic (same as before)
            ell_factor = chunk_ell * (chunk_ell + 1) / (2 * np.pi)
            D_ell = chunk_Cell * ell_factor
            
            # Log-modulus transform
            D_ell_scaled_raw = D_ell * 1e12
            D_ell_scaled = np.sign(D_ell_scaled_raw) * np.log1p(np.abs(D_ell_scaled_raw))
        
            # Interpolate to target_dim
            ell_interp = np.linspace(chunk_ell.min(), chunk_ell.max(), target_dim)
            f_interp = interp1d(chunk_ell, D_ell_scaled, kind='cubic', bounds_error=False, fill_value='extrapolate')
            D_ell_interp = f_interp(ell_interp)
        
            # Interpolate error for noise addition
            with np.errstate(divide='ignore', invalid='ignore'):
                rel_err = np.abs(chunk_err) / (np.abs(chunk_Cell) + 1e-20)
            rel_err = np.clip(rel_err, 0.01, 1.0) # Clip relative error
            
            f_err = interp1d(chunk_ell, rel_err, kind='linear', bounds_error=False, fill_value='extrapolate')
            rel_err_interp = f_err(ell_interp)
            
            # Create a few realizations for this chunk to robustify
            # Increased to 10 to ensure we get >10 samples even if only 1-2 chunks are valid
            n_realizations = 10
            for r_idx in range(n_realizations):
                # Add noise based on error using deterministic seed
                # Seed based on chunk properties and realization index
                chunk_seed = int(np.abs(chunk_Cell.sum()) * 1000) + int(l_min) + r_idx
                rng = np.random.RandomState(chunk_seed % 2**32)
                noise = rng.normal(0, rel_err_interp) * np.abs(D_ell_interp)
                sample = D_ell_interp + noise
                sample = np.nan_to_num(sample, nan=0.0, posinf=10.0, neginf=-10.0)
                samples.append(sample)
                
                # Metadata for physics calculation
                # We store the mean ell and mean C_ell (scaled) for this chunk
                # as a proxy for "location" in frequency space
                metadata_list.append({
                    'l_min': float(l_min),
                    'l_max': float(l_max),
                    'l_mean': float(np.mean(chunk_ell)),
                    'C_ell_mean': float(np.mean(chunk_Cell)),
                    'D_ell_mean': float(np.mean(D_ell)),
                    'error_mean': float(np.mean(chunk_err)),
                    'realization': r_idx,
                    'type': 'CMB_Power_Spectrum'
                })

        if not samples:
            # Fallback if no chunks worked (e.g. sparse data)
            return np.zeros((1, target_dim)), [{}]
        
        return np.array(samples), metadata_list

    def augment_data(self, features: np.ndarray, modality: str) -> np.ndarray:
        """
        Apply physics-informed data augmentation.
        """
        augmented = features.copy()
        
        # Use deterministic seed based on feature content
        seed_val = int(np.abs(features.sum()) * 1000) % 2**32
        rng = np.random.RandomState(seed_val)
        
        if modality.startswith('cmb_'):
            # CMB noise is ell-dependent, but here we have feature vectors
            # Add generic noise
            noise = rng.normal(0, 0.1, size=augmented.shape)
            augmented = augmented + noise
        elif modality.startswith('bao_'):
            noise = rng.normal(0, 0.01, size=augmented.shape)
            augmented = augmented * (1 + noise)
        elif modality.startswith('void_'):
            noise = rng.normal(0, 0.05, size=augmented.shape)
            augmented = augmented + noise
        else:
            noise = rng.normal(0, 0.02, size=augmented.shape)
            augmented = augmented + noise
            
        return augmented
