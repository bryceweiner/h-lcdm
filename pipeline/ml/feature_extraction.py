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
from typing import Dict, Any, Optional, List
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
    
    def extract_features_from_data(self, data: Dict[str, Any], encoder_dims: Dict[str, int]) -> Dict[str, np.ndarray]:
        """
        Extract features from actual cosmological data.
        
        Parameters:
            data: Dictionary of loaded cosmological data
            encoder_dims: Expected dimensions for each modality
            
        Returns:
            Dictionary mapping modality names to feature arrays
        """
        features = {}
        
        for modality, dim in encoder_dims.items():
            if modality not in data or data[modality] is None:
                features[modality] = np.array([])
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
                            features[modality] = self._process_cmb_spectrum(ell, C_ell, C_ell_err, dim)
                        else:
                            features[modality] = np.array([])
                    else:
                        features[modality] = np.array([])
                
                elif modality.startswith('bao_'):
                    # BAO measurement data
                    if isinstance(modality_data, pd.DataFrame):
                        features[modality] = self._extract_bao_features(modality_data, dim)
                    elif isinstance(modality_data, dict):
                        features[modality] = self._extract_bao_features_from_dict(modality_data, dim)
                    else:
                        features[modality] = np.array([])
                
                elif modality.startswith('void_'):
                    # Void catalog data
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        features[modality] = self._extract_void_features(modality_data, dim)
                    else:
                        features[modality] = np.array([])
                
                elif modality == 'galaxy':
                    # Galaxy catalog data
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        features[modality] = self._extract_galaxy_features(modality_data, dim)
                    else:
                        features[modality] = np.array([])
                
                elif modality == 'frb':
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        features[modality] = self._extract_frb_features(modality_data, dim)
                    else:
                        features[modality] = np.array([])
                
                elif modality == 'lyman_alpha':
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        features[modality] = self._extract_lyman_alpha_features(modality_data, dim)
                    else:
                        features[modality] = np.array([])
                
                elif modality == 'jwst':
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        features[modality] = self._extract_jwst_features(modality_data, dim)
                    else:
                        features[modality] = np.array([])
                
                elif modality.startswith('gw_'):
                    if isinstance(modality_data, pd.DataFrame) and not modality_data.empty:
                        features[modality] = self._extract_gw_features(modality_data, dim)
                    else:
                        features[modality] = np.array([])
                
                else:
                    features[modality] = np.array([])
                    
            except Exception as e:
                self.logger.warning(f"Failed to extract features for {modality}: {e}")
                features[modality] = np.array([])
        
        return features

    def _extract_dataframe_features(self, df: pd.DataFrame, target_dim: int, 
                                  priority_cols: List[str] = []) -> np.ndarray:
        """
        Generic extraction with deterministic column ordering.
        
        Note: Does NOT perform Z-score normalization to avoid batch statistics leakage.
        Normalization is handled by the neural network layers (BatchNorm) or downstream scalers.
        """
        if df.empty:
            return np.zeros((1, target_dim))
            
        # Identify available priority columns
        present_priority = [c for c in priority_cols if c in df.columns]
        
        # Get other numeric columns sorted alphabetically for determinism
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        other_cols = sorted([c for c in numeric_cols if c not in present_priority])
        
        # Final column order
        ordered_cols = present_priority + other_cols
        
        if not ordered_cols:
            return np.zeros((len(df), target_dim))
            
        features = df[ordered_cols].values
        
        # Pad or truncate
        if features.shape[1] < target_dim:
            padding = np.zeros((features.shape[0], target_dim - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > target_dim:
            features = features[:, :target_dim]
            
        # Clean but DO NOT Normalize
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
        
        return features

    def _extract_void_features(self, void_df: pd.DataFrame, target_dim: int) -> np.ndarray:
        """
        Extract void features enforcing [RA, Dec, Z, ...] order.
        """
        # Common names for coordinates
        ra_names = ['ra', 'RA', 'Ra', 'right_ascension']
        dec_names = ['dec', 'DEC', 'Dec', 'declination']
        z_names = ['z', 'Z', 'redshift', 'Redshift']
        
        # Find which names exist
        ra_col = next((c for c in ra_names if c in void_df.columns), None)
        dec_col = next((c for c in dec_names if c in void_df.columns), None)
        z_col = next((c for c in z_names if c in void_df.columns), None)
        
        priority_cols = []
        if ra_col and dec_col and z_col:
            priority_cols = [ra_col, dec_col, z_col]
            
        return self._extract_dataframe_features(void_df, target_dim, priority_cols)

    def _extract_galaxy_features(self, galaxy_df: pd.DataFrame, target_dim: int) -> np.ndarray:
        """
        Extract galaxy features with strict grouping for GalaxyEncoder compatibility.
        """
        if galaxy_df.empty:
            return np.zeros((1, target_dim))

        all_cols = galaxy_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Define keyword matchers
        photo_keys = ['mag', 'color', 'flux', '_u', '_g', '_r', '_i', '_z', 'band']
        morph_keys = ['size', 'radius', 'ellip', 'conc', 'petro', 'deV', 'exp']
        spec_keys = ['z', 'redshift', 'spec', 'line', 'alpha', 'oiii', 'h_beta']
        
        # Categorize columns
        photo_cols = []
        morph_cols = []
        spec_cols = []
        cluster_cols = []
        
        used_cols = set()
        
        # 1. Spectroscopic (Prioritize Z)
        for col in all_cols:
            if any(k in col.lower() for k in spec_keys):
                spec_cols.append(col)
                used_cols.add(col)
        spec_cols.sort()
        
        # 2. Morphological
        for col in all_cols:
            if col in used_cols: continue
            if any(k in col.lower() for k in morph_keys):
                morph_cols.append(col)
                used_cols.add(col)
        morph_cols.sort()
        
        # 3. Photometric
        for col in all_cols:
            if col in used_cols: continue
            if any(k in col.lower() for k in photo_keys):
                photo_cols.append(col)
                used_cols.add(col)
        photo_cols.sort()
        
        # 4. Remaining (Clustering/Environment)
        for col in all_cols:
            if col not in used_cols:
                cluster_cols.append(col)
        cluster_cols.sort()
        
        # Construct ordered list
        final_cols = []
        
        # Fill Photometric (pad or truncate to 10)
        final_cols.extend(photo_cols[:10])
        final_cols.extend([None] * (10 - len(photo_cols[:10])))
        
        # Fill Morphological (pad or truncate to 5)
        final_cols.extend(morph_cols[:5])
        final_cols.extend([None] * (5 - len(morph_cols[:5])))
        
        # Fill Spectroscopic (pad or truncate to 10)
        final_cols.extend(spec_cols[:10])
        final_cols.extend([None] * (10 - len(spec_cols[:10])))
        
        # Fill Clustering (rest)
        final_cols.extend(cluster_cols)
        
        # Extract data
        n_samples = len(galaxy_df)
        features = np.zeros((n_samples, target_dim))
        
        current_idx = 0
        for col in final_cols:
            if current_idx >= target_dim:
                break
                
            if col is not None and col in galaxy_df.columns:
                features[:, current_idx] = galaxy_df[col].values
            # Else leave as 0.0 (padding)
            
            current_idx += 1
            
        # Clean but DO NOT Normalize
        features = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            
        return features

    def _extract_bao_features(self, bao_df: pd.DataFrame, target_dim: int) -> np.ndarray:
        """Extract features from BAO DataFrame."""
        return self._extract_dataframe_features(bao_df, target_dim, priority_cols=['z', 'redshift'])

    def _extract_frb_features(self, frb_df: pd.DataFrame, target_dim: int) -> np.ndarray:
        return self._extract_dataframe_features(frb_df, target_dim)

    def _extract_lyman_alpha_features(self, lyman_df: pd.DataFrame, target_dim: int) -> np.ndarray:
        return self._extract_dataframe_features(lyman_df, target_dim)

    def _extract_jwst_features(self, jwst_df: pd.DataFrame, target_dim: int) -> np.ndarray:
        return self._extract_dataframe_features(jwst_df, target_dim)

    def _extract_gw_features(self, gw_df: pd.DataFrame, target_dim: int) -> np.ndarray:
        return self._extract_dataframe_features(gw_df, target_dim)

    def _extract_bao_features_from_dict(self, bao_dict: Dict[str, Any], target_dim: int) -> np.ndarray:
        """Extract features from BAO dict format."""
        # Try to extract arrays from dict
        arrays = []
        for key in ['z', 'D_M_over_r_d', 'D_H_over_r_d', 'error']:
            if key in bao_dict and isinstance(bao_dict[key], np.ndarray):
                arrays.append(bao_dict[key])
        
        if len(arrays) == 0:
            return np.zeros((1, target_dim))
        
        features = np.column_stack(arrays)
        
        # Pad or truncate to target dimension
        if features.shape[1] < target_dim:
            padding = np.zeros((features.shape[0], target_dim - features.shape[1]))
            features = np.hstack([features, padding])
        elif features.shape[1] > target_dim:
            features = features[:, :target_dim]
        
        return features

    def clean_extracted_features(self, extracted_features: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Clean extracted features by removing NaN/inf values."""
        cleaned = {}
        for modality, features in extracted_features.items():
            if len(features) == 0:
                cleaned[modality] = np.array([])
                continue
            
            if len(features.shape) == 2:
                valid_mask = ~(np.isnan(features).any(axis=1) | np.isinf(features).any(axis=1))
                features_clean = features[valid_mask]
                features_clean = np.nan_to_num(features_clean, nan=0.0, posinf=1e10, neginf=-1e10)
            else:
                features_clean = np.nan_to_num(features, nan=0.0, posinf=1e10, neginf=-1e10)
            
            if len(features_clean) < 10:
                if len(features_clean) > 0:
                    n_repeats = (10 // len(features_clean)) + 1
                    features_clean = np.tile(features_clean, (n_repeats, 1))[:10]
                else:
                    self.logger.warning(f"Modality {modality} has no valid features after cleaning, skipping")
                    cleaned[modality] = np.array([])
                    continue
            
            cleaned[modality] = features_clean
        return cleaned

    def _process_cmb_spectrum(self, ell: np.ndarray, C_ell: np.ndarray, 
                             C_ell_err: Optional[np.ndarray], target_dim: int) -> np.ndarray:
        """
        Process CMB power spectrum to target dimension using physical scaling.
        """
        if len(C_ell) == 0:
            return np.zeros((1, target_dim))
        
        C_ell = np.nan_to_num(C_ell, nan=0.0, posinf=1e10, neginf=-1e10)
        ell = np.nan_to_num(ell, nan=0.0, posinf=1e10, neginf=-1e10)
        
        if len(ell) == 0 or len(C_ell) == 0:
            return np.zeros((100, target_dim))
        
        if ell.min() == ell.max():
            ell = np.linspace(2, 2500, len(ell))
        
        ell_factor = ell * (ell + 1) / (2 * np.pi)
        ell_factor[ell < 2] = 0
        D_ell = C_ell * ell_factor
        
        # Ensure D_ell is non-negative for log1p
        # Some TE/EE spectra can be negative (correlation), but log1p requires positive input
        # For power spectra that can be negative, we shift or take absolute, or use sinh scaling
        # Standard practice for C_ell is usually D_ell * ell(ell+1)/2pi is positive for TT
        # But for TE it can be negative. 
        
        # If we detect significant negative values (more than noise), we might be dealing with TE
        # or just noise.
        
        # Safe log-modulus transform: sign(x) * log(1 + |x|)
        # This preserves sign while compressing dynamic range
        D_ell_scaled_raw = D_ell * 1e12
        D_ell_scaled = np.sign(D_ell_scaled_raw) * np.log1p(np.abs(D_ell_scaled_raw))
        
        ell_interp = np.logspace(np.log10(max(2, ell.min())), np.log10(ell.max()), target_dim)
        f_interp = interp1d(ell, D_ell_scaled, kind='cubic', bounds_error=False, fill_value='extrapolate')
        D_ell_interp = f_interp(ell_interp)
        D_ell_interp = np.nan_to_num(D_ell_interp, nan=0.0, posinf=10.0, neginf=0.0)
        
        C_ell_err_interp = None
        if C_ell_err is not None and len(C_ell_err) > 0:
             D_ell_err = C_ell_err * ell_factor
             # Use absolute fractional error for interpolation to avoid negative scales
             with np.errstate(divide='ignore', invalid='ignore'):
                # Calculate fractional error relative to envelope
                D_ell_frac_err = np.abs(D_ell_err) / (np.abs(D_ell) + 1e-20)
             
             D_ell_frac_err = np.nan_to_num(D_ell_frac_err, nan=0.1, posinf=1.0, neginf=0.1)
             f_err_interp = interp1d(ell, D_ell_frac_err, kind='linear', bounds_error=False, fill_value='extrapolate')
             C_ell_err_interp = f_err_interp(ell_interp)
             # Ensure interpolated error scale is strictly positive
             C_ell_err_interp = np.abs(C_ell_err_interp) + 1e-9
        
        n_samples = max(100, len(C_ell) // 10)
        samples = []
        
        for _ in range(n_samples):
            if C_ell_err_interp is not None:
                # scale must be non-negative
                noise = np.random.normal(0, C_ell_err_interp)
            else:
                cv_variance = np.sqrt(2.0 / (2.0 * ell_interp + 1.0))
                noise = np.random.normal(0, cv_variance)
                
            sample = D_ell_interp + noise
            sample = np.nan_to_num(sample, nan=0.0, posinf=10.0, neginf=0.0)
            samples.append(sample)
        
        return np.array(samples)

    def augment_data(self, features: np.ndarray, modality: str) -> np.ndarray:
        """
        Apply physics-informed data augmentation.
        """
        augmented = features.copy()
        batch_size, dim = augmented.shape
        
        if modality.startswith('cmb_'):
            ell = np.logspace(np.log10(2), np.log10(2500), dim)
            sigma_cv = np.sqrt(2.0 / (2.0 * ell + 1.0))
            noise = np.random.normal(0, 1, size=augmented.shape) * sigma_cv
            augmented = augmented + noise
        elif modality.startswith('bao_'):
            noise = np.random.normal(0, 0.01, size=augmented.shape)
            augmented = augmented * (1 + noise)
        elif modality.startswith('void_'):
            noise = np.random.normal(0, 0.05, size=augmented.shape)
            augmented = augmented + noise
        else:
            noise_scale = 0.05
            noise = np.random.normal(0, noise_scale, size=augmented.shape)
            augmented = augmented + noise
        
        dropout_rate = 0.1
        mask = np.random.random(augmented.shape) > dropout_rate
        augmented = augmented * mask
        
        augmented = np.nan_to_num(augmented, nan=0.0, posinf=1e10, neginf=-1e10)
        return augmented
