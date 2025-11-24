"""
Galaxy Feature Extractor
=======================

Extracts features from galaxy catalogs for ML training.
Includes photometric, morphological, and clustering features
that may reveal H-ΛCDM signatures in galaxy formation.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats
from sklearn.neighbors import NearestNeighbors


class GalaxyFeatureExtractor:
    """
    Extract features from galaxy catalogs for ML analysis.

    Focuses on galaxy properties, colors, morphologies, and
    clustering patterns that might reveal H-ΛCDM signatures.
    """

    def __init__(self):
        """Initialize galaxy feature extractor."""
        pass

    def extract_features(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract comprehensive features from galaxy catalog.

        Parameters:
            galaxy_catalog: DataFrame with galaxy properties

        Returns:
            dict: Extracted features
        """
        if galaxy_catalog.empty:
            return {'error': 'Empty galaxy catalog'}

        features = {}

        # Basic catalog statistics
        features.update(self._extract_basic_statistics(galaxy_catalog))

        # Photometric features
        features.update(self._extract_photometric_features(galaxy_catalog))

        # Morphological features
        features.update(self._extract_morphological_features(galaxy_catalog))

        # Color features
        features.update(self._extract_color_features(galaxy_catalog))

        # Spatial clustering features
        features.update(self._extract_clustering_features(galaxy_catalog))

        # Redshift evolution features
        features.update(self._extract_redshift_features(galaxy_catalog))

        return features

    def _extract_basic_statistics(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic galaxy catalog statistics."""
        n_galaxies = len(galaxy_catalog)

        features = {'n_galaxies': n_galaxies}

        # Magnitude statistics
        if 'r_mag' in galaxy_catalog.columns:
            r_mags = galaxy_catalog['r_mag'].values
            features.update({
                'mean_r_mag': float(np.mean(r_mags)),
                'r_mag_std': float(np.std(r_mags)),
                'median_r_mag': float(np.median(r_mags)),
                'mag_range': float(np.max(r_mags) - np.min(r_mags))
            })

        # Redshift statistics
        if 'z' in galaxy_catalog.columns:
            redshifts = galaxy_catalog['z'].values
            features.update({
                'mean_redshift': float(np.mean(redshifts)),
                'redshift_std': float(np.std(redshifts)),
                'redshift_range': float(np.max(redshifts) - np.min(redshifts))
            })

        return features

    def _extract_photometric_features(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract photometric features."""
        features = {}

        # Magnitude distributions
        mag_cols = ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag']
        available_mags = [col for col in mag_cols if col in galaxy_catalog.columns]

        if available_mags:
            for mag_col in available_mags:
                mags = galaxy_catalog[mag_col].values
                features.update({
                    f'{mag_col}_mean': float(np.mean(mags)),
                    f'{mag_col}_std': float(np.std(mags)),
                    f'{mag_col}_skewness': float(stats.skew(mags))
                })

        return features

    def _extract_morphological_features(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract morphological features."""
        features = {}

        # Concentration index
        if 'concentration' in galaxy_catalog.columns:
            concentrations = galaxy_catalog['concentration'].values
            features.update({
                'mean_concentration': float(np.mean(concentrations)),
                'concentration_std': float(np.std(concentrations)),
                'early_type_fraction': float(np.mean(concentrations > 3.0)),  # Concentrated = early type
                'late_type_fraction': float(np.mean(concentrations < 2.5))   # Extended = late type
            })

        # Size features
        size_cols = ['petrosian_radius', 'half_light_radius']
        for size_col in size_cols:
            if size_col in galaxy_catalog.columns:
                sizes = galaxy_catalog[size_col].values
                features.update({
                    f'{size_col}_mean': float(np.mean(sizes)),
                    f'{size_col}_std': float(np.std(sizes)),
                    f'{size_col}_median': float(np.median(sizes))
                })

        return features

    def _extract_color_features(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract color features."""
        features = {}

        # Color indices
        color_cols = ['u_minus_r', 'g_minus_r', 'r_minus_i']
        for color_col in color_cols:
            if color_col in galaxy_catalog.columns:
                colors = galaxy_catalog[color_col].values
                features.update({
                    f'{color_col}_mean': float(np.mean(colors)),
                    f'{color_col}_std': float(np.std(colors)),
                    f'{color_col}_median': float(np.median(colors)),
                    f'{color_col}_skewness': float(stats.skew(colors))
                })

                # Color bimodality (red vs blue sequence)
                if color_col == 'u_minus_r':
                    red_cutoff = 2.2  # Rough red/blue separator
                    red_fraction = float(np.mean(colors > red_cutoff))
                    features['red_fraction'] = red_fraction
                    features['blue_fraction'] = 1 - red_fraction

        return features

    def _extract_clustering_features(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract galaxy clustering features."""
        features = {}

        # Spatial clustering analysis
        if all(col in galaxy_catalog.columns for col in ['ra', 'dec', 'z']):
            ra = galaxy_catalog['ra'].values
            dec = galaxy_catalog['dec'].values
            z = galaxy_catalog['z'].values

            # Convert to Cartesian (simplified for small volumes)
            ra_rad = np.radians(ra)
            dec_rad = np.radians(dec)

            # Simplified 3D positions
            x = z * np.cos(dec_rad) * np.cos(ra_rad)
            y = z * np.cos(dec_rad) * np.sin(ra_rad)
            z_coord = z * np.sin(dec_rad)

            positions = np.column_stack([x, y, z_coord])

            # Nearest neighbor statistics
            if len(positions) > 1:
                n_neighbors = min(10, len(positions) - 1)
                nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(positions)
                distances, indices = nbrs.kneighbors(positions)

                # Mean distance to k-th nearest neighbor
                for k in [1, 3, 5]:
                    if k < n_neighbors:
                        mean_dist_k = np.mean(distances[:, k])
                        std_dist_k = np.std(distances[:, k])
                        features.update({
                            f'mean_dist_k{k}': float(mean_dist_k),
                            f'std_dist_k{k}': float(std_dist_k)
                        })

        return features

    def _extract_redshift_features(self, galaxy_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract redshift evolution features."""
        features = {}

        if 'z' in galaxy_catalog.columns:
            redshifts = galaxy_catalog['z'].values

            # Redshift distribution
            features.update({
                'redshift_bins': np.histogram(redshifts, bins=10)[0].tolist(),
                'redshift_percentiles': np.percentile(redshifts, [25, 50, 75]).tolist()
            })

            # Luminosity evolution (if magnitudes available)
            if 'r_mag' in galaxy_catalog.columns:
                r_mags = galaxy_catalog['r_mag'].values

                # Binned by redshift
                z_bins = np.linspace(np.min(redshifts), np.max(redshifts), 6)
                mag_evolution = []

                for i in range(len(z_bins) - 1):
                    mask = (redshifts >= z_bins[i]) & (redshifts < z_bins[i+1])
                    if np.sum(mask) > 0:
                        mean_mag = np.mean(r_mags[mask])
                        mag_evolution.append(mean_mag)

                features['magnitude_evolution'] = mag_evolution

        return features

    def augment_galaxy_data(self, galaxy_catalog: pd.DataFrame,
                           augmentation_type: str = 'noise') -> pd.DataFrame:
        """
        Apply data augmentation for contrastive learning.

        Parameters:
            galaxy_catalog: Original galaxy catalog
            augmentation_type: Type of augmentation

        Returns:
            pd.DataFrame: Augmented galaxy catalog
        """
        augmented = galaxy_catalog.copy()

        if augmentation_type == 'photometric_noise':
            # Add noise to magnitudes
            mag_cols = ['u_mag', 'g_mag', 'r_mag', 'i_mag', 'z_mag']
            for mag_col in mag_cols:
                if mag_col in augmented.columns:
                    noise_level = 0.05  # 5% photometric error
                    augmented[mag_col] += np.random.normal(0, noise_level, len(augmented))

        elif augmentation_type == 'redshift_uncertainty':
            # Add redshift errors
            if 'z' in augmented.columns:
                z_err = augmented['z'] * 0.05  # 5% redshift error
                augmented['z'] += np.random.normal(0, z_err)

        elif augmentation_type == 'morphology_variation':
            # Vary morphological parameters
            morph_cols = ['concentration', 'petrosian_radius']
            for morph_col in morph_cols:
                if morph_col in augmented.columns:
                    variation = np.random.uniform(0.9, 1.1, len(augmented))
                    augmented[morph_col] *= variation

        return augmented
