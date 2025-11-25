"""
Void Feature Extractor
=====================

Extracts features from cosmic void catalogs for ML training.
Includes morphological, topological, and alignment features
that may reveal E8×E8 heterotic signatures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import NearestNeighbors


class VoidFeatureExtractor:
    """
    Extract features from cosmic void catalogs for ML analysis.

    Focuses on void properties, alignments, and network topology
    that might reveal E8×E8 heterotic structure signatures.
    """

    def __init__(self):
        """Initialize void feature extractor."""
        pass

    def extract_features(self, void_catalog: pd.DataFrame) -> Dict[str, Any]:
        """
        Extract comprehensive features from void catalog.

        Parameters:
            void_catalog: DataFrame with void properties

        Returns:
            dict: Extracted features
        """
        if void_catalog.empty:
            return {'error': 'Empty void catalog'}

        features = {}

        # Basic void statistics
        features.update(self._extract_basic_statistics(void_catalog))

        # Morphological features
        features.update(self._extract_morphological_features(void_catalog))

        # Spatial distribution features
        features.update(self._extract_spatial_features(void_catalog))

        # Network topology features
        features.update(self._extract_network_features(void_catalog))

        # Clustering and correlation features
        features.update(self._extract_clustering_features(void_catalog))

        # Survey-specific features
        features.update(self._extract_survey_features(void_catalog))

        return features

    def _extract_basic_statistics(self, void_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic void catalog statistics."""
        n_voids = len(void_catalog)

        # Size statistics (assuming radius column exists)
        if 'radius_mpc' in void_catalog.columns:
            radii = void_catalog['radius_mpc'].values
            features = {
                'n_voids': n_voids,
                'mean_radius': float(np.mean(radii)),
                'std_radius': float(np.std(radii)),
                'median_radius': float(np.median(radii)),
                'min_radius': float(np.min(radii)),
                'max_radius': float(np.max(radii)),
                'radius_skewness': float(stats.skew(radii)),
                'radius_kurtosis': float(stats.kurtosis(radii))
            }
        else:
            features = {'n_voids': n_voids}

        # Volume statistics
        if 'volume_mpc3' in void_catalog.columns:
            volumes = void_catalog['volume_mpc3'].values
            features.update({
                'total_volume': float(np.sum(volumes)),
                'mean_volume': float(np.mean(volumes)),
                'volume_std': float(np.std(volumes)),
                'volume_skewness': float(stats.skew(volumes))
            })

        # Density statistics
        if 'central_density' in void_catalog.columns:
            densities = void_catalog['central_density'].values
            features.update({
                'mean_density': float(np.mean(densities)),
                'density_std': float(np.std(densities)),
                'density_range': float(np.max(densities) - np.min(densities))
            })

        return features

    def _extract_morphological_features(self, void_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract void morphological features."""
        features = {}

        # Ellipticity/asphericity
        if 'ellipticity' in void_catalog.columns:
            ellipticities = void_catalog['ellipticity'].values
            features.update({
                'mean_ellipticity': float(np.mean(ellipticities)),
                'ellipticity_std': float(np.std(ellipticities)),
                'spherical_fraction': float(np.mean(ellipticities < 0.3))  # Roughly spherical
            })

        if 'asphericity' in void_catalog.columns:
            asphericities = void_catalog['asphericity'].values
            features.update({
                'mean_asphericity': float(np.mean(asphericities)),
                'asphericity_std': float(np.std(asphericities))
            })

        # Shape parameters
        shape_cols = ['radius_los_mpc', 'radius_transverse_mpc']
        if all(col in void_catalog.columns for col in shape_cols):
            r_los = void_catalog['radius_los_mpc'].values
            r_trans = void_catalog['radius_transverse_mpc'].values

            # Axial ratios
            axial_ratios = r_los / (r_trans + 1e-10)  # Avoid division by zero
            features.update({
                'mean_axial_ratio': float(np.mean(axial_ratios)),
                'axial_ratio_std': float(np.std(axial_ratios)),
                'prolate_fraction': float(np.mean(axial_ratios > 1.5)),
                'oblate_fraction': float(np.mean(axial_ratios < 0.67))
            })

        return features

    def _extract_spatial_features(self, void_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract spatial distribution features."""
        features = {}

        # Position statistics
        if all(col in void_catalog.columns for col in ['ra_deg', 'dec_deg', 'redshift']):
            ra = void_catalog['ra_deg'].values
            dec = void_catalog['dec_deg'].values
            z = void_catalog['redshift'].values

            features.update({
                'ra_range': float(np.max(ra) - np.min(ra)),
                'dec_range': float(np.max(dec) - np.min(dec)),
                'z_range': float(np.max(z) - np.min(z)),
                'ra_center': float(np.mean(ra)),
                'dec_center': float(np.mean(dec)),
                'z_center': float(np.mean(z))
            })

            # Comoving distance if available
            if 'comoving_distance' in void_catalog.columns:
                comoving = void_catalog['comoving_distance'].values
                features.update({
                    'comoving_range': float(np.max(comoving) - np.min(comoving)),
                    'mean_comoving_distance': float(np.mean(comoving))
                })

        return features


    def _extract_network_features(self, void_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract void network topology features."""
        features = {}

        # Void network analysis
        if all(col in void_catalog.columns for col in ['ra_deg', 'dec_deg', 'redshift']):
            # Convert to 3D coordinates (simplified Cartesian)
            ra = np.radians(void_catalog['ra_deg'].values)
            dec = np.radians(void_catalog['dec_deg'].values)
            z = void_catalog['redshift'].values

            # Simplified 3D positions (assuming small angular scales)
            # In reality, would use proper cosmology for comoving coordinates
            x = z * np.cos(dec) * np.cos(ra)
            y = z * np.cos(dec) * np.sin(ra)
            z_coord = z * np.sin(dec)

            positions = np.column_stack([x, y, z_coord])

            # Nearest neighbor analysis
            if len(positions) > 1:
                nbrs = NearestNeighbors(n_neighbors=min(5, len(positions))).fit(positions)
                distances, indices = nbrs.kneighbors(positions)

                # Mean nearest neighbor distance
                mean_nn_distance = np.mean(distances[:, 1])  # Skip self (distance 0)
                std_nn_distance = np.std(distances[:, 1])

                features.update({
                    'mean_nn_distance': float(mean_nn_distance),
                    'nn_distance_std': float(std_nn_distance),
                    'nn_distance_skewness': float(stats.skew(distances[:, 1]))
                })

                # Void clustering coefficient (simplified)
                # Count mutual nearest neighbors
                mutual_nn = 0
                for i in range(len(indices)):
                    for j in indices[i, 1:]:  # Skip self
                        if i in indices[j, 1:]:  # Mutual neighbor
                            mutual_nn += 1

                clustering_coeff = mutual_nn / (len(positions) * 4)  # Normalized
                features['network_clustering_coeff'] = float(clustering_coeff)

        return features

    def _extract_clustering_features(self, void_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract void clustering and correlation features."""
        features = {}

        # Size clustering (do large voids cluster?)
        if 'radius_mpc' in void_catalog.columns:
            radii = void_catalog['radius_mpc'].values

            # Size-rank correlation
            size_ranks = stats.rankdata(radii)
            if all(col in void_catalog.columns for col in ['ra_deg', 'dec_deg', 'redshift']):
                ra = void_catalog['ra_deg'].values
                dec = void_catalog['dec_deg'].values
                z = void_catalog['redshift'].values

                # Simple spatial rank correlation
                ra_ranks = stats.rankdata(ra)
                dec_ranks = stats.rankdata(dec)
                z_ranks = stats.rankdata(z)

                # Spearman correlations
                ra_size_corr = stats.spearmanr(ra_ranks, size_ranks)[0]
                dec_size_corr = stats.spearmanr(dec_ranks, size_ranks)[0]
                z_size_corr = stats.spearmanr(z_ranks, size_ranks)[0]

                features.update({
                    'size_position_corr_ra': float(ra_size_corr),
                    'size_position_corr_dec': float(dec_size_corr),
                    'size_position_corr_z': float(z_size_corr)
                })

        return features

    def _extract_survey_features(self, void_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract survey-specific features."""
        features = {}

        if 'survey' in void_catalog.columns:
            survey_counts = void_catalog['survey'].value_counts()
            features['survey_counts'] = survey_counts.to_dict()

            # Survey diversity
            n_surveys = len(survey_counts)
            features['n_surveys'] = n_surveys
            features['survey_entropy'] = float(stats.entropy(survey_counts.values))

        if 'algorithm' in void_catalog.columns:
            algo_counts = void_catalog['algorithm'].value_counts()
            features['algorithm_counts'] = algo_counts.to_dict()

        return features


    def augment_void_data(self, void_catalog: pd.DataFrame,
                         augmentation_type: str = 'noise') -> pd.DataFrame:
        """
        Apply data augmentation for contrastive learning.

        Parameters:
            void_catalog: Original void catalog
            augmentation_type: Type of augmentation

        Returns:
            pd.DataFrame: Augmented void catalog
        """
        augmented = void_catalog.copy()

        if augmentation_type == 'noise':
            # Add noise to positions
            if 'ra_deg' in augmented.columns:
                augmented['ra_deg'] += np.random.normal(0, 0.1, len(augmented))
            if 'dec_deg' in augmented.columns:
                augmented['dec_deg'] += np.random.normal(0, 0.1, len(augmented))

        elif augmentation_type == 'scaling':
            # Randomly scale void sizes
            if 'radius_mpc' in augmented.columns:
                scale_factors = np.random.uniform(0.9, 1.1, len(augmented))
                augmented['radius_mpc'] *= scale_factors

        elif augmentation_type == 'rotation':
            # Rotate orientation angles
            if 'orientation_deg' in augmented.columns:
                rotation = np.random.uniform(0, 360)
                augmented['orientation_deg'] = (augmented['orientation_deg'] + rotation) % 360

        return augmented
