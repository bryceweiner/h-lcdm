"""
JWST Feature Extractor
=====================

Extracts features from JWST early galaxy catalogs for ML training.
Includes high-redshift galaxy properties, mass estimates, and
formation signatures that may reveal anti-viscosity limits.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats


class JWSTFeatureExtractor:
    """
    Extract features from JWST early galaxy data for ML analysis.

    Focuses on high-redshift galaxy properties and mass distributions
    that might reveal H-ΛCDM anti-viscosity signatures.
    """

    def extract_features(self, jwst_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from JWST catalog."""
        if jwst_catalog.empty:
            return {'error': 'Empty JWST catalog'}

        features = {}

        # Basic catalog statistics
        features.update(self._extract_basic_statistics(jwst_catalog))

        # Photometric features
        features.update(self._extract_photometric_features(jwst_catalog))

        # Morphological features
        features.update(self._extract_morphological_features(jwst_catalog))

        # Mass and redshift features
        features.update(self._extract_mass_redshift_features(jwst_catalog))

        return features

    def _extract_basic_statistics(self, jwst_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic JWST catalog statistics."""
        n_galaxies = len(jwst_catalog)
        features = {'n_galaxies': n_galaxies}

        if 'redshift' in jwst_catalog.columns:
            redshifts = jwst_catalog['redshift'].values
            features.update({
                'mean_redshift': float(np.mean(redshifts)),
                'redshift_std': float(np.std(redshifts)),
                'redshift_range': float(np.max(redshifts) - np.min(redshifts)),
                'high_z_fraction': float(np.mean(redshifts > 10))  # z > 10 galaxies
            })

        return features

    def _extract_photometric_features(self, jwst_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract photometric features."""
        features = {}

        # NIRCam magnitudes
        nir_filters = ['f200w_mag', 'f277w_mag', 'f356w_mag']
        for filt in nir_filters:
            if filt in jwst_catalog.columns:
                mags = jwst_catalog[filt].values
                features.update({
                    f'{filt}_mean': float(np.mean(mags)),
                    f'{filt}_std': float(np.std(mags)),
                    f'{filt}_median': float(np.median(mags))
                })

                # Color gradients
                if filt == 'f200w_mag' and 'f356w_mag' in jwst_catalog.columns:
                    color = jwst_catalog['f200w_mag'] - jwst_catalog['f356w_mag']
                    features['f200w_minus_f356w_mean'] = float(np.mean(color))

        return features

    def _extract_morphological_features(self, jwst_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract morphological features."""
        features = {}

        if 'half_light_radius_pix' in jwst_catalog.columns:
            radii = jwst_catalog['half_light_radius_pix'].values
            features.update({
                'mean_half_light_radius': float(np.mean(radii)),
                'radius_std': float(np.std(radii)),
                'compact_fraction': float(np.mean(radii < 0.5))  # Very compact galaxies
            })

        if 'sersic_index' in jwst_catalog.columns:
            sersic = jwst_catalog['sersic_index'].values
            features.update({
                'mean_sersic_index': float(np.mean(sersic)),
                'sersic_std': float(np.std(sersic)),
                'disk_fraction': float(np.mean(sersic < 2)),  # Disk-like
                'spheroid_fraction': float(np.mean(sersic > 2.5))  # Spheroid-like
            })

        return features

    def _extract_mass_redshift_features(self, jwst_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract mass and redshift evolution features."""
        features = {}

        if 'mass_log_msol' in jwst_catalog.columns:
            masses = jwst_catalog['mass_log_msol'].values
            features.update({
                'mean_log_mass': float(np.mean(masses)),
                'mass_std': float(np.std(masses)),
                'mass_range': float(np.max(masses) - np.min(masses)),
                'high_mass_fraction': float(np.mean(masses > 10))  # >10^10 Msun
            })

            # Mass-redshift relation
            if 'redshift' in jwst_catalog.columns:
                redshifts = jwst_catalog['redshift'].values
                mass_z_corr = stats.spearmanr(masses, redshifts)[0]
                features['mass_redshift_correlation'] = float(mass_z_corr)

                # Test for mass limits (potential anti-viscosity signature)
                z_bins = np.linspace(8, 15, 8)
                max_masses = []

                for i in range(len(z_bins) - 1):
                    mask = (redshifts >= z_bins[i]) & (redshifts < z_bins[i+1])
                    if np.sum(mask) > 0:
                        max_mass = np.max(masses[mask])
                        max_masses.append(max_mass)

                if max_masses:
                    features['max_mass_evolution'] = max_masses
                    features['mass_limit_slope'] = float(np.polyfit(z_bins[:-1], max_masses, 1)[0])

        return features

    def augment_jwst_data(self, jwst_catalog: pd.DataFrame,
                         augmentation_type: str = 'photometric') -> pd.DataFrame:
        """Apply data augmentation."""
        augmented = jwst_catalog.copy()

        if augmentation_type == 'photometric':
            # Add photometric noise
            nir_filters = ['f200w_mag', 'f277w_mag', 'f356w_mag']
            for filt in nir_filters:
                if filt in augmented.columns:
                    noise = np.random.normal(0, 0.1, len(augmented))
                    augmented[filt] += noise

        return augmented
