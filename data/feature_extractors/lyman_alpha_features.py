"""
Lyman-alpha Feature Extractor
============================

Extracts features from Lyman-alpha forest spectra for ML training.
Includes optical depth evolution, flux statistics, and correlation
features that may reveal phase transition signatures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats


class LymanAlphaFeatureExtractor:
    """
    Extract features from Lyman-alpha forest data for ML analysis.

    Focuses on optical depth patterns, flux statistics, and
    redshift evolution that might reveal phase transitions.
    """

    def extract_features(self, lyman_alpha_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from Lyman-alpha data."""
        if lyman_alpha_data.empty:
            return {'error': 'Empty Lyman-alpha data'}

        features = {}

        # Process each spectrum
        all_features = []
        for _, spectrum in lyman_alpha_data.iterrows():
            spectrum_features = self._extract_spectrum_features(spectrum)
            all_features.append(spectrum_features)

        # Aggregate features across spectra
        features.update(self._aggregate_spectrum_features(all_features))

        # Cross-spectrum correlations
        features.update(self._extract_cross_spectrum_features(lyman_alpha_data))

        return features

    def _extract_spectrum_features(self, spectrum: pd.Series) -> Dict[str, Any]:
        """Extract features from single Lyman-alpha spectrum."""
        features = {}

        if 'flux' in spectrum.index and 'wavelength' in spectrum.index:
            flux = np.array(spectrum['flux'])
            wavelength = np.array(spectrum['wavelength'])

            # Basic flux statistics
            features.update({
                'mean_flux': float(np.mean(flux)),
                'flux_std': float(np.std(flux)),
                'min_flux': float(np.min(flux)),
                'max_flux': float(np.max(flux)),
                'flux_skewness': float(stats.skew(flux))
            })

            # Optical depth if available
            if 'optical_depth' in spectrum.index:
                tau = np.array(spectrum['optical_depth'])
                features.update({
                    'mean_tau': float(np.mean(tau)),
                    'tau_std': float(np.std(tau)),
                    'max_tau': float(np.max(tau))
                })

                # Transmission spikes (Lyman-alpha forest features)
                transmission = np.exp(-tau)
                spikes = transmission > 0.9  # High transmission regions
                features['transmission_spike_fraction'] = float(np.mean(spikes))

        if 'redshift' in spectrum.index:
            features['redshift'] = float(spectrum['redshift'])

        return features

    def _aggregate_spectrum_features(self, all_features: List[Dict]) -> Dict[str, Any]:
        """Aggregate features across all spectra."""
        if not all_features:
            return {}

        # Convert to arrays for statistical analysis
        feature_names = all_features[0].keys()
        aggregated = {}

        for feature_name in feature_names:
            values = [f.get(feature_name) for f in all_features if feature_name in f]
            if values:
                values = np.array(values)
                aggregated.update({
                    f'{feature_name}_mean': float(np.mean(values)),
                    f'{feature_name}_std': float(np.std(values)),
                    f'{feature_name}_median': float(np.median(values))
                })

        return aggregated

    def _extract_cross_spectrum_features(self, lyman_alpha_data: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from cross-spectrum correlations."""
        features = {}

        if 'redshift' in lyman_alpha_data.columns and len(lyman_alpha_data) > 1:
            redshifts = lyman_alpha_data['redshift'].values

            # Redshift evolution of mean flux
            if 'flux' in lyman_alpha_data.columns:
                mean_fluxes = []
                for _, spectrum in lyman_alpha_data.iterrows():
                    if 'flux' in spectrum.index:
                        flux = np.array(spectrum['flux'])
                        mean_fluxes.append(np.mean(flux))

                if len(mean_fluxes) > 1:
                    flux_evolution_corr = stats.spearmanr(redshifts, mean_fluxes)[0]
                    features['flux_redshift_evolution'] = float(flux_evolution_corr)

        return features

    def augment_lyman_alpha_data(self, lyman_alpha_data: pd.DataFrame,
                                augmentation_type: str = 'noise') -> pd.DataFrame:
        """Apply data augmentation."""
        augmented = lyman_alpha_data.copy()

        if augmentation_type == 'flux_noise':
            if 'flux' in augmented.columns:
                for i, flux in enumerate(augmented['flux']):
                    noise = np.random.normal(0, 0.05, len(flux))
                    augmented.at[i, 'flux'] = flux * (1 + noise)

        return augmented
