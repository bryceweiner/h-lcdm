"""
FRB Feature Extractor
====================

Extracts features from Fast Radio Burst catalogs for ML training.
Includes timing, dispersion measure, and redshift features
that may reveal Little Bang information saturation signatures.
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from scipy import stats


class FRBFeatureExtractor:
    """
    Extract features from FRB catalogs for ML analysis.

    Focuses on timing patterns, dispersion measures, and
    redshift distributions that might reveal information saturation.
    """

    def extract_features(self, frb_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract features from FRB catalog."""
        if frb_catalog.empty:
            return {'error': 'Empty FRB catalog'}

        features = {}

        # Basic statistics
        features.update(self._extract_basic_statistics(frb_catalog))

        # Timing features
        features.update(self._extract_timing_features(frb_catalog))

        # Dispersion measure features
        features.update(self._extract_dm_features(frb_catalog))

        # Redshift features
        features.update(self._extract_redshift_features(frb_catalog))

        # Energy/rate features
        features.update(self._extract_energy_features(frb_catalog))

        return features

    def _extract_basic_statistics(self, frb_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract basic FRB catalog statistics."""
        n_frbs = len(frb_catalog)
        features = {'n_frbs': n_frbs}

        # Position statistics
        if 'ra' in frb_catalog.columns and 'dec' in frb_catalog.columns:
            ra = frb_catalog['ra'].values
            dec = frb_catalog['dec'].values
            features.update({
                'ra_range': float(np.max(ra) - np.min(ra)),
                'dec_range': float(np.max(dec) - np.min(dec)),
                'position_spread': float(np.std(ra) + np.std(dec))
            })

        return features

    def _extract_timing_features(self, frb_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract FRB timing features."""
        features = {}

        if 'time_interval_days' in frb_catalog.columns:
            intervals = frb_catalog['time_interval_days'].values
            intervals = intervals[intervals > 0]  # Remove zeros

            if len(intervals) > 0:
                features.update({
                    'mean_interval': float(np.mean(intervals)),
                    'interval_std': float(np.std(intervals)),
                    'min_interval': float(np.min(intervals)),
                    'max_interval': float(np.max(intervals)),
                    'interval_skewness': float(stats.skew(intervals))
                })

                # Test for periodicity (FFT)
                if len(intervals) > 10:
                    fft = np.fft.fft(intervals - np.mean(intervals))
                    power = np.abs(fft)**2
                    peak_power = np.max(power[1:])  # Exclude DC
                    mean_power = np.mean(power[1:])
                    features['periodicity_power_ratio'] = float(peak_power / mean_power)

        return features

    def _extract_dm_features(self, frb_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract dispersion measure features."""
        features = {}

        if 'dm' in frb_catalog.columns:
            dms = frb_catalog['dm'].values
            features.update({
                'mean_dm': float(np.mean(dms)),
                'dm_std': float(np.std(dms)),
                'min_dm': float(np.min(dms)),
                'max_dm': float(np.max(dms)),
                'dm_skewness': float(stats.skew(dms))
            })

            # DM-redshift correlation
            if 'redshift' in frb_catalog.columns:
                redshifts = frb_catalog['redshift'].values
                dm_z_corr = stats.spearmanr(dms, redshifts)[0]
                features['dm_redshift_correlation'] = float(dm_z_corr)

        return features

    def _extract_redshift_features(self, frb_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract redshift distribution features."""
        features = {}

        if 'redshift' in frb_catalog.columns:
            redshifts = frb_catalog['redshift'].values
            features.update({
                'mean_redshift': float(np.mean(redshifts)),
                'redshift_std': float(np.std(redshifts)),
                'redshift_range': float(np.max(redshifts) - np.min(redshifts))
            })

        return features

    def _extract_energy_features(self, frb_catalog: pd.DataFrame) -> Dict[str, Any]:
        """Extract energy and rate features."""
        features = {}

        if 'fluence_jy_ms' in frb_catalog.columns:
            fluences = frb_catalog['fluence_jy_ms'].values
            features.update({
                'mean_fluence': float(np.mean(fluences)),
                'fluence_std': float(np.std(fluences)),
                'fluence_range': float(np.max(fluences) - np.min(fluences))
            })

        return features

    def augment_frb_data(self, frb_catalog: pd.DataFrame,
                        augmentation_type: str = 'noise') -> pd.DataFrame:
        """Apply data augmentation."""
        augmented = frb_catalog.copy()

        if augmentation_type == 'timing_noise':
            if 'time_interval_days' in augmented.columns:
                noise = np.random.normal(0, 0.1, len(augmented))
                augmented['time_interval_days'] *= (1 + noise)

        return augmented
