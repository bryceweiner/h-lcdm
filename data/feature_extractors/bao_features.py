"""
BAO Feature Extractor
====================

Extracts features from BAO measurements for ML training.
Includes distance measurements, redshift evolution, and
correlation structure features.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from scipy import stats
from scipy.spatial.distance import pdist, squareform


class BAOFeatureExtractor:
    """
    Extract features from BAO distance measurements for ML analysis.

    Focuses on distance ratios, redshift evolution, and measurement
    correlations that might reveal sound horizon enhancement.
    """

    def __init__(self):
        """Initialize BAO feature extractor."""
        pass

    def extract_features(self, bao_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract comprehensive features from BAO data.

        Parameters:
            bao_data: BAO dataset from DataLoader

        Returns:
            dict: Extracted features
        """
        measurements = bao_data.get('measurements', [])
        correlation_matrix = bao_data.get('correlation_matrix')

        if not measurements:
            return {'error': 'No BAO measurements found'}

        features = {}

        # Basic measurement features
        features.update(self._extract_measurement_features(measurements))

        # Redshift evolution features
        features.update(self._extract_redshift_features(measurements))

        # Correlation structure features
        if correlation_matrix is not None:
            features.update(self._extract_correlation_features(correlation_matrix))

        # Distance ratio features
        features.update(self._extract_distance_features(measurements))

        # Statistical consistency features
        features.update(self._extract_consistency_features(measurements))

        # Measurement quality features
        features.update(self._extract_quality_features(measurements))

        return features

    def _extract_measurement_features(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Extract basic measurement statistics."""
        values = np.array([m['value'] for m in measurements])
        errors = np.array([m['error'] for m in measurements])
        redshifts = np.array([m['z'] for m in measurements])

        return {
            'n_measurements': len(measurements),
            'mean_value': float(np.mean(values)),
            'std_value': float(np.std(values)),
            'median_value': float(np.median(values)),
            'min_value': float(np.min(values)),
            'max_value': float(np.max(values)),
            'range_value': float(np.max(values) - np.min(values)),
            'mean_error': float(np.mean(errors)),
            'error_std': float(np.std(errors)),
            'mean_redshift': float(np.mean(redshifts)),
            'redshift_range': float(np.max(redshifts) - np.min(redshifts))
        }

    def _extract_redshift_features(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Extract redshift evolution features."""
        z_values = np.array([m['z'] for m in measurements])
        bao_values = np.array([m['value'] for m in measurements])

        # Sort by redshift
        sort_idx = np.argsort(z_values)
        z_sorted = z_values[sort_idx]
        bao_sorted = bao_values[sort_idx]

        features = {
            'redshifts': z_sorted.tolist(),
            'bao_values': bao_sorted.tolist()
        }

        # Linear trend with redshift
        if len(z_sorted) >= 2:
            slope, intercept = np.polyfit(z_sorted, bao_sorted, 1)
            residuals = bao_sorted - (slope * z_sorted + intercept)
            r_squared = 1 - np.var(residuals) / np.var(bao_sorted)

            features.update({
                'redshift_slope': float(slope),
                'redshift_intercept': float(intercept),
                'redshift_trend_r_squared': float(r_squared)
            })

        # Second-order polynomial fit
        if len(z_sorted) >= 3:
            coeffs = np.polyfit(z_sorted, bao_sorted, 2)
            poly_pred = np.polyval(coeffs, z_sorted)
            poly_r_squared = self._calculate_r_squared(bao_sorted, poly_pred)

            features.update({
                'poly_coeffs': coeffs.tolist(),
                'poly_r_squared': float(poly_r_squared)
            })

        # Evolution statistics
        if len(z_sorted) >= 2:
            bao_diffs = np.diff(bao_sorted)
            z_diffs = np.diff(z_sorted)

            evolution_rate = bao_diffs / z_diffs
            features.update({
                'mean_evolution_rate': float(np.mean(evolution_rate)),
                'evolution_rate_std': float(np.std(evolution_rate)),
                'max_evolution_rate': float(np.max(np.abs(evolution_rate)))
            })

        return features

    def _extract_correlation_features(self, correlation_matrix: np.ndarray) -> Dict[str, Any]:
        """Extract correlation matrix features."""
        # Basic correlation statistics
        corr_flat = correlation_matrix[np.triu_indices_from(correlation_matrix, k=1)]

        features = {
            'mean_correlation': float(np.mean(corr_flat)),
            'std_correlation': float(np.std(corr_flat)),
            'min_correlation': float(np.min(corr_flat)),
            'max_correlation': float(np.max(corr_flat)),
            'correlation_matrix_size': int(correlation_matrix.shape[0])
        }

        # Eigenvalue analysis (information about correlation structure)
        try:
            eigenvals = np.linalg.eigvals(correlation_matrix)
            eigenvals = np.real(eigenvals)  # Ensure real values

            features.update({
                'eigenvalues': eigenvals.tolist(),
                'max_eigenvalue': float(np.max(eigenvals)),
                'eigenvalue_ratio': float(np.max(eigenvals) / np.min(eigenvals)) if np.min(eigenvals) > 0 else 0,
                'condition_number': float(np.linalg.cond(correlation_matrix))
            })
        except np.linalg.LinAlgError:
            features.update({
                'eigenvalues': [],
                'max_eigenvalue': 0.0,
                'eigenvalue_ratio': 0.0,
                'condition_number': float('inf')
            })

        # Correlation clustering (groups of highly correlated measurements)
        strong_corr_threshold = 0.7
        n_strong_correlations = np.sum(np.abs(corr_flat) > strong_corr_threshold)

        features['n_strong_correlations'] = int(n_strong_correlations)

        return features

    def _extract_distance_features(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Extract distance ratio features."""
        values = np.array([m['value'] for m in measurements])

        features = {}

        # Ratios between measurements (potential scale indicators)
        if len(values) >= 2:
            ratios = []
            for i in range(len(values)):
                for j in range(i+1, len(values)):
                    ratio = values[i] / values[j] if values[j] != 0 else 0
                    ratios.append(ratio)

            features.update({
                'measurement_ratios': ratios,
                'mean_ratio': float(np.mean(ratios)),
                'ratio_std': float(np.std(ratios)),
                'min_ratio': float(np.min(ratios)),
                'max_ratio': float(np.max(ratios))
            })

        # Distance from expected ΛCDM values (blinded feature)
        # This provides a scale for how measurements deviate from baseline
        expected_lcdm = 13.0  # Rough ΛCDM expectation at z~0.5
        deviations = values - expected_lcdm
        normalized_deviations = deviations / expected_lcdm

        features.update({
            'deviations_from_lcdm': deviations.tolist(),
            'normalized_deviations': normalized_deviations.tolist(),
            'mean_deviation': float(np.mean(deviations)),
            'deviation_std': float(np.std(deviations))
        })

        return features

    def _extract_consistency_features(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Extract statistical consistency features."""
        values = np.array([m['value'] for m in measurements])
        errors = np.array([m['error'] for m in measurements])

        # Chi-squared like consistency check
        weighted_mean = np.sum(values / errors**2) / np.sum(1/errors**2)
        chi_squared = np.sum(((values - weighted_mean) / errors)**2)
        reduced_chi_squared = chi_squared / (len(values) - 1) if len(values) > 1 else 0

        features = {
            'weighted_mean': float(weighted_mean),
            'chi_squared': float(chi_squared),
            'reduced_chi_squared': float(reduced_chi_squared),
            'consistency_p_value': float(1 - stats.chi2.cdf(chi_squared, len(values) - 1))
        }

        # Outlier detection
        z_scores = (values - weighted_mean) / errors
        n_outliers = np.sum(np.abs(z_scores) > 3)  # 3σ outliers

        features.update({
            'z_scores': z_scores.tolist(),
            'n_outliers': int(n_outliers),
            'max_z_score': float(np.max(np.abs(z_scores)))
        })

        return features

    def _extract_quality_features(self, measurements: List[Dict]) -> Dict[str, Any]:
        """Extract measurement quality features."""
        errors = np.array([m['error'] for m in measurements])
        values = np.array([m['value'] for m in measurements])

        # Signal-to-noise ratios
        snr = values / errors

        # Error distributions
        features = {
            'snr_values': snr.tolist(),
            'mean_snr': float(np.mean(snr)),
            'median_snr': float(np.median(snr)),
            'min_snr': float(np.min(snr)),
            'max_snr': float(np.max(snr)),
            'snr_std': float(np.std(snr))
        }

        # Precision indicators
        relative_errors = errors / values
        features.update({
            'relative_errors': relative_errors.tolist(),
            'mean_relative_error': float(np.mean(relative_errors)),
            'median_relative_error': float(np.median(relative_errors)),
            'worst_precision': float(np.max(relative_errors))
        })

        return features

    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared coefficient."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def augment_bao_data(self, measurements: List[Dict],
                        augmentation_type: str = 'noise') -> List[Dict]:
        """
        Apply data augmentation for contrastive learning.

        Parameters:
            measurements: Original BAO measurements
            augmentation_type: Type of augmentation

        Returns:
            list: Augmented measurements
        """
        augmented = []

        for measurement in measurements:
            aug_measurement = measurement.copy()

            if augmentation_type == 'noise':
                # Add random noise to value
                noise_level = np.random.uniform(0.01, 0.05)  # 1-5% noise
                aug_measurement['value'] *= (1 + np.random.normal(0, noise_level))

            elif augmentation_type == 'error_variation':
                # Vary error bars
                error_factor = np.random.uniform(0.8, 1.2)
                aug_measurement['error'] *= error_factor

            elif augmentation_type == 'correlation':
                # Modify correlation (if exists) - simplified
                pass  # Correlation augmentation more complex

            augmented.append(aug_measurement)

        return augmented
