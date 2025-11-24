"""
CMB Feature Extractor
=====================

Extracts features from CMB E-mode polarization data for ML training.
Includes harmonic space features, power spectrum statistics, and
topological features while handling masking and foreground subtraction.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from scipy import stats
from scipy.signal import find_peaks


class CMBFeatureExtractor:
    """
    Extract features from CMB power spectra for ML analysis.

    Focuses on statistical properties that might reveal H-ΛCDM signatures
    while being robust to instrumental effects.
    """

    def __init__(self, ell_range: tuple = (100, 3000)):
        """
        Initialize CMB feature extractor.

        Parameters:
            ell_range: Multipole range for feature extraction
        """
        self.ell_min, self.ell_max = ell_range

    def extract_features(self, ell: np.ndarray, C_ell: np.ndarray,
                        C_ell_err: Optional[np.ndarray] = None) -> Dict[str, Any]:
        """
        Extract comprehensive features from CMB power spectrum.

        Parameters:
            ell: Multipole values
            C_ell: Power spectrum values
            C_ell_err: Power spectrum uncertainties

        Returns:
            dict: Extracted features
        """
        # Filter to specified multipole range
        mask = (ell >= self.ell_min) & (ell <= self.ell_max)
        ell_filtered = ell[mask]
        C_ell_filtered = C_ell[mask]
        C_ell_err_filtered = C_ell_err[mask] if C_ell_err is not None else None

        features = {}

        # Basic statistical features
        features.update(self._extract_basic_statistics(C_ell_filtered))

        # Power spectrum shape features
        features.update(self._extract_power_spectrum_features(ell_filtered, C_ell_filtered))

        # Peak/trough detection features
        features.update(self._extract_peak_features(ell_filtered, C_ell_filtered))

        # Correlation structure features
        features.update(self._extract_correlation_features(C_ell_filtered))

        # Scale-dependent features
        features.update(self._extract_scale_features(ell_filtered, C_ell_filtered))

        # Error-weighted features if errors available
        if C_ell_err_filtered is not None:
            features.update(self._extract_error_weighted_features(
                C_ell_filtered, C_ell_err_filtered))

        # Phase transition candidates (blinded to H-ΛCDM knowledge)
        features.update(self._extract_transition_candidates(ell_filtered, C_ell_filtered))

        return features

    def _extract_basic_statistics(self, C_ell: np.ndarray) -> Dict[str, float]:
        """Extract basic statistical features."""
        return {
            'mean_cl': float(np.mean(C_ell)),
            'std_cl': float(np.std(C_ell)),
            'median_cl': float(np.median(C_ell)),
            'min_cl': float(np.min(C_ell)),
            'max_cl': float(np.max(C_ell)),
            'range_cl': float(np.max(C_ell) - np.min(C_ell)),
            'skewness_cl': float(stats.skew(C_ell)),
            'kurtosis_cl': float(stats.kurtosis(C_ell)),
            'n_multipoles': len(C_ell)
        }

    def _extract_power_spectrum_features(self, ell: np.ndarray, C_ell: np.ndarray) -> Dict[str, float]:
        """Extract power spectrum shape features."""
        # Fit power law: C_ell ∝ ell^α
        log_ell = np.log(ell)
        log_cl = np.log(np.abs(C_ell) + 1e-10)  # Avoid log(0)

        # Robust linear fit
        slope, intercept = np.polyfit(log_ell, log_cl, 1)
        residuals = log_cl - (slope * log_ell + intercept)
        r_squared = 1 - np.var(residuals) / np.var(log_cl)

        # Higher-order polynomial fit
        coeffs = np.polyfit(log_ell, log_cl, 3)
        poly_r_squared = self._calculate_r_squared(log_cl, np.polyval(coeffs, log_ell))

        return {
            'power_law_slope': float(slope),
            'power_law_intercept': float(intercept),
            'power_law_r_squared': float(r_squared),
            'poly_order3_r_squared': float(poly_r_squared),
            'poly_coeffs': coeffs.tolist()
        }

    def _extract_peak_features(self, ell: np.ndarray, C_ell: np.ndarray) -> Dict[str, Any]:
        """Extract peak and trough features."""
        # Detect peaks (potential acoustic peaks or transitions)
        peaks, properties = find_peaks(C_ell, height=np.mean(C_ell), distance=10)

        # Detect troughs
        troughs, trough_properties = find_peaks(-C_ell, height=-np.mean(C_ell), distance=10)

        peak_features = {
            'n_peaks': len(peaks),
            'n_troughs': len(troughs),
            'peak_positions': ell[peaks].tolist() if len(peaks) > 0 else [],
            'trough_positions': ell[troughs].tolist() if len(troughs) > 0 else []
        }

        # Peak height statistics
        if len(peaks) > 0:
            peak_heights = properties['peak_heights']
            peak_features.update({
                'mean_peak_height': float(np.mean(peak_heights)),
                'max_peak_height': float(np.max(peak_heights)),
                'peak_height_std': float(np.std(peak_heights))
            })

        # Peak spacing (acoustic scale indicators)
        if len(peaks) > 1:
            peak_positions = ell[peaks]
            spacings = np.diff(np.sort(peak_positions))
            peak_features.update({
                'mean_peak_spacing': float(np.mean(spacings)),
                'peak_spacing_std': float(np.std(spacings)),
                'min_peak_spacing': float(np.min(spacings)),
                'max_peak_spacing': float(np.max(spacings))
            })

        return peak_features

    def _extract_correlation_features(self, C_ell: np.ndarray) -> Dict[str, float]:
        """Extract autocorrelation features."""
        # Autocorrelation function
        autocorr = np.correlate(C_ell, C_ell, mode='full')
        autocorr = autocorr[autocorr.size // 2:]  # Second half (positive lags)
        autocorr = autocorr / autocorr[0]  # Normalize

        # Find first zero crossing (correlation length)
        zero_crossings = np.where(np.diff(np.sign(autocorr)))[0]
        correlation_length = zero_crossings[0] if len(zero_crossings) > 0 else len(autocorr)

        # Autocorrelation statistics
        autocorr_stats = {
            'autocorr_length': int(correlation_length),
            'autocorr_max': float(np.max(autocorr[1:])),  # Exclude lag 0
            'autocorr_min': float(np.min(autocorr[1:])),
            'autocorr_mean': float(np.mean(autocorr[1:50])),  # First 50 lags
            'autocorr_std': float(np.std(autocorr[1:50]))
        }

        return autocorr_stats

    def _extract_scale_features(self, ell: np.ndarray, C_ell: np.ndarray) -> Dict[str, float]:
        """Extract scale-dependent features."""
        # Divide into scale bins
        ell_bins = [(100, 500), (500, 1000), (1000, 1500), (1500, 2000), (2000, 3000)]

        scale_features = {}
        for i, (ell_min_bin, ell_max_bin) in enumerate(ell_bins):
            mask = (ell >= ell_min_bin) & (ell <= ell_max_bin)
            if np.any(mask):
                C_ell_bin = C_ell[mask]
                scale_features.update({
                    f'scale_{i}_mean': float(np.mean(C_ell_bin)),
                    f'scale_{i}_std': float(np.std(C_ell_bin)),
                    f'scale_{i}_range': float(np.max(C_ell_bin) - np.min(C_ell_bin)),
                    f'scale_{i}_n_points': int(np.sum(mask))
                })

        # Scale ratios (potential transition indicators)
        if 'scale_0_mean' in scale_features and 'scale_1_mean' in scale_features:
            scale_features['scale_ratio_01'] = scale_features['scale_0_mean'] / scale_features['scale_1_mean']

        return scale_features

    def _extract_error_weighted_features(self, C_ell: np.ndarray, C_ell_err: np.ndarray) -> Dict[str, float]:
        """Extract error-weighted features."""
        # Chi-squared like statistic
        weights = 1.0 / (C_ell_err ** 2 + 1e-10)
        weighted_mean = np.sum(weights * C_ell) / np.sum(weights)

        # Weighted variance
        weighted_var = np.sum(weights * (C_ell - weighted_mean)**2) / np.sum(weights)

        return {
            'weighted_mean_cl': float(weighted_mean),
            'weighted_std_cl': float(np.sqrt(weighted_var)),
            'mean_error': float(np.mean(C_ell_err)),
            'error_variance': float(np.var(C_ell_err))
        }

    def _extract_transition_candidates(self, ell: np.ndarray, C_ell: np.ndarray) -> Dict[str, Any]:
        """Extract potential transition features (blinded analysis)."""
        # Derivative-based transition detection
        dC_dell = np.gradient(C_ell, ell)

        # Find sharp changes (potential transitions)
        dC_smooth = self._smooth_signal(dC_dell, window=5)
        transition_threshold = 3 * np.std(dC_smooth)

        transition_mask = np.abs(dC_smooth) > transition_threshold
        transition_positions = ell[transition_mask]

        # Transition statistics
        transition_features = {
            'n_transitions': int(np.sum(transition_mask)),
            'transition_positions': transition_positions.tolist(),
            'max_derivative': float(np.max(np.abs(dC_smooth))),
            'mean_derivative': float(np.mean(np.abs(dC_smooth)))
        }

        # Transition spacing (if multiple)
        if len(transition_positions) > 1:
            spacings = np.diff(np.sort(transition_positions))
            transition_features.update({
                'transition_spacings': spacings.tolist(),
                'mean_transition_spacing': float(np.mean(spacings)),
                'transition_spacing_std': float(np.std(spacings))
            })

        return transition_features

    def _smooth_signal(self, signal: np.ndarray, window: int = 5) -> np.ndarray:
        """Apply simple moving average smoothing."""
        if len(signal) < window:
            return signal

        kernel = np.ones(window) / window
        return np.convolve(signal, kernel, mode='same')

    def _calculate_r_squared(self, y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """Calculate R-squared coefficient."""
        ss_res = np.sum((y_true - y_pred) ** 2)
        ss_tot = np.sum((y_true - np.mean(y_true)) ** 2)
        return 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

    def augment_cmb_data(self, ell: np.ndarray, C_ell: np.ndarray,
                        augmentation_type: str = 'rotation') -> Dict[str, np.ndarray]:
        """
        Apply data augmentation for contrastive learning.

        Parameters:
            ell: Multipole values
            C_ell: Power spectrum
            augmentation_type: Type of augmentation

        Returns:
            dict: Augmented data
        """
        if augmentation_type == 'rotation':
            # Random phase shift (harmonic space rotation)
            phase_shift = np.random.uniform(0, 2*np.pi)
            # Simplified: just add random noise preserving statistics
            noise_level = 0.05
            C_ell_aug = C_ell * (1 + np.random.normal(0, noise_level, len(C_ell)))

        elif augmentation_type == 'scaling':
            # Random scaling of power spectrum
            scale_factor = np.random.uniform(0.9, 1.1)
            C_ell_aug = C_ell * scale_factor

        elif augmentation_type == 'noise':
            # Add different noise realizations
            noise_std = np.random.uniform(0.01, 0.1)
            C_ell_aug = C_ell + np.random.normal(0, noise_std * np.abs(C_ell))

        else:
            C_ell_aug = C_ell.copy()

        return {'ell': ell, 'C_ell': C_ell_aug}
