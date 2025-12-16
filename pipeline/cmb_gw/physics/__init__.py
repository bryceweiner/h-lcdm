"""
Physics modules for evolving G(z) calculations.
"""

from .evolving_g import G_ratio, H_evolving_G, c_s_baryon_photon
from .sound_horizon import (
    sound_horizon_evolving_G, 
    sound_horizon_lcdm,
    sound_horizon_camb,
    z_drag_eisenstein_hu,
    CAMB_AVAILABLE
)
from .growth_factor import growth_factor_evolving_G, void_size_ratio
from .luminosity_distance import luminosity_distance_evolving_G, dL_residual
from .cmb_peaks import cmb_peak_ratios_evolving_G
from .cmb_spectra import (
    compute_lcdm_cmb_spectrum,
    compute_cmb_residuals,
    get_acoustic_peak_predictions
)

# N-body calibration (requires nbodykit, optional)
try:
    from .nbody_void_calibration import (
        NBODYVoidCalibration,
        quick_calibration,
        NBODYKIT_AVAILABLE
    )
    _nbody_exports = [
        'NBODYVoidCalibration',
        'quick_calibration',
        'NBODYKIT_AVAILABLE'
    ]
except ImportError:
    _nbody_exports = []

# CAMB-based evolving G (optional)
try:
    from .camb_evolving_g import (
        CAMBEvolvingG,
        sound_horizon_evolving_G_camb,
        cmb_spectrum_evolving_G_camb,
        cmb_peak_ratios_evolving_G_camb
    )
    _camb_eg_exports = [
        'CAMBEvolvingG',
        'sound_horizon_evolving_G_camb',
        'cmb_spectrum_evolving_G_camb',
        'cmb_peak_ratios_evolving_G_camb'
    ]
except ImportError:
    _camb_eg_exports = []

__all__ = [
    'G_ratio',
    'H_evolving_G',
    'c_s_baryon_photon',
    'sound_horizon_evolving_G',
    'sound_horizon_lcdm',
    'sound_horizon_camb',
    'z_drag_eisenstein_hu',
    'CAMB_AVAILABLE',
    'growth_factor_evolving_G',
    'void_size_ratio',
    'luminosity_distance_evolving_G',
    'dL_residual',
    'cmb_peak_ratios_evolving_G',
    'compute_lcdm_cmb_spectrum',
    'compute_cmb_residuals',
    'get_acoustic_peak_predictions'
] + _nbody_exports + _camb_eg_exports

