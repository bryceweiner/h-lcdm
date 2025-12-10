"""
Residual Analysis Module - Phase 1
===================================

Computes residuals: ΔC_ℓ^XY = C_ℓ^XY,obs - C_ℓ^XY,ΛCDM(θ_best-fit)

Uses survey-specific ΛCDM best-fit parameters to avoid bias from parameter choice.
"""

import numpy as np
from typing import Dict, Tuple, Optional, Any
import logging
from scipy.interpolate import interp1d

from .camb_interface import CAMBInterface

logger = logging.getLogger(__name__)


class ResidualAnalyzer:
    """
    Compute residuals between observed and theoretical ΛCDM CMB power spectra.
    
    Uses survey-specific best-fit ΛCDM parameters to compute theoretical
    predictions, then subtracts from observed data to get residuals.
    """
    
    # Survey-specific ΛCDM best-fit parameters
    # Note: ACT DR6 and SPT-3G use Planck 2018 as baseline
    # TODO: Update with actual ACT DR6 and SPT-3G best-fit parameters when available
    SURVEY_PARAMS = {
        'planck_2018': {
            'ombh2': 0.02237,      # Ω_b h²
            'omch2': 0.1200,       # Ω_c h²
            'tau': 0.0544,         # optical depth
            'ns': 0.9649,          # scalar spectral index
            'As': 2.1e-9,          # scalar amplitude (at k=0.05 Mpc⁻¹)
            'H0': 67.36,           # Hubble constant (km/s/Mpc)
            'mnu': 0.06,           # sum of neutrino masses (eV)
        },
        'act_dr6': {
            # Using Planck 2018 as baseline (ACT DR6 typically consistent)
            'ombh2': 0.02237,
            'omch2': 0.1200,
            'tau': 0.0544,
            'ns': 0.9649,
            'As': 2.1e-9,
            'H0': 67.36,
            'mnu': 0.06,
        },
        'spt3g': {
            # Using Planck 2018 as baseline (SPT-3G typically consistent)
            'ombh2': 0.02237,
            'omch2': 0.1200,
            'tau': 0.0544,
            'ns': 0.9649,
            'As': 2.1e-9,
            'H0': 67.36,
            'mnu': 0.06,
        },
    }
    
    def __init__(self, camb_interface: Optional[CAMBInterface] = None):
        """
        Initialize residual analyzer.
        
        Parameters:
            camb_interface: CAMB interface instance (creates new if None)
        """
        self.camb = camb_interface or CAMBInterface()
    
    def compute_residuals(
        self,
        survey: str,
        spectrum: str,
        ell_obs: np.ndarray,
        cl_obs: np.ndarray,
        cl_err: np.ndarray
    ) -> Dict[str, np.ndarray]:
        """
        Compute residuals: ΔC_ℓ^XY = C_ℓ^XY,obs - C_ℓ^XY,ΛCDM(θ_best-fit)
        
        Parameters:
            survey: Survey name ('planck_2018', 'act_dr6', 'spt3g')
            spectrum: Spectrum type ('TT', 'TE', 'EE')
            ell_obs: Observed multipoles
            cl_obs: Observed C_ℓ^XY values
            cl_err: 1σ uncertainties on observed values
            
        Returns:
            dict with keys:
                - ell: Multipole array
                - residual: ΔC_ℓ = C_ℓ^obs - C_ℓ^ΛCDM
                - residual_fraction: ΔC_ℓ / C_ℓ^ΛCDM
                - cl_obs: Observed values
                - cl_theory: Theoretical ΛCDM values
                - cl_err: Observational errors
        """
        if survey not in self.SURVEY_PARAMS:
            raise ValueError(f"Unknown survey: {survey}. Must be one of {list(self.SURVEY_PARAMS.keys())}")
        
        if spectrum not in ['TT', 'TE', 'EE']:
            raise ValueError(f"Unknown spectrum: {spectrum}. Must be TT, TE, or EE")
        
        # Get survey-specific best-fit parameters
        params = self.SURVEY_PARAMS[survey].copy()
        
        # Compute theoretical ΛCDM spectrum
        # Determine lmax from observed data
        lmax = int(np.max(ell_obs)) + 200  # Extra range for interpolation
        
        # Compute full theoretical spectrum
        if spectrum == 'TT':
            ell_theory, cl_theory_full = self.camb.compute_cl_tt(params, lmax=lmax)
        elif spectrum == 'TE':
            ell_theory, cl_theory_full = self.camb.compute_cl_te(params, lmax=lmax)
        elif spectrum == 'EE':
            ell_theory, cl_theory_full = self.camb.compute_cl_ee(params, lmax=lmax)
        
        # Interpolate to observed multipoles
        interp_func = interp1d(
            ell_theory,
            cl_theory_full,
            kind='linear',
            bounds_error=False,
            fill_value=np.nan
        )
        cl_theory = interp_func(ell_obs)
        
        # Remove any NaN values (outside valid range)
        valid_mask = np.isfinite(cl_theory) & np.isfinite(cl_obs) & np.isfinite(cl_err)
        
        if not valid_mask.any():
            logger.warning(f"No valid data points for {survey} {spectrum}")
            return {
                'ell': ell_obs,
                'residual': np.full_like(ell_obs, np.nan),
                'residual_fraction': np.full_like(ell_obs, np.nan),
                'cl_obs': cl_obs,
                'cl_theory': cl_theory,
                'cl_err': cl_err,
            }
        
        # Compute residuals
        residual = cl_obs - cl_theory
        
        # Compute fractional residuals (avoid division by zero)
        residual_fraction = np.where(
            cl_theory != 0,
            residual / cl_theory,
            np.nan
        )
        
        return {
            'ell': ell_obs[valid_mask],
            'residual': residual[valid_mask],
            'residual_fraction': residual_fraction[valid_mask],
            'cl_obs': cl_obs[valid_mask],
            'cl_theory': cl_theory[valid_mask],
            'cl_err': cl_err[valid_mask],
        }
    
    def compute_all_residuals(
        self,
        datasets: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray, np.ndarray]]]
    ) -> Dict[str, Dict[str, Dict[str, np.ndarray]]]:
        """
        Compute TT, TE, EE residuals for all surveys.
        
        Parameters:
            datasets: Dictionary with survey names as keys, each containing:
                - 'TT': (ell, cl, cl_err)
                - 'TE': (ell, cl, cl_err)
                - 'EE': (ell, cl, cl_err)
        
        Returns:
            dict: Nested dictionary with structure:
                residuals[survey][spectrum] = residual_dict
        """
        results = {}
        
        for survey_name, survey_data in datasets.items():
            # Normalize survey name
            survey_key = survey_name.lower().replace('-', '_')
            if survey_key not in self.SURVEY_PARAMS:
                logger.warning(f"Skipping unknown survey: {survey_name}")
                continue
            
            results[survey_name] = {}
            
            for spectrum in ['TT', 'TE', 'EE']:
                if spectrum not in survey_data:
                    logger.warning(f"No {spectrum} data for {survey_name}")
                    continue
                
                ell, cl, cl_err = survey_data[spectrum]
                
                # DEBUG: Check actual data range before residual computation
                logger.info(f"DEBUG {survey_name} {spectrum} data range:")
                logger.info(f"  ell range: [{np.min(ell):.0f}, {np.max(ell):.0f}], n_points={len(ell)}")
                logger.info(f"  cl range: [{np.min(cl):.2e}, {np.max(cl):.2e}]")
                logger.info(f"  cl[0:3]: {cl[0:3]}")
                
                try:
                    residual_dict = self.compute_residuals(
                        survey_key,
                        spectrum,
                        ell,
                        cl,
                        cl_err
                    )
                    results[survey_name][spectrum] = residual_dict
                    logger.info(
                        f"✓ Computed {spectrum} residuals for {survey_name}: "
                        f"{len(residual_dict['ell'])} multipoles"
                    )
                except Exception as e:
                    logger.error(f"✗ Failed to compute {spectrum} residuals for {survey_name}: {e}")
                    results[survey_name][spectrum] = None
        
        return results

