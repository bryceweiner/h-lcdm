"""
CAMB Interface for CMB Power Spectrum Computation
==================================================

Wrapper for CAMB Boltzmann solver to compute theoretical CMB power spectra.
Provides interface for both ΛCDM and H-ΛCDM model predictions.

References:
    - CAMB: https://camb.readthedocs.io/
    - Planck 2018 parameters: arXiv:1807.06209
"""

import numpy as np
from typing import Tuple, Dict, Any, Optional
import logging

try:
    import camb
    from camb import model, initialpower
    CAMB_AVAILABLE = True
except ImportError:
    CAMB_AVAILABLE = False
    logging.warning("CAMB not available. Install with: pip install camb")

logger = logging.getLogger(__name__)


class CAMBInterface:
    """
    CAMB wrapper for CMB power spectrum computation.
    
    Provides methods to compute theoretical CMB TT power spectra using
    the CAMB Boltzmann solver. Used by both ΛCDM and H-ΛCDM models.
    """
    
    # Planck 2018 best-fit parameters (Table 2, arXiv:1807.06209)
    PLANCK_2018_PARAMS = {
        'ombh2': 0.02237,      # Ω_b h²
        'omch2': 0.1200,       # Ω_c h²
        'tau': 0.0544,         # optical depth
        'ns': 0.9649,          # scalar spectral index
        'As': 2.1e-9,          # scalar amplitude (at k=0.05 Mpc⁻¹)
        'H0': 67.36,           # Hubble constant (km/s/Mpc)
        'mnu': 0.06,           # sum of neutrino masses (eV)
    }
    
    def __init__(self):
        """Initialize CAMB interface."""
        if not CAMB_AVAILABLE:
            raise ImportError(
                "CAMB is required but not installed. "
                "Install with: pip install camb"
            )
        self.camb = camb
    
    def compute_cl_tt(
        self,
        params: Dict[str, float],
        lmax: int = 3000,
        lmin: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute C_ℓ^TT power spectrum using CAMB.
        
        Parameters:
            params: Dictionary with cosmological parameters:
                - ombh2: Baryon density parameter × h²
                - omch2: Cold dark matter density parameter × h²
                - tau: Optical depth to reionization
                - ns: Scalar spectral index
                - As: Scalar amplitude at k=0.05 Mpc⁻¹
                - H0: Hubble constant (km/s/Mpc)
                - mnu: Sum of neutrino masses (eV, optional)
            lmax: Maximum multipole to compute
            lmin: Minimum multipole to compute
            
        Returns:
            tuple: (ell, C_ell) where:
                - ell: Multipole array
                - C_ell: Power spectrum in μK²
        """
        # Set up CAMB parameters
        pars = camb.CAMBparams()
        
        # Set cosmological parameters
        # Get mnu value (use same default for consistency)
        mnu_value = params.get('mnu', self.PLANCK_2018_PARAMS['mnu'])
        
        pars.set_cosmology(
            H0=params.get('H0', self.PLANCK_2018_PARAMS['H0']),
            ombh2=params.get('ombh2', self.PLANCK_2018_PARAMS['ombh2']),
            omch2=params.get('omch2', self.PLANCK_2018_PARAMS['omch2']),
            tau=params.get('tau', self.PLANCK_2018_PARAMS['tau']),
            mnu=mnu_value,
            num_massive_neutrinos=1 if mnu_value > 0 else 0
        )
        
        # Set primordial power spectrum
        pars.InitPower.set_params(
            ns=params.get('ns', self.PLANCK_2018_PARAMS['ns']),
            As=params.get('As', self.PLANCK_2018_PARAMS['As']),
            r=params.get('r', 0.0)  # Tensor-to-scalar ratio (default 0)
        )
        
        # Set accuracy and output options
        pars.set_for_lmax(lmax, lens_potential_accuracy=0)
        pars.WantTensors = False  # Scalar only for TT spectrum
        
        # Compute results
        results = camb.get_results(pars)
        
        # Get TT power spectrum
        # get_cmb_power_spectra with CMB_unit='muK' returns D_ell = ell(ell+1)C_ell/(2pi), not C_ell!
        # This is a CAMB quirk despite the function name
        powers = results.get_cmb_power_spectra(lmax=lmax, CMB_unit='muK')
        dl_tt = powers['total'][:, 0]  # TT spectrum in D_ell format (first column)
        
        # Create multipole array
        ell = np.arange(len(dl_tt))
        
        # Convert D_ell to C_ell to match data loader convention
        # C_ell = D_ell * 2pi / (ell(ell+1))
        with np.errstate(divide='ignore', invalid='ignore'):
            cl_tt = dl_tt * (2 * np.pi) / (ell * (ell + 1))
            # Handle ell=0,1 where division is undefined
            cl_tt[0] = 0.0
            cl_tt[1] = 0.0
        
        # Return only requested range
        mask = (ell >= lmin) & (ell <= lmax)
        return ell[mask], cl_tt[mask]
    
    def compute_cl_te(
        self,
        params: Dict[str, float],
        lmax: int = 3000,
        lmin: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute C_ℓ^TE power spectrum using CAMB.
        
        Parameters:
            params: Dictionary with cosmological parameters (see compute_cl_tt)
            lmax: Maximum multipole to compute
            lmin: Minimum multipole to compute
            
        Returns:
            tuple: (ell, C_ell) where:
                - ell: Multipole array
                - C_ell: TE power spectrum in μK²
        """
        # Set up CAMB parameters (same as compute_cl_tt)
        pars = camb.CAMBparams()
        
        # Get mnu value (use same default for consistency)
        mnu_value = params.get('mnu', self.PLANCK_2018_PARAMS['mnu'])
        
        pars.set_cosmology(
            H0=params.get('H0', self.PLANCK_2018_PARAMS['H0']),
            ombh2=params.get('ombh2', self.PLANCK_2018_PARAMS['ombh2']),
            omch2=params.get('omch2', self.PLANCK_2018_PARAMS['omch2']),
            tau=params.get('tau', self.PLANCK_2018_PARAMS['tau']),
            mnu=mnu_value,
            num_massive_neutrinos=1 if mnu_value > 0 else 0
        )
        
        # Set primordial power spectrum
        pars.InitPower.set_params(
            ns=params.get('ns', self.PLANCK_2018_PARAMS['ns']),
            As=params.get('As', self.PLANCK_2018_PARAMS['As']),
            r=params.get('r', 0.0)
        )
        
        # Set accuracy and output options
        pars.set_for_lmax(lmax, lens_potential_accuracy=0)
        pars.WantTensors = False
        
        # Compute results
        results = camb.get_results(pars)
        
        # Get TE power spectrum
        # get_cmb_power_spectra with CMB_unit='muK' returns D_ell = ell(ell+1)C_ell/(2pi), not C_ell!
        powers = results.get_cmb_power_spectra(lmax=lmax, CMB_unit='muK')
        dl_te = powers['total'][:, 3]  # TE spectrum in D_ell format (4th column: 0=TT, 1=EE, 2=BB, 3=TE)
        
        # Create multipole array
        ell = np.arange(len(dl_te))
        
        # Convert D_ell to C_ell to match data loader convention
        with np.errstate(divide='ignore', invalid='ignore'):
            cl_te = dl_te * (2 * np.pi) / (ell * (ell + 1))
            cl_te[0] = 0.0
            cl_te[1] = 0.0
        
        # Return only requested range
        mask = (ell >= lmin) & (ell <= lmax)
        return ell[mask], cl_te[mask]
    
    def compute_cl_ee(
        self,
        params: Dict[str, float],
        lmax: int = 3000,
        lmin: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute C_ℓ^EE power spectrum using CAMB.
        
        Parameters:
            params: Dictionary with cosmological parameters (see compute_cl_tt)
            lmax: Maximum multipole to compute
            lmin: Minimum multipole to compute
            
        Returns:
            tuple: (ell, C_ell) where:
                - ell: Multipole array
                - C_ell: EE power spectrum in μK²
        """
        # Set up CAMB parameters (same as compute_cl_tt)
        pars = camb.CAMBparams()
        
        # Get mnu value (use same default for consistency)
        mnu_value = params.get('mnu', self.PLANCK_2018_PARAMS['mnu'])
        
        pars.set_cosmology(
            H0=params.get('H0', self.PLANCK_2018_PARAMS['H0']),
            ombh2=params.get('ombh2', self.PLANCK_2018_PARAMS['ombh2']),
            omch2=params.get('omch2', self.PLANCK_2018_PARAMS['omch2']),
            tau=params.get('tau', self.PLANCK_2018_PARAMS['tau']),
            mnu=mnu_value,
            num_massive_neutrinos=1 if mnu_value > 0 else 0
        )
        
        # Set primordial power spectrum
        pars.InitPower.set_params(
            ns=params.get('ns', self.PLANCK_2018_PARAMS['ns']),
            As=params.get('As', self.PLANCK_2018_PARAMS['As']),
            r=params.get('r', 0.0)
        )
        
        # Set accuracy and output options
        pars.set_for_lmax(lmax, lens_potential_accuracy=0)
        pars.WantTensors = False
        
        # Compute results
        results = camb.get_results(pars)
        
        # Get EE power spectrum
        # get_cmb_power_spectra with CMB_unit='muK' returns D_ell = ell(ell+1)C_ell/(2pi), not C_ell!
        powers = results.get_cmb_power_spectra(lmax=lmax, CMB_unit='muK')
        dl_ee = powers['total'][:, 1]  # EE spectrum in D_ell format (2nd column: 0=TT, 1=EE, 2=BB, 3=TE)
        
        # Create multipole array
        ell = np.arange(len(dl_ee))
        
        # Convert D_ell to C_ell to match data loader convention
        with np.errstate(divide='ignore', invalid='ignore'):
            cl_ee = dl_ee * (2 * np.pi) / (ell * (ell + 1))
            cl_ee[0] = 0.0
            cl_ee[1] = 0.0
        
        # Return only requested range
        mask = (ell >= lmin) & (ell <= lmax)
        return ell[mask], cl_ee[mask]
    
    def compute_dl_tt(
        self,
        params: Dict[str, float],
        lmax: int = 3000,
        lmin: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute D_ℓ^TT = ℓ(ℓ+1)C_ℓ/(2π) using CAMB.
        
        Parameters:
            params: Dictionary with cosmological parameters (see compute_cl_tt)
            lmax: Maximum multipole to compute
            lmin: Minimum multipole to compute
            
        Returns:
            tuple: (ell, D_ell) where:
                - ell: Multipole array
                - D_ell: Power spectrum D_ℓ in μK²
        """
        ell, cl_tt = self.compute_cl_tt(params, lmax, lmin)
        
        # Convert C_ell to D_ell
        # D_ell = ell(ell+1) * C_ell / (2π)
        dl_tt = ell * (ell + 1) * cl_tt / (2 * np.pi)
        
        return ell, dl_tt
    
    def compute_cl_tt_at_multipoles(
        self,
        params: Dict[str, float],
        ell_obs: np.ndarray,
        lmax: int = 3000
    ) -> np.ndarray:
        """
        Compute C_ℓ^TT at specific observed multipoles.
        
        Computes full spectrum then interpolates to observed ℓ values.
        
        Parameters:
            params: Dictionary with cosmological parameters
            ell_obs: Array of multipoles at which to evaluate
            lmax: Maximum multipole for CAMB computation
            
        Returns:
            np.ndarray: C_ℓ^TT values at ell_obs
        """
        ell_full, cl_full = self.compute_cl_tt(params, lmax=lmax)
        
        # Interpolate to observed multipoles
        from scipy.interpolate import interp1d
        
        # Only interpolate within valid range
        valid_mask = (ell_obs >= ell_full.min()) & (ell_obs <= ell_full.max())
        cl_interp = np.full_like(ell_obs, np.nan)
        
        if valid_mask.any():
            interp_func = interp1d(
                ell_full,
                cl_full,
                kind='linear',
                bounds_error=False,
                fill_value=np.nan
            )
            cl_interp[valid_mask] = interp_func(ell_obs[valid_mask])
        
        return cl_interp

