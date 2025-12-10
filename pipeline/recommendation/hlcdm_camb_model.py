"""
H-ΛCDM CAMB Model - Physically Correct Implementation
=======================================================

From bao_resolution_qit.tex:

H-ΛCDM differs from ΛCDM through quantum Zeno backreaction during recombination.
The α = -5.7 coefficient modifies acoustic transport:
    ∂v/∂t = -∇P/ρ - ∇Φ + α·γ·∇²v

This creates negative effective damping (anti-viscosity) that enhances the sound 
horizon by ~2.18%:
    r_s^H-ΛCDM = r_s^ΛCDM × [1 - α(γ/H)] = 150.71 Mpc

CRITICAL: H-ΛCDM uses standard ΛCDM cosmology. The α coefficient does NOT modify:
- Background evolution (Λ remains constant)
- CMB power spectra (C_ℓ^TT, C_ℓ^TE, C_ℓ^EE)
- Angular diameter distance D_A

The enhanced r_s matters only for BAO analysis, where r_s is the standard ruler.
"""

import numpy as np
from typing import Dict, Tuple, Optional
import logging
import scipy.constants as const

from .camb_interface import CAMBInterface

logger = logging.getLogger(__name__)


class HLCDMCAMBModel:
    """
    H-ΛCDM model: Coherent Acoustic Enhancement at Recombination.
    
    Physical mechanism (bao_resolution_qit.tex):
    - Thomson scattering rate Γ_T/H ~ 10⁹ at z≈1100 (quantum Zeno regime)
    - Information processing rate γ ≈ H/ln(S_max) ~ 10⁻¹⁶ s⁻¹
    - Coherent acoustic enhancement α = -5.7 (from Lindblad-Zeno scaling)
    - Enhanced sound horizon r_s = 150.71 Mpc (2.18% larger than ΛCDM)
    
    For CMB analysis: H-ΛCDM and ΛCDM predict identical power spectra.
    For BAO analysis: Use enhanced r_s = 150.71 Mpc as the standard ruler.
    """
    
    # Coherent acoustic enhancement coefficient (Eq. 168-172)
    ALPHA = -5.7
    
    # Standard ΛCDM sound horizon (Planck 2018)
    R_S_LCDM = 147.5  # Mpc
    
    # Enhanced sound horizon from quantum Zeno backreaction
    R_S_HLCDM = 150.71  # Mpc
    
    # Hubble parameter at recombination
    H_RECOMBINATION = 4.47e-14  # s⁻¹ at z≈1100
    
    def __init__(self, camb_interface: Optional[CAMBInterface] = None):
        """
        Initialize H-ΛCDM model.
        
        Parameters:
            camb_interface: CAMB interface for computing spectra
        """
        self.camb = camb_interface or CAMBInterface()
    
    def compute_gamma(self, H: float) -> float:
        """
        Compute holographic information processing rate.
        
        From bao_resolution_qit.tex Eq. 56-59:
            γ = H / ln(S_max)
            S_max = πc⁵/(GℏH²)
        
        This is the coarse-grained rate derived from Bekenstein bound and
        Margolus-Levitin limit.
        
        Parameters:
            H: Hubble parameter in km/s/Mpc
            
        Returns:
            float: γ in km/s/Mpc
        """
        # Convert H to SI units
        H_si = H * 1e3 / 3.086e22  # km/s/Mpc -> s⁻¹
        
        # Compute ln(S_max) = ln(πc⁵/(GℏH²))
        numerator = np.pi * (const.c ** 5)
        denominator = const.G * const.hbar * (H_si ** 2)
        ln_S_max = np.log(numerator / denominator)
        
        # γ = H / ln(S_max)
        gamma_si = H_si / ln_S_max
        
        # Convert back to km/s/Mpc
        gamma = gamma_si * 3.086e22 / 1e3
        
        return gamma
    
    def compute_sound_horizon_enhancement(self, params: Dict[str, float] = None) -> float:
        """
        Compute sound horizon enhancement factor.
        
        From bao_resolution_qit.tex Eq. 196-201:
            r_s^H-ΛCDM = r_s^ΛCDM × [1 - α(γ/H)]
        
        At recombination (z≈1100):
            H ≈ 4.47×10⁻¹⁴ s⁻¹
            γ ≈ 1.7×10⁻¹⁶ s⁻¹
            γ/H ≈ 0.0038
            α = -5.7
            Enhancement = 1 - (-5.7)(0.0038) = 1.02177 ≈ 2.18%
        
        Parameters:
            params: Cosmological parameters (uses H_RECOMBINATION if None)
            
        Returns:
            float: Enhancement factor [1 - α(γ/H)]
        """
        # Use recombination H value
        if params is None:
            H_recomb_kmsmpc = self.H_RECOMBINATION * 3.086e22 / 1e3
        else:
            # Could compute H(z=1100) from params, but constant is sufficient
            H_recomb_kmsmpc = self.H_RECOMBINATION * 3.086e22 / 1e3
        
        # Compute γ/H at recombination
        gamma = self.compute_gamma(H_recomb_kmsmpc)
        gamma_over_H = gamma / H_recomb_kmsmpc
        
        # Enhancement factor
        enhancement = 1.0 - self.ALPHA * gamma_over_H
        
        logger.info(f"Sound horizon enhancement: {enhancement:.6f} ({(enhancement-1)*100:.2f}%)")
        logger.info(f"  r_s: ΛCDM={self.R_S_LCDM} Mpc → H-ΛCDM={self.R_S_HLCDM} Mpc")
        
        return enhancement
    
    def compute_hlcdm_spectrum(
        self,
        params: Dict[str, float],
        spectrum: str = 'TT',
        lmax: int = 3000,
        lmin: int = 2
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute H-ΛCDM CMB power spectrum.
        
        IMPORTANT: Returns standard ΛCDM spectrum!
        
        The α coefficient affects recombination transport, which enhances r_s.
        However, the CMB angular scale θ_s = r_s/D_A is unchanged because
        both r_s and D_A are enhanced by the same factor.
        
        The enhanced r_s matters for BAO analysis (where r_s is the ruler),
        NOT for CMB power spectra.
        
        Parameters:
            params: Cosmological parameters
            spectrum: 'TT', 'TE', or 'EE'
            lmax: Maximum multipole
            lmin: Minimum multipole
            
        Returns:
            tuple: (ell, C_ell) - Standard ΛCDM power spectrum in μK²
        """
        # H-ΛCDM uses standard ΛCDM for CMB
        if spectrum == 'TT':
            return self.camb.compute_cl_tt(params, lmax=lmax, lmin=lmin)
        elif spectrum == 'TE':
            return self.camb.compute_cl_te(params, lmax=lmax, lmin=lmin)
        elif spectrum == 'EE':
            return self.camb.compute_cl_ee(params, lmax=lmax, lmin=lmin)
        else:
            raise ValueError(f"Unknown spectrum: {spectrum}")
    
    def compare_with_lcdm(
        self,
        params: Dict[str, float],
        ell_obs: np.ndarray,
        cl_obs: np.ndarray,
        cl_err: np.ndarray,
        spectrum: str = 'TT'
    ) -> Dict[str, float]:
        """
        Compare H-ΛCDM and ΛCDM fits to CMB data.
        
        For CMB: Both models predict identical spectra, so χ² values are equal.
        The difference is in the sound horizon used for BAO analysis.
        
        Parameters:
            params: Cosmological parameters
            ell_obs: Observed multipoles
            cl_obs: Observed C_ell in μK²
            cl_err: Errors in μK²
            spectrum: 'TT', 'TE', or 'EE'
            
        Returns:
            dict with:
                - chi2_lcdm: χ² for ΛCDM
                - chi2_hlcdm: χ² for H-ΛCDM (same as ΛCDM)
                - delta_chi2: 0.0 (no difference for CMB)
                - r_s_lcdm: 147.5 Mpc
                - r_s_hlcdm: 150.71 Mpc
        """
        # Compute spectrum (same for both models)
        if spectrum == 'TT':
            ell_theory, cl_theory = self.camb.compute_cl_tt(params, lmax=int(np.max(ell_obs)) + 100)
        elif spectrum == 'TE':
            ell_theory, cl_theory = self.camb.compute_cl_te(params, lmax=int(np.max(ell_obs)) + 100)
        elif spectrum == 'EE':
            ell_theory, cl_theory = self.camb.compute_cl_ee(params, lmax=int(np.max(ell_obs)) + 100)
        else:
            raise ValueError(f"Unknown spectrum: {spectrum}")
        
        # Interpolate to observed multipoles
        from scipy.interpolate import interp1d
        interp_func = interp1d(ell_theory, cl_theory, kind='linear', bounds_error=False, fill_value=np.nan)
        cl_theory_obs = interp_func(ell_obs)
        
        # Compute χ²
        valid_mask = np.isfinite(cl_theory_obs) & np.isfinite(cl_obs) & np.isfinite(cl_err) & (cl_err > 0)
        
        if not valid_mask.any():
            logger.warning(f"No valid data for {spectrum}")
            return {
                'chi2_lcdm': np.nan,
                'chi2_hlcdm': np.nan,
                'delta_chi2': 0.0,
                'r_s_lcdm': self.R_S_LCDM,
                'r_s_hlcdm': self.R_S_HLCDM,
            }
        
        chi2 = np.sum(((cl_obs[valid_mask] - cl_theory_obs[valid_mask]) / cl_err[valid_mask]) ** 2)
        
        # Both models give same χ² for CMB
        logger.info(f"{spectrum}: χ² = {chi2:.2f} (same for ΛCDM and H-ΛCDM)")
        logger.info(f"  Difference is in BAO ruler: r_s={self.R_S_LCDM} Mpc (ΛCDM) vs {self.R_S_HLCDM} Mpc (H-ΛCDM)")
        
        return {
            'chi2_lcdm': chi2,
            'chi2_hlcdm': chi2,  # Same for CMB!
            'delta_chi2': 0.0,
            'r_s_lcdm': self.R_S_LCDM,
            'r_s_hlcdm': self.R_S_HLCDM,
        }
