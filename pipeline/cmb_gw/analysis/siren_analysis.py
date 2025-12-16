"""
Standard Siren Analysis
=======================

TEST 3: Fit β parameter from GW standard siren luminosity distances.

This module:
1. Loads GW event catalog with EM counterparts
2. Compares observed d_L to ΛCDM predictions
3. Fits β to minimize χ²
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional
from scipy.optimize import minimize
from hlcdm.parameters import HLCDM_PARAMS
from ..physics.luminosity_distance import luminosity_distance_evolving_G
from data.loader import DataLoader


def analyze_standard_sirens(
    gw_catalog: Optional[pd.DataFrame] = None,
    omega_m: Optional[float] = None,
    H0_cmb: Optional[float] = None,
    H0_local: Optional[float] = None,
    beta_range: tuple = (-0.1, 0.3)
) -> Dict[str, Any]:
    """
    Compare GW-inferred distances to ΛCDM predictions and fit β.
    
    This implements TEST 3 from docs/cmb_gw.md.
    
    Parameters:
    -----------
    gw_catalog : DataFrame, optional
        GW event catalog with columns [event_id, z, dL, dL_err].
        If None, loads from DataLoader
    omega_m : float, optional
        Matter density parameter. If None, uses HLCDM_PARAMS.OMEGA_M
    H0_cmb : float, optional
        Hubble constant from CMB (km/s/Mpc). If None, converts from HLCDM_PARAMS.H0
    H0_local : float, optional
        Hubble constant from local measurements (km/s/Mpc). Not used in fit, for reference.
    beta_range : tuple
        (min, max) range for β fitting
        
    Returns:
    --------
    dict
        Analysis results containing:
        - 'beta_fit': Best-fit β
        - 'beta_err': Uncertainty on β
        - 'chi2_lcdm': χ² for ΛCDM (β=0)
        - 'chi2_evolving': χ² for evolving G
        - 'delta_chi2': χ² improvement
        - 'n_events': Number of events used
    """
    if omega_m is None:
        omega_m = HLCDM_PARAMS.OMEGA_M
    
    # Convert H0 from s⁻¹ to km/s/Mpc if needed
    if H0_cmb is None:
        H0_s_per_s = HLCDM_PARAMS.H0
        H0_km_s_Mpc = H0_s_per_s / 3.24e-20
    else:
        H0_km_s_Mpc = H0_cmb
    
    # Load GW data if not provided
    if gw_catalog is None:
        data_loader = DataLoader()
        try:
            gw_catalog = data_loader.load_gw_data(detector='all')
        except Exception:
            return {
                'beta_fit': np.nan,
                'beta_err': np.nan,
                'chi2_lcdm': np.nan,
                'chi2_evolving': np.nan,
                'delta_chi2': np.nan,
                'n_events': 0
            }
    
    # Filter events with valid redshift and distance
    valid_mask = (
        gw_catalog['redshift'].notna() &
        gw_catalog['luminosity_distance'].notna() &
        (gw_catalog['redshift'] > 0) &
        (gw_catalog['redshift'] < 2.0)  # Reasonable redshift range
    )
    
    gw_valid = gw_catalog[valid_mask].copy()
    
    if len(gw_valid) == 0:
        return {
            'beta_fit': np.nan,
            'beta_err': np.nan,
            'chi2_lcdm': np.nan,
            'chi2_evolving': np.nan,
            'delta_chi2': np.nan,
            'n_events': 0
        }
    
    # Estimate distance uncertainties if not available
    if 'dL_err' not in gw_valid.columns or gw_valid['dL_err'].isna().all():
        # Rough estimate: ~10% uncertainty for GW standard sirens
        gw_valid['dL_err'] = gw_valid['luminosity_distance'] * 0.1
    
    def chi2_func(beta):
        """Compute χ² for given β"""
        total = 0.0
        for _, row in gw_valid.iterrows():
            z = row['redshift']
            dL_obs = row['luminosity_distance']
            dL_err = row['dL_err']
            
            dL_model = luminosity_distance_evolving_G(z, omega_m, H0_km_s_Mpc, beta)
            total += ((dL_obs - dL_model) / dL_err)**2
        
        return total
    
    # Fit β
    result = minimize(chi2_func, x0=0.0, bounds=[beta_range], method='L-BFGS-B')
    beta_fit = float(result.x[0])  # Ensure Python float, not numpy scalar
    
    # Compute χ² values
    chi2_lcdm = float(chi2_func(0.0))
    chi2_evolving = float(chi2_func(beta_fit))
    delta_chi2 = chi2_lcdm - chi2_evolving
    
    # Fisher matrix for error estimate
    eps = 0.001
    d2chi2 = (chi2_func(beta_fit + eps) - 2*chi2_func(beta_fit) + chi2_func(beta_fit - eps)) / eps**2
    beta_err = float(np.sqrt(2.0 / d2chi2)) if d2chi2 > 0 else float(np.inf)
    
    return {
        'beta_fit': beta_fit,
        'beta_err': beta_err,
        'chi2_lcdm': chi2_lcdm,
        'chi2_evolving': chi2_evolving,
        'delta_chi2': delta_chi2,
        'n_events': len(gw_valid)
    }

