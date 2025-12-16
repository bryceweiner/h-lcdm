"""
BAO Sound Horizon Analysis
==========================

TEST 1: Infer sound horizon r_s from BAO measurements and fit β parameter.

This module:
1. Loads BAO data (D_V/r_d or D_M/r_d measurements)
2. Infers observational r_s using fiducial ΛCDM cosmology
3. Compares to theoretical r_s(β) predictions
4. Fits β to minimize χ²

For ΛCDM baseline, uses CAMB-based sound_horizon_lcdm() which returns
the precise value r_s = 147.09 Mpc (Planck 2018).

IMPORTANT METHODOLOGICAL CAVEAT:
--------------------------------
The "observed" r_s is NOT a direct measurement. BAO surveys measure the ratio
D/r_d (distance/sound horizon). To extract r_s, we must assume a fiducial
cosmology to compute D. This makes the "inferred" r_s dependent on the
assumed H0 and Ω_m.

The correct interpretation is:
- If inferred r_s > ΛCDM prediction → BAO distances are LARGER than expected
  → Could indicate weaker early gravity (positive β in H-ΛCDM)
- If inferred r_s < ΛCDM prediction → BAO distances are SMALLER than expected
  → Could indicate stronger early gravity (negative β)

For a fully self-consistent analysis, one should fit cosmological parameters
jointly with β using the raw D/r_d ratios, not the inferred r_s values.
"""

import numpy as np
from typing import Dict, Any, List, Optional
from scipy.optimize import minimize
from scipy.stats import chi2
from hlcdm.parameters import HLCDM_PARAMS
from ..physics.sound_horizon import sound_horizon_evolving_G, sound_horizon_lcdm, CAMB_AVAILABLE
from data.loader import DataLoader

# Planck 2018 radiation density (photons + neutrinos)
# Ω_r h² = 4.18e-5, with h = 0.6736 → Ω_r = 9.21e-5
OMEGA_R = 9.21e-5


def infer_r_s_from_bao(
    bao_data: Dict[str, Any],
    omega_m: Optional[float] = None,
    H0: Optional[float] = None
) -> Dict[str, Any]:
    """
    Infer sound horizon r_s from BAO distance measurements.
    
    IMPORTANT: This function uses a fiducial cosmology to compute distances.
    The "inferred" r_s depends on the assumed cosmological parameters!
    This is NOT an independent measurement of r_s but rather a consistency check.
    
    BAO surveys measure D_V/r_d or D_M/r_d (distance ratios).
    To extract r_s, we compute the fiducial distance D_V or D_M and divide.
    
    Parameters:
    -----------
    bao_data : dict
        BAO data dictionary with:
        - 'measurements': list of {'z', 'value', 'error'} dicts
        - 'measurement_type': 'D_V/r_d' or 'D_M/r_d' (at dataset level)
    omega_m : float, optional
        Matter density parameter. If None, uses HLCDM_PARAMS.OMEGA_M
    H0 : float, optional
        Hubble constant in km/s/Mpc. If None, converts from HLCDM_PARAMS.H0
        
    Returns:
    --------
    dict
        Results containing:
        - 'r_s_inferred': array of inferred r_s values (Mpc)
        - 'r_s_mean': weighted mean r_s (Mpc)
        - 'r_s_err': uncertainty on mean (Mpc)
        - 'n_measurements': number of measurements used
        - 'measurement_type': type of BAO measurement used
    """
    if omega_m is None:
        omega_m = HLCDM_PARAMS.OMEGA_M
    
    if H0 is None:
        H0_s_per_s = HLCDM_PARAMS.H0
        H0_km_s_Mpc = H0_s_per_s / 3.24e-20
    else:
        H0_km_s_Mpc = H0
    
    measurements = bao_data.get('measurements', [])
    if not measurements:
        return {
            'r_s_inferred': np.array([]),
            'r_s_mean': np.nan,
            'r_s_err': np.nan,
            'n_measurements': 0,
            'measurement_type': None
        }
    
    # CRITICAL FIX: Get measurement_type from DATASET level, not measurement level
    # The measurement_type is stored at the dataset level (e.g., 'D_M/r_d')
    dataset_meas_type = bao_data.get('measurement_type', 'D_V/r_d')
    
    # Speed of light in km/s
    c_km_s = HLCDM_PARAMS.C / 1000.0
    
    # Define consistent H(z) including radiation
    omega_L = 1 - omega_m - OMEGA_R  # Flat universe
    
    def H_at_z(z):
        """ΛCDM Hubble parameter including radiation."""
        return H0_km_s_Mpc * np.sqrt(
            OMEGA_R * (1 + z)**4 + 
            omega_m * (1 + z)**3 + 
            omega_L
        )
    
    r_s_inferred = []
    weights = []
    
    from scipy.integrate import quad
    
    for meas in measurements:
        z = meas['z']
        ratio_obs = meas['value']  # D_V/r_d or D_M/r_d
        ratio_err = meas['error']
        
        # Per-measurement type overrides dataset type if present
        meas_type = meas.get('measurement_type', dataset_meas_type)
        
        # Comoving distance: D_c = D_M = c ∫₀ᶻ dz' / H(z')
        def integrand(zp):
            return c_km_s / H_at_z(zp)
        
        D_M, _ = quad(integrand, 0, z, limit=1000)
        
        # Angular diameter distance D_A = D_M / (1+z)
        D_A = D_M / (1 + z)
        
        if 'D_V' in meas_type:
            # Volume-averaged distance:
            # D_V(z) = [(1+z)² D_A²(z) × cz/H(z)]^(1/3)
            # Note: Use consistent H(z) including radiation
            H_z = H_at_z(z)
            D_V = ((1 + z)**2 * D_A**2 * c_km_s * z / H_z)**(1.0/3.0)
            D_fiducial = D_V
        else:
            # Comoving distance (transverse): D_M = c ∫₀ᶻ dz'/H(z')
            D_fiducial = D_M
        
        # Infer r_s: r_s = D_fiducial / (D_fiducial/r_s)
        r_s = D_fiducial / ratio_obs
        
        # Uncertainty propagation: δr_s = r_s × (δratio / ratio)
        r_s_err = r_s * (ratio_err / ratio_obs)
        
        r_s_inferred.append(r_s)
        weights.append(1.0 / r_s_err**2)
    
    r_s_inferred = np.array(r_s_inferred)
    weights = np.array(weights)
    
    # Weighted mean
    if len(r_s_inferred) > 0:
        r_s_mean = np.sum(r_s_inferred * weights) / np.sum(weights)
        r_s_err = np.sqrt(1.0 / np.sum(weights))
    else:
        r_s_mean = np.nan
        r_s_err = np.nan
    
    return {
        'r_s_inferred': r_s_inferred,
        'r_s_mean': r_s_mean,
        'r_s_err': r_s_err,
        'n_measurements': len(r_s_inferred),
        'measurement_type': dataset_meas_type
    }


def analyze_bao_sound_horizon_joint_fit(
    datasets: Optional[List[str]] = None,
    omega_b: float = 0.049,
    omega_m: Optional[float] = None,
    H0: Optional[float] = None,
    beta_range: tuple = (-0.1, 0.5)
) -> Dict[str, Any]:
    """
    Joint fit of β to raw BAO D/r_d measurements (PROPER METHOD).
    
    This directly fits β to the observed D_M/r_d or D_V/r_d ratios without
    circular inference. For each β, we compute:
    - r_d(β) using evolving G(z) physics
    - D_M(z) or D_V(z) using ΛCDM expansion (β affects r_d, not late-time H(z))
    - Predicted ratio D/r_d(β)
    - χ² compared to observations
    
    This avoids the circularity of inferring r_s from a fiducial cosmology.
    
    Parameters:
    -----------
    datasets : list of str, optional
        BAO datasets to analyze. If None, uses ['boss_dr12', 'desi', 'eboss']
    omega_b : float
        Baryon density parameter (default: 0.049 from Planck 2018)
    omega_m : float, optional
        Matter density parameter. If None, uses HLCDM_PARAMS.OMEGA_M
    H0 : float, optional
        Hubble constant in km/s/Mpc. If None, converts from HLCDM_PARAMS.H0
    beta_range : tuple
        (min, max) range for β fitting
        
    Returns:
    --------
    dict
        Analysis results containing:
        - 'beta_fit': Best-fit β
        - 'beta_err': Uncertainty on β
        - 'chi2_lcdm': χ² for ΛCDM (β=0)
        - 'chi2_evolving': χ² for best-fit β
        - 'delta_chi2': χ² improvement
        - 'n_measurements': Number of BAO measurements
        - 'r_s_lcdm': ΛCDM r_s for reference (Mpc)
        - 'r_s_fit': Best-fit r_s(β) (Mpc)
    """
    if datasets is None:
        datasets = ['boss_dr12', 'desi', 'eboss']
    
    if omega_m is None:
        omega_m = HLCDM_PARAMS.OMEGA_M
    
    if H0 is None:
        H0_s_per_s = HLCDM_PARAMS.H0
        H0_km_s_Mpc = H0_s_per_s / 3.24e-20
    else:
        H0_km_s_Mpc = H0
    
    # Load all BAO measurements
    data_loader = DataLoader()
    all_measurements = []
    
    for dataset_name in datasets:
        try:
            bao_data = data_loader.load_bao_data(dataset_name)
            meas_type = bao_data.get('measurement_type', 'D_M/r_d')
            
            for meas in bao_data.get('measurements', []):
                all_measurements.append({
                    'z': meas['z'],
                    'value': meas['value'],
                    'error': meas['error'],
                    'type': meas.get('measurement_type', meas_type),
                    'dataset': dataset_name
                })
        except Exception as e:
            print(f"Warning: Failed to load {dataset_name}: {e}")
            continue
    
    if not all_measurements:
        r_s_lcdm = sound_horizon_lcdm(omega_b, omega_m, H0)
        return {
            'beta_fit': np.nan,
            'beta_err': np.nan,
            'chi2_lcdm': np.nan,
            'chi2_evolving': np.nan,
            'delta_chi2': np.nan,
            'n_measurements': 0,
            'r_s_lcdm': r_s_lcdm,
            'r_s_fit': np.nan
        }
    
    # Speed of light in km/s
    c_km_s = HLCDM_PARAMS.C / 1000.0
    omega_L = 1 - omega_m - OMEGA_R
    
    def H_at_z(z):
        """ΛCDM Hubble parameter including radiation."""
        return H0_km_s_Mpc * np.sqrt(
            OMEGA_R * (1 + z)**4 + 
            omega_m * (1 + z)**3 + 
            omega_L
        )
    
    def compute_D_over_rd(z, meas_type, beta):
        """
        Compute predicted D/r_d ratio for given β.
        
        Key insight: β affects r_d through early-universe physics,
        but late-time expansion H(z) remains ΛCDM at z << z_eq.
        """
        # Compute D_M(z) using ΛCDM expansion
        from scipy.integrate import quad
        D_M, _ = quad(lambda zp: c_km_s / H_at_z(zp), 0, z, limit=1000)
        
        if 'D_V' in meas_type:
            # Volume-averaged distance
            D_A = D_M / (1 + z)
            H_z = H_at_z(z)
            D = ((1 + z)**2 * D_A**2 * c_km_s * z / H_z)**(1.0/3.0)
        else:
            # Comoving transverse distance
            D = D_M
        
        # Compute r_d(β) including evolving G effects
        r_d = sound_horizon_evolving_G(omega_b, omega_m, H0_km_s_Mpc, beta)
        
        return D / r_d
    
    # χ² function: compare predicted to observed D/r_d ratios
    def chi2_func(beta):
        chi2 = 0.0
        for meas in all_measurements:
            z = meas['z']
            obs_ratio = meas['value']
            obs_err = meas['error']
            meas_type = meas['type']
            
            pred_ratio = compute_D_over_rd(z, meas_type, beta)
            chi2 += ((obs_ratio - pred_ratio) / obs_err)**2
        
        return chi2
    
    # Minimize χ²
    result = minimize(chi2_func, x0=0.1, bounds=[beta_range], method='L-BFGS-B')
    beta_fit = float(result.x[0])
    
    # Compute χ² values
    chi2_lcdm = chi2_func(0.0)
    chi2_evolving = chi2_func(beta_fit)
    delta_chi2 = chi2_lcdm - chi2_evolving
    
    # Estimate β uncertainty using Fisher matrix
    eps = 0.001
    d2chi2 = (chi2_func(beta_fit + eps) - 2*chi2_func(beta_fit) + chi2_func(beta_fit - eps)) / eps**2
    beta_err = np.sqrt(2.0 / d2chi2) if d2chi2 > 0 else np.inf
    
    # Compute r_s for reference
    r_s_lcdm = sound_horizon_lcdm(omega_b, omega_m, H0_km_s_Mpc)
    r_s_fit = sound_horizon_evolving_G(omega_b, omega_m, H0_km_s_Mpc, beta_fit)
    
    return {
        'beta_fit': float(beta_fit),
        'beta_err': float(beta_err),
        'chi2_lcdm': float(chi2_lcdm),
        'chi2_evolving': float(chi2_evolving),
        'delta_chi2': float(delta_chi2),
        'n_measurements': len(all_measurements),
        'r_s_lcdm': float(r_s_lcdm),
        'r_s_fit': float(r_s_fit)
    }


def analyze_bao_sound_horizon(
    datasets: Optional[List[str]] = None,
    omega_b: float = 0.049,
    omega_m: Optional[float] = None,
    H0: Optional[float] = None,
    beta_range: tuple = (-0.1, 0.5),
    use_joint_fit: bool = True
) -> Dict[str, Any]:
    """
    Analyze BAO sound horizon measurements and fit β parameter.
    
    This implements TEST 1 from docs/cmb_gw.md.
    
    Parameters:
    -----------
    datasets : list of str, optional
        BAO datasets to analyze. If None, uses ['boss_dr12', 'desi', 'eboss']
    omega_b : float
        Baryon density parameter (default: 0.049 from Planck 2018)
    omega_m : float, optional
        Matter density parameter. If None, uses HLCDM_PARAMS.OMEGA_M
    H0 : float, optional
        Hubble constant in km/s/Mpc. If None, converts from HLCDM_PARAMS.H0
    beta_range : tuple
        (min, max) range for β fitting
    use_joint_fit : bool
        If True (default), use proper joint fit to raw D/r_d ratios.
        If False, use legacy method (infer r_s from fiducial cosmology).
        
    Returns:
    --------
    dict
        Analysis results containing:
        - 'r_s_observed': Weighted mean r_s from BAO (Mpc) [legacy method only]
        - 'r_s_observed_err': Uncertainty (Mpc) [legacy method only]
        - 'r_s_lcdm': ΛCDM prediction (Mpc)
        - 'r_s_fit': Best-fit r_s(β) (Mpc)
        - 'beta_fit': Best-fit β
        - 'beta_err': Uncertainty on β
        - 'chi2_lcdm': χ² for ΛCDM (β=0)
        - 'chi2_evolving': χ² for best-fit β
        - 'delta_chi2': χ² improvement
        - 'n_measurements': Number of measurements used
        - 'method': 'joint_fit' or 'legacy'
    """
    if use_joint_fit:
        # Proper method: fit β directly to D/r_d ratios
        result = analyze_bao_sound_horizon_joint_fit(
            datasets, omega_b, omega_m, H0, beta_range
        )
        result['method'] = 'joint_fit'
        # Add placeholders for legacy fields
        result['r_s_observed'] = result['r_s_fit']
        result['r_s_observed_err'] = np.nan
        result['n_datasets'] = result['n_measurements']
        return result
    
    # Legacy method: infer r_s from fiducial cosmology (for comparison)
    if datasets is None:
        datasets = ['boss_dr12', 'desi', 'eboss']
    
    if omega_m is None:
        omega_m = HLCDM_PARAMS.OMEGA_M
    
    # Load BAO data
    data_loader = DataLoader()
    all_r_s_inferred = []
    all_weights = []
    
    for dataset_name in datasets:
        try:
            bao_data = data_loader.load_bao_data(dataset_name)
            inference = infer_r_s_from_bao(bao_data, omega_m, H0)
            
            if inference['n_measurements'] > 0:
                all_r_s_inferred.append(inference['r_s_mean'])
                all_weights.append(1.0 / inference['r_s_err']**2)
        except Exception:
            # Skip datasets that fail to load
            continue
    
    if not all_r_s_inferred:
        r_s_lcdm = sound_horizon_lcdm(omega_b, omega_m, H0)
        return {
            'r_s_observed': np.nan,
            'r_s_observed_err': np.nan,
            'r_s_lcdm': r_s_lcdm,
            'r_s_fit': np.nan,
            'beta_fit': np.nan,
            'beta_err': np.nan,
            'chi2_lcdm': np.nan,
            'chi2_evolving': np.nan,
            'delta_chi2': np.nan,
            'n_datasets': 0,
            'n_measurements': 0,
            'method': 'legacy'
        }
    
    # Combined weighted mean r_s
    all_r_s_inferred = np.array(all_r_s_inferred)
    all_weights = np.array(all_weights)
    r_s_observed = np.sum(all_r_s_inferred * all_weights) / np.sum(all_weights)
    r_s_observed_err = np.sqrt(1.0 / np.sum(all_weights))
    
    # ΛCDM sound horizon
    r_s_lcdm = sound_horizon_lcdm(omega_b, omega_m, H0)
    
    # Fit β to minimize χ²
    def chi2_func(beta):
        r_s_theory = sound_horizon_evolving_G(omega_b, omega_m, H0, beta)
        chi2_val = ((r_s_observed - r_s_theory) / r_s_observed_err)**2
        return chi2_val
    
    # Minimize χ²
    result = minimize(chi2_func, x0=0.0, bounds=[beta_range], method='L-BFGS-B')
    beta_fit = float(result.x[0])
    
    # Compute χ² values
    chi2_lcdm = float(chi2_func(0.0))
    chi2_evolving = float(chi2_func(beta_fit))
    delta_chi2 = chi2_lcdm - chi2_evolving
    
    # Estimate β uncertainty using Fisher matrix
    eps = 0.001
    d2chi2 = (chi2_func(beta_fit + eps) - 2*chi2_func(beta_fit) + chi2_func(beta_fit - eps)) / eps**2
    beta_err = float(np.sqrt(2.0 / d2chi2)) if d2chi2 > 0 else np.inf
    
    r_s_fit = sound_horizon_evolving_G(omega_b, omega_m, H0, beta_fit)
    
    return {
        'r_s_observed': float(r_s_observed),
        'r_s_observed_err': float(r_s_observed_err),
        'r_s_lcdm': float(r_s_lcdm),
        'r_s_fit': float(r_s_fit),
        'beta_fit': beta_fit,
        'beta_err': beta_err,
        'chi2_lcdm': chi2_lcdm,
        'chi2_evolving': chi2_evolving,
        'delta_chi2': float(delta_chi2),
        'n_datasets': len(all_r_s_inferred),
        'n_measurements': len(all_r_s_inferred),  # Approximate
        'method': 'legacy'
    }

