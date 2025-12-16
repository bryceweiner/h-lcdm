"""
Zeno ISW Calculation
====================

Calculate ISW contribution from Zeno regime unwinding at recombination,
following the formalism in bao_resolution_qit.tex (Weiner 2025).

The quantum Zeno effect at recombination creates a coherence-protected regime
that unwinds over timescale τ_Zeno ~ 1.3×10^5 yr. As M(z) = Γ_T(z)/H(z) declines,
deferred decoherent entropy precipitates, creating time-varying potentials
that generate an ISW contribution distinct from standard late-time ISW.

References:
- Weiner (2025): "EDE-like Resolution of BAO via QIT" (bao_resolution_qit.tex)
- Sachs & Wolfe (1967): ApJ 147, 73 (ISW formalism)
- Planck Collaboration (2016): A&A 594, A21 (ISW measurements)
"""

import numpy as np
from typing import Dict, Any, Tuple, Optional
from scipy import integrate, interpolate
from scipy.special import erf
import logging

logger = logging.getLogger(__name__)

# Physical constants
c = 299792.458  # km/s
T_CMB = 2.725  # K (CMB temperature today)

# QTEP parameters from bao_resolution_qit.tex
QTEP_RATIO = np.log(2) / (1 - np.log(2))  # ≈ 2.257
S_COH = np.log(2)  # Coherent entropy (nats)
S_DECOH = np.log(2) - 1  # Decoherent entropy (nats), ≈ -0.307

# Recombination parameters (Planck 2018)
Z_RECOMB = 1089  # Recombination redshift
Z_STAR = 1089    # Last scattering surface
DELTA_Z_RECOMB = 200  # Width of recombination interval

# Zeno parameters from §2.4 of BAO paper
ZENO_DURATION_YR = 1.3e5  # Duration of strong Zeno regime (years)
N_ZENO_SCATTERINGS = 1.2e7  # Thomson scatterings per baryon during Zeno interval


def calculate_hubble_rate(z: float, 
                          H0: float = 67.4,  # km/s/Mpc (Planck 2018)
                          Om0: float = 0.315,
                          OL0: float = 0.685,
                          Or0: float = 9.3e-5) -> float:
    """
    Calculate Hubble rate H(z) in km/s/Mpc.
    
    Parameters:
        z: Redshift
        H0: Hubble constant today
        Om0: Matter density parameter
        OL0: Dark energy density parameter
        Or0: Radiation density parameter
        
    Returns:
        H(z) in km/s/Mpc
    """
    return H0 * np.sqrt(Om0 * (1+z)**3 + Or0 * (1+z)**4 + OL0)


def calculate_thomson_rate(z: float,
                           H0: float = 67.4,
                           Om0: float = 0.315,
                           Ob0: float = 0.0493) -> float:
    """
    Calculate Thomson scattering rate Γ_T(z) during recombination.
    
    Following standard recombination theory (Peebles 1968, Seager+ 1999).
    
    Parameters:
        z: Redshift
        H0: Hubble constant
        Om0: Total matter density
        Ob0: Baryon density parameter
        
    Returns:
        Γ_T(z) in s^-1
    """
    # Electron density: n_e = n_e0 × (1+z)^3 × X_e(z)
    # where n_e0 = Ω_b ρ_c / m_p
    
    # Critical density today in g/cm^3
    rho_crit_0 = 1.878e-29 * (H0/100)**2  # g/cm^3
    
    # Baryon density today
    rho_b_0 = Ob0 * rho_crit_0  # g/cm^3
    
    # Number density of baryons today
    m_p = 1.673e-24  # Proton mass in g
    n_b_0 = rho_b_0 / m_p  # cm^-3
    
    # Ionization fraction during recombination (Peebles approximation)
    # X_e ≈ 1 for z > z_rec, drops sharply at z ~ z_rec
    if z > Z_RECOMB + DELTA_Z_RECOMB/2:
        X_e = 1.0
    elif z < Z_RECOMB - DELTA_Z_RECOMB/2:
        X_e = 1e-4  # Residual ionization
    else:
        # Smooth transition (visibility function shape)
        delta_z = (z - Z_RECOMB) / (DELTA_Z_RECOMB/2)
        X_e = 0.5 * (1 + erf(-delta_z))  # Smooth decline
    
    # Electron density at redshift z
    n_e = n_b_0 * (1 + z)**3 * X_e  # cm^-3
    
    # Thomson cross section
    sigma_T = 6.652e-25  # cm^2
    
    # Thomson rate
    c_cm_s = 2.998e10  # cm/s
    Gamma_T = n_e * sigma_T * c_cm_s  # s^-1
    
    return Gamma_T


def calculate_monitoring_strength(z: float) -> float:
    """
    Calculate dimensionless monitoring strength M(z) = Γ_T(z)/H(z).
    
    This quantifies the Zeno regime strength. When M >> 1, the system
    is in the quantum Zeno regime.
    
    Parameters:
        z: Redshift
        
    Returns:
        M(z) (dimensionless)
    """
    H_z = calculate_hubble_rate(z)
    Gamma_T = calculate_thomson_rate(z)
    
    # Convert H to s^-1 (H is in km/s/Mpc)
    Mpc_to_km = 3.086e19  # km
    H_z_SI = H_z / Mpc_to_km  # s^-1
    
    return Gamma_T / H_z_SI


def calculate_holographic_rate(z: float) -> float:
    """
    Calculate holographic information processing rate γ(z).
    
    From §2.2 of BAO paper:
    γ = H(z) / ln(N_P) where N_P ~ π c^5 / (G ℏ H^2)
    
    Parameters:
        z: Redshift
        
    Returns:
        γ(z) in s^-1
    """
    H_z = calculate_hubble_rate(z)
    
    # Convert to SI
    Mpc_to_km = 3.086e19
    H_z_SI = H_z / Mpc_to_km  # s^-1
    
    # Holographic entropy (very large number ~ 10^113)
    # N_P ~ π c^5 / (G ℏ H^2)
    # We use ln(N_P) ~ 260 at z ~ 1089 (from BAO paper)
    
    # More precisely: ln(N_P) = ln(π) + 5ln(c) - ln(G) - ln(ℏ) - 2ln(H)
    # For numerical stability, use the fact that ln(N_P) ~ 260 ± 5 near recombination
    ln_N_P = 260.0
    
    gamma = H_z_SI / ln_N_P
    
    return gamma


def calculate_visibility_function(z: float,
                                  z_star: float = Z_STAR,
                                  delta_z: float = DELTA_Z_RECOMB) -> float:
    """
    Calculate recombination visibility function g(z) = -dτ/dz e^(-τ).
    
    Approximated as Gaussian centered at z_star with width delta_z.
    
    Parameters:
        z: Redshift
        z_star: Peak of visibility function
        delta_z: Width of recombination
        
    Returns:
        g(z) (normalized)
    """
    # Gaussian approximation
    sigma = delta_z / 2.355  # FWHM to sigma
    g_z = (1 / (sigma * np.sqrt(2*np.pi))) * np.exp(-0.5 * ((z - z_star)/sigma)**2)
    
    return g_z


def calculate_zeno_isw_contribution(z_obs: float = 0,
                                    ell: float = 18,
                                    alpha: float = -5.7,
                                    z_range: Optional[Tuple[float, float]] = None
                                    ) -> Dict[str, Any]:
    """
    Calculate ISW contribution from Zeno regime unwinding.
    
    As M(z) = Γ_T/H declines through recombination, deferred decoherent
    entropy precipitates, creating time-varying gravitational potentials.
    This generates an ISW contribution:
    
    (δT/T)_ISW,Zeno = -2 ∫ (dΦ/dη) e^(-τ) dη
    
    Where Φ(z) evolves due to entropy precipitation following QTEP dynamics.
    
    Parameters:
        z_obs: Observation redshift (0 for today)
        ell: Multipole moment (for angular scale)
        alpha: Coherent acoustic enhancement coefficient
        z_range: Redshift range for integration (default: Zeno interval)
        
    Returns:
        Dictionary with ISW contribution and diagnostics
    """
    if z_range is None:
        # Zeno regime: z_star ± Δz/2
        z_min = Z_STAR - DELTA_Z_RECOMB
        z_max = Z_STAR + DELTA_Z_RECOMB/2
    else:
        z_min, z_max = z_range
    
    # Redshift grid
    z_grid = np.linspace(z_max, z_min, 500)
    
    # Calculate monitoring strength M(z) on grid
    M_grid = np.array([calculate_monitoring_strength(z) for z in z_grid])
    
    # Calculate dM/dz (rate of Zeno regime decline)
    dM_dz = np.gradient(M_grid, z_grid)
    
    # Visibility function g(z)
    g_grid = np.array([calculate_visibility_function(z) for z in z_grid])
    
    # Calculate holographic rate γ(z)
    gamma_grid = np.array([calculate_holographic_rate(z) for z in z_grid])
    
    # Potential evolution from entropy precipitation during Zeno unwinding
    # 
    # Physical reasoning:
    # 1. At recombination, primordial potential: Φ ~ 10^-5 (Sachs-Wolfe)
    # 2. As Zeno regime unwinds (M drops from ~10^9 to ~1), deferred entropy precipitates
    # 3. This creates time-varying stress-energy → dΦ/dt ≠ 0 → ISW effect
    #
    # Dimensional analysis:
    # - Φ: dimensionless, ~10^-5
    # - dΦ/dη: units of Φ per conformal time, ~Φ×H
    # - ISW: (δT/T)_ISW ~ ∫ (dΦ/dη) dη ~ Φ × (Δη/η) over Zeno interval
    #
    # Scale estimate:
    # - Δη/η during Zeno interval ~ Δt_Zeno × H ~ 10^5 yr × 10^-14 s^-1 ~ 3×10^-2
    # - So: (δT/T)_ISW ~ Φ × (Δη/η) × (entropy fraction)
    #                   ~ 10^-5 × 3×10^-2 × 0.13 ~ 4×10^-8
    # - This is subdominant to primary (as expected for ISW at ℓ~18)
    
    H_grid = np.array([calculate_hubble_rate(z) for z in z_grid])
    
    # Convert H to s^-1
    Mpc_to_km = 3.086e19
    H_grid_SI = H_grid / Mpc_to_km
    
    # Primordial potential amplitude (Sachs-Wolfe normalization)
    # From CMB: δT/T ~ Φ/3, so Φ ~ 3 × 10^-5 ~ 10^-4 for typical fluctuations
    Phi_primordial = 3.0 * 2.6e-5  # ~ 10^-4
    
    # Potential evolution rate: dΦ/dη
    # The Zeno coherence creates an effective "freezing" of potential decay
    # As M drops, this freezing releases → potential evolves
    #
    # Fractional change in Φ due to entropy precipitation:
    # ΔΦ/Φ ~ (ΔS_decoh / S_total) × (α × γ/H)
    #       ~ |S_decoh|/S_total × coherent_enhancement
    #       ~ 0.136 × |α × γ/H|
    
    # Normalized M decline rate (dimensionless)
    M_normalized = M_grid / np.max(M_grid)
    dM_norm_dz = np.gradient(M_normalized, z_grid)
    
    # Potential derivative with respect to redshift
    # dΦ/dz ~ Φ × (dM_norm/dz) × (entropy_fraction) × (coherent_factor)
    entropy_fraction = abs(S_DECOH) / (S_COH + abs(S_DECOH))  # ~ 0.136
    coherent_factor = abs(alpha) * QTEP_RATIO  # ~ 5.7 × 2.257 ~ 13
    
    dPhi_dz = Phi_primordial * dM_norm_dz * entropy_fraction * coherent_factor
    
    # Convert to conformal time derivative: dΦ/dη = (dΦ/dz) × (dz/dη)
    # where dz/dη = -(1+z) H(z)
    dz_deta = -(1 + z_grid) * H_grid_SI
    dPhi_deta = dPhi_dz * dz_deta
    
    # ISW integrand: 2 × (dΦ/dη) × g(z) × (visibility suppression)
    # The factor of 2 is standard ISW convention
    # Visibility suppression: e^(-τ) ≈ 1 during recombination (optically thick → thin)
    
    integrand = -2.0 * dPhi_deta * g_grid
    
    # Integrate over conformal time
    # dη = -dz / [(1+z) H(z)]
    d_eta = -1.0 / ((1 + z_grid) * H_grid_SI)
    
    # Numerical integration (trapezoidal)
    delta_T_over_T_ISW_Zeno = np.trapz(integrand * d_eta, z_grid)
    
    # Diagnostics
    M_peak = np.max(M_grid)
    M_final = M_grid[-1]
    gamma_rec = calculate_holographic_rate(Z_STAR)
    
    # Total entropy precipitated (fraction of total)
    # ΔS_precip / S_total ~ ∫ (dM/dt) dt over Zeno interval
    # This should be ~ |S_decoh| / S_total ≈ 0.307/2.257 ≈ 0.136
    
    return {
        'delta_T_over_T_ISW_Zeno': float(delta_T_over_T_ISW_Zeno),
        'z_range': (z_min, z_max),
        'M_peak': float(M_peak),
        'M_final': float(M_final),
        'gamma_recomb': float(gamma_rec),
        'alpha': alpha,
        'QTEP_ratio': QTEP_RATIO,
        'mechanism': 'Zeno_unwinding',
        'reference': 'Weiner (2025) bao_resolution_qit.tex §2.4-2.5'
    }


def calculate_late_time_isw(z_obs: float = 0,
                            ell: float = 18,
                            Om0: float = 0.315,
                            OL0: float = 0.685) -> float:
    """
    Calculate standard late-time ISW from Λ-dominated era (z ~ 0.1-10).
    
    This is the conventional ISW effect from gravitational potentials
    decaying as dark energy becomes dominant.
    
    Simplified analytical approximation for large-scale modes (ℓ < 50).
    
    Parameters:
        z_obs: Observation redshift
        ell: Multipole moment
        Om0: Matter density
        OL0: Dark energy density
        
    Returns:
        δT/T from late-time ISW
    """
    # Approximate ISW contribution for Λ-dominated era
    # At large scales (ℓ ~ 10-50), late-time ISW contributes ~ 10^-6 level
    # for Cold Spot at ℓ ~ 18
    
    # Analytical approximation (Kofman & Starobinskii 1985):
    # (δT/T)_ISW ≈ -2 Φ_initial × [Ω_Λ / (Ω_m + Ω_Λ)]^(1/2) × f(ℓ)
    
    # For ℓ ~ 18 (Cold Spot scale), ISW is subdominant to primary
    # Typical magnitude: ~ 10^-6 to 10^-5
    
    # Use scaling: ISW ~ 10^-6 × (1 + OL0/Om0) for ℓ ~ 20
    ISW_amplitude = 1e-6 * (1 + OL0/Om0) * (20.0/ell)**2
    
    # Sign: potentials decay → negative contribution to cold spots
    delta_T_over_T_ISW_late = -ISW_amplitude
    
    return delta_T_over_T_ISW_late


def calculate_full_temperature_signal(delta_eta_over_eta_primary: float,
                                     ell: float = 18,
                                     alpha: float = -5.7,
                                     include_zeno_isw: bool = True,
                                     include_late_isw: bool = True) -> Dict[str, Any]:
    """
    Calculate full CMB temperature signal including all contributions.
    
    Total signal:
    δT/T_total = δT/T_primary + δT/T_ISW,Zeno + δT/T_ISW,late + ...
    
    Where:
    - Primary: Sachs-Wolfe at recombination (δT/T = Φ/3 = δη/η from QTEP)
    - Zeno ISW: From coherence unwinding
    - Late ISW: From Λ-dominated era
    
    Parameters:
        delta_eta_over_eta_primary: Primary efficiency variation δη/η at z~1089
        ell: Multipole moment (angular scale)
        alpha: Coherent acoustic enhancement coefficient
        include_zeno_isw: Include Zeno unwinding ISW
        include_late_isw: Include standard late-time ISW
        
    Returns:
        Dictionary with total signal and components
    """
    # Primary Sachs-Wolfe: δT/T = δη/η (QTEP prediction)
    delta_T_over_T_primary = delta_eta_over_eta_primary
    
    # Zeno ISW contribution
    if include_zeno_isw:
        zeno_result = calculate_zeno_isw_contribution(ell=ell, alpha=alpha)
        delta_T_over_T_ISW_Zeno = zeno_result['delta_T_over_T_ISW_Zeno']
        zeno_diagnostics = zeno_result
    else:
        delta_T_over_T_ISW_Zeno = 0.0
        zeno_diagnostics = {}
    
    # Late-time ISW contribution
    if include_late_isw:
        delta_T_over_T_ISW_late = calculate_late_time_isw(ell=ell)
    else:
        delta_T_over_T_ISW_late = 0.0
    
    # Total signal
    delta_T_over_T_total = (delta_T_over_T_primary + 
                           delta_T_over_T_ISW_Zeno + 
                           delta_T_over_T_ISW_late)
    
    return {
        'delta_T_over_T_total': float(delta_T_over_T_total),
        'delta_T_over_T_primary': float(delta_T_over_T_primary),
        'delta_T_over_T_ISW_Zeno': float(delta_T_over_T_ISW_Zeno),
        'delta_T_over_T_ISW_late': float(delta_T_over_T_ISW_late),
        'ell': ell,
        'alpha': alpha,
        'zeno_diagnostics': zeno_diagnostics,
        'method': 'full_signal_chain',
        'epoch_matching': 'recombination_to_today',
        'reference': 'Weiner (2025) + Sachs & Wolfe (1967)'
    }

