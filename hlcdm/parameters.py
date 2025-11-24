"""
HLCDM Parameters - Single Point of Truth
========================================

All physical constants, cosmological parameters, and configuration values
used throughout the Holographic Lambda Model (H-ΛCDM) analysis.

This module serves as the single source of truth for all parameters and constants,
ensuring consistency across all pipelines and calculations.

References:
    - Paper Methods section, line ~193 for physical constants
    - Paper line ~75 for recombination parameters
    - Paper line ~117 for QTEP ratio
"""

import numpy as np
from typing import Dict, List, Any


class HLCDMParameters:
    """
    Holographic Lambda Model Parameters - Single Point of Truth

    This class contains all physical constants, cosmological parameters,
    and configuration values used throughout the H-ΛCDM analysis.

    All parameters are immutable and serve as the authoritative source
    for calculations across all pipelines.
    """

    def __init__(self):
        """Initialize all parameters. All values are constants."""

        # ============================================================================
        # PHYSICAL CONSTANTS
        # ============================================================================
        # Reference: Paper Methods section, line ~193

        self.C = 2.998e8              # m/s - speed of light
        self.C_LIGHT = self.C          # Alias for compatibility
        self.HBAR = 1.055e-34         # J·s - reduced Planck constant
        self.G = 6.674e-11            # m³ kg⁻¹ s⁻² - gravitational constant
        self.G_NEWTON = self.G        # Alias for compatibility
        self.SIGMA_T = 6.65e-29       # m² - Thomson scattering cross section


        # ============================================================================
        # COSMOLOGICAL PARAMETERS
        # ============================================================================

        # Hubble constant and matter densities
        self.H0 = 2.18e-18            # s⁻¹ - Hubble constant (H0 = 67 km/s/Mpc)
        self.OMEGA_M = 0.315          # Matter density parameter
        self.OMEGA_LAMBDA = 0.685     # Dark energy density parameter


        # ============================================================================
        # RECOMBINATION PARAMETERS
        # ============================================================================
        # Reference: Paper line ~75

        self.Z_RECOMB = 1100          # Recombination redshift

        # Hubble parameter at z=1100 (matter-dominated epoch)
        # H(z) = H0 × sqrt(Ω_m(1+z)³ + Ω_Λ)
        # At z=1100: H ≈ H0 × sqrt(0.315 × 1101³) ≈ H0 × 2.02×10⁴
        self.H_RECOMB = self.H0 * np.sqrt(self.OMEGA_M * (self.Z_RECOMB + 1)**3 + self.OMEGA_LAMBDA)  # ≈ 4.4×10⁻¹⁴ s⁻¹


        # ============================================================================
        # COSMOLOGICAL EPOCHS
        # ============================================================================

        self.Z_DRAG = 1059            # Drag epoch redshift (baryon-photon decoupling)
        self.Z_EQ = 3402              # Matter-radiation equality redshift


        # ============================================================================
        # OBSERVATIONAL VALUES
        # ============================================================================

        # Planck 2018
        self.SIGMA_8_PLANCK = 0.811   # σ8 amplitude
        self.S8_PLANCK = 0.834        # S8 = σ8(Ωm/0.3)^0.5 (±0.016)
        self.OMEGA_M_PLANCK = 0.315   # Matter density (±0.007)

        # DES Year 3
        self.S8_DES_Y3 = 0.776        # S8 value (±0.017)
        self.OMEGA_M_DES_Y3 = 0.298   # Matter density (+0.007/-0.007)
        self.RS_DES_Y3 = 143.6        # BAO scale in Mpc (±1.7)

        # BOSS DR12
        self.RS_BOSS = 147.47         # BAO scale in Mpc (±0.59)


        # ============================================================================
        # BAO DATA
        # ============================================================================

        # BOSS DR12 BAO data (REAL measurements)
        # Reference: Alam et al. MNRAS 470, 2617 (2017), arXiv:1607.03155
        # Table 2: Consensus BAO distance measurements
        # Using D_M/r_d (transverse direction) for cosmological tests

        # BOSS DR12 consensus measurements
        self.BOSS_DR12_BAO_DATA = [
            # Effective z, D_M(z)/r_d, error (from combining galaxy samples)
            {'z': 0.38, 'value': 10.27, 'error': 0.15},  # LOWZ sample
            {'z': 0.51, 'value': 13.37, 'error': 0.15},  # CMASS z1
            {'z': 0.61, 'value': 15.23, 'error': 0.17},  # CMASS z2
        ]

        # Published correlation matrix (arXiv:1607.03155)
        # Bins correlated due to overlapping redshift windows
        self.BOSS_DR12_CORRELATION = np.array([
            [1.00, 0.61, 0.49],
            [0.61, 1.00, 0.71],
            [0.49, 0.71, 1.00]
        ])

        # Compatibility: Use BOSS DR12 as default BAO data
        self.DES_Y3_BAO_DATA = self.BOSS_DR12_BAO_DATA


        # ============================================================================
        # MATTER DENSITY DATA
        # ============================================================================

        self.MATTER_DENSITY_DATA = {
            'surveys': ['DES Y1', 'DES Y3'],
            'omega_m_measured': [0.267, 0.298],
            'omega_m_hu': [0.268, 0.298],
            'measured_errors': [[0.017, 0.030], [0.007, 0.007]],
            'hu_errors': [[0.018, 0.018], [0.007, 0.007]]
        }


        # ============================================================================
        # QUANTUM-THERMODYNAMIC ENTROPY PARTITION (QTEP)
        # ============================================================================
        # Reference: Paper line ~117

        # Coherent entropy: S_coh = ln(2) ≈ 0.693 nats
        self.S_COH = np.log(2)

        # Decoherent entropy: S_decoh = ln(2) - 1 ≈ -0.307 nats
        self.S_DECOH = np.log(2) - 1

        # QTEP ratio: S_coh / |S_decoh| = ln(2) / (1 - ln(2)) ≈ 2.257
        self.QTEP_RATIO = np.log(2) / (1 - np.log(2))


        # ============================================================================
        # PLANCK UNITS
        # ============================================================================
        # Reference: Supplementary Note 2, Step 2

        self.T_PLANCK = np.sqrt(self.HBAR * self.G / self.C**5)   # Planck time ≈ 5.39×10⁻⁴⁴ s
        self.M_PLANCK = np.sqrt(self.HBAR * self.C / self.G)      # Planck mass ≈ 2.18×10⁻⁸ kg
        self.RHO_PLANCK = self.C**5 / (self.HBAR * self.G**2)     # Planck density ≈ 5.16×10⁹⁶ kg/m³
        self.L_PLANCK = np.sqrt(self.HBAR * self.G / self.C**3)   # Planck length ≈ 1.62×10⁻³⁵ m


        # ============================================================================
        # COSMOLOGICAL CONSTANT
        # ============================================================================
        # Reference: Paper Introduction, line ~15

        self.LAMBDA_OBS = 1.1e-52     # m⁻² - observed cosmological constant


        # ============================================================================
        # OUTPUT CONFIGURATION
        # ============================================================================

        self.OUTPUT_LOG = 'analysis_unified.log'
        self.OUTPUT_JSON = 'analysis_unified.json'
        self.OUTPUT_FIGURE = 'phase_transitions_analysis.pdf'


        # ============================================================================
        # PAPER LINE REFERENCES
        # ============================================================================
        # Map formulas to specific lines in phase_transitions_discovery.tex

        self.PAPER_REFERENCES = {
            'gamma_theoretical': 'gamma_theoretical_derivation.tex line 29, 159',  # γ = H/ln(πc⁵/GℏH²)
            'qtep_ratio': 'line 167',                  # S_coh/|S_decoh| = 2.257
            'quantization_condition': 'line 109',      # γℓ/H = nπ/2
            'expansion_factor': 'line 117',            # f = γ_theory/γ_obs
            'vacuum_energy_scaling': 'line 115',       # ρ_Λ,eff ∝ [γ_theory/γ_obs]²
            'lambda_formula': 'line 134',              # Λ = 8πG ρ_P (γ×t_P)² × QTEP
            'physical_scale': 'line 212',              # Scale conversion at z_recomb
            'harmonic_ratio': 'line 99',               # ℓ_{n+1}/ℓ_n with corrections
        }


        # ============================================================================
        # VERSION INFORMATION
        # ============================================================================

        self.__version__ = "1.0.0"
        self.__author__ = "Bryce Weiner"
        self.__paper__ = "Holographic Lambda Model (H-ΛCDM) Analysis"


    # ============================================================================
    # PROPERTIES FOR EASY ACCESS
    # ============================================================================

    @property
    def version(self) -> str:
        """Get version information."""
        return self.__version__

    @property
    def author(self) -> str:
        """Get author information."""
        return self.__author__

    @property
    def paper(self) -> str:
        """Get paper title."""
        return self.__paper__

    def get_hubble_at_redshift(self, z: float) -> float:
        """
        Calculate Hubble parameter at given redshift.

        H(z) = H0 × sqrt(Ω_m(1+z)³ + Ω_Λ)

        Parameters:
            z (float): Redshift

        Returns:
            float: Hubble parameter in s⁻¹
        """
        return self.H0 * np.sqrt(self.OMEGA_M * (1 + z)**3 + self.OMEGA_LAMBDA)

    def get_all_constants(self) -> Dict[str, Any]:
        """
        Get all physical constants as a dictionary.

        Returns:
            dict: All constants
        """
        constants = {}
        for attr in dir(self):
            if not attr.startswith('_') and not callable(getattr(self, attr)):
                constants[attr] = getattr(self, attr)
        return constants


# ============================================================================
# GLOBAL INSTANCE - SINGLE POINT OF TRUTH
# ============================================================================

# Create the single global instance
HLCDM_PARAMS = HLCDMParameters()

# For backward compatibility, expose constants directly at module level
C = HLCDM_PARAMS.C
C_LIGHT = HLCDM_PARAMS.C_LIGHT
HBAR = HLCDM_PARAMS.HBAR
G = HLCDM_PARAMS.G
G_NEWTON = HLCDM_PARAMS.G_NEWTON
H0 = HLCDM_PARAMS.H0
OMEGA_M = HLCDM_PARAMS.OMEGA_M
OMEGA_LAMBDA = HLCDM_PARAMS.OMEGA_LAMBDA
SIGMA_T = HLCDM_PARAMS.SIGMA_T
Z_RECOMB = HLCDM_PARAMS.Z_RECOMB
H_RECOMB = HLCDM_PARAMS.H_RECOMB
Z_DRAG = HLCDM_PARAMS.Z_DRAG
Z_EQ = HLCDM_PARAMS.Z_EQ
SIGMA_8_PLANCK = HLCDM_PARAMS.SIGMA_8_PLANCK
S8_PLANCK = HLCDM_PARAMS.S8_PLANCK
OMEGA_M_PLANCK = HLCDM_PARAMS.OMEGA_M_PLANCK
S8_DES_Y3 = HLCDM_PARAMS.S8_DES_Y3
OMEGA_M_DES_Y3 = HLCDM_PARAMS.OMEGA_M_DES_Y3
RS_DES_Y3 = HLCDM_PARAMS.RS_DES_Y3
RS_BOSS = HLCDM_PARAMS.RS_BOSS
BOSS_DR12_BAO_DATA = HLCDM_PARAMS.BOSS_DR12_BAO_DATA
BOSS_DR12_CORRELATION = HLCDM_PARAMS.BOSS_DR12_CORRELATION
DES_Y3_BAO_DATA = HLCDM_PARAMS.DES_Y3_BAO_DATA
MATTER_DENSITY_DATA = HLCDM_PARAMS.MATTER_DENSITY_DATA
S_COH = HLCDM_PARAMS.S_COH
S_DECOH = HLCDM_PARAMS.S_DECOH
QTEP_RATIO = HLCDM_PARAMS.QTEP_RATIO
T_PLANCK = HLCDM_PARAMS.T_PLANCK
M_PLANCK = HLCDM_PARAMS.M_PLANCK
RHO_PLANCK = HLCDM_PARAMS.RHO_PLANCK
L_PLANCK = HLCDM_PARAMS.L_PLANCK
LAMBDA_OBS = HLCDM_PARAMS.LAMBDA_OBS
OUTPUT_LOG = HLCDM_PARAMS.OUTPUT_LOG
OUTPUT_JSON = HLCDM_PARAMS.OUTPUT_JSON
OUTPUT_FIGURE = HLCDM_PARAMS.OUTPUT_FIGURE
PAPER_REFERENCES = HLCDM_PARAMS.PAPER_REFERENCES
__version__ = HLCDM_PARAMS.version
__author__ = HLCDM_PARAMS.author
__paper__ = HLCDM_PARAMS.paper
