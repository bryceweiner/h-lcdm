"""
Cosmology primitives for the expansion-enhancement test.

Everything here works in km/s/Mpc for H and Mpc for distances — natural units
for MCMC priors on H₀. Conversions to/from the internal s⁻¹ conventions used
in ``hlcdm.parameters`` are local to functions that need them.

Design goal: Model A (ΛCDM) and Model B (framework) share ONE distance-ladder
code path. They differ only in the H(z) callable that the distance integrator
consumes, so the integrator cannot be a source of model-dependent bias.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Callable, Literal

import numpy as np

from hlcdm.cosmology import HLCDMCosmology
from hlcdm.parameters import HLCDM_PARAMS


# Recombination redshift (single source of truth: hlcdm/parameters.py:63).
Z_REC: float = float(HLCDM_PARAMS.Z_RECOMB)

# Speed of light in km/s. Distances in Mpc when dividing by H [km/s/Mpc].
C_KMS: float = HLCDM_PARAMS.C / 1000.0  # 2.998e5 km/s

# Conversion: km/s/Mpc → s⁻¹. Used when calling γ(H) which wants H in s⁻¹.
# 1 Mpc = 3.0857e22 m, so 1 km/s/Mpc = 1000 / 3.0857e22 s⁻¹ = 3.2408e-20 s⁻¹.
KMS_MPC_TO_INV_S: float = 1000.0 / 3.0857e22

EpsilonMode = Literal["constant", "qtep", "residuals"]

# Radiation density today, expressed as Ω_r h².
# Planck 2018 TT,TE,EE+lowE: photons (T_CMB = 2.7255 K) + 3.046 relativistic
# neutrinos → Ω_r h² ≈ 4.18e-5. Omitting this causes a ~0.5% bias in D_M(z_rec)
# — enough to push θ* fits off by ~50σ.
OMEGA_R_H2: float = 4.18e-5


# -----------------------------------------------------------------------------
# Hubble parameter
# -----------------------------------------------------------------------------

def H_LCDM(z: float | np.ndarray, H0: float, Om: float) -> float | np.ndarray:
    """Flat-ΛCDM H(z) in km/s/Mpc, including radiation.

    H²(z) = H₀² [Ω_m(1+z)³ + Ω_r(1+z)⁴ + Ω_Λ], with Ω_Λ = 1 − Ω_m − Ω_r and
    Ω_r = (Ω_r h²)/h². Radiation matters at z_rec (~30% of E² there) but
    vanishes rapidly for z < few.
    """
    h = H0 / 100.0
    Or = OMEGA_R_H2 / (h * h)
    OL = 1.0 - Om - Or
    return H0 * np.sqrt(Om * (1.0 + z) ** 3 + Or * (1.0 + z) ** 4 + OL)


def epsilon_profile(
    z: float | np.ndarray,
    eps: float,
    mode: EpsilonMode = "constant",
) -> float | np.ndarray:
    """ε(z) for the framework enhancement factor.

    - ``constant``: ε(z) = eps for z < Z_REC, else 0. Hard step at z_rec (the
      BAO/SNe data all sit below z_rec so each data point's integrand sees a
      uniform enhancement; only the CMB θ* integral crosses the step).
    - ``qtep``: ε(z) = eps × γ(z)/γ(0) for z < Z_REC, else 0. Uses γ(z) from
      ``hlcdm/cosmology.py:286-303`` evaluated on reference ΛCDM H(z) to keep
      the ε(z) shape deterministic across MCMC samples (only its amplitude
      ε₀ = eps varies).
    - ``residuals``: ε(z) = eps × signed-IVW Planck 2018 TT/TE/EE residual
      z-score, mapped via ℓ(z) = π D_M(z*)/(D_M(z*)−D_M(z)) (post-recombination
      horizon re-entry). Shape is precomputed from Planck data in a fixed
      reference cosmology; see ``cmb_residuals.epsilon_shape``. Outside the
      Planck ℓ-coverage the shape zero-extends. Signed by construction — may
      flip sign locally; see monotonicity caveat in that module.
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    below = z_arr < Z_REC

    if mode == "constant":
        out = np.where(below, eps, 0.0)
    elif mode == "qtep":
        ratio = _gamma_ratio(z_arr)
        out = np.where(below, eps * ratio, 0.0)
    elif mode == "residuals":
        # Lazy import to keep cosmology.py free of Planck I/O side effects.
        from .cmb_residuals import epsilon_shape
        shape = epsilon_shape(z_arr)
        out = np.where(below, eps * shape, 0.0)
    else:
        raise ValueError(f"Unknown epsilon mode: {mode!r}")

    return out if np.ndim(z) else float(out[0])


def H_framework(
    z: float | np.ndarray,
    H0: float,
    Om: float,
    eps: float,
    mode: EpsilonMode = "constant",
) -> float | np.ndarray:
    """H(z) with framework Zeno-enhancement applied for z < Z_REC."""
    base = H_LCDM(z, H0, Om)
    return base * (1.0 + epsilon_profile(z, eps, mode=mode))


def make_H_callable(
    H0: float, Om: float, eps: float = 0.0, mode: EpsilonMode = "constant"
) -> Callable[[np.ndarray], np.ndarray]:
    """Return a vectorized H(z) [km/s/Mpc] with the given cosmology + ε mode.

    Having a single callable lets the distance-ladder functions work for
    Model A (eps=0) and Model B (eps>0) via one code path.
    """
    def H(z: float | np.ndarray) -> float | np.ndarray:
        return H_framework(z, H0, Om, eps, mode=mode)

    return H


# -----------------------------------------------------------------------------
# Distance ladder — all take an H(z) callable in km/s/Mpc, return Mpc
# -----------------------------------------------------------------------------

def D_C(z: float | np.ndarray, H_func: Callable, n_steps: int = 512) -> np.ndarray:
    """Comoving distance D_C(z) = c ∫₀^z dz'/H(z') in Mpc.

    Optimized for vectorized evaluation. Builds a shared fine grid on
    [0, z_max], trapezoid-integrates ``c / H`` to get the cumulative distance,
    then interpolates to each requested z. Reduces cost from O(N_z · n_steps)
    to O(n_steps + N_z) — critical for the Pantheon+ block (N_z ≈ 1700).
    """
    z_arr = np.atleast_1d(np.asarray(z, dtype=float))
    z_max = float(z_arr.max())
    if z_max <= 0.0:
        return np.zeros_like(z_arr) if np.ndim(z) else 0.0

    # Shared grid from 0 to z_max with n_steps+1 points.
    grid = np.linspace(0.0, z_max, n_steps + 1)
    integrand = C_KMS / H_func(grid)

    # Cumulative trapezoid integration — np.cumsum with midpoint averaging.
    dz = np.diff(grid)
    increments = 0.5 * (integrand[:-1] + integrand[1:]) * dz
    cumulative = np.concatenate(([0.0], np.cumsum(increments)))

    # Interpolate the cumulative D_C onto each requested z.
    out = np.interp(z_arr, grid, cumulative)
    out = np.where(z_arr > 0.0, out, 0.0)
    return out if np.ndim(z) else float(out[0])


def D_M(z, H_func, n_steps: int = 512):
    """Transverse comoving distance (flat universe: equal to D_C)."""
    return D_C(z, H_func, n_steps=n_steps)


def D_H(z, H_func):
    """Hubble distance c/H(z) in Mpc."""
    return C_KMS / H_func(z)


def D_A(z, H_func, n_steps: int = 512):
    """Angular diameter distance D_M(z)/(1+z) in Mpc."""
    return D_M(z, H_func, n_steps=n_steps) / (1.0 + np.asarray(z))


def D_L(z, H_func, n_steps: int = 512):
    """Luminosity distance (1+z)·D_M(z) in Mpc."""
    return D_M(z, H_func, n_steps=n_steps) * (1.0 + np.asarray(z))


def D_V(z, H_func, n_steps: int = 512):
    """BAO volume-averaged distance [z·D_M²·D_H]^(1/3) in Mpc."""
    dm = D_M(z, H_func, n_steps=n_steps)
    dh = D_H(z, H_func)
    return np.cbrt(np.asarray(z) * dm * dm * dh)


def theta_star(H_func, r_d: float, z_rec: float = Z_REC, n_steps: int = 2048) -> float:
    """CMB acoustic-peak angular scale θ* = r_d / D_M(z_rec).

    Flat universe: θ* = r_s_comoving × (1+z*) / D_A_physical(z*) (1+z*)
                      = r_s_comoving / D_M(z*).

    Uses ``scipy.integrate.quad`` for the high-z comoving-distance integral
    because linear-spaced Simpson converges slowly between z=0 (where 1/H
    is large) and z=z_rec (where it is orders of magnitude smaller).
    ``n_steps`` is retained for signature compatibility but unused.
    """
    from scipy.integrate import quad

    def integrand(zp):
        return 1.0 / float(np.atleast_1d(H_func(zp))[0])

    # Break the integration at z=5 so quad can spend effort where the
    # integrand varies most (low z) without wasting samples on the smooth tail.
    lo, _ = quad(integrand, 0.0, 5.0, limit=200, epsrel=1e-8)
    hi, _ = quad(integrand, 5.0, z_rec, limit=200, epsrel=1e-8)
    dm = C_KMS * (lo + hi)
    return r_d / dm


# -----------------------------------------------------------------------------
# γ-ratio helper for ε_QTEP(z)
# -----------------------------------------------------------------------------

# Reference cosmology fixing the ε(z) SHAPE (amplitude is eps, a free param).
# Using Planck 2018 mean H0 / Ω_m makes ε_QTEP(z) a deterministic function of
# z — independent of the MCMC sample — so fitted ε₀ is interpretable.
_REF_H0_KMS_MPC: float = 67.4
_REF_OMEGA_M: float = 0.315


@lru_cache(maxsize=1)
def _gamma0() -> float:
    """γ(z=0) [s⁻¹] evaluated on the reference ΛCDM cosmology."""
    H0_inv_s = _REF_H0_KMS_MPC * KMS_MPC_TO_INV_S
    return float(HLCDMCosmology.gamma_theoretical(H0_inv_s))


def _gamma_ratio(z: np.ndarray) -> np.ndarray:
    """γ(z)/γ(0) on the reference cosmology. Vectorized.

    ``HLCDMCosmology.gamma_theoretical`` is pure numpy, so we can feed it an
    array of H values in one shot — no python loop needed.
    """
    H_kms = H_LCDM(z, _REF_H0_KMS_MPC, _REF_OMEGA_M)
    H_inv_s = np.asarray(H_kms) * KMS_MPC_TO_INV_S
    return HLCDMCosmology.gamma_theoretical(H_inv_s) / _gamma0()


# -----------------------------------------------------------------------------
# Distance modulus for SNe
# -----------------------------------------------------------------------------

def mu_model(z: np.ndarray, H_func: Callable, n_steps: int = 512) -> np.ndarray:
    """Distance modulus μ(z) = 5 log₁₀(D_L(z) [Mpc]) + 25.

    The ``+25`` converts 10 pc → Mpc baseline in the usual SN-cosmology form.
    Pantheon+ likelihood then analytically marginalizes the absolute-magnitude
    nuisance M, so only relative μ(z) matters — but we still add the +25 so
    intermediate diagnostics have correct absolute magnitudes.
    """
    dl = D_L(z, H_func, n_steps=n_steps)
    return 5.0 * np.log10(np.asarray(dl)) + 25.0
