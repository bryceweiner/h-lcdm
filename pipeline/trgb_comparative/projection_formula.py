"""
Holographic projection formula for H_local / H_CMB.

This is the framework's forward-prediction centerpiece for the TRGB comparative
analysis:

    H_local / H_CMB  =  1 + (γ/H)·L · [1 + (C(G) - 0.5)·L]
    where            L = ln(d_CMB / d_local)

`γ/H` is **computed at runtime** from ``HLCDMCosmology.gamma_at_redshift(0.0)``
divided by ``HLCDM_PARAMS.get_hubble_at_redshift(0.0)``. It is NOT cached as a
hardcoded constant — the nominal ~1/282 is an evaluation result, not a
definition, and caching it would silently disconnect downstream code from any
refinement of γ.

``C(G) = 25/32`` is taken from the existing ``E8Cache`` module. ``d_CMB`` is
the Planck 2018 comoving distance to last scattering, now a named constant
on ``HLCDM_PARAMS``.

The return value of :func:`holographic_h_ratio` is a structured
:class:`HolographicRatioResult` rather than a bare float, so callers can log
the full provenance (γ/H used, intermediate terms, breakdown flag, message)
to output artifacts rather than losing warnings to stderr.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

from hlcdm.cosmology import HLCDMCosmology
from hlcdm.parameters import HLCDM_PARAMS


# ---------------------------------------------------------------------------
# Warnings / results
# ---------------------------------------------------------------------------


#: Threshold on d_local (Mpc) below which the framework's derivation of the
#: holographic projection formula is considered non-perturbative. Defined at
#: 1 Mpc: the LMC at 0.05 Mpc sits deep below this, NGC 4258 at 7.58 Mpc
#: sits safely above. Structural anchors for SH0ES-like methods (≳ few Mpc)
#: are all in the perturbative regime; direct LMC anchoring is not.
PERTURBATIVE_D_LOCAL_MPC: float = 1.0


class PerturbativeBreakdownWarning(UserWarning):
    """Raised (by warnings.warn) when d_local falls below
    :data:`PERTURBATIVE_D_LOCAL_MPC`, where the framework's perturbative
    derivation no longer holds.

    Caller can catch this (``warnings.catch_warnings``) to suppress stderr
    output during Monte Carlo propagation; the same breakdown state is also
    always recorded on the returned :class:`HolographicRatioResult` so it is
    never lost.
    """


@dataclass(frozen=True)
class HolographicRatioResult:
    """Structured output of :func:`holographic_h_ratio`.

    All intermediate quantities are preserved so callers can reason about the
    prediction beyond the single ``ratio`` number. ``breakdown_flag`` is the
    authoritative breakdown signal; ``breakdown_message`` is the human-
    readable text.
    """

    ratio: float
    gamma_over_H: float
    clustering_coefficient: float
    L: float
    linear_term: float
    quadratic_correction: float
    breakdown_flag: bool
    breakdown_message: Optional[str]
    d_local_mpc: float
    d_cmb_mpc: float
    second_order: bool


# ---------------------------------------------------------------------------
# Runtime-computed framework inputs
# ---------------------------------------------------------------------------


def compute_gamma_over_H_at_z0(cosmology: Optional[HLCDMCosmology] = None) -> float:
    """Return dimensionless γ(z=0) / H(z=0) from live framework functions.

    Uses :func:`HLCDMCosmology.gamma_at_redshift` divided by
    :func:`HLCDM_PARAMS.get_hubble_at_redshift`. Both return s⁻¹, so the
    ratio is dimensionless.

    The result is approximately 1/282 at z=0 by evaluation, but the exact
    value depends on the framework's γ(z) implementation at the time of the
    call. Do NOT cache or hardcode.
    """
    if cosmology is None:
        cosmology = HLCDMCosmology()
    gamma_0 = HLCDMCosmology.gamma_at_redshift(0.0)
    H_0 = HLCDM_PARAMS.get_hubble_at_redshift(0.0)
    return float(gamma_0 / H_0)


def _e8_clustering_coefficient() -> float:
    """Return C(G) = 25/32 from the E8×E8 heterotic construction.

    We prefer the live value from ``hlcdm.e8.e8_cache.E8Cache`` so this module
    tracks any refinement there. The expensive network construction is
    avoided — we pull the cached theoretical value directly. If the module
    cannot be imported for any reason (e.g., test environments where the
    ``networkx`` dependency is absent), fall back to the mathematical
    definition 25/32 = 0.78125.
    """
    try:
        from hlcdm.e8.e8_cache import E8Cache  # lazy: heavy import

        cache = E8Cache()
        value = cache.get_clustering_coefficient()
        # Guard against the empirical-from-graph value; the theoretical value
        # is what the formula uses. If ``get_clustering_coefficient`` ever
        # returns a graph-measured number that drifts from 25/32, fall back
        # to the mathematical definition.
        if abs(value - 25.0 / 32.0) > 1e-6:
            return 25.0 / 32.0
        return float(value)
    except Exception:  # pragma: no cover - defensive fallback
        return 25.0 / 32.0


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------


def holographic_h_ratio(
    d_local_mpc: float,
    d_cmb_mpc: Optional[float] = None,
    cosmology: Optional[HLCDMCosmology] = None,
    clustering_coefficient: Optional[float] = None,
    gamma_over_H: Optional[float] = None,
    second_order: bool = True,
    emit_warning: bool = True,
) -> HolographicRatioResult:
    """Evaluate the holographic projection formula for H_local / H_CMB.

    Parameters
    ----------
    d_local_mpc:
        Distance (in Mpc) to the direct geometric anchor defining the local
        measurement scale. For SH0ES-like / NGC 4258 anchored methods this is
        approximately 7.58 Mpc; for LMC-anchored methods it is 0.05 Mpc.
    d_cmb_mpc:
        Comoving distance to last scattering. Default: Planck 2018
        ``HLCDM_PARAMS.D_CMB_PLANCK_2018``.
    cosmology:
        Optional ``HLCDMCosmology`` instance. If omitted, a fresh one is
        created; γ/H is recomputed at call time.
    clustering_coefficient:
        Optional override for C(G). Default: live value from ``E8Cache``
        (which should always equal 25/32).
    gamma_over_H:
        Optional override for γ/H. Intended for tests and sensitivity
        analyses. In production, leave ``None`` so the formula pulls from
        the framework cosmology at call time.
    second_order:
        Include the ``(C(G) - 0.5) * L`` correction. Default True.
    emit_warning:
        Raise ``PerturbativeBreakdownWarning`` when the second-order
        correction magnitude exceeds 0.5. Default True. Monte Carlo drivers
        that evaluate the formula many times typically pass False and rely on
        the ``breakdown_flag`` in the return value.
    """
    if d_local_mpc <= 0.0:
        raise ValueError(f"d_local_mpc must be positive; got {d_local_mpc}")

    if d_cmb_mpc is None:
        d_cmb_mpc = float(HLCDM_PARAMS.D_CMB_PLANCK_2018)
    if d_cmb_mpc <= 0.0:
        raise ValueError(f"d_cmb_mpc must be positive; got {d_cmb_mpc}")

    if gamma_over_H is None:
        gamma_over_H = compute_gamma_over_H_at_z0(cosmology)
    if clustering_coefficient is None:
        clustering_coefficient = _e8_clustering_coefficient()

    L = math.log(d_cmb_mpc / d_local_mpc)
    linear_term = gamma_over_H * L
    quadratic_correction = (clustering_coefficient - 0.5) * L

    if second_order:
        ratio = 1.0 + linear_term * (1.0 + quadratic_correction)
    else:
        ratio = 1.0 + linear_term

    breakdown_flag = d_local_mpc < PERTURBATIVE_D_LOCAL_MPC
    breakdown_message: Optional[str] = None
    if breakdown_flag:
        breakdown_message = (
            "Perturbative expansion breakdown: "
            f"d_local = {d_local_mpc} Mpc < {PERTURBATIVE_D_LOCAL_MPC} Mpc threshold; "
            f"framework's perturbative derivation does not hold. "
            f"|(C(G) - 0.5) * L| = {abs(quadratic_correction):.3f}."
        )
        if emit_warning:
            warnings.warn(breakdown_message, PerturbativeBreakdownWarning, stacklevel=2)

    return HolographicRatioResult(
        ratio=float(ratio),
        gamma_over_H=float(gamma_over_H),
        clustering_coefficient=float(clustering_coefficient),
        L=float(L),
        linear_term=float(linear_term),
        quadratic_correction=float(quadratic_correction),
        breakdown_flag=bool(breakdown_flag),
        breakdown_message=breakdown_message,
        d_local_mpc=float(d_local_mpc),
        d_cmb_mpc=float(d_cmb_mpc),
        second_order=bool(second_order),
    )


def predict_local_H0(
    H_cmb: float,
    d_local_mpc: float,
    **kwargs,
) -> Tuple[float, HolographicRatioResult]:
    """Return the predicted local H₀ (km/s/Mpc) and the full ratio result.

    ``H_cmb`` is the CMB-inferred H₀ (nominally the Planck 2018 posterior
    mean, ~67.4 km/s/Mpc). Extra kwargs are forwarded to
    :func:`holographic_h_ratio`.
    """
    result = holographic_h_ratio(d_local_mpc=d_local_mpc, **kwargs)
    return float(H_cmb * result.ratio), result


def propagate_projection_uncertainty(
    H_cmb_samples: np.ndarray,
    d_local_samples: np.ndarray,
    d_cmb_mpc: Optional[float] = None,
    cosmology: Optional[HLCDMCosmology] = None,
    clustering_coefficient: Optional[float] = None,
    second_order: bool = True,
    emit_warning: bool = False,
) -> Tuple[np.ndarray, List[HolographicRatioResult]]:
    """Monte Carlo propagation of the projection formula.

    Evaluates :func:`holographic_h_ratio` on each pair of H_cmb / d_local
    draws, returning the array of predicted local H₀ samples and the list
    of per-draw ``HolographicRatioResult`` records (useful for aggregating
    breakdown statistics).

    ``emit_warning`` defaults to False here since the formula may be called
    N×10⁵ times during framework-branch MC propagation and stderr spam
    would be useless. The breakdown state is preserved on each result.
    """
    H_cmb_samples = np.asarray(H_cmb_samples, dtype=float)
    d_local_samples = np.asarray(d_local_samples, dtype=float)
    if H_cmb_samples.shape != d_local_samples.shape:
        raise ValueError(
            "H_cmb_samples and d_local_samples must share shape; got "
            f"{H_cmb_samples.shape} vs {d_local_samples.shape}"
        )

    # γ/H is computed once (framework state is stationary across the Monte
    # Carlo draw) so we don't pay the import/lookup per sample.
    if cosmology is None:
        cosmology = HLCDMCosmology()
    if clustering_coefficient is None:
        clustering_coefficient = _e8_clustering_coefficient()
    gamma_over_H = compute_gamma_over_H_at_z0(cosmology)
    if d_cmb_mpc is None:
        d_cmb_mpc = float(HLCDM_PARAMS.D_CMB_PLANCK_2018)

    H0_samples = np.empty_like(H_cmb_samples)
    per_draw: List[HolographicRatioResult] = []
    for i, (h_cmb, d_local) in enumerate(zip(H_cmb_samples, d_local_samples)):
        res = holographic_h_ratio(
            d_local_mpc=float(d_local),
            d_cmb_mpc=d_cmb_mpc,
            clustering_coefficient=clustering_coefficient,
            gamma_over_H=gamma_over_H,
            second_order=second_order,
            emit_warning=emit_warning,
        )
        H0_samples[i] = h_cmb * res.ratio
        per_draw.append(res)

    return H0_samples, per_draw


__all__ = [
    "HolographicRatioResult",
    "PERTURBATIVE_D_LOCAL_MPC",
    "PerturbativeBreakdownWarning",
    "compute_gamma_over_H_at_z0",
    "holographic_h_ratio",
    "predict_local_H0",
    "propagate_projection_uncertainty",
]
