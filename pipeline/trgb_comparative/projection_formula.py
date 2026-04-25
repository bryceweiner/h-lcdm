"""
Holographic projection formula for H_local / H_CMB.

This is the framework's forward-prediction centerpiece for the TRGB comparative
analysis:

    H_local / H_CMB  =  1 + (γ/H)·L · [1 + C(G)·L]
    where            L = ln(d_CMB / d_local)

`γ/H` is **computed at runtime** from ``HLCDMCosmology.gamma_at_redshift(0.0)``
divided by ``HLCDM_PARAMS.get_hubble_at_redshift(0.0)``. It is NOT cached as a
hardcoded constant — the nominal ~1/282 is an evaluation result, not a
definition, and caching it would silently disconnect downstream code from any
refinement of γ.

``C(G)`` is the clustering coefficient of the E8×E8 root-system adjacency
graph under **Convention A** (canonical root-system graph: edges where
⟨α,β⟩ = +1). It is sourced live from
:func:`e8_heterotic.get_e8_clustering_coefficient`, which evaluates to
27/55 ≈ 0.4909 for Convention A.

**2026-04-25 correction**: an earlier version of this module evaluated
the second-order correction as ``(C(G) - 0.5) * L``, treating ``b = 0.5``
as a forward-only ansatz. ``b`` had no first-principles derivation and
has been removed entirely (not set to zero — removed). The corrected
formula is ``[1 + C(G) * L]``. See ``docs/correction_log.md`` and
``results/preregistration_addendum.md`` for the full audit trail. At
NGC 4258 (d_local = 7.58 Mpc) the corrected formula predicts H_local
≈ 75.82 km/s/Mpc; at LMC (d_local = 0.05 Mpc) ≈ 88.85 km/s/Mpc, both
shifted upward from the pre-correction values.

``d_CMB`` is the Planck 2018 comoving distance to last scattering, a
named constant on ``HLCDM_PARAMS``.

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


#: Threshold on |C(G) * L| above which the projection formula's truncated
#: perturbative expansion (kept to first order inside the bracket
#: ``[1 + C(G) * L]``) is no longer self-consistent: the next-order
#: term in the expansion has magnitude ≥ the leading term it corrects.
#:
#: This is the **physics-motivated breakdown criterion** for Form 1:
#:
#:     ratio = 1 + (γ/H) · L · [1 + C(G) · L]
#:
#: where the bracket is a first-order Taylor truncation in the small
#: parameter ``β·L = C(G)·L``. Truncation is justified when ``|β·L| < 1``.
#: When ``|β·L| ≥ 1``, the dropped ``(β·L)²`` term has magnitude ≥
#: ``β·L`` itself and the truncation is no longer perturbatively valid.
#:
#: **Replaces the legacy ``d_local < 1 Mpc`` heuristic** that was
#: carried over from the pre-2026-04-25 formula. That heuristic was a
#: geometric scale criterion, not a property of Form 1's expansion;
#: under the corrected formula it has no first-principles standing.
#:
#: With C(G) = 27/55 (Convention A) and d_CMB = 13869.7 Mpc, the
#: |β·L| = 1 contour sits at d_local ≈ 1885 Mpc — i.e., the strict
#: perturbative regime is essentially the whole observable universe;
#: TRGB-anchored measurements (d_local ≈ a few to tens of Mpc) all sit
#: outside it. The flag is therefore a `caution` annotation rather
#: than an `invalid prediction` annotation; the formula's evaluation
#: remains finite and the prediction value is still reported.
PERTURBATIVE_BETA_L_THRESHOLD: float = 1.0

#: Legacy alias retained so external callers that imported the old
#: ``d_local < 1 Mpc`` constant continue to import without raising
#: ImportError. NOT used by the breakdown check anymore.
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
    linear_term: float                  # (γ/H) * L
    quadratic_correction: float         # C(G) * L (post-2026-04-25 correction)
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


#: Adjacency convention for the E8×E8 root-system graph.
#: Convention A is the canonical root-system graph (⟨α,β⟩ = +1 for edges).
#: Other conventions in the e8-heterotic-network package: 'B' (⟨α,β⟩ ≠ 0),
#: 'C' (|⟨α,β⟩| = 1), 'D' (⟨α,β⟩ = -1). Convention A clustering
#: coefficient is 27/55 ≈ 0.4909.
E8_ADJACENCY_CONVENTION: str = "A"


def _e8_clustering_coefficient() -> float:
    """Return C(G) for the E8×E8 root-system graph under Convention A.

    Sourced live from
    :func:`e8_heterotic.get_e8_clustering_coefficient`, the canonical
    adjacency-graph clustering coefficient under Convention A
    (⟨α,β⟩ = +1 for edges). Convention A evaluates to 27/55 ≈ 0.4909.

    The previous local ``hlcdm.e8.e8_cache.E8Cache`` returned 25/32 =
    0.78125 (a literature claim, not an actual computation on the
    canonical adjacency graph) and has been removed; the migration to
    the external ``e8-heterotic-network`` package corrects this.
    """
    from e8_heterotic import get_e8_clustering_coefficient
    return float(get_e8_clustering_coefficient(E8_ADJACENCY_CONVENTION))


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
    **_rejected_kwargs,
) -> HolographicRatioResult:
    """Evaluate the holographic projection formula for H_local / H_CMB.

    The formula is::

        H_local / H_CMB  =  1 + (γ/H) · L · [1 + C(G) · L]
        where            L = ln(d_CMB / d_local)

    No ``b`` parameter is accepted. Passing ``b=...`` (or any other
    unrecognised keyword) raises :class:`TypeError`.

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
        Optional override for C(G). Default: live value from
        :func:`e8_heterotic.get_e8_clustering_coefficient` under
        Convention A (canonical root-system graph), which evaluates to
        27/55 ≈ 0.4909.
    gamma_over_H:
        Optional override for γ/H. Intended for tests and sensitivity
        analyses. In production, leave ``None`` so the formula pulls from
        the framework cosmology at call time.
    second_order:
        Include the ``C(G) * L`` correction. Default True. With
        ``second_order=False`` the formula reduces to
        ``1 + (γ/H) · L``.
    emit_warning:
        Raise ``PerturbativeBreakdownWarning`` when ``d_local < 1 Mpc``
        (the perturbative-derivation domain boundary). Default True. Monte
        Carlo drivers that evaluate the formula many times typically pass
        False and rely on the ``breakdown_flag`` in the return value.
    """
    # Reject deprecated `b` (and any other unrecognised) kwargs explicitly.
    # The 2026-04-25 correction removed the `(C(G) - b) * L` form; passing
    # `b=…` is a code-archaeology error and is rejected loudly rather than
    # silently ignored.
    if _rejected_kwargs:
        raise TypeError(
            f"holographic_h_ratio() received unexpected keyword arguments: "
            f"{sorted(_rejected_kwargs)}. The pre-2026-04-25 `b` parameter "
            "was removed from the formula (see docs/correction_log.md); "
            "pass only the documented arguments."
        )

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
    # Second-order correction is C(G) * L (post-2026-04-25; pre-correction
    # form was (C(G) - 0.5) * L).
    quadratic_correction = clustering_coefficient * L

    if second_order:
        ratio = 1.0 + linear_term * (1.0 + quadratic_correction)
    else:
        ratio = 1.0 + linear_term

    # Physics-motivated breakdown criterion under Form 1: the bracket's
    # first-order Taylor truncation `[1 + β·L]` is self-consistent only
    # when |β·L| < 1. When |β·L| ≥ 1 the dropped (β·L)² term has
    # magnitude ≥ the kept (β·L) term, so the truncation is no longer
    # perturbatively justified.
    beta_L_magnitude = abs(quadratic_correction)        # |C(G) · L|
    breakdown_flag = beta_L_magnitude >= PERTURBATIVE_BETA_L_THRESHOLD
    breakdown_message: Optional[str] = None
    if breakdown_flag:
        breakdown_message = (
            "Perturbative expansion breakdown: "
            f"|C(G) * L| = {beta_L_magnitude:.3f} ≥ "
            f"{PERTURBATIVE_BETA_L_THRESHOLD} threshold "
            f"(d_local = {d_local_mpc} Mpc, L = {L:.3f}). "
            "The first-order Taylor truncation in β = C(G) is no longer "
            "self-consistent; the dropped (β·L)² term is comparable to "
            "or larger than the kept (β·L) term. The prediction is "
            "still finite and reported as a number, but should be "
            "treated as outside the strict perturbative validity region "
            "of Form 1."
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
