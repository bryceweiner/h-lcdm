"""
Holographic projection formula for H_local / H_CMB.

This is the framework's forward-prediction centerpiece for the TRGB comparative
analysis:

    H_local / H_CMB  =  1 + (γ/H) · L
    where            L = ln(d_CMB / d_local)

`γ/H` is **computed at runtime** from ``HLCDMCosmology.gamma_at_redshift(0.0)``
divided by ``HLCDM_PARAMS.get_hubble_at_redshift(0.0)``. It is NOT cached as a
hardcoded constant — the nominal ~1/282 is an evaluation result, not a
definition, and caching it would silently disconnect downstream code from any
refinement of γ.

``d_CMB`` is the Planck 2018 comoving distance to last scattering, a named
constant on ``HLCDM_PARAMS``.

**Correction history**:

- 2026-04-25 (commit ``0be9fb5``): removed the ``b = 0.5`` ansatz that had
  the formula reading ``[1 + (C(G) − b) · L]``; ``b`` had no first-principles
  derivation. The intermediate-corrected formula was
  ``1 + a · (γ/H)·L · [1 + C(G)·L]`` with ``a = 1``.
- 2026-04-25 (later in same commit cycle): removed the ``C(G)·L`` second-order
  term as well. The bracket reduces to unity and the formula collapses to
  ``1 + a · (γ/H)·L``. ``C(G)`` had been retained as a forward-only ansatz
  that no longer survives once the algebra is taken seriously.
- 2026-04-25 (this commit): removed the leading ``a`` amplitude prefactor
  itself. ``a`` had always been 1 in published derivations and existed only
  as an unjustified free amplitude knob. The formula's final form is
  ``1 + (γ/H)·L`` with no free parameters beyond runtime-computed γ/H,
  observed d_local, and the Planck d_CMB. See ``docs/correction_log.md``
  and ``results/preregistration_addendum.md`` for the full audit trail.

Reference predictions under the linear form (``H_CMB = 67.4 km/s/Mpc``,
``γ/H = 1/281.7``, ``d_CMB = 13869.7 Mpc``):

    NGC 4258 (d_local = 7.58 Mpc):   H_local = 67.4 · 1.02666 ≈ 69.20 km/s/Mpc
    LMC      (d_local = 0.05 Mpc):   H_local = 67.4 · 1.04449 ≈ 70.40 km/s/Mpc

The return value of :func:`holographic_h_ratio` is a structured
:class:`HolographicRatioResult` rather than a bare float, so callers can log
the full provenance (γ/H used, intermediate terms, breakdown flag, message)
to output artifacts rather than losing warnings to stderr.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from hlcdm.cosmology import HLCDMCosmology
from hlcdm.parameters import HLCDM_PARAMS


# ---------------------------------------------------------------------------
# Warnings / results
# ---------------------------------------------------------------------------


#: Threshold on ``|γ/H · L|`` above which the linear-form projection
#: formula's first-order Taylor truncation in the small parameter
#: ``γ/H · L`` is no longer self-consistent.
#:
#: Under the linear form ``ratio = 1 + (γ/H) · L``, this is the analogue
#: of the Form-1 ``|C(G) · L| ≥ 1`` criterion: the formula is the
#: leading-order term in a Taylor expansion of some underlying functional
#: form, and the truncation is justified only when the small parameter is
#: < 1. With ``γ/H ≈ 1/282`` the contour ``|γ/H · L| = 1`` sits at
#: ``L = 282``, i.e. ``d_local = d_CMB / e^{282}`` — astronomically
#: smaller than any TRGB anchor used in practice. The flag therefore
#: never fires for realistic distance-ladder anchors but is preserved for
#: defense in depth (e.g., extreme-d_local sensitivity tests).
PERTURBATIVE_LINEAR_THRESHOLD: float = 1.0


class PerturbativeBreakdownWarning(UserWarning):
    """Raised (by ``warnings.warn``) when ``|γ/H · L| ≥
    PERTURBATIVE_LINEAR_THRESHOLD``: the linear-form projection formula's
    leading-order truncation is no longer perturbatively self-consistent.
    For all realistic TRGB-anchored measurements this never fires.

    Caller can catch this (``warnings.catch_warnings``) to suppress stderr
    output during Monte Carlo propagation; the breakdown state is also
    always recorded on the returned :class:`HolographicRatioResult`.
    """


@dataclass(frozen=True)
class HolographicRatioResult:
    """Structured output of :func:`holographic_h_ratio`.

    All intermediate quantities are preserved so callers can reason about
    the prediction beyond the single ``ratio`` number. The struct holds
    only quantities that exist in the linear-form formula
    ``ratio = 1 + (γ/H) · L``: there is no clustering coefficient and no
    quadratic correction term.
    """

    ratio: float
    gamma_over_H: float
    L: float
    linear_term: float                  # (γ/H) * L
    breakdown_flag: bool
    breakdown_message: Optional[str]
    d_local_mpc: float
    d_cmb_mpc: float


# ---------------------------------------------------------------------------
# Runtime-computed framework inputs
# ---------------------------------------------------------------------------


def compute_gamma_over_H_at_z0(cosmology: Optional[HLCDMCosmology] = None) -> float:
    """Return dimensionless γ(z=0) / H(z=0) from live framework functions.

    Uses :func:`HLCDMCosmology.gamma_at_redshift` divided by
    :func:`HLCDM_PARAMS.get_hubble_at_redshift`. Both return s⁻¹, so the
    ratio is dimensionless.

    The result is approximately 1/282 at z=0 by evaluation, but the exact
    value depends on the framework's γ(z) implementation at the time of
    the call. Do NOT cache or hardcode.
    """
    if cosmology is None:
        cosmology = HLCDMCosmology()
    gamma_0 = HLCDMCosmology.gamma_at_redshift(0.0)
    H_0 = HLCDM_PARAMS.get_hubble_at_redshift(0.0)
    return float(gamma_0 / H_0)


# ---------------------------------------------------------------------------
# Core formula
# ---------------------------------------------------------------------------


def holographic_h_ratio(
    d_local_mpc: float,
    d_cmb_mpc: Optional[float] = None,
    cosmology: Optional[HLCDMCosmology] = None,
    gamma_over_H: Optional[float] = None,
    emit_warning: bool = True,
    **_rejected_kwargs,
) -> HolographicRatioResult:
    """Evaluate the holographic projection formula for H_local / H_CMB.

    The formula is::

        H_local / H_CMB  =  1 + (γ/H) · L
        where            L = ln(d_CMB / d_local)

    No clustering coefficient, no quadratic term, no offset parameter,
    and no overall amplitude prefactor is accepted. Passing any of
    ``a=…``, ``b=…``, ``clustering_coefficient=…``, ``C_graph=…``,
    ``second_order=…``, etc. raises :class:`TypeError`. The historical
    ``a`` prefactor (the leading coefficient of the (γ/H)·L term) was
    always 1 in published derivations and has been dropped along with
    ``b`` and ``C(G)``: the linear form admits no free amplitude.

    Parameters
    ----------
    d_local_mpc:
        Distance (in Mpc) to the direct geometric anchor defining the
        local measurement scale. For SH0ES-like / NGC 4258 anchored
        methods this is approximately 7.58 Mpc; for LMC-anchored methods
        it is 0.05 Mpc.
    d_cmb_mpc:
        Comoving distance to last scattering. Default: Planck 2018
        ``HLCDM_PARAMS.D_CMB_PLANCK_2018``.
    cosmology:
        Optional ``HLCDMCosmology`` instance. If omitted, a fresh one is
        created; γ/H is recomputed at call time.
    gamma_over_H:
        Optional override for γ/H. Intended for tests and sensitivity
        analyses. In production, leave ``None`` so the formula pulls from
        the framework cosmology at call time.
    emit_warning:
        Raise ``PerturbativeBreakdownWarning`` when ``|γ/H · L| ≥ 1``
        (the linear-form's perturbative-validity boundary). Default True.
        Monte Carlo drivers that evaluate the formula many times typically
        pass False and rely on the ``breakdown_flag`` in the return value.
    """
    # Reject deprecated kwargs explicitly. The 2026-04-25 corrections
    # removed the `b` parameter, the `C(G)` term, and the leading `a`
    # amplitude prefactor in three steps; passing `a=…` / `b=…` /
    # `clustering_coefficient=…` / `C_graph=…` / `second_order=…` is a
    # code-archaeology error and is rejected loudly.
    if _rejected_kwargs:
        raise TypeError(
            f"holographic_h_ratio() received unexpected keyword arguments: "
            f"{sorted(_rejected_kwargs)}. The 2026-04-25 corrections "
            "removed the `b` parameter (commit 0be9fb5), the `C(G)` "
            "term, and the leading `a` amplitude prefactor from the "
            "formula. The linear form has no clustering / quadratic / "
            "offset / amplitude parameter to override; see "
            "docs/correction_log.md."
        )

    if d_local_mpc <= 0.0:
        raise ValueError(f"d_local_mpc must be positive; got {d_local_mpc}")

    if d_cmb_mpc is None:
        d_cmb_mpc = float(HLCDM_PARAMS.D_CMB_PLANCK_2018)
    if d_cmb_mpc <= 0.0:
        raise ValueError(f"d_cmb_mpc must be positive; got {d_cmb_mpc}")

    if gamma_over_H is None:
        gamma_over_H = compute_gamma_over_H_at_z0(cosmology)

    L = math.log(d_cmb_mpc / d_local_mpc)
    linear_term = gamma_over_H * L
    ratio = 1.0 + linear_term

    # Linear-form perturbative breakdown criterion: |γ/H · L| ≥ 1.
    # With γ/H ≈ 1/282, this never triggers for realistic distance-
    # ladder anchors (NGC 4258 d_local = 7.58 Mpc gives γ/H · L ≈ 0.027;
    # LMC d_local = 0.05 Mpc gives ≈ 0.045). Retained as a defense-in-
    # depth flag in case of extreme-d_local sensitivity tests.
    linear_magnitude = abs(linear_term)
    breakdown_flag = linear_magnitude >= PERTURBATIVE_LINEAR_THRESHOLD
    breakdown_message: Optional[str] = None
    if breakdown_flag:
        breakdown_message = (
            "Perturbative expansion breakdown: "
            f"|γ/H · L| = {linear_magnitude:.3f} ≥ "
            f"{PERTURBATIVE_LINEAR_THRESHOLD} threshold "
            f"(d_local = {d_local_mpc} Mpc, L = {L:.3f}, "
            f"γ/H = {gamma_over_H:.6f}). The linear-form Taylor "
            "truncation is no longer perturbatively justified. The "
            "prediction value is still finite but should be treated "
            "as outside the strict linear regime."
        )
        if emit_warning:
            warnings.warn(breakdown_message, PerturbativeBreakdownWarning, stacklevel=2)

    return HolographicRatioResult(
        ratio=float(ratio),
        gamma_over_H=float(gamma_over_H),
        L=float(L),
        linear_term=float(linear_term),
        breakdown_flag=bool(breakdown_flag),
        breakdown_message=breakdown_message,
        d_local_mpc=float(d_local_mpc),
        d_cmb_mpc=float(d_cmb_mpc),
    )


def predict_local_H0(
    H_cmb: float,
    d_local_mpc: float,
    **kwargs,
) -> Tuple[float, HolographicRatioResult]:
    """Return the predicted local H₀ (km/s/Mpc) and the full ratio result.

    ``H_cmb`` is the CMB-inferred H₀ (nominally the Planck 2018 posterior
    mean, ~67.4 km/s/Mpc). Extra kwargs are forwarded to
    :func:`holographic_h_ratio` (which rejects any unsupported kwarg).
    """
    result = holographic_h_ratio(d_local_mpc=d_local_mpc, **kwargs)
    return float(H_cmb * result.ratio), result


def propagate_projection_uncertainty(
    H_cmb_samples: np.ndarray,
    d_local_samples: np.ndarray,
    d_cmb_mpc: Optional[float] = None,
    cosmology: Optional[HLCDMCosmology] = None,
    emit_warning: bool = False,
) -> Tuple[np.ndarray, List[HolographicRatioResult]]:
    """Monte Carlo propagation of the linear-form projection formula.

    Evaluates :func:`holographic_h_ratio` on each pair of H_cmb /
    d_local draws, returning the array of predicted local H₀ samples
    and the list of per-draw ``HolographicRatioResult`` records (useful
    for aggregating breakdown statistics).

    ``emit_warning`` defaults to False here since the formula may be
    called N×10⁵ times during framework-branch MC propagation and stderr
    spam would be useless. The breakdown state is preserved on each
    result.
    """
    H_cmb_samples = np.asarray(H_cmb_samples, dtype=float)
    d_local_samples = np.asarray(d_local_samples, dtype=float)
    if H_cmb_samples.shape != d_local_samples.shape:
        raise ValueError(
            "H_cmb_samples and d_local_samples must share shape; got "
            f"{H_cmb_samples.shape} vs {d_local_samples.shape}"
        )

    # γ/H is computed once (framework state is stationary across the
    # Monte Carlo draw) so we don't pay the import/lookup per sample.
    if cosmology is None:
        cosmology = HLCDMCosmology()
    gamma_over_H = compute_gamma_over_H_at_z0(cosmology)
    if d_cmb_mpc is None:
        d_cmb_mpc = float(HLCDM_PARAMS.D_CMB_PLANCK_2018)

    H0_samples = np.empty_like(H_cmb_samples)
    per_draw: List[HolographicRatioResult] = []
    for i, (h_cmb, d_local) in enumerate(zip(H_cmb_samples, d_local_samples)):
        res = holographic_h_ratio(
            d_local_mpc=float(d_local),
            d_cmb_mpc=d_cmb_mpc,
            gamma_over_H=gamma_over_H,
            emit_warning=emit_warning,
        )
        H0_samples[i] = h_cmb * res.ratio
        per_draw.append(res)

    return H0_samples, per_draw


__all__ = [
    "HolographicRatioResult",
    "PERTURBATIVE_LINEAR_THRESHOLD",
    "PerturbativeBreakdownWarning",
    "compute_gamma_over_H_at_z0",
    "holographic_h_ratio",
    "predict_local_H0",
    "propagate_projection_uncertainty",
]
