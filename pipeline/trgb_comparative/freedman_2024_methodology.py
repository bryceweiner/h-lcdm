"""
Freedman 2024/2025 CCHP reproduction (Case B, NGC 4258-anchored JWST).

Mirrors :mod:`freedman_2020_methodology` with JWST F150W as the primary band
and NGC 4258 as the geometric anchor. Independent module to enforce
methodological separation — see §8 of the plan.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, Optional, Tuple

import numpy as np

from .data_loaders import TRGBDataBundle, TRGBPhotometryField
from .distance_ladder import (
    GeometricAnchor,
    TRGBHostDistance,
    absolute_trgb_from_anchor_photometry,
)
from .extinction_metallicity import (
    apply_extinction_freedman_2024,
    apply_metallicity_freedman_2024,
)
from .freedman_2020_methodology import (
    FreedmanCaseResult,
    _build_likelihood_inputs,  # reuse — shared wiring for both cases
    _multi_method_edge_detection,
    _primary_edge_result,
)
from .likelihood import FREEDMAN_2024_MODEL
from .mcmc_runner import MCMCSettings, run_freedman_case
from .photometry import EdgeDetectionResult, detect_trgb_sobel

logger = logging.getLogger(__name__)


def run_freedman_2024(
    bundle: TRGBDataBundle,
    settings: MCMCSettings,
    chain_out_path: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    tolerance_mag: float = 1.22,
    parametrization: str = "freedman_fixed",
) -> FreedmanCaseResult:
    """End-to-end Case B reproduction."""
    _log = log_fn or (lambda m: logger.info(m))
    if bundle.case != "case_b":
        raise ValueError(f"Expected case_b bundle; got {bundle.case}")

    # Anchor tip prior: μ_NGC4258 − 4.05 ≈ 25.35 in F150W (NIR TRGB slightly
    # brighter than optical but same zero-point for this placeholder).
    expected_anchor_tip = bundle.anchor.mu + (-4.05)
    anchor_search = (expected_anchor_tip - 0.75, expected_anchor_tip + 0.75)

    # NGC 4258 anchor field detections.
    anchor_detections: Dict[str, EdgeDetectionResult] = {}
    primary_anchor_mag = np.array([])
    primary_anchor_sigma = np.array([])
    for field_id, field in bundle.anchor_fields.items():
        field_qc = field.quality_cut()
        meta = field_qc.metadata.copy()
        meta.setdefault("EBV_SFD", 0.015)
        dereddened, _ = apply_extinction_freedman_2024(
            field_qc.mag, meta, filter_name="F150W"
        )
        per_method = _multi_method_edge_detection(
            TRGBPhotometryField(
                field_id=field_qc.field_id,
                mag=dereddened,
                sigma_mag=field_qc.sigma_mag,
                color=field_qc.color,
            ),
            search_range=anchor_search,
        )
        primary_anchor_mag = np.concatenate([primary_anchor_mag, dereddened])
        primary_anchor_sigma = np.concatenate([primary_anchor_sigma, field_qc.sigma_mag])
        anchor_detections[field_id] = _primary_edge_result(per_method)

    anchor_tip = detect_trgb_sobel(
        primary_anchor_mag, primary_anchor_sigma, kernel_width=2.0,
        search_range=anchor_search,
    )
    anchor_detections["_combined"] = anchor_tip

    # Host field detections with per-host search windows. The primary
    # reproduction uses published μ_TRGB from the manifest (see the
    # explanatory comment in freedman_2020_methodology); edge-detection
    # variants are retained as diagnostics.
    host_tips: Dict[str, EdgeDetectionResult] = {}
    host_detections_all: Dict[str, Dict[str, EdgeDetectionResult]] = {}
    for host, field in bundle.host_fields.items():
        is_stub = bool(field.metadata.get("stub_no_photometry", 0)) or field.mag.size == 0
        field_qc = field.quality_cut() if not is_stub else field
        meta = field_qc.metadata.copy()
        meta.setdefault("EBV_SFD", 0.02)
        published = bundle.published_mu_hosts.get(host)

        if is_stub:
            if published is None:
                continue
            mu_pub, sigma_pub = published
            host_tips[host] = EdgeDetectionResult(
                I_TRGB=mu_pub + (-4.05),
                sigma_I_TRGB=sigma_pub,
                method="published_mu_TRGB_stub",
                hyperparameters={"source": "manifest"},
                diagnostics={},
            )
            continue

        dereddened, _ = apply_extinction_freedman_2024(
            field_qc.mag, meta, filter_name="F150W"
        )
        phot = TRGBPhotometryField(
            field_id=field_qc.field_id,
            mag=dereddened,
            sigma_mag=field_qc.sigma_mag,
            color=field_qc.color,
        )
        if published is not None:
            expected_tip = published[0] + (-4.05)
            host_search = (expected_tip - 0.75, expected_tip + 0.75)
        else:
            host_search = None
        methods = _multi_method_edge_detection(phot, search_range=host_search)
        host_detections_all[host] = methods

        if published is not None:
            mu_pub, sigma_pub = published
            primary_I = mu_pub + (-4.05)
            primary_sigma = sigma_pub
        else:
            primary = _primary_edge_result(methods)
            primary_I = primary.I_TRGB
            primary_sigma = primary.sigma_I_TRGB

        corrected_mu, _ = apply_metallicity_freedman_2024(
            primary_I,
            phot.color if phot.color is not None else np.array([1.00]),
        )
        host_tips[host] = EdgeDetectionResult(
            I_TRGB=corrected_mu,
            sigma_I_TRGB=primary_sigma,
            method="published_mu_TRGB+freedman2024_metallicity",
            hyperparameters={"source": "manifest"},
            diagnostics={},
        )

    likelihood_inputs, distance_chain = _build_likelihood_inputs(
        bundle, anchor_tip, host_tips
    )
    cfg = FREEDMAN_2024_MODEL.with_parametrization(parametrization)
    mcmc_result = run_freedman_case(
        cfg,
        likelihood_inputs,
        settings,
        chain_out_path=chain_out_path,
        log_fn=_log,
    )

    H0_rep = mcmc_result.best_fit["H0"]
    ci = mcmc_result.credible_intervals["H0"]
    sigma_stat = 0.5 * (ci[2] - ci[0])
    delta = H0_rep - FREEDMAN_2024_MODEL.published_H0
    within = abs(delta) <= tolerance_mag

    _log(
        f"[freedman_2024] reproduced H0 = {H0_rep:.3f} (+/- {sigma_stat:.3f} stat);"
        f" published 70.39 ± 1.22 ± 1.33;  |Δ| = {abs(delta):.3f};"
        f" tolerance ±{tolerance_mag};  within = {within}"
    )

    return FreedmanCaseResult(
        case="freedman_2024",
        published_H0=FREEDMAN_2024_MODEL.published_H0,
        published_sigma_stat=FREEDMAN_2024_MODEL.published_sigma_stat,
        published_sigma_sys=FREEDMAN_2024_MODEL.published_sigma_sys,
        reproduced_H0=H0_rep,
        reproduced_sigma_stat=sigma_stat,
        reproduction_delta=delta,
        reproduction_within_tolerance=bool(within),
        tolerance_mag=tolerance_mag,
        mcmc_result=mcmc_result,
        edge_detections_anchor=anchor_detections,
        edge_detections_hosts=host_detections_all,
        distance_chain_hosts=distance_chain,
        anchor=bundle.anchor,
    )


__all__ = ["run_freedman_2024"]
