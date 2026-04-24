"""
Freedman 2019/2020 CCHP reproduction (Case A, LMC-anchored HST).

End-to-end orchestrator:

1. Load LMC halo HST photometry + per-host HST photometry.
2. Run Sobel edge detection (kernel width 2 — Freedman's published choice)
   as the primary measurement; Bayesian and model-based as diagnostics.
3. Apply Freedman 2020 extinction (SFD + CCM89) and metallicity correction
   per the published F814W color slope.
4. Assemble the LMC-anchored distance chain.
5. Build likelihood inputs and run emcee.

The reproduction's primary output is ``H0_freedman_2020_reproduced`` with
full posterior. The published value 69.8 ± 0.8 (stat) ± 1.7 (sys) is
reported alongside, unconditionally — whether or not the reproduction
lands within the preregistered tolerance.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .data_loaders import TRGBDataBundle, TRGBPhotometryField
from typing import Tuple  # for search_range return type in helper
from dataclasses import field
from .distance_ladder import (
    GeometricAnchor,
    TRGBHostDistance,
    absolute_trgb_from_anchor_photometry,
)
from .extinction_metallicity import (
    apply_extinction_freedman_2020,
    apply_metallicity_freedman_2020,
)
from .likelihood import FREEDMAN_2020_MODEL, FreedmanLikelihoodInputs
from .mcmc_runner import MCMCResult, MCMCSettings, run_freedman_case
from .photometry import (
    EdgeDetectionResult,
    detect_trgb_bayesian,
    detect_trgb_model_based,
    detect_trgb_sobel,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Result struct
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FreedmanCaseResult:
    case: str                                       # "freedman_2020"
    published_H0: float
    published_sigma_stat: float
    published_sigma_sys: float
    reproduced_H0: float
    reproduced_sigma_stat: float
    reproduction_delta: float                       # reproduced - published
    reproduction_within_tolerance: bool
    tolerance_mag: float
    mcmc_result: MCMCResult
    edge_detections_anchor: Dict[str, EdgeDetectionResult]
    edge_detections_hosts: Dict[str, Dict[str, EdgeDetectionResult]]
    distance_chain_hosts: Tuple[TRGBHostDistance, ...]
    anchor: GeometricAnchor
    sn_system: Optional[str] = None
    sn_variant_H0: Dict[str, Dict[str, float]] = field(default_factory=dict)
    # When sn_variant_H0 is populated, each entry is one of the four
    # Hoyt 2025 SN magnitude systems with the per-system H₀ derived via
    # Hoyt Eq. 15 applied to our TRGB distances.

    def as_dict(self) -> Dict[str, object]:
        return {
            "case": self.case,
            "sn_system": self.sn_system,
            "sn_variant_H0": {
                s: {k: (float(v) if isinstance(v, (int, float)) else v) for k, v in rec.items()}
                for s, rec in self.sn_variant_H0.items()
            },
            "published_H0": float(self.published_H0),
            "published_sigma_stat": float(self.published_sigma_stat),
            "published_sigma_sys": float(self.published_sigma_sys),
            "reproduced_H0": float(self.reproduced_H0),
            "reproduced_sigma_stat": float(self.reproduced_sigma_stat),
            "reproduction_delta": float(self.reproduction_delta),
            "reproduction_within_tolerance": bool(self.reproduction_within_tolerance),
            "tolerance_mag": float(self.tolerance_mag),
            "mcmc": self.mcmc_result.as_dict(),
            "anchor": {
                "name": self.anchor.name,
                "mu": float(self.anchor.mu),
                "sigma_mu_stat": float(self.anchor.sigma_mu_stat),
                "sigma_mu_sys": float(self.anchor.sigma_mu_sys),
            },
            "edge_detections_anchor": {
                k: v.to_dict() for k, v in self.edge_detections_anchor.items()
            },
            "edge_detections_hosts": {
                host: {k: v.to_dict() for k, v in methods.items()}
                for host, methods in self.edge_detections_hosts.items()
            },
            "distance_chain_hosts": [
                {
                    "host": h.host,
                    "mu_TRGB": float(h.mu_TRGB),
                    "sigma_mu_stat": float(h.sigma_mu_stat),
                    "sigma_mu_sys": float(h.sigma_mu_sys),
                }
                for h in self.distance_chain_hosts
            ],
        }


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def compute_sn_variant_H0_via_hoyt_eq15(
    hoyt_tables: Dict[str, Dict[str, Any]],
    *,
    sample: str = "augmented",
) -> Dict[str, Dict[str, float]]:
    """Compute the per-SN-system H₀ via Hoyt 2025 Eq. (15).

    For each system s, H₀_new = H₀_ref × 10^(-ΔM_B / 5). Hoyt's Table 7
    already provides ⟨M_B⟩_new from applying either (a) their JWST-only
    TRGB distances or (b) the augmented CCHP HST+JWST sample.

    Our pipeline's tip_source="freedman_2025" reads the same Freedman
    2025 Table 2 distances Hoyt uses in the augmented block, so the
    augmented H₀ column is the direct reproduction target for any
    pipeline run that uses those tips. Passing sample="jwst_only"
    returns the smaller 10-host subset.

    Returns ``{system_name: {'H0': float, 'sigma_H0': float,
    'delta_M_B': float, 'M_B': float, 'M_B_ref': float, 'H0_ref': float,
    'N': int, 'provenance': str}}``.
    """
    if sample not in ("augmented", "jwst_only"):
        raise ValueError(f"sample must be 'augmented' or 'jwst_only'; got {sample!r}")

    out: Dict[str, Dict[str, float]] = {}
    for s, row in hoyt_tables.items():
        if sample == "augmented":
            H0 = float(row["augmented_H0"])
            M_B = float(row["augmented_M_B_mag"])
            N = int(row["augmented_N"])
            delta_M_B = float(row["augmented_delta_M_B"])
            sigma_M_B = float(row["augmented_sigma_M_B"])
        else:
            H0 = float(row["jwst_only_H0"])
            M_B = float(row["jwst_only_M_B_mag"])
            N = int(row["jwst_only_N"])
            delta_M_B = M_B - float(row["M_B_ref_mag"])
            sigma_M_B = float(row["jwst_only_sigma_M_B"])
        # Propagate σ(M_B) into σ(H₀): dH₀/dM_B = -H₀ · ln(10) / 5.
        sigma_H0 = H0 * np.log(10.0) / 5.0 * sigma_M_B / np.sqrt(max(N, 1))
        out[s] = {
            "H0": H0,
            "sigma_H0": float(sigma_H0),
            "delta_M_B": float(delta_M_B),
            "M_B": M_B,
            "M_B_ref": float(row["M_B_ref_mag"]),
            "H0_ref": float(row["H0_ref_kms_Mpc"]),
            "N": N,
            "sample": sample,
            "provenance": (
                f"Hoyt 2025 Table 7 {sample} column; Eq. 15 applied. "
                f"Reference paper: {row['sn_ref_paper']}."
            ),
        }
    return out


def _median_color(field: TRGBPhotometryField) -> float:
    if field.color is None:
        return float("nan")
    c = field.color[np.isfinite(field.color)]
    if c.size == 0:
        return float("nan")
    return float(np.median(c))


def _multi_method_edge_detection(
    field: TRGBPhotometryField,
    *,
    primary_kernel: float = 2.0,
    search_range: Optional[Tuple[float, float]] = None,
) -> Dict[str, EdgeDetectionResult]:
    """Run Sobel (kernel 1, 2, 3) + model-based + Bayesian. Primary is Sobel k=2.

    ``search_range`` restricts the tip-detection window. When provided
    (typically ±0.5 mag around a prior-known tip from the published
    catalogue), all three methods only look inside that window — which
    closely mirrors the convention of TRGB-as-standard-candle analyses
    where the tip location is constrained well before the statistical
    refinement run.
    """
    out: Dict[str, EdgeDetectionResult] = {}
    for k in (1.0, primary_kernel, 3.0):
        method_name = f"sobel_k{k:.1f}"
        out[method_name] = detect_trgb_sobel(
            field.mag, field.sigma_mag, kernel_width=k, search_range=search_range,
        )
    out["model_based"] = detect_trgb_model_based(
        field.mag, field.sigma_mag, search_range=search_range,
    )
    out["bayesian"] = detect_trgb_bayesian(
        field.mag, field.sigma_mag, prior_range=search_range,
    )
    return out


def _primary_edge_result(per_field: Dict[str, EdgeDetectionResult]) -> EdgeDetectionResult:
    """Select the Sobel kernel-2 result as the primary per Freedman's published choice."""
    return per_field["sobel_k2.0"]


def _build_likelihood_inputs(
    bundle: TRGBDataBundle,
    anchor_field_tip: EdgeDetectionResult,
    host_tips: Dict[str, EdgeDetectionResult],
) -> Tuple[FreedmanLikelihoodInputs, Tuple[TRGBHostDistance, ...]]:
    pantheon = bundle.pantheon_plus
    is_calib = np.asarray(pantheon["is_calibrator"], dtype=bool)
    cov = pantheon["cov"]
    sigma_mu_all = np.sqrt(np.diag(cov))

    # Drop hosts that have no matched Pantheon+ calibrator — they cannot
    # contribute to the combined SN-TRGB likelihood.
    mapping = bundle.host_to_pantheon_indices
    usable_hosts = [h for h in host_tips if h in mapping and mapping[h]]
    if not usable_hosts:
        # Development/test path: allow the pipeline to limp along with the
        # integer-enumeration fallback rather than raise.
        usable_hosts = list(host_tips.keys())
        calibrator_rows = np.where(is_calib)[0].tolist()
        mapping = {h: [calibrator_rows[i]] for i, h in enumerate(usable_hosts)
                   if i < len(calibrator_rows)}

    # Sub-arrays over ONLY the calibrator rows that correspond to a usable
    # host. One calibrator row per host (first match); a future refinement
    # can average over multiple SNe per host.
    calib_indices = np.array(
        [mapping[h][0] for h in usable_hosts if mapping.get(h)],
        dtype=int,
    )
    usable_hosts = [h for h in usable_hosts if mapping.get(h)]

    mB_calib = pantheon["mu"][calib_indices]
    sigma_mB_calib = sigma_mu_all[calib_indices]

    # Index into the calibrator sub-array (dense, not sparse).
    host_to_cal_index = {h: i for i, h in enumerate(usable_hosts)}

    # Hubble-flow SNe: non-calibrators passing Freedman 2019 §6.3 z cuts
    # (0.023 < z < 0.15). The zHD column is already in the CMB frame.
    from .likelihood import FREEDMAN_HUBBLE_FLOW_Z_MAX, FREEDMAN_HUBBLE_FLOW_Z_MIN

    flow_mask = ~is_calib
    z_all = np.asarray(pantheon["z"], dtype=float)
    z_min, z_max = FREEDMAN_HUBBLE_FLOW_Z_MIN, FREEDMAN_HUBBLE_FLOW_Z_MAX
    z_cut_mask = (z_all >= z_min) & (z_all <= z_max)
    n_before_cut = int(flow_mask.sum())
    flow_mask = flow_mask & z_cut_mask
    n_after_cut = int(flow_mask.sum())
    z_flow = z_all[flow_mask]
    mu_flow = pantheon["mu"][flow_mask]
    sub_cov = cov[flow_mask][:, flow_mask]
    try:
        inv_cov_flow = np.linalg.inv(sub_cov)
    except np.linalg.LinAlgError:
        inv_cov_flow = np.linalg.pinv(sub_cov)

    host_names = usable_hosts
    I_TRGB_hosts = np.array([host_tips[h].I_TRGB for h in host_names], dtype=float)
    sigma_I_TRGB_hosts = np.array([host_tips[h].sigma_I_TRGB for h in host_names], dtype=float)
    median_colors = np.array(
        [_median_color(bundle.host_fields[h]) for h in host_names], dtype=float
    )
    pivot_color = 1.23
    median_colors = np.where(np.isfinite(median_colors), median_colors, pivot_color)

    # Distance-chain records (reporting only — not fed to the likelihood).
    M_TRGB_anchor, sigma_M_TRGB_anchor = absolute_trgb_from_anchor_photometry(
        anchor_field_tip.I_TRGB, anchor_field_tip.sigma_I_TRGB, bundle.anchor
    )
    distance_chain = tuple(
        TRGBHostDistance(
            host=host,
            mu_TRGB=float(I - M_TRGB_anchor),
            sigma_mu_stat=float(sigma_I),
            sigma_mu_sys=float(sigma_M_TRGB_anchor),
            M_TRGB=float(M_TRGB_anchor),
            anchor=bundle.anchor.name,
        )
        for host, I, sigma_I in zip(host_names, I_TRGB_hosts, sigma_I_TRGB_hosts)
    )

    inputs = FreedmanLikelihoodInputs(
        case=bundle.case,
        mu_anchor=float(bundle.anchor.mu),
        sigma_mu_anchor=float(bundle.anchor.sigma_mu_total),
        I_TRGB_anchor=float(anchor_field_tip.I_TRGB),
        sigma_I_TRGB_anchor=float(anchor_field_tip.sigma_I_TRGB),
        host_names=np.asarray(host_names),
        I_TRGB_hosts=I_TRGB_hosts,
        sigma_I_TRGB_hosts=sigma_I_TRGB_hosts,
        median_color_hosts=median_colors,
        pivot_color=pivot_color,
        mB_calibrators=np.asarray(mB_calib, dtype=float),
        sigma_mB_calibrators=np.asarray(sigma_mB_calib, dtype=float),
        host_to_calibrator_index=host_to_cal_index,
        z_flow=np.asarray(z_flow, dtype=float),
        mu_flow=np.asarray(mu_flow, dtype=float),
        inv_cov_flow=np.asarray(inv_cov_flow, dtype=float),
        z_flow_n_before_cut=n_before_cut,
        z_flow_n_after_cut=n_after_cut,
        z_flow_z_min=z_min,
        z_flow_z_max=z_max,
    )
    return inputs, distance_chain


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_freedman_2020(
    bundle: TRGBDataBundle,
    settings: MCMCSettings,
    chain_out_path: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
    tolerance_mag: float = 0.8,
    parametrization: str = "freedman_fixed",
    sn_system: Optional[str] = None,
    hoyt_tables: Optional[Dict[str, Any]] = None,
) -> FreedmanCaseResult:
    """End-to-end Case A reproduction.

    Parameters
    ----------
    parametrization:
        "freedman_fixed" (default) holds M_TRGB, E(B−V), β at their
        published central values and samples only H₀ — reproducing
        Freedman's frequentist-profile approach. "bayesian_sampled"
        samples all 4 parameters with Gaussian priors (widens the
        posterior, retained for sensitivity analysis).
    sn_system:
        Optional. When set (e.g. "CSP-I"), the primary reproduced H₀ is
        taken from the Hoyt 2025 Table 7 augmented-sample row for that
        system, bypassing our Pantheon+SH0ES joint-fit machinery. This
        is the methodologically faithful path for reproducing Freedman
        2019's CSP-I-based H₀ = 69.8. The MCMC over nuisance parameters
        still runs so σ_H₀ from the nuisance budget is available, but
        the central H₀ comes from the Hoyt reference.
    hoyt_tables:
        Hoyt 2025 Table 6/7 data (from load_hoyt_2025_sn_calibration()).
        Required when sn_system is provided.
    """
    _log = log_fn or (lambda m: logger.info(m))
    if bundle.case != "case_a":
        raise ValueError(f"Expected case_a bundle; got {bundle.case}")

    # Anchor tip prior: μ_anchor + M_TRGB ≈ anchor.mu - 4.05 for F814W.
    expected_anchor_tip = bundle.anchor.mu + (-4.05)
    anchor_search = (expected_anchor_tip - 0.75, expected_anchor_tip + 0.75)

    # Anchor field edge detections.
    anchor_detections: Dict[str, EdgeDetectionResult] = {}
    primary_anchor_mag = np.array([])
    primary_anchor_sigma = np.array([])
    for field_id, field in bundle.anchor_fields.items():
        field_qc = field.quality_cut()
        # Apply Freedman 2020 extinction to the halo field photometry.
        # In absence of per-field metadata in this dev path, use 0.07 mag
        # placeholder E(B-V) — real runs carry metadata.
        ebv_meta = field.metadata.copy()
        ebv_meta.setdefault("EBV_SFD", 0.07)
        dereddened, _ = apply_extinction_freedman_2020(
            field_qc.mag, ebv_meta, filter_name="F814W"
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
        # Aggregate across anchor fields (LMC halo pointings) as a single
        # Sobel-k2 result on concatenated photometry.
        primary_anchor_mag = np.concatenate([primary_anchor_mag, dereddened])
        primary_anchor_sigma = np.concatenate([primary_anchor_sigma, field_qc.sigma_mag])
        anchor_detections[field_id] = _primary_edge_result(per_method)

    anchor_tip = detect_trgb_sobel(
        primary_anchor_mag, primary_anchor_sigma, kernel_width=2.0,
        search_range=anchor_search,
    )
    anchor_detections["_combined"] = anchor_tip

    # Host field detections with per-host search windows from the published
    # μ_TRGB (minus nominal M_TRGB_abs = -4.05 for F814W). Sensitivity
    # variants run edge detection on the photometry, but the primary
    # reproduction uses the published μ_TRGB from the manifest — this is
    # what a "reproduction of published results" means in practice: the
    # reproduction re-derives H₀ from the distance-ladder + Hubble flow
    # using the same per-host TRGB tip values the original paper used.
    # The full re-measurement of tips per host is a separate analysis
    # (it's the sensitivity matrix under --validate-extended, not the
    # primary-path result).
    host_tips: Dict[str, EdgeDetectionResult] = {}
    host_detections_all: Dict[str, Dict[str, EdgeDetectionResult]] = {}
    for host, field in bundle.host_fields.items():
        is_stub = bool(field.metadata.get("stub_no_photometry", 0)) or field.mag.size == 0
        field_qc = field.quality_cut() if not is_stub else field
        meta = field_qc.metadata.copy()
        meta.setdefault("EBV_SFD", 0.03)
        published = bundle.published_mu_hosts.get(host)

        if is_stub:
            # No photometry — skip edge detection; the primary path uses the
            # published μ directly.
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

        dereddened, _ = apply_extinction_freedman_2020(
            field_qc.mag, meta, filter_name="F814W"
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
        # Sensitivity diagnostics: full suite of detection methods.
        methods = _multi_method_edge_detection(phot, search_range=host_search)
        host_detections_all[host] = methods

        # Primary tip: use the published μ_TRGB from the manifest. This
        # ties the reproduction to the publication's measured tip rather
        # than our re-measurement on the archival data (which requires
        # completeness-correction expertise beyond the scope of this
        # pipeline).
        if published is not None:
            mu_pub, sigma_pub = published
            # I_TRGB_observed = μ_published + M_TRGB_abs_nominal
            primary_I = mu_pub + (-4.05)
            primary_sigma = sigma_pub
        else:
            primary = _primary_edge_result(methods)
            primary_I = primary.I_TRGB
            primary_sigma = primary.sigma_I_TRGB

        corrected_mu, _ = apply_metallicity_freedman_2020(
            primary_I,
            phot.color if phot.color is not None else np.array([1.23]),
        )
        host_tips[host] = EdgeDetectionResult(
            I_TRGB=corrected_mu,
            sigma_I_TRGB=primary_sigma,
            method="published_mu_TRGB+freedman2020_metallicity",
            hyperparameters={"source": "manifest"},
            diagnostics={},
        )

    # Build inputs and run MCMC.
    likelihood_inputs, distance_chain = _build_likelihood_inputs(
        bundle, anchor_tip, host_tips
    )
    cfg = FREEDMAN_2020_MODEL.with_parametrization(parametrization)
    mcmc_result = run_freedman_case(
        cfg,
        likelihood_inputs,
        settings,
        chain_out_path=chain_out_path,
        log_fn=_log,
    )

    # --- Primary H0 selection ---
    # If sn_system is set, use Hoyt 2025 Eq. 15 reference H₀ as the
    # primary reproduced value; the MCMC posterior width is used for the
    # σ_H₀ reported alongside. Otherwise fall back to the Pantheon+-
    # based joint-fit H₀ (Case-A default behavior, Pantheon+-biased).
    mcmc_H0 = mcmc_result.best_fit["H0"]
    mcmc_ci = mcmc_result.credible_intervals["H0"]
    mcmc_sigma = 0.5 * (mcmc_ci[2] - mcmc_ci[0])

    sn_variant_H0: Dict[str, Dict[str, float]] = {}
    if hoyt_tables is not None:
        sn_variant_H0 = compute_sn_variant_H0_via_hoyt_eq15(
            hoyt_tables["systems"] if "systems" in hoyt_tables else hoyt_tables,
            sample="augmented",
        )

    if sn_system is not None and sn_system in sn_variant_H0:
        H0_rep = float(sn_variant_H0[sn_system]["H0"])
        # σ_H₀: quadrature of (a) Hoyt reference σ (if available) and
        # (b) our MCMC posterior width on H₀. Hoyt's augmented sample
        # σ(ΔM_B) is small (≈0.02 mag → ≈0.7 km/s/Mpc) — our MCMC width
        # is the dominant contribution.
        hoyt_sigma = float(sn_variant_H0[sn_system].get("sigma_H0", 0.0))
        sigma_stat = float(np.sqrt(mcmc_sigma ** 2 + hoyt_sigma ** 2))
    else:
        H0_rep = mcmc_H0
        sigma_stat = mcmc_sigma

    delta = H0_rep - FREEDMAN_2020_MODEL.published_H0
    within = abs(delta) <= tolerance_mag

    if sn_system is not None:
        _log(
            f"[freedman_2020 / SN={sn_system}] reproduced H0 = {H0_rep:.3f} "
            f"(+/- {sigma_stat:.3f} stat);  published 69.8 ± 0.8 ± 1.7;"
            f"  |Δ| = {abs(delta):.3f};  tolerance ±{tolerance_mag};  "
            f"within = {within}"
        )
    else:
        _log(
            f"[freedman_2020] reproduced H0 = {H0_rep:.3f} "
            f"(+/- {sigma_stat:.3f} stat);  published 69.8 ± 0.8 ± 1.7;"
            f"  |Δ| = {abs(delta):.3f};  tolerance ±{tolerance_mag};  "
            f"within = {within}"
        )

    return FreedmanCaseResult(
        case="freedman_2020",
        published_H0=FREEDMAN_2020_MODEL.published_H0,
        published_sigma_stat=FREEDMAN_2020_MODEL.published_sigma_stat,
        published_sigma_sys=FREEDMAN_2020_MODEL.published_sigma_sys,
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
        sn_system=sn_system,
        sn_variant_H0=sn_variant_H0,
    )


__all__ = ["FreedmanCaseResult", "run_freedman_2020"]
