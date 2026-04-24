"""Per-(case, SN-system) chain factories and orchestrator.

Two chain modes are supported (one per photometric-system family):

- **Uddin 8-parameter** (CSP-I / CSP-II): the full 8-parameter SNooPy
  likelihood from Uddin 2023 H0CSP.py is run on the unified Uddin
  ``B_trgb_update3.csv`` data, filtered to the requested release era
  (CSP-I or CSP-II). Case A uses LMC-anchored μ_TRGB from the file;
  Case B swaps the calibrator block for NGC 4258-anchored μ_bar values
  from Freedman 2025 Table 2 (dropping calibrators whose host is absent
  from the 2025 sample).

- **Simple 1-parameter** (SuperCal / Pantheon+SH0ES): the input
  magnitudes are already fully standardized, so the chain samples H₀
  alone (profiling M_B analytically) against a pre-standardized
  (calibrator m_B, μ_TRGB) + (flow m_B, z) dataset.

Public API:

- :func:`build_chain_for(case, system, loader)` — returns a
  ``ChainPlan`` describing the inputs and MCMC mode.
- :func:`run_all_chains(case, loader, settings, chains_dir, log_fn)` —
  executes all four system chains for one case, returning a dict of
  results keyed by system id.
- :func:`run_all_chains_both_cases(loader, settings, chains_dir, log_fn)`
  — runs the full 8-chain matrix, returning a nested dict
  ``{case_id: {system_id: result_dict}}``.

All results implement ``.as_dict()`` and carry the R̂ convergence
diagnostic — the pipeline refuses to promote any posterior that does
not meet the R̂ < 1.01 convergence gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Union

import numpy as np

from data.loader import DataLoader, DataUnavailableError

from .data_loaders import SN_TO_HOST
from .mcmc_runner import MCMCSettings
from .sn_chain import SNChainData, SNChainResult, run_sn_chain
from .uddin_csp_chain import (
    UddinChainResult,
    build_uddin_inputs_from_loader_dataset,
    run_uddin_csp_chain,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Chain plan (discriminated-union style)
# =============================================================================


@dataclass
class ChainPlan:
    """Instructions for one (case, system) MCMC chain.

    ``mode == 'uddin_8param'``: ``uddin_inputs`` is an ``UddinCSPInputs``
    object and the runner uses :func:`run_uddin_csp_chain`.
    ``mode == 'simple_1param'``: ``simple_data`` is an ``SNChainData``
    object and the runner uses :func:`run_sn_chain`.
    """
    case: str
    system: str
    system_label: str
    mode: str                                   # 'uddin_8param' or 'simple_1param'
    published_target_H0: float
    published_sigma_stat: float
    published_sigma_sys: float
    notes: str = ""
    uddin_inputs: Optional[object] = None
    simple_data: Optional[SNChainData] = None


# =============================================================================
# Helpers (host name canonicalization, μ_TRGB lookups)
# =============================================================================


def _canon_host(s: str) -> str:
    if not isinstance(s, str):
        return ""
    t = s.strip()
    if t.upper().startswith("NGC"):
        rest = t[3:].lstrip()
        return f"NGC {rest}"
    if t.upper().startswith("N") and t[1:].isdigit():
        return f"NGC {t[1:]}"
    if t.upper().startswith("UGC"):
        rest = t[3:].lstrip()
        return f"UGC {rest}"
    if t.upper().startswith("U") and t[1:].isdigit():
        return f"UGC {t[1:]}"
    if t.upper().startswith("M") and t[1:].isdigit():
        return f"M{t[1:]}"
    return t


def _canon_sn(name: str) -> str:
    if not isinstance(name, str):
        return ""
    n = name.strip()
    if n.upper().startswith("SN"):
        n = n[2:].lstrip()
    return n


def _per_host_mu_freedman_2019(loader: DataLoader) -> Dict[str, Tuple[float, float]]:
    tbl = loader.load_freedman_2019_table3()
    return {_canon_host(h): (float(rec["mu"]), float(rec["sigma"]))
            for h, rec in tbl["hosts"].items()}


def _per_host_mu_freedman_2025(loader: DataLoader) -> Dict[str, Tuple[float, float]]:
    tbl = loader.load_freedman_2025_table2()
    return {_canon_host(h): (float(rec["mu_bar"]), float(rec["sigma_bar"]))
            for h, rec in tbl["hosts"].items()}


def _freedman_2019_table3_by_sn(loader: DataLoader) -> Dict[str, Dict[str, float]]:
    """Per-SN calibrator records from Freedman 2019 Table 3."""
    import pandas as pd
    from pathlib import Path
    p = Path("data") / "catalogs" / "freedman_2019_table3.csv"
    if not p.exists():
        raise DataUnavailableError(f"Freedman 2019 Table 3 CSV missing at {p}")
    df = pd.read_csv(p, comment="#")
    out: Dict[str, Dict[str, float]] = {}
    for _, row in df.iterrows():
        out[_canon_sn(str(row["SN"]))] = {
            "host_canon": str(row["host_canon"]),
            "mu_TRGB": _f(row.get("mu_TRGB")),
            "sigma_T": _f(row.get("sigma_T")),
            "m_B_CSP": _f(row.get("m_B_CSP")),
            "sigma_B_CSP": _f(row.get("sigma_B_CSP")),
            "m_B_SuperCal": _f(row.get("m_B_SuperCal")),
            "sigma_B_SC": _f(row.get("sigma_B_SC")),
        }
    return out


def _f(x) -> float:
    import math
    import pandas as pd
    try:
        if x is None or (isinstance(x, float) and math.isnan(x)):
            return float("nan")
        if isinstance(x, str) and not x.strip():
            return float("nan")
        v = float(x)
        if pd.isna(v):
            return float("nan")
        return v
    except Exception:
        return float("nan")


# Hubble-flow z cuts (Freedman 2019 §6.3).
HUBBLE_FLOW_Z_MIN: float = 0.023
HUBBLE_FLOW_Z_MAX: float = 0.15


def _apply_hubble_flow_cut(
    zcmb: np.ndarray, *others: np.ndarray,
    z_min: float = HUBBLE_FLOW_Z_MIN, z_max: float = HUBBLE_FLOW_Z_MAX,
) -> Tuple[np.ndarray, ...]:
    mask = (zcmb >= z_min) & (zcmb <= z_max)
    return (zcmb[mask],) + tuple(a[mask] for a in others)


# =============================================================================
# Plan builders — 8 entries (4 systems × 2 cases)
# =============================================================================


def _plan_uddin(
    loader: DataLoader, case: str, system: str, system_label: str,
    flow_sample_filter: str, notes: str,
    published: Tuple[float, float, float],
    f25_mu: Optional[Dict[str, Tuple[float, float]]] = None,
) -> ChainPlan:
    uddin = loader.load_uddin_h0csp_trgb_dataset()
    inputs = build_uddin_inputs_from_loader_dataset(
        uddin,
        flow_sample_filter=flow_sample_filter,
        case_b_host_mu_override=f25_mu,
    )
    return ChainPlan(
        case=case,
        system=system,
        system_label=system_label,
        mode="uddin_8param",
        published_target_H0=published[0],
        published_sigma_stat=published[1],
        published_sigma_sys=published[2],
        notes=notes,
        uddin_inputs=inputs,
    )


def _plan_simple_supercal(
    loader: DataLoader, case: str,
    mu_override: Optional[Dict[str, Tuple[float, float]]] = None,
    published: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ChainPlan:
    f19 = _freedman_2019_table3_by_sn(loader)
    cal_sn, cal_host, cal_mB, cal_sigma, cal_mu, cal_sigma_mu = [], [], [], [], [], []
    for sn, rec in f19.items():
        mB = rec["m_B_SuperCal"]; sigma = rec["sigma_B_SC"]
        if not (np.isfinite(mB) and np.isfinite(sigma)):
            continue
        hc = _canon_host(rec["host_canon"])
        if mu_override is not None:
            if hc not in mu_override:
                continue
            mu, smu = mu_override[hc]
        else:
            mu, smu = rec["mu_TRGB"], rec["sigma_T"]
            if not (np.isfinite(mu) and np.isfinite(smu)):
                continue
        cal_sn.append(sn); cal_host.append(hc)
        cal_mB.append(mB); cal_sigma.append(sigma)
        cal_mu.append(mu); cal_sigma_mu.append(smu)

    if len(cal_sn) < 3:
        raise DataUnavailableError(
            f"{case}/supercal: fewer than 3 calibrators after matching."
        )

    pantheon_2018 = loader.load_pantheon_2018()
    cal_norm = {_canon_sn(s) for s in cal_sn}
    pan_names = np.asarray([_canon_sn(str(n)) for n in pantheon_2018["SN_name"]])
    not_cal = ~np.isin(pan_names, list(cal_norm))
    zcmb, mBf, sigmaf, snf = _apply_hubble_flow_cut(
        pantheon_2018["zcmb"][not_cal], pantheon_2018["mb"][not_cal],
        pantheon_2018["dmb"][not_cal], pan_names[not_cal],
    )

    data = SNChainData(
        case=case, system="supercal",
        system_label=f"{case.replace('_', ' ').title()} / SuperCal "
                     f"(F19 Table 3 m_B^SC + Pantheon 2018 flow)",
        calibrator_sn_names=np.asarray(cal_sn),
        calibrator_hosts=np.asarray(cal_host),
        calibrator_mB=np.asarray(cal_mB, dtype=float),
        calibrator_sigma_mB=np.asarray(cal_sigma, dtype=float),
        calibrator_mu_TRGB=np.asarray(cal_mu, dtype=float),
        calibrator_sigma_mu_TRGB=np.asarray(cal_sigma_mu, dtype=float),
        flow_sn_names=np.asarray(snf),
        flow_zcmb=np.asarray(zcmb, dtype=float),
        flow_mB=np.asarray(mBf, dtype=float),
        flow_sigma_mB=np.asarray(sigmaf, dtype=float),
        published_target_H0=published[0],
        published_sigma_stat=published[1],
        published_sigma_sys=published[2],
        notes=(
            "SuperCal photometric system chain: calibrator m_B from "
            "Freedman 2019 Table 3 m_B^SuperCal column; Hubble flow from "
            "Pantheon 2018 (Scolnic 2018), which uses the SuperCal "
            "cross-calibration (Scolnic 2015)."
        ),
    )
    return ChainPlan(
        case=case, system="supercal",
        system_label=data.system_label, mode="simple_1param",
        published_target_H0=published[0],
        published_sigma_stat=published[1],
        published_sigma_sys=published[2],
        notes=data.notes, simple_data=data,
    )


def _plan_simple_pantheon_plus(
    loader: DataLoader, case: str,
    mu_override: Dict[str, Tuple[float, float]],
    published: Tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> ChainPlan:
    pp = loader.load_pantheon_plus()
    cid = np.asarray(pp["CID"])
    is_cal = np.asarray(pp["is_calibrator"], dtype=bool)
    z_hd = np.asarray(pp["z"], dtype=float)
    mu_obs = np.asarray(pp["mu"], dtype=float)
    M_B_ref_sh0es = -19.253                               # Brout 2022
    mB_synth = mu_obs + M_B_ref_sh0es
    sigma_mB_synth = np.sqrt(np.diag(pp["cov"]))

    cal_sn, cal_host, cal_mB, cal_sigma, cal_mu, cal_sigma_mu = [], [], [], [], [], []
    for i in np.where(is_cal)[0]:
        sn = str(cid[i]); host = SN_TO_HOST.get(sn)
        if host is None:
            continue
        hc = _canon_host(host)
        if hc not in mu_override:
            continue
        mu, smu = mu_override[hc]
        cal_sn.append(sn); cal_host.append(hc)
        cal_mB.append(float(mB_synth[i])); cal_sigma.append(float(sigma_mB_synth[i]))
        cal_mu.append(float(mu)); cal_sigma_mu.append(float(smu))

    if len(cal_sn) < 3:
        raise DataUnavailableError(
            f"{case}/pantheon_plus: fewer than 3 SN_TO_HOST ∩ μ-override "
            "matches available."
        )

    flow_mask = ~is_cal
    zcmb, mBf, sigmaf, snf = _apply_hubble_flow_cut(
        z_hd[flow_mask], mB_synth[flow_mask], sigma_mB_synth[flow_mask],
        cid[flow_mask],
    )
    data = SNChainData(
        case=case, system="pantheon_plus",
        system_label=f"{case.replace('_', ' ').title()} / Pantheon+SH0ES "
                     "(Pantheon+ calibrators + flow, case-specific μ)",
        calibrator_sn_names=np.asarray(cal_sn),
        calibrator_hosts=np.asarray(cal_host),
        calibrator_mB=np.asarray(cal_mB, dtype=float),
        calibrator_sigma_mB=np.asarray(cal_sigma, dtype=float),
        calibrator_mu_TRGB=np.asarray(cal_mu, dtype=float),
        calibrator_sigma_mu_TRGB=np.asarray(cal_sigma_mu, dtype=float),
        flow_sn_names=np.asarray(snf),
        flow_zcmb=np.asarray(zcmb, dtype=float),
        flow_mB=np.asarray(mBf, dtype=float),
        flow_sigma_mB=np.asarray(sigmaf, dtype=float),
        published_target_H0=published[0],
        published_sigma_stat=published[1],
        published_sigma_sys=published[2],
        notes=(
            "Pantheon+SH0ES chain: Pantheon+ IS_CALIBRATOR rows cross-"
            "matched by SN name to the case-specific TRGB distance "
            "(Freedman 2019 Table 3 for Case A, Freedman 2025 Table 2 for "
            "Case B). Flow: Pantheon+ non-calibrator rows in [0.023, 0.15]."
        ),
    )
    return ChainPlan(
        case=case, system="pantheon_plus",
        system_label=data.system_label, mode="simple_1param",
        published_target_H0=published[0],
        published_sigma_stat=published[1],
        published_sigma_sys=published[2],
        notes=data.notes, simple_data=data,
    )


# ----- Per-case dispatchers ---------------------------------------------------


PUBLISHED_CASE_A = (69.8, 0.8, 1.7)
PUBLISHED_CASE_B = (70.39, 1.22, 1.33)
ALL_SYSTEMS: Tuple[str, ...] = ("csp_i", "csp_ii", "supercal", "pantheon_plus")


def build_chain_for(case: str, system: str, loader: DataLoader) -> ChainPlan:
    """Build the ChainPlan for one (case, system) pair."""
    if case == "case_a":
        if system == "csp_i":
            return _plan_uddin(
                loader, "case_a", "csp_i",
                "Case A / CSP-I (Uddin 8-param, LMC-anchored μ)",
                flow_sample_filter="CSPI",
                notes=(
                    "Uddin 2023 8-parameter SNooPy MCMC on unified "
                    "B_trgb_update3.csv, restricted to the CSP-I era "
                    "flow sub-sample. Calibrators: Uddin's 20-host TRGB "
                    "block with LMC-anchored μ_TRGB (Freedman 2019 scale)."
                ),
                published=PUBLISHED_CASE_A,
            )
        if system == "csp_ii":
            return _plan_uddin(
                loader, "case_a", "csp_ii",
                "Case A / CSP-II (Uddin 8-param, LMC-anchored μ)",
                flow_sample_filter="CSPII",
                notes=(
                    "Uddin 8-parameter MCMC on the CSP-II era flow "
                    "sub-sample with the same LMC-anchored calibrator "
                    "block as Case A / CSP-I."
                ),
                published=PUBLISHED_CASE_A,
            )
        if system == "supercal":
            return _plan_simple_supercal(loader, "case_a", mu_override=None,
                                         published=PUBLISHED_CASE_A)
        if system == "pantheon_plus":
            return _plan_simple_pantheon_plus(
                loader, "case_a",
                mu_override=_per_host_mu_freedman_2019(loader),
                published=PUBLISHED_CASE_A,
            )
    elif case == "case_b":
        f25 = _per_host_mu_freedman_2025(loader)
        if system == "csp_i":
            return _plan_uddin(
                loader, "case_b", "csp_i",
                "Case B / CSP-I (Uddin 8-param, NGC 4258-anchored μ)",
                flow_sample_filter="CSPI",
                notes=(
                    "Uddin 2023 8-parameter MCMC on the CSP-I era flow "
                    "sub-sample, with the calibrator block restricted to "
                    "hosts present in Freedman 2025 Table 2 and re-anchored "
                    "to the NGC 4258 scale (μ_bar from that table)."
                ),
                published=PUBLISHED_CASE_B, f25_mu=f25,
            )
        if system == "csp_ii":
            return _plan_uddin(
                loader, "case_b", "csp_ii",
                "Case B / CSP-II (Uddin 8-param, NGC 4258-anchored μ)",
                flow_sample_filter="CSPII",
                notes=(
                    "Uddin 2023 8-parameter MCMC on the CSP-II era flow, "
                    "calibrators re-anchored to NGC 4258 via Freedman 2025 "
                    "Table 2."
                ),
                published=PUBLISHED_CASE_B, f25_mu=f25,
            )
        if system == "supercal":
            return _plan_simple_supercal(loader, "case_b", mu_override=f25,
                                         published=PUBLISHED_CASE_B)
        if system == "pantheon_plus":
            return _plan_simple_pantheon_plus(loader, "case_b",
                                              mu_override=f25,
                                              published=PUBLISHED_CASE_B)
    raise ValueError(f"Unknown (case, system) = ({case!r}, {system!r})")


# =============================================================================
# Orchestrator
# =============================================================================


ChainResult = Union[SNChainResult, UddinChainResult]


def _run_one(
    plan: ChainPlan,
    settings: MCMCSettings,
    chain_out_path: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> ChainResult:
    if plan.mode == "uddin_8param":
        return run_uddin_csp_chain(
            plan.uddin_inputs, settings,
            case=plan.case, system=plan.system, system_label=plan.system_label,
            published_target_H0=plan.published_target_H0,
            published_sigma_stat=plan.published_sigma_stat,
            published_sigma_sys=plan.published_sigma_sys,
            notes=plan.notes,
            chain_out_path=chain_out_path, log_fn=log_fn,
        )
    elif plan.mode == "simple_1param":
        res = run_sn_chain(
            plan.simple_data, settings,
            chain_out_path=chain_out_path, log_fn=log_fn,
        )
        return res
    else:
        raise ValueError(f"Unknown chain mode: {plan.mode!r}")


def run_all_chains(
    case: str,
    loader: DataLoader,
    settings: MCMCSettings,
    chains_dir: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Dict[str, object]]:
    """Run all 4 system chains for one case. Returns a dict of dicts."""
    _log = log_fn or (lambda m: logger.info(m))
    out: Dict[str, Dict[str, object]] = {}
    for system in ALL_SYSTEMS:
        try:
            plan = build_chain_for(case, system, loader)
        except DataUnavailableError as exc:
            _log(f"[{case}/{system}] SKIPPED — {exc}")
            out[system] = {
                "case": case,
                "system": system,
                "error": str(exc),
                "skipped": True,
            }
            continue
        chain_out = None
        if chains_dir is not None:
            chain_out = Path(chains_dir) / f"{case}_{system}.npz"
        try:
            result = _run_one(plan, settings, chain_out_path=chain_out, log_fn=_log)
            out[system] = result.as_dict()
            out[system]["mode"] = plan.mode
        except Exception as exc:
            _log(f"[{case}/{system}] FAILED — {type(exc).__name__}: {exc}")
            out[system] = {
                "case": case, "system": system,
                "error": f"{type(exc).__name__}: {exc}",
                "failed": True,
            }
    return out


def run_all_chains_both_cases(
    loader: DataLoader,
    settings: MCMCSettings,
    chains_dir: Optional[Path] = None,
    log_fn: Optional[Callable[[str], None]] = None,
) -> Dict[str, Dict[str, Dict[str, object]]]:
    """Run the full 8-chain matrix. Returns ``{case: {system: result_dict}}``."""
    return {
        "case_a": run_all_chains("case_a", loader, settings, chains_dir, log_fn),
        "case_b": run_all_chains("case_b", loader, settings, chains_dir, log_fn),
    }


__all__ = [
    "ALL_SYSTEMS",
    "ChainPlan",
    "build_chain_for",
    "run_all_chains",
    "run_all_chains_both_cases",
]
