"""Full-calibrator-sample chain factories — corrected per audit Recommendation 1.

The original :mod:`pipeline.trgb_comparative.sn_chain_factories` operates on
the intersection of Uddin 2023's TRGB calibrator subset with each Freedman
case's published TRGB hosts, which silently reduces Case B chains from the
published 24-SN augmented sample down to 7 calibrators. That intersection
logic was not part of either Freedman paper's published methodology and
materially biases the H₀ posterior.

This module provides factories that operate on the **full calibrator sample
published by each Freedman paper**:

- **Case A (target H₀ = 69.8 km/s/Mpc)**: 18-SN F2019 Table 3 sample
  (15 unique hosts), with TRGB distances from F2019 column
  `mu_TRGB^F19,F21`. Uddin's `calibrators_trgb_f19.csv` reproduces this
  sample exactly; SNooPy parameters come from `B_all_noj21.csv`.
- **Case B primary (target H₀ = 70.39 km/s/Mpc)**: 24-SN F2025 Table 3
  augmented sample (20 unique hosts), with TRGB distances from F2025
  column `mu_TRGB^CCHP` (the inverse-variance-weighted HST+JWST combined
  distance). All 24 SNe have full SNooPy parameters in Uddin's
  `B_all_noj21.csv`.
- **Case B sensitivity variant (target H₀ = 68.81 km/s/Mpc)**: 11-SN F2025
  Table 2 JWST-only subset (10 unique hosts), with TRGB distances from
  F2025 column `mu_bar` (TRGB+JAGB averaged JWST distances).

Per-photometric-system coverage:

================  ==========  ==========  ==========  ==========
System            Case A      Case B aug  Case B JWST Coverage notes
================  ==========  ==========  ==========  ==========
CSP-I/II (Uddin)  18/18       24/24       11/11       Uddin B_all_noj21 has all SNooPy params
SuperCal (F19 SC) 14/18       14/24       6/11        F19 Table 3 m_B^SuperCal column; covered by Hoyt augmented_N=14
Pantheon+SH0ES    11/18       16/24       9/11        Pantheon+ IS_CALIBRATOR rows
================  ==========  ==========  ==========  ==========

The factories explicitly report coverage gaps; a chain with reduced N is
labelled `_partial` or carries a `coverage_notes` field. No silent
reduction.

Naming convention: every factory returns a :class:`ChainPlan` with
``case`` = `case_a` / `case_b` / `case_b_jwst_only` and ``system`` matching
the SN photometric system. The corresponding chain output filenames carry
a `_full` suffix to distinguish them from the legacy intersection chains.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.loader import DataLoader, DataUnavailableError

from .data_loaders import SN_TO_HOST
from .sn_chain import SNChainData
from .sn_chain_factories import (
    ChainPlan,
    HUBBLE_FLOW_Z_MAX,
    HUBBLE_FLOW_Z_MIN,
    _apply_hubble_flow_cut,
    _canon_host,
    _canon_sn,
    _f,
)
from .uddin_csp_chain import UddinCSPInputs

logger = logging.getLogger(__name__)


# =============================================================================
# Calibrator extraction utility
# =============================================================================


@dataclass(frozen=True)
class CalibratorSpec:
    """One calibrator: (SN, host) plus the canonical TRGB distance to use."""
    sn_name: str           # without the "SN" prefix (e.g. "1981B")
    host_canon: str        # canonical host name (e.g. "NGC 4536")
    mu_TRGB: float
    sigma_mu_TRGB: float


def _f2019_calibrators(loader: DataLoader) -> List[CalibratorSpec]:
    """18-SN Freedman 2019 Table 3 TRGB-calibrator sample."""
    f19 = loader.load_freedman_2019_table3()
    out: List[CalibratorSpec] = []
    # The transcribed dict from the loader keys hosts; we need per-SN.
    # Re-read the CSV for per-row access.
    from pathlib import Path
    df = pd.read_csv(Path("data") / "catalogs" / "freedman_2019_table3.csv",
                     comment='#')
    for _, row in df.iterrows():
        mu = _f(row.get('mu_TRGB'))
        sigma = _f(row.get('sigma_T'))
        if not (np.isfinite(mu) and np.isfinite(sigma)):
            continue                                     # Cepheid-only row
        out.append(CalibratorSpec(
            sn_name=_canon_sn(str(row['SN'])),
            host_canon=_canon_host(str(row['host_canon'])),
            mu_TRGB=float(mu),
            sigma_mu_TRGB=float(sigma),
        ))
    return out


def _f2025_augmented_calibrators(loader: DataLoader) -> List[CalibratorSpec]:
    """24-SN Freedman 2025 Table 3 augmented HST+JWST TRGB sample."""
    t3 = loader.load_freedman_2025_table3()
    aug = t3['augmented']
    return [
        CalibratorSpec(
            sn_name=_canon_sn(str(sn)),
            host_canon=_canon_host(str(hc)),
            mu_TRGB=float(mu),
            sigma_mu_TRGB=float(sigma),
        )
        for sn, hc, mu, sigma in zip(
            aug['SN_name'], aug['host_canon'],
            aug['mu_TRGB_CCHP'], aug['sigma_TRGB_CCHP'],
        )
    ]


def _f2025_jwst_only_calibrators(loader: DataLoader) -> List[CalibratorSpec]:
    """11-SN F2025 Table 2 JWST-only sample (TRGB+JAGB averaged μ_bar)."""
    t2 = loader.load_freedman_2025_table2()
    out: List[CalibratorSpec] = []
    # Loader returns hosts dict; we need per-SN. Re-read the CSV.
    from pathlib import Path
    df = pd.read_csv(Path("data") / "catalogs" / "freedman_2025_table2.csv",
                     comment='#')
    for _, row in df.iterrows():
        out.append(CalibratorSpec(
            sn_name=_canon_sn(str(row['SN'])),
            host_canon=_canon_host(str(row['host_canon'])),
            mu_TRGB=float(row['mu_bar']),
            sigma_mu_TRGB=float(row['sigma_bar']),
        ))
    return out


# =============================================================================
# Coverage report
# =============================================================================


@dataclass
class CoverageReport:
    """Per-(case, system) calibrator coverage summary."""
    case: str
    system: str
    requested_cal_count: int           # full calibrator sample size
    matched_cal_count: int             # found in this photometric system
    missing_sn_names: List[str]        # not in this system
    notes: str = ""


# =============================================================================
# Build Uddin CSP inputs (CSP-I or CSP-II) on a full calibrator sample
# =============================================================================


def build_uddin_inputs_full(
    loader: DataLoader,
    cal_specs: List[CalibratorSpec],
    flow_sample_filter: str,
) -> Tuple[UddinCSPInputs, CoverageReport]:
    """Assemble UddinCSPInputs for the given calibrator sample + CSP flow filter.

    For each SN in `cal_specs`, pull SNooPy fit parameters (Mmax, st, BV,
    eMmax, est, eBV, EBVmw, covMs, covBV_M, m_hostmass, ml, mu_hostmass)
    from Uddin 2023 ``B_all_noj21.csv``. The ``covBVs`` cross-covariance
    is not present in the broad-sample file; rows pulled from there get
    ``covBVs = 0`` (a small approximation since the term enters the
    variance only as ``2·β·fac·covBVs ~ 2·2.87·1·1e-5 ~ 6e-5 mag^2``).

    The ``flow_sample_filter`` selects which CSP era of the Hubble-flow
    sub-sample to use (`'CSPI'` or `'CSPII'`).
    """
    # Load Uddin's full B_all sample for SNooPy params per SN.
    base = loader.downloaded_data_dir / 'uddin_h0csp'
    flow_path = base / 'B_all_noj21.csv'
    udf_full = pd.read_csv(flow_path, sep=r'\s+')
    udf_full['sn_clean'] = udf_full['sn'].astype(str).str.replace('SN', '', regex=False)

    # Also load B_trgb_update3 to recover covBVs for the SNe that have it.
    trgb_path = base / 'B_trgb_update3.csv'
    trgb_df = pd.read_csv(trgb_path)
    trgb_df['sn_clean'] = trgb_df['sn'].astype(str).str.replace('SN', '', regex=False)
    covbvs_lookup = dict(zip(trgb_df['sn_clean'], trgb_df['covBVs']))

    # Build per-SN cal arrays for the 8-param likelihood input.
    matched: List[Dict[str, float]] = []
    missing: List[str] = []
    for spec in cal_specs:
        rows = udf_full[udf_full['sn_clean'] == spec.sn_name]
        if len(rows) == 0:
            missing.append(spec.sn_name)
            continue
        r = rows.iloc[0]                                  # Uddin lists each SN once
        cov_bvs = float(covbvs_lookup.get(spec.sn_name, 0.0))
        matched.append({
            'sn_name': spec.sn_name,
            'host_canon': spec.host_canon,
            'zhel': float(r['zhel']),
            'zcmb': float(r['zcmb']),
            'st': float(r['st']),
            'est': float(r['est']),
            'Mmax': float(r['Mmax']),
            'eMmax': float(r['eMmax']),
            'BV': float(r['BV']),
            'eBV': float(r['eBV']),
            'covMs': float(r['covMs']),
            'covBV_M': float(r['covBV_M']),
            'covBVs': cov_bvs,                            # 0 if not in B_trgb_update3
            'm_hostmass': float(r['m']),
            'ml_hostmass': float(r['ml']),
            'mu_hostmass': float(r['mu']),
            'mu_TRGB': float(spec.mu_TRGB),               # use Freedman's distance
            'sigma_mu_TRGB': float(spec.sigma_mu_TRGB),
        })

    coverage = CoverageReport(
        case='?', system='?',                             # filled by caller
        requested_cal_count=len(cal_specs),
        matched_cal_count=len(matched),
        missing_sn_names=missing,
    )

    if len(matched) < 3:
        raise DataUnavailableError(
            f"Uddin full-cal chain: only {len(matched)} of "
            f"{len(cal_specs)} calibrators have SNooPy parameters in "
            f"Uddin B_all_noj21.csv. Missing: {missing}"
        )

    # Hubble-flow block: same filter as before — restrict to the requested CSP era.
    if flow_sample_filter not in ('CSPI', 'CSPII', 'both'):
        raise ValueError(f"flow_sample_filter must be 'CSPI'/'CSPII'/'both'")
    flow_full = udf_full.copy()
    if flow_sample_filter != 'both':
        flow_full = flow_full[flow_full['sample'] == flow_sample_filter]
    # Drop calibrator rows from the flow.
    cal_sn_clean = {c.sn_name for c in cal_specs}
    flow_full = flow_full[~flow_full['sn_clean'].isin(cal_sn_clean)]
    # Apply Hubble-flow z cuts.
    zcmb = flow_full['zcmb'].to_numpy(dtype=float)
    z_mask = (zcmb >= HUBBLE_FLOW_Z_MIN) & (zcmb <= HUBBLE_FLOW_Z_MAX)
    flow_full = flow_full[z_mask].reset_index(drop=True)

    # Assemble concatenated calibrator + flow arrays in the
    # UddinCSPInputs schema.
    n_cal = len(matched)
    n_flow = len(flow_full)
    is_cal = np.concatenate([np.ones(n_cal, dtype=bool), np.zeros(n_flow, dtype=bool)])

    def _vec(field: str, dtype=float) -> np.ndarray:
        cal_part = np.asarray([row[field] for row in matched], dtype=dtype)
        if field in ('zhel', 'zcmb', 'st', 'est', 'Mmax', 'eMmax', 'BV',
                     'eBV', 'covMs', 'covBV_M'):
            flow_col = field
        elif field == 'covBVs':
            # Not in B_all_noj21 — set to 0 for flow rows.
            return np.concatenate([cal_part, np.zeros(n_flow, dtype=dtype)])
        elif field == 'm_hostmass':
            flow_col = 'm'
        elif field == 'ml_hostmass':
            flow_col = 'ml'
        elif field == 'mu_hostmass':
            flow_col = 'mu'
        elif field == 'mu_TRGB':
            return np.concatenate([cal_part, np.zeros(n_flow, dtype=dtype)])  # ignored for flow
        elif field == 'sigma_mu_TRGB':
            return np.concatenate([cal_part, np.full(n_flow, -1.0)])  # flagged
        else:
            raise ValueError(f"Unknown field {field}")
        flow_part = flow_full[flow_col].to_numpy(dtype=dtype)
        return np.concatenate([cal_part, flow_part])

    sn_names = np.concatenate([
        np.asarray([row['sn_name'] for row in matched]),
        flow_full['sn_clean'].to_numpy(),
    ])

    em = 0.5 * ((_vec('mu_hostmass') - _vec('m_hostmass'))
                + (_vec('m_hostmass') - _vec('ml_hostmass')))
    em = np.where(em == 0.0, 0.005, em)

    median_mhost = float(np.median(_vec('m_hostmass')[np.isfinite(_vec('m_hostmass'))]))

    # dist column: TRGB μ for calibrators, arbitrary for flow.
    dist = _vec('mu_TRGB')                                # for cals: from Freedman; flow: 0
    edist = _vec('sigma_mu_TRGB')
    # For flow rows, mu_TRGB is unused (likelihood uses distmod model);
    # fill with sentinel.

    inputs = UddinCSPInputs(
        sn_names=sn_names,
        zhel=_vec('zhel'),
        zcmb=_vec('zcmb'),
        st=_vec('st'),
        est=_vec('est'),
        Mmax=_vec('Mmax'),
        eMmax=_vec('eMmax'),
        BV=_vec('BV'),
        eBV=_vec('eBV'),
        covMs=_vec('covMs'),
        covBV_M=_vec('covBV_M'),
        covBVs=_vec('covBVs'),
        m_hostmass=_vec('m_hostmass'),
        ml_hostmass=_vec('ml_hostmass'),
        mu_hostmass=_vec('mu_hostmass'),
        dist=dist,
        edist=edist,
        is_cal=is_cal,
        median_mhost=median_mhost,
        em_hostmass=em,
    )
    return inputs, coverage


# =============================================================================
# Build SuperCal SNChainData on a full calibrator sample
# =============================================================================


def build_supercal_inputs_full(
    loader: DataLoader,
    cal_specs: List[CalibratorSpec],
) -> Tuple[SNChainData, CoverageReport]:
    """SuperCal calibrators: F19 Table 3 m_B^SC + Pantheon 2018 Hubble flow.

    Coverage gap: only those SNe in `cal_specs` whose F19 Table 3 row
    carries an `m_B_SuperCal` value enter the chain. The remaining
    SNe are reported in the CoverageReport as missing.
    """
    from pathlib import Path
    f19 = pd.read_csv(Path("data") / "catalogs" / "freedman_2019_table3.csv",
                      comment='#')
    f19['sn_clean'] = f19['SN'].astype(str)
    f19_lookup = {row['sn_clean']: row for _, row in f19.iterrows()}

    matched_sn, matched_host, matched_mB, matched_sigma_mB = [], [], [], []
    matched_mu, matched_sigma_mu = [], []
    missing: List[str] = []
    for spec in cal_specs:
        rec = f19_lookup.get(spec.sn_name)
        if rec is None:
            missing.append(spec.sn_name)
            continue
        mB = _f(rec.get('m_B_SuperCal'))
        sigma = _f(rec.get('sigma_B_SC'))
        if not (np.isfinite(mB) and np.isfinite(sigma)):
            missing.append(spec.sn_name)
            continue
        matched_sn.append(spec.sn_name)
        matched_host.append(spec.host_canon)
        matched_mB.append(mB)
        matched_sigma_mB.append(sigma)
        matched_mu.append(spec.mu_TRGB)
        matched_sigma_mu.append(spec.sigma_mu_TRGB)

    coverage = CoverageReport(
        case='?', system='supercal',
        requested_cal_count=len(cal_specs),
        matched_cal_count=len(matched_sn),
        missing_sn_names=missing,
        notes=("SuperCal chain only sees SNe whose F2019 Table 3 row "
               "has an m_B^SuperCal value; this is the source of the "
               "Hoyt augmented_N=14 vs. 24-SN augmented sample size."),
    )

    if len(matched_sn) < 3:
        raise DataUnavailableError(
            f"SuperCal full-cal chain: only {len(matched_sn)} of "
            f"{len(cal_specs)} calibrators have F19 m_B^SuperCal."
        )

    # Pantheon 2018 Hubble-flow block, excluding calibrators by name.
    p2018 = loader.load_pantheon_2018()
    pan_names = np.asarray([_canon_sn(str(n)) for n in p2018['SN_name']])
    cal_set = {_canon_sn(s) for s in matched_sn}
    not_cal = ~np.isin(pan_names, list(cal_set))
    zcmb, mBf, sigmaf, snf = _apply_hubble_flow_cut(
        p2018['zcmb'][not_cal], p2018['mb'][not_cal],
        p2018['dmb'][not_cal], pan_names[not_cal],
    )

    data = SNChainData(
        case='?', system='supercal',
        system_label='SuperCal full-cal (F19 m_B^SC + Pantheon 2018 flow)',
        calibrator_sn_names=np.asarray(matched_sn),
        calibrator_hosts=np.asarray(matched_host),
        calibrator_mB=np.asarray(matched_mB, dtype=float),
        calibrator_sigma_mB=np.asarray(matched_sigma_mB, dtype=float),
        calibrator_mu_TRGB=np.asarray(matched_mu, dtype=float),
        calibrator_sigma_mu_TRGB=np.asarray(matched_sigma_mu, dtype=float),
        flow_sn_names=np.asarray(snf),
        flow_zcmb=np.asarray(zcmb, dtype=float),
        flow_mB=np.asarray(mBf, dtype=float),
        flow_sigma_mB=np.asarray(sigmaf, dtype=float),
        notes=coverage.notes,
    )
    return data, coverage


# =============================================================================
# Build Pantheon+SH0ES SNChainData on a full calibrator sample
# =============================================================================


def build_pantheon_plus_inputs_full(
    loader: DataLoader,
    cal_specs: List[CalibratorSpec],
) -> Tuple[SNChainData, CoverageReport]:
    """Pantheon+SH0ES calibrators: IS_CALIBRATOR rows ∩ cal_specs.

    Coverage gap: only those SNe in `cal_specs` that appear as
    IS_CALIBRATOR rows in Pantheon+SH0ES.dat (cross-matched by CID
    via the existing SN_TO_HOST mapping inverse) enter the chain.
    Pantheon+ does not include some F19 hosts (e.g., NGC 1316, NGC 1404).
    """
    pp = loader.load_pantheon_plus()
    cid = np.asarray(pp['CID']).astype(str)
    is_cal = np.asarray(pp['is_calibrator'], dtype=bool)
    z_hd = np.asarray(pp['z'], dtype=float)
    mu_obs = np.asarray(pp['mu'], dtype=float)
    M_B_ref_sh0es = -19.253                                 # Brout 2022
    mB_synth = mu_obs + M_B_ref_sh0es
    sigma_mB_synth = np.sqrt(np.diag(pp['cov']))

    # Pantheon+ uses CID matching the SN name (no "SN" prefix in CIDs).
    pp_cal_lookup: Dict[str, int] = {}
    for i in np.where(is_cal)[0]:
        sn_clean = _canon_sn(str(cid[i]))
        # Pantheon+ may have multiple rows per SN (different surveys) — keep first.
        pp_cal_lookup.setdefault(sn_clean, int(i))

    matched_sn, matched_host, matched_mB, matched_sigma_mB = [], [], [], []
    matched_mu, matched_sigma_mu = [], []
    missing: List[str] = []
    for spec in cal_specs:
        idx = pp_cal_lookup.get(spec.sn_name)
        if idx is None:
            missing.append(spec.sn_name)
            continue
        matched_sn.append(spec.sn_name)
        matched_host.append(spec.host_canon)
        matched_mB.append(float(mB_synth[idx]))
        matched_sigma_mB.append(float(sigma_mB_synth[idx]))
        matched_mu.append(spec.mu_TRGB)
        matched_sigma_mu.append(spec.sigma_mu_TRGB)

    coverage = CoverageReport(
        case='?', system='pantheon_plus',
        requested_cal_count=len(cal_specs),
        matched_cal_count=len(matched_sn),
        missing_sn_names=missing,
        notes=("Pantheon+SH0ES IS_CALIBRATOR rows do not include some "
               "Freedman hosts (e.g., NGC 1316, NGC 1404); those SNe "
               "are reported as missing rather than dropped silently."),
    )

    if len(matched_sn) < 3:
        raise DataUnavailableError(
            f"Pantheon+ full-cal chain: only {len(matched_sn)} of "
            f"{len(cal_specs)} calibrators are Pantheon+ IS_CALIBRATOR rows."
        )

    flow_mask = ~is_cal
    zcmb, mBf, sigmaf, snf = _apply_hubble_flow_cut(
        z_hd[flow_mask], mB_synth[flow_mask], sigma_mB_synth[flow_mask],
        cid[flow_mask],
    )

    data = SNChainData(
        case='?', system='pantheon_plus',
        system_label='Pantheon+SH0ES full-cal (IS_CALIBRATOR ∩ Freedman sample)',
        calibrator_sn_names=np.asarray(matched_sn),
        calibrator_hosts=np.asarray(matched_host),
        calibrator_mB=np.asarray(matched_mB, dtype=float),
        calibrator_sigma_mB=np.asarray(matched_sigma_mB, dtype=float),
        calibrator_mu_TRGB=np.asarray(matched_mu, dtype=float),
        calibrator_sigma_mu_TRGB=np.asarray(matched_sigma_mu, dtype=float),
        flow_sn_names=np.asarray(snf),
        flow_zcmb=np.asarray(zcmb, dtype=float),
        flow_mB=np.asarray(mBf, dtype=float),
        flow_sigma_mB=np.asarray(sigmaf, dtype=float),
        notes=coverage.notes,
    )
    return data, coverage


# =============================================================================
# Top-level chain plan factories — full-cal versions
# =============================================================================


PUBLISHED_CASE_A_FULL = (69.8, 0.8, 1.7)
PUBLISHED_CASE_B_AUG_FULL = (70.39, 1.22, 1.33)
PUBLISHED_CASE_B_JWST_FULL = (68.81, 1.80, 0.0)            # JWST-only Table 4 row


def _make_chain_plan(
    case: str, system: str, system_label: str, mode: str,
    published: Tuple[float, float, float],
    notes: str,
    uddin_inputs: Optional[UddinCSPInputs] = None,
    simple_data: Optional[SNChainData] = None,
) -> ChainPlan:
    if simple_data is not None:
        # Inject the case/system into the SNChainData's case field for output naming.
        simple_data = SNChainData(
            case=case, system=system,
            system_label=system_label,
            calibrator_sn_names=simple_data.calibrator_sn_names,
            calibrator_hosts=simple_data.calibrator_hosts,
            calibrator_mB=simple_data.calibrator_mB,
            calibrator_sigma_mB=simple_data.calibrator_sigma_mB,
            calibrator_mu_TRGB=simple_data.calibrator_mu_TRGB,
            calibrator_sigma_mu_TRGB=simple_data.calibrator_sigma_mu_TRGB,
            flow_sn_names=simple_data.flow_sn_names,
            flow_zcmb=simple_data.flow_zcmb,
            flow_mB=simple_data.flow_mB,
            flow_sigma_mB=simple_data.flow_sigma_mB,
            published_target_H0=published[0],
            published_sigma_stat=published[1],
            published_sigma_sys=published[2],
            notes=notes,
        )
    return ChainPlan(
        case=case, system=system, system_label=system_label, mode=mode,
        published_target_H0=published[0],
        published_sigma_stat=published[1],
        published_sigma_sys=published[2],
        notes=notes,
        uddin_inputs=uddin_inputs,
        simple_data=simple_data,
    )


def build_chain_full(
    case: str, system: str, loader: DataLoader,
) -> Tuple[ChainPlan, CoverageReport]:
    """Build the full-calibrator-sample ChainPlan + coverage report.

    case ∈ {'case_a', 'case_b', 'case_b_jwst_only'}
    system ∈ {'csp_i', 'csp_ii', 'supercal', 'pantheon_plus'}
    """
    if case == 'case_a':
        cal_specs = _f2019_calibrators(loader)            # 18 SNe
        published = PUBLISHED_CASE_A_FULL
        anchor_label = 'F2019 Table 3 mu_TRGB^F19,F21'
    elif case == 'case_b':
        cal_specs = _f2025_augmented_calibrators(loader)  # 24 SNe
        published = PUBLISHED_CASE_B_AUG_FULL
        anchor_label = 'F2025 Table 3 mu_TRGB^CCHP (augmented HST+JWST)'
    elif case == 'case_b_jwst_only':
        cal_specs = _f2025_jwst_only_calibrators(loader)  # 11 SNe
        published = PUBLISHED_CASE_B_JWST_FULL
        anchor_label = 'F2025 Table 2 mu_bar (JWST-only TRGB+JAGB)'
    else:
        raise ValueError(f"Unknown case: {case!r}")

    if system in ('csp_i', 'csp_ii'):
        flow_filter = 'CSPI' if system == 'csp_i' else 'CSPII'
        inputs, coverage = build_uddin_inputs_full(loader, cal_specs, flow_filter)
        coverage = CoverageReport(
            case=case, system=system,
            requested_cal_count=coverage.requested_cal_count,
            matched_cal_count=coverage.matched_cal_count,
            missing_sn_names=coverage.missing_sn_names,
            notes=(f"Uddin 8-parameter likelihood; calibrator distances "
                   f"from {anchor_label}; SNooPy params from Uddin "
                   f"B_all_noj21.csv (covBVs=0 for SNe absent from "
                   f"B_trgb_update3)."),
        )
        plan = _make_chain_plan(
            case=case, system=system,
            system_label=f"{case} / {system} (full-cal: {anchor_label})",
            mode='uddin_8param', published=published,
            notes=coverage.notes,
            uddin_inputs=inputs,
        )
        return plan, coverage

    if system == 'supercal':
        data, coverage = build_supercal_inputs_full(loader, cal_specs)
        coverage = CoverageReport(
            case=case, system=system,
            requested_cal_count=coverage.requested_cal_count,
            matched_cal_count=coverage.matched_cal_count,
            missing_sn_names=coverage.missing_sn_names,
            notes=(f"SuperCal m_B from F2019 Table 3; flow from "
                   f"Pantheon 2018; calibrator distances from "
                   f"{anchor_label}. {coverage.notes}"),
        )
        plan = _make_chain_plan(
            case=case, system=system,
            system_label=f"{case} / supercal (full-cal: {anchor_label})",
            mode='simple_1param', published=published,
            notes=coverage.notes,
            simple_data=data,
        )
        return plan, coverage

    if system == 'pantheon_plus':
        data, coverage = build_pantheon_plus_inputs_full(loader, cal_specs)
        coverage = CoverageReport(
            case=case, system=system,
            requested_cal_count=coverage.requested_cal_count,
            matched_cal_count=coverage.matched_cal_count,
            missing_sn_names=coverage.missing_sn_names,
            notes=(f"Pantheon+SH0ES IS_CALIBRATOR rows; flow from "
                   f"non-calibrator Pantheon+ rows; calibrator distances "
                   f"from {anchor_label}. {coverage.notes}"),
        )
        plan = _make_chain_plan(
            case=case, system=system,
            system_label=f"{case} / pantheon_plus (full-cal: {anchor_label})",
            mode='simple_1param', published=published,
            notes=coverage.notes,
            simple_data=data,
        )
        return plan, coverage

    raise ValueError(f"Unknown system: {system!r}")


def all_chains_full(
    loader: DataLoader,
    *,
    include_jwst_only_sensitivity: bool = True,
) -> Dict[str, Dict[str, Tuple[ChainPlan, CoverageReport]]]:
    """Build all full-calibrator-sample plans + coverage reports.

    Returns ``{case: {system: (ChainPlan, CoverageReport)}}`` for
    cases {'case_a', 'case_b'} (and 'case_b_jwst_only' if
    `include_jwst_only_sensitivity`) and systems
    {'csp_i', 'csp_ii', 'supercal', 'pantheon_plus'}.
    """
    cases = ['case_a', 'case_b']
    if include_jwst_only_sensitivity:
        cases.append('case_b_jwst_only')
    systems = ('csp_i', 'csp_ii', 'supercal', 'pantheon_plus')
    out: Dict[str, Dict[str, Tuple[ChainPlan, CoverageReport]]] = {}
    for case in cases:
        out[case] = {}
        for system in systems:
            try:
                out[case][system] = build_chain_full(case, system, loader)
            except DataUnavailableError as exc:
                logger.warning(f"[{case}/{system}] full-cal plan unavailable: {exc}")
                out[case][system] = (None, CoverageReport(
                    case=case, system=system,
                    requested_cal_count=0, matched_cal_count=0,
                    missing_sn_names=[], notes=f"unavailable: {exc}",
                ))
    return out


__all__ = [
    "CalibratorSpec",
    "CoverageReport",
    "all_chains_full",
    "build_chain_full",
    "build_pantheon_plus_inputs_full",
    "build_supercal_inputs_full",
    "build_uddin_inputs_full",
]
