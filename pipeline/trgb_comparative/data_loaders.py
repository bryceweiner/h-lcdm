"""
Pipeline-level wrappers around data/loader.py for the TRGB analysis.

Returns typed dataclasses rather than raw dicts, matching the pattern in
:mod:`pipeline.expansion_enhancement.data_loaders`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from data.loader import DataLoader, DataUnavailableError

from .distance_ladder import GeometricAnchor


# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TRGBPhotometryField:
    """Photometry of a single field (LMC halo pointing, SN host pointing)."""

    field_id: str
    mag: np.ndarray
    sigma_mag: np.ndarray
    color: Optional[np.ndarray] = None
    sigma_color: Optional[np.ndarray] = None
    flag: Optional[np.ndarray] = None
    published_mu_TRGB: Optional[float] = None
    published_sigma_mu: Optional[float] = None
    metadata: Dict[str, float] = field(default_factory=dict)

    @property
    def n_stars(self) -> int:
        return int(self.mag.size)

    def quality_cut(self) -> "TRGBPhotometryField":
        """Return a copy with only quality-flag-0 stars."""
        if self.flag is None:
            return self
        mask = self.flag == 0
        return TRGBPhotometryField(
            field_id=self.field_id,
            mag=self.mag[mask],
            sigma_mag=self.sigma_mag[mask],
            color=None if self.color is None else self.color[mask],
            sigma_color=None if self.sigma_color is None else self.sigma_color[mask],
            flag=self.flag[mask],
            published_mu_TRGB=self.published_mu_TRGB,
            published_sigma_mu=self.published_sigma_mu,
            metadata=dict(self.metadata),
        )


@dataclass
class TRGBDataBundle:
    """Everything a Freedman methodology module needs, per case."""

    case: str                                                # "case_a" or "case_b"
    anchor: GeometricAnchor
    anchor_fields: Dict[str, TRGBPhotometryField]
    host_fields: Dict[str, TRGBPhotometryField]
    pantheon_plus: Dict[str, np.ndarray]                     # re-exposed for downstream fit
    published_mu_hosts: Dict[str, Tuple[float, float]]       # host -> (mu, sigma)
    host_to_pantheon_indices: Dict[str, List[int]] = field(default_factory=dict)
    # host name -> list of Pantheon+ row indices (absolute, not per-calibrator-subset)
    # Populated via RA/Dec crossmatch; an empty dict means no mapping built.

    tip_source: str = "anand_2022"
    # Source of the per-host published μ_TRGB values that `published_mu_hosts`
    # carries. Valid values: "freedman_2019", "freedman_2025", "anand_2022".
    # The Freedman reproduction branches report this in the output so the
    # reader knows which published H₀ target the reproduction is aiming at.


# ---------------------------------------------------------------------------
# Loader helpers
# ---------------------------------------------------------------------------


def _photometry_from_dict(d: Dict[str, np.ndarray], field_id: str,
                          primary_mag: str, primary_err: str,
                          color_bands: Optional[Tuple[str, str]] = None,
                          metadata: Optional[Dict[str, float]] = None,
                          published_mu: Optional[float] = None,
                          published_sigma: Optional[float] = None,
                          flag_key: str = "flag") -> TRGBPhotometryField:
    mag = np.asarray(d[primary_mag], dtype=float)
    sigma = np.asarray(d[primary_err], dtype=float)
    color = None
    sigma_color = None
    if color_bands is not None:
        b1_mag, b2_mag = color_bands
        if b1_mag in d and b2_mag in d and d[b1_mag] is not None and d[b2_mag] is not None:
            color = np.asarray(d[b1_mag], dtype=float) - np.asarray(d[b2_mag], dtype=float)
    flag = None
    if flag_key in d and d[flag_key] is not None:
        flag = np.asarray(d[flag_key], dtype=int)
    return TRGBPhotometryField(
        field_id=field_id,
        mag=mag,
        sigma_mag=sigma,
        color=color,
        sigma_color=sigma_color,
        flag=flag,
        published_mu_TRGB=published_mu,
        published_sigma_mu=published_sigma,
        metadata=dict(metadata or {}),
    )


def _make_anchor_from_dict(d: Dict[str, object], canonical_name: str) -> GeometricAnchor:
    return GeometricAnchor(
        name=canonical_name,
        mu=float(d["mu"]),
        sigma_mu_stat=float(d["sigma_mu_stat"]),
        sigma_mu_sys=float(d["sigma_mu_sys"]),
        reference=str(d.get("reference", "")),
    )


# Hardcoded SN → host mapping for Pantheon+ calibrators. Covers the
# well-known calibrator set; derived from positional crossmatch of
# Pantheon+SH0ES calibrator RA/Dec against our EDD host coordinate set.
# Used when the Pantheon+ data file does not include explicit host names.
SN_TO_HOST = {
    "2011fe": "NGC 5457", "2012cg": "NGC 4424", "1981B": "NGC 4536",
    "2013aa": "NGC 5643", "2017cbv": "NGC 5643", "2001el": "NGC 1448",
    "2021pit": "NGC 1448", "2011by": "NGC 3972", "1998aq": "NGC 3982",
    "1990N": "NGC 4639", "2005df": "NGC 1559", "2012fr": "NGC 1365",
    "1994ae": "NGC 3370", "2007af": "NGC 5584", "2012ht": "NGC 3447",
    "2013dy": "NGC 7250", "2015F": "NGC 2442", "2002fk": "NGC 1309",
    "1995al": "NGC 3021", "1998dh": "NGC 7541", "2005cf": "NGC 5917",
    "2007sr": "NGC 4038", "2009Y": "NGC 5728", "2011iv": "NGC 1404",
    "2006dd": "NGC 1316",
}


def build_host_to_pantheon_indices(
    pantheon: Dict[str, np.ndarray],
    host_names: List[str],
) -> Dict[str, List[int]]:
    """Crossmatch Pantheon+ calibrator rows to host galaxies.

    Uses the hardcoded :data:`SN_TO_HOST` SN-name → host map. For each
    Pantheon+ calibrator whose ``CID`` is in SN_TO_HOST, the row is
    assigned to that host.

    Returns ``{host: [pantheon_row_indices]}`` for each host that has at
    least one match.
    """
    if "CID" not in pantheon:
        return {}
    cids = np.asarray(pantheon["CID"])
    is_calib = np.asarray(pantheon["is_calibrator"], dtype=bool)
    out: Dict[str, List[int]] = {}
    for idx in np.where(is_calib)[0]:
        sn = str(cids[idx]).strip()
        host = SN_TO_HOST.get(sn)
        if host is None or host not in host_names:
            continue
        out.setdefault(host, []).append(int(idx))
    return out


# ---------------------------------------------------------------------------
# Case loaders
# ---------------------------------------------------------------------------


def load_case_a(
    loader: DataLoader,
    *,
    strict: bool = True,
    tip_source: str = "freedman_2019",
) -> TRGBDataBundle:
    """Freedman 2019/2020 HST, LMC-anchored.

    Parameters
    ----------
    tip_source:
        Source of per-host published μ_TRGB values. ``"freedman_2019"`` is
        the primary reproduction path; it reads from Freedman 2019 Table 3
        transcription. ``"anand_2022"`` uses the independent Anand 2021
        reduction via EDD — reported as a cross-check rather than the
        primary reproduction target. ``"freedman_2025"`` is valid if a
        Case-A-compatible 2025 subset is requested; it reads JWST-era
        distances and is typically reserved for Case B.

    When ``strict`` is False and any data are missing, the loader returns a
    bundle with empty host_fields / anchor_fields so higher-level code can
    proceed (tests mostly) — callers must verify before running MCMC.
    """
    if tip_source not in ("freedman_2019", "freedman_2025", "anand_2022"):
        raise ValueError(f"Unknown tip_source: {tip_source!r}")

    anchor = _make_anchor_from_dict(loader.load_pietrzynski_lmc_distance(), "LMC")

    # Per-host A_F814W from Freedman 2019 Table 1 — authoritative foreground
    # extinction values used by the CCHP TRGB reduction.
    try:
        table1 = loader.load_freedman_2019_table1()
        a_f814w_by_host = {h: v["A_F814W"] for h, v in table1["hosts"].items()}
    except Exception:
        a_f814w_by_host = {}

    # Per-host published μ_TRGB depending on tip_source. Freedman 2019
    # Table 3 is the primary target; Anand 2022 is an independent-reduction
    # cross-check.
    published_by_source: Dict[str, Tuple[float, float]] = {}
    if tip_source == "freedman_2019":
        try:
            f19 = loader.load_freedman_2019_table3()
            for h, rec in f19["hosts"].items():
                published_by_source[h] = (float(rec["mu"]), float(rec["sigma"]))
        except Exception:
            if strict:
                raise
    elif tip_source == "freedman_2025":
        try:
            f25 = loader.load_freedman_2025_table2()
            for h, rec in f25["hosts"].items():
                published_by_source[h] = (float(rec["mu_bar"]), float(rec["sigma_bar"]))
        except Exception:
            if strict:
                raise
    # "anand_2022" is left empty here; it falls back to each host's manifest
    # value (Anand 2021 EDD reduction, which the manifest carries).

    anchor_fields: Dict[str, TRGBPhotometryField] = {}
    host_fields: Dict[str, TRGBPhotometryField] = {}
    published_mu_hosts: Dict[str, Tuple[float, float]] = {}
    try:
        lmc = loader.load_lmc_trgb_hst_photometry()
        unique_fields = np.unique(lmc["field_id"])
        for f_id in unique_fields:
            mask = lmc["field_id"] == f_id
            sub = {
                "F814W": lmc["F814W"][mask],
                "F814W_err": lmc["F814W_err"][mask],
                "F555W": None if lmc["F555W"] is None else lmc["F555W"][mask],
                "F555W_err": None if lmc["F555W_err"] is None else lmc["F555W_err"][mask],
                "flag": lmc["flag"][mask] if lmc["flag"] is not None else None,
            }
            anchor_fields[str(f_id)] = _photometry_from_dict(
                sub, field_id=str(f_id),
                primary_mag="F814W", primary_err="F814W_err",
                color_bands=("F555W", "F814W"),
            )
    except DataUnavailableError:
        if strict:
            raise

    try:
        sn_hst = loader.load_sn_host_trgb_hst_photometry()
        for host, rec in sn_hst["hosts"].items():
            photom = {
                "F814W": rec["F814W"],
                "F814W_err": rec["F814W_err"],
                "F555W": rec.get("F555W"),
                "F555W_err": rec.get("F555W_err"),
                "flag": rec.get("flag"),
            }
            host_metadata: Dict[str, float] = {}
            if host in a_f814w_by_host:
                host_metadata["A_F814W"] = a_f814w_by_host[host]
            # Tip source: freedman_2019 / freedman_2025 override the manifest
            # (Anand 2021) value with the paper's published μ_TRGB.
            chosen_mu, chosen_sigma = (
                rec["published_mu_TRGB"],
                rec["published_sigma_mu"],
            )
            if host in published_by_source:
                chosen_mu, chosen_sigma = published_by_source[host]
            host_fields[host] = _photometry_from_dict(
                photom, field_id=host,
                primary_mag="F814W", primary_err="F814W_err",
                color_bands=("F555W", "F814W"),
                metadata=host_metadata,
                published_mu=chosen_mu,
                published_sigma=chosen_sigma,
            )
            published_mu_hosts[host] = (chosen_mu, chosen_sigma)
    except DataUnavailableError:
        if strict:
            raise

    # Hosts that are in Freedman 2019 Table 3 but NOT in our HST photometry
    # manifest should still appear as host entries for fits that rely on
    # published μ_TRGB rather than the raw photometry. (Edge-detection
    # variants will skip them.)
    if tip_source == "freedman_2019":
        for h, (mu, sigma) in published_by_source.items():
            if h not in host_fields and h != "LMC":
                # Create a stub TRGBPhotometryField with empty photometry
                # arrays so the likelihood can still use the published μ.
                import numpy as _np
                host_fields[h] = TRGBPhotometryField(
                    field_id=h,
                    mag=_np.array([]),
                    sigma_mag=_np.array([]),
                    color=None,
                    sigma_color=None,
                    flag=None,
                    published_mu_TRGB=mu,
                    published_sigma_mu=sigma,
                    metadata={"A_F814W": a_f814w_by_host.get(h, 0.03),
                              "stub_no_photometry": 1.0},
                )
                published_mu_hosts[h] = (mu, sigma)

    pantheon_plus = loader.load_pantheon_plus()
    host_to_pantheon = build_host_to_pantheon_indices(
        pantheon_plus, list(host_fields.keys())
    )

    return TRGBDataBundle(
        case="case_a",
        anchor=anchor,
        anchor_fields=anchor_fields,
        host_fields=host_fields,
        pantheon_plus=pantheon_plus,
        published_mu_hosts=published_mu_hosts,
        host_to_pantheon_indices=host_to_pantheon,
        tip_source=tip_source,
    )


def load_case_b(
    loader: DataLoader,
    *,
    strict: bool = True,
    tip_source: str = "freedman_2025",
) -> TRGBDataBundle:
    """Freedman 2024/2025 JWST, NGC 4258-anchored.

    Parameters
    ----------
    tip_source:
        Source of per-host published μ_TRGB. Default ``"freedman_2025"``
        reads the JWST Table 2 values. ``"anand_2022"`` falls back to the
        manifest's EDD/Anand reduction (HST-era, cross-check only).

    The JWST sample uses F150W as the primary band with F090W−F150W color.
    The NGC 4258 anchor's TRGB photometry for F150W comes from the same
    manifest (treated as one of the 'hosts' for bookkeeping).
    """
    if tip_source not in ("freedman_2019", "freedman_2025", "anand_2022"):
        raise ValueError(f"Unknown tip_source: {tip_source!r}")

    anchor = _make_anchor_from_dict(loader.load_reid_ngc4258_distance(), "NGC_4258")

    published_by_source: Dict[str, Tuple[float, float]] = {}
    if tip_source == "freedman_2025":
        try:
            f25 = loader.load_freedman_2025_table2()
            for h, rec in f25["hosts"].items():
                published_by_source[h] = (float(rec["mu_bar"]), float(rec["sigma_bar"]))
        except Exception:
            if strict:
                raise
    elif tip_source == "freedman_2019":
        try:
            f19 = loader.load_freedman_2019_table3()
            for h, rec in f19["hosts"].items():
                published_by_source[h] = (float(rec["mu"]), float(rec["sigma"]))
        except Exception:
            if strict:
                raise

    host_fields: Dict[str, TRGBPhotometryField] = {}
    anchor_fields: Dict[str, TRGBPhotometryField] = {}
    published_mu_hosts: Dict[str, Tuple[float, float]] = {}
    try:
        sn_jwst = loader.load_sn_host_trgb_jwst_photometry()
        for host, rec in sn_jwst["hosts"].items():
            photom = {
                "F150W": rec["F150W"],
                "F150W_err": rec["F150W_err"],
                "F090W": rec["F090W"],
                "F090W_err": rec["F090W_err"],
                "flag": rec.get("flag"),
            }
            chosen_mu, chosen_sigma = (
                rec["published_mu_TRGB"],
                rec["published_sigma_mu"],
            )
            if host in published_by_source:
                chosen_mu, chosen_sigma = published_by_source[host]
            field = _photometry_from_dict(
                photom, field_id=host,
                primary_mag="F150W", primary_err="F150W_err",
                color_bands=("F090W", "F150W"),
                published_mu=chosen_mu,
                published_sigma=chosen_sigma,
            )
            if host.upper() in ("NGC 4258", "NGC_4258", "NGC4258"):
                anchor_fields[host] = field
            else:
                host_fields[host] = field
                published_mu_hosts[host] = (chosen_mu, chosen_sigma)
    except DataUnavailableError:
        if strict:
            raise

    # For tip_source="freedman_2025", include hosts from Table 2 that have
    # no direct photometry entry (photometry stub with only published μ).
    if tip_source == "freedman_2025":
        import numpy as _np
        for h, (mu, sigma) in published_by_source.items():
            if h == "NGC 4258" or h.upper().replace(" ", "") == "NGC4258":
                continue
            if h not in host_fields:
                host_fields[h] = TRGBPhotometryField(
                    field_id=h,
                    mag=_np.array([]),
                    sigma_mag=_np.array([]),
                    color=None,
                    sigma_color=None,
                    flag=None,
                    published_mu_TRGB=mu,
                    published_sigma_mu=sigma,
                    metadata={"A_F150W": 0.015, "stub_no_photometry": 1.0},
                )
                published_mu_hosts[h] = (mu, sigma)

    pantheon_plus = loader.load_pantheon_plus()
    host_to_pantheon = build_host_to_pantheon_indices(
        pantheon_plus, list(host_fields.keys())
    )

    return TRGBDataBundle(
        case="case_b",
        anchor=anchor,
        anchor_fields=anchor_fields,
        host_fields=host_fields,
        pantheon_plus=pantheon_plus,
        published_mu_hosts=published_mu_hosts,
        host_to_pantheon_indices=host_to_pantheon,
        tip_source=tip_source,
    )


def load_all(
    loader: Optional[DataLoader] = None,
    *,
    strict: bool = True,
    tip_source_a: str = "freedman_2019",
    tip_source_b: str = "freedman_2025",
) -> Dict[str, TRGBDataBundle]:
    """Load both cases. With strict=False, returns bundles even if host
    photometry is missing (useful during development + dry runs).

    Per-case tip sources default to the authoritative published paper for
    that case (Freedman 2019 Table 3 for Case A, Freedman 2025 Table 2
    for Case B).
    """
    if loader is None:
        loader = DataLoader()
    return {
        "case_a": load_case_a(loader, strict=strict, tip_source=tip_source_a),
        "case_b": load_case_b(loader, strict=strict, tip_source=tip_source_b),
    }


__all__ = [
    "TRGBDataBundle",
    "TRGBPhotometryField",
    "load_all",
    "load_case_a",
    "load_case_b",
]
