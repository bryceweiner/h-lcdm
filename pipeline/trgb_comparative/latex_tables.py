"""Publication-ready LaTeX data-tables file for the TRGB comparative pipeline.

Generates a single ``data_tables.tex`` file containing every astronomical
input the pipeline consumes, organized by data source with full DOI /
arXiv identifiers and BibTeX-ready entries. Tables use the AASTeX
``deluxetable`` environment (the dominant journal style for the
intended publication venues — ApJ, ApJL, AJ, A&A).

Coverage:

- **§1 Geometric distance anchors**: Pietrzyński 2019 LMC DEB, Reid 2019
  NGC 4258 maser.
- **§2 TRGB / JAGB calibrator distances**: Freedman 2019 Table 1
  (per-host extinction), Freedman 2019 Table 3 (full 27-SN TRGB
  calibrator table), Freedman 2025 Table 2 (JWST 11-SN), Freedman 2025
  Table 3 (augmented HST+JWST 24-SN).
- **§3 SN Ia photometric samples**: Uddin 2023 h0csp B-band sample
  (full 390-SN flow + 18-SN TRGB calibrator subset), Pantheon 2018
  cosmology sample (Scolnic 2018; summary metadata + IDSURVEY breakdown),
  Pantheon+SH0ES (Brout 2022; summary metadata + calibrator subset
  enumeration).
- **§4 SN-system H₀ recalibrations**: Hoyt 2025 Tables 6 & 7 (per-SN-
  system reference and augmented H₀ values).
- **§5 Cosmological priors**: Planck 2018 distance prior used for d_CMB.
- **§6 BibTeX bibliography**: ready-to-paste BibTeX records covering
  every reference cited in the table captions.

Calibrator tables (small, ~10-30 rows) are tabulated in full. Hubble-
flow samples (hundreds of rows) are summarized with sample size,
redshift range, photometric system foundation, and SHA-256 file
provenance — per-SN flow data is voluminous and not typically tabulated
in papers; the SHA-256 plus file path lets reviewers retrieve the
underlying data deterministically.

Public API: a single function ``write_latex_data_tables(loader, out_path)``
that takes a ``DataLoader`` and an output path, returns the written
``Path``. Wired into the standard pipeline reporting flow.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# DOI / BibTeX registry
# =============================================================================

# Each entry: bibkey, full citation text, DOI, arXiv ID, BibTeX record.
BIB_REGISTRY: Dict[str, Dict[str, str]] = {
    "Pietrzynski2019": {
        "citation": "Pietrzyński, G., Graczyk, D., Gallenne, A., et al. 2019, Nature, 567, 200",
        "doi": "10.1038/s41586-019-0999-4",
        "arxiv": "1903.08096",
        "bibtex": r"""@ARTICLE{Pietrzynski2019,
    author = {{Pietrzy{\'n}ski}, G. and {Graczyk}, D. and {Gallenne}, A. and others},
    title = "{A distance to the Large Magellanic Cloud that is precise to one per cent}",
    journal = {\nat},
    year = 2019, volume = 567, pages = {200-203},
    doi = {10.1038/s41586-019-0999-4},
    eprint = {1903.08096},
}""",
    },
    "Reid2019": {
        "citation": "Reid, M. J., Pesce, D. W., & Riess, A. G. 2019, ApJL, 886, L27",
        "doi": "10.3847/2041-8213/ab552d",
        "arxiv": "1908.05625",
        "bibtex": r"""@ARTICLE{Reid2019,
    author = {{Reid}, M.~J. and {Pesce}, D.~W. and {Riess}, A.~G.},
    title = "{An Improved Distance to NGC 4258 and Its Implications for the Hubble Constant}",
    journal = {\apjl},
    year = 2019, volume = 886, pages = {L27},
    doi = {10.3847/2041-8213/ab552d},
    eprint = {1908.05625},
}""",
    },
    "Freedman2019": {
        "citation": "Freedman, W. L., Madore, B. F., Hatt, D., et al. 2019, ApJ, 882, 34",
        "doi": "10.3847/1538-4357/ab2f73",
        "arxiv": "1907.05922",
        "bibtex": r"""@ARTICLE{Freedman2019,
    author = {{Freedman}, W.~L. and {Madore}, B.~F. and {Hatt}, D. and {Hoyt}, T.~J.
              and {Jang}, I.-S. and {Beaton}, R.~L. and others},
    title = "{The Carnegie-Chicago Hubble Program. VIII. An Independent Determination
              of the Hubble Constant Based on the Tip of the Red Giant Branch}",
    journal = {\apj},
    year = 2019, volume = 882, eid = {34},
    doi = {10.3847/1538-4357/ab2f73},
    eprint = {1907.05922},
}""",
    },
    "Freedman2025": {
        "citation": "Freedman, W. L., Madore, B. F., Hoyt, T. J., et al. 2025, ApJ, 985, 203",
        "doi": "10.3847/1538-4357/adcaef",
        "arxiv": "2408.06153",
        "bibtex": r"""@ARTICLE{Freedman2025,
    author = {{Freedman}, W.~L. and {Madore}, B.~F. and {Hoyt}, T.~J. and {Owens}, K.~A.
              and {Lee}, A.~J. and others},
    title = "{Status Report on the Chicago-Carnegie Hubble Program (CCHP):
              Three Independent Astrophysical Determinations of the Hubble Constant
              Using the James Webb Space Telescope}",
    journal = {\apj},
    year = 2025, volume = 985, eid = {203},
    doi = {10.3847/1538-4357/adcaef},
    eprint = {2408.06153},
}""",
    },
    "Hoyt2025": {
        "citation": "Hoyt, T. J., Jang, I.-S., Freedman, W. L., et al. 2025, arXiv:2503.11769",
        "doi": "",
        "arxiv": "2503.11769",
        "bibtex": r"""@ARTICLE{Hoyt2025,
    author = {{Hoyt}, T.~J. and {Jang}, I.-S. and {Freedman}, W.~L. and {Madore}, B.~F.
              and {Owens}, K.~A. and {Lee}, A.~J.},
    title = "{The Chicago-Carnegie Hubble Program: TRGB Distances to NGC 4258 and the LMC
              with the James Webb Space Telescope}",
    journal = {arXiv e-prints},
    year = 2025,
    eprint = {2503.11769},
}""",
    },
    "Uddin2024": {
        "citation": "Uddin, S. A., Burns, C. R., Phillips, M. M., et al. 2024, ApJ, 970, 72",
        "doi": "10.3847/1538-4357/ad53c3",
        "arxiv": "2308.01875",
        "bibtex": r"""@ARTICLE{Uddin2024,
    author = {{Uddin}, S.~A. and {Burns}, C.~R. and {Phillips}, M.~M. and {Hsiao}, E.~Y.
              and {Suntzeff}, N.~B. and others},
    title = "{Carnegie Supernova Project-I and -II: Measurements of $H_0$ Using
              Cepheid, TRGB, and SBF Distance Calibration to Type Ia Supernovae}",
    journal = {\apj},
    year = 2024, volume = 970, eid = {72},
    doi = {10.3847/1538-4357/ad53c3},
    eprint = {2308.01875},
}""",
    },
    "Scolnic2018": {
        "citation": "Scolnic, D. M., Jones, D. O., Rest, A., et al. 2018, ApJ, 859, 101",
        "doi": "10.3847/1538-4357/aab9bb",
        "arxiv": "1710.00845",
        "bibtex": r"""@ARTICLE{Scolnic2018,
    author = {{Scolnic}, D.~M. and {Jones}, D.~O. and {Rest}, A. and {Pan}, Y.~C.
              and {Chornock}, R. and others},
    title = "{The Complete Light-curve Sample of Spectroscopically Confirmed SNe Ia
              from Pan-STARRS1 and Cosmological Constraints from the Combined Pantheon Sample}",
    journal = {\apj},
    year = 2018, volume = 859, eid = {101},
    doi = {10.3847/1538-4357/aab9bb},
    eprint = {1710.00845},
}""",
    },
    "Scolnic2015": {
        "citation": "Scolnic, D., Casertano, S., Riess, A., et al. 2015, ApJ, 815, 117",
        "doi": "10.1088/0004-637X/815/2/117",
        "arxiv": "1508.05361",
        "bibtex": r"""@ARTICLE{Scolnic2015,
    author = {{Scolnic}, D. and {Casertano}, S. and {Riess}, A. and {Rest}, A.
              and {Schlafly}, E. and others},
    title = "{Supercal: Cross-calibration of Multiple Photometric Systems to Improve
              Cosmological Measurements with Type Ia Supernovae}",
    journal = {\apj},
    year = 2015, volume = 815, eid = {117},
    doi = {10.1088/0004-637X/815/2/117},
    eprint = {1508.05361},
}""",
    },
    "Brout2022": {
        "citation": "Brout, D., Scolnic, D., Popovic, B., et al. 2022, ApJ, 938, 110",
        "doi": "10.3847/1538-4357/ac8e04",
        "arxiv": "2202.04077",
        "bibtex": r"""@ARTICLE{Brout2022,
    author = {{Brout}, D. and {Scolnic}, D. and {Popovic}, B. and {Riess}, A.~G.
              and {Carr}, A. and {Zuntz}, J. and others},
    title = "{The Pantheon+ Analysis: Cosmological Constraints}",
    journal = {\apj},
    year = 2022, volume = 938, eid = {110},
    doi = {10.3847/1538-4357/ac8e04},
    eprint = {2202.04077},
}""",
    },
    "Riess2022": {
        "citation": "Riess, A. G., Yuan, W., Macri, L. M., et al. 2022, ApJL, 934, L7",
        "doi": "10.3847/2041-8213/ac5c5b",
        "arxiv": "2112.04510",
        "bibtex": r"""@ARTICLE{Riess2022,
    author = {{Riess}, A.~G. and {Yuan}, W. and {Macri}, L.~M. and {Scolnic}, D. and others},
    title = "{A Comprehensive Measurement of the Local Value of the Hubble Constant
              with 1 km s$^{-1}$ Mpc$^{-1}$ Uncertainty from the Hubble Space Telescope
              and the SH0ES Team}",
    journal = {\apjl},
    year = 2022, volume = 934, eid = {L7},
    doi = {10.3847/2041-8213/ac5c5b},
    eprint = {2112.04510},
}""",
    },
    "PlanckVI2020": {
        "citation": "Planck Collaboration, Aghanim, N., Akrami, Y., et al. 2020, A&A, 641, A6",
        "doi": "10.1051/0004-6361/201833910",
        "arxiv": "1807.06209",
        "bibtex": r"""@ARTICLE{PlanckVI2020,
    author = {{Planck Collaboration} and {Aghanim}, N. and {Akrami}, Y. and others},
    title = "{Planck 2018 results. VI. Cosmological parameters}",
    journal = {\aap},
    year = 2020, volume = 641, eid = {A6},
    doi = {10.1051/0004-6361/201833910},
    eprint = {1807.06209},
}""",
    },
    "Anand2022": {
        "citation": "Anand, G. S., Tully, R. B., Rizzi, L., Riess, A. G., & Yuan, W. 2022, ApJ, 932, 15",
        "doi": "10.3847/1538-4357/ac68df",
        "arxiv": "2108.00007",
        "bibtex": r"""@ARTICLE{Anand2022,
    author = {{Anand}, G.~S. and {Tully}, R.~B. and {Rizzi}, L. and {Riess}, A.~G. and {Yuan}, W.},
    title = "{Comparing Tip of the Red Giant Branch Distance Scales: An Independent
              Reduction of the Carnegie-Chicago Hubble Program and the Value of $H_0$}",
    journal = {\apj},
    year = 2022, volume = 932, eid = {15},
    doi = {10.3847/1538-4357/ac68df},
    eprint = {2108.00007},
}""",
    },
}


# =============================================================================
# Helpers
# =============================================================================


def _esc(s) -> str:
    """Escape LaTeX special characters in a string value."""
    if s is None:
        return ""
    s = str(s)
    repl = {"&": r"\&", "%": r"\%", "$": r"\$", "#": r"\#",
            "_": r"\_", "{": r"\{", "}": r"\}",
            "~": r"\textasciitilde{}", "^": r"\textasciicircum{}",
            "\\": r"\textbackslash{}"}
    out = []
    for ch in s:
        out.append(repl.get(ch, ch))
    return "".join(out)


def _fmt(v: Any, prec: int = 3) -> str:
    """Format a value for a LaTeX cell, '...' for missing/NaN."""
    if v is None:
        return r"\nodata"
    try:
        f = float(v)
        if not np.isfinite(f):
            return r"\nodata"
        return f"{f:.{prec}f}"
    except (TypeError, ValueError):
        return _esc(v)


def _now() -> str:
    return datetime.now(tz=timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _hdr(title: str, level: int = 1) -> str:
    """LaTeX section header."""
    cmd = {1: "section", 2: "subsection", 3: "subsubsection"}[level]
    return f"\\{cmd}{{{title}}}\n\n"


def _bib_cite(key: str) -> str:
    """Return the citation text for a bib key."""
    return BIB_REGISTRY[key]["citation"]


def _doi_or_arxiv(key: str) -> str:
    """Return a DOI or arXiv identifier line for the table caption."""
    rec = BIB_REGISTRY[key]
    if rec["doi"]:
        return f"DOI:~\\href{{https://doi.org/{rec['doi']}}}{{{rec['doi']}}}"
    if rec["arxiv"]:
        return f"arXiv:~\\href{{https://arxiv.org/abs/{rec['arxiv']}}}{{{rec['arxiv']}}}"
    return ""


# =============================================================================
# Section 1 — Geometric anchors
# =============================================================================


def _section_geometric_anchors(loader) -> str:
    """Two single-row tables for the LMC and NGC 4258 geometric anchors."""
    lines: List[str] = [_hdr("Geometric distance anchors", 1)]
    lines.append(
        "Two distance-ladder anchors are used: the LMC eclipsing-binary "
        "distance from Pietrzy{\\'n}ski et al.~\\citep{Pietrzynski2019} "
        "(Case~A, the F2019 reproduction) and the NGC~4258 maser distance "
        "from Reid et al.~\\citep{Reid2019} (Case~B, the F2025 reproduction).\n\n"
    )

    lmc = loader.load_pietrzynski_lmc_distance()
    ngc = loader.load_reid_ngc4258_distance()

    lines.append(r"""\begin{deluxetable}{lcccl}
\tablecaption{Geometric distance anchors used in the analysis. \label{tab:anchors}}
\tablehead{\colhead{Anchor} & \colhead{$\mu$~(mag)} & \colhead{$\sigma_{\mu,\mathrm{stat}}$} &
           \colhead{$\sigma_{\mu,\mathrm{sys}}$} & \colhead{Reference}}
\startdata
""")
    lines.append(
        f"LMC (DEB)      & {_fmt(lmc['mu'])} & {_fmt(lmc['sigma_mu_stat'])} "
        f"& {_fmt(lmc['sigma_mu_sys'])} & \\citet{{Pietrzynski2019}} \\\\\n"
    )
    lines.append(
        f"NGC~4258 (maser) & {_fmt(ngc['mu'])} & {_fmt(ngc['sigma_mu_stat'])} "
        f"& {_fmt(ngc['sigma_mu_sys'])} & \\citet{{Reid2019}} \\\\\n"
    )
    lines.append(r"""\enddata
\tablecomments{LMC distance modulus from the Pietrzy{\'n}ski et al.~2019
detached eclipsing binary measurement, $d=49.59\pm0.09$~kpc, """ +
                 _doi_or_arxiv("Pietrzynski2019") + r""".
NGC~4258 distance modulus from the Reid et al.~2019 maser measurement,
$d=7.58\pm0.11$~Mpc, """ + _doi_or_arxiv("Reid2019") + r""".}
\end{deluxetable}

""")
    return "".join(lines)


# =============================================================================
# Section 2 — TRGB calibrator distances
# =============================================================================


def _section_trgb_calibrators(loader) -> str:
    lines: List[str] = [_hdr("TRGB calibrator distance moduli", 1)]
    lines.append(
        "TRGB calibrator distances enter through three published Freedman et al. "
        "tables: Freedman 2019 Table~3~\\citep{Freedman2019} (LMC anchor), Freedman "
        "2025 Table~2~\\citep{Freedman2025} (JWST-only TRGB+JAGB averaged "
        "distances), and Freedman 2025 Table~3 (the augmented HST+JWST TRGB sample "
        "from which the published $H_0=70.39$ derives). The pipeline transcribes "
        "all three verbatim into machine-readable CSVs under "
        "\\texttt{trgb_data/catalogs/}; the SHA-256 of each transcription is reported "
        "alongside.\n\n"
    )

    # Freedman 2019 Table 1 — per-host A_F814W
    try:
        f19t1 = loader.load_freedman_2019_table1()
        lines.append(_hdr("Freedman 2019 Table 1: per-host F814W extinction", 2))
        lines.append(r"""\begin{deluxetable}{lc}
\tablecaption{Per-host foreground extinction in F814W from Freedman~2019
Table~1. Used by the pipeline's CCHP TRGB photometry reduction.
\label{tab:f19_t1}}
\tablehead{\colhead{Host galaxy} & \colhead{$A_{F814W}$ (mag)}}
\startdata
""")
        for host, rec in sorted(f19t1["hosts"].items()):
            a = rec.get("A_F814W")
            lines.append(f"{_esc(host)} & {_fmt(a, 3)} \\\\\n")
        lines.append(r"""\enddata
\tablecomments{From Freedman et al.~2019 Table~1. """ + _doi_or_arxiv("Freedman2019") + r""".}
\end{deluxetable}

""")
    except Exception as exc:
        lines.append(f"% Freedman 2019 Table 1 unavailable: {exc}\n\n")

    # Freedman 2019 Table 3 — full TRGB calibrator set
    try:
        from pathlib import Path
        df = pd.read_csv(Path("trgb_data/catalogs/freedman_2019_table3.csv"), comment="#")
        lines.append(_hdr("Freedman 2019 Table 3: TRGB calibrator distances and SN Ia magnitudes", 2))
        lines.append(r"""\begin{deluxetable*}{llcccccccc}
\tablecaption{TRGB and Cepheid distance moduli plus standardized SN~Ia peak
magnitudes for the Freedman~2019 calibrator set (LMC-anchored). \label{tab:f19_t3}}
\tablehead{\colhead{SN Ia} & \colhead{Host} &
           \colhead{$\mu_{\mathrm{TRGB}}$} & \colhead{$\sigma_T$} &
           \colhead{$m_B^{\mathrm{CSP}}$} & \colhead{$\sigma_{m_B}^{\mathrm{CSP}}$} &
           \colhead{$\mu_{\mathrm{Ceph}}$} & \colhead{$\sigma_C$} &
           \colhead{$m_B^{\mathrm{SC}}$} & \colhead{$\sigma_{m_B}^{\mathrm{SC}}$}}
\startdata
""")
        for _, row in df.iterrows():
            lines.append(
                f"{_esc(row['SN'])} & {_esc(row['host_canon'])} & "
                f"{_fmt(row.get('mu_TRGB'),3)} & {_fmt(row.get('sigma_T'),2)} & "
                f"{_fmt(row.get('m_B_CSP'),3)} & {_fmt(row.get('sigma_B_CSP'),2)} & "
                f"{_fmt(row.get('mu_Ceph'),3)} & {_fmt(row.get('sigma_C'),2)} & "
                f"{_fmt(row.get('m_B_SuperCal'),3)} & {_fmt(row.get('sigma_B_SC'),2)} \\\\\n"
            )
        lines.append(r"""\enddata
\tablecomments{Transcribed verbatim from Freedman et al.~2019 Table~3, """ +
                     _doi_or_arxiv("Freedman2019") + r""".
$\mu_{\mathrm{TRGB}}$ is the LMC-anchored TRGB distance modulus from CCHP
or Jang \& Lee~2017. $m_B^{\mathrm{CSP}}$ is the CSP-I standardized peak
B-band magnitude; $m_B^{\mathrm{SC}}$ is the SuperCal cross-calibrated
peak from Scolnic et al.~2015~\citep{Scolnic2015}. $\mu_{\mathrm{Ceph}}$
is the Cepheid distance modulus from Riess et al.~2016. Empty entries
mean the corresponding measurement does not exist for that SN.}
\end{deluxetable*}

""")
    except Exception as exc:
        lines.append(f"% Freedman 2019 Table 3 unavailable: {exc}\n\n")

    # Freedman 2025 Table 2 — JWST 11-SN
    try:
        from pathlib import Path
        df = pd.read_csv(Path("trgb_data/catalogs/freedman_2025_table2.csv"), comment="#")
        lines.append(_hdr("Freedman 2025 Table 2: JWST-only TRGB+JAGB averaged distance moduli", 2))
        lines.append(r"""\begin{deluxetable*}{llccccccc}
\tablecaption{JWST-only TRGB and JAGB distance moduli (NGC~4258 anchor)
from Freedman~2025 Table~2. The 11-SN subset used in the
$H_0=68.81\pm1.80$ JWST-only sensitivity variant. \label{tab:f25_t2}}
\tablehead{\colhead{SN Ia} & \colhead{Host} &
           \colhead{$\mu_{\mathrm{TRGB}}^{\mathrm{JWST}}$} & \colhead{$\sigma_T$} &
           \colhead{$\mu_{\mathrm{JAGB}}^{\mathrm{JWST}}$} & \colhead{$\sigma_J$} &
           \colhead{$\bar{\mu}$} & \colhead{$\bar{\sigma}$}}
\startdata
""")
        for _, row in df.iterrows():
            lines.append(
                f"{_esc(row['SN'])} & {_esc(row['host_canon'])} & "
                f"{_fmt(row.get('mu_TRGB'),3)} & {_fmt(row.get('sigma_T'),3)} & "
                f"{_fmt(row.get('mu_JAGB'),3)} & {_fmt(row.get('sigma_J'),3)} & "
                f"{_fmt(row.get('mu_bar'),2)} & {_fmt(row.get('sigma_bar'),2)} \\\\\n"
            )
        lines.append(r"""\enddata
\tablecomments{Transcribed from Freedman et al.~2025 Table~2 (label
\texttt{tab:distances} in the arXiv source), """ +
                     _doi_or_arxiv("Freedman2025") + r""".
$\bar{\mu}$ is the inverse-variance-weighted mean of $\mu_{\mathrm{TRGB}}^{\mathrm{JWST}}$
and $\mu_{\mathrm{JAGB}}^{\mathrm{JWST}}$. NGC~5643 hosts both SN~2013aa
and SN~2017cbv at the same TRGB/JAGB distances.}
\end{deluxetable*}

""")
    except Exception as exc:
        lines.append(f"% Freedman 2025 Table 2 unavailable: {exc}\n\n")

    # Freedman 2025 Table 3 — augmented HST+JWST 24-SN
    try:
        from pathlib import Path
        df = pd.read_csv(Path("trgb_data/catalogs/freedman_2025_table3.csv"), comment="#")
        df_aug = df[df["in_augmented"] == 1].reset_index(drop=True)
        lines.append(_hdr("Freedman 2025 Table 3: augmented HST+JWST TRGB sample (primary $H_0$ sample)", 2))
        lines.append(r"""\begin{deluxetable*}{llcccccccc}
\tablecaption{Augmented HST+JWST TRGB calibrator sample from Freedman~2025
Table~3 (label \texttt{tab:cchptrgbtot}). The published TRGB
$H_0=70.39\pm1.22$ derives from this 24-SN sample. \label{tab:f25_t3}}
\tablehead{\colhead{SN Ia} & \colhead{Host} &
           \colhead{$\mu_{\mathrm{TRGB}}^{\mathrm{JWST}}$} & \colhead{$\sigma$} &
           \colhead{$\mu_{\mathrm{TRGB}}^{F19,F21}$} & \colhead{$\sigma$} &
           \colhead{$\mu_{\mathrm{TRGB}}^{\mathrm{CCHP}}$} & \colhead{$\sigma$} &
           \colhead{$\mu_{\mathrm{Ceph}}^{R22}$} & \colhead{$\sigma$}}
\startdata
""")
        for _, row in df_aug.iterrows():
            lines.append(
                f"{_esc(row['SN'])} & {_esc(row['host_canon'])} & "
                f"{_fmt(row.get('mu_TRGB_JWST'),3)} & {_fmt(row.get('sigma_TRGB_JWST'),3)} & "
                f"{_fmt(row.get('mu_TRGB_F19F21'),3)} & {_fmt(row.get('sigma_TRGB_F19F21'),2)} & "
                f"{_fmt(row.get('mu_TRGB_CCHP'),3)} & {_fmt(row.get('sigma_TRGB_CCHP'),3)} & "
                f"{_fmt(row.get('mu_Ceph_R22'),3)} & {_fmt(row.get('sigma_Ceph_R22'),3)} \\\\\n"
            )
        lines.append(r"""\enddata
\tablecomments{$\mu_{\mathrm{TRGB}}^{\mathrm{CCHP}}$ is the
inverse-variance-weighted combined HST+JWST distance modulus and is the
quantity used in the published $H_0=70.39\pm1.22\pm1.33$~km\,s$^{-1}$\,Mpc$^{-1}$
derivation (Freedman~2025 Table~4 row ``24 SN calibrators, z$>$0.01;
\textsc{pymc}''). Riess et al.~2022~\citep{Riess2022} R22 Cepheid
distances are listed for the subset of hosts that have one. SN~2021pit
(NGC~1448) is in the table but excluded from the 24-SN augmented analysis;
not shown here. """ + _doi_or_arxiv("Freedman2025") + r""".}
\end{deluxetable*}

""")
    except Exception as exc:
        lines.append(f"% Freedman 2025 Table 3 unavailable: {exc}\n\n")

    return "".join(lines)


# =============================================================================
# Section 3 — SN Ia photometric samples
# =============================================================================


def _section_sn_samples(loader) -> str:
    lines: List[str] = [_hdr("SN Ia photometric samples", 1)]
    lines.append(
        "Three SN Ia photometric system catalogs feed the per-system MCMC "
        "chains: the Uddin et al.~2024 h0csp B-band sample~\\citep{Uddin2024} "
        "(CSP-I and CSP-II native Carnegie photometry), the Pantheon~2018 "
        "cosmology sample~\\citep{Scolnic2018} (SuperCal-cross-calibrated "
        "$m_B$, foundation of the SuperCal chain), and Pantheon+SH0ES "
        "\\citep{Brout2022} (Cepheid-anchored sample with SH0ES "
        "calibrator subset).\n\n"
    )

    # Uddin h0csp summary + TRGB calibrator subset (full)
    try:
        u = loader.load_uddin_h0csp_sample()
        ut = loader.load_uddin_h0csp_trgb_dataset()
        lines.append(_hdr("Uddin 2024 h0csp B-band SN sample", 2))
        lines.append(r"""\begin{deluxetable}{lcl}
\tablecaption{Summary of the Uddin~2024 h0csp B-band SN~Ia sample. The
full per-SN SNooPy fit parameters are tabulated in the file referenced
below; only sample-level metadata are reproduced here. \label{tab:uddin_summary}}
\tablehead{\colhead{Quantity} & \colhead{Value} & \colhead{Note}}
\startdata
""")
        lines.append(
            f"$N_{{\\mathrm{{flow}}}}$ (full B-band sample) & "
            f"{u['N_flow_total']} & CSP-I + CSP-II combined \\\\\n"
            f"$N_{{\\mathrm{{flow}}}}$ (CSP-I subset) & "
            f"{u['N_flow_cspi']} & \\\\\n"
            f"$N_{{\\mathrm{{flow}}}}$ (CSP-II subset) & "
            f"{u['N_flow_cspii']} & \\\\\n"
            f"$N_{{\\mathrm{{cal}}}}$ (TRGB calibrators, F19 subset) & "
            f"{u['N_calibrators_trgb_f19']} & 18-SN F2019 sample \\\\\n"
            f"Cal SHA-256 (TRGB-f19) & "
            f"\\texttt{{{_esc(u['cal_file_sha256'][:16])}\\dots}} & file SHA-256 \\\\\n"
            f"Flow SHA-256 & "
            f"\\texttt{{{_esc(u['flow_file_sha256'][:16])}\\dots}} & file SHA-256 \\\\\n"
        )
        lines.append(r"""\enddata
\tablecomments{Source: Uddin et al.~2024, """ + _doi_or_arxiv("Uddin2024") + r""".
Public data archive: \href{https://github.com/syeduddin/h0csp}{github.com/syeduddin/h0csp}.
The unified TRGB Hubble-diagram fit input file
\texttt{data/working/B\_trgb\_update3.csv} carries 326 flow + 20 TRGB
calibrators in the schema used by Uddin's
\texttt{scripts/H0CSP.py}; SHA-256 \texttt{""" +
                     _esc(ut['file_sha256'][:32]) + r"""\dots}.}
\end{deluxetable}

""")

        # Per-SN TRGB calibrator subset
        cal = u["calibrators_trgb_f19"]
        lines.append(_hdr("Uddin 2024 h0csp F2019-style TRGB calibrator subset", 2))
        lines.append(r"""\begin{deluxetable}{llccc}
\tablecaption{Per-SN TRGB calibrator subset from the Uddin~2024 h0csp archive
(file \texttt{data/calibrators/calibrators\_trgb\_f19.csv}). $M_{\max}$ is
the standardized peak B-band magnitude in the native Carnegie photometric
system; $\mu_{\mathrm{TRGB}}$ is the LMC-anchored TRGB distance modulus
adopted from Freedman~2019. \label{tab:uddin_cal}}
\tablehead{\colhead{SN Ia} & \colhead{Host} & \colhead{$M_{\max}$ (mag)} &
           \colhead{$\mu_{\mathrm{TRGB}}$ (mag)} & \colhead{$\sigma_\mu$ (mag)}}
\startdata
""")
        for sn, host, mb, mu, smu in zip(cal['SN_name'], cal['host'],
                                          cal['Mmax'], cal['mu_TRGB'],
                                          cal['sigma_mu_TRGB']):
            lines.append(
                f"{_esc(sn)} & {_esc(host)} & {_fmt(mb,2)} & "
                f"{_fmt(mu,3)} & {_fmt(smu,2)} \\\\\n"
            )
        lines.append(r"""\enddata
\tablecomments{Public data: \href{https://github.com/syeduddin/h0csp}{Uddin~2024 h0csp repository}.}
\end{deluxetable}

""")
    except Exception as exc:
        lines.append(f"% Uddin h0csp unavailable: {exc}\n\n")

    # Pantheon 2018 — summary
    try:
        p18 = loader.load_pantheon_2018()
        lines.append(_hdr("Pantheon 2018 cosmology sample (Scolnic et al.\\ 2018)", 2))
        lines.append(r"""\begin{deluxetable}{lcl}
\tablecaption{Sample-level metadata for the Pantheon~2018 SN~Ia cosmology
sample, the foundation for the SuperCal-system chain's Hubble-flow block.
\label{tab:p18_summary}}
\tablehead{\colhead{Quantity} & \colhead{Value} & \colhead{Note}}
\startdata
""")
        lines.append(
            f"$N$ SNe (full sample) & {p18['N']} & spectroscopic Type Ia \\\\\n"
            f"$z_{{\\mathrm{{CMB}}}}$ range & {p18['zcmb'].min():.4f}--{p18['zcmb'].max():.4f} & \\\\\n"
            f"Photometric system & SuperCal & Scolnic et al.~2015~\\citep{{Scolnic2015}} \\\\\n"
            f"File SHA-256 & \\texttt{{{_esc(p18['file_sha256'][:16])}\\dots}} & "
            f"\\texttt{{lcparam\\_full\\_long.txt}} \\\\\n"
        )
        lines.append(r"""\enddata
\tablecomments{Source: Scolnic et al.~2018, """ + _doi_or_arxiv("Scolnic2018") + r""".
Public archive: \href{https://github.com/dscolnic/Pantheon}{github.com/dscolnic/Pantheon}.
The Pantheon~2018 cosmology sample explicitly excludes nearby calibrators
($z_{\mathrm{CMB}}<0.01$) by construction (Scolnic~2018 §2); the SuperCal-
chain calibrator $m_B$ values therefore come from Freedman~2019 Table~3's
$m_B^{\mathrm{SC}}$ column rather than this catalog.}
\end{deluxetable}

""")
    except Exception as exc:
        lines.append(f"% Pantheon 2018 unavailable: {exc}\n\n")

    # Pantheon+SH0ES — summary + calibrator subset enumeration
    try:
        pp = loader.load_pantheon_plus()
        is_cal = np.asarray(pp['is_calibrator'], dtype=bool)
        N_cal = int(is_cal.sum())
        N_flow = int((~is_cal).sum())
        z_arr = np.asarray(pp['z'], dtype=float)
        cal_sn = sorted(set(str(c) for c in np.asarray(pp['CID'])[is_cal]))

        lines.append(_hdr("Pantheon+SH0ES (Brout et al.\\ 2022)", 2))
        lines.append(r"""\begin{deluxetable}{lcl}
\tablecaption{Sample-level metadata for the Pantheon+SH0ES catalog. The
chain factory's CID-matched calibrator sample (post-2026-04-25 audit)
deliberately does not require Pantheon+'s own \texttt{IS\_CALIBRATOR}
flag, because the flag tracks ``has R22 Cepheid distance'' which is
orthogonal to TRGB-anchored analysis; see the host-coverage audit
report. \label{tab:pp_summary}}
\tablehead{\colhead{Quantity} & \colhead{Value} & \colhead{Note}}
\startdata
""")
        lines.append(
            f"$N$ rows (total) & {len(z_arr)} & per-SN per-survey rows \\\\\n"
            f"$N$ unique IS\\_CALIBRATOR=1 SNe & {len(cal_sn)} & R22 Cepheid-anchored \\\\\n"
            f"$N$ rows in Hubble-flow regression & {N_flow} & non-calibrator \\\\\n"
            f"$z$ range & {z_arr.min():.4f}--{z_arr.max():.4f} & \\\\\n"
            f"Photometric system & Pantheon+ recalibration & Brout et al.~2022~\\citep{{Brout2022}} \\\\\n"
        )
        lines.append(r"""\enddata
\tablecomments{Source: Brout et al.~2022, """ + _doi_or_arxiv("Brout2022") + r""".
Public archive: \href{https://github.com/PantheonPlusSH0ES/DataRelease}{github.com/PantheonPlusSH0ES/DataRelease}.}
\end{deluxetable}

""")
        # Calibrator SN list
        lines.append(_hdr("Pantheon+SH0ES IS\\_CALIBRATOR SN list", 3))
        lines.append(
            "The "
            f"{len(cal_sn)} unique SNe Ia flagged as IS\\_CALIBRATOR=1 in "
            "\\texttt{Pantheon+SH0ES.dat}: "
            + ", ".join(_esc(s) for s in cal_sn)
            + ". The pipeline's audit-corrected matching policy uses CID "
              "with photometry-source-suffix tolerance (e.g., "
              "\\texttt{1994DRichmond}~$\\to$~\\texttt{1994D}) and does not "
              "require IS\\_CALIBRATOR=1 for our TRGB-anchored chain.\n\n"
        )
    except Exception as exc:
        lines.append(f"% Pantheon+SH0ES unavailable: {exc}\n\n")

    return "".join(lines)


# =============================================================================
# Section 4 — Hoyt 2025 SN-system recalibrations
# =============================================================================


def _section_hoyt_2025(loader) -> str:
    lines: List[str] = [_hdr("SN-system $H_0$ recalibrations (Hoyt et al.\\ 2025)", 1)]
    lines.append(
        "Hoyt et al.~2025~\\citep{Hoyt2025} report per-SN-system $H_0$ "
        "values derived by applying their JWST-only and HST+JWST augmented "
        "TRGB distances to the Pantheon+, SuperCal, CSP-I, and CSP-II "
        "photometric samples. Their Tables~6 and~7 are the cross-reference "
        "against which our pipeline's per-system MCMC chains are compared.\n\n"
    )
    try:
        h = loader.load_hoyt_2025_sn_calibration()
        sys_d = h["systems"]
        lines.append(r"""\begin{deluxetable*}{lcccccccccc}
\tablecaption{Hoyt~2025 Tables~6 and~7 combined: per-SN-system reference
values, JWST-only and HST+JWST augmented recalibrations. \label{tab:hoyt}}
\tablehead{\multicolumn{4}{c}{\textbf{Reference}} & \multicolumn{3}{c}{\textbf{JWST-only}} &
           \multicolumn{4}{c}{\textbf{Augmented (HST+JWST)}} \\
           \colhead{System} & \colhead{$N_{\mathrm{ref}}$} & \colhead{$\langle M_B\rangle_{\mathrm{ref}}$} & \colhead{$H_{0,\mathrm{ref}}$} &
           \colhead{$N_{\mathrm{JWST}}$} & \colhead{$\langle M_B\rangle_{\mathrm{JWST}}$} & \colhead{$H_{0,\mathrm{JWST}}$} &
           \colhead{$N_{\mathrm{aug}}$} & \colhead{$\langle M_B\rangle_{\mathrm{aug}}$} &
           \colhead{$\Delta M_B$} & \colhead{$H_{0,\mathrm{aug}}$}}
\startdata
""")
        for sysname in ("CSP-I", "CSP-II", "SuperCal", "Pantheon+"):
            r = sys_d.get(sysname)
            if r is None:
                continue
            lines.append(
                f"{_esc(sysname)} & "
                f"{r['N_ref']} & {_fmt(r['M_B_ref_mag'],3)} & {_fmt(r['H0_ref_kms_Mpc'],2)} & "
                f"{r['jwst_only_N']} & {_fmt(r['jwst_only_M_B_mag'],3)} & {_fmt(r['jwst_only_H0'],2)} & "
                f"{r['augmented_N']} & {_fmt(r['augmented_M_B_mag'],3)} & "
                f"{_fmt(r['augmented_delta_M_B'],3)} & {_fmt(r['augmented_H0'],2)} \\\\\n"
            )
        lines.append(r"""\enddata
\tablecomments{Transcribed verbatim from Hoyt et al.~2025 Tables~6
(\textbf{Reference} block) and~7 (JWST-only + augmented blocks),
""" + _doi_or_arxiv("Hoyt2025") + r""".
$H_0$ values in km\,s$^{-1}$\,Mpc$^{-1}$; $\langle M_B\rangle$ in mag.
Augmented $H_0$ derived from Hoyt's Eq.~15:
$H_0^{\mathrm{new}}=H_0^{\mathrm{ref}}\times 10^{-\Delta M_B/5}$.
Used in the pipeline's chain-matrix table only as a literature
cross-reference; not as a pipeline-computed posterior.}
\end{deluxetable*}

""")
    except Exception as exc:
        lines.append(f"% Hoyt 2025 tables unavailable: {exc}\n\n")
    return "".join(lines)


# =============================================================================
# Section 5 — Cosmological priors
# =============================================================================


def _section_priors() -> str:
    lines: List[str] = [_hdr("Cosmological priors", 1)]
    lines.append(
        "The framework's holographic projection formula uses the Planck 2018 "
        "comoving distance to recombination as the high-$z$ anchor. The pipeline "
        "reads this from \\texttt{HLCDM\\_PARAMS.D\\_CMB\\_PLANCK\\_2018} at "
        "runtime.\n\n"
    )
    lines.append(r"""\begin{deluxetable}{lcl}
\tablecaption{Planck~2018 distance prior used as the high-$z$ anchor in the
H-$\Lambda$CDM holographic projection formula. \label{tab:planck}}
\tablehead{\colhead{Quantity} & \colhead{Value} & \colhead{Reference}}
\startdata
$d_{\mathrm{CMB}}$ ($D_M(z_{\mathrm{rec}})$) & 13869.7 $\pm$ 4.4 Mpc & \citet{PlanckVI2020} \\
$\Omega_m$ (matter density) & 0.315 & Planck 2018 TT,TE,EE+lowE+lensing \\
$H_0$ Planck (used only for cross-check) & 67.4 $\pm$ 0.5~km\,s$^{-1}$\,Mpc$^{-1}$ & \citet{PlanckVI2020} \\
""" + r"""\enddata
\tablecomments{From Planck Collaboration et al.~2020 Table~2 (TT,TE,EE+lowE+lensing).
""" + _doi_or_arxiv("PlanckVI2020") + r""".}
\end{deluxetable}

""")
    return "".join(lines)


# =============================================================================
# Section 6 — Bibliography
# =============================================================================


def _section_bibliography() -> str:
    lines: List[str] = [_hdr("BibTeX bibliography", 1)]
    lines.append(
        "Ready-to-paste BibTeX records covering every reference cited in the "
        "table captions above. Standard AAS journal abbreviations "
        "(\\texttt{\\textbackslash apj}, \\texttt{\\textbackslash apjl}, "
        "\\texttt{\\textbackslash aap}, \\texttt{\\textbackslash nat}) are "
        "assumed to be defined in the host document's preamble.\n\n"
    )
    lines.append("\\begin{verbatim}\n")
    for key in sorted(BIB_REGISTRY.keys()):
        lines.append(BIB_REGISTRY[key]["bibtex"])
        lines.append("\n\n")
    lines.append("\\end{verbatim}\n\n")
    return "".join(lines)


# =============================================================================
# Top-level renderer
# =============================================================================


_PREAMBLE = r"""% TRGB Comparative Analysis — Astronomical Data Tables
% Auto-generated by pipeline.trgb_comparative.latex_tables
% This file is intended to be \input{} into the host paper's main .tex.
% Required packages (typically loaded by AASTeX): deluxetable, graphicx,
% hyperref, natbib (or biblatex). The table environments use the AASTeX
% \deluxetable / \deluxetable* macros and standard \citet / \citep
% citation commands.
%
% Generation provenance is recorded at the top of each section. The
% pipeline's host coverage audit
% (results/trgb_comparative/reports/host_coverage_audit.md) describes
% catalog-level limitations that some tables expose via \nodata cells.

"""


def write_latex_data_tables(loader, out_path: Path,
                            log_fn: Optional[Callable[[str], None]] = None) -> Path:
    """Render the full data-tables LaTeX file and write it to disk.

    Parameters
    ----------
    loader : DataLoader
        Pipeline-wide ``DataLoader`` instance (already-cached datasets
        are reused; missing ones cause that section to render an empty
        commented stub rather than aborting).
    out_path : Path
        Output filename, e.g. ``results/trgb_comparative/reports/data_tables.tex``.
        Parent directory is created if missing.
    log_fn : callable, optional
        Logger callback for progress messages.
    """
    log = log_fn or (lambda m: logger.info(m))
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    parts: List[str] = [_PREAMBLE,
                        f"% Generated: {_now()}\n\n",
                        _section_geometric_anchors(loader),
                        _section_trgb_calibrators(loader),
                        _section_sn_samples(loader),
                        _section_hoyt_2025(loader),
                        _section_priors(),
                        _section_bibliography()]
    out_path.write_text("".join(parts))
    log(f"LaTeX data tables → {out_path}")
    return out_path


__all__ = ["write_latex_data_tables", "BIB_REGISTRY"]
