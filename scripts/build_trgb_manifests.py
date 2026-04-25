#!/usr/bin/env python3
"""
Build TRGB photometry CSVs and manifest from cached EDD downloads.

Handles two DOLPHOT .phot.WEB formats found in the EDD archive:
  A) "mag1/mag2" generic columns (37 cols; `mag_mag1`, `mag_mag2`)
  B) "named filter" columns (43+ cols; `m_F555W`, `m_F814W`)

Writes:
  trgb_data/downloaded_data/trgb/sn_host_trgb_hst/{host}.csv  — F814W, F555W, errors, flag
  trgb_data/downloaded_data/trgb/sn_host_trgb_hst/manifest.csv  — host, published μ_TRGB, σ, reference

Run from repo root. Requires cached raw photometry under
  trgb_data/downloaded_data/trgb/sn_host_trgb_hst/raw/{host}.phot.WEB
and a parsed EDD summary at /tmp/edd_parsed.json (produced by the download step).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


RAW_DIR = Path("trgb_data/downloaded_data/trgb/sn_host_trgb_hst/raw")
OUT_DIR = Path("trgb_data/downloaded_data/trgb/sn_host_trgb_hst")

# Hosts that must NEVER appear in the Case A manifest: NGC 4258 is the
# Case B anchor galaxy (per Reid 2019 maser distance). Including it as a
# Case A host pollutes the LMC-anchored distance ladder. See diagnostic
# report § Bug 1.
CASE_A_EXCLUDED_HOSTS = {"NGC 4258"}


def _detect_format(first_line: str, second_line: str) -> str:
    """Return 'A' for mag1/mag2 style, 'B' for named-filter style."""
    if "GLOBAL SOLUTION" in first_line or first_line.strip().startswith("|"):
        return "B"
    if "m_F" in first_line:
        return "B"
    return "A"


def _parse_format_a(path: Path) -> pd.DataFrame:
    """Format A: 37 columns, whitespace separated, first line is header.

    Prefers ``mag_mag1`` / ``mag_mag2`` (Vega-calibrated) but falls back to
    ``inst_vega_mag*`` when those are populated with 99.999 placeholders.
    """
    with open(path) as f:
        header = f.readline().strip().split()
    df = pd.read_csv(path, sep=r"\s+", skiprows=1, names=header, low_memory=False)

    def best_mag(df, primary: str, secondary: str) -> pd.Series:
        col = df[primary].astype(float)
        if (col > 90).mean() > 0.5:
            # mostly placeholders → fall back
            return df[secondary].astype(float)
        return col

    def best_err(df, primary: str, fallback_value: float = 0.01) -> pd.Series:
        col = df[primary].astype(float)
        if col.abs().sum() < 1e-9:
            # all zeros → replace with a per-star floor
            return pd.Series([fallback_value] * len(col), index=col.index)
        return col

    out = pd.DataFrame(
        {
            "F814W": best_mag(df, "mag_mag2", "inst_vega_mag2"),
            "F814W_err": best_err(df, "mag2_err"),
            "F555W": best_mag(df, "mag_mag1", "inst_vega_mag1"),
            "F555W_err": best_err(df, "mag1_err"),
            "flag": df["flag_mag2"].astype(int),
        }
    )
    return out


def _parse_format_b(path: Path) -> pd.DataFrame:
    """Format B: global-solution header, then a column-name row, then data.

    Column names we need: `m_F814W`, `m_F555W`, `m_err`, plus `Eflag`. The
    per-filter block is 16 columns wide. In multi-exposure files the F814W
    header repeats; we take the FIRST F814W block (usually the combined).
    """
    with open(path) as f:
        lines = f.readlines()
    if len(lines) < 3:
        raise ValueError(f"{path} has fewer than 3 lines")
    # Column-name row is usually line 2 (index 1).
    header_line = lines[1].strip()
    column_names = header_line.split()

    # Data rows start from line 3 (index 2).
    data = []
    for line in lines[2:]:
        parts = line.split()
        if not parts:
            continue
        data.append(parts)

    # Data row width is >= len(column_names); extra columns are repeated
    # filter blocks. We keep only up to the first F814W block plus any
    # columns before it.
    # Find the positions of m_F555W, m_F814W, m_err, and Eflag in the header.
    def idx(name: str, start: int = 0) -> int:
        for i in range(start, len(column_names)):
            if column_names[i] == name:
                return i
        return -1

    # Some CCHP reductions use F606W instead of F555W as the blue filter.
    i_f555 = idx("m_F555W")
    blue_name = "F555W"
    if i_f555 < 0:
        i_f555 = idx("m_F606W")
        blue_name = "F606W"
    if i_f555 < 0:
        raise ValueError(f"{path} missing m_F555W and m_F606W in header")
    # m_err of the blue block is the next m_err after m_F555W/F606W (after m_V).
    i_f555_err = idx("m_err", i_f555)
    i_f814 = idx("m_F814W", i_f555_err)
    i_f814_err = idx("m_err", i_f814)
    # Eflag of F814W block is the first Eflag after m_F814W_err. Some
    # older WFPC2 reductions (e.g. LMC) omit Eflag entirely; fall back to
    # a constant "passes" flag in that case.
    i_eflag = idx("Eflag", i_f814_err)

    if min(i_f555, i_f555_err, i_f814, i_f814_err) < 0:
        raise ValueError(f"{path} could not locate filter columns in header")

    # Pull those specific indices out of each data row. Data rows may be
    # WIDER than the column_names list (multi-filter extras), but the
    # indices of the first F555W/F814W blocks stay the same.
    vals = []
    needed = max(i_f555, i_f555_err, i_f814, i_f814_err, i_eflag if i_eflag >= 0 else 0)
    for row in data:
        if len(row) <= needed:
            continue
        try:
            v555 = float(row[i_f555])
            v555e = float(row[i_f555_err])
            v814 = float(row[i_f814])
            v814e = float(row[i_f814_err])
            flag = int(row[i_eflag]) if i_eflag >= 0 else 0
        except ValueError:
            continue
        vals.append((v814, v814e, v555, v555e, flag))

    # Note: ``F555W`` column in the output may actually be F606W for hosts
    # that were imaged with the (V, I) = (F606W, F814W) pair. The color
    # meaning is preserved; downstream metallicity corrections are
    # color-based and accept either blue filter.
    df = pd.DataFrame(
        vals, columns=["F814W", "F814W_err", "F555W", "F555W_err", "flag"]
    )
    return df


def parse_one(path: Path) -> pd.DataFrame:
    with open(path) as f:
        l1 = f.readline()
        l2 = f.readline()
    fmt = _detect_format(l1, l2)
    if fmt == "A":
        return _parse_format_a(path)
    return _parse_format_b(path)


def _sigma_from_edd_record(rec: dict) -> Optional[float]:
    """Interpret eDM_lo / eDM_hi from the parsed EDD HTML record.

    In EDD's HTML table, some rows store the σ on distance modulus as a
    small number (0.05-0.15 mag); others accidentally store lo/hi bound
    absolute values. We guard by requiring 0 < σ < 1.
    """
    for key in ("eDM_lo", "eDM_hi"):
        try:
            v = float(rec.get(key))
        except (TypeError, ValueError):
            continue
        if 0.0 < v < 1.0:
            return v
        # If v > 1, it's probably the bound itself (DM_tip + σ); infer σ.
        try:
            dm = float(rec["DM_tip"])
            diff = abs(v - dm)
            if 0.0 < diff < 1.0:
                return diff
        except (KeyError, TypeError, ValueError):
            continue
    return None


def main() -> int:
    edd_json = Path("/tmp/edd_parsed.json")
    if not edd_json.exists():
        print(f"Missing {edd_json} — run the EDD download step first.")
        return 1

    with open(edd_json) as f:
        parsed = json.load(f)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    manifest_rows = []
    for host, rec in parsed.items():
        if host in CASE_A_EXCLUDED_HOSTS:
            print(f"  {host}: excluded from Case A manifest (anchor-galaxy contamination)")
            continue
        raw = RAW_DIR / f'{host.replace(" ", "_")}.phot.WEB'
        if not raw.exists() or raw.stat().st_size < 10000:
            print(f"  {host}: no raw photometry")
            continue
        try:
            df = parse_one(raw)
        except Exception as exc:
            print(f"  {host}: parse failed — {exc}")
            continue
        df = df[(df["F814W"] > 15) & (df["F814W"] < 30)]
        if df.empty:
            print(f"  {host}: empty after sanity cuts")
            continue

        csv_path = OUT_DIR / f"{host}.csv"
        df.to_csv(csv_path, index=False)

        try:
            mu = float(rec.get("DM_tip"))
        except (TypeError, ValueError):
            print(f"  {host}: no published DM_tip — skipping manifest entry")
            continue
        sigma = _sigma_from_edd_record(rec)
        if sigma is None:
            # Per EDD Anand 2021 paper, typical σ(μ) ≈ 0.1 mag for TRGB distances.
            sigma = 0.10

        manifest_rows.append(
            {
                "host": host,
                "published_mu_TRGB": mu,
                "published_sigma_mu": sigma,
                "reference": "Anand 2021 reduction via EDD kcmd",
            }
        )
        print(f"  {host}: {len(df):,} stars, μ_TRGB={mu:.3f} ± {sigma:.3f}")

    manifest = pd.DataFrame(manifest_rows)
    manifest.to_csv(OUT_DIR / "manifest.csv", index=False)
    print(f"\nmanifest: {len(manifest)} hosts -> {OUT_DIR / 'manifest.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
