"""
Extinction and metallicity corrections, per Freedman paper.

Each of the two CCHP reproductions (Freedman 2019/2020 HST and Freedman
2024/2025 JWST) uses its own reddening law and metallicity treatment. The
corresponding ``apply_*_freedman_{2020,2024}`` functions implement the
published choices. Sensitivity variants (Green et al. 2019 3D dust,
Jang & Lee 2017 metallicity) are surfaced separately and are NEVER called
by the primary reproduction paths — they are only run in the sensitivity
analysis stage.

SFD foreground extinction uses the Schlegel, Finkbeiner & Davis 1998 map.
For field-level E(B-V), we rely on pre-extracted values passed in as part
of the photometry metadata rather than re-querying dust maps here (the
field coordinates are a constant subset known from the preregistration).

The Cardelli, Clayton & Mathis 1989 extinction law provides filter-
specific ratios A_λ / E(B-V).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


# ---------------------------------------------------------------------------
# CCM 1989 / Schlafly & Finkbeiner 2011 filter coefficients A_λ / E(B-V)
# R_V = 3.1 unless specified otherwise.
# ---------------------------------------------------------------------------

A_OVER_EBV_CCM89 = {
    # Optical / HST
    "F814W": 1.526,        # Schlafly & Finkbeiner 2011, Table 6
    "F555W": 2.755,
    "F606W": 2.488,
    "F160W": 0.512,        # HST WFC3 IR
    # JWST NIRCam (Freedman 2024 conversion; A_λ/A_V per CCM89 extrapolation)
    "F090W": 1.138,        # per Gordon et al. 2023 / CCM89 extension
    "F115W": 0.851,
    "F150W": 0.590,
    "F200W": 0.387,
}


@dataclass(frozen=True)
class ExtinctionCorrection:
    filter_name: str
    EBV: float
    A_filter: float             # A_λ in mag
    coefficient: float          # A_λ / E(B-V)
    prescription: str
    systematic_budget: float    # quadrature-summed σ(A_λ), mag


def _ebv_from_field(metadata: Dict[str, float], key: str = "EBV_SFD") -> float:
    """Retrieve field E(B-V) from metadata or raise if missing."""
    if key not in metadata:
        raise KeyError(
            f"Field metadata missing {key!r}. Preload SFD values in data_loaders."
        )
    return float(metadata[key])


# ---------------------------------------------------------------------------
# Extinction — Freedman 2019/2020 HST pipeline
# ---------------------------------------------------------------------------


def apply_extinction_freedman_2020(
    mag: np.ndarray,
    metadata: Dict[str, float],
    filter_name: str = "F814W",
) -> tuple:
    """Return (mag_dereddened, ExtinctionCorrection) under Freedman 2019/2020.

    Prefers ``A_F814W`` from Freedman 2019 Table 1 (published per-host
    foreground extinction). Falls back to computing A from metadata's
    ``EBV_SFD`` if the per-host value is not provided — that code path
    is only reachable for hosts absent from Freedman's Table 1, and
    should be treated as a sensitivity-variant placeholder.

    Uses SFD E(B-V) with Cardelli 1989 R_V = 3.1 extinction law when
    falling back. Filter coefficients per Schlafly & Finkbeiner 2011
    Table 6.
    """
    coeff = A_OVER_EBV_CCM89.get(filter_name)
    if coeff is None:
        raise ValueError(f"Unknown filter for Freedman 2020 extinction: {filter_name!r}")

    if filter_name == "F814W" and "A_F814W" in metadata:
        # Authoritative Freedman 2019 Table 1 value.
        A = float(metadata["A_F814W"])
        ebv = A / coeff                          # back-derive E(B-V) for provenance
        prescription = "freedman_2019_table1_A_F814W"
        sigma_total = max(0.14 * A, 0.01)        # conservative 14 % stat on A + 0.01 zero-point
    else:
        ebv = _ebv_from_field(metadata, "EBV_SFD")
        A = coeff * ebv
        sigma_A_sfd = 0.14 * A
        sigma_A_zero = 0.01
        sigma_total = float(np.sqrt(sigma_A_sfd ** 2 + sigma_A_zero ** 2))
        prescription = "freedman_2020_sfd_ccm89_placeholder_ebv"

    return mag - A, ExtinctionCorrection(
        filter_name=filter_name,
        EBV=ebv,
        A_filter=A,
        coefficient=coeff,
        prescription=prescription,
        systematic_budget=sigma_total,
    )


# ---------------------------------------------------------------------------
# Extinction — Freedman 2024/2025 JWST pipeline
# ---------------------------------------------------------------------------


def apply_extinction_freedman_2024(
    mag: np.ndarray,
    metadata: Dict[str, float],
    filter_name: str = "F150W",
) -> tuple:
    """Return (mag_dereddened, ExtinctionCorrection) under CCHP 2024 JWST."""
    ebv = _ebv_from_field(metadata, "EBV_SFD")
    coeff = A_OVER_EBV_CCM89.get(filter_name)
    if coeff is None:
        raise ValueError(f"Unknown filter for Freedman 2024 extinction: {filter_name!r}")
    A = coeff * ebv
    # JWST NIRCam: A_λ is small, but the extrapolation uncertainty is larger
    # than in the optical (CCM89 → JWST NIR extension).
    sigma_A_sfd = 0.14 * A
    sigma_A_extrap = 0.03 * A  # NIR extinction-law extrapolation
    sigma_A_zero = 0.02
    sigma_total = float(np.sqrt(sigma_A_sfd ** 2 + sigma_A_extrap ** 2 + sigma_A_zero ** 2))
    return mag - A, ExtinctionCorrection(
        filter_name=filter_name,
        EBV=ebv,
        A_filter=A,
        coefficient=coeff,
        prescription="freedman_2024_sfd_ccm89_nir",
        systematic_budget=sigma_total,
    )


# ---------------------------------------------------------------------------
# Extinction sensitivity variants
# ---------------------------------------------------------------------------


def apply_extinction_sensitivity_variant(
    mag: np.ndarray,
    metadata: Dict[str, float],
    filter_name: str,
    variant: str,
) -> tuple:
    """Sensitivity-only extinction prescription. NOT used by primary paths."""
    if variant == "green2019_3d":
        key = "EBV_GREEN2019"
    elif variant == "planck2014":
        key = "EBV_PLANCK2014"
    else:
        raise ValueError(f"Unknown extinction variant: {variant!r}")

    ebv = _ebv_from_field(metadata, key)
    coeff = A_OVER_EBV_CCM89.get(filter_name)
    if coeff is None:
        raise ValueError(f"Unknown filter for variant extinction: {filter_name!r}")
    A = coeff * ebv
    sigma_A = 0.14 * A
    return mag - A, ExtinctionCorrection(
        filter_name=filter_name,
        EBV=ebv,
        A_filter=A,
        coefficient=coeff,
        prescription=f"sensitivity_{variant}_ccm89",
        systematic_budget=sigma_A,
    )


# ---------------------------------------------------------------------------
# Metallicity — Freedman 2019/2020
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MetallicityCorrection:
    slope_beta: float           # dI_TRGB / d(color)
    pivot_color: float
    prescription: str


def apply_metallicity_freedman_2020(
    mu_TRGB: float,
    colors: np.ndarray,
    beta: float = 0.2,
    pivot_color: float = 1.23,
) -> tuple:
    """Freedman 2019/2020 color–magnitude metallicity correction.

    Their published slope in F814W is approximately dM / d(V-I) ≈ 0.2 mag
    per mag of color, with a pivot near V-I = 1.23 for metal-poor halo
    RGB. We apply this as a per-field median-color correction.
    """
    median_color = float(np.median(colors[np.isfinite(colors)]))
    delta = beta * (median_color - pivot_color)
    mu_corrected = float(mu_TRGB - delta)
    return mu_corrected, MetallicityCorrection(
        slope_beta=beta,
        pivot_color=pivot_color,
        prescription="freedman_2020_color_slope",
    )


# ---------------------------------------------------------------------------
# Metallicity — Freedman 2024/2025 JWST
# ---------------------------------------------------------------------------


def apply_metallicity_freedman_2024(
    mu_TRGB: float,
    colors: np.ndarray,
    beta: float = 0.08,
    pivot_color: float = 1.00,
) -> tuple:
    """CCHP 2024 JWST color-based metallicity correction.

    NIR TRGB color slope is shallower than the optical; CCHP 2024 reports
    dM/d(F090W-F150W) ≈ 0.08.
    """
    median_color = float(np.median(colors[np.isfinite(colors)]))
    delta = beta * (median_color - pivot_color)
    mu_corrected = float(mu_TRGB - delta)
    return mu_corrected, MetallicityCorrection(
        slope_beta=beta,
        pivot_color=pivot_color,
        prescription="freedman_2024_jwst_nir_slope",
    )


# ---------------------------------------------------------------------------
# Metallicity sensitivity variants
# ---------------------------------------------------------------------------


def apply_metallicity_sensitivity_variant(
    mu_TRGB: float,
    colors: np.ndarray,
    variant: str,
) -> tuple:
    """Alternative metallicity prescriptions for sensitivity analysis."""
    if variant == "rizzi2007":
        beta, pivot = 0.217, 1.23   # Rizzi, Tully et al. 2007
        prescription = "rizzi2007"
    elif variant == "jang_lee_2017":
        beta, pivot = 0.091, 1.23   # Jang & Lee 2017 quartic-flattened slope
        prescription = "jang_lee_2017"
    elif variant == "flat":
        beta, pivot = 0.0, 1.23
        prescription = "flat_no_correction"
    else:
        raise ValueError(f"Unknown metallicity variant: {variant!r}")

    median_color = float(np.median(colors[np.isfinite(colors)]))
    delta = beta * (median_color - pivot)
    return float(mu_TRGB - delta), MetallicityCorrection(
        slope_beta=beta, pivot_color=pivot, prescription=prescription
    )


__all__ = [
    "A_OVER_EBV_CCM89",
    "ExtinctionCorrection",
    "MetallicityCorrection",
    "apply_extinction_freedman_2020",
    "apply_extinction_freedman_2024",
    "apply_extinction_sensitivity_variant",
    "apply_metallicity_freedman_2020",
    "apply_metallicity_freedman_2024",
    "apply_metallicity_sensitivity_variant",
]
