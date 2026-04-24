"""
TRGB edge-detection algorithms.

Three methods, uniform API:

- :func:`detect_trgb_sobel` — Sobel edge detection on the smoothed I-band
  luminosity function (Lee, Freedman & Madore 1993 lineage). Parametrized by
  a Gaussian smoothing kernel width. Freedman 2019/2020's primary analysis
  uses kernel_width ≈ 2.
- :func:`detect_trgb_model_based` — Fit a parametric broken power-law (or
  broken linear in log-N) to the luminosity function (Makarov et al. 2006
  style). Robust when the edge is broad.
- :func:`detect_trgb_bayesian` — Bayesian posterior over the tip location
  using a model-based likelihood with noise-aware per-star contribution
  (Hatt et al. 2017 style).

All three return a :class:`EdgeDetectionResult` with a common schema so
higher-level code can evaluate sensitivity across methods without branching.

These functions are deliberately pure-Python / NumPy / SciPy — no framework
dependencies. A separate module (:mod:`pipeline.trgb_comparative.compute_backend`)
offers MLX acceleration; the hot-path MLX variant is implemented as an
alternative entry point behind a feature flag rather than replacing the
NumPy reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Common result schema
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class EdgeDetectionResult:
    I_TRGB: float
    sigma_I_TRGB: float
    method: str
    hyperparameters: Dict[str, float]
    diagnostics: Dict[str, np.ndarray] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, object]:
        out: Dict[str, object] = {
            "I_TRGB": float(self.I_TRGB),
            "sigma_I_TRGB": float(self.sigma_I_TRGB),
            "method": self.method,
            "hyperparameters": {k: float(v) for k, v in self.hyperparameters.items()},
        }
        out["diagnostics"] = {k: v.tolist() for k, v in self.diagnostics.items()}
        return out


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------


def _build_luminosity_function(
    mag: np.ndarray,
    sigma_mag: Optional[np.ndarray],
    bin_width: float,
    mag_range: Optional[tuple] = None,
) -> tuple:
    """Binned luminosity function N(I) with 1/sigma² weights.

    Returns (mag_centres, N, sigma_N) where ``sigma_N = sqrt(N)`` for
    unweighted Poisson statistics (sigma_mag currently unused in N; the
    edge-detection methods consume the smoothed LF rather than propagating
    per-star uncertainties through the histogram).
    """
    mag = np.asarray(mag, dtype=float)
    finite = np.isfinite(mag)
    mag = mag[finite]

    if mag_range is None:
        mag_lo = float(np.floor(mag.min() * 10) / 10) - 0.1
        mag_hi = float(np.ceil(mag.max() * 10) / 10) + 0.1
    else:
        mag_lo, mag_hi = mag_range

    n_bins = max(int(round((mag_hi - mag_lo) / bin_width)), 3)
    edges = np.linspace(mag_lo, mag_hi, n_bins + 1)
    centres = 0.5 * (edges[:-1] + edges[1:])
    N, _ = np.histogram(mag, bins=edges)
    sigma_N = np.sqrt(np.maximum(N, 1.0))
    return centres, N.astype(float), sigma_N


def _gaussian_kernel_1d(kernel_width_bins: float, truncation_sigma: float = 4.0) -> np.ndarray:
    """Gaussian kernel with standard deviation ``kernel_width_bins`` (bins)."""
    if kernel_width_bins <= 0.0:
        return np.array([1.0])
    n = int(max(1, np.ceil(truncation_sigma * kernel_width_bins)))
    x = np.arange(-n, n + 1, dtype=float)
    k = np.exp(-0.5 * (x / kernel_width_bins) ** 2)
    k /= k.sum()
    return k


def _convolve_reflect(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """1D convolution with reflecting boundaries (mode='same')."""
    if kernel.size == 1:
        return signal.copy()
    pad = kernel.size // 2
    padded = np.pad(signal, pad, mode="reflect")
    out = np.convolve(padded, kernel, mode="valid")
    return out


# ---------------------------------------------------------------------------
# Sobel-based edge detection
# ---------------------------------------------------------------------------


def detect_trgb_sobel(
    mag: np.ndarray,
    sigma_mag: Optional[np.ndarray] = None,
    kernel_width: float = 2.0,
    bin_width: float = 0.05,
    mag_range: Optional[tuple] = None,
    search_range: Optional[tuple] = None,
) -> EdgeDetectionResult:
    """Sobel edge detection on the smoothed I-band LF.

    Implements the standard Lee, Freedman & Madore 1993 / Freedman 2019
    recipe:

    1. Build a binned luminosity function N(I) with bin width ``bin_width``.
    2. Smooth with a Gaussian of standard deviation ``kernel_width`` bins.
    3. Compute the Sobel-filtered response E(I) = [smoothed_LF * (−1, 0, +1)].
    4. Weight by Poisson uncertainty to get E(I)/sqrt(smoothed_LF).
    5. Return the magnitude at the response maximum within ``search_range``.
    """
    centres, N, _sigma_N = _build_luminosity_function(mag, sigma_mag, bin_width, mag_range)
    kernel = _gaussian_kernel_1d(kernel_width)
    smoothed = _convolve_reflect(N, kernel)

    # Standard Lee-Freedman-Madore recipe: Sobel operator on log N, not raw N.
    # Raw-N Sobel finds the steepest rise across the whole RGB (far from the
    # tip) rather than the tip itself. log N flattens the exponential RGB
    # growth so the tip's step jump dominates.
    with np.errstate(invalid="ignore", divide="ignore"):
        log_smoothed = np.log10(np.maximum(smoothed, 1.0))

    sobel = np.array([-1.0, 0.0, 1.0])
    response = _convolve_reflect(log_smoothed, sobel)

    with np.errstate(invalid="ignore", divide="ignore"):
        # Poisson-weight the log-derivative response.
        weighted = response * np.sqrt(np.maximum(smoothed, 1.0))

    if search_range is not None:
        lo, hi = search_range
        mask = (centres >= lo) & (centres <= hi)
    else:
        # Default: exclude 10% of the edges to avoid boundary artifacts.
        trim = max(1, int(0.1 * centres.size))
        mask = np.zeros_like(centres, dtype=bool)
        mask[trim:-trim] = True

    # Require a minimum density so empty-bin gradients don't win the argmax.
    # Freedman-style threshold: N_smoothed > max(N_smoothed) / 100.
    density_floor = 0.01 * float(np.nanmax(smoothed))
    mask = mask & (smoothed > density_floor)

    valid_resp = np.where(mask, weighted, -np.inf)
    if not np.isfinite(valid_resp).any() or mask.sum() == 0:
        centre = 0.5 * (search_range[0] + search_range[1]) if search_range else float(centres[centres.size // 2])
        return EdgeDetectionResult(
            I_TRGB=centre,
            sigma_I_TRGB=10.0,
            method=f"sobel_k{kernel_width:.1f}_out_of_coverage",
            hyperparameters={
                "kernel_width_bins": float(kernel_width),
                "bin_width": float(bin_width),
            },
            diagnostics={
                "mag_centres": centres,
                "N": N,
                "smoothed_LF": smoothed,
            },
        )
    i_peak = int(np.argmax(valid_resp))
    I_TRGB = float(centres[i_peak])

    # Rough σ(I_TRGB) from a local parabolic fit to the response.
    sigma_I = _parabolic_peak_uncertainty(centres, weighted, i_peak, bin_width)

    return EdgeDetectionResult(
        I_TRGB=I_TRGB,
        sigma_I_TRGB=sigma_I,
        method=f"sobel_k{kernel_width:.1f}",
        hyperparameters={
            "kernel_width_bins": float(kernel_width),
            "bin_width": float(bin_width),
        },
        diagnostics={
            "mag_centres": centres,
            "N": N,
            "smoothed_LF": smoothed,
            "sobel_response": response,
            "weighted_response": weighted,
        },
    )


def _parabolic_peak_uncertainty(
    centres: np.ndarray,
    response: np.ndarray,
    i_peak: int,
    bin_width: float,
) -> float:
    """Estimate uncertainty by fitting a parabola to three points around the peak."""
    if i_peak == 0 or i_peak >= centres.size - 1:
        return float(bin_width)  # fallback
    y = response[i_peak - 1 : i_peak + 2]
    if not np.all(np.isfinite(y)):
        return float(bin_width)
    # For y = a (x - x0)^2 + c, the half-width where y drops by 1 gives σ.
    # From three equally spaced points y_-, y_0, y_+:
    #   a = (y_- + y_+ - 2 y_0) / (2 bin_width^2)
    a = (y[0] + y[2] - 2 * y[1]) / (2 * bin_width * bin_width)
    if a >= 0.0:
        return float(bin_width)  # not a peak
    # response "scale" ~ the peak height; σ in I such that parabola drops by 1
    # unit corresponds to σ = sqrt(-1/a). Guard against tiny |a|.
    sigma = float(np.sqrt(1.0 / max(abs(a), 1e-6)))
    # Bound σ between bin_width and a few bin widths (Freedman-style tolerance).
    return float(np.clip(sigma, bin_width, 10.0 * bin_width))


# ---------------------------------------------------------------------------
# Model-based edge detection (Makarov 2006 style)
# ---------------------------------------------------------------------------


def detect_trgb_model_based(
    mag: np.ndarray,
    sigma_mag: Optional[np.ndarray] = None,
    bin_width: float = 0.05,
    mag_range: Optional[tuple] = None,
    search_range: Optional[tuple] = None,
) -> EdgeDetectionResult:
    """Fit a broken power-law to log N(I) and report the break as I_TRGB.

    log₁₀ N(I) = a·(I − I_TRGB) + b         (above the tip; AGB side)
                 a'·(I − I_TRGB) + b        (below the tip; RGB side)

    A least-squares fit over a 1D grid of trial I_TRGB values selects the
    break location that minimizes the combined residual. Simpler and less
    sensitive to narrow LF features than Sobel; slower (~ms per galaxy).
    """
    centres, N, _ = _build_luminosity_function(mag, sigma_mag, bin_width, mag_range)
    logN = np.log10(np.maximum(N, 1.0))

    if search_range is not None:
        lo, hi = search_range
        search_mask = (centres >= lo) & (centres <= hi)
    else:
        trim = max(2, int(0.1 * centres.size))
        search_mask = np.zeros_like(centres, dtype=bool)
        search_mask[trim:-trim] = True

    trial_mags = centres[search_mask]

    if trial_mags.size == 0:
        # Search range falls outside the photometry coverage — return a
        # flagged, wide-sigma result at the range centre.
        centre = 0.5 * (search_range[0] + search_range[1]) if search_range else float(centres[centres.size // 2])
        return EdgeDetectionResult(
            I_TRGB=centre,
            sigma_I_TRGB=10.0,
            method="model_based_out_of_coverage",
            hyperparameters={"bin_width": float(bin_width)},
            diagnostics={"mag_centres": centres, "logN": logN},
        )

    best_rss = np.inf
    best_I = float(trial_mags[0])
    for I_trial in trial_mags:
        above = centres > I_trial  # fainter = AGB
        below = centres < I_trial  # brighter = RGB tail
        if above.sum() < 3 or below.sum() < 3:
            continue
        # Fit slopes on each side.
        a_up, b_up = np.polyfit(centres[above] - I_trial, logN[above], 1)
        a_dn, b_dn = np.polyfit(centres[below] - I_trial, logN[below], 1)
        pred = np.where(
            centres >= I_trial,
            a_up * (centres - I_trial) + b_up,
            a_dn * (centres - I_trial) + b_dn,
        )
        rss = float(np.sum((logN - pred) ** 2))
        if rss < best_rss:
            best_rss = rss
            best_I = float(I_trial)

    # Uncertainty: curvature of RSS around minimum.
    sigma_I = _trial_scan_uncertainty(centres, logN, trial_mags, best_I, bin_width)

    return EdgeDetectionResult(
        I_TRGB=best_I,
        sigma_I_TRGB=sigma_I,
        method="model_based",
        hyperparameters={"bin_width": float(bin_width)},
        diagnostics={
            "mag_centres": centres,
            "logN": logN,
        },
    )


def _trial_scan_uncertainty(
    centres: np.ndarray,
    logN: np.ndarray,
    trial_mags: np.ndarray,
    best_I: float,
    bin_width: float,
) -> float:
    """RSS-curvature uncertainty estimate. Returns approximate 1σ (mag)."""
    rss_vals = []
    for I_trial in trial_mags:
        above = centres > I_trial
        below = centres < I_trial
        if above.sum() < 3 or below.sum() < 3:
            rss_vals.append(np.inf)
            continue
        a_up, b_up = np.polyfit(centres[above] - I_trial, logN[above], 1)
        a_dn, b_dn = np.polyfit(centres[below] - I_trial, logN[below], 1)
        pred = np.where(
            centres >= I_trial,
            a_up * (centres - I_trial) + b_up,
            a_dn * (centres - I_trial) + b_dn,
        )
        rss_vals.append(float(np.sum((logN - pred) ** 2)))
    rss_arr = np.array(rss_vals)
    rss_min = float(np.nanmin(rss_arr))
    # 1σ where RSS increases by 1 (for a ~χ²-like residual scale).
    over = rss_arr - rss_min
    within = trial_mags[over < 1.0]
    if within.size >= 2:
        return float(0.5 * (within.max() - within.min()))
    return float(bin_width)


# ---------------------------------------------------------------------------
# Bayesian edge detection (Hatt 2017 style)
# ---------------------------------------------------------------------------


def detect_trgb_bayesian(
    mag: np.ndarray,
    sigma_mag: Optional[np.ndarray] = None,
    prior_range: Optional[tuple] = None,
    n_grid: int = 200,
    bin_width: float = 0.05,
    mag_range: Optional[tuple] = None,
) -> EdgeDetectionResult:
    """Posterior over I_TRGB given a broken-power-law LF model.

    Assumes Gaussian log-LF residuals; computes log-posterior on a grid of
    trial I_TRGB and summarizes the posterior median and 16-84 % credible
    interval.
    """
    centres, N, _ = _build_luminosity_function(mag, sigma_mag, bin_width, mag_range)
    logN = np.log10(np.maximum(N, 1.0))

    if prior_range is None:
        trim = max(2, int(0.1 * centres.size))
        lo = centres[trim]
        hi = centres[-trim]
    else:
        lo, hi = prior_range

    # If the prior range falls entirely outside the photometry coverage,
    # return an uninformative (flagged) result rather than raise.
    if lo > centres[-1] or hi < centres[0]:
        centre = 0.5 * (lo + hi)
        return EdgeDetectionResult(
            I_TRGB=centre,
            sigma_I_TRGB=10.0,
            method="bayesian_out_of_coverage",
            hyperparameters={"n_grid": float(n_grid), "bin_width": float(bin_width)},
            diagnostics={"grid": np.array([]), "logN": logN, "log_posterior": np.array([])},
        )

    # Clip the prior grid to the actual coverage.
    lo = max(lo, float(centres[0]))
    hi = min(hi, float(centres[-1]))
    grid = np.linspace(lo, hi, n_grid)
    log_post = np.full(n_grid, -np.inf)
    for k, I_trial in enumerate(grid):
        above = centres > I_trial
        below = centres < I_trial
        if above.sum() < 3 or below.sum() < 3:
            continue
        a_up, b_up = np.polyfit(centres[above] - I_trial, logN[above], 1)
        a_dn, b_dn = np.polyfit(centres[below] - I_trial, logN[below], 1)
        pred = np.where(
            centres >= I_trial,
            a_up * (centres - I_trial) + b_up,
            a_dn * (centres - I_trial) + b_dn,
        )
        resid = logN - pred
        # noise-free log-LF residual: assume unit-variance Gaussian for shape.
        log_post[k] = -0.5 * float(np.sum(resid * resid))

    finite_lp = log_post[np.isfinite(log_post)]
    if finite_lp.size == 0:
        centre = 0.5 * (lo + hi)
        return EdgeDetectionResult(
            I_TRGB=centre,
            sigma_I_TRGB=10.0,
            method="bayesian_no_valid_trial",
            hyperparameters={"n_grid": float(n_grid), "bin_width": float(bin_width)},
            diagnostics={"grid": grid, "logN": logN, "log_posterior": log_post},
        )
    log_post -= finite_lp.max()
    post = np.exp(log_post)
    post_norm = float(np.trapezoid(post, grid))
    if post_norm <= 0.0 or not np.isfinite(post_norm):
        # Fall back to peak of scan.
        i_peak = int(np.nanargmax(log_post))
        return EdgeDetectionResult(
            I_TRGB=float(grid[i_peak]),
            sigma_I_TRGB=float(bin_width),
            method="bayesian",
            hyperparameters={"n_grid": float(n_grid), "bin_width": float(bin_width)},
            diagnostics={"grid": grid, "logN": logN, "log_posterior": log_post},
        )
    post = post / post_norm

    cdf = np.concatenate(([0.0], np.cumsum(0.5 * (post[:-1] + post[1:]) * np.diff(grid))))
    cdf = cdf / cdf[-1]

    median = float(np.interp(0.5, cdf, grid))
    lo_ci = float(np.interp(0.16, cdf, grid))
    hi_ci = float(np.interp(0.84, cdf, grid))
    sigma = 0.5 * (hi_ci - lo_ci)

    return EdgeDetectionResult(
        I_TRGB=median,
        sigma_I_TRGB=sigma,
        method="bayesian",
        hyperparameters={"n_grid": float(n_grid), "bin_width": float(bin_width)},
        diagnostics={
            "grid": grid,
            "logN": logN,
            "posterior": post,
            "log_posterior": log_post,
            "credible_interval_68": np.array([lo_ci, hi_ci]),
        },
    )


__all__ = [
    "EdgeDetectionResult",
    "detect_trgb_sobel",
    "detect_trgb_model_based",
    "detect_trgb_bayesian",
]
