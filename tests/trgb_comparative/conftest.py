"""
Fixtures for the TRGB comparative test suite.

Synthesizes realistic RGB+AGB CMDs so edge detection and pipeline wiring
can be exercised without access to real archival photometry. Mocks for
likelihood inputs keep MCMC smoke tests cheap.
"""

from __future__ import annotations

import numpy as np
import pytest


def synth_rgb_agb_mags(
    I_TRGB_true: float = 20.0,
    n_rgb_total: int = 5000,
    rgb_density_tip: float = 400.0,      # stars per mag at the tip (RGB side)
    agb_density_tip: float = 40.0,       # stars per mag at the tip (AGB side; ~10x lower)
    rgb_slope: float = 0.30,
    agb_slope: float = 0.30,
    completeness_mag: float = 2.5,
    agb_range_mag: float = 2.0,
    noise_sigma: float = 0.03,
    seed: int = 0,
) -> tuple:
    """Generate realistic RGB/AGB magnitudes with a sharp factor-of-10 break at I_TRGB.

    The key feature of a real TRGB is a near-discontinuous jump in N(I)
    at the tip — AGB density is ~10× smaller than RGB density at the tip.

    Returns (mag, sigma_mag).
    """
    rng = np.random.default_rng(seed)
    # RGB draws: density ∝ rgb_density_tip * 10^(rgb_slope * (I - I_TRGB))
    # integrated to N_rgb total over [I_TRGB, I_TRGB + completeness_mag]:
    #    N_rgb = rgb_density_tip * (10^(rgb_slope * completeness_mag) - 1) / (rgb_slope * ln(10))
    n_rgb = int(rgb_density_tip * (10.0 ** (rgb_slope * completeness_mag) - 1.0) / (rgb_slope * np.log(10.0)))
    u_rgb = rng.uniform(0.0, 1.0, n_rgb)
    I_rgb = I_TRGB_true + np.log10(
        1.0 + u_rgb * (10.0 ** (rgb_slope * completeness_mag) - 1.0)
    ) / rgb_slope

    # AGB draws: density ∝ agb_density_tip * 10^(agb_slope * (I - I_TRGB)) for I < I_TRGB.
    n_agb = int(agb_density_tip * (1.0 - 10.0 ** (-agb_slope * agb_range_mag)) / (agb_slope * np.log(10.0)))
    u_agb = rng.uniform(0.0, 1.0, n_agb)
    I_agb = I_TRGB_true + np.log10(
        1.0 - u_agb * (1.0 - 10.0 ** (-agb_slope * agb_range_mag))
    ) / agb_slope

    mags = np.concatenate([I_rgb, I_agb])
    mags = mags + rng.normal(0.0, noise_sigma, size=mags.size)
    sigma = np.full_like(mags, noise_sigma)
    return mags, sigma


@pytest.fixture
def synthetic_cmd_with_tip_at_20():
    mag, sigma = synth_rgb_agb_mags(I_TRGB_true=20.0, seed=0)
    return mag, sigma


@pytest.fixture
def synthetic_cmd_with_tip_at_24():
    mag, sigma = synth_rgb_agb_mags(I_TRGB_true=24.0, seed=1)
    return mag, sigma
