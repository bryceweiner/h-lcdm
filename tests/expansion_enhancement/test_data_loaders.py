"""
Data loader tests: shapes, units, PSD covariances.

The DESI DR1 and Planck θ* loaders are offline (hardcoded published values), so
they're exercised unconditionally. The Pantheon+ loader hits the network, so
it's gated behind RUN_E2E=1.
"""

import os

import numpy as np
import pytest

from data.loader import DataLoader
from pipeline.expansion_enhancement.data_loaders import load_bao, load_cmb


def test_desi_dr1_shape_and_psd():
    loader = DataLoader()
    raw = loader.load_desi_dr1_bao_full()
    n = len(raw["z"])
    assert n == 12, f"Expected 12 DESI DR1 rows, got {n}"
    # Published z-effective bins (DESI 2024 Table 1).
    z_unique = sorted(set(raw["z"].tolist()))
    assert z_unique == [0.295, 0.510, 0.706, 0.930, 1.317, 1.491, 2.330]
    # Covariance positive-definite.
    w = np.linalg.eigvalsh(raw["cov"])
    assert np.all(w > 0), f"DESI DR1 covariance not PSD: eigenvalues {w}"
    # Intra-bin correlations are negative (D_M/D_H anti-correlated).
    # First such pair: rows 1,2 at z=0.510.
    rho = raw["cov"][1, 2] / np.sqrt(raw["cov"][1, 1] * raw["cov"][2, 2])
    assert rho < 0


def test_planck_theta_star():
    loader = DataLoader()
    tp = loader.load_planck_2018_theta_star()
    # D_M(z*) ~ 13870 Mpc with ~4 Mpc uncertainty.
    assert 13000 < tp["D_M_z_rec"] < 14500
    assert 0 < tp["sigma_D_M"] < 20
    # θ* still exposed for reporting.
    assert 0.01 < tp["theta_star"] < 0.011


def test_bao_wrapper_returns_inv_cov():
    """Pipeline-level load_bao must include a usable inverse covariance."""
    loader = DataLoader()
    bao = load_bao(loader)
    identity = bao.cov @ bao.inv_cov
    np.testing.assert_allclose(identity, np.eye(bao.cov.shape[0]), atol=1e-8)


def test_cmb_wrapper():
    loader = DataLoader()
    cmb = load_cmb(loader)
    assert cmb.sigma_D_M > 0
    assert cmb.D_M_z_rec > 0


@pytest.mark.skipif(os.environ.get("RUN_E2E") != "1", reason="Pantheon+ download requires network")
def test_pantheon_plus_loader():
    from pipeline.expansion_enhancement.data_loaders import load_sn

    loader = DataLoader()
    sn = load_sn(loader)
    assert sn.z.size > 1000
    assert sn.cov.shape == (sn.z.size, sn.z.size)
    # Covariance must be symmetric.
    np.testing.assert_allclose(sn.cov, sn.cov.T, atol=1e-8)
