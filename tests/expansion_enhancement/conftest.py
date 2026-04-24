"""
Fixtures for the expansion-enhancement test suite.

Mocked data lets likelihood / MCMC tests run without hitting the network.
Real Pantheon+ downloads are exercised only by the end-to-end test, which is
skipped unless ``RUN_E2E=1`` is set in the environment.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

from pipeline.expansion_enhancement.data_loaders import (
    BAOData,
    CMBDistancePrior,
    ExpansionDataBundle,
    SNData,
    _safe_inverse,
)
from pipeline.expansion_enhancement.cosmology import D_A, D_H, D_M, D_V, mu_model, make_H_callable
from pipeline.expansion_enhancement.likelihood import MODEL_A, R_D_LCDM


@pytest.fixture
def fiducial_params():
    """Planck 2018-like central cosmology."""
    return {"H0": 67.4, "Om": 0.315, "eps": 0.0}


@pytest.fixture
def fake_bao(fiducial_params):
    """Synthetic DESI-DR1-shaped BAO dataset generated from the fiducial cosmology.

    If we feed the fiducial params back in, χ²(BAO) must be 0 to high
    precision — an internal consistency check for the pipeline.
    """
    # Mimic the DESI DR1 layout but with only a few points so tests are fast.
    rows = [
        (0.295, "D_V/r_d", 0.10),
        (0.510, "D_M/r_d", 0.10),
        (0.510, "D_H/r_d", 0.10),
        (0.930, "D_M/r_d", 0.15),
        (2.330, "D_M/r_d", 0.30),
    ]
    z = np.array([r[0] for r in rows])
    kind = [r[1] for r in rows]
    error = np.array([r[2] for r in rows])

    H = make_H_callable(fiducial_params["H0"], fiducial_params["Om"], 0.0, mode="constant")
    value = np.zeros(len(rows))
    for i, (zi, k) in enumerate(zip(z, kind)):
        if k == "D_M/r_d":
            value[i] = float(np.atleast_1d(D_M(zi, H))[0]) / R_D_LCDM
        elif k == "D_H/r_d":
            value[i] = float(np.atleast_1d(D_H(zi, H))[0]) / R_D_LCDM
        elif k == "D_V/r_d":
            value[i] = float(np.atleast_1d(D_V(zi, H))[0]) / R_D_LCDM
    cov = np.diag(error ** 2)
    return BAOData(z=z, kind=kind, value=value, error=error, cov=cov, inv_cov=_safe_inverse(cov))


@pytest.fixture
def fake_sn(fiducial_params):
    """Synthetic SN distance moduli at DESI-like redshifts."""
    rng = np.random.default_rng(42)
    z = np.array(sorted(rng.uniform(0.01, 2.3, size=50)))
    H = make_H_callable(fiducial_params["H0"], fiducial_params["Om"], 0.0, mode="constant")
    mu = mu_model(z, H)
    sigma = 0.15
    cov = np.eye(z.size) * sigma ** 2
    return SNData(
        z=z,
        mu=mu,
        cov=cov,
        inv_cov=_safe_inverse(cov),
        is_calibrator=np.zeros(z.size, dtype=bool),
        M_cepheid=np.zeros(z.size),
    )


@pytest.fixture
def fake_cmb(fiducial_params):
    """Synthetic D_M(z*) computed with the SAME D_M integrator the likelihood
    uses, so χ²_CMB(truth)=0 exactly rather than up to integrator noise."""
    from pipeline.expansion_enhancement.cosmology import D_M as _D_M

    H = make_H_callable(fiducial_params["H0"], fiducial_params["Om"], 0.0, mode="constant")
    dm = float(np.atleast_1d(_D_M(1089.80, H, n_steps=16384))[0])
    return CMBDistancePrior(D_M_z_rec=dm, sigma_D_M=1.0, z_rec=1089.80)


@pytest.fixture
def fake_bundle(fake_bao, fake_sn, fake_cmb):
    return ExpansionDataBundle(bao=fake_bao, sn=fake_sn, cmb=fake_cmb)


def _skip_if_no_network(fn):
    return pytest.mark.skipif(
        os.environ.get("RUN_E2E") != "1",
        reason="End-to-end test requires network (Pantheon+ download). Set RUN_E2E=1 to enable.",
    )(fn)


@pytest.fixture
def skip_if_no_network():
    return _skip_if_no_network
