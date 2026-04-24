"""Distance-ladder unit tests."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.trgb_comparative.distance_ladder import (
    GeometricAnchor,
    absolute_trgb_from_anchor_photometry,
    assemble_distance_chain,
    fit_hubble_flow,
    host_distance_modulus_from_trgb,
)


def test_geometric_anchor_distance_conversion():
    lmc = GeometricAnchor(name="LMC", mu=18.477, sigma_mu_stat=0.026,
                          sigma_mu_sys=0.024, reference="Pietrzyński 2019")
    # μ = 18.477 → d = 10^(18.477/5 + 1) pc = 10^4.6954 pc ≈ 49.6 kpc
    assert lmc.distance_mpc == pytest.approx(0.0496, rel=0.01)


def test_absolute_trgb_and_host_distance_roundtrip():
    lmc = GeometricAnchor(name="LMC", mu=18.477, sigma_mu_stat=0.026,
                          sigma_mu_sys=0.024, reference="Pietrzyński 2019")
    M, sigma_M = absolute_trgb_from_anchor_photometry(
        I_TRGB_observed=14.37, sigma_I_TRGB=0.05, anchor=lmc
    )
    assert M == pytest.approx(14.37 - 18.477)
    # Reconstruct a mock host at μ=30.
    host = host_distance_modulus_from_trgb(
        I_TRGB_host=30.0 + M,
        sigma_I_TRGB_host=0.05,
        M_TRGB=M, sigma_M_TRGB=sigma_M,
        anchor_name=lmc.name, host_name="NGC_XYZ",
    )
    assert host.mu_TRGB == pytest.approx(30.0, abs=1e-6)


def test_fit_hubble_flow_on_mock_lcdm_recovers_input_H0():
    from pipeline.expansion_enhancement.cosmology import make_H_callable, mu_model
    true_H0 = 70.5
    z_flow = np.linspace(0.02, 0.15, 30)
    H = make_H_callable(true_H0, 0.315, 0.0, mode="constant")
    mu_flow = mu_model(z_flow, H) + 19.25  # add an arbitrary M_B offset
    # Identity covariance with sigma = 0.05 mag per SN.
    inv_cov = np.eye(z_flow.size) / 0.05 ** 2
    fit = fit_hubble_flow(z_flow, mu_flow, inv_cov, Om=0.315)
    assert abs(fit.H0 - true_H0) < 0.5, f"expected {true_H0}; got {fit.H0:.2f}"


def test_assemble_distance_chain_raises_on_invalid_case():
    lmc = GeometricAnchor(name="LMC", mu=18.477, sigma_mu_stat=0.026,
                          sigma_mu_sys=0.024, reference="Pietrzyński 2019")
    with pytest.raises(ValueError):
        assemble_distance_chain("bogus", lmc, 14.37, 0.05, [("X", 26.0, 0.1)])
