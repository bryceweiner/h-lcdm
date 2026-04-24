"""
Cosmology-module tests: ΛCDM limit, distance ladder, ε prescriptions.
"""

import numpy as np
import pytest

from pipeline.expansion_enhancement import cosmology as cos


def test_H_framework_recovers_LCDM_when_eps_zero():
    """ε=0 must exactly reproduce the standard ΛCDM H(z)."""
    z = np.linspace(0.01, 5.0, 100)
    base = cos.H_LCDM(z, 67.4, 0.315)
    framework = cos.H_framework(z, 67.4, 0.315, 0.0, mode="constant")
    np.testing.assert_allclose(framework, base, rtol=0, atol=1e-12)

    framework_qtep = cos.H_framework(z, 67.4, 0.315, 0.0, mode="qtep")
    np.testing.assert_allclose(framework_qtep, base, rtol=0, atol=1e-12)


def test_epsilon_constant_step_discontinuity():
    """Constant-ε step turns OFF above z_rec and ON below."""
    eps = 0.022
    # Below and above z_rec.
    assert cos.epsilon_profile(500.0, eps, "constant") == pytest.approx(eps)
    assert cos.epsilon_profile(2000.0, eps, "constant") == pytest.approx(0.0)
    # Just-below vs just-above crossing boundary.
    assert cos.epsilon_profile(cos.Z_REC - 1, eps, "constant") == pytest.approx(eps)
    assert cos.epsilon_profile(cos.Z_REC + 1, eps, "constant") == pytest.approx(0.0)


def test_epsilon_qtep_monotone_in_z():
    """ε_QTEP(z) = eps * γ(z)/γ(0) grows with z (γ grows with H grows with z)."""
    eps = 0.02
    zs = np.array([0.0, 0.5, 1.0, 2.0, 10.0])
    vals = cos.epsilon_profile(zs, eps, "qtep")
    # Monotone non-decreasing.
    diffs = np.diff(vals)
    assert np.all(diffs >= -1e-12), f"ε_QTEP(z) not monotone: {vals}"
    # Above z_rec it's zero.
    assert cos.epsilon_profile(cos.Z_REC + 10, eps, "qtep") == pytest.approx(0.0)


def test_distance_ladder_relations():
    """Internal consistency: D_L = (1+z) D_M; D_A = D_M/(1+z); D_H = c/H."""
    H = cos.make_H_callable(67.4, 0.315, 0.0, mode="constant")
    z = 0.5
    dm = float(np.atleast_1d(cos.D_M(z, H))[0])
    dl = float(np.atleast_1d(cos.D_L(z, H))[0])
    da = float(np.atleast_1d(cos.D_A(z, H))[0])
    dh = float(cos.D_H(z, H))

    assert dl == pytest.approx(dm * (1.0 + z))
    assert da == pytest.approx(dm / (1.0 + z))
    assert dh == pytest.approx(cos.C_KMS / cos.H_LCDM(z, 67.4, 0.315))


def test_distance_agrees_with_astropy():
    """Sanity: D_M(z) agrees with astropy FlatLambdaCDM at the 0.1% level."""
    from astropy.cosmology import FlatLambdaCDM

    H0, Om = 67.4, 0.315
    astro = FlatLambdaCDM(H0=H0, Om0=Om)
    H = cos.make_H_callable(H0, Om, 0.0, mode="constant")
    for zi in [0.1, 0.5, 1.0, 2.0, 3.0]:
        d_ours = float(np.atleast_1d(cos.D_M(zi, H))[0])
        d_ref = astro.comoving_distance(zi).value
        assert d_ours == pytest.approx(d_ref, rel=1e-3), (zi, d_ours, d_ref)


def test_epsilon_increases_Hz_below_z_rec():
    """With ε>0, H_framework > H_LCDM for z < z_rec."""
    zs = np.array([0.1, 0.5, 1.0, 500.0])
    base = cos.H_LCDM(zs, 67.4, 0.315)
    enh = cos.H_framework(zs, 67.4, 0.315, 0.02, mode="constant")
    assert np.all(enh > base)
    # At z above z_rec, enhancement is zero.
    z_high = 2000.0
    assert cos.H_framework(z_high, 67.4, 0.315, 0.02, "constant") == pytest.approx(
        cos.H_LCDM(z_high, 67.4, 0.315)
    )


def test_theta_star_sensitivity_to_eps():
    """Enhancing H(z) for z < z_rec shrinks D_M(z_rec); θ* ≈ r_d/D_M therefore rises."""
    H_lcdm = cos.make_H_callable(67.4, 0.315, 0.0, mode="constant")
    H_fw = cos.make_H_callable(67.4, 0.315, 0.02, mode="constant")
    theta_lcdm = cos.theta_star(H_lcdm, r_d=147.5)
    theta_fw = cos.theta_star(H_fw, r_d=150.71)
    # Both effects push θ* up (r_d larger, D_M smaller), sanity-check direction.
    assert theta_fw > theta_lcdm
