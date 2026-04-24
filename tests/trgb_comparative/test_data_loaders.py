"""Schema / availability tests for data loaders."""

from __future__ import annotations

import pytest

from data.loader import DataLoader, DataUnavailableError


def test_lmc_distance_has_expected_schema():
    d = DataLoader().load_pietrzynski_lmc_distance()
    for key in ("mu", "sigma_mu_stat", "sigma_mu_sys", "d_kpc", "reference"):
        assert key in d, f"missing key {key!r}"
    assert d["mu"] == pytest.approx(18.477, abs=1e-3)
    assert d["d_kpc"] == pytest.approx(49.59, abs=0.1)


def test_ngc4258_distance_has_expected_schema():
    d = DataLoader().load_reid_ngc4258_distance()
    for key in ("mu", "sigma_mu_stat", "sigma_mu_sys", "d_mpc", "reference"):
        assert key in d
    assert d["mu"] == pytest.approx(29.397, abs=1e-3)
    assert d["d_mpc"] == pytest.approx(7.58, abs=0.1)


def test_missing_photometry_raises_clean_error():
    dl = DataLoader(downloaded_data_dir="/tmp/__trgb_test_nonexistent__")
    with pytest.raises(DataUnavailableError):
        dl.load_lmc_trgb_hst_photometry()
    with pytest.raises(DataUnavailableError):
        dl.load_sn_host_trgb_hst_photometry()
    with pytest.raises(DataUnavailableError):
        dl.load_sn_host_trgb_jwst_photometry()
    with pytest.raises(DataUnavailableError):
        dl.load_anand_trgb_catalog()


def test_check_data_availability_reports_trgb_keys():
    dl = DataLoader()
    avail = dl.check_data_availability()
    for key in (
        "lmc_trgb_hst",
        "sn_host_trgb_hst_manifest",
        "sn_host_trgb_jwst_manifest",
        "anand_trgb_catalog",
    ):
        assert key in avail
