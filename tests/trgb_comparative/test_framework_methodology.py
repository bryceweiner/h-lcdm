"""Framework behavior tests — the framework is prediction-only."""

from __future__ import annotations

import numpy as np

from pipeline.trgb_comparative.framework_methodology import (
    FrameworkMethodology,
    FrameworkPrediction,
)


def test_framework_predicts_case_b_reliable_regime():
    fw = FrameworkMethodology()
    pred = fw.predict(
        label="test_ngc4258_anchor",
        d_local_mpc=7.58, sigma_d_local_mpc=0.08,
        n_samples=5000, seed=0,
    )
    assert isinstance(pred, FrameworkPrediction)
    assert pred.breakdown_fraction < 0.05
    assert 70.0 < pred.H0_median < 76.0
    assert pred.H0_low < pred.H0_median < pred.H0_high


def test_framework_predicts_case_a_breakdown_regime():
    fw = FrameworkMethodology()
    pred = fw.predict(
        label="test_lmc_anchor",
        d_local_mpc=0.0496, sigma_d_local_mpc=0.0009,
        n_samples=5000, seed=0,
    )
    assert pred.breakdown_fraction > 0.95
    assert pred.breakdown_flag_any
    assert 78.0 < pred.H0_median < 85.0
    assert len(pred.breakdown_messages) >= 1


def test_framework_has_no_data_loading_imports():
    """Framework methodology module must not import data loaders."""
    import pipeline.trgb_comparative.framework_methodology as fm
    src = __import__("inspect").getsource(fm)
    assert "data.loader" not in src, (
        "framework_methodology.py must not import from data.loader — "
        "framework is prediction only."
    )
    assert "DataLoader" not in src


def test_framework_prediction_dict_has_expected_keys():
    fw = FrameworkMethodology()
    pred = fw.predict(
        label="test",
        d_local_mpc=7.58, sigma_d_local_mpc=0.08,
        n_samples=200, seed=0,
    )
    d = pred.as_dict()
    for k in ("label", "H0_median", "H0_low", "H0_high", "breakdown_fraction",
              "breakdown_flag_any", "breakdown_messages", "inputs", "n_samples"):
        assert k in d
