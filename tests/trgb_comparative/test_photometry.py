"""Edge-detection method tests on synthetic RGB/AGB CMDs."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.trgb_comparative.photometry import (
    EdgeDetectionResult,
    detect_trgb_bayesian,
    detect_trgb_model_based,
    detect_trgb_sobel,
)


def test_sobel_recovers_tip_within_one_mag(synthetic_cmd_with_tip_at_20):
    mag, sigma = synthetic_cmd_with_tip_at_20
    # Give the detector a tip-region prior window; real pipelines always do.
    res = detect_trgb_sobel(mag, sigma, kernel_width=2.0, search_range=(19.5, 20.5))
    assert isinstance(res, EdgeDetectionResult)
    assert abs(res.I_TRGB - 20.0) < 1.0, f"Sobel tip off by {res.I_TRGB - 20.0:.3f}"
    assert res.sigma_I_TRGB > 0


def test_model_based_recovers_tip_within_half_mag(synthetic_cmd_with_tip_at_20):
    mag, sigma = synthetic_cmd_with_tip_at_20
    res = detect_trgb_model_based(mag, sigma, search_range=(19.5, 20.5))
    assert abs(res.I_TRGB - 20.0) < 0.5


def test_bayesian_recovers_tip_within_half_mag(synthetic_cmd_with_tip_at_20):
    mag, sigma = synthetic_cmd_with_tip_at_20
    res = detect_trgb_bayesian(mag, sigma, prior_range=(19.0, 21.0))
    assert abs(res.I_TRGB - 20.0) < 0.5


def test_all_methods_agree_roughly(synthetic_cmd_with_tip_at_24):
    mag, sigma = synthetic_cmd_with_tip_at_24
    s = detect_trgb_sobel(mag, sigma, kernel_width=2.0, search_range=(23.5, 24.5))
    m = detect_trgb_model_based(mag, sigma, search_range=(23.5, 24.5))
    b = detect_trgb_bayesian(mag, sigma, prior_range=(23.0, 25.0))
    tips = np.array([s.I_TRGB, m.I_TRGB, b.I_TRGB])
    assert np.all(np.abs(tips - 24.0) < 1.0)


def test_edge_detection_result_to_dict_serializable(synthetic_cmd_with_tip_at_20):
    mag, sigma = synthetic_cmd_with_tip_at_20
    res = detect_trgb_sobel(mag, sigma)
    d = res.to_dict()
    assert d["method"].startswith("sobel")
    assert "diagnostics" in d
    assert isinstance(d["diagnostics"], dict)
