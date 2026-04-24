"""NumPy vs MLX parity tests (skipped when MLX not installed)."""

from __future__ import annotations

import numpy as np
import pytest

from pipeline.trgb_comparative.compute_backend import (
    HAS_MLX,
    available_backends,
    get_backend,
)


def test_numpy_backend_available_and_callable():
    be = get_backend("numpy")
    assert be.name == "numpy"
    x = np.array([1.0, 2.0, 3.0])
    y = be.log(x)
    assert np.allclose(y, np.log(x))


def test_auto_backend_selects_mlx_when_available():
    be = get_backend("auto")
    expected = "mlx" if HAS_MLX else "numpy"
    assert be.name == expected


def test_available_backends_always_includes_numpy():
    assert "numpy" in available_backends()


def test_mlx_requested_without_install_raises():
    if HAS_MLX:
        pytest.skip("MLX installed; import-error path not reachable")
    with pytest.raises(ImportError):
        get_backend("mlx")


@pytest.mark.skipif(not HAS_MLX, reason="mlx not installed")
def test_mlx_numpy_parity_on_log():
    np_be = get_backend("numpy")
    mlx_be = get_backend("mlx")
    x = np.linspace(0.1, 10.0, 100)
    np_log = np_be.log(x)
    mlx_log = mlx_be.to_numpy(mlx_be.log(x))
    # MLX is float32 — loosen tolerance accordingly.
    assert np.allclose(np_log, mlx_log, atol=1e-5)


def test_unknown_preference_raises():
    with pytest.raises(ValueError):
        get_backend("cuda")
