"""
Compute-backend abstraction for the TRGB comparative pipeline.

Provides a thin wrapper that exposes a NumPy-compatible API backed by either
NumPy (reference implementation) or Apple MLX (accelerated path on Apple
Silicon). All call sites should go through the backend returned by
:func:`get_backend`. NumPy is always the reference; MLX results must agree
with NumPy within documented tolerance (see tests/test_compute_backend.py).

Design choice: we do NOT monkey-patch numpy or replace numpy in callers.
The backend exposes just the few vectorized primitives we actually use in
hot paths (elementwise math, convolution, percentile, random draws). Callers
that only need numpy keep importing numpy directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Optional

import numpy as np

try:  # pragma: no cover - import-only branch
    import mlx.core as mx  # type: ignore

    HAS_MLX = True
except ImportError:  # pragma: no cover - absence is the common path on non-mac
    mx = None  # type: ignore
    HAS_MLX = False


# ---------------------------------------------------------------------------
# Backend dataclasses
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _BackendBase:
    """Common fields surfaced to callers."""

    name: str
    dtype_default: str
    mlx_available: bool

    def to_numpy(self, x: Any) -> np.ndarray:  # pragma: no cover - overridden
        raise NotImplementedError

    def asarray(self, x: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    # Vectorized math surface (just what hot paths need).
    def log(self, x: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def exp(self, x: Any) -> Any:  # pragma: no cover - overridden
        raise NotImplementedError

    def normal(self, mean: float, sigma: float, size: int, seed: Optional[int] = None) -> Any:
        raise NotImplementedError  # pragma: no cover - overridden


class _NumpyBackend(_BackendBase):
    """Reference NumPy backend. Always available."""

    def __init__(self) -> None:
        object.__setattr__(self, "name", "numpy")
        object.__setattr__(self, "dtype_default", "float64")
        object.__setattr__(self, "mlx_available", HAS_MLX)

    def to_numpy(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def asarray(self, x: Any) -> np.ndarray:
        return np.asarray(x)

    def log(self, x: Any) -> np.ndarray:
        return np.log(np.asarray(x))

    def exp(self, x: Any) -> np.ndarray:
        return np.exp(np.asarray(x))

    def normal(self, mean: float, sigma: float, size: int, seed: Optional[int] = None) -> np.ndarray:
        rng = np.random.default_rng(seed)
        return rng.normal(loc=mean, scale=sigma, size=size)


class _MLXBackend(_BackendBase):
    """MLX-accelerated backend. Falls back to NumPy for ops not on hot path."""

    def __init__(self) -> None:
        object.__setattr__(self, "name", "mlx")
        # MLX uses float32 by default on most hardware; float64 support is
        # limited. We expose float32 as the default dtype; tests tolerate
        # the precision difference.
        object.__setattr__(self, "dtype_default", "float32")
        object.__setattr__(self, "mlx_available", True)

    def to_numpy(self, x: Any) -> np.ndarray:
        if mx is not None and isinstance(x, mx.array):  # type: ignore[attr-defined]
            return np.asarray(x)
        return np.asarray(x)

    def asarray(self, x: Any) -> Any:
        if mx is None:
            return np.asarray(x)
        if isinstance(x, mx.array):  # type: ignore[attr-defined]
            return x
        return mx.array(np.asarray(x, dtype=np.float32))

    def log(self, x: Any) -> Any:
        if mx is None:
            return np.log(np.asarray(x))
        return mx.log(self.asarray(x))

    def exp(self, x: Any) -> Any:
        if mx is None:
            return np.exp(np.asarray(x))
        return mx.exp(self.asarray(x))

    def normal(self, mean: float, sigma: float, size: int, seed: Optional[int] = None) -> Any:
        # MLX's random has a different seeding API; we delegate to NumPy's RNG
        # for reproducibility and then upload. Sampling overhead is small.
        rng = np.random.default_rng(seed)
        draws = rng.normal(loc=mean, scale=sigma, size=size).astype(np.float32)
        if mx is None:
            return draws
        return mx.array(draws)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


_NUMPY_SINGLETON: Optional[_NumpyBackend] = None
_MLX_SINGLETON: Optional[_MLXBackend] = None


def get_backend(prefer: str = "auto") -> _BackendBase:
    """Return the requested backend.

    Parameters
    ----------
    prefer:
        One of ``{"auto", "mlx", "numpy"}``. ``"auto"`` picks MLX when it is
        installed, otherwise NumPy. ``"mlx"`` requires MLX to be available
        (raises ImportError otherwise). ``"numpy"`` always returns NumPy.
    """
    global _NUMPY_SINGLETON, _MLX_SINGLETON

    if prefer not in ("auto", "mlx", "numpy"):
        raise ValueError(f"unknown backend preference: {prefer!r}")

    if prefer == "numpy":
        if _NUMPY_SINGLETON is None:
            _NUMPY_SINGLETON = _NumpyBackend()
        return _NUMPY_SINGLETON

    if prefer == "mlx":
        if not HAS_MLX:
            raise ImportError("MLX backend requested but mlx is not installed")
        if _MLX_SINGLETON is None:
            _MLX_SINGLETON = _MLXBackend()
        return _MLX_SINGLETON

    # auto
    if HAS_MLX:
        if _MLX_SINGLETON is None:
            _MLX_SINGLETON = _MLXBackend()
        return _MLX_SINGLETON
    if _NUMPY_SINGLETON is None:
        _NUMPY_SINGLETON = _NumpyBackend()
    return _NUMPY_SINGLETON


def available_backends() -> list[str]:
    """List of backends the current environment supports."""
    out = ["numpy"]
    if HAS_MLX:
        out.append("mlx")
    return out
