"""
Thin pipeline-level wrappers around data/loader.py.

Each loader returns a self-contained dict of numpy arrays ready to feed into
:mod:`pipeline.expansion_enhancement.likelihood`. Keeping the pipeline wrappers
separate from the raw loaders lets us (a) precompute inverse-covariance
Cholesky factors once, and (b) present a uniform API regardless of what the
DataLoader calls its columns.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import numpy as np

from data.loader import DataLoader


@dataclass
class BAOData:
    z: np.ndarray
    kind: List[str]
    value: np.ndarray
    error: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray


@dataclass
class SNData:
    z: np.ndarray
    mu: np.ndarray
    cov: np.ndarray
    inv_cov: np.ndarray
    is_calibrator: np.ndarray
    M_cepheid: np.ndarray


@dataclass
class CMBDistancePrior:
    """CMB constraint on the comoving transverse distance to last scattering.

    Preferred over a θ* constraint because D_M(z*) is independent of the
    r_s* vs r_d distinction — see ``load_planck_2018_theta_star`` docstring.
    ``theta_star`` / ``sigma_theta_star`` are retained only for reporting.
    """

    D_M_z_rec: float
    sigma_D_M: float
    z_rec: float
    theta_star: float = 0.0      # reporting-only, not used in likelihood
    sigma_theta_star: float = 0.0


# Backward-compat alias so older code paths still import cleanly.
CMBThetaStar = CMBDistancePrior


@dataclass
class ExpansionDataBundle:
    bao: BAOData
    sn: SNData
    cmb: CMBDistancePrior

    @property
    def n_data(self) -> int:
        return self.bao.value.size + self.sn.z.size + 1


def _safe_inverse(cov: np.ndarray) -> np.ndarray:
    """Numerically stable inverse via Cholesky, falling back to pinv."""
    try:
        L = np.linalg.cholesky(cov)
        return np.linalg.solve(L.T, np.linalg.solve(L, np.eye(cov.shape[0])))
    except np.linalg.LinAlgError:
        return np.linalg.pinv(cov)


def load_bao(loader: DataLoader) -> BAOData:
    raw = loader.load_desi_dr1_bao_full()
    return BAOData(
        z=np.asarray(raw['z'], dtype=float),
        kind=list(raw['type']),
        value=np.asarray(raw['value'], dtype=float),
        error=np.asarray(raw['error'], dtype=float),
        cov=np.asarray(raw['cov'], dtype=float),
        inv_cov=_safe_inverse(np.asarray(raw['cov'], dtype=float)),
    )


def load_sn(loader: DataLoader) -> SNData:
    raw = loader.load_pantheon_plus()
    cov = np.asarray(raw['cov'], dtype=float)
    return SNData(
        z=np.asarray(raw['z'], dtype=float),
        mu=np.asarray(raw['mu'], dtype=float),
        cov=cov,
        inv_cov=_safe_inverse(cov),
        is_calibrator=np.asarray(raw['is_calibrator'], dtype=bool),
        M_cepheid=np.asarray(raw['M_cepheid'], dtype=float),
    )


def load_cmb(loader: DataLoader) -> CMBDistancePrior:
    raw = loader.load_planck_2018_theta_star()
    return CMBDistancePrior(
        D_M_z_rec=float(raw['D_M_z_rec']),
        sigma_D_M=float(raw['sigma_D_M']),
        z_rec=float(raw['z_rec']),
        theta_star=float(raw.get('theta_star', 0.0)),
        sigma_theta_star=float(raw.get('sigma_theta_star', 0.0)),
    )


def load_all(loader: DataLoader | None = None) -> ExpansionDataBundle:
    if loader is None:
        loader = DataLoader()
    return ExpansionDataBundle(
        bao=load_bao(loader),
        sn=load_sn(loader),
        cmb=load_cmb(loader),
    )
