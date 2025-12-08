"""
MCMC Inference Utilities
========================

Provides physically motivated covariance builders and a lightweight MCMC
engine (HMC) that can run on Apple's MPS (or CUDA/CPU fallback) to evaluate
Gaussian likelihoods over cosmological summary statistics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional

import numpy as np
import torch

from hlcdm.parameters import HLCDM_PARAMS


# -----------------------------------------------------------------------------
# Covariance construction helpers
# -----------------------------------------------------------------------------

def build_bao_covariance(sigmas: np.ndarray, corr: np.ndarray) -> np.ndarray:
    """
    Construct BAO covariance from 1σ errors and a published correlation matrix.

    C_ij = rho_ij * sigma_i * sigma_j
    """
    sigmas = np.asarray(sigmas)
    corr = np.asarray(corr)
    if corr.shape[0] != corr.shape[1]:
        raise ValueError("Correlation matrix must be square.")
    if corr.shape[0] != sigmas.shape[0]:
        raise ValueError("Correlation and sigma length mismatch.")
    outer = np.outer(sigmas, sigmas)
    return corr * outer


def hartlap_inverse(cov: np.ndarray, n_mock: int) -> np.ndarray:
    """
    Apply Hartlap correction to the inverse covariance when using a finite
    number of mocks (n_mock > dim + 2).
    """
    cov = np.asarray(cov)
    dim = cov.shape[0]
    if n_mock <= dim + 2:
        # Fallback: return pseudo-inverse without correction
        return np.linalg.pinv(cov)
    correction = (n_mock - dim - 2) / (n_mock - 1)
    return correction * np.linalg.pinv(cov)


def block_diagonal(blocks: List[np.ndarray]) -> np.ndarray:
    """Create a block-diagonal matrix from a list of square blocks."""
    sizes = [b.shape[0] for b in blocks if b.size > 0]
    total = sum(sizes)
    cov = np.zeros((total, total))
    offset = 0
    for b in blocks:
        n = b.shape[0]
        cov[offset:offset + n, offset:offset + n] = b
        offset += n
    return cov


# -----------------------------------------------------------------------------
# Data vector assembly
# -----------------------------------------------------------------------------

@dataclass
class DataVector:
    values: np.ndarray
    covariance: np.ndarray
    labels: List[str]

    def to_torch(self, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
        return (
            torch.tensor(self.values, dtype=torch.float32, device=device),
            torch.tensor(self.covariance, dtype=torch.float32, device=device),
        )


def assemble_data_vector(
    bao_values: Optional[np.ndarray],
    bao_sigmas: Optional[np.ndarray],
    bao_corr: Optional[np.ndarray],
    cmb_bandpowers: Optional[np.ndarray],
    cmb_cov: Optional[np.ndarray],
    other_blocks: Optional[List[Tuple[np.ndarray, np.ndarray, List[str]]]] = None,
) -> DataVector:
    """
    Assemble a joint data vector and block-diagonal covariance.

    Parameters
    ----------
    bao_values : np.ndarray or None
        BAO measurements (e.g., D_M/r_d). Shape (n_bao,)
    bao_sigmas : np.ndarray or None
        1σ uncertainties for BAO values. Shape (n_bao,)
    bao_corr : np.ndarray or None
        Published correlation matrix for BAO bins. Shape (n_bao, n_bao)
    cmb_bandpowers : np.ndarray or None
        Compressed CMB bandpowers (e.g., low/mid/high-ℓ TT/TE/EE). Shape (n_cmb,)
    cmb_cov : np.ndarray or None
        Covariance for the CMB bandpowers. Shape (n_cmb, n_cmb)
    other_blocks : list of (values, cov, labels) or None
        Additional probes (voids, galaxies, FRB, GW) already compressed.
    """
    values: List[np.ndarray] = []
    cov_blocks: List[np.ndarray] = []
    labels: List[str] = []

    if bao_values is not None and bao_sigmas is not None and bao_corr is not None:
        cov_bao = build_bao_covariance(bao_sigmas, bao_corr)
        values.append(bao_values)
        cov_blocks.append(cov_bao)
        labels.extend([f"BAO_{i}" for i in range(len(bao_values))])

    if cmb_bandpowers is not None and cmb_cov is not None:
        values.append(cmb_bandpowers)
        cov_blocks.append(cmb_cov)
        labels.extend([f"CMB_{i}" for i in range(len(cmb_bandpowers))])

    if other_blocks:
        for vec, cov, lbs in other_blocks:
            values.append(vec)
            cov_blocks.append(cov)
            labels.extend(lbs)

    if not values:
        raise ValueError("No data provided to assemble_data_vector.")

    joint_values = np.concatenate(values)
    joint_cov = block_diagonal(cov_blocks)
    return DataVector(values=joint_values, covariance=joint_cov, labels=labels)


# -----------------------------------------------------------------------------
# Model predictions (placeholder hooks)
# -----------------------------------------------------------------------------

def _e_z(omega_m: float, omega_lambda: float, z: float) -> float:
    """Dimensionless expansion rate E(z) for flat background."""
    return math.sqrt(omega_m * (1 + z) ** 3 + omega_lambda)


def _comoving_distance_mpc(omega_m: float, omega_lambda: float, h0: float, z: float, n_steps: int = 256) -> float:
    """
    Comoving distance D_C(z) = c ∫_0^z dz'/H(z') in Mpc,
    using Simpson integration with n_steps intervals.
    """
    if z <= 0:
        return 0.0
    zs = np.linspace(0, z, n_steps + 1)
    ez = np.sqrt(omega_m * (1 + zs) ** 3 + omega_lambda)
    integrand = 1.0 / ez
    dz = z / n_steps
    # Simpson's rule
    s = integrand[0] + integrand[-1] + 4 * integrand[1:-1:2].sum() + 2 * integrand[2:-2:2].sum()
    dc = HLCDM_PARAMS.C / h0 * dz / 3.0 * s  # in meters
    return dc / 3.086e22  # to Mpc


def bao_model_prediction(z: np.ndarray, params: Dict[str, float]) -> np.ndarray:
    """
    Physically motivated BAO prediction: D_M(z)/r_s with an enhanced sound horizon.

    r_s is allowed to differ from the ΛCDM fiducial (e.g., 150.71 Mpc from the
    coherent acoustic enhancement in docs/bao_resolution_qit.tex).
    """
    h0 = params.get("H0", HLCDM_PARAMS.H0)
    omega_m = params.get("omega_m", HLCDM_PARAMS.OMEGA_M)
    omega_lambda = params.get("omega_lambda", HLCDM_PARAMS.OMEGA_LAMBDA)
    # Allow an enhanced sound horizon; default to HLCDM_PARAMS.RS_BOSS if not provided.
    rs = params.get("r_s", 150.71)  # Mpc (enhanced value from QIT BAO resolution)

    dm_over_rs = []
    for zi in z:
        dc = _comoving_distance_mpc(omega_m, omega_lambda, h0, zi)
        dm = dc  # flat universe: D_M = D_C
        dm_over_rs.append(dm / rs)

    return np.array(dm_over_rs)


# -----------------------------------------------------------------------------
# HMC Sampler (lightweight, MPS-aware)
# -----------------------------------------------------------------------------

def gaussian_log_likelihood_torch(
    data_vec: torch.Tensor,
    model_vec: torch.Tensor,
    inv_cov: torch.Tensor,
) -> torch.Tensor:
    """
    Gaussian log-likelihood (up to additive constant): -0.5 * Δ^T C^{-1} Δ
    """
    delta = data_vec - model_vec
    # If inv_cov is diagonal vector, allow broadcasting
    if inv_cov.dim() == 1:
        quad = (delta ** 2 * inv_cov).sum()
    else:
        quad = torch.matmul(delta, torch.matmul(inv_cov, delta))
    return -0.5 * quad


def hmc_sample(
    log_prob_fn,
    initial_theta: torch.Tensor,
    step_size: float = 0.01,
    n_leapfrog: int = 10,
    num_samples: int = 1000,
    burn_in: int = 200,
    device: Optional[torch.device] = None,
):
    """
    Minimal HMC sampler. Returns samples and acceptance stats.
    This is intentionally lightweight; for production use consider Pyro/NumPyro.
    """
    theta = initial_theta.clone().detach()
    theta.requires_grad_(True)
    dim = theta.shape[-1]
    samples = []
    accepted = 0

    for n in range(num_samples + burn_in):
        # Draw momentum
        p = torch.randn_like(theta)
        current_theta = theta.clone()
        current_p = p.clone()

        # Compute current log prob
        lp = log_prob_fn(theta)
        grad = torch.autograd.grad(lp, theta, create_graph=False)[0]

        # Leapfrog
        theta_proposed = theta.clone()
        p = p + 0.5 * step_size * grad
        for _ in range(n_leapfrog):
            theta_proposed = theta_proposed + step_size * p
            theta_proposed.requires_grad_(True)
            lp_prop = log_prob_fn(theta_proposed)
            grad_prop = torch.autograd.grad(lp_prop, theta_proposed, create_graph=False)[0]
            p = p + step_size * grad_prop
        # Final half step
        p = p + 0.5 * step_size * grad_prop

        # Metropolis acceptance
        lp_new = log_prob_fn(theta_proposed)
        kin_current = 0.5 * torch.sum(current_p ** 2)
        kin_new = 0.5 * torch.sum(p ** 2)
        log_accept = lp_new - lp - (kin_new - kin_current)
        if torch.log(torch.rand(1, device=theta.device)) < log_accept:
            theta = theta_proposed.detach()
            theta.requires_grad_(True)
            accepted += 1
        else:
            theta = current_theta
            theta.requires_grad_(True)

        if n >= burn_in:
            samples.append(theta.detach().cpu().numpy())

    acc_rate = accepted / float(num_samples + burn_in)
    return np.array(samples), acc_rate


# -----------------------------------------------------------------------------
# High-level runner
# -----------------------------------------------------------------------------

@dataclass
class MCMCRunConfig:
    step_size: float = 0.01
    n_leapfrog: int = 10
    num_samples: int = 1000
    burn_in: int = 200
    device: str = "auto"  # "mps", "cuda", "cpu"


class MCMCRunner:
    """
    Minimal MCMC driver that builds the likelihood and launches HMC on MPS/CUDA/CPU.
    """

    def __init__(self, run_config: Optional[MCMCRunConfig] = None):
        self.config = run_config or MCMCRunConfig()
        self.device = self._select_device(self.config.device)

    def _select_device(self, pref: str) -> torch.device:
        if pref == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if pref == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if pref == "auto":
            if torch.backends.mps.is_available():
                return torch.device("mps")
            if torch.cuda.is_available():
                return torch.device("cuda")
        return torch.device("cpu")

    def run(
        self,
        data_vector: DataVector,
        model_fn,
        params_init: Dict[str, float],
        param_order: List[str],
        prior_logprob=None,
    ) -> Dict[str, Any]:
        """
        Run HMC on the provided data vector and model function.

        model_fn: callable(params_dict) -> np.ndarray matching data_vector.values shape
        params_init: dict of initial parameter values
        param_order: list defining the order of parameters in the vector
        prior_logprob: optional callable(theta_vector) -> log prior
        """
        # Prepare data on device
        d_t, cov_t = data_vector.to_torch(self.device)
        inv_cov = torch.linalg.pinv(cov_t)

        theta0 = torch.tensor([params_init[p] for p in param_order], dtype=torch.float32, device=self.device)

        def log_prob(theta_vec: torch.Tensor) -> torch.Tensor:
            params = {p: float(theta_vec[i]) for i, p in enumerate(param_order)}
            model_np = model_fn(params)
            model_t = torch.tensor(model_np, dtype=torch.float32, device=self.device)
            lp = gaussian_log_likelihood_torch(d_t, model_t, inv_cov)
            if prior_logprob is not None:
                lp = lp + prior_logprob(theta_vec)
            return lp

        samples, acc_rate = hmc_sample(
            log_prob_fn=log_prob,
            initial_theta=theta0,
            step_size=self.config.step_size,
            n_leapfrog=self.config.n_leapfrog,
            num_samples=self.config.num_samples,
            burn_in=self.config.burn_in,
            device=self.device,
        )

        return {
            "samples": samples,
            "acceptance_rate": acc_rate,
            "param_order": param_order,
            "device": str(self.device),
        }


