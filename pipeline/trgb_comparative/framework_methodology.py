"""
Framework forward prediction — prediction only, no photometry, no MCMC.

The framework does not "observe" — it predicts. Given a d_local value and
H_CMB with uncertainty, this module evaluates the holographic projection
formula and returns a Monte Carlo posterior over the framework-predicted
local H₀.

There is no ``H0_framework_observed``. The authoritative outputs are
``H0_framework_predicted_lmc_anchor`` (Case A) and
``H0_framework_predicted_ngc4258_anchor`` (Case B).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .monte_carlo_propagation import PlanckH0Posterior, run_monte_carlo


@dataclass(frozen=True)
class FrameworkPrediction:
    """Monte Carlo summary of the framework-predicted local H₀."""

    label: str                              # e.g. "framework_predicted_ngc4258_anchor"
    H0_samples: np.ndarray
    H0_median: float
    H0_low: float                           # 16th percentile
    H0_high: float                          # 84th percentile
    breakdown_fraction: float
    breakdown_flag_any: bool
    breakdown_messages: List[str]
    inputs: Dict[str, float]

    def as_dict(self, include_samples: bool = False) -> Dict[str, object]:
        out: Dict[str, object] = {
            "label": self.label,
            "H0_median": float(self.H0_median),
            "H0_low": float(self.H0_low),
            "H0_high": float(self.H0_high),
            "breakdown_fraction": float(self.breakdown_fraction),
            "breakdown_flag_any": bool(self.breakdown_flag_any),
            "breakdown_messages": list(self.breakdown_messages),
            "inputs": dict(self.inputs),
            "n_samples": int(self.H0_samples.size),
        }
        if include_samples:
            out["H0_samples"] = self.H0_samples.tolist()
        return out


class FrameworkMethodology:
    """Forward prediction only. Evaluates the projection formula via MC."""

    def predict(
        self,
        label: str,
        d_local_mpc: float,
        sigma_d_local_mpc: float,
        H_cmb_posterior: Optional[PlanckH0Posterior] = None,
        n_samples: int = 50_000,
        seed: int = 42,
    ) -> FrameworkPrediction:
        H0_samples, per_draw = run_monte_carlo(
            d_local_mpc=d_local_mpc,
            sigma_d_local_mpc=sigma_d_local_mpc,
            H_cmb_posterior=H_cmb_posterior,
            n_samples=n_samples,
            seed=seed,
        )

        flags = np.array([r.breakdown_flag for r in per_draw], dtype=bool)
        msgs = sorted(
            {r.breakdown_message for r in per_draw if r.breakdown_message is not None}
        )

        if H_cmb_posterior is None:
            H_cmb_posterior = PlanckH0Posterior()

        return FrameworkPrediction(
            label=label,
            H0_samples=H0_samples,
            H0_median=float(np.median(H0_samples)),
            H0_low=float(np.percentile(H0_samples, 16)),
            H0_high=float(np.percentile(H0_samples, 84)),
            breakdown_fraction=float(flags.mean()),
            breakdown_flag_any=bool(flags.any()),
            breakdown_messages=msgs,
            inputs={
                "H_cmb_mean": float(H_cmb_posterior.mean),
                "H_cmb_sigma": float(H_cmb_posterior.sigma),
                "d_local_mpc": float(d_local_mpc),
                "sigma_d_local_mpc": float(sigma_d_local_mpc),
                "gamma_over_H": float(per_draw[0].gamma_over_H) if per_draw else float("nan"),
                "d_cmb_mpc": float(per_draw[0].d_cmb_mpc) if per_draw else float("nan"),
            },
        )


__all__ = ["FrameworkMethodology", "FrameworkPrediction"]
