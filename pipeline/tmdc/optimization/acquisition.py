"""
Physics-Informed Acquisition Function
=====================================

Custom acquisition function with QTEP physics knowledge for Bayesian Optimization.
"""

import numpy as np
from scipy.stats import norm
from typing import Tuple, Union

from hlcdm.parameters import HLCDM_PARAMS

class PhysicsInformedAcquisition:
    """
    Implements physics-informed acquisition function.
    Combines Expected Improvement (EI) with physical priors.
    """
    
    def __init__(self, xi: float = 0.01, resonance_weight: float = 0.5):
        """
        Initialize acquisition function.
        
        Args:
            xi: Exploration-exploitation trade-off parameter for EI
            resonance_weight: Weight for the resonance prior bonus
        """
        self.xi = xi
        self.resonance_weight = resonance_weight
        
    def _expected_improvement(self, X: np.ndarray, mean: np.ndarray, std: np.ndarray, 
                              y_best: float) -> np.ndarray:
        """
        Calculate standard Expected Improvement.
        """
        # Avoid division by zero
        std = np.maximum(std, 1e-9)
        
        imp = mean - y_best - self.xi
        Z = imp / std
        ei = imp * norm.cdf(Z) + std * norm.pdf(Z)
        return ei

    def _gamma_resonance_prior(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate bonus for γ-resonant configurations.
        
        Based on WSe2 physics (docs/wse_2.md):
        - Primary magic angle ~1.2 degrees
        - Broad flat-band window 1.0 - 3.0 degrees
        """
        # Primary target
        target_angles = [1.2]
        
        # Secondary targets to guide optimizer into the flat-band window
        # Spaced to cover 1-3 degree range
        secondary_targets = [1.8, 2.4, 3.0]
        
        sigma_primary = 0.2
        sigma_secondary = 0.4
        
        # Calculate proximity of interlayer twist angles to targets
        # X shape: (n_samples, n_dims) where n_dims=6 (interlayer twists)
        
        n_samples, n_dims = X.shape
        prior_score = np.zeros(n_samples)
        
        for i in range(n_samples):
            # X[i] contains relative twist angles (deltas) directly
            deltas = X[i]
            
            # Sum scores across all interfaces
            total_score = 0.0
            for delta in deltas:
                val = abs(delta)
                
                # Primary match
                primary_match = max([np.exp(-0.5 * ((val - t) / sigma_primary) ** 2) for t in target_angles])
                
                # Secondary match (weighted lower)
                secondary_match = max([np.exp(-0.5 * ((val - t) / sigma_secondary) ** 2) for t in secondary_targets])
                
                total_score += max(primary_match, 0.5 * secondary_match)
            
            # Average match quality across all 6 interfaces
            prior_score[i] = total_score / len(deltas)
            
        return prior_score

    def _bounds_penalty(self, X: np.ndarray) -> np.ndarray:
        """
        Penalize configurations near parameter bounds.
        """
        # Bounds from bayesian_opt.py: [0.1, 4.0]
        lower_bound = 0.1
        upper_bound = 4.0
        # Margin where penalty applies (0.1 degrees)
        margin = 0.1
        
        penalty = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            for angle in X[i]:
                # Penalty for being within margin of bounds
                # Quadratic penalty for smoothness
                if angle < lower_bound + margin:
                    # Normalized distance from safe zone
                    dist = (lower_bound + margin - angle) / margin
                    penalty[i] -= 0.5 * (dist ** 2)
                elif angle > upper_bound - margin:
                    dist = (angle - (upper_bound - margin)) / margin
                    penalty[i] -= 0.5 * (dist ** 2)
        
        return penalty

    def evaluate(self, X: np.ndarray, gp_model, y_best: float) -> np.ndarray:
        """
        Compute acquisition value for candidate points X.
        
        Args:
            X: Candidate points (n_samples, n_dims)
            gp_model: Trained Gaussian Process model
            y_best: Best observed value so far
            
        Returns:
            Acquisition values
        """
        mean, std = gp_model.predict(X, return_std=True)
        
        # Expected Improvement
        ei = self._expected_improvement(X, mean, std, y_best)
        
        # Physics Prior
        prior = self._gamma_resonance_prior(X)
        
        # Bounds Penalty (keep away from non-physical edge solutions)
        bounds_penalty = self._bounds_penalty(X)
        
        # Combine
        # We want to maximize this value
        # Add penalty (which is negative)
        return ei + self.resonance_weight * prior + bounds_penalty

