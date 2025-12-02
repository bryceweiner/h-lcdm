"""
Bayesian Optimization Framework
===============================

Manages the Bayesian Optimization process for finding optimal twist angles.
"""

import numpy as np
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, Matern, ConstantKernel, WhiteKernel
from scipy.optimize import minimize
from typing import Tuple, List, Dict, Optional

from .acquisition import PhysicsInformedAcquisition

# Suppress sklearn convergence warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

class TMDCBayesianOptimizer:
    """
    Bayesian Optimizer for 7-layer TMDC architecture.
    """
    
    def __init__(self, bounds: List[Tuple[float, float]], random_state: int = 42):
        """
        Initialize optimizer.
        
        Args:
            bounds: List of (min, max) for each dimension
            random_state: Random seed
        """
        self.bounds = np.array(bounds)
        self.dim = len(bounds)
        self.rng = np.random.RandomState(random_state)
        
        # Define Kernel: Constant * (RBF + Matern)
        # We want a flexible kernel that can handle smooth variations (RBF) 
        # and rougher physics transitions (Matern)
        # Expanded bounds to prevent convergence warnings
        kernel = ConstantKernel(1.0, constant_value_bounds=(1e-6, 1e3)) * (
            RBF(length_scale=1.0, length_scale_bounds=(0.01, 100.0)) + 
            Matern(length_scale=1.0, length_scale_bounds=(0.01, 100.0), nu=2.5)
        ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-8, 1e-2))
        
        self.gp = GaussianProcessRegressor(
            kernel=kernel,
            alpha=1e-3, # Small thermodynamic noise floor
            normalize_y=True,
            n_restarts_optimizer=10,
            random_state=random_state
        )
        
        self.acquisition = PhysicsInformedAcquisition()
        
        self.X_train = []
        self.y_train = []
        self._y_scaled = []
        self._target_max = 1e4
        self._jitter = 1e-4
        
    def update(self, X: np.ndarray, y: float):
        """
        Update the model with new observation.
        
        Args:
            X: Parameter vector (1D array)
            y: Objective value
        """
        if not np.isfinite(y):
            raise ValueError("Non-finite objective received by GP")

        y_clipped = float(np.clip(y, 0.0, self._target_max))
        y_scaled = self._transform_target(y_clipped)

        x_noisy = X + self.rng.normal(scale=self._jitter, size=X.shape)

        self.X_train.append(x_noisy)
        self.y_train.append(y_clipped)
        self._y_scaled.append(y_scaled)

        # Refit model
        self.gp.fit(np.array(self.X_train), np.array(self._y_scaled))
        
    def suggest_next_point(self, n_restarts: int = 25) -> np.ndarray:
        """
        Propose next point to evaluate by optimizing acquisition function.
        
        Args:
            n_restarts: Number of random starting points for optimization
            
        Returns:
            Next point parameter vector
        """
        if not self.X_train:
            # Random initial point if no data
            return self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            
        # Best observed value so far (maximization)
        y_best = np.max(self.y_train)
        
        # Function to minimize (negative acquisition)
        def objective(x):
            x = x.reshape(1, -1)
            acq_value = self.acquisition.evaluate(x, self.gp, y_best)
            return -acq_value[0]
        
        best_x = None
        best_acq = float('inf')
        
        # Multi-start optimization
        for _ in range(n_restarts):
            x0 = self.rng.uniform(self.bounds[:, 0], self.bounds[:, 1])
            res = minimize(
                objective, 
                x0=x0, 
                bounds=self.bounds, 
                method='L-BFGS-B'
            )
            
            if res.fun < best_acq:
                best_acq = res.fun
                best_x = res.x
                
        return best_x

    def get_best_params(self) -> Tuple[np.ndarray, float]:
        """
        Get best parameters found so far.
        """
        if not self.y_train:
            return None, None
        
        idx = np.argmax(self.y_train)
        return self.X_train[idx], self.y_train[idx]

    @staticmethod
    def _transform_target(y: float) -> float:
        return float(np.log1p(y))


def setup_bayesian_optimization(dim: int,
                                angle_bounds: Tuple[float, float] = (0.1, 4.0),
                                random_state: int = 42) -> TMDCBayesianOptimizer:
    """
    Configure Bayesian optimization for an N-dimensional interlayer twist space.
    
    Args:
        dim: Number of interlayer twists (n_layers - 1).
        angle_bounds: (min, max) bounds for each Δθ component (degrees).
        random_state: Seed for reproducibility.
        
    Returns:
        Configured optimization instance.
    """
    bounds = [angle_bounds for _ in range(dim)]
    optimizer = TMDCBayesianOptimizer(bounds=bounds, random_state=random_state)
    return optimizer

