"""
TMDC Validation Suite
=====================

Validation tests for physics engines and optimization framework.
"""

import numpy as np
import logging
from typing import Dict, Any

from pipeline.tmdc.physics.moire import calculate_moire_properties
from pipeline.tmdc.physics.tight_binding import construct_tb_hamiltonian, solve_eigenvalue_problem
from pipeline.tmdc.simulation.objective import objective_function, objective_diagnostics
from hlcdm.parameters import HLCDM_PARAMS
from pipeline.tmdc.optimization.bayesian_opt import setup_bayesian_optimization

logger = logging.getLogger(__name__)

def validate_physics_engines() -> Dict[str, Any]:
    """
    Test suite for physics simulation components.
    """
    results = {}
    
    # Test 1: Verify moiré period calculations against WSe2 literature
    # WSe2 (a=0.33 nm), theta=1.2 deg (magic angle)
    # Period approx a / theta_rad = 0.33 / (1.2 * pi / 180) = 15.75 nm
    # Using formula: a / (2 sin(theta/2))
    props = calculate_moire_properties(0, np.deg2rad(1.2))
    period = props['period']
    expected_period = 15.76
    results['moire_period_test'] = {
        'passed': abs(period - expected_period) < 0.5,
        'value': period,
        'expected': expected_period
    }
    
    # Test 2: Compare band structure with known small-angle results
    # Check if bandwidth decreases for small angles (flat bands)
    # Not implementing full comparison, just a sanity check on output shape/values
    angles = np.zeros(7) # Aligned
    angles[1] = np.deg2rad(1.2) # WSe2 Magic angle twist
    moire_data = {'0_1': calculate_moire_properties(0, angles[1])}
    H = construct_tb_hamiltonian(angles, moire_data, k_points=10)
    band_res = solve_eigenvalue_problem(H)
    results['band_structure_test'] = {
        'passed': band_res['eigenvalues'].shape == (10, 28),
        'flatness': band_res['flatness_metric']
    }
    
    # Test 3: Validate QTEP ratio calculations
    # Should equal ~2.257 (from central parameters module)
    qtep_ratio = HLCDM_PARAMS.QTEP_RATIO
    results['qtep_ratio_test'] = {
        'passed': abs(qtep_ratio - 2.257) < 0.01,
        'value': qtep_ratio
    }

    # Test 4: N-layer scaling sanity (5,7,9 layers)
    uniform_5 = objective_diagnostics(np.ones(4) * 1.2, n_layers=5)
    uniform_7 = objective_diagnostics(np.ones(6) * 1.2, n_layers=7)
    uniform_9 = objective_diagnostics(np.ones(8) * 1.2, n_layers=9)
    base_amp_5 = uniform_5['base_amplification']
    base_amp_7 = uniform_7['base_amplification']
    base_amp_9 = uniform_9['base_amplification']
    strain_5 = uniform_5['total_strain_energy']
    strain_7 = uniform_7['total_strain_energy']
    strain_9 = uniform_9['total_strain_energy']

    results['n_layer_scaling_test'] = {
        'passed': (
            base_amp_5 < base_amp_7 < base_amp_9 and
            strain_5 < strain_7 < strain_9
        ),
        'base_amplifications': {
            'n5': base_amp_5,
            'n7': base_amp_7,
            'n9': base_amp_9
        },
        'strain_energy': {
            'n5': strain_5,
            'n7': strain_7,
            'n9': strain_9
        }
    }
    
    return results

def validate_optimization() -> Dict[str, Any]:
    """
    Test Bayesian optimization convergence and physics constraints.
    """
    results = {}
    
    # Test 1: Known optimal configuration recovery
    # We can't easily test this without a ground truth function, 
    # but we can check if it respects bounds
    optimizer = setup_bayesian_optimization(dim=6)
    next_point = optimizer.suggest_next_point(n_restarts=5)
    
    # Test 2: Constraint satisfaction
    # Bounds: [0.1, 4.0]
    in_bounds = True
    for i, val in enumerate(next_point):
        if val < 0.1 or val > 4.0:
            in_bounds = False
            break
            
    results['bounds_test'] = {
        'passed': in_bounds,
        'point': next_point.tolist()
    }

    # Test 3: Multi-run diversity (brief BO restarts)
    def _mini_bo(seed: int):
        dim = 6
        mini_optimizer = setup_bayesian_optimization(dim=dim, random_state=seed)
        rng = np.random.default_rng(seed)
        # Random seeding
        for _ in range(2):
            point = rng.uniform(0.1, 4.0, dim)
            val = objective_function(point)
            mini_optimizer.update(point, val)
        # Few BO steps
        for _ in range(3):
            point = mini_optimizer.suggest_next_point(n_restarts=5)
            val = objective_function(point)
            mini_optimizer.update(point, val)
        best_val = float(np.max(mini_optimizer.y_train))
        best_idx = int(np.argmax(mini_optimizer.y_train))
        return best_val, best_idx + 1

    mini_runs = [_mini_bo(seed) for seed in range(3)]
    best_vals = np.array([val for val, _ in mini_runs])
    convergence_counts = [count for _, count in mini_runs]
    diversity_passed = (
        np.std(best_vals) > 1e-3 or len(set(convergence_counts)) > 1
    )
    results['multi_run_diversity_test'] = {
        'passed': bool(diversity_passed),
        'best_values': best_vals.tolist(),
        'convergence_counts': convergence_counts
    }
    
    return results

