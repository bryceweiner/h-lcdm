"""
Test configuration and fixtures for H-ΛCDM analysis framework.

Provides common test fixtures and utilities for unit testing.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import json

from pipeline.common.base_pipeline import AnalysisPipeline


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def mock_pipeline(temp_output_dir):
    """Create a mock pipeline instance for testing."""
    class MockPipeline(AnalysisPipeline):
        def __init__(self):
            super().__init__("mock_pipeline", str(temp_output_dir))

        def run(self, context=None):
            return {"test": "data"}

        def validate(self, context=None):
            return {"validation": "passed"}

        def validate_extended(self, context=None):
            return {"extended_validation": "passed"}

    return MockPipeline()


@pytest.fixture
def sample_p_values():
    """Sample p-values for multiple testing correction tests."""
    np.random.seed(42)
    return np.array([0.01, 0.03, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50])


@pytest.fixture
def sample_covariance_matrix():
    """Sample covariance matrix for testing."""
    np.random.seed(42)
    n = 5
    # Create a positive definite covariance matrix
    A = np.random.randn(n, n)
    cov = A @ A.T + np.eye(n) * 0.1
    return cov


@pytest.fixture
def sample_mcmc_samples():
    """Sample MCMC chains for convergence testing."""
    np.random.seed(42)
    n_samples = 1000
    n_chains = 3

    # Simulate converging chains
    true_value = 1.0
    samples = {}
    for param in ['param1', 'param2']:
        chain_samples = []
        for chain in range(n_chains):
            # Add chain-specific offset that decreases (convergence)
            offset = 0.5 * np.exp(-chain * 0.5)
            chain_data = np.random.normal(true_value + offset, 0.1, n_samples)
            chain_samples.extend(chain_data)
        samples[param] = chain_samples

    return samples


@pytest.fixture
def sample_systematic_components():
    """Sample systematic error components for testing."""
    return {
        'calibration': 0.02,
        'foreground': 0.03,
        'beam': 0.01,
        'atmosphere': 0.015
    }


@pytest.fixture
def sample_bao_data():
    """Sample BAO data for testing."""
    return [
        {'z': 0.38, 'value': 10.27, 'error': 0.15},
        {'z': 0.51, 'value': 13.37, 'error': 0.15},
        {'z': 0.61, 'value': 15.23, 'error': 0.17}
    ]


@pytest.fixture
def sample_cmb_data():
    """Sample CMB data for testing."""
    return {
        'act_dr6': {
            'C_ell': np.random.lognormal(0, 0.5, 100),
            'ell': np.arange(30, 130)
        },
        'planck_2018': {
            'C_ell': np.random.lognormal(0, 0.5, 100),
            'ell': np.arange(30, 130)
        }
    }


@pytest.fixture
def sample_void_catalog():
    """Sample void catalog for testing."""
    np.random.seed(42)
    n_voids = 50
    return pd.DataFrame({
        'radius_Mpc': np.random.lognormal(1.5, 0.3, n_voids),
        'density_contrast': np.random.normal(-0.8, 0.2, n_voids),
        'volume_Mpc3': np.random.lognormal(3.0, 0.5, n_voids),
        'x': np.random.uniform(0, 1000, n_voids),
        'y': np.random.uniform(0, 1000, n_voids),
        'z': np.random.uniform(0, 1000, n_voids)
    })


@pytest.fixture
def sample_test_results():
    """Sample test results for HLCDM meta-analysis."""
    return {
        'jwst': {'evidence': 2.5, 'p_value': 0.012},
        'lyman_alpha': {'evidence': 1.8, 'p_value': 0.035},
        'frb': {'evidence': 3.2, 'p_value': 0.001},
        'e8_ml': {'evidence': 1.2, 'p_value': 0.230},
        'e8_chiral': {'evidence': 2.8, 'p_value': 0.005},
        'temporal_cascade': {'evidence': 1.5, 'p_value': 0.133}
    }
