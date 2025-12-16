"""
Unit tests for joint analysis modules.
"""

import numpy as np
import pytest
from pipeline.cmb_gw.joint.consistency import joint_consistency_check
from pipeline.cmb_gw.joint.verdict import final_verdict


class TestConsistencyCheck:
    """Test parameter consistency checking."""
    
    def test_consistent_betas(self):
        """Test consistency check with consistent β values."""
        test_results = {
            'sound_horizon': {'beta_fit': 0.20, 'beta_err': 0.05},
            'sirens': {'beta_fit': 0.18, 'beta_err': 0.06},
            'peaks': {'beta_fit': 0.22, 'beta_err': 0.08}
        }
        
        result = joint_consistency_check(test_results)
        
        assert 'beta_combined' in result
        assert 'beta_combined_err' in result
        assert 'consistent' in result
        assert result['beta_combined'] > 0
        assert result['beta_combined_err'] > 0
    
    def test_inconsistent_betas(self):
        """Test consistency check with inconsistent β values."""
        test_results = {
            'sound_horizon': {'beta_fit': 0.20, 'beta_err': 0.01},
            'sirens': {'beta_fit': -0.10, 'beta_err': 0.01},  # Very different
            'peaks': {'beta_fit': 0.25, 'beta_err': 0.01}
        }
        
        result = joint_consistency_check(test_results)
        
        # Should detect inconsistency
        assert result['p_value'] < 0.05  # Low p-value indicates inconsistency
    
    def test_empty_results(self):
        """Test with no valid results."""
        test_results = {}
        result = joint_consistency_check(test_results)
        
        assert np.isnan(result['beta_combined'])
        assert not result['consistent']


class TestVerdict:
    """Test final verdict determination."""
    
    def test_strong_positive_verdict(self):
        """Test STRONG_POSITIVE verdict criteria."""
        joint_results = {
            'beta_combined': 0.20,
            'beta_combined_err': 0.05,  # 4σ from zero
            'consistent': True,
            'p_value': 0.1
        }
        
        individual_results = {
            'sound_horizon': {'r_s_observed': 150.5, 'r_s_lcdm': 147.0, 'delta_chi2': 5.0},
            'voids': {'R_v_ratio': 1.06, 'delta_chi2': 4.5},
            'sirens': {'delta_chi2': 6.0},
            'peaks': {'delta_chi2': 5.5},
            'coherence': {'enhancement_ratio': 3.0}
        }
        
        verdict = final_verdict(joint_results, individual_results)
        
        assert verdict['verdict'] == 'STRONG_POSITIVE'
        assert verdict['n_criteria_met'] >= 5
    
    def test_null_verdict(self):
        """Test NULL verdict (β consistent with zero)."""
        joint_results = {
            'beta_combined': 0.02,
            'beta_combined_err': 0.05,  # < 2σ from zero
            'consistent': True,
            'p_value': 0.1
        }
        
        individual_results = {
            'sound_horizon': {'r_s_observed': 147.2, 'r_s_lcdm': 147.0, 'delta_chi2': 1.0},
            'voids': {'R_v_ratio': 1.01},
            'sirens': {'delta_chi2': 0.5},
            'peaks': {'delta_chi2': 1.0},
            'coherence': {'enhancement_ratio': 1.1}
        }
        
        verdict = final_verdict(joint_results, individual_results)
        
        assert verdict['verdict'] == 'NULL'
        assert verdict['n_criteria_met'] < 3

