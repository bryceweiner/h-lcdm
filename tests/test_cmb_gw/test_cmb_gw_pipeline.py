"""
Integration tests for CMB-GW pipeline.
"""

import pytest
from pathlib import Path
from pipeline.cmb_gw import CMBGWPipeline


class TestCMBGWPipeline:
    """Test CMB-GW pipeline integration."""
    
    @pytest.fixture
    def pipeline(self, temp_output_dir):
        """Create CMB-GW pipeline instance."""
        return CMBGWPipeline(str(temp_output_dir))
    
    def test_pipeline_initialization(self, pipeline):
        """Test pipeline initialization."""
        assert pipeline.name == "cmb_gw"
        assert hasattr(pipeline, 'data_loader')
        assert pipeline.base_output_dir.exists()
    
    def test_pipeline_run_structure(self, pipeline):
        """Test that pipeline.run() returns expected structure."""
        # Use minimal context to avoid long execution
        context = {
            'bao_datasets': ['boss_dr12'],  # Single dataset for speed
            'void_surveys': ['sdss_dr7_douglass']  # Single survey
        }
        
        results = pipeline.run(context)
        
        # Check that all five tests are present
        assert 'sound_horizon' in results
        assert 'voids' in results
        assert 'sirens' in results
        assert 'peaks' in results
        assert 'coherence' in results
        assert 'joint_consistency' in results
        assert 'verdict' in results
    
    def test_validation_basic(self, pipeline):
        """Test basic validation."""
        # Run pipeline first
        pipeline.results = pipeline.run({'bao_datasets': ['boss_dr12']})
        
        validation = pipeline.validate()
        
        assert 'null_hypothesis_test' in validation
        assert 'consistency_test' in validation
    
    def test_validation_extended(self, pipeline):
        """Test extended validation."""
        validation = pipeline.validate_extended()
        
        assert 'extended_validation' in validation
    
    def test_results_structure(self, pipeline):
        """Test that results have expected schema."""
        results = pipeline.run({'bao_datasets': ['boss_dr12']})
        
        # Check TEST 1 structure
        if 'error' not in results.get('sound_horizon', {}):
            test1 = results['sound_horizon']
            assert 'r_s_observed' in test1 or 'error' in test1
            assert 'beta_fit' in test1 or 'error' in test1
        
        # Check TEST 2 (voids) structure - should use literature calibration
        if 'error' not in results.get('voids', {}):
            test2 = results['voids']
            assert 'beta_fit' in test2 or 'error' in test2
            assert 'beta_err' in test2 or 'error' in test2
            assert 'methodology' in test2 or 'error' in test2
            if 'methodology' in test2:
                assert test2['methodology'] == 'LITERATURE_CALIBRATED'
            assert 'include_in_joint' in test2 or 'error' in test2
            if 'include_in_joint' in test2:
                assert test2['include_in_joint'] is True  # Should be included (rigorous)
            # Verify no qualitative caveats
            assert 'caveat' not in test2 or test2.get('caveat') != 'QUALITATIVE_ONLY'
            # Verify citations present
            assert 'citations' in test2 or 'error' in test2
        
        # Check joint consistency structure
        if 'error' not in results.get('joint_consistency', {}):
            joint = results['joint_consistency']
            assert 'beta_combined' in joint or 'error' in joint
        
        # Check verdict structure
        if 'error' not in results.get('verdict', {}):
            verdict = results['verdict']
            assert 'verdict' in verdict
            assert verdict['verdict'] in ['STRONG_POSITIVE', 'TENTATIVE_POSITIVE', 'NULL']

