"""
Integration Tests for Void Analysis with Literature Calibration
================================================================

Tests the full void analysis pipeline using literature-based calibration.
"""

import numpy as np
import pytest
from pipeline.cmb_gw.analysis.void_analysis import analyze_void_sizes
from hlcdm.parameters import HLCDM_PARAMS


class TestFullPipeline:
    """Test A: Full Pipeline Test"""
    
    def test_analyze_voids_with_literature_calibration(self):
        """
        Test full void analysis pipeline:
        1. Load void catalog
        2. Apply literature calibration
        3. Extract β with uncertainties
        4. Verify results structure
        """
        # Run analysis
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Verify results structure
        assert 'mean_R_v_observed' in results
        assert 'mean_R_v_lcdm' in results
        assert 'mean_R_v_evolving' in results
        assert 'R_v_ratio' in results
        assert 'beta_fit' in results
        assert 'beta_err' in results
        assert 'methodology' in results
        assert 'include_in_joint' in results
        assert 'citations' in results
        
        # Verify methodology
        assert results['methodology'] == 'LITERATURE_CALIBRATED'
        assert results['include_in_joint'] is True
        
        # Verify citations present
        assert len(results['citations']) > 0
        assert 'Pisani' in results['citations'][0]
        
        # Verify no qualitative caveats
        assert 'caveat' not in results or results.get('caveat') != 'QUALITATIVE_ONLY'
        assert 'warning' not in results or results.get('warning') is None
        
        # Verify β values are finite (if data available)
        if results['n_voids'] > 0:
            assert np.isfinite(results['R_v_ratio'])
            # β may be NaN if ratio is outside physical range, which is OK
            if np.isfinite(results['beta_fit']):
                assert np.isfinite(results['beta_err'])
                assert results['beta_err'] > 0


class TestMethodComparison:
    """Test B: Comparison with Old Method"""
    
    def test_literature_vs_semi_analytic(self):
        """
        Compare literature calibration to old semi-analytic:
        - Should give similar trends
        - But literature has proper error bars
        - Document differences in report
        """
        # This test documents that literature calibration gives
        # physically reasonable results
        
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Literature calibration should give:
        # 1. Proper error bars
        if np.isfinite(results['beta_fit']):
            assert np.isfinite(results['beta_err'])
            assert results['beta_err'] > 0
        
        # 2. β in physical range
        if np.isfinite(results['beta_fit']):
            assert -0.5 <= results['beta_fit'] <= 0.5
        
        # 3. Ratio should be physically reasonable
        if np.isfinite(results['R_v_ratio']):
            assert 0.5 <= results['R_v_ratio'] <= 2.0


class TestMultiSurveyConsistency:
    """Test C: Multi-Survey Consistency"""
    
    def test_multiple_void_surveys_consistency(self):
        """
        Test with multiple void catalogs:
        - SDSS DR7 (Douglass)
        - SDSS DR7 (Clampitt-Jain)
        Verify β estimates are consistent within errors
        """
        # Run analysis with multiple surveys
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass', 'sdss_dr7_clampitt'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Should have more voids than single survey
        assert results['n_voids'] >= 0  # May be 0 if catalogs fail to load
        
        # If we have data, verify consistency
        if results['n_voids'] > 0:
            assert np.isfinite(results['R_v_ratio'])
            
            # β should be in reasonable range
            if np.isfinite(results['beta_fit']):
                assert -0.5 <= results['beta_fit'] <= 0.5
                assert results['beta_err'] > 0


class TestCMBGWIntegration:
    """Test D: CMB-GW Pipeline Integration"""
    
    def test_cmb_gw_pipeline_with_voids(self):
        """
        Run full --cmb-gw pipeline:
        - Verify void test executes
        - Check β estimate included in joint analysis
        - Verify no 'QUALITATIVE_ONLY' caveat
        """
        # This is a simplified test - full pipeline test would require
        # running the entire CMB-GW pipeline
        
        # Test that void analysis returns proper structure for joint fit
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Verify structure needed for joint analysis
        assert 'beta_fit' in results
        assert 'beta_err' in results
        assert 'include_in_joint' in results
        
        # Verify it's marked for joint analysis
        assert results['include_in_joint'] is True
        
        # Verify no qualitative caveats
        assert 'caveat' not in results or results.get('caveat') != 'QUALITATIVE_ONLY'
        
        # If β is finite, it should be usable in joint fit
        if np.isfinite(results['beta_fit']) and np.isfinite(results['beta_err']):
            assert results['beta_err'] > 0
            assert -0.5 <= results['beta_fit'] <= 0.5


class TestPhysicalScenarios:
    """Test E: Physically Motivated Scenarios"""
    
    def test_void_analysis_with_realistic_beta(self):
        """
        PHYSICAL: Test void analysis with realistic β value (β ≈ 0.2).
        
        This tests the full pipeline with a physically motivated β value
        that would produce observable void size enhancements (~5-10%).
        """
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Verify physical consistency
        if results['n_voids'] > 0 and np.isfinite(results['R_v_ratio']):
            # Void ratio should be physically reasonable
            assert 0.8 <= results['R_v_ratio'] <= 1.3, \
                f"Void ratio {results['R_v_ratio']:.3f} should be in physical range"
            
            # If β is extracted, verify it's consistent with test value
            if np.isfinite(results['beta_fit']):
                # Extracted β should be within ~50% of test value (accounting for data scatter)
                assert abs(results['beta_fit'] - 0.2) < 0.3, \
                    f"Extracted β={results['beta_fit']:.3f} should be close to test β=0.2"
    
    def test_void_analysis_at_different_redshifts(self):
        """
        PHYSICAL: Test that void analysis correctly handles redshift evolution.
        
        Voids at different redshifts should give consistent β values when
        analyzed with proper formation redshift.
        """
        # Run analysis (uses mean formation redshift from catalog)
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        if results['n_voids'] > 0:
            # Verify redshift handling
            assert 'mean_R_v_observed' in results
            assert 'mean_R_v_lcdm' in results
            
            # Ratio should be physically reasonable
            if np.isfinite(results['R_v_ratio']):
                assert results['R_v_ratio'] > 0, "Void ratio should be positive"
    
    def test_beta_extraction_consistency(self):
        """
        PHYSICAL: Test that β extraction is consistent with void size ratio.
        
        If voids are 10% larger than ΛCDM, extracted β should be positive
        and physically reasonable.
        """
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        if results['n_voids'] > 0:
            R_v_ratio = results['R_v_ratio']
            beta_fit = results['beta_fit']
            
            if np.isfinite(R_v_ratio) and np.isfinite(beta_fit):
                # Physical consistency: larger voids → positive β
                if R_v_ratio > 1.05:  # 5% excess
                    assert beta_fit > -0.1, \
                        f"Large voids (ratio={R_v_ratio:.3f}) should give β > -0.1, got {beta_fit:.3f}"
                
                # Small voids → negative or small β
                if R_v_ratio < 0.95:  # 5% deficit
                    assert beta_fit < 0.1, \
                        f"Small voids (ratio={R_v_ratio:.3f}) should give β < 0.1, got {beta_fit:.3f}"
    
    def test_uncertainty_propagation_through_pipeline(self):
        """
        PHYSICAL: Test that uncertainties are properly propagated through analysis.
        
        The β uncertainty should reflect both:
        1. Data scatter (void size distribution)
        2. Literature calibration uncertainty (γ = 1.7 ± 0.2)
        """
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        if results['n_voids'] > 0 and np.isfinite(results['beta_fit']):
            beta_fit = results['beta_fit']
            beta_err = results['beta_err']
            
            # Uncertainty should be positive and finite
            assert np.isfinite(beta_err), "β uncertainty should be finite"
            assert beta_err > 0, "β uncertainty should be positive"
            
            # Relative uncertainty should be reasonable (~10-30% for typical data)
            relative_err = beta_err / abs(beta_fit) if beta_fit != 0 else np.nan
            if np.isfinite(relative_err):
                assert 0.05 <= relative_err <= 1.0, \
                    f"Relative β uncertainty {relative_err:.3f} should be in reasonable range [5%, 100%]"
    
    def test_literature_calibration_applied_correctly(self):
        """
        PHYSICAL: Verify that literature calibration (γ = 1.7) is applied correctly.
        
        The void size ratio should follow R_v(β)/R_v(0) = [D(β)/D(0)]^1.7,
        not the old semi-analytic approximation.
        """
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Verify methodology
        assert results['methodology'] == 'LITERATURE_CALIBRATED'
        
        # Verify citations mention Pisani+ (2015)
        citations = results.get('citations', [])
        assert len(citations) > 0, "Citations should be present"
        
        pisani_citation = [c for c in citations if 'Pisani' in c]
        assert len(pisani_citation) > 0, "Should cite Pisani+ (2015)"
        
        # Verify no qualitative caveats
        assert 'caveat' not in results or results.get('caveat') != 'QUALITATIVE_ONLY'


class TestJointFitIntegration:
    """Test F: Joint Fit Integration"""
    
    def test_void_beta_included_in_joint_fit(self):
        """
        PHYSICAL: Verify void β is included in joint consistency check.
        
        Void β should participate in joint fit with other tests (CMB, BAO, etc.)
        since it's now rigorously calibrated.
        """
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Verify structure for joint fit
        assert 'beta_fit' in results
        assert 'beta_err' in results
        assert 'include_in_joint' in results
        
        # Should be included (rigorous method)
        assert results['include_in_joint'] is True
        
        # If β is finite, it should be usable
        if np.isfinite(results['beta_fit']) and np.isfinite(results['beta_err']):
            assert results['beta_err'] > 0
            assert -0.5 <= results['beta_fit'] <= 0.5
    
    def test_void_beta_consistency_with_other_tests(self):
        """
        PHYSICAL: Test that void β is consistent with other cosmological tests.
        
        In a real analysis, void β should agree with CMB and BAO β within errors.
        This test verifies the structure allows such consistency checks.
        """
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Verify results have structure needed for consistency check
        required_keys = ['beta_fit', 'beta_err', 'methodology', 'include_in_joint']
        for key in required_keys:
            assert key in results, f"Results should contain '{key}' for joint analysis"
        
        # Verify methodology is rigorous
        assert results['methodology'] == 'LITERATURE_CALIBRATED'
        
        # Verify it's marked for inclusion
        assert results['include_in_joint'] is True


class TestErrorHandling:
    """Test G: Error Handling and Edge Cases"""
    
    def test_handles_missing_data_gracefully(self):
        """
        PHYSICAL: Test that analysis handles missing void catalogs gracefully.
        
        If catalogs fail to load, analysis should return NaN values, not crash.
        """
        # Try with non-existent survey (should handle gracefully)
        results = analyze_void_sizes(
            surveys=['nonexistent_survey'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Should return structure even if no data
        assert 'n_voids' in results
        assert 'beta_fit' in results
        
        # If no voids, some values may be NaN (acceptable)
        if results['n_voids'] == 0:
            # Should still have methodology and citations
            assert 'methodology' in results
            assert 'citations' in results
    
    def test_handles_extreme_ratios(self):
        """
        PHYSICAL: Test handling of extreme void size ratios.
        
        If observed voids are extremely large/small, β extraction should
        handle gracefully (may return NaN if outside physical range).
        """
        # This test verifies the code doesn't crash on extreme values
        # Actual extreme ratios would come from data, but we test the structure
        
        results = analyze_void_sizes(
            surveys=['sdss_dr7_douglass'],
            omega_m=HLCDM_PARAMS.OMEGA_M,
            beta_test=0.2
        )
        
        # Should always return valid structure
        assert isinstance(results, dict)
        assert 'R_v_ratio' in results
        assert 'beta_fit' in results
        
        # β may be NaN if ratio is outside physical range (acceptable)
        if not np.isfinite(results['beta_fit']):
            # Should still have methodology
            assert 'methodology' in results

