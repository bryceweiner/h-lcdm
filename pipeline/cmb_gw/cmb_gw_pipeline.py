"""
CMB-GW Pipeline - Evolving Gravitational Constant Test Suite
=============================================================

Main pipeline orchestrating all five tests and joint analysis.

This implements the complete protocol from docs/cmb_gw.md:
- TEST 1: Sound horizon enhancement from BAO
- TEST 2: Void size distribution
- TEST 3: Standard siren luminosity distances
- TEST 4: CMB peak height ratios
- TEST 5: Cross-modal coherence at acoustic scale
- Joint analysis: Parameter consistency and final verdict
"""

import numpy as np
from typing import Dict, Any, Optional
from ..common.base_pipeline import AnalysisPipeline
from .analysis.bao_sound_horizon import analyze_bao_sound_horizon
from .analysis.void_analysis import analyze_void_sizes
from .analysis.siren_analysis import analyze_standard_sirens
from .analysis.peak_analysis import fit_peak_ratios_to_data, measure_peak_ratios
from .analysis.coherence_analysis import cross_modal_coherence_at_harmonics, compute_cmb_residuals
from .joint.consistency import joint_consistency_check
from .joint.verdict import final_verdict
from data.loader import DataLoader
from hlcdm.parameters import HLCDM_PARAMS


class CMBGWPipeline(AnalysisPipeline):
    """
    CMB-GW evolving G(z) analysis pipeline.
    
    Tests the evolving gravitational constant hypothesis through five
    independent observational probes with joint parameter consistency analysis.
    """
    
    def __init__(self, output_dir: str = "results"):
        """
        Initialize CMB-GW pipeline.
        
        Parameters:
        -----------
        output_dir : str
            Output directory for results
        """
        super().__init__("cmb_gw", output_dir)
        
        self.update_metadata('description', 'Evolving G(z) test suite: sound horizon, voids, sirens, CMB peaks, coherence')
        self.update_metadata('protocol', 'docs/cmb_gw.md')
        self.update_metadata('tests', ['sound_horizon', 'voids', 'sirens', 'peaks', 'coherence'])
        
        # Initialize data loader
        self.data_loader = DataLoader(log_file=self.log_file)
    
    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute all five tests and joint analysis.
        
        Parameters:
        -----------
        context : dict, optional
            Analysis parameters:
            - 'bao_datasets': List of BAO datasets (default: ['boss_dr12', 'desi', 'eboss'])
            - 'void_surveys': List of void surveys (default: ['sdss_dr7_douglass', 'sdss_dr7_clampitt'])
            - 'omega_b': Baryon density (default: 0.049)
            - 'omega_m': Matter density (default: HLCDM_PARAMS.OMEGA_M)
            - 'H0': Hubble constant in km/s/Mpc (default: from HLCDM_PARAMS)
            
        Returns:
        --------
        dict
            Complete analysis results with all five tests and joint analysis
        """
        self.log_progress("Starting CMB-GW evolving G(z) analysis...")
        
        if context is None:
            context = {}
        
        # Parse context parameters
        bao_datasets = context.get('bao_datasets', ['boss_dr12', 'desi', 'eboss'])
        void_surveys = context.get('void_surveys', ['sdss_dr7_douglass', 'sdss_dr7_clampitt'])
        omega_b = context.get('omega_b', 0.049)
        omega_m = context.get('omega_m', None)  # Will use HLCDM_PARAMS default
        H0 = context.get('H0', None)  # Will use HLCDM_PARAMS default
        
        results = {}
        
        # TEST 1: Sound Horizon Enhancement
        self.log_progress("Running TEST 1: Sound horizon enhancement from BAO...")
        try:
            test1_results = analyze_bao_sound_horizon(
                datasets=bao_datasets,
                omega_b=omega_b,
                omega_m=omega_m,
                H0=H0
            )
            results['sound_horizon'] = test1_results
            beta_fit = float(test1_results.get('beta_fit', np.nan)) if np.isfinite(test1_results.get('beta_fit', np.nan)) else np.nan
            beta_err = float(test1_results.get('beta_err', np.nan)) if np.isfinite(test1_results.get('beta_err', np.nan)) else np.nan
            self.log_progress(f"  TEST 1 complete: β = {beta_fit:.3f} ± {beta_err:.3f}")
        except Exception as e:
            self.log_progress(f"  TEST 1 failed: {e}")
            results['sound_horizon'] = {'error': str(e)}
        
        # TEST 2: Void Size Distribution
        self.log_progress("Running TEST 2: Void size distribution...")
        try:
            test2_results = analyze_void_sizes(
                surveys=void_surveys,
                omega_m=omega_m,
                use_nbody_calibration=True,  # Use N-body calibration if available
                run_calibration_if_needed=False  # Don't run calibration automatically
            )
            results['voids'] = test2_results
            
            # Log methodology used
            methodology = test2_results.get('methodology', 'UNKNOWN')
            if methodology == 'NBODY_CALIBRATED':
                self.log_progress(f"  TEST 2 complete (N-body calibrated): R_v ratio = {test2_results.get('R_v_ratio', np.nan):.3f}")
                beta_fit = float(test2_results.get('beta_fit', np.nan)) if 'beta_fit' in test2_results else np.nan
                beta_err = float(test2_results.get('beta_err', np.nan)) if np.isfinite(test2_results.get('beta_err', np.nan)) else np.nan
                if np.isfinite(beta_fit):
                    self.log_progress(f"  β (voids, rigorous) = {beta_fit:.3f} ± {beta_err:.3f}")
            else:
                self.log_progress(f"  TEST 2 complete (analytic approximation): R_v ratio = {test2_results.get('R_v_ratio', np.nan):.3f}")
                self.log_progress(f"  WARNING: N-body calibration not available. β estimate is QUALITATIVE ONLY.")
        except Exception as e:
            self.log_progress(f"  TEST 2 failed: {e}")
            results['voids'] = {'error': str(e)}
        
        # TEST 3: Standard Siren Luminosity Distances
        self.log_progress("Running TEST 3: Standard siren luminosity distances...")
        try:
            test3_results = analyze_standard_sirens(
                omega_m=omega_m,
                H0_cmb=H0
            )
            results['sirens'] = test3_results
            beta_fit = float(test3_results.get('beta_fit', np.nan)) if np.isfinite(test3_results.get('beta_fit', np.nan)) else np.nan
            beta_err = float(test3_results.get('beta_err', np.nan)) if np.isfinite(test3_results.get('beta_err', np.nan)) else np.nan
            self.log_progress(f"  TEST 3 complete: β = {beta_fit:.3f} ± {beta_err:.3f}")
        except Exception as e:
            self.log_progress(f"  TEST 3 failed: {e}")
            results['sirens'] = {'error': str(e)}
        
        # TEST 4: CMB Peak Height Ratios
        self.log_progress("Running TEST 4: CMB peak height ratios...")
        try:
            # Load CMB data and measure peaks
            planck_data = self.data_loader.load_planck_2018()
            planck_peaks = None
            
            if planck_data is None:
                self.log_progress("  TEST 4: planck_data is None")
            elif 'TT' not in planck_data:
                self.log_progress(f"  TEST 4: 'TT' not in planck_data. Available keys: {list(planck_data.keys())}")
            else:
                ell, Cl_TT, Cl_err = planck_data['TT']
                self.log_progress(f"  TEST 4: Loaded TT spectrum with {len(ell)} multipoles, ell range: {ell.min():.0f}-{ell.max():.0f}")
                self.log_progress(f"  TEST 4: Cl_TT range: {Cl_TT.min():.2e} to {Cl_TT.max():.2e}, mean: {Cl_TT.mean():.2e}")
                
                # Ensure arrays are valid
                if len(ell) > 0 and len(Cl_TT) > 0:
                    # Don't filter out negative values - they're valid in CMB data
                    valid_mask = np.isfinite(Cl_TT) & np.isfinite(ell)
                    if np.sum(valid_mask) < 10:
                        self.log_progress(f"  TEST 4: Not enough valid data points ({np.sum(valid_mask)})")
                    else:
                        ell_clean = ell[valid_mask]
                        Cl_TT_clean = Cl_TT[valid_mask]
                        
                        # Log data statistics (data-driven, no hardcoded positions)
                        self.log_progress(f"  TEST 4: Data statistics - mean: {np.mean(Cl_TT_clean):.2e}, std: {np.std(Cl_TT_clean):.2e}")
                        
                        measured = measure_peak_ratios(Cl_TT_clean, ell_clean)
                        
                        if measured is None:
                            self.log_progress("  TEST 4: measure_peak_ratios returned None")
                        elif np.isnan(measured.get('R21', np.nan)) or np.isnan(measured.get('R31', np.nan)):
                            self.log_progress(f"  TEST 4: Peak measurement failed. R21={measured.get('R21')}, R31={measured.get('R31')}")
                            if 'peak_ells' in measured and len(measured['peak_ells']) > 0:
                                self.log_progress(f"  TEST 4: Found {len(measured['peak_ells'])} peaks at ell={measured.get('peak_ells', [])}")
                            elif 'n_peaks_found' in measured:
                                self.log_progress(f"  TEST 4: Found {measured['n_peaks_found']} peaks total")
                            if 'debug_info' in measured:
                                debug = measured['debug_info']
                                prominence = debug.get('prominence_data_driven', debug.get('prominence_threshold'))
                                self.log_progress(f"  TEST 4: Debug - prominence={prominence:.2e if prominence else 'N/A'}, distance={debug.get('distance_indices', 'N/A')}")
                        else:
                            # DATA-DRIVEN error estimation from measurement uncertainties
                            from .analysis.peak_analysis import estimate_peak_ratio_uncertainty
                            if 'peak_ells' in measured and 'peak_amps' in measured and len(measured['peak_ells']) >= 3:
                                R21_err, R31_err = estimate_peak_ratio_uncertainty(
                                    Cl_TT_clean, Cl_err[valid_mask], ell_clean,
                                    measured['peak_ells'], measured['peak_amps']
                                )
                            else:
                                # Fallback: estimate from ratio of std/mean
                                frac_unc = np.std(Cl_TT_clean) / np.mean(np.abs(Cl_TT_clean)) if np.mean(np.abs(Cl_TT_clean)) > 0 else 0.1
                                R21_err = frac_unc * abs(measured['R21']) if measured['R21'] != 0 else frac_unc
                                R31_err = frac_unc * abs(measured['R31']) if measured['R31'] != 0 else frac_unc
                            
                            # Ensure valid uncertainties
                            R21_err = R21_err if np.isfinite(R21_err) and R21_err > 0 else 0.1 * abs(measured['R21'])
                            R31_err = R31_err if np.isfinite(R31_err) and R31_err > 0 else 0.1 * abs(measured['R31'])
                            
                            planck_peaks = {
                                'R21': float(measured['R21']),
                                'R21_err': float(R21_err),
                                'R31': float(measured['R31']),
                                'R31_err': float(R31_err)
                            }
                            self.log_progress(f"  TEST 4: Measured peaks - R21={planck_peaks['R21']:.3f}±{planck_peaks['R21_err']:.3f}, R31={planck_peaks['R31']:.3f}±{planck_peaks['R31_err']:.3f}")
                            if 'peak_ells' in measured and len(measured['peak_ells']) > 0:
                                self.log_progress(f"  TEST 4: Peak positions at ell={measured['peak_ells']} (data-driven)")
                else:
                    self.log_progress(f"  TEST 4: Empty arrays - len(ell)={len(ell)}, len(Cl_TT)={len(Cl_TT)}")
            
            omega_c = (omega_m or HLCDM_PARAMS.OMEGA_M) - omega_b
            test4_results = fit_peak_ratios_to_data(
                planck_peaks=planck_peaks,
                omega_b=omega_b,
                omega_c=omega_c,
                H0=H0
            )
            results['peaks'] = test4_results
            
            if planck_peaks is None:
                self.log_progress("  TEST 4: No peak data available, fitting skipped")
            
            beta_fit = float(test4_results.get('beta_fit', np.nan)) if np.isfinite(test4_results.get('beta_fit', np.nan)) else np.nan
            if not np.isnan(beta_fit):
                self.log_progress(f"  TEST 4 complete: β = {beta_fit:.3f}")
            else:
                self.log_progress(f"  TEST 4 complete: No valid peak data found (beta_fit={test4_results.get('beta_fit')})")
        except Exception as e:
            self.log_progress(f"  TEST 4 failed: {e}")
            import traceback
            self.log_progress(f"  Traceback: {traceback.format_exc()}")
            results['peaks'] = {'error': str(e)}
        
        # TEST 5: Cross-Modal Coherence
        self.log_progress("Running TEST 5: Cross-modal coherence at acoustic scale...")
        try:
            residual_TT, residual_TE, residual_EE, ell = compute_cmb_residuals('planck_2018')
            
            if len(ell) > 0:
                test5_results = cross_modal_coherence_at_harmonics(
                    residual_TT, residual_TE, residual_EE, ell
                )
                results['coherence'] = test5_results
                self.log_progress(f"  TEST 5 complete: Enhancement ratio = {test5_results.get('enhancement_ratio', np.nan):.3f}")
            else:
                results['coherence'] = {'error': 'Could not compute residuals'}
        except Exception as e:
            self.log_progress(f"  TEST 5 failed: {e}")
            results['coherence'] = {'error': str(e)}
        
        # Joint Analysis: Parameter Consistency
        self.log_progress("Running joint consistency check...")
        try:
            joint_results = joint_consistency_check(results)
            results['joint_consistency'] = joint_results
            beta_combined = float(joint_results.get('beta_combined', np.nan)) if np.isfinite(joint_results.get('beta_combined', np.nan)) else np.nan
            beta_combined_err = float(joint_results.get('beta_combined_err', np.nan)) if np.isfinite(joint_results.get('beta_combined_err', np.nan)) else np.nan
            self.log_progress(f"  Joint analysis: β_combined = {beta_combined:.3f} ± {beta_combined_err:.3f}")
        except Exception as e:
            self.log_progress(f"  Joint analysis failed: {e}")
            results['joint_consistency'] = {'error': str(e)}
        
        # Final Verdict
        self.log_progress("Determining final verdict...")
        try:
            verdict_results = final_verdict(
                results.get('joint_consistency', {}),
                results
            )
            results['verdict'] = verdict_results
            self.log_progress(f"  Verdict: {verdict_results.get('verdict', 'UNKNOWN')}")
        except Exception as e:
            self.log_progress(f"  Verdict determination failed: {e}")
            results['verdict'] = {'error': str(e)}
        
        self.log_progress("CMB-GW analysis complete")
        
        return results
    
    def validate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform basic statistical validation.
        
        Parameters:
        -----------
        context : dict, optional
            Validation parameters
            
        Returns:
        --------
        dict
            Validation results
        """
        self.log_progress("Running basic validation...")
        
        # Run main analysis if not already done
        if not self.results:
            self.results = self.run(context)
        
        validation_results = {
            'null_hypothesis_test': {
                'null_hypothesis': 'β = 0 (no G evolution)',
                'beta_combined': self.results.get('joint_consistency', {}).get('beta_combined', np.nan),
                'beta_err': self.results.get('joint_consistency', {}).get('beta_combined_err', np.nan),
                'p_value': self._compute_null_p_value()
            },
            'consistency_test': {
                'tests_consistent': self.results.get('joint_consistency', {}).get('consistent', False),
                'p_value': self.results.get('joint_consistency', {}).get('p_value', np.nan)
            }
        }
        
        return validation_results
    
    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform extended validation (Monte Carlo, bootstrap, etc.).
        
        Parameters:
        -----------
        context : dict, optional
            Extended validation parameters
            
        Returns:
        --------
        dict
            Extended validation results
        """
        self.log_progress("Running extended validation...")
        
        # For now, extended validation is a placeholder
        # Full implementation would include:
        # - Monte Carlo simulations with varying β
        # - Bootstrap resampling of datasets
        # - Cross-validation across different data subsets
        
        return {
            'extended_validation': 'Not yet implemented',
            'note': 'Extended validation requires Monte Carlo simulations and bootstrap resampling'
        }
    
    def _compute_null_p_value(self) -> float:
        """Compute p-value for null hypothesis (β = 0)."""
        joint = self.results.get('joint_consistency', {})
        beta = joint.get('beta_combined', 0)
        beta_err = joint.get('beta_combined_err', np.inf)
        
        if beta_err > 0:
            # Two-tailed test: p = 2 × (1 - Φ(|β/σ|))
            from scipy.stats import norm
            z_score = abs(beta / beta_err)
            p_value = 2 * (1 - norm.cdf(z_score))
            return p_value
        return np.nan

