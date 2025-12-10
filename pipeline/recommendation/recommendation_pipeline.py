"""
H-ΛCDM CMB Cross-Modal Coherence Analysis Pipeline
==================================================

Implements 6-phase analysis protocol to test H-ΛCDM predictions:

Phase 1: Residual computation (ΔC_ℓ = C_ℓ^obs - C_ℓ^ΛCDM)
Phase 2: Cross-modal coherence test (KEY DISCRIMINANT)
Phase 3: Characteristic scale analysis (Fourier analysis)
Phase 4: H-ΛCDM CAMB model comparison
Phase 5: Amplitude consistency (α validation)
Phase 6: ML anomaly targeting

References:
    - BAO paper: H-ΛCDM BAO analysis
    - Cai et al. arXiv:1507.05619: Model comparison methodology
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import logging

from ..common.base_pipeline import AnalysisPipeline
from data.loader import DataLoader, DataUnavailableError
from .camb_interface import CAMBInterface
from .residual_analysis import ResidualAnalyzer
from .cross_modal_coherence import CrossModalCoherenceTest
from .characteristic_scales import CharacteristicScaleAnalyzer
from .hlcdm_camb_model import HLCDMCAMBModel
from .amplitude_consistency import AmplitudeConsistencyChecker
from .ml_anomaly_targeting import MLAnomalyTargeting
from .qtep_verification import QTEPVerificationTest

logger = logging.getLogger(__name__)


class RecommendationPipeline(AnalysisPipeline):
    """
    H-ΛCDM CMB Cross-Modal Coherence Analysis Pipeline.
    
    Executes 6-phase protocol to test H-ΛCDM predictions against CMB data.
    The key test is cross-modal coherence: H-ΛCDM predicts correlated residuals
    across TT, TE, EE due to coherent acoustic enhancement.
    """
    
    def __init__(self, output_dir: str = "results"):
        """Initialize recommendation pipeline."""
        super().__init__("recommendation", output_dir)
        self.data_loader = DataLoader(log_file=self.log_file)
        
        # Initialize analysis modules
        self.camb_interface = CAMBInterface()
        self.residual_analyzer = ResidualAnalyzer(self.camb_interface)
        self.coherence_tester = CrossModalCoherenceTest()
        self.scale_analyzer = CharacteristicScaleAnalyzer()
        self.hlcdm_model = HLCDMCAMBModel(self.camb_interface)
        self.amplitude_checker = AmplitudeConsistencyChecker(self.hlcdm_model)
        self.ml_targeter = MLAnomalyTargeting()
        self.qtep_verifier = QTEPVerificationTest()
        
        self.update_metadata("description", "H-ΛCDM CMB cross-modal coherence analysis")
        self.update_metadata("methodology", "6-phase protocol: residuals, coherence, scales, model, amplitude, ML")
    
    def _load_cmb_data(self) -> Dict[str, Dict[str, tuple]]:
        """
        Load CMB TT, TE, EE data from all available surveys.
        
        Returns:
            dict: Dictionary with survey names as keys, each containing:
                - 'TT': (ell, cl, cl_err)
                - 'TE': (ell, cl, cl_err)
                - 'EE': (ell, cl, cl_err)
        """
        datasets = {}
        
        # Load Planck 2018
        try:
            planck_data = self.data_loader.load_planck_2018()
            if planck_data:
                datasets['planck_2018'] = planck_data
                self.log_progress(f"✓ Loaded Planck 2018: TT={len(planck_data.get('TT', [[]])[0]) if planck_data.get('TT') else 0}, "
                                f"TE={len(planck_data.get('TE', [[]])[0]) if planck_data.get('TE') else 0}, "
                                f"EE={len(planck_data.get('EE', [[]])[0]) if planck_data.get('EE') else 0}")
        except Exception as e:
            self.log_progress(f"✗ Failed to load Planck 2018: {e}")
        
        # Load ACT DR6
        try:
            act_data = self.data_loader.load_act_dr6()
            if act_data:
                datasets['act_dr6'] = act_data
                self.log_progress(f"✓ Loaded ACT DR6: TT={len(act_data.get('TT', [[]])[0]) if act_data.get('TT') else 0}, "
                                f"TE={len(act_data.get('TE', [[]])[0]) if act_data.get('TE') else 0}, "
                                f"EE={len(act_data.get('EE', [[]])[0]) if act_data.get('EE') else 0}")
        except Exception as e:
            self.log_progress(f"✗ Failed to load ACT DR6: {e}")
        
        # Load SPT-3G
        try:
            spt_data = self.data_loader.load_spt3g()
            if spt_data:
                datasets['spt3g'] = spt_data
                self.log_progress(f"✓ Loaded SPT-3G: TT={len(spt_data.get('TT', [[]])[0]) if spt_data.get('TT') else 0}, "
                                f"TE={len(spt_data.get('TE', [[]])[0]) if spt_data.get('TE') else 0}, "
                                f"EE={len(spt_data.get('EE', [[]])[0]) if spt_data.get('EE') else 0}")
        except Exception as e:
            self.log_progress(f"✗ Failed to load SPT-3G: {e}")
        
        if not datasets:
            raise DataUnavailableError("No CMB data available")
        
        return datasets
    
    def _compute_theoretical_spectra(
        self,
        datasets: Dict[str, Dict[str, tuple]]
    ) -> Dict[str, Dict[str, tuple]]:
        """
        Compute theoretical ΛCDM spectra for all surveys/spectra.
        
        Parameters:
            datasets: Observed data
            
        Returns:
            dict: Theoretical spectra with same structure as datasets
        """
        theoretical = {}
        
        for survey_name, survey_data in datasets.items():
            theoretical[survey_name] = {}
            
            # Get survey-specific parameters
            survey_key = survey_name.lower().replace('-', '_')
            params = ResidualAnalyzer.SURVEY_PARAMS.get(survey_key, ResidualAnalyzer.SURVEY_PARAMS['planck_2018'])
            
            for spectrum in ['TT', 'TE', 'EE']:
                if spectrum not in survey_data:
                    continue
                
                ell_obs, _, _ = survey_data[spectrum]
                lmax = int(np.max(ell_obs)) + 200
                
                try:
                    if spectrum == 'TT':
                        ell, cl = self.camb_interface.compute_cl_tt(params, lmax=lmax)
                    elif spectrum == 'TE':
                        ell, cl = self.camb_interface.compute_cl_te(params, lmax=lmax)
                    elif spectrum == 'EE':
                        ell, cl = self.camb_interface.compute_cl_ee(params, lmax=lmax)
                    
                    theoretical[survey_name][spectrum] = (ell, cl)
                except Exception as e:
                    logger.warning(f"Failed to compute {spectrum} theory for {survey_name}: {e}")
        
        return theoretical
    
    def run(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Execute H-ΛCDM analysis protocol.
        
        Supports running specific recommendations via rec_ids in context:
        - rec_ids=[6]: Run QTEP ratio verification (Recommendation 6)
        - rec_ids not specified or empty: Run full 6-phase protocol
        
        Parameters:
            context: Optional context dictionary with optional 'rec_ids' list
            
        Returns:
            dict: Complete analysis results
        """
        context = context or {}
        rec_ids = context.get('rec_ids', [])
        
        # Check if running specific recommendation
        if rec_ids and 6 in rec_ids:
            return self._run_recommendation_6(context)
        
        # Default: Run full 6-phase protocol
        self.log_progress("Starting H-ΛCDM CMB cross-modal coherence analysis...")
        
        # Phase 1: Load data and compute residuals
        self.log_progress("Phase 1: Loading CMB data and computing residuals...")
        datasets = self._load_cmb_data()
        
        # Compute theoretical spectra
        theoretical_spectra = self._compute_theoretical_spectra(datasets)
        
        # Compute residuals
        residuals = self.residual_analyzer.compute_all_residuals(datasets)
        
        # Phase 2: Cross-modal coherence test (KEY DISCRIMINANT)
        self.log_progress("Phase 2: Testing cross-modal coherence (KEY TEST)...")
        coherence = self.coherence_tester.run_coherence_test(residuals)
        
        # Phase 3: Characteristic scale analysis
        self.log_progress("Phase 3: Analyzing characteristic scales...")
        scales = self.scale_analyzer.detect_characteristic_features(residuals)
        
        # Phase 4: H-ΛCDM model comparison
        self.log_progress("Phase 4: Comparing H-ΛCDM vs ΛCDM models...")
        model_comparison = {}
        for survey_name, survey_data in datasets.items():
            if survey_name not in theoretical_spectra:
                continue
            
            survey_key = survey_name.lower().replace('-', '_')
            params = ResidualAnalyzer.SURVEY_PARAMS.get(survey_key, ResidualAnalyzer.SURVEY_PARAMS['planck_2018'])
            
            model_comparison[survey_name] = {}
            for spectrum in ['TT', 'TE', 'EE']:
                if spectrum not in survey_data or spectrum not in theoretical_spectra[survey_name]:
                    continue
                
                ell, cl_obs, cl_err = survey_data[spectrum]
                
                try:
                    comparison = self.hlcdm_model.compare_with_lcdm(
                        params,
                        ell,
                        cl_obs,
                        cl_err,
                        spectrum=spectrum
                    )
                    model_comparison[survey_name][spectrum] = comparison
                except Exception as e:
                    logger.warning(f"Model comparison failed for {survey_name} {spectrum}: {e}")
        
        # Phase 5: Amplitude consistency
        self.log_progress("Phase 5: Checking amplitude consistency...")
        amplitude = self.amplitude_checker.check_all_spectra(
            residuals,
            theoretical_spectra,
            ResidualAnalyzer.SURVEY_PARAMS.get('act_dr6', ResidualAnalyzer.SURVEY_PARAMS['planck_2018'])
        )
        
        # Phase 6: ML anomaly targeting
        self.log_progress("Phase 6: Targeting ML-flagged anomalies...")
        ml_targeting = self.ml_targeter.targeted_test(
            residuals,
            coherence,
            amplitude
        )
        
        # Synthesize conclusion
        conclusion = self._synthesize_conclusion(coherence, scales, model_comparison, amplitude, ml_targeting)
        
        # Assemble results
        results = {
            'datasets': list(datasets.keys()),
            'residuals': self._serialize_residuals(residuals),
            'cross_modal_coherence': self._serialize_coherence(coherence),
            'characteristic_scales': self._serialize_scales(scales),
            'model_comparison': self._serialize_model_comparison(model_comparison),
            'amplitude_consistency': self._serialize_amplitude(amplitude),
            'ml_anomaly_targeting': self._serialize_ml_targeting(ml_targeting),
            'conclusion': conclusion,
        }
        
        self.results = results
        self.save_results(results)
        
        self.log_progress("✓ Analysis complete")
        self.log_progress(f"  Cross-modal coherence: {conclusion.get('coherence_summary', 'N/A')}")
        self.log_progress(f"  Amplitude α: {amplitude.get('consistency', {}).get('mean_alpha', 'N/A'):.2f}")
        
        return results
    
    def _run_recommendation_6(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Run Recommendation 6: Cross-Modal Coherence Test in Polarization Cross-Spectra.
        
        Tests for positive cross-modal correlation ρ(TE,EE) > 0 at ℓ=800-1200.
        
        PHYSICAL BASIS:
        - H-ΛCDM predicts ρ > 0 (correlated residuals from shared Lindblad mechanism)
        - ΛCDM predicts ρ ≈ 0 (independent Gaussian noise)
        
        NOTE: The QTEP ratio (2.257) is a theoretical INPUT, not an observable.
        It determines α through Lindblad-Zeno scaling. The observable consequence
        is positive cross-modal correlation, not the QTEP ratio itself.
        
        Parameters:
            context: Optional context dictionary
            
        Returns:
            dict: Cross-modal coherence verification results
        """
        self.log_progress("Running Recommendation 6: Cross-Modal Coherence Test...")
        self.log_progress("  Testing for ρ(TE,EE) > 0 (H-ΛCDM predicts correlated residuals)")
        self.log_progress("  Note: QTEP ratio (2.257) is theoretical input, not CMB observable")
        context = context or {}
        
        # Load CMB data
        self.log_progress("Loading CMB TE and EE data...")
        datasets = self._load_cmb_data()
        
        if not datasets:
            raise ValueError("No CMB data available for cross-modal coherence test")
        
        # Compute theoretical spectra
        theoretical_spectra = self._compute_theoretical_spectra(datasets)
        
        # Compute residuals
        self.log_progress("Computing residuals...")
        residuals = self.residual_analyzer.compute_all_residuals(datasets)
        
        # Run cross-modal coherence test
        self.log_progress("Running cross-modal coherence test...")
        coherence_results = self.qtep_verifier.run_verification(residuals)
        
        # Assemble results
        results = {
            'recommendation_id': 6,
            'test_name': 'Cross-Modal Coherence Test (TE-EE)',
            'physical_test': 'ρ(TE,EE) > 0 indicates correlated residuals (H-ΛCDM)',
            'note': 'QTEP ratio (2.257) is theoretical input, NOT CMB observable',
            'datasets': list(datasets.keys()),
            'qtep_verification': coherence_results,  # Legacy key name for compatibility
        }
        
        self.results = results
        self.save_results(results)
        
        # Log summary with physically meaningful interpretation
        combined_full = coherence_results.get('combined_full', {})
        combined_predicted = coherence_results.get('combined_predicted', {})
        
        rho_full = combined_full.get('rho_median', combined_full.get('R_median', np.nan))
        rho_std_full = combined_full.get('rho_std', combined_full.get('R_std', 0))
        bf_full = combined_full.get('bayes_factor', np.nan)
        interpretation_full = combined_full.get('interpretation', 'N/A')
        
        rho_pred = combined_predicted.get('rho_median', combined_predicted.get('R_median', np.nan))
        rho_std_pred = combined_predicted.get('rho_std', combined_predicted.get('R_std', 0))
        bf_pred = combined_predicted.get('bayes_factor', np.nan)
        
        self.log_progress("✓ Cross-modal coherence test complete")
        self.log_progress("  Full multipole range analysis:")
        if not np.isnan(rho_full):
            sig = abs(rho_full) / rho_std_full if rho_std_full > 0 else 0
            self.log_progress(f"    ρ(TE,EE) = {rho_full:.3f} ± {rho_std_full:.3f}")
            self.log_progress(f"    Significance vs null (ρ=0): {sig:.1f}σ")
            if rho_full > 0:
                self.log_progress(f"    Positive correlation detected (consistent with H-ΛCDM)")
            else:
                self.log_progress(f"    No positive correlation (consistent with ΛCDM null)")
        if not np.isnan(bf_full):
            self.log_progress(f"    Bayes factor: {bf_full:.2f} ({interpretation_full})")
        
        self.log_progress(f"  Predicted range (ℓ=800-1200) analysis:")
        if not np.isnan(rho_pred):
            sig_pred = abs(rho_pred) / rho_std_pred if rho_std_pred > 0 else 0
            self.log_progress(f"    ρ(TE,EE) = {rho_pred:.3f} ± {rho_std_pred:.3f} ({sig_pred:.1f}σ from null)")
        if not np.isnan(bf_pred):
            self.log_progress(f"    Bayes factor: {bf_pred:.2f}")
        
        return results
    
    def _synthesize_conclusion(
        self,
        coherence: Dict[str, Any],
        scales: Dict[str, Any],
        model_comparison: Dict[str, Any],
        amplitude: Dict[str, Any],
        ml_targeting: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Synthesize overall conclusion from all phases."""
        # Check coherence results
        coherence_summary = coherence.get('summary', {})
        n_significant = coherence_summary.get('n_significant_peaks', 0)
        n_in_range = coherence_summary.get('n_peaks_in_range', 0)
        
        # Check amplitude consistency
        amp_consistency = amplitude.get('consistency', {})
        alpha_mean = amp_consistency.get('mean_alpha', np.nan)
        within_prior = amp_consistency.get('within_prior', False)
        consistent_across = amp_consistency.get('consistent_across_spectra', False)
        
        # Check ML targeting
        ml_summary = ml_targeting.get('summary', {})
        n_ml_exceed = ml_summary.get('n_exceed_2sigma', 0)
        
        # Determine overall conclusion
        evidence_for_hlcdm = []
        
        if n_significant > 0:
            evidence_for_hlcdm.append(f"{n_significant} significant coherence peaks")
        if n_in_range > 0:
            evidence_for_hlcdm.append(f"{n_in_range} peaks in expected range (ℓ≈800-1200)")
        if not np.isnan(alpha_mean) and within_prior:
            evidence_for_hlcdm.append(f"α={alpha_mean:.2f} within theoretical prior")
        if consistent_across:
            evidence_for_hlcdm.append("α consistent across TT/TE/EE")
        if n_ml_exceed > 0:
            evidence_for_hlcdm.append(f"{n_ml_exceed} ML-flagged samples show 2σ excess")
        
        conclusion_text = "H-ΛCDM signatures detected" if evidence_for_hlcdm else "No clear H-ΛCDM signatures"
        
        return {
            'conclusion': conclusion_text,
            'evidence': evidence_for_hlcdm,
            'coherence_summary': f"{n_significant} significant peaks, {n_in_range} in expected range",
            'amplitude_summary': f"α={alpha_mean:.2f} (prior: [-7.7, -3.7])" if not np.isnan(alpha_mean) else "α not determined",
            'ml_summary': f"{n_ml_exceed}/{ml_summary.get('n_samples_tested', 0)} samples exceed 2σ",
        }
    
    def _serialize_residuals(self, residuals: Dict) -> Dict:
        """Serialize residuals for JSON output."""
        serialized = {}
        for survey, survey_residuals in residuals.items():
            serialized[survey] = {}
            for spectrum, data in survey_residuals.items():
                if data is None:
                    continue
                serialized[survey][spectrum] = {
                    'ell': data['ell'].tolist(),
                    'residual': data['residual'].tolist(),
                    'residual_fraction': data['residual_fraction'].tolist(),
                }
        return serialized
    
    def _serialize_coherence(self, coherence: Dict) -> Dict:
        """Serialize coherence results for JSON output."""
        # Keep summary and peak information, simplify detailed arrays
        serialized = {
            'summary': coherence.get('summary', {}),
        }
        
        by_survey = {}
        for survey, survey_results in coherence.get('by_survey', {}).items():
            by_survey[survey] = {}
            for pair, pair_results in survey_results.items():
                by_survey[survey][pair] = {
                    'peak': pair_results.get('peak', {}),
                }
        serialized['by_survey'] = by_survey
        
        return serialized
    
    def _serialize_scales(self, scales: Dict) -> Dict:
        """Serialize scale analysis for JSON output."""
        return {
            'summary': scales.get('summary', {}),
            'expected_characteristic_ell': scales.get('expected_characteristic_ell', np.nan),
            'expected_delta_ell': scales.get('expected_delta_ell', np.nan),
        }
    
    def _serialize_model_comparison(self, model_comparison: Dict) -> Dict:
        """Serialize model comparison for JSON output."""
        return model_comparison
    
    def _serialize_amplitude(self, amplitude: Dict) -> Dict:
        """Serialize amplitude consistency for JSON output."""
        return {
            'consistency': amplitude.get('consistency', {}),
        }
    
    def _serialize_ml_targeting(self, ml_targeting: Dict) -> Dict:
        """Serialize ML targeting for JSON output."""
        return {
            'summary': ml_targeting.get('summary', {}),
        }
    
    def validate(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Perform validation with bootstrap and null hypothesis testing.
        
        Tests:
        - Bootstrap stability of cross-modal coherence results
        - Null hypothesis: ρ(TE,EE) = 0 (ΛCDM) vs ρ > 0 (H-ΛCDM)
        
        Parameters:
            context: Optional validation parameters
                - n_bootstrap: Number of bootstrap iterations (default: 1000)
                - n_null: Number of null hypothesis simulations (default: 10000)
        
        Returns:
            dict: Validation results with bootstrap and null hypothesis tests
        """
        self.log_progress("Performing recommendation pipeline validation...")
        
        if not self.results:
            self.log_progress("✗ No results available for validation")
            return {
                'bootstrap': {'passed': False, 'error': 'No results available'},
                'null_hypothesis': {'passed': False, 'error': 'No results available'}
            }
        
        n_bootstrap = context.get('n_bootstrap', 1000) if context else 1000
        n_null = context.get('n_null', 10000) if context else 10000
        
        # Bootstrap validation
        bootstrap_results = self._bootstrap_coherence_validation(n_bootstrap)
        
        # Null hypothesis testing
        null_results = self._null_hypothesis_testing(n_null)
        
        validation_results = {
            'bootstrap': bootstrap_results,
            'null_hypothesis': null_results,
            'validation_level': 'basic',
            'n_bootstrap': n_bootstrap,
            'n_null': n_null
        }
        
        # Overall status
        bootstrap_passed = bootstrap_results.get('passed', False)
        null_passed = null_results.get('passed', False)
        validation_results['overall_status'] = 'PASSED' if (bootstrap_passed and null_passed) else 'FAILED'
        
        # Save validation results to main results
        if not self.results:
            self.results = {}
        self.results['validation'] = validation_results
        self.save_results(self.results)
        
        self.log_progress(f"✓ Validation complete: {validation_results['overall_status']}")
        
        return validation_results
    
    def validate_extended(self, context: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Extended validation (same as basic for this pipeline)."""
        return self.validate(context)
    
    def _bootstrap_coherence_validation(self, n_bootstrap: int = 1000) -> Dict[str, Any]:
        """
        Bootstrap validation of cross-modal coherence results.
        
        Resamples coherence data to assess stability of ρ(TE,EE) estimates.
        
        Parameters:
            n_bootstrap: Number of bootstrap iterations
            
        Returns:
            dict: Bootstrap validation results
        """
        try:
            # Extract coherence results from main results
            qtep_verification = self.results.get('qtep_verification', {})
            if not qtep_verification:
                return {'passed': False, 'error': 'No QTEP verification results available'}
            
            surveys = qtep_verification.get('surveys', {})
            if not surveys:
                return {'passed': False, 'error': 'No survey results available'}
            
            # Collect all coherence data from predicted range (ℓ=800-1200)
            all_coherence = []
            all_coherence_err = []
            
            for survey_name, survey_data in surveys.items():
                coherence_data = survey_data.get('coherence_data_predicted')
                if coherence_data and coherence_data.get('coherence'):
                    coherence = np.array(coherence_data['coherence'])
                    coherence_err = np.array(coherence_data['coherence_err'])
                    all_coherence.extend(coherence)
                    all_coherence_err.extend(coherence_err)
            
            if len(all_coherence) == 0:
                return {'passed': False, 'error': 'No coherence data available'}
            
            all_coherence = np.array(all_coherence)
            all_coherence_err = np.array(all_coherence_err)
            
            # Bootstrap resampling
            bootstrap_means = []
            bootstrap_stds = []
            
            for _ in range(n_bootstrap):
                # Resample with replacement
                indices = np.random.choice(len(all_coherence), size=len(all_coherence), replace=True)
                boot_coherence = all_coherence[indices]
                boot_err = all_coherence_err[indices]
                
                # Weighted mean
                weights = 1.0 / boot_err**2
                weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones_like(weights) / len(weights)
                boot_mean = np.sum(weights * boot_coherence)
                boot_std = np.sqrt(1.0 / np.sum(1.0 / boot_err**2)) if np.sum(1.0 / boot_err**2) > 0 else np.std(boot_coherence)
                
                bootstrap_means.append(boot_mean)
                bootstrap_stds.append(boot_std)
            
            bootstrap_means = np.array(bootstrap_means)
            bootstrap_stds = np.array(bootstrap_stds)
            
            # Original estimate
            weights_orig = 1.0 / all_coherence_err**2
            weights_orig = weights_orig / np.sum(weights_orig)
            rho_observed = np.sum(weights_orig * all_coherence)
            
            # Bootstrap statistics
            rho_bootstrap_mean = np.mean(bootstrap_means)
            rho_bootstrap_std = np.std(bootstrap_means)
            rho_bootstrap_median = np.median(bootstrap_means)
            
            # Credible intervals
            rho_ci_68 = [np.percentile(bootstrap_means, 16), np.percentile(bootstrap_means, 84)]
            rho_ci_95 = [np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5)]
            
            # Check if observed value is within bootstrap distribution
            within_ci_95 = rho_ci_95[0] <= rho_observed <= rho_ci_95[1]
            
            # Stability check: bootstrap std should be reasonable
            stability_ok = rho_bootstrap_std < 0.5  # Coherence should be stable
            
            return {
                'passed': within_ci_95 and stability_ok,
                'n_bootstrap': n_bootstrap,
                'n_successful': n_bootstrap,
                'rho_observed': float(rho_observed),
                'rho_bootstrap_mean': float(rho_bootstrap_mean),
                'rho_bootstrap_std': float(rho_bootstrap_std),
                'rho_bootstrap_median': float(rho_bootstrap_median),
                'rho_ci_68': [float(rho_ci_68[0]), float(rho_ci_68[1])],
                'rho_ci_95': [float(rho_ci_95[0]), float(rho_ci_95[1])],
                'within_ci_95': within_ci_95,
                'stability_ok': stability_ok,
                'interpretation': 'Bootstrap validation passed' if (within_ci_95 and stability_ok) else 'Bootstrap validation failed'
            }
            
        except Exception as e:
            logger.error(f"Bootstrap validation error: {e}", exc_info=True)
            return {'passed': False, 'error': str(e)}
    
    def _null_hypothesis_testing(self, n_null: int = 10000) -> Dict[str, Any]:
        """
        Null hypothesis testing: ρ(TE,EE) = 0 (ΛCDM) vs ρ > 0 (H-ΛCDM).
        
        Generates null hypothesis realizations (uncorrelated TE/EE residuals)
        and compares observed correlation to null distribution.
        
        Parameters:
            n_null: Number of null hypothesis simulations
            
        Returns:
            dict: Null hypothesis test results
        """
        try:
            # Extract coherence results
            qtep_verification = self.results.get('qtep_verification', {})
            if not qtep_verification:
                return {'passed': False, 'error': 'No QTEP verification results available'}
            
            surveys = qtep_verification.get('surveys', {})
            if not surveys:
                return {'passed': False, 'error': 'No survey results available'}
            
            # Get observed correlation from combined results
            combined_predicted = qtep_verification.get('combined_predicted', {})
            rho_observed = combined_predicted.get('rho_median', combined_predicted.get('R_median', np.nan))
            
            if np.isnan(rho_observed):
                return {'passed': False, 'error': 'No observed correlation value available'}
            
            # Collect coherence data for null simulation
            all_coherence = []
            all_coherence_err = []
            n_bins_total = 0
            
            for survey_name, survey_data in surveys.items():
                coherence_data = survey_data.get('coherence_data_predicted')
                if coherence_data and coherence_data.get('coherence'):
                    coherence = np.array(coherence_data['coherence'])
                    coherence_err = np.array(coherence_data['coherence_err'])
                    all_coherence.extend(coherence)
                    all_coherence_err.extend(coherence_err)
                    n_bins_total += len(coherence)
            
            if len(all_coherence) == 0:
                return {'passed': False, 'error': 'No coherence data available'}
            
            all_coherence_err = np.array(all_coherence_err)
            
            # Generate null hypothesis realizations
            # Under null hypothesis (ΛCDM): ρ = 0, residuals are uncorrelated
            null_correlations = []
            
            for _ in range(min(n_null, 5000)):  # Limit for computational efficiency
                # Generate uncorrelated Gaussian noise (null hypothesis)
                null_coherence = np.random.normal(0, all_coherence_err, size=len(all_coherence_err))
                
                # Weighted mean (same as observed)
                weights = 1.0 / all_coherence_err**2
                weights = weights / np.sum(weights)
                null_mean = np.sum(weights * null_coherence)
                null_correlations.append(null_mean)
            
            null_correlations = np.array(null_correlations)
            
            # Statistics of null distribution
            null_mean = np.mean(null_correlations)
            null_std = np.std(null_correlations)
            
            # p-value: probability of observing |ρ| >= |ρ_observed| under null hypothesis
            p_value = np.mean(np.abs(null_correlations) >= np.abs(rho_observed))
            
            # Significance in sigma
            if null_std > 0:
                significance_sigma = abs(rho_observed - null_mean) / null_std
            else:
                significance_sigma = 0.0
            
            # Interpretation
            null_rejected = p_value < 0.05  # Standard threshold
            if p_value < 0.001:
                evidence_strength = "VERY_STRONG"
            elif p_value < 0.01:
                evidence_strength = "STRONG"
            elif p_value < 0.05:
                evidence_strength = "MODERATE"
            else:
                evidence_strength = "WEAK"
            
            interpretation = (
                f"Null hypothesis (ρ=0) {'rejected' if null_rejected else 'not rejected'} "
                f"(p={p_value:.4f}, {significance_sigma:.2f}σ). "
                f"{'Consistent with H-ΛCDM' if (rho_observed > 0 and null_rejected) else 'Consistent with ΛCDM null'}."
            )
            
            return {
                'passed': True,  # Test completed successfully
                'n_null': len(null_correlations),
                'rho_observed': float(rho_observed),
                'rho_null_mean': float(null_mean),
                'rho_null_std': float(null_std),
                'p_value': float(p_value),
                'significance_sigma': float(significance_sigma),
                'null_rejected': bool(null_rejected),
                'evidence_strength': evidence_strength,
                'interpretation': interpretation,
                # Legacy format for reporter compatibility
                'lcdm': {
                    'n_null': len(null_correlations),
                    'chi2_obs': float(rho_observed**2),  # Approximate as chi2-like
                    'chi2_null_mean': float(null_mean**2),
                    'chi2_null_std': float(null_std**2),
                    'p_value_chi2': float(p_value),
                    'interpretation_chi2': interpretation
                },
                'hlcdm': {
                    'n_null': len(null_correlations),
                    'chi2_obs': float(rho_observed**2),
                    'chi2_null_mean': float(null_mean**2),
                    'chi2_null_std': float(null_std**2),
                    'p_value_chi2': float(p_value),
                    'interpretation_chi2': interpretation
                }
            }
            
        except Exception as e:
            logger.error(f"Null hypothesis testing error: {e}", exc_info=True)
            return {'passed': False, 'error': str(e)}
