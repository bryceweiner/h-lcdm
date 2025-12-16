"""
Final Verdict Determination
============================

Determine overall support for evolving G hypothesis based on joint analysis.

This module implements the verdict criteria from docs/cmb_gw.md:
- STRONG_POSITIVE: β > 2σ from 0, tests consistent, ≥5/6 criteria met
- TENTATIVE_POSITIVE: ≥3/6 criteria met
- NULL: β consistent with 0, <3 criteria met
"""

import numpy as np
from typing import Dict, Any


def final_verdict(
    joint_results: Dict[str, Any],
    individual_results: Dict[str, Dict[str, Any]]
) -> Dict[str, Any]:
    """
    Determine overall support for evolving G hypothesis.
    
    This implements the final verdict logic from docs/cmb_gw.md.
    
    Parameters:
    -----------
    joint_results : dict
        Results from joint_consistency_check()
    individual_results : dict
        Individual test results with keys:
        - 'sound_horizon': TEST 1 results
        - 'voids': TEST 2 results
        - 'sirens': TEST 3 results
        - 'peaks': TEST 4 results
        - 'coherence': TEST 5 results
        
    Returns:
    --------
    dict
        Verdict results containing:
        - 'verdict': 'STRONG_POSITIVE', 'TENTATIVE_POSITIVE', or 'NULL'
        - 'interpretation': Text interpretation
        - 'criteria': Dictionary of criteria checks
        - 'n_criteria_met': Number of criteria satisfied
    """
    beta_combined = joint_results.get('beta_combined', 0.0)
    beta_combined_err = joint_results.get('beta_combined_err', np.inf)
    consistent = joint_results.get('consistent', False)
    
    # Criteria for POSITIVE detection
    criteria = {
        'beta_nonzero': beta_combined > 2 * beta_combined_err if beta_combined_err > 0 else False,
        'tests_consistent': consistent,
        'sound_horizon_enhanced': False,
        'voids_enlarged': False,
        'coherence_elevated': False,
        'n_tests_significant': 0
    }
    
    # Check sound horizon enhancement
    sound_horizon = individual_results.get('sound_horizon', {})
    if 'r_s_observed' in sound_horizon and 'r_s_lcdm' in sound_horizon:
        r_s_obs = sound_horizon.get('r_s_observed', 0)
        r_s_lcdm = sound_horizon.get('r_s_lcdm', 0)
        criteria['sound_horizon_enhanced'] = r_s_obs > 149.0 if np.isfinite(r_s_obs) else False
    
    # Check void enlargement
    voids = individual_results.get('voids', {})
    if 'R_v_ratio' in voids:
        R_v_ratio = voids.get('R_v_ratio', 1.0)
        criteria['voids_enlarged'] = R_v_ratio > 1.05 if np.isfinite(R_v_ratio) else False
    
    # Check coherence enhancement
    coherence = individual_results.get('coherence', {})
    if 'enhancement_ratio' in coherence:
        enh_ratio = coherence.get('enhancement_ratio', 1.0)
        criteria['coherence_elevated'] = enh_ratio > 2.0 if np.isfinite(enh_ratio) else False
    
    # Count significant tests (Δχ² > 4)
    for test_name, test_result in individual_results.items():
        delta_chi2 = test_result.get('delta_chi2', 0)
        if np.isfinite(delta_chi2) and delta_chi2 > 4:
            criteria['n_tests_significant'] += 1
    
    n_criteria_met = sum(criteria.values())
    
    # Determine verdict
    if n_criteria_met >= 5 and criteria['beta_nonzero'] and criteria['tests_consistent']:
        verdict = 'STRONG_POSITIVE'
        interpretation = (
            f"Strong evidence for evolving G(z):\n"
            f"- Combined β = {beta_combined:.3f} ± {beta_combined_err:.3f}\n"
            f"- {n_criteria_met}/6 criteria met\n"
            f"- Independent tests yield consistent β values (p = {joint_results.get('p_value', 0):.3f})\n"
            f"- Physical interpretation: G_eff was ~{100*beta_combined:.1f}% weaker at recombination"
        )
    elif n_criteria_met >= 3:
        verdict = 'TENTATIVE_POSITIVE'
        interpretation = (
            f"Tentative evidence for evolving G(z):\n"
            f"- Combined β = {beta_combined:.3f} ± {beta_combined_err:.3f}\n"
            f"- {n_criteria_met}/6 criteria met\n"
            f"- Some tests show deviation but not all consistent\n"
            f"- Further investigation recommended"
        )
    else:
        verdict = 'NULL'
        interpretation = (
            f"No significant evidence for evolving G(z):\n"
            f"- β consistent with zero: {beta_combined:.3f} ± {beta_combined_err:.3f}\n"
            f"- Only {n_criteria_met}/6 criteria met\n"
            f"- Data consistent with standard ΛCDM (constant G)"
        )
    
    return {
        'verdict': verdict,
        'interpretation': interpretation,
        'criteria': criteria,
        'n_criteria_met': n_criteria_met
    }

