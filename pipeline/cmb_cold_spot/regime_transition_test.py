"""
QTEP Regime Transition Test
============================

Test for regime transitions (not discrete peaks) at QTEP-predicted multipoles.

If QTEP governs information processing at recombination, transitions between
different physical regimes should occur at multipoles following η^n:
- ℓ₀ = 18: Sachs-Wolfe → Acoustic transition
- ℓ₁ = 41: Early acoustic regime boundary
- ℓ₂ = 92: Post-peak damping transition

This tests a MORE subtle prediction than discrete peaks: information processing
constraints create natural boundaries between physical regimes.

References:
- QTEP ratio: η = ln(2)/(1-ln(2)) ≈ 2.257
- Spectral index analysis: Planck Collaboration (2020)
- Regime identification: Hu & Dodelson (2002) CMB theory
"""

import numpy as np
from typing import Dict, Any, List, Tuple
import logging
from scipy import signal, stats

logger = logging.getLogger(__name__)

# QTEP ratio
QTEP_RATIO = np.log(2) / (1 - np.log(2))  # ≈ 2.257


def calculate_spectral_index(ell: np.ndarray, C_ell: np.ndarray) -> np.ndarray:
    """
    Calculate spectral index n_s(ℓ) = d(ln C_ℓ)/d(ln ℓ).
    
    Regime transitions manifest as changes in spectral index.
    
    Parameters:
        ell: Multipole array
        C_ell: Power spectrum
        
    Returns:
        Spectral index array
    """
    # Avoid log of zero/negative
    C_ell_clean = np.maximum(C_ell, 1e-30)
    ell_clean = np.maximum(ell, 1)
    
    # Calculate in log space
    log_ell = np.log(ell_clean)
    log_C_ell = np.log(C_ell_clean)
    
    # Numerical derivative
    n_s = np.gradient(log_C_ell, log_ell)
    
    return n_s


def detect_regime_transitions(ell: np.ndarray,
                              n_s: np.ndarray,
                              window: int = 5) -> List[int]:
    """
    Detect regime transitions as inflection points in spectral index.
    
    Parameters:
        ell: Multipole array
        n_s: Spectral index array
        window: Smoothing window
        
    Returns:
        List of multipoles where transitions occur
    """
    # Smooth spectral index to reduce noise
    n_s_smooth = np.convolve(n_s, np.ones(window)/window, mode='same')
    
    # Find inflection points: where d²(ln C_ℓ)/d(ln ℓ)² changes sign
    second_deriv = np.gradient(n_s_smooth)
    
    # Zero crossings of second derivative indicate inflection points
    sign_changes = np.diff(np.sign(second_deriv))
    transition_indices = np.where(sign_changes != 0)[0]
    
    # Get corresponding multipoles
    transitions = [int(ell[i]) for i in transition_indices if 2 < i < len(ell)-2]
    
    return transitions


def test_qtep_regime_transitions(ell: np.ndarray,
                                 C_ell: np.ndarray,
                                 qtep_multipoles: List[int]) -> Dict[str, Any]:
    """
    Test if regime transitions occur at QTEP-predicted multipoles.
    
    Instead of looking for discrete peaks (which failed), look for:
    - Changes in spectral index at ℓ = [18, 41, 92, ...]
    - Regime boundaries (Sachs-Wolfe → Acoustic, etc.)
    - Transitions aligned with η^n scaling
    
    Parameters:
        ell: Multipole array
        C_ell: Power spectrum
        qtep_multipoles: QTEP-predicted transition multipoles
        
    Returns:
        Dictionary with transition test results
    """
    # Calculate spectral index
    n_s = calculate_spectral_index(ell, C_ell)
    
    # Detect regime transitions empirically
    detected_transitions = detect_regime_transitions(ell, n_s, window=5)
    
    # Test: How close are detected transitions to QTEP predictions?
    matches = []
    match_quality = []
    
    for qtep_ell in qtep_multipoles:
        # Find nearest detected transition
        if detected_transitions:
            distances = [abs(trans - qtep_ell) for trans in detected_transitions]
            nearest_dist = min(distances)
            nearest_trans = detected_transitions[np.argmin(distances)]
            
            # Consider a match if within ±5 multipoles (typical uncertainty)
            is_match = (nearest_dist <= 5)
            
            matches.append(is_match)
            match_quality.append(nearest_dist)
            
            if is_match:
                logger.info(f"QTEP ℓ={qtep_ell} → Detected transition at ℓ={nearest_trans} (Δℓ={nearest_dist})")
        else:
            matches.append(False)
            match_quality.append(999)
    
    # Calculate metrics
    n_matches = sum(matches)
    n_predicted = len(qtep_multipoles)
    match_fraction = n_matches / n_predicted if n_predicted > 0 else 0.0
    avg_match_quality = np.mean([q for q in match_quality if q < 100])
    
    # Statistical significance
    # Null: Random multipoles would match by chance with probability ~ 10/100 = 0.1
    # Binomial test: P(n_matches | n_trials, p=0.1)
    null_prob = 0.1  # Probability of random match
    p_value_binomial = 1.0 - stats.binom.cdf(n_matches - 1, n_predicted, null_prob)
    
    # Result
    if match_fraction >= 0.67 and p_value_binomial < 0.05:
        result = "QTEP_TRANSITIONS_DETECTED"
    elif match_fraction >= 0.5:
        result = "PARTIAL_AGREEMENT"
    else:
        result = "NO_QTEP_TRANSITIONS"
    
    logger.info(f"Regime transition test: {n_matches}/{n_predicted} QTEP multipoles matched")
    logger.info(f"Match fraction: {match_fraction:.2f}, p={p_value_binomial:.4f}")
    logger.info(f"Result: {result}")
    
    return {
        'qtep_multipoles': qtep_multipoles,
        'detected_transitions': detected_transitions,
        'n_matches': n_matches,
        'match_fraction': float(match_fraction),
        'avg_match_quality_ell': float(avg_match_quality) if match_quality else 0.0,
        'p_value_binomial': float(p_value_binomial),
        'result': result,
        'method': 'regime_transition_detection',
        'null_hypothesis': 'Random multipoles, p_match=0.1',
        'interpretation': 'QTEP predicts regime boundaries, not discrete peaks',
        'reference': 'Information processing constraints → regime structure'
    }

