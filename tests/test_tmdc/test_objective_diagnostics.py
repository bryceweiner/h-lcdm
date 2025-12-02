import numpy as np

from pipeline.tmdc.simulation.objective import objective_diagnostics


def test_objective_diagnostics_basic_structure():
    """
    Sanity check that objective_diagnostics returns the expected keys and types
    for a representative 7-layer configuration.
    
    This does not assert specific numerical values (which depend on the full
    TMDC physics stack), only that the decomposition is well-formed and finite.
    """
    # Simple alternating pattern near the MoS2 magic angle difference (~3.8 deg)
    angles_deg = np.array([0.9, 4.7, 0.9, 4.7, 0.9, 4.7, 0.9], dtype=float)
    
    diag = objective_diagnostics(angles_deg)
    
    # Required keys
    for key in [
        "base_amplification",
        "chain_penalty",
        "total_strain_energy",
        "strain_penalty_factor",
        "moire_couplings",
        "min_coupling",
        "max_coupling",
        "mean_coupling",
    ]:
        assert key in diag, f"Missing key in diagnostics: {key}"
    
    # Basic type/finite checks
    assert isinstance(diag["base_amplification"], float)
    assert np.isfinite(diag["base_amplification"])
    assert isinstance(diag["chain_penalty"], float)
    assert 0.0 <= diag["chain_penalty"] <= 1.0
    assert isinstance(diag["strain_penalty_factor"], float)
    assert diag["strain_penalty_factor"] >= 0.0
    
    moire_couplings = diag["moire_couplings"]
    assert isinstance(moire_couplings, list)
    # For a 7-layer stack we expect 6 interfaces
    assert len(moire_couplings) == 6


