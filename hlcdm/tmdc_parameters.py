"""
TMDC Material Parameters
========================

Physical constants for Transition Metal Dichalcogenides, focused on WSe2.

References:
- Devakul et al., "Magic in twisted transition metal dichalcogenide bilayers," Nat. Commun. 12, 6730 (2021).
- Foutty et al., "Mapping twist-tuned multiband topology in bilayer WSe2," Science 381, eadi4728 (2023).
- An et al., "Interaction effects and superconductivity signatures in twisted double-bilayer WSe2," Nanoscale Horiz. 5, 1309 (2020).
- Zhang et al., "Flat bands in twisted bilayer transition metal dichalcogenides," arXiv:1910.13068.
"""

class WSe2Parameters:
    """
    Physical parameters for Tungsten Diselenide (WSe2).
    """
    # Structural Constants
    # Lattice constant a [Å] -> [nm]
    # Reference: docs/wse_2.md (approx 3.3 Å)
    LATTICE_CONSTANT_A_NM = 0.33  # 3.3 Å
    
    # Interlayer distance [nm]
    # Typical vdW gap for TMDCs
    INTERLAYER_DISTANCE_D_NM = 0.65 # 6.5 Å
    
    # Electronic Constants
    # Effective mass (m*/m0)
    # WSe2 has heavier holes than MoS2. Approx 0.35-0.45.
    EFFECTIVE_MASS = 0.4 
    
    # Spin-Orbit Coupling [eV]
    # Strong SOC in WSe2, valence band splitting ~450-460 meV
    SOC_LAMBDA_EV = 0.46 
    
    # Hopping Parameters [eV]
    # Nearest-neighbor hopping
    HOPPING_T1_EV = 1.1 
    # Next-nearest-neighbor hopping
    HOPPING_T2_EV = 0.15
    
    # Magic Angle Physics
    # Primary topological magic angle (Devakul et al.)
    MAGIC_ANGLE_PRIMARY_DEG = 1.2
    
    # Broad flat-band window (An et al., Zhang et al.)
    FLAT_BAND_WINDOW_DEG = (1.0, 3.0)
    
    # Moiré Potential Scale
    # VdW coupling strength gamma0 [eV] at optimal stacking
    VDW_COUPLING_GAMMA0_EV = 0.18  # Slightly stronger than MoS2 due to larger orbitals? Keeping similar scale.

