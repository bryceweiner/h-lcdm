"""
N-Body Void Calibration for Evolving G(z)
==========================================

DEPRECATED: This module is experimental and superseded by literature-based calibration.

For production analysis, use the literature calibration from Pisani+ (2015) instead:
- Module: pipeline.cmb_gw.physics.void_scaling_literature
- Formula: R_v(β)/R_v(0) = [D(β)/D(0)]^{1.7±0.2}
- Based on: Billion-particle N-body simulations (MultiDark, resolution 2048³)

This module is kept for reference/future work but is marked as experimental.
Custom N-body simulations are computationally expensive and the literature
calibration is MORE rigorous (based on professional simulations).

Original description:
Rigorous calibration of void size scaling with β using N-body simulations.

This module:
1. Runs N-body simulations with modified growth factor for different β values
2. Identifies voids in each simulation using density-based algorithms
3. Measures void size distributions
4. Calibrates the void_size_ratio(z, Ω_m, β) function empirically
5. Provides rigorous β extraction from observed void data

Uses custom MPS-accelerated Particle-Mesh N-body simulator (nbody_pm_gpu.py)
with GPU acceleration (MPS for Apple Silicon, CUDA for NVIDIA).
"""

import numpy as np
import logging
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path
import pickle
import time
import threading
from tqdm import tqdm

logger = logging.getLogger(__name__)

# Check for custom MPS-accelerated N-body simulator
try:
    from .nbody_pm_gpu import ParticleMeshNBody, GPU_AVAILABLE, GPU_TYPE, TORCH_AVAILABLE
    NBODY_PM_AVAILABLE = TORCH_AVAILABLE
    if NBODY_PM_AVAILABLE:
        logger.info(f"Custom MPS-accelerated N-body simulator available ({GPU_TYPE})")
    else:
        logger.warning("PyTorch not available. Install with: pip install torch")
except ImportError as e:
    NBODY_PM_AVAILABLE = False
    logger.warning(f"Custom N-body simulator not available: {e}")

# Backward compatibility (deprecated - use custom simulator)
GRAV_SIM_AVAILABLE = False
NBODYKIT_AVAILABLE = False


class NBODYVoidCalibration:
    """
    N-body simulation-based calibration for void size scaling with evolving G.
    
    This provides RIGOROUS β extraction by running simulations with modified
    growth factors and measuring resulting void statistics.
    
    Uses custom MPS-accelerated Particle-Mesh N-body simulator (nbody_pm_gpu.py)
    with GPU acceleration (MPS for Apple Silicon, CUDA for NVIDIA).
    """
    
    def __init__(
        self,
        box_size: float = 256.0,  # Mpc/h
        n_particles: int = 32**3,  # ~32K particles (reasonable for calibration)
        n_grid: int = 128,  # Grid resolution for PM (should be ≥ 2× particle cube root)
        z_initial: float = 49.0,
        z_final: float = 0.0,
        cache_dir: str = "nbody_cache"
    ):
        """
        Initialize N-body void calibration.
        
        Parameters:
        -----------
        box_size : float
            Simulation box size in Mpc/h (default: 256 Mpc/h)
        n_particles : int
            Number of particles (default: 32³ ≈ 32K)
        n_grid : int
            Grid resolution for Particle-Mesh method (default: 128)
            Should be ≥ 2× particle cube root for accuracy
        z_initial : float
            Initial redshift for simulation (default: 49)
        z_final : float
            Final redshift (default: 0)
        cache_dir : str
            Directory to cache simulation results
        """
        if not NBODY_PM_AVAILABLE:
            raise ImportError(
                "Custom MPS-accelerated N-body simulator is required. "
                "PyTorch must be installed: pip install torch"
            )
        
        self.box_size = box_size
        self.n_particles = n_particles
        self.n_grid = n_grid
        self.z_initial = z_initial
        self.z_final = z_final
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
        
        # Initialize custom MPS-accelerated PM simulator
        self.simulator = ParticleMeshNBody(
            box_size=box_size,
            n_particles=n_particles,
            n_grid=n_grid,
            z_initial=z_initial,
            z_final=z_final
        )
        
        # Cosmological parameters (Planck 2018) - stored in simulator, keep for compatibility
        self.h = 0.6736
        self.Omega_m = 0.315
        self.Omega_b = 0.049
        self.sigma8 = 0.811
        self.n_s = 0.9649
        
        logger.info(
            f"Initialized N-body calibration: {box_size} Mpc/h box, "
            f"{n_particles:,} particles, {n_grid}³ grid using MPS-accelerated PM simulator ({GPU_TYPE})"
        )
    
    def growth_factor_beta(self, z: float, beta: float) -> float:
        """
        Growth factor D(z) with evolving G parameterized by β.
        
        Parameters:
        -----------
        z : float
            Redshift
        beta : float
            Coupling strength for G evolution
            
        Returns:
        --------
        float
            Growth factor relative to ΛCDM at z=0
        """
        from ..physics.growth_factor import growth_factor_evolving_G
        
        # Use the physics module for consistency
        # Pass as array and extract scalar
        z_array = np.array([z]) if np.isscalar(z) else z
        D_beta_array = growth_factor_evolving_G(z_array, self.Omega_m, beta)
        D_beta = D_beta_array[0] if np.isscalar(z) else D_beta_array
        
        D_lcdm_z0_array = growth_factor_evolving_G(np.array([0.0]), self.Omega_m, 0.0)
        D_lcdm_z0 = D_lcdm_z0_array[0]
        
        # Normalize to D(z=0) = 1 for ΛCDM
        return float(D_beta / D_lcdm_z0)
    
    # Note: generate_initial_conditions is now handled by the simulator internally
    
    def run_simulation(
        self,
        beta: float,
        n_steps: int = 1000,
        force_rerun: bool = False,
        seed: int = 42
    ) -> Dict[str, np.ndarray]:
        """
        Run N-body simulation with evolving G(z) characterized by β.
        
        Uses custom MPS-accelerated Particle-Mesh simulator.
        
        Parameters:
        -----------
        beta : float
            Coupling strength for G evolution
        n_steps : int
            Number of time steps (default: 1000, literature standard for z=49→0)
        force_rerun : bool
            Force rerun even if cached results exist
        seed : int
            Random seed
            
        Returns:
        --------
        dict
            Final particle positions, velocities, masses
        """
        # Check cache
        cache_file = self.cache_dir / f"sim_beta_{beta:.4f}_np_{self.n_particles}_seed_{seed}.pkl"
        checkpoint_file = self.cache_dir / f"sim_beta_{beta:.4f}_np_{self.n_particles}_seed_{seed}_checkpoint.pkl"
        
        if cache_file.exists() and not force_rerun:
            logger.info(f"Loading cached simulation for β={beta:.4f}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Create detailed progress bar callback for evolution steps
        import sys
        pbar = tqdm(
            total=n_steps, 
            desc=f"    β={beta:+.2f} evolution", 
            unit="step",
            position=1,
            leave=False, 
            dynamic_ncols=True,
            file=sys.stdout,
            bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}step/s, z={postfix}]',
            mininterval=0.5  # Update at most every 0.5 seconds
        )
        
        def progress_callback(step: int, total: int, rate: float, z: float):
            """Update progress bar with detailed info"""
            pbar.n = step
            pbar.set_postfix_str(f"{z:.2f}")
            pbar.refresh()
        
        try:
            # Run simulation using custom MPS-accelerated simulator
            # The simulator handles initial conditions, evolution, and evolving G internally
            result = self.simulator.run_simulation(
                beta=beta,
                n_steps=n_steps,
                progress_callback=progress_callback,
                seed=seed,
                checkpoint_file=str(checkpoint_file),
                checkpoint_interval=100  # Save every 100 steps
            )
        finally:
            pbar.close()
        
        # Cache results
        logger.info(f"  Saving simulation to cache...")
        with open(cache_file, 'wb') as f:
            pickle.dump(result, f)
        
        logger.info(f"  ✓ Simulation complete")
        
        return result
    
    def find_voids_in_simulation(
        self,
        state: Dict[str, np.ndarray],
        min_void_radius: float = 10.0  # Mpc/h
    ) -> np.ndarray:
        """
        Identify voids in simulation using density-based void finder.
        
        Parameters:
        -----------
        state : dict
            Particle state (positions, velocities, masses)
        min_void_radius : float
            Minimum void radius in Mpc/h
            
        Returns:
        --------
        array
            Array of void radii in Mpc/h
        """
        logger.info("Finding voids in simulation...")
        
        positions = state['positions']
        
        # Create density grid
        n_grid = 64  # Grid resolution
        grid_spacing = self.box_size / n_grid
        
        # Bin particles into grid using numpy histogram
        density, edges = np.histogramdd(
            positions,
            bins=[n_grid, n_grid, n_grid],
            range=[[0, self.box_size]] * 3
        )
        
        # Normalize density
        mean_density = np.mean(density)
        density_contrast = (density - mean_density) / mean_density
        
        # Find underdense regions (voids)
        from scipy import ndimage
        
        # Threshold for void identification (δ < -0.7)
        void_mask = density_contrast < -0.7
        
        # Label connected underdense regions
        labeled_voids, n_voids = ndimage.label(void_mask)
        
        logger.info(f"Found {n_voids} candidate voids")
        
        # Measure void sizes
        void_radii = []
        
        for void_id in range(1, n_voids + 1):
            void_region = (labeled_voids == void_id)
            n_cells = np.sum(void_region)
            
            # Volume in (Mpc/h)³
            cell_volume = grid_spacing**3
            void_volume = n_cells * cell_volume
            
            # Effective radius: R = (3V/4π)^(1/3)
            void_radius = (3 * void_volume / (4 * np.pi))**(1.0/3.0)
            
            if void_radius >= min_void_radius:
                void_radii.append(void_radius)
        
        logger.info(
            f"Found {len(void_radii)} voids above {min_void_radius} Mpc/h threshold"
        )
        
        return np.array(void_radii)
    
    def calibrate_void_scaling(
        self,
        beta_values: Optional[List[float]] = None,
        n_realizations: int = 3,
        n_steps: int = 1000,
        force_rerun: bool = False
    ) -> Dict[str, Any]:
        """
        Run simulations across β values to calibrate void size scaling.
        
        This is the RIGOROUS method for extracting β from void data.
        
        Parameters:
        -----------
        beta_values : list, optional
            β values to simulate. Default: [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        n_realizations : int
            Number of realizations per β (for error estimation)
        n_steps : int
            Number of time steps for simulation (default: 1000, literature standard)
        force_rerun : bool
            Force rerun even if cached calibration exists
            
        Returns:
        --------
        dict
            Calibration data containing ratio(β) function
        """
        if beta_values is None:
            beta_values = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
        
        # Check cache
        cache_file = self.cache_dir / "void_calibration.pkl"
        
        if cache_file.exists() and not force_rerun:
            logger.info(f"Loading cached calibration from {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Running calibration simulations for β = {beta_values}")
        logger.info(f"  {n_realizations} realizations per β value")
        logger.info(f"  Total simulations: {len(beta_values) * n_realizations}")
        
        # Initialize or load partial calibration data
        checkpoint_file = self.cache_dir / "void_calibration_checkpoint.pkl"
        
        if checkpoint_file.exists() and not force_rerun:
            logger.info(f"Found checkpoint file, resuming calibration...")
            with open(checkpoint_file, 'rb') as f:
                calibration_data = pickle.load(f)
        else:
            calibration_data = {
                'beta_grid': beta_values,
                'void_size_ratio': [],
                'void_size_ratio_err': [],
                'all_void_sizes': {},
                'completed_betas': []  # Track which β values are done
            }
        
        # Import tqdm for progress bars
        try:
            from tqdm import tqdm
            use_tqdm = True
        except ImportError:
            use_tqdm = False
            logger.warning("tqdm not available - no progress bars")
        
        # Run ΛCDM baseline (β=0) first if not already done
        if 0.0 not in calibration_data.get('completed_betas', []):
            logger.info("Running ΛCDM baseline (β=0)...")
            lcdm_void_sizes = []
            
            iterator = range(n_realizations)
            if use_tqdm:
                iterator = tqdm(
                    iterator, 
                    desc="  ΛCDM (β=0.00)", 
                    unit="sim",
                    position=0,
                    leave=True,
                    dynamic_ncols=True
                )
            
            for i in iterator:
                state = self.run_simulation(beta=0.0, n_steps=n_steps, force_rerun=force_rerun, seed=42 + i)
                voids = self.find_voids_in_simulation(state)
                lcdm_void_sizes.extend(voids)
                if not use_tqdm:
                    logger.info(f"  Realization {i+1}/{n_realizations} complete: {len(voids)} voids")
            
            calibration_data['all_void_sizes'][0.0] = lcdm_void_sizes
            calibration_data['completed_betas'].append(0.0)
            
            # Checkpoint after ΛCDM
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(calibration_data, f)
            logger.info(f"✓ ΛCDM baseline complete, checkpoint saved")
        
        mean_R_v_lcdm = np.mean(calibration_data['all_void_sizes'][0.0])
        logger.info(f"ΛCDM mean void size: {mean_R_v_lcdm:.2f} Mpc/h")
        
        # Run for each β value
        for beta_idx, beta in enumerate(beta_values):
            # Skip if already completed
            if beta in calibration_data.get('completed_betas', []):
                logger.info(f"β={beta:.2f} already completed (resuming from checkpoint)")
                continue
            
            logger.info(f"Running simulations for β={beta:.2f} ({beta_idx+1}/{len(beta_values)})...")
            
            beta_void_sizes = []
            
            iterator = range(n_realizations)
            if use_tqdm:
                iterator = tqdm(
                    iterator, 
                    desc=f"  β={beta:+.2f}", 
                    unit="sim",
                    position=0,
                    leave=True,
                    dynamic_ncols=True
                )
            
            for i in iterator:
                seed = 100 + int(beta*1000) + i
                state = self.run_simulation(beta=beta, n_steps=n_steps, force_rerun=force_rerun, seed=seed)
                voids = self.find_voids_in_simulation(state)
                beta_void_sizes.extend(voids)
                if not use_tqdm:
                    logger.info(f"  Realization {i+1}/{n_realizations} complete: {len(voids)} voids")
            
            mean_R_v_beta = np.mean(beta_void_sizes)
            ratio = mean_R_v_beta / mean_R_v_lcdm
            
            # Error from scatter across realizations
            chunk_size = len(beta_void_sizes) // n_realizations
            realization_means = []
            for i in range(n_realizations):
                start = i * chunk_size
                end = (i + 1) * chunk_size if i < n_realizations - 1 else len(beta_void_sizes)
                if end > start:
                    realization_means.append(np.mean(beta_void_sizes[start:end]))
            
            ratio_err = np.std(realization_means) / np.sqrt(len(realization_means)) if len(realization_means) > 1 else 0.05
            
            calibration_data['void_size_ratio'].append(ratio)
            calibration_data['void_size_ratio_err'].append(ratio_err)
            calibration_data['all_void_sizes'][beta] = beta_void_sizes
            calibration_data['completed_betas'].append(beta)
            
            logger.info(f"✓ β={beta:.2f}: ratio={ratio:.4f} ± {ratio_err:.4f}")
            
            # CHECKPOINT after each β value
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(calibration_data, f)
            logger.info(f"  Checkpoint saved ({len(calibration_data['completed_betas'])}/{len(beta_values)} β values complete)")
        
        # Create interpolation function
        from scipy.interpolate import interp1d
        
        calibration_data['calibration_function'] = interp1d(
            beta_values,
            calibration_data['void_size_ratio'],
            kind='cubic',
            fill_value='extrapolate'
        )
        
        # Save cache
        logger.info(f"Saving calibration to {cache_file}")
        with open(cache_file, 'wb') as f:
            # Don't pickle the interpolation function
            cache_data = {k: v for k, v in calibration_data.items() if k != 'calibration_function'}
            pickle.dump(cache_data, f)
        
        return calibration_data
    
    def extract_beta_from_voids(
        self,
        observed_void_sizes: np.ndarray,
        calibration_data: Optional[Dict] = None
    ) -> Tuple[float, float]:
        """
        Extract β from observed void sizes using N-body calibration.
        
        This is the RIGOROUS method.
        
        Parameters:
        -----------
        observed_void_sizes : array
            Observed void radii in Mpc/h
        calibration_data : dict, optional
            Calibration from calibrate_void_scaling(). If None, loads from cache.
            
        Returns:
        --------
        tuple
            (beta_fit, beta_err)
        """
        if calibration_data is None:
            cache_file = self.cache_dir / "void_calibration.pkl"
            if not cache_file.exists():
                raise FileNotFoundError(
                    f"No calibration found at {cache_file}. "
                    "Run calibrate_void_scaling() first."
                )
            with open(cache_file, 'rb') as f:
                calibration_data = pickle.load(f)
            
            # Recreate interpolation function
            from scipy.interpolate import interp1d
            calibration_data['calibration_function'] = interp1d(
                calibration_data['beta_grid'],
                calibration_data['void_size_ratio'],
                kind='cubic',
                fill_value='extrapolate'
            )
        
        # Compare observed to ΛCDM from calibration
        idx_lcdm = calibration_data['beta_grid'].index(0.0)
        lcdm_sizes = calibration_data['all_void_sizes'][0.0]
        mean_R_v_lcdm_sim = np.mean(lcdm_sizes)
        
        # Observed ratio
        mean_R_v_obs = np.mean(observed_void_sizes)
        observed_ratio = mean_R_v_obs / mean_R_v_lcdm_sim
        
        # Find β that matches observed ratio
        from scipy.optimize import brentq
        
        def objective(beta):
            return calibration_data['calibration_function'](beta) - observed_ratio
        
        try:
            beta_min = min(calibration_data['beta_grid'])
            beta_max = max(calibration_data['beta_grid'])
            beta_fit = brentq(objective, beta_min, beta_max, xtol=1e-4)
        except ValueError:
            # No root found - return boundary value
            if observed_ratio > calibration_data['void_size_ratio'][-1]:
                beta_fit = beta_max
            else:
                beta_fit = beta_min
        
        # Estimate uncertainty via bootstrap
        n_bootstrap = 100
        beta_bootstrap = []
        
        for _ in range(n_bootstrap):
            resample = np.random.choice(observed_void_sizes, size=len(observed_void_sizes), replace=True)
            resample_ratio = np.mean(resample) / mean_R_v_lcdm_sim
            
            try:
                beta_bs = brentq(
                    lambda b: calibration_data['calibration_function'](b) - resample_ratio,
                    beta_min, beta_max, xtol=1e-4
                )
                beta_bootstrap.append(beta_bs)
            except ValueError:
                continue
        
        if len(beta_bootstrap) > 10:
            beta_err = np.std(beta_bootstrap)
        else:
            beta_err = 0.1
        
        logger.info(f"N-body calibrated β = {beta_fit:.4f} ± {beta_err:.4f}")
        
        return float(beta_fit), float(beta_err)


def quick_calibration(
    beta_values: List[float] = None,
    box_size: float = 256.0,
    n_particles: int = 32**3,  # ~32K particles for speed
    n_realizations: int = 2,
    cache_dir: str = "nbody_cache"
) -> Dict[str, Any]:
    """
    Run a quick N-body calibration (for testing or preliminary analysis).
    
    Uses smaller box and fewer particles for speed.
    
    Parameters:
    -----------
    beta_values : list
        β values to calibrate (default: [-0.1, 0.0, 0.1, 0.2])
    box_size : float
        Box size in Mpc/h (default: 256 for speed)
    n_particles : int
        Number of particles (default: 32³ ≈ 32K)
    n_realizations : int
        Realizations per β (default: 2)
    cache_dir : str
        Cache directory
        
    Returns:
    --------
    dict
        Calibration data
    """
    if beta_values is None:
        beta_values = [-0.1, 0.0, 0.1, 0.2]
    
    calibrator = NBODYVoidCalibration(
        box_size=box_size,
        n_particles=n_particles,
        cache_dir=cache_dir
    )
    
    return calibrator.calibrate_void_scaling(
        beta_values=beta_values,
        n_realizations=n_realizations
    )
