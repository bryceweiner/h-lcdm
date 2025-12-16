#!/usr/bin/env python3
"""
Run N-Body Void Calibration for Evolving G
===========================================

This script runs N-body simulations to calibrate the void_size_ratio(β) function.
The calibration is required for RIGOROUS β extraction from void data.

Usage:
------
# Quick calibration (for testing, ~10-30 minutes):
python scripts/run_nbody_calibration.py --quick

# Production calibration (high resolution, several hours):
python scripts/run_nbody_calibration.py --production

# Custom calibration:
python scripts/run_nbody_calibration.py --box-size 512 --n-particles 256 --beta-range -0.2,0.3,7 --n-realizations 5

Requirements:
-------------
- PyTorch: pip install torch (for MPS/CUDA GPU acceleration)
- GPU acceleration recommended (MPS for Apple Silicon, CUDA for NVIDIA)

Output:
-------
Calibration data saved to nbody_cache/void_calibration.pkl
This file is automatically used by the --cmb-gw pipeline for rigorous β extraction.
"""

import argparse
import sys
import logging
from pathlib import Path
import numpy as np

# Add project root to Python path
project_root = Path(__file__).parent.parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Check for custom MPS-accelerated N-body simulator
try:
    from pipeline.cmb_gw.physics.nbody_void_calibration import (
        NBODYVoidCalibration,
        quick_calibration,
        NBODY_PM_AVAILABLE
    )
    if not NBODY_PM_AVAILABLE:
        logger.error("Custom MPS-accelerated N-body simulator not available.")
        logger.error("PyTorch is required. Install with: pip install torch")
        sys.exit(1)
except ImportError as e:
    logger.error(f"Failed to import N-body calibration: {e}")
    logger.error("Make sure you're running from the project root directory")
    sys.exit(1)

# Check for GPU
try:
    import torch
    if torch.backends.mps.is_available():
        logger.info("✓ Apple MPS GPU detected")
    elif torch.cuda.is_available():
        logger.info(f"✓ CUDA GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("No GPU detected - using CPU (slower)")
except ImportError:
    logger.info("PyTorch not available - no GPU acceleration")


def parse_beta_range(beta_str):
    """Parse beta range string like '-0.2,0.3,7' to list of values."""
    if ',' in beta_str:
        parts = beta_str.split(',')
        if len(parts) == 3:
            start, end, n = float(parts[0]), float(parts[1]), int(parts[2])
            return list(np.linspace(start, end, n))
        else:
            return [float(x) for x in parts]
    else:
        raise ValueError("Beta range must be 'start,end,n' or comma-separated list")


def main():
    parser = argparse.ArgumentParser(
        description="Run N-body calibration for void size scaling with evolving G",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    # Preset modes
    parser.add_argument(
        '--quick',
        action='store_true',
        help='Quick calibration (128 Mpc/h box, 128³ particles, ~10-30 min)'
    )
    parser.add_argument(
        '--production',
        action='store_true',
        help='Production calibration (512 Mpc/h box, 256³ particles, several hours)'
    )
    
    # Custom parameters
    parser.add_argument(
        '--box-size',
        type=float,
        help='Box size in Mpc/h (default: 256 for custom, or preset values)'
    )
    parser.add_argument(
        '--n-particles',
        type=int,
        help='Number of particles as integer or N³ (e.g., 128 for 128³). Default: 128 for custom'
    )
    parser.add_argument(
        '--beta-range',
        type=str,
        help='Beta values as "start,end,n" or comma-separated list. Default: -0.2,0.3,6'
    )
    parser.add_argument(
        '--n-realizations',
        type=int,
        default=3,
        help='Number of realizations per β value (default: 3)'
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='nbody_cache',
        help='Directory for caching simulations (default: nbody_cache)'
    )
    parser.add_argument(
        '--force',
        action='store_true',
        help='Force rerun even if cached calibration exists'
    )
    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate calibration plots (requires matplotlib)'
    )
    
    args = parser.parse_args()
    
    # Determine parameters
    if args.quick:
        box_size = 128.0
        n_particles_1d = 32  # 32³ = 33K (fast for PM method)
        n_grid = 64  # Grid resolution (2× particle cube root)
        n_steps = 200  # Reasonable for testing
        beta_values = [-0.1, 0.0, 0.1, 0.2]
        n_realizations = 2
        logger.info("Using QUICK calibration preset")
    elif args.production:
        # SCIENTIFICALLY RIGOROUS PARAMETERS (validated against literature)
        # See docs/NBODY_PARAMETER_VALIDATION.md for full justification
        #
        # Key requirements for void studies (Pisani+ 2015, Nadathur & Hotchkiss 2014):
        # - Particle count: ≥128³ (~2M) for accurate void finding
        # - Force resolution: ≤2 Mpc/h for 10-20 Mpc/h voids
        # - Time resolution: Δa < 0.01 for Leapfrog stability
        # - Realizations: ≥5 for statistical significance
        box_size = 512.0
        n_particles_1d = 128  # 128³ = 2,097,152 particles (production standard)
        n_grid = 256  # Grid resolution (2× particle cube root, 2 Mpc/h cells)
        n_steps = 500  # Adequate time resolution (Δa ≈ 0.002)
        beta_values = list(np.linspace(-0.2, 0.3, 8))
        n_realizations = 5  # Sufficient for σ/√5 ≈ 0.45σ error reduction
        logger.info("Using PRODUCTION calibration preset (scientifically validated)")
    else:
        box_size = args.box_size if args.box_size else 256.0
        n_particles_1d = args.n_particles if args.n_particles else 128
        n_grid = max(64, n_particles_1d * 2)  # Grid should be ≥ 2× particle cube root
        n_steps = 500  # Balanced for custom runs
        beta_values = parse_beta_range(args.beta_range) if args.beta_range else list(np.linspace(-0.2, 0.3, 6))
        n_realizations = args.n_realizations
        logger.info("Using CUSTOM calibration parameters")
    
    n_particles = n_particles_1d**3
    
    # Print configuration
    logger.info("="*80)
    logger.info("N-BODY VOID CALIBRATION FOR EVOLVING G")
    logger.info("="*80)
    logger.info(f"Box size: {box_size} Mpc/h")
    logger.info(f"N particles: {n_particles_1d}³ = {n_particles:,}")
    logger.info(f"Grid resolution: {n_grid}³")
    logger.info(f"Time steps: {n_steps} (z=49→0)")
    logger.info(f"Method: Particle-Mesh (MPS-accelerated)")
    logger.info(f"β values: {beta_values}")
    logger.info(f"Realizations per β: {n_realizations}")
    logger.info(f"Total simulations: {len(beta_values) * n_realizations}")
    logger.info(f"Cache directory: {args.cache_dir}")
    logger.info("="*80)
    
    # Estimate runtime
    # PM method scales as: N (particles) + N_grid log N_grid (FFT) × n_steps
    # Rough estimate: ~0.5-1 minute per simulation for 32³ with 200 steps on MPS
    base_time = 0.5  # minutes for 32³ with 200 steps on MPS GPU
    # Scale with particle count and steps
    particle_scale = (n_particles / (32**3))
    grid_scale = (n_grid**3 * np.log(n_grid**3)) / (64**3 * np.log(64**3))
    step_scale = (n_steps / 200)
    time_per_sim = base_time * particle_scale * grid_scale * step_scale
    total_time = time_per_sim * len(beta_values) * n_realizations
    
    logger.info(f"Estimated runtime: {total_time:.1f} minutes ({total_time/60:.1f} hours)")
    
    # Confirm
    if not args.quick:
        logger.info("Starting in 3 seconds... (Ctrl+C to cancel)")
        import time
        time.sleep(3)
    
    # Create calibrator
    calibrator = NBODYVoidCalibration(
        box_size=box_size,
        n_particles=n_particles,
        n_grid=n_grid,
        cache_dir=args.cache_dir
    )
    
    # Run calibration
    logger.info("Starting N-body calibration...")
    try:
        calibration_data = calibrator.calibrate_void_scaling(
            beta_values=beta_values,
            n_realizations=n_realizations,
            n_steps=n_steps,
            force_rerun=args.force
        )
        
        logger.info("="*80)
        logger.info("CALIBRATION COMPLETE")
        logger.info("="*80)
        logger.info(f"Results saved to: {args.cache_dir}/void_calibration.pkl")
        logger.info("")
        logger.info("Calibration results:")
        for i, (beta, ratio, err) in enumerate(zip(
            calibration_data['beta_grid'],
            calibration_data['void_size_ratio'],
            calibration_data['void_size_ratio_err']
        )):
            logger.info(f"  β = {beta:6.3f}  →  R_v/R_v(ΛCDM) = {ratio:.4f} ± {err:.4f}")
        
        # Generate plots if requested
        if args.plot:
            try:
                import matplotlib.pyplot as plt
                
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
                
                # Plot 1: Void size ratio vs β
                ax1.errorbar(
                    calibration_data['beta_grid'],
                    calibration_data['void_size_ratio'],
                    yerr=calibration_data['void_size_ratio_err'],
                    marker='o', linestyle='-', capsize=5
                )
                ax1.axhline(1.0, color='k', linestyle='--', alpha=0.3, label='ΛCDM')
                ax1.axvline(0.0, color='k', linestyle='--', alpha=0.3)
                ax1.set_xlabel('β (evolving G coupling)', fontsize=12)
                ax1.set_ylabel('R_v / R_v(ΛCDM)', fontsize=12)
                ax1.set_title('N-body Void Size Calibration', fontsize=14, fontweight='bold')
                ax1.grid(True, alpha=0.3)
                ax1.legend()
                
                # Plot 2: Void size distributions for different β
                ax2.set_xlabel('Void radius (Mpc/h)', fontsize=12)
                ax2.set_ylabel('Probability density', fontsize=12)
                ax2.set_title('Void Size Distributions', fontsize=14, fontweight='bold')
                
                colors = plt.cm.viridis(np.linspace(0, 1, len(beta_values)))
                for beta, color in zip(beta_values, colors):
                    if beta in calibration_data['all_void_sizes']:
                        voids = calibration_data['all_void_sizes'][beta]
                        ax2.hist(
                            voids, bins=20, alpha=0.5, density=True,
                            color=color, label=f'β = {beta:.2f}'
                        )
                
                ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                plt.tight_layout()
                
                # Save plot
                plot_file = Path(args.cache_dir) / "calibration_plot.png"
                plt.savefig(plot_file, dpi=150, bbox_inches='tight')
                logger.info(f"Calibration plot saved to: {plot_file}")
                
                # Show if interactive
                # plt.show()
                
            except ImportError:
                logger.warning("matplotlib not available - skipping plots")
            except Exception as e:
                logger.error(f"Failed to generate plots: {e}")
        
        logger.info("="*80)
        logger.info("Calibration complete! Use with --cmb-gw pipeline:")
        logger.info("  python main.py --cmb-gw")
        logger.info("="*80)
        
        return 0
        
    except Exception as e:
        logger.error(f"Calibration failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())

