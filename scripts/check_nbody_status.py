#!/usr/bin/env python3
"""
Check N-Body Calibration Status
================================

Monitor the progress of ongoing N-body void calibration.

Usage:
------
python scripts/check_nbody_status.py
python scripts/check_nbody_status.py --cache-dir nbody_cache
"""

import argparse
from pathlib import Path
import pickle
import time
from datetime import datetime, timedelta

def check_status(cache_dir="nbody_cache"):
    """Check calibration status and display progress."""
    
    cache_path = Path(cache_dir)
    
    print("="*80)
    print("N-BODY CALIBRATION STATUS CHECK")
    print("="*80)
    print()
    
    # Check for checkpoint file
    checkpoint_file = cache_path / "void_calibration_checkpoint.pkl"
    final_file = cache_path / "void_calibration.pkl"
    
    if final_file.exists():
        print("✅ CALIBRATION COMPLETE!")
        print()
        with open(final_file, 'rb') as f:
            data = pickle.load(f)
        
        print(f"Final calibration data:")
        print(f"  β values: {data['beta_grid']}")
        print(f"  Ratios: {[f'{r:.4f}' for r in data['void_size_ratio']]}")
        print()
        print("Ready to use with: python main.py --cmb-gw")
        print("="*80)
        return
    
    if not checkpoint_file.exists():
        print("❌ NO CALIBRATION IN PROGRESS")
        print()
        print("No checkpoint file found at:", checkpoint_file)
        print()
        print("To start calibration:")
        print("  python scripts/run_nbody_calibration.py --quick")
        print("="*80)
        return
    
    # Load checkpoint
    with open(checkpoint_file, 'rb') as f:
        data = pickle.load(f)
    
    print("⏳ CALIBRATION IN PROGRESS")
    print()
    
    # Get checkpoint modification time
    checkpoint_mtime = checkpoint_file.stat().st_mtime
    checkpoint_time = datetime.fromtimestamp(checkpoint_mtime)
    time_since_update = time.time() - checkpoint_mtime
    
    print(f"Last checkpoint: {checkpoint_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Time since update: {timedelta(seconds=int(time_since_update))}")
    print()
    
    # Progress
    total_betas = len(data['beta_grid'])
    completed = len(data.get('completed_betas', []))
    
    print(f"Progress: {completed}/{total_betas} β values complete ({100*completed/total_betas:.0f}%)")
    print()
    
    if completed > 0:
        print("Completed β values:")
        for beta in data['completed_betas']:
            if beta in data['all_void_sizes']:
                n_voids = len(data['all_void_sizes'][beta])
                print(f"  ✓ β = {beta:6.2f}  ({n_voids} voids)")
        print()
    
    if completed < total_betas:
        remaining = total_betas - completed
        print(f"Remaining: {remaining} β values")
        for beta in data['beta_grid']:
            if beta not in data.get('completed_betas', []):
                print(f"  ⏳ β = {beta:6.2f}")
        print()
    
    # Check individual simulation cache
    print("Cached simulations:")
    sim_files = list(cache_path.glob("sim_beta_*.pkl"))
    if sim_files:
        print(f"  {len(sim_files)} simulation snapshots cached")
        # Group by beta
        beta_counts = {}
        for f in sim_files:
            # Parse filename: sim_beta_{beta:.4f}_np_{n}_seed_{seed}.pkl
            try:
                parts = f.stem.split('_')
                beta_val = float(parts[2])
                beta_counts[beta_val] = beta_counts.get(beta_val, 0) + 1
            except:
                pass
        
        for beta, count in sorted(beta_counts.items()):
            print(f"    β = {beta:6.2f}: {count} cached")
    else:
        print("  No cached simulations found")
    
    print()
    
    # Time estimate
    if completed > 0 and time_since_update < 3600:  # Active in last hour
        # Estimate time per beta
        # This is rough - actual time varies
        if time_since_update > 60:  # At least 1 minute since last update
            time_per_beta = time_since_update / max(1, completed - len(data.get('completed_betas', [])[:-1]))
            estimated_remaining = time_per_beta * remaining
            
            print(f"Estimated time remaining: {timedelta(seconds=int(estimated_remaining))}")
            print(f"  (Based on recent progress, rough estimate)")
    elif time_since_update > 7200:  # > 2 hours
        print("⚠️  WARNING: No progress in > 2 hours")
        print("   Calibration may have stalled. Check the process.")
    
    print()
    print("="*80)
    print()
    print("To monitor in real-time, run:")
    print("  watch -n 10 python scripts/check_nbody_status.py")


def main():
    parser = argparse.ArgumentParser(
        description="Check N-body calibration status"
    )
    parser.add_argument(
        '--cache-dir',
        type=str,
        default='nbody_cache',
        help='Cache directory to check (default: nbody_cache)'
    )
    
    args = parser.parse_args()
    check_status(args.cache_dir)


if __name__ == "__main__":
    main()

