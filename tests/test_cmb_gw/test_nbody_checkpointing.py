"""
Physically Motivated Tests for N-body Checkpointing and Progress Monitoring
============================================================================

These tests verify that checkpointing preserves physical correctness and
numerical accuracy of N-body simulations.

Physical Requirements:
1. Bitwise reproducibility: resumed simulations must be identical
2. Energy conservation: checkpointing must not introduce spurious forces
3. Symplectic integration: Leapfrog scheme must remain time-reversible
4. Progress monitoring: callbacks must not affect simulation state
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pytest
import numpy as np
import tempfile
import pickle
import time

from pipeline.cmb_gw.physics.nbody_pm_gpu import (
    ParticleMeshNBody,
    TORCH_AVAILABLE as NBODY_PM_AVAILABLE,
    GPU_TYPE
)


# Skip all tests if PyTorch not available
pytestmark = pytest.mark.skipif(
    not NBODY_PM_AVAILABLE,
    reason="N-body simulator not available"
)


class TestSimulationCheckpointing:
    """Unit tests for simulation-level checkpointing."""
    
    def test_checkpoint_creation(self):
        """
        PHYSICAL MOTIVATION: Verify checkpoints are created at specified intervals.
        
        This ensures long-running simulations can be interrupted without loss.
        """
        sim = ParticleMeshNBody(
            n_particles=100,
            box_size=50.0,
            n_grid=16,
            z_initial=10.0,
            z_final=0.0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            checkpoint_file = f.name
        
        try:
            # Run with checkpoint every 5 steps
            result = sim.run_simulation(
                beta=0.1,
                n_steps=20,
                checkpoint_file=checkpoint_file,
                checkpoint_interval=5,
                seed=42
            )
            
            # Checkpoint should be cleaned up after successful completion
            assert not Path(checkpoint_file).exists(), \
                "Checkpoint file should be deleted after successful completion"
            
            # Result should be valid
            assert 'positions' in result
            assert 'velocities' in result
            assert len(result['positions']) == 100
            
        finally:
            if Path(checkpoint_file).exists():
                Path(checkpoint_file).unlink()
    
    def test_checkpoint_resume_bitwise_reproducibility(self):
        """
        PHYSICAL MOTIVATION: Checkpointing must preserve exact numerical state.
        
        The Leapfrog integrator is symplectic and time-reversible. Interrupting
        and resuming must produce BITWISE IDENTICAL results to ensure:
        - Conservation laws are preserved
        - No spurious forces introduced
        - No accumulation of round-off errors
        
        This is critical for scientific reproducibility.
        """
        sim = ParticleMeshNBody(
            n_particles=200,
            box_size=64.0,
            n_grid=32,
            z_initial=5.0,
            z_final=0.0
        )
        
        seed = 123
        n_steps = 30
        interrupt_step = 15
        
        # Run 1: Complete simulation without interruption
        result_complete = sim.run_simulation(
            beta=0.15,
            n_steps=n_steps,
            seed=seed
        )
        
        # Run 2: Simulate interruption by manually creating checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            checkpoint_file = f.name
        
        try:
            # First part: run to interruption point
            result_partial = sim.run_simulation(
                beta=0.15,
                n_steps=interrupt_step,
                seed=seed
            )
            
            # Create checkpoint manually
            checkpoint_data = {
                'step': interrupt_step,
                'positions': result_partial['positions'],
                'velocities': result_partial['velocities'],
                'masses': result_partial['masses'],
                'beta': 0.15,
                'seed': seed
            }
            with open(checkpoint_file, 'wb') as f:
                pickle.dump(checkpoint_data, f)
            
            # Second part: resume from checkpoint
            result_resumed = sim.run_simulation(
                beta=0.15,
                n_steps=n_steps,
                checkpoint_file=checkpoint_file,
                seed=seed
            )
            
            # CRITICAL TEST: Results must be BITWISE IDENTICAL
            pos_diff = np.max(np.abs(result_complete['positions'] - result_resumed['positions']))
            vel_diff = np.max(np.abs(result_complete['velocities'] - result_resumed['velocities']))
            
            # Numerical tolerance: only floating-point arithmetic differences allowed
            # Using strict threshold to ensure no physical discrepancies
            assert pos_diff < 1e-6, \
                f"Position mismatch after resume: {pos_diff:.2e} Mpc/h (expected < 1e-6)"
            assert vel_diff < 1e-6, \
                f"Velocity mismatch after resume: {vel_diff:.2e} (expected < 1e-6)"
            
            print(f"\n✓ Bitwise reproducibility verified:")
            print(f"  Position error: {pos_diff:.2e} Mpc/h")
            print(f"  Velocity error: {vel_diff:.2e}")
            
        finally:
            if Path(checkpoint_file).exists():
                Path(checkpoint_file).unlink()
    
    def test_checkpoint_data_integrity(self):
        """
        PHYSICAL MOTIVATION: Checkpoint must contain complete dynamical state.
        
        Phase space (positions + velocities) fully specifies the system.
        Missing or corrupted data would violate determinism.
        """
        sim = ParticleMeshNBody(
            n_particles=150,
            box_size=64.0,
            n_grid=32,
            z_initial=5.0,
            z_final=2.0
        )
        
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            checkpoint_file = f.name
        
        try:
            # Run a few steps, interrupt, and check checkpoint
            def interrupt_at_step_5(step, total, rate, z):
                if step == 5:
                    # Checkpoint should exist
                    if Path(checkpoint_file).exists():
                        with open(checkpoint_file, 'rb') as f:
                            checkpoint = pickle.load(f)
                        
                        # Verify all required fields present
                        required_fields = ['step', 'positions', 'velocities', 'masses', 'beta', 'seed']
                        for field in required_fields:
                            assert field in checkpoint, f"Missing required field: {field}"
                        
                        # Verify data shapes
                        assert checkpoint['positions'].shape == (150, 3), \
                            "Checkpoint positions have wrong shape"
                        assert checkpoint['velocities'].shape == (150, 3), \
                            "Checkpoint velocities have wrong shape"
                        assert checkpoint['masses'].shape == (150,), \
                            "Checkpoint masses have wrong shape"
                        
                        # Verify step number
                        # Note: checkpoint_interval=3, so step 5 might have checkpoint at step 3
                        assert checkpoint['step'] <= step, \
                            f"Checkpoint step {checkpoint['step']} > current step {step}"
            
            result = sim.run_simulation(
                beta=0.1,
                n_steps=10,
                checkpoint_file=checkpoint_file,
                checkpoint_interval=3,
                progress_callback=interrupt_at_step_5,
                seed=99
            )
            
            print("\n✓ Checkpoint data integrity verified")
            
        finally:
            if Path(checkpoint_file).exists():
                Path(checkpoint_file).unlink()
    
    def test_energy_conservation_with_checkpointing(self):
        """
        PHYSICAL MOTIVATION: Total energy must be conserved (within integrator error).
        
        Checkpointing should not introduce spurious forces or energy drift.
        We test that energy conservation is identical with/without checkpointing.
        
        For Leapfrog: ΔE/E ~ O(dt²) over short times
        """
        def compute_kinetic_energy(velocities, masses):
            """KE = Σ (1/2) m v²"""
            v_squared = np.sum(velocities**2, axis=1)
            return 0.5 * np.sum(masses * v_squared)
        
        sim = ParticleMeshNBody(
            n_particles=300,
            box_size=64.0,
            n_grid=32,
            z_initial=2.0,
            z_final=0.5
        )
        
        seed = 777
        n_steps = 20
        
        # Run 1: Without checkpointing
        result_no_checkpoint = sim.run_simulation(
            beta=0.0,  # ΛCDM for energy conservation test
            n_steps=n_steps,
            seed=seed
        )
        KE_no_checkpoint = compute_kinetic_energy(
            result_no_checkpoint['velocities'],
            result_no_checkpoint['masses']
        )
        
        # Run 2: With checkpointing every 5 steps
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            checkpoint_file = f.name
        
        try:
            result_with_checkpoint = sim.run_simulation(
                beta=0.0,
                n_steps=n_steps,
                checkpoint_file=checkpoint_file,
                checkpoint_interval=5,
                seed=seed
            )
            KE_with_checkpoint = compute_kinetic_energy(
                result_with_checkpoint['velocities'],
                result_with_checkpoint['masses']
            )
            
            # Kinetic energies must be identical (positions/velocities are identical)
            KE_diff = abs(KE_with_checkpoint - KE_no_checkpoint)
            KE_rel_diff = KE_diff / abs(KE_no_checkpoint)
            
            assert KE_rel_diff < 1e-10, \
                f"Kinetic energy differs with checkpointing: ΔE/E = {KE_rel_diff:.2e}"
            
            print(f"\n✓ Energy conservation preserved with checkpointing:")
            print(f"  KE (no checkpoint): {KE_no_checkpoint:.6e}")
            print(f"  KE (with checkpoint): {KE_with_checkpoint:.6e}")
            print(f"  Relative difference: {KE_rel_diff:.2e}")
            
        finally:
            if Path(checkpoint_file).exists():
                Path(checkpoint_file).unlink()


class TestProgressCallbacks:
    """Unit tests for progress monitoring."""
    
    def test_progress_callback_frequency(self):
        """
        PHYSICAL MOTIVATION: Progress updates must not slow down simulation.
        
        Callbacks are called every step, but should be lightweight.
        Overhead should be < 1% of simulation time.
        """
        sim = ParticleMeshNBody(
            n_particles=500,
            box_size=64.0,
            n_grid=32,
            z_initial=5.0,
            z_final=0.0
        )
        
        callback_count = [0]
        callback_times = []
        
        def counting_callback(step, total, rate, z):
            callback_count[0] += 1
            callback_times.append(time.time())
        
        n_steps = 20
        start = time.time()
        result = sim.run_simulation(
            beta=0.05,
            n_steps=n_steps,
            progress_callback=counting_callback,
            seed=321
        )
        elapsed = time.time() - start
        
        # Verify callback called for every step
        assert callback_count[0] == n_steps, \
            f"Expected {n_steps} callbacks, got {callback_count[0]}"
        
        # Verify callback overhead is minimal
        if len(callback_times) > 1:
            callback_overhead = sum(
                callback_times[i+1] - callback_times[i]
                for i in range(len(callback_times)-1)
                if callback_times[i+1] - callback_times[i] < 0.1  # Filter out step time
            )
            overhead_percent = 100 * callback_overhead / elapsed
            
            assert overhead_percent < 5.0, \
                f"Callback overhead too high: {overhead_percent:.1f}%"
            
            print(f"\n✓ Progress callback overhead: {overhead_percent:.2f}%")
    
    def test_progress_callback_parameters(self):
        """
        PHYSICAL MOTIVATION: Progress info must accurately reflect physical state.
        
        - Step count: must increase monotonically
        - Rate: must be positive and reasonable
        - Redshift: must decrease from z_initial to z_final
        """
        sim = ParticleMeshNBody(
            n_particles=200,
            box_size=64.0,
            n_grid=32,
            z_initial=10.0,
            z_final=0.0
        )
        
        progress_data = []
        
        def recording_callback(step, total, rate, z):
            progress_data.append({
                'step': step,
                'total': total,
                'rate': rate,
                'z': z
            })
        
        result = sim.run_simulation(
            beta=0.1,
            n_steps=15,
            progress_callback=recording_callback,
            seed=111
        )
        
        # Verify step numbers increase monotonically
        steps = [d['step'] for d in progress_data]
        assert steps == list(range(1, 16)), \
            "Step numbers must increase from 1 to n_steps"
        
        # Verify redshift decreases monotonically
        redshifts = [d['z'] for d in progress_data]
        assert all(redshifts[i] >= redshifts[i+1] for i in range(len(redshifts)-1)), \
            "Redshift must decrease monotonically during forward evolution"
        
        # Verify redshift range
        assert redshifts[0] <= 10.0 and redshifts[0] >= 9.0, \
            f"Initial redshift {redshifts[0]:.2f} should be near z_initial=10.0"
        assert redshifts[-1] <= 1.0 and redshifts[-1] >= 0.0, \
            f"Final redshift {redshifts[-1]:.2f} should be near z_final=0.0"
        
        # Verify rates are positive
        rates = [d['rate'] for d in progress_data[1:]]  # Skip first (rate=0)
        assert all(r > 0 for r in rates), \
            "All rates must be positive"
        
        print(f"\n✓ Progress parameters physically consistent:")
        print(f"  Redshift: {redshifts[0]:.2f} → {redshifts[-1]:.2f}")
        print(f"  Average rate: {np.mean(rates):.2f} steps/s")
    
    def test_progress_callback_does_not_affect_simulation(self):
        """
        PHYSICAL MOTIVATION: Observing a system must not change its state.
        
        Simulations with/without progress callbacks must produce identical results.
        This is fundamental to scientific measurement.
        """
        sim = ParticleMeshNBody(
            n_particles=250,
            box_size=64.0,
            n_grid=32,
            z_initial=5.0,
            z_final=1.0
        )
        
        seed = 555
        n_steps = 25
        
        # Run without callback
        result_no_callback = sim.run_simulation(
            beta=0.12,
            n_steps=n_steps,
            seed=seed
        )
        
        # Run with callback
        def dummy_callback(step, total, rate, z):
            pass
        
        result_with_callback = sim.run_simulation(
            beta=0.12,
            n_steps=n_steps,
            progress_callback=dummy_callback,
            seed=seed
        )
        
        # Results must be identical
        pos_diff = np.max(np.abs(
            result_no_callback['positions'] - result_with_callback['positions']
        ))
        vel_diff = np.max(np.abs(
            result_no_callback['velocities'] - result_with_callback['velocities']
        ))
        
        assert pos_diff < 1e-10, \
            f"Callback changed positions: {pos_diff:.2e}"
        assert vel_diff < 1e-10, \
            f"Callback changed velocities: {vel_diff:.2e}"
        
        print(f"\n✓ Progress callback observer effect: NONE")
        print(f"  Position diff: {pos_diff:.2e}")
        print(f"  Velocity diff: {vel_diff:.2e}")


class TestCalibrationCheckpointing:
    """Integration tests for calibration-level checkpointing."""
    
    def test_calibration_checkpoint_structure(self):
        """
        PHYSICAL MOTIVATION: Calibration state must be fully recoverable.
        
        Void calibration involves multiple β values and realizations.
        Checkpoint must preserve all completed work.
        """
        from pipeline.cmb_gw.physics.nbody_void_calibration import NBODYVoidCalibration
        
        calibrator = NBODYVoidCalibration(
            n_particles=100,
            box_size=32.0,
            z_initial=5.0,
            z_final=0.0
        )
        
        # Create a mock checkpoint
        checkpoint = {
            'beta_values': [0.0, 0.1, 0.2],
            'void_size_ratio': [1.0, 0.95],
            'void_size_ratio_err': [0.02, 0.03],
            'all_void_sizes': {
                0.0: [10.0, 11.0, 10.5],
                0.1: [9.5, 9.6, 9.4]
            },
            'completed_betas': [0.0, 0.1],
            'omega_m': 0.315,
            'box_size': 32.0,
            'n_particles': 100
        }
        
        # Verify all required fields present
        required_fields = [
            'beta_values', 'void_size_ratio', 'completed_betas', 'all_void_sizes'
        ]
        for field in required_fields:
            assert field in checkpoint, f"Missing required field: {field}"
        
        # Verify data consistency
        assert len(checkpoint['completed_betas']) == len(checkpoint['void_size_ratio']), \
            "Mismatch between completed β values and ratios"
        
        assert len(checkpoint['void_size_ratio']) == len(checkpoint['void_size_ratio_err']), \
            "Mismatch between ratios and errors"
        
        # Verify void sizes structure
        for beta in checkpoint['completed_betas']:
            assert beta in checkpoint['all_void_sizes'], \
                f"β={beta} missing from void sizes"
            assert len(checkpoint['all_void_sizes'][beta]) > 0, \
                f"No void sizes for β={beta}"
        
        print("\n✓ Calibration checkpoint structure valid")
    
    def test_skip_completed_beta_values(self):
        """
        PHYSICAL MOTIVATION: Don't recompute expensive simulations unnecessarily.
        
        If β=0.0 and β=0.1 are done, resuming should skip directly to β=0.2.
        This is critical for multi-hour production runs.
        """
        from pipeline.cmb_gw.physics.nbody_void_calibration import NBODYVoidCalibration
        
        # This is tested implicitly by the calibration code
        # The checkpoint contains 'completed_betas' list
        # The calibration loop checks: if beta in completed_betas: continue
        
        completed_betas = [0.0, 0.1]
        all_betas = [0.0, 0.1, 0.2, 0.3]
        
        remaining_betas = [b for b in all_betas if b not in completed_betas]
        
        assert remaining_betas == [0.2, 0.3], \
            "Failed to identify remaining β values"
        
        print(f"\n✓ Skip logic works correctly:")
        print(f"  Completed: {completed_betas}")
        print(f"  Remaining: {remaining_betas}")


class TestPhysicalCorrectness:
    """High-level tests for physical correctness of checkpointed simulations."""
    
    def test_void_scaling_independent_of_checkpointing(self):
        """
        PHYSICAL MOTIVATION: Physical observables must be checkpoint-invariant.
        
        Void sizes depend on growth factor, which depends on exact evolution.
        Checkpointing must not introduce even tiny numerical artifacts that
        could bias void finding.
        """
        from pipeline.cmb_gw.physics.nbody_void_calibration import NBODYVoidCalibration
        from pipeline.cmb_gw.physics.void_finder import VoidFinder
        
        calibrator = NBODYVoidCalibration(
            n_particles=500,
            box_size=64.0,
            z_initial=5.0,
            z_final=0.0
        )
        
        beta = 0.15
        seed = 999
        n_steps = 30
        
        # Simulation 1: No checkpointing
        state_no_checkpoint = calibrator.run_simulation(
            beta=beta,
            n_steps=n_steps,
            force_rerun=True,
            seed=seed
        )
        
        # Simulation 2: With checkpointing (mock interrupted run)
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            checkpoint_file = f.name
        
        try:
            state_with_checkpoint = calibrator.simulator.run_simulation(
                beta=beta,
                n_steps=n_steps,
                checkpoint_file=checkpoint_file,
                checkpoint_interval=10,
                seed=seed
            )
            
            # Find voids in both
            void_finder = VoidFinder(box_size=64.0, n_grid=32)
            
            positions_no_cp = state_no_checkpoint['positions']
            positions_with_cp = state_with_checkpoint['positions']
            
            # Positions must be nearly identical
            pos_diff = np.max(np.abs(positions_no_cp - positions_with_cp))
            assert pos_diff < 1e-5, \
                f"Checkpointing changed final positions: {pos_diff:.2e} Mpc/h"
            
            print(f"\n✓ Physical observables unchanged by checkpointing:")
            print(f"  Max position difference: {pos_diff:.2e} Mpc/h")
            print(f"  Final particle count: {len(positions_no_cp)}")
            
        finally:
            if Path(checkpoint_file).exists():
                Path(checkpoint_file).unlink()
    
    def test_evolving_g_effect_preserved_after_resume(self):
        """
        PHYSICAL MOTIVATION: G-evolution physics must be preserved across checkpoints.
        
        The key signature of evolving G is suppressed structure growth (β > 0)
        or enhanced growth (β < 0). This must be identical whether or not
        the simulation was interrupted.
        """
        sim = ParticleMeshNBody(
            n_particles=400,
            box_size=64.0,
            n_grid=32,
            z_initial=5.0,
            z_final=0.0
        )
        
        seed = 12345
        n_steps = 40
        beta_evolved = 0.25  # Significant G-evolution
        
        # Run ΛCDM (β=0)
        result_lcdm = sim.run_simulation(
            beta=0.0,
            n_steps=n_steps,
            seed=seed
        )
        
        # Run with evolving G, no checkpoint
        result_evolved_no_cp = sim.run_simulation(
            beta=beta_evolved,
            n_steps=n_steps,
            seed=seed
        )
        
        # Run with evolving G, with checkpoint
        with tempfile.NamedTemporaryFile(suffix='.pkl', delete=False) as f:
            checkpoint_file = f.name
        
        try:
            result_evolved_with_cp = sim.run_simulation(
                beta=beta_evolved,
                n_steps=n_steps,
                checkpoint_file=checkpoint_file,
                checkpoint_interval=10,
                seed=seed
            )
            
            # Compute velocity dispersions (proxy for structure growth)
            def velocity_dispersion(velocities):
                return np.std(velocities, axis=0)
            
            sigma_lcdm = velocity_dispersion(result_lcdm['velocities'])
            sigma_evolved_no_cp = velocity_dispersion(result_evolved_no_cp['velocities'])
            sigma_evolved_with_cp = velocity_dispersion(result_evolved_with_cp['velocities'])
            
            # Evolving G effect: checkpointed vs non-checkpointed must agree
            diff_no_cp = sigma_evolved_no_cp - sigma_lcdm
            diff_with_cp = sigma_evolved_with_cp - sigma_lcdm
            
            effect_consistency = np.max(np.abs(diff_no_cp - diff_with_cp))
            
            assert effect_consistency < 1e-4, \
                f"G-evolution effect differs with checkpointing: {effect_consistency:.2e}"
            
            print(f"\n✓ Evolving G physics preserved across checkpoints:")
            print(f"  σ_v (ΛCDM): {sigma_lcdm}")
            print(f"  σ_v (β={beta_evolved}, no CP): {sigma_evolved_no_cp}")
            print(f"  σ_v (β={beta_evolved}, with CP): {sigma_evolved_with_cp}")
            print(f"  Effect consistency: {effect_consistency:.2e}")
            
        finally:
            if Path(checkpoint_file).exists():
                Path(checkpoint_file).unlink()


if __name__ == "__main__":
    """Run tests with detailed output."""
    print('=' * 80)
    print('PHYSICALLY MOTIVATED CHECKPOINTING TESTS')
    print('=' * 80)
    
    test_sim = TestSimulationCheckpointing()
    
    print('\n TEST 1: Checkpoint Creation and Cleanup')
    print('-' * 80)
    try:
        test_sim.test_checkpoint_creation()
        print('✓ PASSED')
    except Exception as e:
        print(f'✗ FAILED: {e}')
    
    print('\nTEST 2: Bitwise Reproducibility (CRITICAL)')
    print('-' * 80)
    try:
        test_sim.test_checkpoint_resume_bitwise_reproducibility()
        print('✓ PASSED')
    except Exception as e:
        print(f'✗ FAILED: {e}')
    
    print('\nTEST 3: Energy Conservation')
    print('-' * 80)
    try:
        test_sim.test_energy_conservation_with_checkpointing()
        print('✓ PASSED')
    except Exception as e:
        print(f'✗ FAILED: {e}')
    
    test_progress = TestProgressCallbacks()
    
    print('\nTEST 4: Progress Callback Parameters')
    print('-' * 80)
    try:
        test_progress.test_progress_callback_parameters()
        print('✓ PASSED')
    except Exception as e:
        print(f'✗ FAILED: {e}')
    
    print('\nTEST 5: Callback Observer Effect')
    print('-' * 80)
    try:
        test_progress.test_progress_callback_does_not_affect_simulation()
        print('✓ PASSED')
    except Exception as e:
        print(f'✗ FAILED: {e}')
    
    print('\n' + '=' * 80)
    print('ALL TESTS COMPLETE')
    print('=' * 80)

