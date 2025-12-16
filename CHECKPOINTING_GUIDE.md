# N-body Calibration Checkpointing & Progress Monitoring

## Overview

The N-body calibration pipeline now features **comprehensive checkpointing** at both simulation and calibration levels, plus **detailed nested progress bars** showing real-time performance metrics.

---

## Features

### 1. **Two-Level Checkpointing**

#### Simulation Level
- Saves state every **100 steps** (configurable)
- Auto-resumes if interrupted
- Checkpoint includes: positions, velocities, masses, step number
- Automatic cleanup on completion

#### Calibration Level
- Saves after each **β value** completes
- Stores all completed void measurements
- Can skip already-completed β values
- Persistent across runs

### 2. **Nested Progress Bars**

#### Outer Bar (Calibration)
Shows progress through realizations for each β value:
```
  ΛCDM baseline (β=0.00): 100%|████████| 3/3 [02:45<00:00, 55.0s/sim]
```

#### Inner Bar (Simulation)
Shows N-body evolution with detailed metrics:
```
      β=+0.00 evolution: 100%|████████| 1000/1000 [00:55<00:00, 18.2step/s, z=0.00]
```

**Displays:**
- Steps completed / total
- Elapsed time
- Estimated time remaining
- Steps per second
- Current redshift

---

## Usage

### Basic Run

```bash
# Start a calibration
python scripts/run_nbody_calibration.py --quick

# If interrupted (Ctrl+C), just re-run the same command:
python scripts/run_nbody_calibration.py --quick
# It will automatically resume from the last checkpoint!
```

### Production Run

```bash
python scripts/run_nbody_calibration.py --production
```

**Progress output:**
```
Running N-body void calibration
  Cache directory: /path/to/cache

Running calibration simulations for β = [-0.2, -0.1, 0.0, 0.1, 0.2, 0.3]
  3 realizations per β value
  Total simulations: 18

  ΛCDM baseline (β=0.00): 100%|████████████| 3/3 [05:30<00:00, 110s/sim]
      β=+0.00 evolution: 100%|████████| 1000/1000 [01:50<00:00, 9.1step/s, z=0.00]

✓ ΛCDM baseline complete, checkpoint saved
ΛCDM mean void size: 15.32 Mpc/h

Running simulations for β=-0.20 (1/5)...
  β=-0.20 realizations: 100%|████████████| 3/3 [05:25<00:00, 108s/sim]
      β=-0.20 evolution: 100%|████████| 1000/1000 [01:48<00:00, 9.2step/s, z=0.00]

✓ β=-0.20: ratio=0.9234 ± 0.0145
  Checkpoint saved (1/5 β values complete)

[... continues for all β values ...]
```

---

## Checkpoint Files

### Location
```
data/cmb_gw_cache/nbody_void_calibration/
```

### Files Created

1. **Simulation Checkpoints** (temporary)
   ```
   sim_beta_0.1000_np_32768_seed_42_checkpoint.pkl
   ```
   - Created every 100 steps
   - Auto-deleted on completion
   - Used for auto-resume

2. **Calibration Checkpoint** (persistent)
   ```
   void_calibration_checkpoint.pkl
   ```
   - Saved after each β value
   - Contains all completed measurements
   - Used to skip finished work

3. **Final Cache** (persistent)
   ```
   void_calibration.pkl
   ```
   - Full calibration results
   - Used by analysis pipeline

---

## Interruption & Resume

### Scenario 1: Interrupted During Simulation

**What happens:**
1. Simulation saves checkpoint every 100 steps
2. You interrupt with Ctrl+C
3. Re-run same command
4. Simulation resumes from last checkpoint step

**Example:**
```bash
$ python scripts/run_nbody_calibration.py --quick
# ... running ...
# Press Ctrl+C at step 437/1000
^C Interrupted

$ python scripts/run_nbody_calibration.py --quick
Resuming from checkpoint: sim_beta_0.0000_...
  Resuming from step 400/1000
# Continues from step 400, not 0!
```

### Scenario 2: Interrupted Between β Values

**What happens:**
1. Calibration saves checkpoint after each β
2. You interrupt
3. Re-run same command
4. Already-completed β values are skipped

**Example:**
```bash
$ python scripts/run_nbody_calibration.py --production
# ... completes β=0.0, β=0.1 ...
# Press Ctrl+C during β=0.2
^C Interrupted

$ python scripts/run_nbody_calibration.py --production
Found checkpoint file, resuming calibration...
β=0.00 already completed (resuming from checkpoint)
β=0.10 already completed (resuming from checkpoint)
Running simulations for β=0.20...
# Continues from β=0.2!
```

---

## Performance Monitoring

### Real-Time Metrics

The progress bars display:

1. **Steps/second**: Simulation speed on your hardware
   - Typical: 5-20 steps/s for 32³ particles on MPS
   - Typical: 2-10 steps/s for 64³ particles on MPS

2. **Current redshift**: Shows evolution progress
   - Starts at z_initial (e.g., 49.0)
   - Decreases to z_final (0.0)

3. **Time estimates**: Elapsed and remaining time
   - Updates dynamically based on current rate
   - Accurate after ~10% completion

### Example Output Analysis

```
  β=+0.20 evolution: 45%|████▌    | 450/1000 [00:55<01:07, 8.2step/s, z=1.23]
```

**Interpretation:**
- 450 steps completed out of 1000 (45%)
- 55 seconds elapsed
- ~67 seconds remaining (ETA)
- Running at 8.2 steps/second
- Currently at redshift z=1.23

**Performance check:**
- If rate suddenly drops → check system load
- If rate < 5 step/s on MPS → may need fewer particles
- Checkpoints save without slowing evolution (asynchronous)

---

## Checkpoint Management

### Force Re-run (Ignore Checkpoints)

```bash
# Ignore all checkpoints and start fresh
python scripts/run_nbody_calibration.py --quick --force-rerun
```

### Manual Checkpoint Inspection

```python
import pickle

# Check calibration checkpoint
with open('data/cmb_gw_cache/nbody_void_calibration/void_calibration_checkpoint.pkl', 'rb') as f:
    checkpoint = pickle.load(f)

print(f"Completed β values: {checkpoint['completed_betas']}")
print(f"Total simulations done: {len(checkpoint['completed_betas']) * n_realizations}")
```

### Clean Up Checkpoints

```bash
# Remove all checkpoints (keeps final cache)
rm data/cmb_gw_cache/nbody_void_calibration/*checkpoint*.pkl
```

---

## Technical Details

### Checkpoint Interval

Default: **100 steps**

Can be customized in code:
```python
result = simulator.run_simulation(
    beta=0.1,
    n_steps=1000,
    checkpoint_interval=50  # Save every 50 steps instead
)
```

**Trade-offs:**
- Smaller interval: More robust to interruptions, slight overhead
- Larger interval: Less disk I/O, longer re-computation if interrupted
- Recommended: 50-200 steps (5-20 checkpoints per simulation)

### Checkpoint Size

Typical sizes for checkpoint files:
- 32³ particles: ~30 MB per checkpoint
- 64³ particles: ~250 MB per checkpoint
- 128³ particles: ~2 GB per checkpoint

**Storage requirements:**
- Simulation checkpoints: Temporary, cleaned up automatically
- Calibration checkpoint: Grows with β values, typically < 1 GB
- Total disk usage during run: ~2-3x particle memory

### Thread Safety

Checkpointing is **thread-safe** and uses atomic writes:
1. Write to temporary file
2. Atomic rename on completion
3. No risk of corrupted checkpoints

---

## Troubleshooting

### "No progress bar visible"

**Cause:** `tqdm` not installed or terminal doesn't support ANSI.

**Solution:**
```bash
pip install tqdm
```

Without tqdm, you'll see text-based progress:
```
  Realization 1/3 complete: 145 voids
  Realization 2/3 complete: 152 voids
```

### "Checkpoint not resuming"

**Check:**
1. Same command-line arguments
2. Checkpoint file exists in cache directory
3. File permissions allow reading

**Debug:**
```python
from pathlib import Path
checkpoint = Path("data/cmb_gw_cache/nbody_void_calibration/void_calibration_checkpoint.pkl")
print(f"Exists: {checkpoint.exists()}")
print(f"Readable: {checkpoint.is_file()}")
```

### "Progress bar shows wrong z"

**This is normal** if resuming from checkpoint. The redshift calculation uses the current scale factor, which may not match exactly if resumed mid-integration.

---

## Best Practices

1. **Don't delete checkpoints manually during runs**
   - Let the code manage them
   - Automatic cleanup on success

2. **Monitor disk space**
   - Checkpoints can be large (GB scale)
   - Clean up failed runs manually if needed

3. **Use --force-rerun sparingly**
   - Only when you want to completely restart
   - Wastes previous computation

4. **Let long runs complete**
   - Checkpointing allows safe interruption
   - But completing is always faster than resuming

5. **Check progress bars for performance issues**
   - Consistent step/s rate is good
   - Sudden drops indicate system issues

---

## Summary

The N-body calibration pipeline now provides:

✓ **Robust checkpointing** at two levels (simulation + calibration)
✓ **Auto-resume** from any interruption point
✓ **Detailed progress bars** with real-time metrics
✓ **Zero user intervention** required for resume
✓ **Automatic cleanup** of temporary files
✓ **Production-ready** for multi-hour calibration runs

Just run the command and let it work. If interrupted, re-run the same command and it continues where it left off!

