"""One-shot builder for `notebooks/trgb_comparative_reproduction.ipynb`.

Run from repo root:

    python scripts/_build_trgb_comparative_notebook.py

This script constructs the notebook JSON in-place and overwrites the
existing file. The notebook reproduces the full 12-chain matrix +
Uddin positive control + framework forward predictions documented in
``results/trgb_comparative/reports/trgb_comparative_analysis_report.md``.
"""

from __future__ import annotations

import json
from pathlib import Path
from textwrap import dedent


def md(source: str) -> dict:
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": _split(source),
    }


def code(source: str) -> dict:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": _split(source),
    }


def _split(s: str) -> list[str]:
    s = dedent(s).strip("\n")
    lines = s.split("\n")
    return [(line + "\n") if i < len(lines) - 1 else line for i, line in enumerate(lines)]


CELLS: list[dict] = []


CELLS.append(md("""
# TRGB Comparative Analysis — Reproduction Notebook

This notebook reproduces the central numerical results in
`results/trgb_comparative/reports/trgb_comparative_analysis_report.md`
end-to-end on public archival data. A reviewer running every cell
should obtain matrix values matching the report to within MCMC noise
(typically < 0.1 km/s/Mpc on H₀ medians at full chain length).

What is reproduced:

1. The **12-chain matrix** (4 SN photometric systems × 3 calibrator-sample
   variants: Case A 18-host F2019 / Case B 24-SN F2025 augmented /
   Case B 11-SN F2025 JWST-only).
2. The **Uddin 2023 positive-control test** (target H₀ = 70.242 km/s/Mpc,
   tolerance |Δ| ≤ 1.0, R̂ < 1.01) — this validates the 8-parameter
   SNooPy likelihood implementation. **Required to pass before any CSP
   chain result is trusted.**
3. **Framework forward predictions** for both d_local cases
   (LMC = 0.0496 Mpc; NGC 4258 = 7.58 Mpc), with the perturbative-
   breakdown flag surfaced for the LMC anchor.
4. The **cross-case shift** per SN system (Δ(B aug − A) and Δ(B JWST − A)).
5. **Framework-vs-MCMC comparison** with explicit literature-citation vs
   MCMC-posterior labelling.

The Freedman reproduction cells run with no H-ΛCDM framework knowledge;
they stand as independent observational cosmology. The framework cells
operate on the same data via forward prediction only.

**Preregistration**: all methodological choices are frozen in
`trgb_data/prereg/trgb_comparative_preregistration_stage{1,2}.md`.
The main pipeline regenerates them at the start of each run; this
notebook does not change any of them.

**Data and chain locations**:
- Catalog CSVs: `trgb_data/catalogs/`
- Raw downloaded photometry: `trgb_data/downloaded_data/`
- MCMC chain outputs: `trgb_data/chains/`
- 12-chain reproduction matrix CSV: `results/12_chain_matrix.csv`
- Manuscript figures: `figures/manuscript/`
"""))


CELLS.append(md("## 0. Setup"))


CELLS.append(code("""
import os
import sys
import time
from pathlib import Path

repo_root = Path.cwd()
while repo_root != repo_root.parent and not (repo_root / 'main.py').exists():
    repo_root = repo_root.parent
sys.path.insert(0, str(repo_root))
# DataLoader and the catalog paths are resolved relative to cwd, so the
# notebook must run with cwd set to the repo root regardless of where
# the kernel was launched.
os.chdir(repo_root)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

print('repo_root =', repo_root)
print('cwd       =', os.getcwd())
"""))


CELLS.append(md("""
### 0.1 Data availability check

The 12-chain matrix needs four photometric systems' worth of SN data
plus three calibrator-sample tables and both anchor distance constants.
Below we check every prerequisite. If any are missing, run

    python main.py --trgb-comparative-load-data

from the repo root and re-execute the notebook.
"""))


CELLS.append(code("""
from data.loader import DataLoader

loader = DataLoader()
availability = loader.check_data_availability()

REQUIRED_FOR_12_CHAIN_MATRIX = [
    # CSP-I + CSP-II flow + calibrator block (Uddin 8-parameter likelihood)
    'uddin_h0csp_flow',
    'uddin_h0csp_calibrators_trgb_f19',
    # Pantheon 2018 SN sample (SuperCal flow)
    'pantheon_2018',
    # Pantheon+SH0ES SN sample with covariance
    'pantheon_plus',
    # Freedman 2019 Table 3 — Case A 18-host LMC-anchored calibrators
    'freedman_2019_table3',
    'freedman_2019_table1',
    # Freedman 2025 Table 2 — Case B JWST-only 11-SN calibrators
    'freedman_2025_table2',
    # Freedman 2025 Table 3 — Case B augmented 24-SN HST+JWST calibrators
    'freedman_2025_table3',
    # Hoyt 2025 Tables 6/7 — literature reference H₀ per SN system
    'hoyt_2025_tables',
]

print('Required dataset prerequisites for the 12-chain matrix:')
print('-' * 64)
missing = []
for k in REQUIRED_FOR_12_CHAIN_MATRIX:
    ok = availability.get(k, False)
    marker = '✓' if ok else '✗'
    print(f'  {marker} {k}')
    if not ok:
        missing.append(k)

# Preregistration documents (Stage 1 + Stage 2 must exist).
docs_dir = repo_root / 'docs'
prereg_paths = [
    docs_dir / 'trgb_comparative_preregistration_stage1.md',
    docs_dir / 'trgb_comparative_preregistration_stage2.md',
]
print()
print('Preregistration documents:')
print('-' * 64)
for p in prereg_paths:
    marker = '✓' if p.exists() else '✗'
    print(f'  {marker} {p.relative_to(repo_root)}')
    if not p.exists():
        missing.append(p.name)

# Anchor distance constants — published distances baked into the pipeline.
print()
print('Anchor distance constants used by the framework predictor:')
print('  d_local(LMC)       = 0.0496 ± 0.0009 Mpc  (Pietrzynski et al. 2019)')
print('  d_local(NGC 4258)  = 7.58   ± 0.08   Mpc  (Reid et al. 2019)')

if missing:
    print()
    print('!!! Some prerequisites are missing. Run')
    print('        python main.py --trgb-comparative-load-data')
    print('    from the repo root, then re-run this notebook from the top.')
    raise SystemExit('TRGB comparative prerequisites unavailable; halting.')
else:
    print()
    print('All 12-chain prerequisites available. Continue.')
"""))


CELLS.append(md("""
## 1. Framework forward predictions

The framework's holographic projection formula is a pure forward
prediction given d_local. It needs no photometric data. We compute
predictions for both d_local values with Monte Carlo propagation over
the Planck 2018 H_CMB posterior.

The LMC anchor (d_local ≈ 0.05 Mpc) sits in the perturbative-breakdown
regime — the formula's expansion in γ/H · ln(d_CMB/d_local) does not
hold there, so its prediction in that case is flagged as unreliable
and shown for context only.
"""))


CELLS.append(code("""
from pipeline.trgb_comparative.framework_methodology import FrameworkMethodology

fw = FrameworkMethodology()

framework_a = fw.predict(
    label='H0_framework_predicted_lmc_anchor',
    d_local_mpc=0.0496, sigma_d_local_mpc=0.0009,
    n_samples=50_000, seed=42,
)
framework_b = fw.predict(
    label='H0_framework_predicted_ngc4258_anchor',
    d_local_mpc=7.58, sigma_d_local_mpc=0.08,
    n_samples=50_000, seed=43,
)

print(f'Case A (LMC anchor, d_local = 0.0496 Mpc):')
print(f'  H₀ = {framework_a.H0_median:.3f} km/s/Mpc  '
      f'68% CI [{framework_a.H0_low:.3f}, {framework_a.H0_high:.3f}]')
print(f'  perturbative-breakdown fraction: {framework_a.breakdown_fraction:.2f}')
if framework_a.breakdown_fraction > 0.5:
    print('  >>> PERTURBATIVE BREAKDOWN <<<')
    print(f'  >>> {framework_a.breakdown_messages[0]}')
    print('  >>> Framework prediction in this regime is NOT RELIABLE.')

print()
print(f'Case B (NGC 4258 anchor, d_local = 7.58 Mpc):')
print(f'  H₀ = {framework_b.H0_median:.3f} km/s/Mpc  '
      f'68% CI [{framework_b.H0_low:.3f}, {framework_b.H0_high:.3f}]')
print(f'  perturbative-breakdown fraction: {framework_b.breakdown_fraction:.2f}')
print('  Perturbative regime — formula prediction is reliable here.')
"""))


CELLS.append(code("""
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.hist(framework_a.H0_samples, bins=60, alpha=0.55,
        label='Framework Case A (LMC, BREAKDOWN)', color='#F18F01')
ax.hist(framework_b.H0_samples, bins=60, alpha=0.55,
        label='Framework Case B (NGC 4258)', color='#2E86AB')

# Reference H₀ values (literature citations, NOT pipeline MCMC posteriors).
ref_lines = [
    (73.04, '#8B5CF6', 'SH0ES Cepheid (Riess+ 2022): 73.04'),
    (69.80, '#A23B72', 'Freedman 2019 published (LMC): 69.80'),
    (70.39, '#6B7280', 'Freedman 2025 augmented (NGC 4258): 70.39'),
    (68.81, '#9CA3AF', 'Freedman 2025 JWST-only (NGC 4258): 68.81'),
]
for v, c, label in ref_lines:
    ax.axvline(v, ls='--', color=c, label=label, lw=1.2)

ax.set_xlabel(r'$H_0$ [km/s/Mpc]')
ax.set_ylabel('Monte Carlo draws')
ax.set_title('Framework forward predictions vs published reference values')
ax.legend(fontsize=8, loc='upper right')
plt.tight_layout()
plt.show()
"""))


CELLS.append(md("""
## 2. Uddin 2023 positive-control test

This is a **required validation step**. Before trusting any of the
CSP-based MCMC chains in Section 3, we reproduce Uddin 2023's published
H₀ = 70.242 ± 0.724 km/s/Mpc by running the 8-parameter SNooPy
likelihood on the **full** Uddin 2023 calibrator + Hubble-flow sample
(no Freedman intersection). If the pipeline cannot reproduce Uddin's
own number, the 8-parameter likelihood implementation is incorrect and
no downstream CSP result can be trusted — the notebook halts.

Acceptance: |Δ vs 70.242| ≤ 1.0 km/s/Mpc AND R̂_max < 1.01.
"""))


CELLS.append(md("""
### 2.1 MCMC settings — smoke vs full

`MCMC_MODE = 'full'` runs the preregistered settings (64 walkers ×
20 000 steps × 5 000 burn-in). This is what the analysis report uses
and what reproduces the published numbers; expect ~12 min wall time
end-to-end on a typical Apple-silicon workstation (12-chain matrix
~11 min + Uddin positive control ~49 s).

`MCMC_MODE = 'smoke'` runs an under-converged short chain (32 walkers
× 1 000 steps × 250 burn-in) for ~minutes; the resulting H₀ values
are NOT publication quality and will not match the report. Use only
to verify the wiring.
"""))


CELLS.append(code("""
from pipeline.trgb_comparative.mcmc_runner import MCMCSettings

# Set to 'full' to reproduce the analysis-report numbers.
# Set to 'smoke' for a fast wiring check (results NOT publication quality).
MCMC_MODE = 'full'

if MCMC_MODE == 'full':
    settings = MCMCSettings(
        n_walkers=64, n_steps=20_000, n_burnin=5_000,
        seed=42, progress=True,
    )
    print('MCMC mode: FULL (preregistered, publication quality)')
elif MCMC_MODE == 'smoke':
    settings = MCMCSettings(
        n_walkers=32, n_steps=1_000, n_burnin=250,
        seed=42, progress=False,
    )
    print('MCMC mode: SMOKE TEST')
    print('  >>> Results from smoke-test mode are NOT publication quality.')
    print('  >>> H₀ medians will not match the analysis report. R̂ values')
    print('  >>> will likely exceed the 1.01 convergence gate.')
    print('  >>> Set MCMC_MODE = \\'full\\' to reproduce report numbers.')
else:
    raise ValueError(f'Unknown MCMC_MODE: {MCMC_MODE!r}')

print(f'  walkers = {settings.n_walkers}')
print(f'  steps   = {settings.n_steps}')
print(f'  burn-in = {settings.n_burnin}')
print(f'  seed    = {settings.seed}')
"""))


CELLS.append(code("""
from pipeline.trgb_comparative.uddin_csp_chain import (
    build_uddin_inputs_from_loader_dataset,
    run_uddin_csp_chain,
)

uddin = loader.load_uddin_h0csp_trgb_dataset()
uddin_inputs = build_uddin_inputs_from_loader_dataset(
    uddin, flow_sample_filter='both',
)
print(f'Uddin combined positive-control sample: '
      f'n_cal = {uddin_inputs.n_cal}, n_flow = {uddin_inputs.n_flow}')

t0 = time.time()
positive_control = run_uddin_csp_chain(
    uddin_inputs, settings,
    case='positive_control',
    system='uddin_2023_full',
    system_label='Uddin 2023 H0CSP positive control (full CSP-I+CSP-II)',
    published_target_H0=70.242, published_sigma_stat=0.724,
    notes=('Uddin 2023 ApJ 970, 72 8-parameter SNooPy MCMC on full '
           'B_trgb_update3.csv (CSP-I+CSP-II combined). '
           'Target H0 = 70.242 ± 0.724 km/s/Mpc.'),
    log_fn=print,
)
positive_control_wallclock_s = time.time() - t0

target = 70.242
target_sigma = 0.724
delta = positive_control.H0_median - target
acceptance_tol_kmsmpc = 1.0
passed = (abs(delta) <= acceptance_tol_kmsmpc) and positive_control.converged

print()
print('Uddin 2023 positive-control result:')
print(f'  Pipeline H₀ = {positive_control.H0_median:.3f} ± '
      f'{positive_control.H0_sigma:.3f} km/s/Mpc')
print(f'  Target H₀   = {target:.3f} ± {target_sigma:.3f} km/s/Mpc')
print(f'  Δ           = {delta:+.3f} km/s/Mpc  (tolerance ±{acceptance_tol_kmsmpc:.1f})')
print(f'  R̂_max      = {positive_control.rhat_max:.4f} '
      f'({"converged" if positive_control.converged else "NOT converged"}, gate 1.01)')
print(f'  Wall time   = {positive_control_wallclock_s:.1f} s')
print(f'  PASS        = {passed}')

if not passed:
    if MCMC_MODE == 'smoke':
        print()
        print('NOTE: Positive control failed in smoke-test mode. This is expected;')
        print('      smoke-test settings do not achieve the R̂ gate. Switch to')
        print('      MCMC_MODE = \\'full\\' before drawing conclusions.')
    else:
        raise RuntimeError(
            'Uddin 2023 positive-control test FAILED at full MCMC settings. '
            'The 8-parameter SNooPy likelihood implementation is suspect; '
            'downstream CSP chain results cannot be trusted. Investigate '
            'pipeline/trgb_comparative/uddin_csp_chain.py before continuing.'
        )
"""))


CELLS.append(md("""
## 3. 12-chain matrix execution

The primary pipeline-computed reproduction is the **12-chain matrix**:
4 SN photometric systems × 3 calibrator-sample variants. Each chain
operates on the **full** Freedman calibrator sample appropriate to its
case (no Uddin-intersection subsetting).

Cases:

- **case_a**: Freedman 2019 Table 3, LMC anchor, 18 hosts, target H₀ = 69.80
- **case_b**: Freedman 2025 Table 3, NGC 4258 anchor, 24 SN augmented HST+JWST, target H₀ = 70.39
- **case_b_jwst_only**: Freedman 2025 Table 2, NGC 4258 anchor, 11 SN JWST-only, target H₀ = 68.81

Photometric systems:

- **csp_i**: Uddin 8-parameter likelihood, CSP-I flow
- **csp_ii**: Uddin 8-parameter likelihood, CSP-II flow
- **supercal**: simple 1-parameter likelihood, F19 m_B^SC + Pantheon 2018 flow
- **pantheon_plus**: simple 1-parameter likelihood, Pantheon+SH0ES (CID-matched)

For each chain we report the calibrator coverage (req/got), Hubble-flow
SN count, H₀ posterior median + 1σ, R̂_max, the published target, the
delta, and whether the chain falls within the published statistical
uncertainty.
"""))


CELLS.append(code("""
from pipeline.trgb_comparative.full_calibrator_factories import all_chains_full
from pipeline.trgb_comparative.sn_chain_factories import _run_one as _run_chain_plan

# Chain output goes here (these files persist between cells if you re-run).
# Matches the main pipeline's location at trgb_data/chains/.
chains_dir = repo_root / 'trgb_data' / 'chains'
chains_dir.mkdir(parents=True, exist_ok=True)

CASES = ('case_a', 'case_b', 'case_b_jwst_only')
SYSTEMS = ('csp_i', 'csp_ii', 'supercal', 'pantheon_plus')

print('Building chain plans (full Freedman calibrator samples per case)…')
plans = all_chains_full(loader, include_jwst_only_sensitivity=True)

# Print coverage up front so the reviewer sees req/got before chains run.
print()
print(f"{'case':<18s} {'system':<14s} {'req':>4s} {'got':>4s}  notes")
for case in CASES:
    for system in SYSTEMS:
        plan, cov = plans[case][system]
        if plan is None:
            print(f'  {case:<16s} {system:<14s} ---  ---  UNAVAILABLE: {cov.notes}')
        else:
            print(f'  {case:<16s} {system:<14s} '
                  f'{cov.requested_cal_count:>3d} {cov.matched_cal_count:>3d}')
"""))


CELLS.append(code("""
chain_results: dict = {case: {} for case in CASES}
chain_wallclock_s: dict = {}

t_total = time.time()
for case in CASES:
    for system in SYSTEMS:
        plan, cov = plans[case][system]
        key = f'{case}/{system}'
        if plan is None:
            print(f'[{key}] SKIPPED — {cov.notes}')
            chain_results[case][system] = {
                'case': case, 'system': system,
                'error': cov.notes, 'skipped': True,
                'n_calibrators_requested': cov.requested_cal_count,
                'n_calibrators_matched': cov.matched_cal_count,
            }
            continue
        chain_path = chains_dir / f'{case}_{system}_full.npz'
        t0 = time.time()
        result = _run_chain_plan(plan, settings, chain_out_path=chain_path,
                                 log_fn=print)
        dt = time.time() - t0
        chain_wallclock_s[key] = dt
        rdict = result.as_dict()
        rdict['mode'] = plan.mode
        rdict['n_calibrators_requested'] = cov.requested_cal_count
        rdict['n_calibrators_matched'] = cov.matched_cal_count
        rdict['missing_sn_names'] = list(cov.missing_sn_names)
        chain_results[case][system] = rdict
        rmax = rdict.get('rhat_max', rdict.get('rhat_H0', float('nan')))
        print(f'[{key}] H₀ = {rdict["H0_median"]:.3f} ± '
              f'{rdict["H0_sigma"]:.3f}  R̂={rmax:.4f}  '
              f'wall={dt:.1f}s  conv={"yes" if rdict.get("converged") else "NO"}')

total_chain_wallclock_s = time.time() - t_total
n_converged = sum(1 for case in CASES for system in SYSTEMS
                  if chain_results[case][system].get('converged'))
print()
print(f'12-chain matrix complete: {n_converged}/12 chains converged at R̂ < 1.01.')
print(f'Total chain wall time: {total_chain_wallclock_s:.1f} s '
      f'({total_chain_wallclock_s/60.0:.1f} min)')
"""))


CELLS.append(code("""
# Render the 12-chain matrix table — same shape as the analysis report.
TARGETS = {
    'case_a': (69.80, 0.80),
    'case_b': (70.39, 1.22),
    'case_b_jwst_only': (68.81, 1.80),
}

rows = []
for case in CASES:
    target_H0, target_sigma_stat = TARGETS[case]
    for system in SYSTEMS:
        r = chain_results[case][system]
        if r.get('skipped') or r.get('failed'):
            rows.append({
                'case': case, 'system': system,
                'cal_req_got': f'{r.get("n_calibrators_requested", 0)}/'
                                f'{r.get("n_calibrators_matched", 0)}',
                'N_flow': '—', 'mode': '—',
                'H0_median': float('nan'), 'H0_sigma': float('nan'),
                'rhat_max': float('nan'), 'converged': False,
                'target_H0': target_H0,
                'delta_mcmc_minus_target': float('nan'),
                'within_pm_stat': False,
            })
            continue
        h0 = float(r['H0_median'])
        sig = float(r['H0_sigma'])
        rmax = float(r.get('rhat_max', r.get('rhat_H0', float('nan'))))
        delta = h0 - target_H0
        within = abs(delta) <= target_sigma_stat
        rows.append({
            'case': case, 'system': system,
            'cal_req_got': f'{r["n_calibrators_requested"]}/'
                            f'{r["n_calibrators_matched"]}',
            'N_flow': int(r.get('n_flow', 0)),
            'mode': r['mode'],
            'H0_median': h0, 'H0_sigma': sig,
            'rhat_max': rmax,
            'converged': bool(r.get('converged', False)),
            'target_H0': target_H0,
            'delta_mcmc_minus_target': delta,
            'within_pm_stat': bool(within),
        })

chain_matrix_df = pd.DataFrame(rows)
display_cols = [
    'case', 'system', 'cal_req_got', 'N_flow', 'mode',
    'H0_median', 'H0_sigma', 'rhat_max', 'converged',
    'target_H0', 'delta_mcmc_minus_target', 'within_pm_stat',
]
fmt = chain_matrix_df[display_cols].copy()
fmt['H0_median'] = fmt['H0_median'].map(lambda v: f'{v:.3f}')
fmt['H0_sigma'] = fmt['H0_sigma'].map(lambda v: f'{v:.3f}')
fmt['rhat_max'] = fmt['rhat_max'].map(lambda v: f'{v:.4f}')
fmt['target_H0'] = fmt['target_H0'].map(lambda v: f'{v:.2f}')
fmt['delta_mcmc_minus_target'] = fmt['delta_mcmc_minus_target'].map(lambda v: f'{v:+.3f}')
fmt['converged'] = fmt['converged'].map(lambda b: '✓' if b else '✗')
fmt['within_pm_stat'] = fmt['within_pm_stat'].map(lambda b: '✓' if b else '✗')
print('12-chain matrix (matches the analysis report\\'s '
      '"Full-calibrator-sample MCMC chain matrix" table):')
print()
print(fmt.to_string(index=False))
"""))


CELLS.append(md("""
## 4. Cross-case shifts per SN system

For each photometric system, we compute the shift in H₀ when moving
from the LMC anchor (Case A) to the NGC 4258 anchor (Case B in both
its augmented and JWST-only flavours). This matches the
"Cross-case shift per system" table in the analysis report.
"""))


CELLS.append(code("""
def _h0(case, system):
    r = chain_results[case][system]
    if r.get('skipped') or r.get('failed'):
        return float('nan')
    return float(r['H0_median'])

shift_rows = []
for system in SYSTEMS:
    a = _h0('case_a', system)
    b = _h0('case_b', system)
    bj = _h0('case_b_jwst_only', system)
    shift_rows.append({
        'system': system,
        'A_H0': a, 'B_aug_H0': b, 'B_jwst_H0': bj,
        'delta_B_aug_minus_A': b - a,
        'delta_B_jwst_minus_A': bj - a,
    })

shift_df = pd.DataFrame(shift_rows)
fmt = shift_df.copy()
for c in ('A_H0', 'B_aug_H0', 'B_jwst_H0'):
    fmt[c] = fmt[c].map(lambda v: f'{v:.3f}')
for c in ('delta_B_aug_minus_A', 'delta_B_jwst_minus_A'):
    fmt[c] = fmt[c].map(lambda v: f'{v:+.3f}')
print('Cross-case shift per SN system (full-cal chains):')
print()
print(fmt.to_string(index=False))
"""))


CELLS.append(md("""
## 5. Framework comparison

For each chain in the 12-chain matrix we compute the statistical
tension between the MCMC posterior and the framework's forward
prediction for the corresponding anchor:

- Case A chains compare against the LMC-anchor framework prediction
  (≈ 70.40 km/s/Mpc).
- Case B (both augmented and JWST-only) chains compare against the
  NGC 4258-anchor framework prediction (≈ 69.20 km/s/Mpc).

Both anchors sit firmly in the perturbative regime
(γ/H · L ≈ 0.045 at LMC; ≈ 0.027 at NGC 4258).

Tension is reported in σ-units (statistical only):

  τ = |H₀_MCMC − H₀_framework| / sqrt(σ²_MCMC + σ²_framework)
"""))


CELLS.append(code("""
def _framework_for(case):
    if case == 'case_a':
        return framework_a, bool(framework_a.breakdown_flag_any)
    return framework_b, bool(framework_b.breakdown_flag_any)

tension_rows = []
for case in CASES:
    fw_pred, breakdown_flagged = _framework_for(case)
    fw_med = float(fw_pred.H0_median)
    fw_sigma = 0.5 * (float(fw_pred.H0_high) - float(fw_pred.H0_low))
    for system in SYSTEMS:
        r = chain_results[case][system]
        if r.get('skipped') or r.get('failed'):
            tension_rows.append({
                'case': case, 'system': system,
                'H0_MCMC': float('nan'), 'sigma_MCMC': float('nan'),
                'H0_framework': fw_med, 'sigma_framework': fw_sigma,
                'tension_sigma_stat_only': float('nan'),
                'breakdown_flagged': breakdown_flagged,
            })
            continue
        h0 = float(r['H0_median']); sig = float(r['H0_sigma'])
        denom = (sig * sig + fw_sigma * fw_sigma) ** 0.5
        tau = abs(h0 - fw_med) / denom if denom > 0 else float('nan')
        tension_rows.append({
            'case': case, 'system': system,
            'H0_MCMC': h0, 'sigma_MCMC': sig,
            'H0_framework': fw_med, 'sigma_framework': fw_sigma,
            'tension_sigma_stat_only': tau,
            'breakdown_flagged': breakdown_flagged,
        })

tension_df = pd.DataFrame(tension_rows)
fmt = tension_df.copy()
for c in ('H0_MCMC', 'sigma_MCMC', 'H0_framework', 'sigma_framework',
          'tension_sigma_stat_only'):
    fmt[c] = fmt[c].map(lambda v: 'nan' if pd.isna(v) else f'{v:.3f}')
fmt['breakdown_flagged'] = fmt['breakdown_flagged'].map(
    lambda b: 'BREAKDOWN' if b else 'reliable')
print('Framework-vs-MCMC tension per chain (statistical only):')
print()
print(fmt.to_string(index=False))
print()
print('Note: γ/H · L ≈ 0.045 at LMC, ≈ 0.027 at NGC 4258 — both well '
      'below the |γ/H · L| ≥ 1 perturbative-breakdown criterion.')
"""))


CELLS.append(md("""
## 6. Variable provenance

Every numerical quantity in this notebook falls into exactly one of
five provenance categories. The table below makes the distinction
explicit so a reviewer can audit which numbers are pipeline-computed
posteriors vs published literature citations vs forward predictions.
"""))


CELLS.append(code("""
hoyt = loader.load_hoyt_2025_sn_calibration()
hoyt_systems = hoyt['systems']

provenance_rows = []

# 1. Pipeline-computed MCMC posteriors (12 chains).
for case in CASES:
    for system in SYSTEMS:
        r = chain_results[case][system]
        if r.get('skipped') or r.get('failed'):
            continue
        provenance_rows.append({
            'value_label': f'{case}/{system}',
            'H0_kms_Mpc': float(r['H0_median']),
            'sigma_kms_Mpc': float(r['H0_sigma']),
            'provenance': 'pipeline_computed_mcmc_posterior',
        })

# 2. Framework forward predictions (2 anchors).
provenance_rows.append({
    'value_label': 'H0_framework_predicted_lmc_anchor',
    'H0_kms_Mpc': float(framework_a.H0_median),
    'sigma_kms_Mpc': 0.5 * (float(framework_a.H0_high) - float(framework_a.H0_low)),
    'provenance': 'framework_forward_prediction (BREAKDOWN flagged)',
})
provenance_rows.append({
    'value_label': 'H0_framework_predicted_ngc4258_anchor',
    'H0_kms_Mpc': float(framework_b.H0_median),
    'sigma_kms_Mpc': 0.5 * (float(framework_b.H0_high) - float(framework_b.H0_low)),
    'provenance': 'framework_forward_prediction (reliable regime)',
})

# 3. Published Freedman target H₀ values (Case A, Case B aug, Case B JWST-only).
provenance_rows.append({
    'value_label': 'H0_freedman_2019_published',
    'H0_kms_Mpc': 69.80, 'sigma_kms_Mpc': 0.80,
    'provenance': 'published_target_value',
})
provenance_rows.append({
    'value_label': 'H0_freedman_2025_table3_augmented_published',
    'H0_kms_Mpc': 70.39, 'sigma_kms_Mpc': 1.22,
    'provenance': 'published_target_value',
})
provenance_rows.append({
    'value_label': 'H0_freedman_2025_table2_jwst_only_published',
    'H0_kms_Mpc': 68.81, 'sigma_kms_Mpc': 1.80,
    'provenance': 'published_target_value',
})

# 4. Hoyt 2025 Table 7 literature reference values (4 SN systems).
hoyt_label_map = {
    'CSP-I': 'hoyt_2025_table7_augmented_CSP-I',
    'CSP-II': 'hoyt_2025_table7_augmented_CSP-II',
    'SuperCal': 'hoyt_2025_table7_augmented_SuperCal',
    'Pantheon+': 'hoyt_2025_table7_augmented_Pantheon+',
}
for sys_key, label in hoyt_label_map.items():
    if sys_key not in hoyt_systems:
        continue
    rec = hoyt_systems[sys_key]
    provenance_rows.append({
        'value_label': label,
        'H0_kms_Mpc': float(rec['augmented_H0']),
        'sigma_kms_Mpc': float('nan'),
        'provenance': 'literature_citation_not_pipeline_mcmc',
    })

# 5. SH0ES Cepheid reference (Riess+ 2022).
provenance_rows.append({
    'value_label': 'H0_sh0es_cepheid_riess2022',
    'H0_kms_Mpc': 73.04, 'sigma_kms_Mpc': 1.04,
    'provenance': 'literature_citation_not_pipeline_mcmc',
})

prov_df = pd.DataFrame(provenance_rows)
fmt = prov_df.copy()
fmt['H0_kms_Mpc'] = fmt['H0_kms_Mpc'].map(lambda v: f'{v:.3f}')
fmt['sigma_kms_Mpc'] = fmt['sigma_kms_Mpc'].map(
    lambda v: 'nan' if pd.isna(v) else f'{v:.3f}')
print('Variable provenance table:')
print()
print(fmt.to_string(index=False))
"""))


CELLS.append(md("""
## 7. Full pipeline reproduction (optional)

The cell below runs the **complete** pipeline end-to-end via
`TRGBComparativePipeline`. This is what is invoked by

    python main.py --trgb-comparative

It produces every chain, figure, and report file in
`results/trgb_comparative/`. It is **not** required to verify the
12-chain matrix above — Sections 2–6 already do that — but it is the
single command that re-generates the published artefacts byte-for-byte.

Wall time at full preregistered settings (64 walkers × 20 000 steps ×
5 000 burn-in): ~12 minutes end-to-end on a typical Apple-silicon
workstation (12-chain matrix ~11 min + Uddin positive control ~49 s
+ framework predictions, tables, and figures).
"""))


CELLS.append(code("""
# Uncomment the lines below to run the full pipeline.

# from pipeline.trgb_comparative import TRGBComparativePipeline
# pipe = TRGBComparativePipeline('results/trgb_comparative')
# out = pipe.run({
#     'short': False,                  # use preregistered (full) MCMC settings
#     'enforce_preregistration': True, # halt unless Stage 1+2 docs exist
#     'strict_data': True,             # halt on any missing dataset
#     'n_framework_samples': 50_000,
# })
# print('Pipeline complete. Report:', out['main']['report'])
"""))


CELLS.append(md("""
## Verification

After running this notebook end-to-end with `MCMC_MODE = 'full'`, the
12-chain matrix table in Section 3 should match the
"Full-calibrator-sample MCMC chain matrix" table in
`results/trgb_comparative/reports/trgb_comparative_analysis_report.md`
to within ~0.1 km/s/Mpc on each H₀ median. If you observe a larger
discrepancy:

1. Check that all preregistration documents are present and unchanged.
2. Verify the data-availability check in Section 0 passed every key.
3. Confirm `MCMC_MODE = 'full'` was set before Section 2 ran.
4. Confirm the Uddin 2023 positive control passed (Section 2).
5. If the discrepancy persists, document it in the cell output above
   and open a reproducibility issue against the pipeline.

**Run wall time**: record the value of `total_chain_wallclock_s` from
Section 3 and `positive_control_wallclock_s` from Section 2 here when
running at full settings.
"""))


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {
                "name": "python",
                "version": "3.11",
            },
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    # Add per-cell ids (nbformat 4.5 wants them).
    for i, cell in enumerate(nb["cells"]):
        cell["id"] = f"cell-{i:02d}"
    out = Path(__file__).resolve().parents[1] / "notebooks" / "trgb_comparative_reproduction.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n",
                   encoding="utf-8")
    print(f"Wrote {out}  ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
