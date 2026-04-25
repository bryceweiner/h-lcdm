# TRGB Comparative Analysis — Preregistration Stage 1

*Generated: 2026-04-25T14:54:08Z*

**Stage 1 is frozen BEFORE any data loading.** Every methodological choice documented below is committed to the repository in this form.

## Framework inputs (runtime-computed, not cached)

- γ/H at z=0: HLCDMCosmology.gamma_at_redshift(0.0) / HLCDM_PARAMS.get_hubble_at_redshift(0.0) — runtime computation, no hardcoded constant.
- C(G) (historical, NOT used): Not used. The 2026-04-25 C(G)-removal correction reduced the projection formula to the linear form 1 + (γ/H) · L, with no C(G) term. The clustering coefficient C(G) and its source (e8-heterotic-network package, Convention A, 27/55 ≈ 0.4909) remain documented in docs/correction_log.md for historical reference but enter the H₀ projection nowhere. The ML pipeline (pipeline/ml/) continues to use Convention A clustering for pattern-detection diagnostics, which are unrelated to the H₀ projection here.
- d_CMB: HLCDM_PARAMS.D_CMB_PLANCK_2018 = 13869.7 Mpc (Planck 2018 VI, A&A 641 A6, Table 2).
- Perturbative-regime breakdown criterion: |γ/H · L| ≥ 1.0. Does not fire for any realistic distance-ladder anchor under the linear form (LMC γ/H · L ≈ 0.045; NGC 4258 ≈ 0.027).

## Reproduction configuration

- Parametrization (primary): freedman_fixed — only H0 is sampled; M_TRGB, E(B-V), and β are held at published central values (Freedman's frequentist-profile approach).
- Parametrization (sensitivity variant): bayesian_sampled — all 4 parameters (H0, M_TRGB, E(B-V), β) sampled with Gaussian priors; retained as a sensitivity-analysis variant.
- Case A tip source: freedman_2019 — per-host μ_TRGB read from trgb_data/catalogs/freedman_2019_table3.csv (Freedman 2019 Table 3 transcription, SHA-256 verified; 15 unique hosts, 18 SN Ia calibrators).
- Case B tip source: freedman_2025 — per-host μ_TRGB read from trgb_data/catalogs/freedman_2025_table2.csv (Freedman 2025 Table 2 transcription; 10 unique hosts with weighted TRGB+JAGB distance moduli).
- Tip source sensitivity variant: anand_2022 — Anand 2021 independent EDD reduction μ_TRGB (variant cross-check against the Freedman-paper primary).
- Per-host extinction: Freedman 2019 Table 1 per-host A_F814W values (trgb_data/catalogs/freedman_2019_table1.csv). Fallback: placeholder EBV_SFD for hosts not in Freedman Table 1 — sensitivity-only path.
- Hubble-flow z cuts: 0.023 ≤ z_CMB ≤ 0.15 (Freedman 2019 §6.3 Supercal subsample). Applied to Pantheon+SH0ES non-calibrator SNe; N_flow ≈ 496. For the Pantheon+-only variant of H₀; CSP/SuperCal variants use Hoyt 2025 Eq. 15 reference values directly (see sn_system_*).
- SN Ia system (Case A primary): CSP-I (Freedman 2019 primary SN sample). Per-system reproduced H₀ computed via Hoyt 2025 Eq. 15 applied to our TRGB distances with the Hoyt 2025 Table 6/7 reference values. Variants: CSP-I, CSP-II, SuperCal, Pantheon+ all computed and reported.
- SN Ia system (Case B primary): CSP-II (Freedman 2025 primary SN sample). Same 4-variant analysis as Case A. The Pantheon+ variant is expected to be +2 km/s/Mpc higher than CSP-II per Hoyt 2025 Section 4; this is documented but not used as the primary number.

### 2026-04-24 Amendment: primary SN sample

2026-04-24 amendment: primary Hubble-flow SN sample for each case is CCHP's own CSP-I/II rather than Pantheon+SH0ES. Pantheon+ calibration is known (Hoyt 2025 §4, 3.1σ significance) to bias H₀ upward by ≈+2 km/s/Mpc relative to CSP, because Pantheon+ μ_SH0ES values are Cepheid/SH0ES-anchored. Retaining Pantheon+ as the fourth variant directly demonstrates this +2 km/s/Mpc shift.

## Case A — Freedman 2019/2020 (LMC anchor)

- Anchor: LMC (Pietrzyński 2019 DEB); μ = 18.477 ± 0.026 (stat) ± 0.024 (sys).
- d_local: 0.0496 Mpc ± 0.0009 Mpc
- Primary band: F814W
- Extinction: Freedman 2019 Table 1 per-host A_F814W values (authoritative paper-tabulated extinction). Freedman's own SFD + CCM89 R_V=3.1 values.
- Metallicity: Freedman 2020 F814W color slope, β = 0.20 (Rizzi 2007), fixed.
- Reproduction tolerance target: ±0.8 km/s/Mpc (stat).
- Sample selection criteria:
  - Hosts in Freedman 2019 ApJ 882, 34 Table 3 (15 unique TRGB hosts, 18 SN Ia calibrators). Each Table 3 host is included with its Freedman-published μ_TRGB value.
  - HST photometry from the Anand 2021 EDD reduction attached where available (10 hosts: NGC 1316, 1365, 1404, 1448, 3627, 4038, 4424, 4526, 4536, 5643); remaining 5 hosts (M101, NGC 1309, 3021, 3368, 3370, 5584) enter as 'photometry stubs' — published μ only.
  - NGC 4258 is excluded from Case A (anchor-galaxy contamination; enforced by scripts/build_trgb_manifests.py CASE_A_EXCLUDED_HOSTS).

## Case B — Freedman 2024/2025 (NGC 4258 anchor)

- Anchor: NGC 4258 (Reid 2019 maser); μ = 29.397 ± 0.024 (stat) ± 0.022 (sys).
- d_local: 7.58 Mpc ± 0.08 Mpc
- Primary band: F150W (reported in the paper); μ_TRGB values used are the weighted TRGB+JAGB μ_bar
- Extinction: Placeholder (JWST NIRCam per-host extinction infrastructure not in pipeline). Sensitivity-only path; primary reproduction uses Freedman 2025 Table 2 published μ_bar directly and therefore does not depend on our per-field extinction.
- Metallicity: Inherits M_TRGB_abs = -4.049 common zero point (Freedman 2025 §14.2: 'F19 and F21 share a common TRGB absolute magnitude zero point').
- Reproduction tolerance target: ±1.22 km/s/Mpc (stat).
- Sample selection criteria:
  - Hosts in Freedman 2025 ApJ 985, 203 Table 2 (10 JWST-observed hosts, 11 SN Ia calibrators). Each is included with its published weighted μ_bar.
  - JWST raw NIRCam photometry was NOT downloaded & reduced; public from MAST (GO-1995, 2875, 3055) but DOLPHOT-level re-reduction is out of scope. We use Freedman 2025's published μ_TRGB values directly (faithful 'reproduction of published' posture).
  - Edge-detection sensitivity variants operate on HST-era photometry where overlap exists (NGC 1365, 1448, 4038, 4424, 4536, 5643).

## Edge detection

- Primary: Published μ_TRGB (bypass). Edge detection runs on the raw photometry for sensitivity diagnostics only; the primary path reads μ_TRGB from the Freedman-paper table (tip_source).
- Sensitivity variants:
  - Sobel kernel width 1.0
  - Sobel kernel width 2.0 (Freedman published choice)
  - Sobel kernel width 3.0
  - Model-based (Makarov 2006-style broken power-law fit)
  - Bayesian (Hatt 2017-style posterior over tip location)

## MCMC

- Settings: 32 walkers × 10000 steps (2000 burn-in), Gelman-Rubin R̂ gate 1.01, seed 42.

### Prior boxes — Freedman 2020

| Parameter | lo | hi | mean | sigma |
| --- | --- | --- | --- | --- |
| H0 | 55.0 | 85.0 |  |  |
| M_TRGB | -5.0 | -3.5 | -4.049 | 0.045 |
| EBV | -0.1 | 0.3 | 0.07 | 0.03 |
| beta | -0.2 | 0.6 | 0.2 | 0.1 |

### Prior boxes — Freedman 2024

| Parameter | lo | hi | mean | sigma |
| --- | --- | --- | --- | --- |
| H0 | 55.0 | 85.0 |  |  |
| M_TRGB | -4.2 | -3.9 | -4.049 | 0.05 |
| EBV | -0.1 | 0.3 | 0.07 | 0.03 |
| beta | -0.2 | 0.6 | 0.2 | 0.1 |

## Compute backend

- Preference: `auto` (MLX when available; NumPy otherwise).

## Sensitivity variants

Sensitivity variants run in a SEPARATE analysis stage; they never feed into the primary reproduction numbers.

- Extinction: green2019_3d
- Metallicity: rizzi2007
- Metallicity: jang_lee_2017

