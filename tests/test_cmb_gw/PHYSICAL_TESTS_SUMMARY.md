# Physically Motivated Tests for Literature-Based Void Calibration

## Overview

This document summarizes the physically motivated unit and integration tests added for the literature-based void calibration implementation. These tests verify not just code correctness, but physical consistency with cosmological principles.

## Unit Tests (`test_void_scaling_literature.py`)

### Test G: Physical Consistency Checks

**Purpose**: Verify that void size scaling follows physical principles from cosmology.

1. **`test_void_ratio_scales_with_redshift`**
   - **Physics**: Void ratio should increase with redshift for β > 0
   - **Reason**: G_eff was weaker in the past (higher z), so growth was more suppressed, leading to larger voids
   - **Verifies**: Monotonic increase of void ratio with z

2. **`test_void_ratio_at_recombination`**
   - **Physics**: At recombination (z ≈ 1100), f(z) ≈ 0.24, so G_eff/G_0 ≈ 1 - 0.24β
   - **Reason**: For β = 0.2, G_eff was ~5% weaker, leading to suppressed growth
   - **Verifies**: Void ratio > 1 at high z, larger than at low z

3. **`test_void_ratio_approaches_unity_at_z_zero`**
   - **Physics**: As z → 0, G_eff → G_0, so void ratio → 1
   - **Reason**: Fundamental requirement: at present day, evolving G matches ΛCDM
   - **Verifies**: All ratios close to 1.0 at low z (within ~1%)

4. **`test_growth_factor_consistency`**
   - **Physics**: Void ratio should scale as [D(β)/D(0)]^γ where γ = 1.7
   - **Reason**: Core physical relation from Pisani+ (2015)
   - **Verifies**: Exact power-law scaling with growth factor ratio

5. **`test_void_size_increases_with_beta_physically`**
   - **Physics**: Larger β → weaker G in past → more suppressed growth → larger voids
   - **Reason**: Core physical prediction of evolving G model
   - **Verifies**: Strict monotonicity of void ratio with β

6. **`test_negative_beta_decreases_void_size`**
   - **Physics**: Negative β → stronger G in past → enhanced growth → smaller voids
   - **Reason**: Tests physical consistency of sign convention
   - **Verifies**: Proper ordering: negative β < ΛCDM < positive β

7. **`test_uncertainty_scales_with_gamma_error`**
   - **Physics**: Uncertainty should scale with γ uncertainty (σ_γ = 0.2)
   - **Reason**: Literature uncertainty comes from power-law exponent uncertainty
   - **Verifies**: Larger β → larger D_ratio deviation → larger uncertainty

### Test H: Literature Value Validation

**Purpose**: Verify that implementation matches literature values and expectations.

1. **`test_gamma_value_matches_pisani_2015`**
   - **Physics**: γ = 1.7 ± 0.2 from Pisani+ (2015, PRD 91, 043513)
   - **Reason**: Core calibration parameter from billion-particle simulations
   - **Verifies**: Exact match to literature values

2. **`test_power_law_exponent_effect`**
   - **Physics**: For typical growth suppression (D_ratio ≈ 0.95), void ratio ≈ 0.95^1.7 ≈ 0.92
   - **Reason**: Tests physically reasonable void size enhancements (~8%)
   - **Verifies**: Power-law scaling gives reasonable results

3. **`test_uncertainty_matches_literature_expectation`**
   - **Physics**: For γ = 1.7 ± 0.2 and D_ratio ≈ 0.95, uncertainty ≈ 1%
   - **Reason**: Consistency with literature uncertainty expectations
   - **Verifies**: Uncertainty propagation matches formula and gives reasonable values

## Integration Tests (`test_void_analysis_literature.py`)

### Test E: Physically Motivated Scenarios

**Purpose**: Test full pipeline with realistic physical scenarios.

1. **`test_void_analysis_with_realistic_beta`**
   - **Physics**: Test with β ≈ 0.2 (realistic value producing ~5-10% void enhancements)
   - **Reason**: Verifies full pipeline with physically motivated β value
   - **Verifies**: Physical consistency, β extraction accuracy

2. **`test_void_analysis_at_different_redshifts`**
   - **Physics**: Test redshift evolution handling
   - **Reason**: Voids at different redshifts should give consistent β values
   - **Verifies**: Proper redshift handling in analysis

3. **`test_beta_extraction_consistency`**
   - **Physics**: β extraction should be consistent with void size ratio
   - **Reason**: Larger voids → positive β, smaller voids → negative/small β
   - **Verifies**: Physical consistency between ratio and extracted β

4. **`test_uncertainty_propagation_through_pipeline`**
   - **Physics**: Uncertainties should reflect data scatter + literature calibration uncertainty
   - **Reason**: Proper error propagation is essential for joint fits
   - **Verifies**: Reasonable uncertainty values (~10-30% relative)

5. **`test_literature_calibration_applied_correctly`**
   - **Physics**: Verify γ = 1.7 calibration is applied, not old semi-analytic
   - **Reason**: Ensures rigorous method is used
   - **Verifies**: Methodology flag, citations present, no qualitative caveats

### Test F: Joint Fit Integration

**Purpose**: Verify void β integrates correctly with other cosmological tests.

1. **`test_void_beta_included_in_joint_fit`**
   - **Physics**: Void β should participate in joint consistency check
   - **Reason**: Rigorous calibration allows inclusion in joint analysis
   - **Verifies**: Structure for joint fit, include_in_joint flag

2. **`test_void_beta_consistency_with_other_tests`**
   - **Physics**: Void β should agree with CMB and BAO β within errors
   - **Reason**: Consistency across independent probes is key discriminant
   - **Verifies**: Structure allows consistency checks

### Test G: Error Handling

**Purpose**: Verify robust handling of edge cases and errors.

1. **`test_handles_missing_data_gracefully`**
   - **Physics**: Missing catalogs should not crash analysis
   - **Reason**: Robust error handling for production use
   - **Verifies**: Graceful degradation, valid structure returned

2. **`test_handles_extreme_ratios`**
   - **Physics**: Extreme void size ratios should be handled gracefully
   - **Reason**: Data may have outliers or systematic errors
   - **Verifies**: No crashes, valid structure even with extreme values

## Test Coverage Summary

### Physical Principles Tested

1. **Redshift Evolution**: Void ratio increases with z for β > 0 ✓
2. **Present-Day Limit**: Ratio → 1 as z → 0 ✓
3. **Growth Factor Scaling**: Exact power-law relation [D(β)/D(0)]^1.7 ✓
4. **Beta Dependence**: Monotonic increase with β ✓
5. **Sign Convention**: Negative β decreases void size ✓
6. **Uncertainty Propagation**: Proper error propagation ✓
7. **Literature Consistency**: Matches Pisani+ (2015) values ✓

### Integration Scenarios Tested

1. **Realistic β Values**: β ≈ 0.2 produces reasonable results ✓
2. **Redshift Handling**: Proper formation redshift usage ✓
3. **Consistency Checks**: Ratio ↔ β extraction consistency ✓
4. **Joint Fit Integration**: Structure for multi-probe analysis ✓
5. **Error Handling**: Graceful degradation on missing/extreme data ✓

## Running the Tests

```bash
# Run all unit tests
pytest tests/test_cmb_gw/test_void_scaling_literature.py -v

# Run all integration tests
pytest tests/test_cmb_gw/test_void_analysis_literature.py -v

# Run specific physical consistency tests
pytest tests/test_cmb_gw/test_void_scaling_literature.py::TestPhysicalConsistency -v

# Run literature validation tests
pytest tests/test_cmb_gw/test_void_scaling_literature.py::TestLiteratureValueValidation -v
```

## Expected Test Counts

- **Unit Tests**: ~30 tests (including original + new physical tests)
- **Integration Tests**: ~12 tests (including original + new scenarios)
- **Total**: ~42 tests covering physical consistency, literature validation, and integration scenarios

## Key Physical Insights Verified

1. **Void sizes scale with growth factor**: R_v(β)/R_v(0) = [D(β)/D(0)]^1.7
2. **Effect increases with redshift**: Higher z → larger void enhancement
3. **Present-day limit**: G_eff(z=0) = G_0 → ratio = 1
4. **Uncertainty from literature**: ~20% precision on β from γ = 1.7 ± 0.2
5. **Consistency requirement**: Void β should agree with CMB/BAO β

These tests ensure the implementation is not just mathematically correct, but physically consistent with cosmological principles and literature-calibrated relations.

