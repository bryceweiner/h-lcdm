from pipeline.common.reporting import recommendation_reporter


def test_recommendation_grok_prompt_enforces_third_person(monkeypatch):
    captured = {}

    class FakeGrok:
        def generate_custom_report(self, prompt):
            captured["prompt"] = prompt
            return "analysis"

    main_results = {
        "recommendations": {
            "rec1": {
                "dataset": "act_dr6_tt",
                "ell_band": [500, 2000],
                "z_eff": 2.1,
                "gamma_ratio": 1.05,
                "anomaly_indices": [1, 2],
                "fourier_summary": {
                    "modulation_amplitude_pct": 1.2,
                    "target_power": 1.0,
                    "power_ratio": 0.9,
                },
            }
        }
    }

    report_text = recommendation_reporter.results(main_results, grok_client=FakeGrok())

    assert "NEVER use first person" in captured["prompt"]
    assert "third-person" in captured["prompt"]
    assert "analysis" in report_text


def test_recommendation_validation_reports_bootstrap_and_null():
    fake_results = {
        "validation": {
            "bandpass_stability": {"passed": True, "relative_shifts": [0.1]},
            "interpolation_sensitivity": {"passed": True, "relative_shift": 0.05},
            "base_amplitude_pct": 1.2,
            "bootstrap": {
                "test": "bootstrap_modulation_null",
                "n_bootstrap": 50,
                "mean_amp_pct": 0.2,
                "std_amp_pct": 0.1,
                "p95_amp_pct": 0.5,
                "observed_amp_pct": 1.2,
                "p_value_ge_obs": 0.04,
                "null_hypothesis_rejected": True,
            },
            "null_hypothesis": {
                "test": "null_modulation_zero_mean",
                "null_hypothesis": "No γ-scaling modulation",
                "observed_amp_pct": 1.2,
                "p_value": 0.04,
                "z_score": 10.0,
                "null_hypothesis_rejected": True,
            },
        }
    }

    text = recommendation_reporter.validation(fake_results)
    assert "Bootstrap Modulation Null" in text
    assert "Null Hypothesis Test" in text
    assert "p-value" in text
    assert "null_hypothesis_rejected" not in text  # should be rendered, not literal key
