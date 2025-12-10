import os
import pathlib

from pipeline.common.reporting import HLambdaDMReporter


def _minimal_gamma_results():
    return {
        "main": {
            "theory_summary": {
                "present_day": {"gamma_s^-1": 1.0, "lambda_m^-2": 1.0},
                "recombination_era": {"redshift": 1100.0, "gamma_s^-1": 2.0, "lambda_m^-2": 2.0},
                "evolution_ratios": {"gamma_recomb/gamma_today": 2.0, "lambda_recomb/lambda_today": 2.0},
                "qtep_ratio": 2.257,
                "key_equations": ["gamma(z) = gamma_0 (1+z)^n"],
            },
            "model_comparison": {"comparison_available": False},
        },
        "validation": {"overall_status": "PASSED"},
    }


def test_generate_pipeline_report_gamma(tmp_path, monkeypatch):
    monkeypatch.setenv("XAI_API_KEY", "dummy")
    reporter = HLambdaDMReporter(output_dir=str(tmp_path))
    report_path = reporter.generate_pipeline_report("gamma", _minimal_gamma_results())
    assert pathlib.Path(report_path).exists()
    content = pathlib.Path(report_path).read_text()
    assert "γ(z=0)" in content


def test_pipeline_header_avoids_first_person():
    reporter = HLambdaDMReporter(output_dir="results")
    header = reporter._generate_pipeline_header("recommendation")
    assert "What are we analyzing?" not in header
    assert "Objective:" in header
    assert "Observables Assessed:" in header

