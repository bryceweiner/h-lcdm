"""End-to-end pipeline smoke test.

Runs TRGBComparativePipeline with synthetic minimal inputs — strict_data=False,
no preregistration enforcement — to verify wiring. The Freedman MCMC branches
skip (no photometry available); the framework predictions always run and
reports are generated.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from pipeline.trgb_comparative import TRGBComparativePipeline


def test_pipeline_smoke_without_data(tmp_path: Path):
    out_dir = tmp_path / "out"
    pipe = TRGBComparativePipeline(str(out_dir))
    results = pipe.run(
        {
            "short": True,
            "enforce_preregistration": False,
            "strict_data": False,
            "n_framework_samples": 500,
        }
    )
    assert "main" in results
    main = results["main"]
    assert main["framework_a"] is not None
    assert main["framework_b"] is not None
    assert main["framework_a"]["breakdown_fraction"] > 0.9
    assert main["framework_b"]["breakdown_fraction"] < 0.1
    # Figures emitted.
    assert "h0_posteriors" in main["figures"]
    # Report file exists.
    assert Path(main["report"]).exists()


def test_pipeline_validate_smoke(tmp_path: Path):
    pipe = TRGBComparativePipeline(str(tmp_path / "out"))
    result = pipe.validate()
    assert result["validation"]["passed"]


def test_pipeline_preregister_stage1_writes_doc(tmp_path: Path, monkeypatch):
    docs = tmp_path / "docs"
    pipe = TRGBComparativePipeline(str(tmp_path / "out"))
    pipe.docs_dir = docs
    out = pipe.preregister_stage1()
    p = Path(out["stage1_path"])
    assert p.exists()
    text = p.read_text()
    assert "Preregistration Stage 1" in text
    assert "LMC" in text and "NGC 4258" in text
