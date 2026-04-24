"""
Stage-level checkpoint / resume for the TRGB pipeline.

First-run cost is ~35-50 hours (data download, photometry across ~40 hosts ×
sensitivity variants, two MCMC branches). Rather than push users to restart
from scratch on any interruption, we mark each logical stage and skip
stages whose inputs have not changed.

Stages:

- ``download``
- ``preregister_stage2``
- ``photometry_case_a``
- ``photometry_case_b``
- ``mcmc_freedman_2020``
- ``mcmc_freedman_2024``
- ``framework_prediction``
- ``reporting``

Stage markers live under ``{output_dir}/checkpoints/{stage}.marker``. Each
marker is a JSON blob with timestamp and a SHA-256 of the concatenated
input SHAs. A stage is skipped when its marker exists and input SHAs match.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional


@dataclass(frozen=True)
class StageResult:
    stage: str
    marker_path: Path
    input_sha: str
    timestamp: float
    cached: bool


def _sha256_of_strings(items: Iterable[str]) -> str:
    h = hashlib.sha256()
    for item in items:
        h.update(item.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _sha256_of_file(path: Path) -> str:
    if not path.exists():
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


class CheckpointManager:
    """Manage stage markers under a given output directory."""

    def __init__(self, output_dir: Path | str) -> None:
        self.output_dir = Path(output_dir)
        self.checkpoints_dir = self.output_dir / "checkpoints"
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Input-hash helpers
    # ------------------------------------------------------------------

    def input_sha(self, *, files: Iterable[Path] = (), strings: Iterable[str] = ()) -> str:
        """Combine file hashes and string inputs into a single SHA-256."""
        hashes = [_sha256_of_file(p) for p in files]
        return _sha256_of_strings(list(strings) + hashes)

    # ------------------------------------------------------------------
    # Marker ops
    # ------------------------------------------------------------------

    def marker_path(self, stage: str) -> Path:
        return self.checkpoints_dir / f"{stage}.marker"

    def is_complete(self, stage: str, input_sha: str) -> bool:
        mp = self.marker_path(stage)
        if not mp.exists():
            return False
        try:
            payload = json.loads(mp.read_text())
        except (OSError, json.JSONDecodeError):
            return False
        return str(payload.get("input_sha")) == input_sha

    def mark_complete(self, stage: str, input_sha: str) -> StageResult:
        mp = self.marker_path(stage)
        payload = {
            "stage": stage,
            "input_sha": input_sha,
            "timestamp": time.time(),
        }
        mp.write_text(json.dumps(payload, indent=2))
        return StageResult(
            stage=stage,
            marker_path=mp,
            input_sha=input_sha,
            timestamp=payload["timestamp"],
            cached=False,
        )

    def skip_or_run(
        self,
        stage: str,
        input_sha: str,
        runner: Optional[callable] = None,
    ) -> StageResult:
        """Return a cached StageResult or call ``runner()`` and mark complete.

        ``runner`` is only invoked when the stage is not already complete;
        its return value is ignored — the contract is that the runner
        writes its own outputs to the output directory and we are just
        tracking stage completion.
        """
        if self.is_complete(stage, input_sha):
            return StageResult(
                stage=stage,
                marker_path=self.marker_path(stage),
                input_sha=input_sha,
                timestamp=self.marker_path(stage).stat().st_mtime,
                cached=True,
            )
        if runner is not None:
            runner()
        return self.mark_complete(stage, input_sha)


__all__ = ["CheckpointManager", "StageResult"]
