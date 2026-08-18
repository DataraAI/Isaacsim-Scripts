"""run_logger.py — structured run logging, no Isaac Sim dependency."""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


@dataclass
class RunLogger:
    output_dir: Path
    pipeline: str
    task: str
    extra_meta: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.run_id = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H-%M-%S") + f"_{self.pipeline}"
        self.run_dir = self.output_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._frames_fh = (self.run_dir / "frames.jsonl").open("a", buffering=1)
        self._events_fh = (self.run_dir / "events.jsonl").open("a", buffering=1)
        self._started_at = datetime.now(timezone.utc)
        self._write_meta("in_progress")

    def log_frame(self, t: int, **fields: Any) -> None:
        self._frames_fh.write(json.dumps({"t": t, **fields}) + "\n")
        self._frames_fh.flush()

    def log_event(self, t: int, event: str, **fields: Any) -> None:
        self._events_fh.write(json.dumps({"t": t, "event": event, **fields}) + "\n")
        self._events_fh.flush()

    def _write_meta(self, outcome: str) -> None:
        git_commit = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, cwd=Path(__file__).parent,
        ).stdout.strip()
        meta = {
            "run_id": self.run_id,
            "pipeline": self.pipeline,
            "task": self.task,
            "git_commit": git_commit,
            "started_at": self._started_at.isoformat(),
            "ended_at": datetime.now(timezone.utc).isoformat() if outcome != "in_progress" else None,
            "outcome": outcome,
            "schema_version": 1,
            **self.extra_meta,
        }
        (self.run_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    def finalize(self, outcome: str) -> None:
        self._frames_fh.close()
        self._events_fh.close()
        self._write_meta(outcome)
