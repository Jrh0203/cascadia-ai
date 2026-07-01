from __future__ import annotations

import json
from pathlib import Path

from frontier_calibrated_adamw_report import summarize_failed_attempts


def test_failed_attempt_summary_counts_only_nonzero_finishes(
    tmp_path: Path,
) -> None:
    path = tmp_path / "events.jsonl"
    path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "event": "finished",
                        "task_id": "a",
                        "return_code": 1,
                        "elapsed_seconds": 2.0,
                    }
                ),
                json.dumps(
                    {
                        "event": "finished",
                        "task_id": "b",
                        "return_code": 0,
                        "elapsed_seconds": 3.0,
                    }
                ),
            ]
        )
        + "\n"
    )
    summary = summarize_failed_attempts(path)
    assert summary == {
        "count": 1,
        "seconds": 2.0,
        "task_ids": ["a"],
    }
