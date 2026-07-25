"""Tests for the lightweight experiment queue."""

import json
import os
import shlex
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.experiment_queue import parse_queue, shell_env

CASCADIAV3 = Path(__file__).resolve().parents[1]
RUNNER = CASCADIAV3 / "scripts" / "run_experiment_queue.sh"


class ExperimentQueueTest(unittest.TestCase):
    def test_parser_accepts_source_revision_as_an_ordinary_value(self) -> None:
        with TemporaryDirectory() as root:
            path = Path(root) / "queue.jsonl"
            path.write_text(
                json.dumps(
                    {
                        "name": "stage",
                        "script": "run.sh",
                        "env": {"SOURCE_REVISION": "informational", "GAMES": 12},
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            stages = parse_queue(str(path))
        self.assertEqual(stages[0]["env"]["SOURCE_REVISION"], "informational")
        self.assertEqual(stages[0]["env"]["GAMES"], "12")

    def test_shell_env_round_trips_spaces_and_quotes(self) -> None:
        rendered = shell_env({"A": "has space", "B": "it's"})
        self.assertEqual(shlex.split(rendered), ["A=has space", "B=it's"])

    def test_runner_executes_without_receipts_or_done_markers(self) -> None:
        with TemporaryDirectory() as root_text:
            root = Path(root_text)
            (root / "cascadiav3" / "logs").mkdir(parents=True)
            (root / "cascadiav3" / "src").symlink_to(CASCADIAV3 / "src")
            (root / "stage.sh").write_text(
                "#!/usr/bin/env bash\nprintf '%s\\n' \"$VALUE\" > output.txt\n",
                encoding="utf-8",
            )
            (root / "queue.jsonl").write_text(
                json.dumps(
                    {"name": "plain", "script": "stage.sh", "env": {"VALUE": "works"}}
                )
                + "\n",
                encoding="utf-8",
            )
            env = dict(os.environ)
            env.update({"ROOT": str(root), "PYTHON": sys.executable})
            result = subprocess.run(
                ["bash", str(RUNNER), "queue.jsonl"],
                cwd=root,
                env=env,
                capture_output=True,
                text=True,
                timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            self.assertEqual((root / "output.txt").read_text(), "works\n")
            self.assertFalse(any((root / "cascadiav3" / "logs").glob("*receipt*")))
            self.assertFalse(any((root / "cascadiav3" / "logs").glob("queue_done_*")))


if __name__ == "__main__":
    unittest.main()
