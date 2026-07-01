import json
import sys
import tempfile
import unittest
from pathlib import Path

from host_local_job_runner import run


class HostLocalJobRunnerTests(unittest.TestCase):
    def test_records_success_and_progress(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            progress = root / "groups"
            progress.mkdir()
            (progress / "row-000.json").write_text("{}")
            status = root / "status.json"
            result = run(
                [sys.executable, "-c", "raise SystemExit(0)"],
                status,
                progress,
                "row-",
                ".json",
                2,
                0.01,
                root / "stdout.log",
                root / "stderr.log",
            )
            self.assertEqual(result, 0)
            value = json.loads(status.read_text())
            self.assertEqual(value["state"], "completed")
            self.assertEqual(value["exit_code"], 0)
            self.assertEqual(value["progress"]["completed_items"], 1)

    def test_records_failure(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            status = Path(temporary) / "status.json"
            result = run(
                [sys.executable, "-c", "raise SystemExit(7)"],
                status,
                None,
                "",
                "",
                None,
                0.01,
                None,
                None,
            )
            self.assertEqual(result, 7)
            value = json.loads(status.read_text())
            self.assertEqual(value["state"], "failed")
            self.assertEqual(value["exit_code"], 7)


if __name__ == "__main__":
    unittest.main()
