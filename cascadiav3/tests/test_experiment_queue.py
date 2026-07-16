"""Contract tests for the experiment-queue JSONL parser and the bash runner.

The parser tests pin the fail-closed validation contract of
cascadiav3.experiment_queue. The integration tests run the real
run_experiment_queue.sh against a sandboxed fake ROOT (symlinked src/ and
lib_waiter.sh, deployed-revision marker instead of git) with sub-second
heartbeat polling, covering env hand-off, failure-continues semantics,
done-marker resume, HOLD gating, and fail-closed queue validation.
"""

import json
import os
import shlex
import subprocess
import sys
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cascadiav3.experiment_queue import (
    missing_scripts,
    parse_queue,
    shell_env,
    shell_stage_line,
)

CASCADIAV3 = Path(__file__).resolve().parents[1]
RUNNER = CASCADIAV3 / "scripts" / "run_experiment_queue.sh"
EXAMPLE_QUEUE = Path(__file__).parent / "fixtures" / "experiment_queue" / "example.jsonl"
SOURCE_REVISION = "testrev0123"


def write_queue(path, stages):
    path.write_text("\n".join(json.dumps(stage) for stage in stages) + "\n", encoding="utf-8")


class ParseQueueTest(unittest.TestCase):
    def parse_lines(self, tmp, lines):
        queue = Path(tmp) / "queue.jsonl"
        queue.write_text("\n".join(lines) + "\n", encoding="utf-8")
        return parse_queue(str(queue))

    def test_valid_file_parses(self) -> None:
        with TemporaryDirectory() as tmp:
            stages = self.parse_lines(
                tmp,
                [
                    "# comment line",
                    "",
                    json.dumps(
                        {
                            "name": "stage_a",
                            "script": "scripts/a.sh",
                            "env": {"GAMES": 100, "TAG": "x"},
                            "notes": "prereg pointer",
                        }
                    ),
                    json.dumps({"name": "stage_b", "script": "scripts/b.sh"}),
                ],
            )
        self.assertEqual(
            stages,
            [
                {
                    "name": "stage_a",
                    "script": "scripts/a.sh",
                    "env": {"GAMES": "100", "TAG": "x"},
                },
                {"name": "stage_b", "script": "scripts/b.sh", "env": {}},
            ],
        )

    def test_example_queue_file_parses(self) -> None:
        stages = parse_queue(str(EXAMPLE_QUEUE))
        self.assertEqual(
            [stage["name"] for stage in stages],
            ["worlds_confirm_resume", "ghost_screen"],
        )
        self.assertEqual(
            stages[1]["env"],
            {"SCREEN_NAME": "ghost_opponents", "EXTRA_FLAGS": "--gumbel-ghost-opponents"},
        )

    def assert_rejects(self, lines, fragment):
        with TemporaryDirectory() as tmp, self.assertRaises(ValueError) as ctx:
            self.parse_lines(tmp, lines)
        self.assertIn(fragment, str(ctx.exception))

    def test_duplicate_names_rejected(self) -> None:
        line = json.dumps({"name": "twice", "script": "a.sh", "env": {}})
        self.assert_rejects([line, line], "duplicate stage name")

    def test_malformed_json_line_rejected_with_line_number(self) -> None:
        good = json.dumps({"name": "ok", "script": "a.sh", "env": {}})
        self.assert_rejects([good, "{not json"], ":2:")

    def test_non_object_line_rejected(self) -> None:
        self.assert_rejects(['["not", "an", "object"]'], "JSON object")

    def test_missing_name_rejected(self) -> None:
        self.assert_rejects([json.dumps({"script": "a.sh"})], "name")

    def test_missing_script_rejected(self) -> None:
        self.assert_rejects([json.dumps({"name": "x"})], "script")

    def test_unsafe_stage_name_rejected(self) -> None:
        # Names become queue_<name>.log / queue_done_<name>; path separators
        # must never reach the filesystem.
        self.assert_rejects([json.dumps({"name": "../escape", "script": "a.sh"})], "name")

    def test_unknown_keys_rejected(self) -> None:
        self.assert_rejects(
            [json.dumps({"name": "x", "script": "a.sh", "evn": {}})], "unknown keys"
        )

    def test_non_dict_env_rejected(self) -> None:
        self.assert_rejects(
            [json.dumps({"name": "x", "script": "a.sh", "env": "GAMES=100"})],
            "env must be a JSON object",
        )

    def test_bad_env_key_rejected(self) -> None:
        self.assert_rejects(
            [json.dumps({"name": "x", "script": "a.sh", "env": {"games": "100"}})],
            "invalid env key",
        )

    def test_reserved_source_revision_env_key_rejected(self) -> None:
        self.assert_rejects(
            [json.dumps({"name": "x", "script": "a.sh", "env": {"SOURCE_REVISION": "abc"}})],
            "reserved",
        )

    def test_control_characters_in_env_value_rejected(self) -> None:
        self.assert_rejects(
            [json.dumps({"name": "x", "script": "a.sh", "env": {"A": "one\ttwo"}})],
            "control characters",
        )

    def test_bool_env_value_rejected(self) -> None:
        self.assert_rejects(
            [json.dumps({"name": "x", "script": "a.sh", "env": {"FLAG": True}})],
            "string or number",
        )

    def test_empty_queue_rejected(self) -> None:
        self.assert_rejects(["# nothing but comments"], "no stages")

    def test_missing_scripts_reported_in_order_without_duplicates(self) -> None:
        with TemporaryDirectory() as tmp:
            present = Path(tmp) / "present.sh"
            present.write_text("#!/usr/bin/env bash\n", encoding="utf-8")
            stages = [
                {"name": "a", "script": "absent_one.sh", "env": {}},
                {"name": "b", "script": "present.sh", "env": {}},
                {"name": "c", "script": "absent_one.sh", "env": {}},
                {"name": "d", "script": "absent_two.sh", "env": {}},
            ]
            self.assertEqual(missing_scripts(stages, tmp), ["absent_one.sh", "absent_two.sh"])


class ShellEnvTest(unittest.TestCase):
    def test_empty_env_renders_empty_string(self) -> None:
        self.assertEqual(shell_env({}), "")

    def test_values_are_always_single_quoted(self) -> None:
        self.assertEqual(shell_env({"TAG": "plain"}), "TAG='plain'")

    def test_quoting_handles_spaces_and_quotes(self) -> None:
        rendered = shell_env({"A": "has space", "B": "it's", "C": 'say "hi"'})
        # Round-trip through shell word splitting: each pair must come back
        # as a single KEY=value token with the original value intact.
        self.assertEqual(
            shlex.split(rendered),
            ["A=has space", "B=it's", 'C=say "hi"'],
        )
        self.assertIn("B='it'\"'\"'s'", rendered)

    def test_numbers_are_normalized_to_strings(self) -> None:
        self.assertEqual(shell_env({"GAMES": 100}), "GAMES='100'")

    def test_bad_key_rejected(self) -> None:
        with self.assertRaises(ValueError):
            shell_env({"lower_case": "x"})
        with self.assertRaises(ValueError):
            shell_env({"9LEADING": "x"})

    def test_shell_stage_line_is_tab_separated(self) -> None:
        stage = {"name": "s1", "script": "run.sh", "env": {"A": "b c"}}
        self.assertEqual(shell_stage_line(stage), "s1\trun.sh\tA='b c'")


class RunnerIntegrationTest(unittest.TestCase):
    """Drive the real bash runner inside a sandboxed fake ROOT."""

    def build_root(self, tmp):
        root = Path(tmp) / "root"
        (root / "cascadiav3" / "logs").mkdir(parents=True)
        (root / "cascadiav3" / "scripts").mkdir()
        (root / "cascadiav3" / "src").symlink_to(CASCADIAV3 / "src")
        (root / "cascadiav3" / "scripts" / "lib_waiter.sh").symlink_to(
            CASCADIAV3 / "scripts" / "lib_waiter.sh"
        )
        # The fake ROOT is not a git worktree, so the runner falls back to
        # the deployed-revision marker (same path the house scripts use).
        (root / "cascadiav3" / "logs" / "exact_k1_deployed_revision.txt").write_text(
            SOURCE_REVISION + "\n", encoding="utf-8"
        )
        (root / "stage_ok.sh").write_text(
            "#!/usr/bin/env bash\n"
            "set -euo pipefail\n"
            'printf \'%s|%s|%s\\n\' "${GREETING:-unset}" "${QUOTED:-unset}"'
            ' "$SOURCE_REVISION" > stage_ok_output.txt\n',
            encoding="utf-8",
        )
        (root / "stage_fail.sh").write_text(
            "#!/usr/bin/env bash\necho 'about to fail'\nexit 3\n", encoding="utf-8"
        )
        return root

    def runner_env(self, root):
        env = dict(os.environ)
        env.update(
            {
                "ROOT": str(root),
                "SOURCE_REVISION": SOURCE_REVISION,
                "WAITER_POLL_SECONDS": "0.05",
                "PYTHON": sys.executable,
            }
        )
        return env

    def run_queue(self, root, queue_rel="queue.jsonl"):
        return subprocess.run(
            ["bash", str(RUNNER), queue_rel],
            cwd=str(root),
            env=self.runner_env(root),
            capture_output=True,
            text=True,
            timeout=60,
        )

    def default_stages(self):
        return [
            {
                "name": "ok_stage",
                "script": "stage_ok.sh",
                "env": {"GREETING": "hello world", "QUOTED": 'it\'s "quoted"'},
            },
            {"name": "fail_stage", "script": "stage_fail.sh", "env": {}},
        ]

    def test_queue_runs_stages_continues_past_failure_and_resumes(self) -> None:
        with TemporaryDirectory() as tmp:
            root = self.build_root(tmp)
            write_queue(root / "queue.jsonl", self.default_stages())
            logs = root / "cascadiav3" / "logs"

            first = self.run_queue(root)
            # fail_stage failed, so the runner exits 1 — after finishing the
            # whole queue and printing the summary table.
            self.assertEqual(first.returncode, 1, first.stdout + first.stderr)
            self.assertIn("ok_stage COMPLETE", first.stdout)
            self.assertIn("fail_stage FAILED exit=3", first.stdout)
            self.assertIn("queue summary", first.stdout)
            self.assertIn("queue finished: 1 complete, 1 failed, 0 skipped", first.stdout)

            # Env hand-off preserved spaces/quotes and pinned SOURCE_REVISION.
            self.assertEqual(
                (root / "stage_ok_output.txt").read_text(encoding="utf-8"),
                f'hello world|it\'s "quoted"|{SOURCE_REVISION}\n',
            )
            # Done marker, per-stage log, and pid file next to the log.
            self.assertTrue((logs / "queue_done_ok_stage").exists())
            self.assertFalse((logs / "queue_done_fail_stage").exists())
            self.assertIn(
                "about to fail",
                (logs / "queue_fail_stage.log").read_text(encoding="utf-8"),
            )
            self.assertTrue((logs / "queue_ok_stage.pid").exists())

            # Rerun: ok_stage is skipped via its done marker; fail_stage
            # reruns (and fails again).
            second = self.run_queue(root)
            self.assertEqual(second.returncode, 1, second.stdout + second.stderr)
            self.assertIn("ok_stage SKIPPED", second.stdout)
            self.assertIn("queue finished: 0 complete, 1 failed, 1 skipped", second.stdout)

    def test_invalid_queue_fails_closed_before_running_anything(self) -> None:
        with TemporaryDirectory() as tmp:
            root = self.build_root(tmp)
            stages = self.default_stages()
            stages[1]["script"] = "does_not_exist.sh"
            write_queue(root / "queue.jsonl", stages)

            result = self.run_queue(root)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("missing scripts", result.stderr)
            self.assertIn("does_not_exist.sh", result.stderr)
            # Fail closed: stage 1 must not have run even though its own
            # script and config were fine.
            self.assertFalse((root / "stage_ok_output.txt").exists())
            self.assertFalse((root / "cascadiav3" / "logs" / "queue_done_ok_stage").exists())

    def test_hold_file_pauses_queue_until_removed(self) -> None:
        with TemporaryDirectory() as tmp:
            root = self.build_root(tmp)
            write_queue(root / "queue.jsonl", [self.default_stages()[0]])
            hold = root / "cascadiav3" / "logs" / "HOLD_experiment_queue"
            hold.touch()

            proc = subprocess.Popen(
                ["bash", str(RUNNER), "queue.jsonl"],
                cwd=str(root),
                env=self.runner_env(root),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
            )
            try:
                time.sleep(1.0)
                self.assertIsNone(proc.poll(), "runner exited while HOLD present")
                self.assertFalse((root / "stage_ok_output.txt").exists())
                hold.unlink()
                stdout, _ = proc.communicate(timeout=30)
            finally:
                if proc.poll() is None:
                    proc.kill()
                    proc.communicate()
            self.assertEqual(proc.returncode, 0, stdout)
            self.assertIn("paused by", stdout)
            self.assertTrue((root / "stage_ok_output.txt").exists())


if __name__ == "__main__":
    unittest.main()
