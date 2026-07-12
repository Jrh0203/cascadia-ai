"""Fail-closed parsing and validation for experiment-queue JSONL files.

Consumed by ``cascadiav3/scripts/run_experiment_queue.sh``: the runner imports
this module from a python3 heredoc, validates the whole queue file BEFORE any
stage runs (AGENTS.md fail-closed rule), and receives one TAB-separated
``name<TAB>script<TAB>env-prefix`` line per stage. Standard library only.

Queue file contract (one JSON object per line; blank lines and full-line
``#`` comments are skipped):

    {"name": "<stage>", "script": "cascadiav3/scripts/run_x.sh",
     "env": {"KEY": "value", ...}, "notes": "optional annotation"}

- ``name``   becomes ``queue_<name>.log`` / ``queue_done_<name>`` filenames,
  so it must match ``[A-Za-z0-9][A-Za-z0-9._-]*`` and be unique in the file.
- ``script`` is a path relative to the repo root (john0:
  ``/home/john0/cascadia``); existence is checked via ``missing_scripts``.
- ``env``    keys must match ``[A-Z][A-Z0-9_]*``. ``SOURCE_REVISION`` is
  reserved — the runner pins it for every stage and a queue entry must not
  try to override it. Values may be strings or numbers (normalized to
  strings); control characters (tabs/newlines) are rejected because the
  shell hand-off format is line- and TAB-delimited.
"""

from __future__ import annotations

import json
import os
import re

ENV_KEY_PATTERN = re.compile(r"^[A-Z][A-Z0-9_]*$")
STAGE_NAME_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")

_ALLOWED_STAGE_KEYS = frozenset({"name", "script", "env", "notes"})
_RESERVED_ENV_KEYS = frozenset({"SOURCE_REVISION"})
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


def _normalize_env(env: dict, where: str) -> dict:
    """Validate an env mapping and normalize its values to strings."""
    if not isinstance(env, dict):
        raise ValueError(f"{where}: env must be a JSON object, got {type(env).__name__}")
    normalized = {}
    for key, value in env.items():
        if not isinstance(key, str) or not ENV_KEY_PATTERN.match(key):
            raise ValueError(f"{where}: invalid env key {key!r} (must match [A-Z][A-Z0-9_]*)")
        if key in _RESERVED_ENV_KEYS:
            raise ValueError(
                f"{where}: env key {key} is reserved (the runner pins it for every stage)"
            )
        if isinstance(value, bool) or not isinstance(value, (str, int, float)):
            raise ValueError(
                f"{where}: env value for {key} must be a string or number,"
                f" got {type(value).__name__}"
            )
        text = value if isinstance(value, str) else str(value)
        if _CONTROL_CHARS.search(text):
            raise ValueError(
                f"{where}: env value for {key} contains control characters"
                " (tabs/newlines are not allowed)"
            )
        normalized[key] = text
    return normalized


def parse_queue(path: str) -> list[dict]:
    """Parse a JSONL queue file into ``[{"name", "script", "env"}, ...]``.

    Raises ValueError (naming the offending line) on: malformed JSON, a
    non-object line, unknown keys, a missing/invalid ``name`` or ``script``,
    a duplicate stage name, a non-dict ``env``, an invalid or reserved env
    key, a non-scalar env value, control characters anywhere, or a queue
    with no stages at all. Fail closed: any error rejects the whole file.
    """
    stages = []
    seen = set()
    with open(path, encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            where = f"{path}:{lineno}"
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{where}: invalid JSON: {exc}") from exc
            if not isinstance(record, dict):
                raise ValueError(f"{where}: each line must be a JSON object")
            unknown = sorted(key for key in record if key not in _ALLOWED_STAGE_KEYS)
            if unknown:
                raise ValueError(
                    f"{where}: unknown keys: {', '.join(unknown)}"
                    " (allowed: name, script, env, notes)"
                )
            name = record.get("name")
            if not isinstance(name, str) or not STAGE_NAME_PATTERN.match(name):
                raise ValueError(
                    f"{where}: name must match [A-Za-z0-9][A-Za-z0-9._-]*"
                    f" (it becomes queue_<name>.log / queue_done_<name>), got {name!r}"
                )
            if name in seen:
                raise ValueError(f"{where}: duplicate stage name {name!r}")
            seen.add(name)
            script = record.get("script")
            if (
                not isinstance(script, str)
                or not script
                or _CONTROL_CHARS.search(script)
            ):
                raise ValueError(
                    f"{where}: script must be a non-empty path string, got {script!r}"
                )
            notes = record.get("notes", "")
            if not isinstance(notes, str):
                raise ValueError(f"{where}: notes must be a string")
            stages.append(
                {
                    "name": name,
                    "script": script,
                    "env": _normalize_env(record.get("env", {}), where),
                }
            )
    if not stages:
        raise ValueError(f"{path}: queue file contains no stages")
    return stages


def _single_quote(value: str) -> str:
    """Wrap value in single quotes, escaping embedded single quotes."""
    return "'" + value.replace("'", "'\"'\"'") + "'"


def shell_env(env: dict) -> str:
    """Render env as a ``KEY='value' ...`` prefix safe to eval before a command.

    Keys are validated against ``[A-Z][A-Z0-9_]*`` and values are always
    single-quoted (embedded single quotes use the standard ``'\"'\"'``
    escape), so the result is safe to splice in front of ``env ... bash``.
    Raises ValueError on any key or value ``parse_queue`` would reject.
    """
    normalized = _normalize_env(env, "shell_env")
    return " ".join(f"{key}={_single_quote(value)}" for key, value in normalized.items())


def missing_scripts(stages: list[dict], root: str = ".") -> list[str]:
    """Return the queue's script paths that do not exist under root.

    Order-preserving and de-duplicated; an empty list means every stage's
    script is present (running the same script twice with different env is
    legitimate and not flagged).
    """
    missing = []
    for stage in stages:
        script = stage["script"]
        if script not in missing and not os.path.isfile(os.path.join(root, script)):
            missing.append(script)
    return missing


def shell_stage_line(stage: dict) -> str:
    """One ``name<TAB>script<TAB>env-prefix`` line for the bash runner.

    TAB-safe by construction: names, scripts, and env values with control
    characters are rejected at parse time.
    """
    return "\t".join((stage["name"], stage["script"], shell_env(stage["env"])))


def _main(argv: list[str]) -> int:
    """CLI: validate a queue file and print the shell hand-off lines.

    ``python3 -m cascadiav3.experiment_queue <queue-file> [--root <dir>]``
    exits non-zero with a message on any validation failure, so a queue can
    be checked locally before it is shipped to john0.
    """
    import argparse

    parser = argparse.ArgumentParser(
        description="Validate an experiment-queue JSONL file (fail closed)."
    )
    parser.add_argument("queue_file", help="path to the JSONL queue file")
    parser.add_argument(
        "--root",
        default=".",
        help="repo root that stage script paths are relative to (default: .)",
    )
    args = parser.parse_args(argv)
    stages = parse_queue(args.queue_file)
    missing = missing_scripts(stages, args.root)
    if missing:
        raise SystemExit(
            f"{args.queue_file}: queue references missing scripts: {', '.join(missing)}"
        )
    for stage in stages:
        print(shell_stage_line(stage))
    return 0


if __name__ == "__main__":
    import sys

    try:
        raise SystemExit(_main(sys.argv[1:]))
    except ValueError as exc:
        raise SystemExit(f"queue validation failed: {exc}")
