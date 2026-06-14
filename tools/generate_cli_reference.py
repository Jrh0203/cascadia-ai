#!/usr/bin/env python3
"""Generate or verify the checked-in Cascadia v2 CLI reference."""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BINARY = ROOT / "target/debug/cascadia-v2"
DEFAULT_OUTPUT = ROOT / "docs/v2/CLI_REFERENCE.md"


def command_help(binary: Path, *arguments: str) -> str:
    result = subprocess.run(
        [str(binary), *arguments, "--help"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.rstrip()


def command_names(top_level_help: str) -> list[str]:
    commands: list[str] = []
    in_commands = False
    for line in top_level_help.splitlines():
        if line == "Commands:":
            in_commands = True
            continue
        if in_commands and line == "Options:":
            break
        if in_commands and line.startswith("  ") and not line.startswith("    "):
            name = line.strip().split(maxsplit=1)[0]
            if name != "help":
                commands.append(name)
    return commands


def render_reference(binary: Path) -> str:
    top_level = command_help(binary)
    sections = [
        "# Cascadia V2 CLI Reference",
        "",
        "Generated from the typed Clap schema. Regenerate with `make cli-docs`.",
        "",
        "## Top Level",
        "",
        "```text",
        top_level,
        "```",
    ]
    for name in command_names(top_level):
        sections.extend(
            [
                "",
                f"## `{name}`",
                "",
                "```text",
                command_help(binary, name),
                "```",
            ]
        )
    return "\n".join(sections) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--check", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rendered = render_reference(args.binary.resolve())
    output = args.output.resolve()
    if args.check:
        if not output.exists() or output.read_text() != rendered:
            raise SystemExit(f"{output} is stale; run make cli-docs")
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(rendered)


if __name__ == "__main__":
    main()
