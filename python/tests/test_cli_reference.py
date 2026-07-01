from __future__ import annotations

import importlib.util
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "generate_cli_reference",
    TOOLS / "generate_cli_reference.py",
)
assert SPEC is not None and SPEC.loader is not None
generator = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(generator)


def test_command_names_reads_only_top_level_commands() -> None:
    help_text = """Cascadia AI v2

Commands:
  benchmark
          Run a benchmark
  compare
          Compare strategies
  help
          Print help

Options:
  -h, --help
"""
    assert generator.command_names(help_text) == ["benchmark", "compare"]
