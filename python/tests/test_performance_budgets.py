from __future__ import annotations

import importlib.util
import json
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[2] / "tools"
SPEC = importlib.util.spec_from_file_location(
    "verify_performance_budgets",
    TOOLS / "verify_performance_budgets.py",
)
assert SPEC is not None and SPEC.loader is not None
budgets = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(budgets)


def write_fixture(tmp_path: Path, actual: float, operator: str, threshold: float) -> Path:
    (tmp_path / "report.json").write_text(json.dumps({"metric": actual}))
    config = {
        "schema_version": 1,
        "profile_id": "test-profile",
        "reports": {"fixture": "report.json"},
        "gates": [
            {
                "id": "fixture-gate",
                "report": "fixture",
                "pointer": "/metric",
                "operator": operator,
                "threshold": threshold,
                "unit": "widgets",
            }
        ],
    }
    path = tmp_path / "budgets.json"
    path.write_text(json.dumps(config))
    return path


def test_json_pointer_supports_objects_arrays_and_escaping() -> None:
    document = {"items": [{"a/b": {"~key": 3}}]}
    assert budgets.json_pointer(document, "/items/0/a~1b/~0key") == 3


def test_qualification_passes_at_inclusive_boundary(tmp_path: Path) -> None:
    config = write_fixture(tmp_path, actual=10.0, operator="<=", threshold=10.0)
    result = budgets.qualify(config, root=tmp_path)
    assert result["passed"]
    assert result["gates"][0]["passed"]
    assert "Verdict: **PASS**" in budgets.render_markdown(result)


def test_qualification_fails_when_minimum_is_missed(tmp_path: Path) -> None:
    config = write_fixture(tmp_path, actual=9.0, operator=">=", threshold=10.0)
    result = budgets.qualify(config, root=tmp_path)
    assert not result["passed"]
    assert not result["gates"][0]["passed"]
