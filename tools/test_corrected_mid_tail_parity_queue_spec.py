from __future__ import annotations

import importlib.util
import json
import stat
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).with_name("corrected_mid_tail_parity_queue_spec.py")
SPEC = importlib.util.spec_from_file_location(
    "corrected_mid_tail_parity_queue_spec",
    MODULE_PATH,
)
assert SPEC is not None and SPEC.loader is not None
queue_spec = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = queue_spec
SPEC.loader.exec_module(queue_spec)


def _implementation() -> dict[str, object]:
    return {
        "identity_kind": "test-implementation",
        "bundle_blake3": "a" * 64,
        "files": [
            {
                "label": "module",
                "relative_file": "python/cascadia_mlx/module.py",
                "bytes": 4,
                "blake3": "b" * 64,
            }
        ],
    }


def _by_id(spec: dict) -> dict[str, dict]:
    return {task["id"]: task for task in spec["tasks"]}


def test_queue_spec_maximizes_four_host_parallelism_without_applying() -> None:
    spec = queue_spec.build_queue_spec(_implementation())
    assert spec["review_status"] == "generated-not-applied"
    assert spec["shared_queue_mutated"] is False
    assert spec["task_count"] == 15
    assert spec["allocation"]["shard_count_by_host"] == {
        "john1": 3,
        "john2": 3,
        "john3": 2,
        "john4": 2,
    }
    assert spec["allocation"]["maximum_concurrent_shards"] == 4
    assert spec["scientific_blake3"] == queue_spec.scientific_blake3(
        {key: value for key, value in spec.items() if key != "scientific_blake3"}
    )


def test_shard_commands_are_pinned_and_use_no_row_limit() -> None:
    spec = queue_spec.build_queue_spec(_implementation())
    tasks = _by_id(spec)
    for shard_index in range(10):
        task = tasks[f"f5p-shard-{shard_index:05d}"]
        host = queue_spec.assigned_host(shard_index)
        assert task["compatible_hosts"] == [host]
        assert task["dependencies"] == ["f5p-source-bundle-fanout"]
        assert task["resources"]["uses_mlx"] is True
        assert "--row-limit" not in task["command"]
        assert task["command"][task["command"].index("--shard-index") + 1] == str(shard_index)
        assert (
            task["command"][task["command"].index("--expected-implementation-blake3") + 1]
            == "a" * 64
        )
        assert task["command"][1] == "-B"
        assert "/source/blake3/" + "a" * 64 + "/" in task["command"][2]


def test_source_bundle_fanout_is_exact_and_precedes_every_shard() -> None:
    spec = queue_spec.build_queue_spec(_implementation())
    task = _by_id(spec)["f5p-source-bundle-fanout"]
    assert task["compatible_hosts"] == ["john1"]
    assert task["dependencies"] == []
    assert task["command"].count("--destination") == 3
    assert task["command"].count("--required-file") == 2
    assert "--verify-tree" in task["command"]
    assert spec["source_bundle"]["fanout_task"] == task["id"]
    assert spec["source_bundle"]["required_files"] == [
        "python/cascadia_mlx/module.py",
        "manifest.json",
    ]


def test_collection_and_dual_aggregate_graph_are_exact() -> None:
    spec = queue_spec.build_queue_spec(_implementation())
    tasks = _by_id(spec)
    collection = tasks["f5p-collect-remote-reports"]
    assert collection["command"].count("--artifact") == 7
    assert len(collection["dependencies"]) == 10
    assert all(
        not source.startswith("john1:")
        for index, source in enumerate(collection["command"])
        if index > 0 and collection["command"][index - 1] == "--artifact"
    )

    forward = tasks["f5p-aggregate-forward"]["command"]
    reverse = tasks["f5p-aggregate-reverse"]["command"]
    forward_reports = [
        forward[index + 1] for index, value in enumerate(forward) if value == "--report"
    ]
    reverse_reports = [
        reverse[index + 1] for index, value in enumerate(reverse) if value == "--report"
    ]
    assert len(forward_reports) == 10
    assert reverse_reports == list(reversed(forward_reports))
    order_proof = tasks["f5p-aggregate-order-proof"]
    assert order_proof["dependencies"] == [
        "f5p-aggregate-forward",
        "f5p-aggregate-reverse",
    ]
    assert order_proof["command"][:2] == ["/usr/bin/cmp", "-s"]


def test_immutable_writer_is_idempotent_and_rejects_drift(tmp_path: Path) -> None:
    output = tmp_path / "queue-spec.json"
    value = queue_spec.build_queue_spec(_implementation())
    assert queue_spec.write_immutable_json(output, value) is False
    assert queue_spec.write_immutable_json(output, value) is True
    assert stat.S_IMODE(output.stat().st_mode) & 0o222 == 0
    changed = json.loads(json.dumps(value))
    changed["shared_queue_mutated"] = True
    with pytest.raises(queue_spec.QueueSpecError, match="differs"):
        queue_spec.write_immutable_json(output, changed)


def test_source_bundle_is_content_addressed_idempotent_and_fail_closed(
    tmp_path: Path,
) -> None:
    relative = Path("python/cascadia_mlx/module.py")
    source = tmp_path / relative
    source.parent.mkdir(parents=True)
    source.write_bytes(b"code")
    implementation = {
        "identity_kind": "test-implementation",
        "bundle_blake3": "c" * 64,
        "files": [
            {
                "label": "module",
                "relative_file": relative.as_posix(),
                "bytes": source.stat().st_size,
                "blake3": queue_spec.checksum_file(source),
            }
        ],
    }

    bundle = queue_spec.prepare_source_bundle(tmp_path, implementation)
    assert bundle == tmp_path / queue_spec.source_bundle_root("c" * 64)
    assert (bundle / relative).read_bytes() == b"code"
    assert stat.S_IMODE((bundle / relative).stat().st_mode) & 0o222 == 0
    assert stat.S_IMODE(bundle.stat().st_mode) & 0o222 == 0
    assert queue_spec.prepare_source_bundle(tmp_path, implementation) == bundle

    (bundle / relative).chmod(0o644)
    (bundle / relative).write_bytes(b"drift")
    with pytest.raises(queue_spec.QueueSpecError, match="differs"):
        queue_spec.prepare_source_bundle(tmp_path, implementation)
