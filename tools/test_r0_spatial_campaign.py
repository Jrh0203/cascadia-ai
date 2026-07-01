from __future__ import annotations

from itertools import pairwise
from pathlib import Path

import pytest
import r0_spatial_campaign as campaign


def _specs() -> list[dict]:
    return campaign.build_task_specs(
        bundle_relative=Path("artifacts/experiments/r0/bundles/example"),
    )


def _by_id(specs: list[dict]) -> dict[str, dict]:
    return {spec["id"]: spec for spec in specs}


def test_dataset_parts_are_disjoint_and_hit_exact_stage_one_totals() -> None:
    parts = campaign.dataset_parts()
    train = [part for part in parts if part.split == "train"]
    validation = [part for part in parts if part.split == "validation"]
    assert sum(part.games for part in train) == 625
    assert sum(part.games * 80 for part in train) == 50_000
    assert sum(part.games for part in validation) == 125
    assert sum(part.games * 80 for part in validation) == 10_000
    for split_parts in (train, validation):
        for left, right in pairwise(split_parts):
            assert left.first_game_index + left.games == right.first_game_index
    assert train[-1].first_game_index + train[-1].games == 200_625
    assert validation[-1].first_game_index + validation[-1].games == 210_125


def test_graph_contains_collection_fanout_and_twelve_process_benchmarks() -> None:
    specs = _specs()
    by_id = _by_id(specs)
    assert len(specs) == 33
    assert len(by_id) == len(specs)
    assert sum(spec["id"].startswith("r0f-collect-") for spec in specs) == 8
    assert sum(spec["id"].startswith("r0f-fanout-") for spec in specs) == 8
    benchmark_ids = [spec["id"] for spec in specs if spec["id"].startswith("r0f-benchmark-shard-")]
    assert len(benchmark_ids) == 12
    for shard_index, host in enumerate(campaign.HOSTS):
        for replicate_index in range(3):
            task = by_id[f"r0f-benchmark-shard-{shard_index}-replicate-{replicate_index}"]
            assert task["compatible_hosts"] == [host]
            assert task["command"][-4:] == [
                "--replicate-index",
                str(replicate_index),
                "--output",
                str(campaign.REMOTE_ROOTS[host] / task["artifact_path"]),
            ]
            assert task["command"].count("--dataset-root") == 8
            assert task["command"][:3] == [
                "/usr/bin/env",
                "-C",
                str(
                    campaign.REMOTE_ROOTS[host] / "artifacts/experiments/r0/bundles/example/source"
                ),
            ]


def test_every_benchmark_waits_for_all_eight_verified_dataset_parts() -> None:
    specs = _specs()
    expected = {
        f"r0f-fanout-{split}-part-{part_index}"
        for split in ("train", "validation")
        for part_index in range(4)
    }
    for spec in specs:
        if spec["id"].startswith("r0f-benchmark-shard-"):
            assert set(spec["dependencies"]) == expected


def test_remote_collection_contains_exactly_nine_nonlocal_reports() -> None:
    task = _by_id(_specs())["r0f-benchmark-report-collection"]
    assert task["command"].count("--artifact") == 9
    assert len(task["dependencies"]) == 12
    assert all("shard-0-" not in value for value in task["command"] if ":" in value)


def test_classifiers_require_three_replicates_and_opposite_report_orders() -> None:
    by_id = _by_id(_specs())
    forward = by_id["r0f-extraction-classification-forward"]["command"]
    reverse = by_id["r0f-extraction-classification-reverse"]["command"]
    assert forward.count("--report") == 12
    assert reverse.count("--report") == 12
    assert forward[forward.index("--required-replicates") + 1] == "3"
    assert reverse[reverse.index("--required-replicates") + 1] == "3"
    forward_reports = [
        forward[index + 1] for index, value in enumerate(forward) if value == "--report"
    ]
    reverse_reports = [
        reverse[index + 1] for index, value in enumerate(reverse) if value == "--report"
    ]
    assert reverse_reports == list(reversed(forward_reports))


@pytest.mark.parametrize(
    ("iterations", "replicates"),
    [(0, 3), (50, 2)],
)
def test_invalid_scientific_budget_is_rejected(iterations: int, replicates: int) -> None:
    with pytest.raises(campaign.CampaignError):
        campaign.build_task_specs(
            bundle_relative=Path("artifacts/bundle"),
            benchmark_iterations=iterations,
            required_replicates=replicates,
        )


def test_bundle_source_validation_requires_complete_provenance_roots() -> None:
    complete_paths = [
        *campaign.REQUIRED_SOURCE_FILES,
        *(f"{prefix}placeholder" for prefix in campaign.REQUIRED_SOURCE_PREFIXES),
    ]
    manifest = {
        "identity": {
            "source_files": [{"path": path} for path in complete_paths],
        }
    }
    campaign.validate_provenance_source_bundle(manifest)
    manifest["identity"]["source_files"] = [
        entry
        for entry in manifest["identity"]["source_files"]
        if entry["path"] != "CASCADIA_V2_GOAL.txt"
    ]
    with pytest.raises(campaign.CampaignError, match="CASCADIA_V2_GOAL"):
        campaign.validate_provenance_source_bundle(manifest)
