import json
from pathlib import Path

import blake3
import s6_topology_campaign as campaign


def _bundle(tmp_path: Path) -> Path:
    binary = tmp_path / campaign.BINARY_NAME
    binary.write_bytes(b"s6 topology test binary")
    identity = {
        "schema_version": 1,
        "experiment_id": campaign.EXPERIMENT_ID,
        "source_files": [],
        "binaries": [
            {
                "name": campaign.BINARY_NAME,
                "bytes": binary.stat().st_size,
                "blake3": campaign.file_blake3(binary),
            }
        ],
    }
    bundle_id = blake3.blake3(
        json.dumps(
            identity,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode()
    ).hexdigest()
    bundle = tmp_path / "repository" / "bundle" / bundle_id
    (bundle / "bin").mkdir(parents=True)
    (bundle / "bin" / campaign.BINARY_NAME).write_bytes(binary.read_bytes())
    (bundle / "bundle.json").write_text(
        json.dumps({"bundle_id": bundle_id, "identity": identity})
    )
    return bundle


def _distribution(count: int, value: int) -> dict:
    return {
        "count": count,
        "minimum": value,
        "mean_milli": value * 1000,
        "median": value,
        "p90": value,
        "p99": value,
        "maximum": value,
    }


def _report(
    shard: campaign.Shard,
    bundle_id: str,
    executable_blake3: str,
) -> dict:
    positions = shard.games * campaign.POSITIONS_PER_GAME
    boards = positions * 4
    metrics = {
        "positions": positions,
        "board_encodings": boards,
        "topology_decoder_checks": boards * 11,
        "topology_decoder_failures": 0,
        "d6_invariance_checks": positions * 12,
        "d6_invariance_failures": 0,
        "adversarial_checks": 4,
        "adversarial_failures": 0,
        "baseline_collision_pairs": 1_000,
        "topology_separated_pairs": 400,
        "path_separated_pairs": 100,
        "random_walk_separated_pairs": 300,
        "spectral_separated_pairs": 350,
        "full_encoding_separated_pairs": 500,
        "long_range_collision_pairs": 200,
        "long_range_separated_pairs": 100,
        "unique_topology_encodings": 500,
        "unique_path_encodings": 50,
        "unique_random_walk_encodings": 400,
        "unique_spectral_encodings": 450,
        "unique_full_encodings": 700,
        "boards_with_long_range_paths": 200,
        "boards_with_geometric_holes": 0,
        "full_separation_rate_ppm": 500_000,
        "encoding_bytes": _distribution(positions, 900),
        "extraction_ns": _distribution(positions, 500_000),
        "isolated_extraction_ns": _distribution(
            campaign.POSITIONS_PER_GAME, 400_000
        ),
        "exactness_gate_pass": True,
        "d6_gate_pass": True,
        "adversarial_gate_pass": True,
        "feature_variation_gate_pass": True,
        "long_range_gate_pass": True,
        "isolated_latency_gate_pass": True,
        "compactness_gate_pass": True,
    }
    scientific = {
        "schema_version": 1,
        "artifact_kind": "relational_feature_census_report",
        "experiment_id": campaign.EXPERIMENT_ID,
        "protocol_id": campaign.PROTOCOL_ID,
        "source_bundle_id": bundle_id,
        "config": {
            "lane": "s6-topology",
            "first_seed": shard.first_seed,
            "games": shard.games,
            "source_bundle_id": bundle_id,
            "host": shard.host,
            "rayon_threads": shard.rayon_threads,
        },
        "corpus": {
            "first_seed": shard.first_seed,
            "games": shard.games,
            "positions": positions,
            "seeds_blake3": "0" * 64,
        },
        "metrics": metrics,
        "passed": True,
        "classification": campaign.PASSING_CLASSIFICATION,
    }
    return {
        "scientific": scientific,
        "scientific_blake3": blake3.blake3(
            campaign.canonical_json(scientific)
        ).hexdigest(),
        "execution": {
            "host": shard.host,
            "started_unix_ms": 100,
            "completed_unix_ms": 110,
            "elapsed_ms": 10,
            "executable_blake3": executable_blake3,
        },
    }


def test_task_graph_pins_four_disjoint_seed_blocks(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    repository = bundle.parents[2]
    tasks = campaign.build_task_specs(repository, bundle)
    by_id = {task["id"]: task for task in tasks}
    assert len(tasks) == 7
    assert len(by_id) == len(tasks)
    run_ids = set()
    seed_blocks = set()
    for shard in campaign.SHARDS:
        task_id = f"{campaign.TASK_PREFIX}-run-{shard.host}"
        run_ids.add(task_id)
        task = by_id[task_id]
        assert task["workload_class"] == "divisible-evidence"
        assert task["compatible_hosts"] == [shard.host]
        assert task["dependencies"] == [f"{campaign.TASK_PREFIX}-fanout"]
        seed_blocks.add(
            task["command"][task["command"].index("--first-seed") + 1]
        )
    assert len(seed_blocks) == 4
    assert (
        set(by_id[f"{campaign.TASK_PREFIX}-collect"]["dependencies"]) == run_ids
    )
    assert (
        by_id[f"{campaign.TASK_PREFIX}-aggregate"]["dependencies"]
        == [f"{campaign.TASK_PREFIX}-collect"]
    )


def test_aggregate_is_order_invariant_and_conservative(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = campaign._bundle_manifest(bundle)
    bundle_id = manifest["bundle_id"]
    executable = manifest["identity"]["binaries"][0]["blake3"]
    reports = [
        (shard, _report(shard, bundle_id, executable))
        for shard in campaign.SHARDS
    ]
    forward = campaign.aggregate_reports(
        reports,
        bundle_id=bundle_id,
        executable_blake3=executable,
    )
    reverse = campaign.aggregate_reports(
        reversed(reports),
        bundle_id=bundle_id,
        executable_blake3=executable,
    )
    assert forward["scientific_blake3"] == reverse["scientific_blake3"]
    assert forward["scientific"]["all_hosts_passed"] is True
    assert (
        forward["scientific"]["totals"]["baseline_collision_pairs"] == 4_000
    )
    assert (
        forward["scientific"]["totals"]["full_encoding_separated_pairs"] == 2_000
    )


def test_tampered_scientific_payload_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = campaign._bundle_manifest(bundle)
    shard = campaign.SHARDS[0]
    report = _report(
        shard,
        manifest["bundle_id"],
        manifest["identity"]["binaries"][0]["blake3"],
    )
    report["scientific"]["metrics"]["positions"] += 1
    try:
        campaign.validate_shard_report(
            report,
            shard,
            bundle_id=manifest["bundle_id"],
            executable_blake3=manifest["identity"]["binaries"][0]["blake3"],
        )
    except campaign.CampaignError as error:
        assert "scientific hash" in str(error)
    else:
        raise AssertionError("tampered report was accepted")
