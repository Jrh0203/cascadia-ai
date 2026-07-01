import json
from pathlib import Path

import blake3
import relational_feature_foundation_campaign as campaign


def _bundle(tmp_path: Path) -> Path:
    binary = tmp_path / "binary"
    binary.write_bytes(b"relational feature test binary")
    binary_blake3 = campaign.file_blake3(binary)
    identity = {
        "schema_version": 1,
        "experiment_id": campaign.CAMPAIGN_ID,
        "source_files": [],
        "binaries": [
            {
                "name": campaign.BINARY_NAME,
                "bytes": binary.stat().st_size,
                "blake3": binary_blake3,
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


def _metrics(lane: campaign.Lane) -> dict:
    positions = lane.games * campaign.POSITIONS_PER_GAME
    if lane.key == "r5":
        return {
            "positions": positions,
            "complete_actions": 10,
            "current_score_decoder_checks": positions * 4,
            "control_affordance_checks": 10,
            "quotient_affordance_underdetermined": 10,
            "local_affordance_checks": 10,
            "local_score_delta_checks": 10,
        }
    if lane.key == "r6":
        return {
            "positions": positions,
            "complete_actions": 10,
            "exact_apply_checks": 10,
            "exact_undo_checks": 10,
        }
    if lane.key == "s3":
        return {
            "positions": positions,
            "board_score_decoder_checks": positions * 4,
            "action_delta_decoder_checks": positions,
            "d6_invariance_checks": positions * 12,
            "boards_with_elk_extensions": 1,
            "boards_with_salmon_continuations": 1,
            "boards_with_hawk_opportunities": 1,
            "boards_with_bear_pair_opportunities": 1,
        }
    feature_scales = {
        f"feature.{index}": {"count": positions * 64} for index in range(154)
    }
    return {
        "positions": positions,
        "sampled_actions": positions * 64,
        "exact_replay_checks": positions * 64,
        "score_delta_checks": positions * 64,
        "feature_field_count": 154,
        "feature_scales": feature_scales,
    }


def _report(
    lane: campaign.Lane,
    bundle_id: str,
    executable_blake3: str,
) -> dict:
    scientific = {
        "schema_version": 1,
        "artifact_kind": "relational_feature_census_report",
        "experiment_id": lane.experiment_id,
        "protocol_id": lane.protocol_id,
        "source_bundle_id": bundle_id,
        "config": {
            "lane": {
                "r5": "r5-quotient",
                "r6": "r6-incremental",
                "s3": "s3-component-motif",
                "s5": "s5-derivatives",
            }[lane.key],
            "first_seed": lane.first_seed,
            "games": lane.games,
            "source_bundle_id": bundle_id,
            "host": lane.host,
            "rayon_threads": lane.rayon_threads,
        },
        "corpus": {
            "first_seed": lane.first_seed,
            "games": lane.games,
            "positions": lane.games * campaign.POSITIONS_PER_GAME,
            "seeds_blake3": "0" * 64,
        },
        "metrics": _metrics(lane),
        "passed": True,
        "classification": lane.passing_classification,
    }
    return {
        "scientific": scientific,
        "scientific_blake3": blake3.blake3(
            campaign.canonical_json(scientific)
        ).hexdigest(),
        "execution": {
            "host": lane.host,
            "started_unix_ms": 100,
            "completed_unix_ms": 110,
            "elapsed_ms": 10,
            "executable_blake3": executable_blake3,
        },
    }


def test_task_graph_runs_four_distinct_host_pinned_lanes(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    repository = bundle.parents[2]
    tasks = campaign.build_task_specs(repository, bundle)
    by_id = {task["id"]: task for task in tasks}
    assert len(tasks) == 7
    assert len(by_id) == len(tasks)

    fanout_id = f"{campaign.TASK_PREFIX}-fanout-bundle"
    run_ids = set()
    seen_seeds = set()
    for lane in campaign.LANES:
        task_id = f"{campaign.TASK_PREFIX}-run-{lane.key}-{lane.host}"
        run_ids.add(task_id)
        task = by_id[task_id]
        assert task["workload_class"] == "independent-experiment"
        assert task["compatible_hosts"] == [lane.host]
        assert task["dependencies"] == [fanout_id]
        assert task["command"][task["command"].index("--first-seed") + 1] == str(
            lane.first_seed
        )
        seen_seeds.add(lane.first_seed)
    assert len(seen_seeds) == 4

    collect = by_id[f"{campaign.TASK_PREFIX}-collect"]
    assert set(collect["dependencies"]) == run_ids
    assert collect["command"].count("--artifact") == 4
    aggregate = by_id[f"{campaign.TASK_PREFIX}-aggregate"]
    assert aggregate["dependencies"] == [f"{campaign.TASK_PREFIX}-collect"]
    assert aggregate["decision_terminal"] is True


def test_aggregate_is_order_invariant_and_hash_bound(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = campaign._bundle_manifest(bundle)
    bundle_id = manifest["bundle_id"]
    executable_blake3 = manifest["identity"]["binaries"][0]["blake3"]
    reports = [
        (lane, _report(lane, bundle_id, executable_blake3))
        for lane in campaign.LANES
    ]
    forward = campaign.aggregate_reports(
        reports,
        bundle_id=bundle_id,
        executable_blake3=executable_blake3,
    )
    reverse = campaign.aggregate_reports(
        reversed(reports),
        bundle_id=bundle_id,
        executable_blake3=executable_blake3,
    )
    assert forward["scientific_blake3"] == reverse["scientific_blake3"]
    assert forward["scientific"]["all_foundations_passed"] is True
    assert (
        forward["scientific"]["classification"]
        == "relational_feature_foundations_authorized"
    )


def test_tampered_scientific_payload_is_rejected(tmp_path: Path) -> None:
    bundle = _bundle(tmp_path)
    manifest = campaign._bundle_manifest(bundle)
    lane = campaign.LANES[0]
    report = _report(
        lane,
        manifest["bundle_id"],
        manifest["identity"]["binaries"][0]["blake3"],
    )
    report["scientific"]["metrics"]["positions"] += 1
    try:
        campaign.validate_lane_report(
            report,
            lane,
            bundle_id=manifest["bundle_id"],
            executable_blake3=manifest["identity"]["binaries"][0]["blake3"],
        )
    except campaign.CampaignError as error:
        assert "scientific hash" in str(error)
    else:
        raise AssertionError("tampered report was accepted")
