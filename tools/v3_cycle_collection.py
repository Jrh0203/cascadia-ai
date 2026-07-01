#!/usr/bin/env python3
"""Execute and validate one topology-free 10K-game V3 expert collection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import v3_phase2_jobs
from cascadia_cluster import ContainerInput, Resources
from v3_model_stage import _digest, bundle_environment, model_identity, stage_model
from v3_phase2_pipeline import (
    PipelineError,
    _client,
    _monitor,
    _validate_fabric,
    _validate_image,
    _write_atomic,
)

V1_POLICY_ID = "canonical-action-qualified-v1-direct-nnue-v4opp-v1"


def expected_policy_seats_for_ranges(
    cycle: int, ranges: list[tuple[int, int]]
) -> tuple[int, int]:
    if not 1 <= cycle <= 10:
        raise PipelineError("expert collection cycle is outside 1..=10")
    v1 = 0
    prior = 0
    for first, games in ranges:
        for game_index in range(first, first + games):
            focal = game_index % 4
            for seat in range(4):
                if seat == focal:
                    continue
                if cycle == 1 or (game_index * 7 + seat * 3) % 10 < 8:
                    v1 += 1
                else:
                    prior += 1
    return v1, prior


def expected_policy_seats(cycle: int) -> tuple[int, int]:
    first = 2_000_000_000 + cycle * 10_000
    return expected_policy_seats_for_ranges(cycle, [(first, 10_000)])


def build_jobs(
    *,
    plan: dict[str, Any],
    store: Any,
    campaign_state: Path,
    v1_weights: Path,
    newest_model: Path,
    prior_models: list[Path],
    prior_bundles_per_shard: int | None = 1,
) -> tuple[list[ContainerInput], str]:
    cycle = int(str(plan["phase"]).split("-")[1])
    newest_identity = model_identity(newest_model)
    unique_prior: list[tuple[Path, str]] = []
    seen = {newest_identity}
    for directory in prior_models:
        identity = model_identity(directory)
        if identity not in seen:
            seen.add(identity)
            unique_prior.append((directory, identity))
    if (cycle == 1 and unique_prior) or (cycle > 1 and not unique_prior):
        raise PipelineError(
            "cycle 1 must have no prior V3 opponent; later cycles require at least one "
            "checkpoint distinct from the newest model"
        )
    newest_stage = stage_model(store, newest_model, "newest")
    common_references = [
        store.stage_file(campaign_state, target="/inputs/control"),
        store.stage_file(v1_weights, target="/inputs/v1"),
        *newest_stage.references,
    ]
    prior_stages = []
    for index, (directory, identity) in enumerate(unique_prior, start=1):
        stage = stage_model(store, directory, f"prior-{index:02d}")
        prior_stages.append((stage, identity))
    jobs = []
    for shard_index, item in enumerate(plan["items"]):
        metadata = dict(item["application_metadata"])
        selected_prior = []
        if prior_stages:
            # A 100-game shard needs the newest focal policy and only one
            # frozen prior opponent. Rotate the prior deterministically across
            # shards so the complete collection covers the full pool without
            # loading every 105-MiB model into every one-GiB worker.
            if prior_bundles_per_shard is None:
                selected_prior = prior_stages
            elif prior_bundles_per_shard == 1:
                selected_prior = [prior_stages[shard_index % len(prior_stages)]]
            else:
                raise PipelineError("prior bundles per shard must be one or all")
        stages = [newest_stage, *(stage for stage, _ in selected_prior)]
        prior_identities = [identity for _, identity in selected_prior]
        references = [
            *common_references,
            *(reference for stage, _ in selected_prior for reference in stage.references),
        ]
        metadata.update(
            {
                "newest_model_id": newest_identity,
                "prior_model_ids": json.dumps(prior_identities, separators=(",", ":")),
            }
        )
        source_args = list(item["args"])
        arguments = []
        cursor = 0
        while cursor < len(source_args):
            if source_args[cursor] == "--v3-model-dir":
                cursor += 2
                continue
            arguments.append(source_args[cursor])
            cursor += 1
        arguments.extend(("--v3-model-dir", newest_stage.materialized_directory))
        for stage, _ in selected_prior:
            arguments.extend(("--v3-model-dir", stage.materialized_directory))
        environment = dict(item["environment"])
        environment["CASCADIA_MODEL_BUNDLES_JSON"] = bundle_environment(stages)
        jobs.append(
            ContainerInput(
                key=item["key"],
                args=tuple(arguments),
                environment=environment,
                inputs=tuple(references),
                application_metadata=metadata,
            )
        )
    return jobs, newest_identity


def _validate_item(item_directory: Path, job: ContainerInput) -> dict[str, int]:
    shards = sorted(item_directory.glob("*.v3g"))
    receipts = sorted(item_directory.glob("*.receipt.json"))
    if len(shards) != 1 or len(receipts) != 1:
        raise PipelineError(f"expert collection artifact set is incomplete for {job.key}")
    shard = shards[0]
    value = json.loads(receipts[0].read_text())
    expected_games = int(job.application_metadata["games"])
    newest = job.application_metadata["newest_model_id"]
    declared_prior = json.loads(job.application_metadata["prior_model_ids"])
    if not isinstance(declared_prior, list) or any(
        not isinstance(identity, str) for identity in declared_prior
    ):
        raise PipelineError(f"expert collection prior-model domain is invalid for {job.key}")
    policy = value.get("policy_seat_games", {})
    allowed_policies = {newest, V1_POLICY_ID, *declared_prior}
    if (
        value.get("schema_id") != "cascadia-v3-collection-shard-receipt-v1"
        or value.get("scientific_eligible") is not True
        or value.get("component") != "expert-iteration"
        or value.get("cycle") != int(job.application_metadata["cycle"])
        or value.get("games") != expected_games
        or value.get("records") != expected_games
        or value.get("newest_model_seats_per_expert_game") != 1
        or value.get("bytes") != shard.stat().st_size
        or value.get("blake3") != _digest(shard)
        or value.get("approved_readiness_sha256") is None
        or not isinstance(policy, dict)
        or any(not isinstance(identity, str) for identity in policy)
        or not set(policy).issubset(allowed_policies)
        or policy.get(newest) != expected_games
        or sum(int(count) for count in policy.values()) != expected_games * 4
    ):
        raise PipelineError(f"expert collection receipt is invalid for {job.key}")
    v1 = int(policy.get(V1_POLICY_ID, 0))
    return {
        "games": expected_games,
        "bytes": shard.stat().st_size,
        "seat_games": expected_games * 4,
        "newest_seat_games": int(policy[newest]),
        "v1_seat_games": v1,
        "prior_v3_seat_games": expected_games * 3 - v1,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_image(args.image)
    state = v3_phase2_jobs._authorized(args.campaign_state)
    phase = str(state.get("phase"))
    if not phase.startswith("cycle-") or not phase.endswith("-collecting"):
        raise PipelineError("expert collection requires a cycle collecting phase")
    cycle = int(phase.split("-")[1])
    client = _client(args.state_directory, args.artifact_directory)
    _validate_fabric(client.api.nodes())
    store = client.object_store
    assert store is not None
    plan = v3_phase2_jobs.build_plan(state, args.image, args.games_per_item)
    jobs, newest = build_jobs(
        plan=plan,
        store=store,
        campaign_state=args.campaign_state,
        v1_weights=args.v1_weights,
        newest_model=args.newest_model,
        prior_models=args.prior_model,
        prior_bundles_per_shard=(None if args.prior_bundles_per_shard == 0 else 1),
    )
    requested_items = set(args.item_key)
    if requested_items:
        known_items = {job.key for job in jobs}
        unknown = sorted(requested_items - known_items)
        if unknown:
            raise PipelineError(f"repair item keys are unknown: {unknown}")
        jobs = [job for job in jobs if job.key in requested_items]
        if len(jobs) != len(requested_items):
            raise PipelineError("repair item selection is not unique")
    completion = _monitor(
        client=client,
        image=args.image,
        jobs=jobs,
        resources=Resources(cpu=1, memory_gib=args.memory_gib, disk_gib=1),
        request_id=args.request_id,
        experiment_id=f"cascadia-v3-expert-cycle-{cycle:02d}-collection",
        artifact_directory=args.artifact_directory,
        progress=args.progress,
        timeout_seconds=12 * 60 * 60,
        validate=_validate_item,
    )
    expected_ranges = [
        (
            int(job.application_metadata["first_game_index"]),
            int(job.application_metadata["games"]),
        )
        for job in jobs
    ]
    expected_games = sum(games for _, games in expected_ranges)
    expected_v1, expected_prior = expected_policy_seats_for_ranges(cycle, expected_ranges)
    totals = completion["totals"]
    if (
        totals.get("games") != expected_games
        or totals.get("seat_games") != expected_games * 4
        or totals.get("newest_seat_games") != expected_games
        or totals.get("v1_seat_games") != expected_v1
        or totals.get("prior_v3_seat_games") != expected_prior
    ):
        raise PipelineError(f"cycle {cycle} policy-seat accounting differs: {totals}")
    completion.update(
        {
            "cycle": cycle,
            "newest_model_id": newest,
            "opponent_mix": {
                "v1_seat_games": expected_v1,
                "prior_v3_seat_games": expected_prior,
                "v1_fraction": expected_v1 / 30_000,
            },
            "manual_host_sharding": False,
            "scheduler_owns_placement": True,
            "repair_mode": bool(requested_items),
            "repair_item_keys": sorted(requested_items),
            "requested_memory_gib": args.memory_gib,
            "protected_seed_values_opened": False,
        }
    )
    _write_atomic(args.completion, completion)
    return completion


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--campaign-state", type=Path, required=True)
    parser.add_argument("--v1-weights", type=Path, required=True)
    parser.add_argument("--newest-model", type=Path, required=True)
    parser.add_argument("--prior-model", type=Path, action="append", default=[])
    parser.add_argument("--games-per-item", type=int, default=100)
    parser.add_argument("--item-key", action="append", default=[])
    parser.add_argument("--memory-gib", type=float, default=1.0)
    parser.add_argument(
        "--prior-bundles-per-shard",
        type=int,
        choices=(0, 1),
        default=1,
        help="mount one rotated prior per shard; zero preserves a legacy all-prior request",
    )
    parser.add_argument("--state-directory", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--request-id", required=True)
    parser.add_argument("--progress", type=Path, required=True)
    parser.add_argument("--completion", type=Path, required=True)
    args = parser.parse_args()
    if args.games_per_item <= 0 or 10_000 % args.games_per_item:
        raise SystemExit("games-per-item must divide 10,000")
    if args.memory_gib <= 0:
        raise SystemExit("memory-gib must be positive")
    try:
        result = run(args)
    except (PipelineError, OSError, ValueError, json.JSONDecodeError) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
