#!/usr/bin/env python3
"""Run the always-valid four-tier promotion test for one V3 expert cycle."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Any

import v3_promotion
import v3_promotion_v1
from cascadia_cluster import ContainerInput, Resources
from v3_checkpoint_lifecycle import compact_completed_run, retire_completed_run
from v3_colima_reclaim import reclaim_completed_increment, reclaim_remote_workers
from v3_model_stage import bundle_environment, model_identity, stage_model
from v3_phase2_pipeline import (
    PipelineError,
    _client,
    _monitor,
    _validate_fabric,
    _validate_image,
    _write_atomic,
)

ROOT = Path("/Users/johnherrick/cascadia-bench/v3-nnue")
REPOSITORY = Path("/Users/johnherrick/cascadia")
PYTHON = REPOSITORY / ".venv/bin/python"
STATE = ROOT / "control/campaign-state.json"
V1 = ROOT / "phase2/inputs/v1/qualified-v1.bin"
TIERS = ("direct", "k32-r64", "k32-r600", "equal-wall-time")
WORKER_TIERS = {
    "direct": "direct",
    "k32-r64": "k32r64",
    "k32-r600": "k32r600",
    "equal-wall-time": "equal-wall-time",
}
PROMOTION_CONTRACT = "v3-conditioned-rollout"
PROMOTION_SEARCH_ENV = {
    "MCE_LMR": "1",
    "MCE_DIVERSE_PREFILTER": "1",
}
PAIR_INCREMENT = 100
# The first four cycles used five-pair shards. Cycle 5 then profiled one-pair,
# pair-major admission: its first look performed 54,339 focal CPU-seconds but
# took 3,233 wall seconds because 400 jobs each staged the same model bundles.
# Four pairs per immutable item cuts bundle staging and scheduler overhead 4x,
# while the 25 items in each long tier still form two balanced waves across the
# 29-CPU fleet. Pair indices, RNG domains, tier totals, and the registered
# sequential test are unchanged. Opened requests always recover their frozen
# shard size and item order from durable state.
PAIRS_PER_ITEM = 4


class CyclePromotionError(ValueError):
    """The promotion pair domain, model identity, or sequential gate failed."""


def promotion_rule(cycle: int) -> Any:
    """Keep Cycle 1 on its opened v1 rule; use the pre-registered v2 thereafter."""
    return v3_promotion_v1 if cycle == 1 else v3_promotion


def worker_tier(tier: str) -> str:
    """Translate the report tier into clap's stable ValueEnum spelling."""
    try:
        return WORKER_TIERS[tier]
    except KeyError as error:
        raise CyclePromotionError(f"unknown promotion tier: {tier}") from error


def request_id(cycle: int, start: int, count: int) -> str:
    return (
        f"v3-cycle-{cycle:02d}-promotion-{start:03d}-{start + count:03d}-"
        f"{PROMOTION_CONTRACT}"
    )


def request_plan(
    cycle: int, start: int, count: int
) -> tuple[int, tuple[tuple[str, int], ...]]:
    """Return the frozen shard size/order for an opened request or a new plan."""
    request = request_id(cycle, start, count)
    path = ROOT / "phase2/control/cluster-client/requests" / f"{request}.json"
    if not path.is_file():
        return PAIRS_PER_ITEM, tuple(
            (tier, first)
            for first in range(start, start + count, PAIRS_PER_ITEM)
            for tier in TIERS
        )
    value = _read(path)
    items = value.get("items")
    if (
        value.get("schema_id") != "cascadia.cluster.managed-request-state.v2"
        or value.get("request_id") != request
        or not isinstance(items, list)
        or not items
    ):
        raise CyclePromotionError("opened promotion request state is invalid")
    metadata = [item.get("job_payload", {}).get("Meta", {}) for item in items]
    recorded = {item.get("cascadia.app.pairs") for item in metadata}
    try:
        pairs_per_item = int(next(iter(recorded)))
    except (StopIteration, TypeError, ValueError) as error:
        raise CyclePromotionError("opened promotion request has no shard size") from error
    try:
        item_order = tuple(
            (
                str(item["cascadia.app.tier"]),
                int(item["cascadia.app.first_pair_index"]),
            )
            for item in metadata
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CyclePromotionError("opened promotion request has invalid item order") from error
    expected = {
        (tier, first)
        for tier in TIERS
        for first in range(start, start + count, pairs_per_item)
    }
    if (
        len(recorded) != 1
        or pairs_per_item <= 0
        or count % pairs_per_item
        or len(items) != len(TIERS) * count // pairs_per_item
        or len(set(item_order)) != len(item_order)
        or set(item_order) != expected
    ):
        raise CyclePromotionError("opened promotion request shard domain is inconsistent")
    return pairs_per_item, item_order


def pairs_per_item_for_request(cycle: int, start: int, count: int) -> int:
    """Compatibility accessor for tests and diagnostics."""
    return request_plan(cycle, start, count)[0]


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _sha256(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def build_jobs(
    *,
    cycle: int,
    start: int,
    count: int,
    state: dict[str, Any],
    store: Any,
    treatment: Path,
    control: Path,
    pairs_per_item: int = PAIRS_PER_ITEM,
    item_order: tuple[tuple[str, int], ...] | None = None,
) -> list[ContainerInput]:
    if start < 0 or count <= 0 or pairs_per_item <= 0 or count % pairs_per_item:
        raise CyclePromotionError("promotion pair range does not divide into worker items")
    treatment_stage = stage_model(store, treatment, "treatment")
    control_stage = stage_model(store, control, "control")
    stages = [treatment_stage, control_stage]
    references = [
        store.stage_file(STATE, target="/inputs/control"),
        store.stage_file(V1, target="/inputs/v1"),
        *treatment_stage.references,
        *control_stage.references,
    ]
    expected_order = {
        (tier, first)
        for tier in TIERS
        for first in range(start, start + count, pairs_per_item)
    }
    if item_order is None:
        item_order = tuple(
            (tier, first)
            for first in range(start, start + count, pairs_per_item)
            for tier in TIERS
        )
    if len(item_order) != len(expected_order) or set(item_order) != expected_order:
        raise CyclePromotionError("promotion item order does not cover the exact tier domain")
    jobs = []
    for tier, first in item_order:
        key = f"{tier}-{first:04d}"
        jobs.append(
            ContainerInput(
                key=key,
                args=(
                    "v3-campaign-worker",
                    "promotion-pairs",
                    "--output",
                    "/outputs/promotion.json",
                    "--treatment-model-dir",
                    treatment_stage.materialized_directory,
                    "--control-model-dir",
                    control_stage.materialized_directory,
                    "--v1-weights",
                    "/inputs/v1/qualified-v1.bin",
                    "--tier",
                    worker_tier(tier),
                    "--first-pair-index",
                    str(first),
                    "--pairs",
                    str(pairs_per_item),
                    "--cycle",
                    str(cycle),
                    "--campaign-state",
                    "/inputs/control/campaign-state.json",
                    "--approved-readiness-sha256",
                    state["approved_readiness_sha256"],
                ),
                environment={
                    **PROMOTION_SEARCH_ENV,
                    "RAYON_NUM_THREADS": "1",
                    "CASCADIA_MODEL_BUNDLES_JSON": bundle_environment(stages),
                },
                inputs=tuple(references),
                application_metadata={
                    "campaign": "cascadia-v3",
                    "stage": "cycle-promotion",
                    "cycle": str(cycle),
                    "tier": tier,
                    "first_pair_index": str(first),
                    "pairs": str(pairs_per_item),
                    "promotion_contract": PROMOTION_CONTRACT,
                },
            )
        )
    return jobs


def _validate_item(directory: Path, job: ContainerInput) -> dict[str, int]:
    path = directory / "promotion.json"
    value = _read(path)
    records = value.get("records", [])
    if (
        value.get("schema_id") != "cascadia-v3-promotion-pair-shard-v1"
        or value.get("passed") is not True
        or value.get("scientific_eligible") is not True
        or value.get("cycle") != int(job.application_metadata["cycle"])
        or value.get("tier") != job.application_metadata["tier"]
        or value.get("first_pair_index")
        != int(job.application_metadata["first_pair_index"])
        or value.get("pairs") != int(job.application_metadata["pairs"])
        or len(records) != int(job.application_metadata["pairs"])
        or any(record.get("integrity_passed") is not True for record in records)
    ):
        raise PipelineError(f"promotion artifact is invalid for {job.key}")
    return {"pairs": len(records), "physical_games": len(records) * 2}


def _run_increment(
    *,
    cycle: int,
    image: str,
    treatment: Path,
    control: Path,
    start: int,
    count: int,
) -> Path:
    directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}/promotion"
    request = request_id(cycle, start, count)
    completion = directory / f"completion-{start:03d}-{start + count:03d}.json"
    if completion.is_file() and _read(completion).get("passed") is True:
        return completion
    state = _read(STATE)
    client = _client(
        ROOT / "phase2/control/cluster-client",
        directory / "accepted",
    )
    _validate_fabric(client.api.nodes())
    store = client.object_store
    assert store is not None
    pairs_per_item, item_order = request_plan(cycle, start, count)
    jobs = build_jobs(
        cycle=cycle,
        start=start,
        count=count,
        state=state,
        store=store,
        treatment=treatment,
        control=control,
        pairs_per_item=pairs_per_item,
        item_order=item_order,
    )
    value = _monitor(
        client=client,
        image=image,
        jobs=jobs,
        resources=Resources(cpu=1, memory_gib=1.0, disk_gib=1),
        request_id=request,
        experiment_id=f"cascadia-v3-cycle-{cycle:02d}-promotion",
        artifact_directory=directory / "accepted",
        progress=directory / f"progress-{start:03d}-{start + count:03d}.json",
        timeout_seconds=24 * 60 * 60,
        validate=_validate_item,
    )
    _write_atomic(completion, value)
    return completion


def _records(cycle: int, pairs: int) -> list[dict[str, Any]]:
    directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}/promotion/accepted"
    by_tier: dict[str, list[dict[str, Any]]] = {tier: [] for tier in TIERS}
    for start in range(0, pairs, PAIR_INCREMENT):
        count = min(PAIR_INCREMENT, pairs - start)
        request_directory = directory / request_id(cycle, start, count)
        for path in sorted(request_directory.rglob("promotion.json")):
            value = _read(path)
            if value.get("cycle") != cycle:
                continue
            tier = value["tier"]
            for record in value["records"]:
                if int(record["pair_index"]) < pairs:
                    by_tier[tier].append(record)
    result = []
    for tier, values in by_tier.items():
        values.sort(key=lambda record: int(record["pair_index"]))
        if [int(record["pair_index"]) for record in values] != list(range(pairs)):
            raise CyclePromotionError(f"{tier} does not contain the exact 0..{pairs - 1} domain")
        treatment_seconds = sum(float(record["treatment_focal_seconds"]) for record in values)
        control_seconds = sum(float(record["control_focal_seconds"]) for record in values)
        resource_regression = treatment_seconds > control_seconds * 1.20
        for record in values:
            result.append(
                {
                    "tier": tier,
                    "pair_index": record["pair_index"],
                    "paired_delta": record["paired_delta"],
                    "integrity_passed": record["integrity_passed"],
                    "resource_regression": resource_regression,
                }
            )
    return result


def _freeze_champion(
    *,
    cycle: int,
    verdict: dict[str, Any],
    treatment: Path,
    treatment_run: Path,
    control: Path,
    control_run: Path,
) -> Path:
    promoted = verdict["verdict"] == "promote"
    source = treatment if promoted else control
    run_dir = treatment_run if promoted else control_run
    destination = ROOT / f"models/cycle-{cycle:02d}-champion"
    if not destination.is_dir():
        temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
        shutil.copytree(source, temporary)
        os.replace(temporary, destination)
    evidence = ROOT / f"phase2/cycles/cycle-{cycle:02d}/promotion/champion.json"
    value = {
        "schema_id": "cascadia-v3-expert-cycle-champion-v1",
        "passed": True,
        "cycle": cycle,
        "promotion_verdict": verdict["verdict"],
        "promoted": promoted,
        "champion_model_dir": str(destination.resolve()),
        "champion_run_dir": str(run_dir.resolve()),
        "model_manifest_sha256": _sha256(destination / "model.json"),
        "weights_sha256": _sha256(destination / "weights.v3q"),
        "promotion_report": str(
            (ROOT / f"phase2/cycles/cycle-{cycle:02d}/promotion/report.json").resolve()
        ),
        "protected_seed_values_opened": False,
    }
    _write_atomic(evidence, value)
    return evidence


def _advance(cycle: int, evidence: Path) -> None:
    destination = (
        f"cycle-{cycle + 1:02d}-collecting"
        if cycle < 10
        else "final_protected_comparison"
    )
    subprocess.run(
        [
            str(PYTHON),
            "tools/v3_campaign.py",
            "advance-phase2",
            "--to",
            destination,
            "--evidence",
            str(evidence),
            "--evidence-sha256",
            _sha256(evidence),
        ],
        cwd=REPOSITORY,
        check=True,
    )


def run(args: argparse.Namespace) -> dict[str, Any]:
    _validate_image(args.image)
    state = _read(STATE)
    if state.get("phase") != f"cycle-{args.cycle:02d}-promotion":
        raise CyclePromotionError("campaign is not at this cycle's promotion gate")
    if model_identity(args.treatment_model) == model_identity(args.control_model):
        raise CyclePromotionError("promotion treatment and control are identical")
    rule = promotion_rule(args.cycle)
    pairs = 0
    verdict: dict[str, Any] = {"verdict": "continue"}
    while verdict["verdict"] == "continue" and pairs < rule.MAX_PAIRS:
        completion = _run_increment(
            cycle=args.cycle,
            image=args.image,
            treatment=args.treatment_model,
            control=args.control_model,
            start=pairs,
            count=PAIR_INCREMENT,
        )
        reclaim_completed_increment(
            completion,
            completion.with_name(
                f"storage-reclaim-{pairs:03d}-{pairs + PAIR_INCREMENT:03d}.json"
            ),
        )
        reclaim_remote_workers(
            completion,
            completion.with_name(
                f"remote-worker-reclaim-{pairs:03d}-{pairs + PAIR_INCREMENT:03d}.json"
            ),
        )
        pairs += PAIR_INCREMENT
        verdict = rule.evaluate(_records(args.cycle, pairs))
    if verdict["verdict"] == "continue":
        raise CyclePromotionError("promotion remained open after the registered maximum")
    report = ROOT / f"phase2/cycles/cycle-{args.cycle:02d}/promotion/report.json"
    result = {
        **verdict,
        "passed": True,
        "cycle": args.cycle,
        "pairs_per_tier": pairs,
        "physical_games": pairs * len(TIERS) * 2,
        "treatment_model_id": model_identity(args.treatment_model),
        "control_model_id": model_identity(args.control_model),
        "protected_seed_values_opened": False,
    }
    _write_atomic(report, result)
    evidence = _freeze_champion(
        cycle=args.cycle,
        verdict=result,
        treatment=args.treatment_model,
        treatment_run=args.treatment_run_dir,
        control=args.control_model,
        control_run=args.control_run_dir,
    )
    if result["verdict"] == "promote":
        compact_completed_run(args.treatment_run_dir)
        retire_completed_run(args.control_run_dir, reason="superseded-by-promoted-cycle")
    else:
        compact_completed_run(args.control_run_dir)
        retire_completed_run(args.treatment_run_dir, reason="cycle-candidate-not-promoted")
    _advance(args.cycle, evidence)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle", type=int, required=True)
    parser.add_argument("--image", required=True)
    parser.add_argument("--treatment-model", type=Path, required=True)
    parser.add_argument("--treatment-run-dir", type=Path, required=True)
    parser.add_argument("--control-model", type=Path, required=True)
    parser.add_argument("--control-run-dir", type=Path, required=True)
    args = parser.parse_args()
    if not 1 <= args.cycle <= 10:
        raise SystemExit("cycle is outside 1..=10")
    try:
        value = run(args)
    except (
        CyclePromotionError,
        PipelineError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(value, sort_keys=True))


if __name__ == "__main__":
    main()
