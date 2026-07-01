#!/usr/bin/env python3
"""Run the sealed 250-pair and 1K/4K all-V3 final Cascadia evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import secrets
import subprocess
import urllib.request
import uuid
from pathlib import Path
from typing import Any

import v3_final_report
from cascadia_cluster import ContainerInput, Resources
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
FINAL = ROOT / "phase2/final"
CLUSTER_API = "http://100.110.109.6:5187/api/v1/cluster"
CLUSTER_HISTORY_API = "http://100.110.109.6:5187/api/v1/cluster/history?range=7d"
SWAP_USED = re.compile(r"used = ([0-9.]+)([KMGT])")


class FinalPipelineError(ValueError):
    """The protected seed, pair, game, or final-report domain is invalid."""


def _read(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _write_text_atomic(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_text(value)
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _http_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=15) as response:
        value = json.loads(response.read())
    if not isinstance(value, dict):
        raise FinalPipelineError(f"telemetry endpoint did not return an object: {url}")
    return value


def _parse_swap_used(value: str) -> int:
    match = SWAP_USED.search(value)
    if match is None:
        raise FinalPipelineError("swap usage output is not parseable")
    scale = {"K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}[match.group(2)]
    return round(float(match.group(1)) * scale)


def _host_swap(host: str) -> dict[str, Any]:
    command = ["/usr/sbin/sysctl", "-n", "vm.swapusage"]
    if host != "john1":
        command = [
            "/usr/bin/ssh",
            "-o",
            "BatchMode=yes",
            "-o",
            "ConnectTimeout=10",
            host,
            "/usr/sbin/sysctl -n vm.swapusage",
        ]
    try:
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=20,
        )
        return {"used_bytes": _parse_swap_used(completed.stdout), "error": None}
    except (OSError, ValueError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
        return {"used_bytes": None, "error": str(error)}


def _resource_snapshot(label: str) -> dict[str, Any]:
    cluster = _http_json(CLUSTER_API)
    nodes = cluster.get("nodes")
    if not isinstance(nodes, list):
        raise FinalPipelineError("cluster telemetry omits nodes")
    selected = {}
    for node in nodes:
        if not isinstance(node, dict) or node.get("id") not in {
            "john1",
            "john2",
            "john3",
            "john4",
        }:
            continue
        selected[str(node["id"])] = {
            field: node.get(field)
            for field in (
                "reachable",
                "health",
                "cores",
                "cpu_percent",
                "memory_used_bytes",
                "memory_total_bytes",
                "memory_used_percent",
                "disk_used_bytes",
                "disk_total_bytes",
                "disk_available_bytes",
                "disk_used_percent",
            )
        }
    if set(selected) != {"john1", "john2", "john3", "john4"}:
        raise FinalPipelineError("cluster telemetry does not cover John1 through John4")
    swaps = {host: _host_swap(host) for host in ("john1", "john2", "john3")}
    return {
        "schema_id": "cascadia-v3-final-resource-snapshot-v1",
        "passed": True,
        "label": label,
        "collected_at_unix_ms": cluster.get("collected_at_unix_ms"),
        "nodes": selected,
        "swap": swaps,
    }


def _snapshot_once(path: Path, label: str) -> dict[str, Any]:
    if path.is_file():
        value = _read(path)
        if value.get("passed") is not True or value.get("label") != label:
            raise FinalPipelineError(f"existing resource snapshot is invalid: {path}")
        return value
    value = _resource_snapshot(label)
    _write_atomic(path, value)
    return value


def _compact_candidate(candidate: dict[str, Any]) -> dict[str, Any]:
    training = candidate.get("training_report", {})
    return {
        "origin": candidate.get("label"),
        "eligible": candidate.get("eligible"),
        "quantized_validation_loss": candidate.get("quantized_validation_loss"),
        "open_game_mean": candidate.get("open_game_mean"),
        "open_nonregression": candidate.get("open_nonregression"),
        "examples_seen": training.get("examples_seen") if isinstance(training, dict) else None,
        "elapsed_seconds": (
            training.get("elapsed_seconds") if isinstance(training, dict) else None
        ),
        "latest_loss": training.get("latest_loss") if isinstance(training, dict) else None,
    }


def _campaign_history() -> list[dict[str, Any]]:
    history = []
    for cycle in range(1, 11):
        directory = ROOT / f"phase2/cycles/cycle-{cycle:02d}"
        paths = {
            "corpus": directory / "corpus.json",
            "teacher": directory / "teacher-label-corpus.json",
            "selection": directory / "training/selection.json",
            "candidate": directory / "training/candidate.json",
            "promotion": directory / "promotion/report.json",
            "champion": directory / "promotion/champion.json",
        }
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FinalPipelineError(
                f"cycle {cycle} is missing final-history artifacts: {missing}"
            )
        values = {name: _read(path) for name, path in paths.items()}
        if any(value.get("passed") is not True for value in values.values()):
            raise FinalPipelineError(f"cycle {cycle} contains non-passing history evidence")
        if (
            values["corpus"].get("cycle") != cycle
            or values["corpus"].get("games") != 10_000
            or values["corpus"].get("training_entries") != 200_000
            or values["teacher"].get("cycle") != cycle
            or values["teacher"].get("roots") != 2_500
            or values["teacher"].get("rollouts") != 1_500_000
            or values["promotion"].get("cycle") != cycle
            or values["champion"].get("cycle") != cycle
        ):
            raise FinalPipelineError(f"cycle {cycle} history counts or identities drifted")
        curves = []
        for origin in (1, 2):
            loss_path = directory / f"training/origin-{origin}/loss.json"
            if not loss_path.is_file():
                raise FinalPipelineError(f"cycle {cycle} origin {origin} loss curve is missing")
            loss = _read(loss_path)
            samples = loss.get("samples")
            if not isinstance(samples, list) or not samples:
                raise FinalPipelineError(f"cycle {cycle} origin {origin} loss curve is empty")
            curves.append({"origin": origin, "samples": samples})
        selection = values["selection"]
        candidates = selection.get("candidates")
        if (
            selection.get("cycle") != cycle
            or not isinstance(candidates, list)
            or len(candidates) != 2
            or selection.get("selected") not in {
                candidate.get("label") for candidate in candidates if isinstance(candidate, dict)
            }
        ):
            raise FinalPipelineError(f"cycle {cycle} training selection is invalid")
        promotion = values["promotion"]
        pairs_per_tier = promotion.get("pairs_per_tier")
        if (
            not isinstance(pairs_per_tier, int)
            or not 100 <= pairs_per_tier <= 500
            or set(promotion.get("tiers", {}))
            != {"direct", "k32-r64", "k32-r600", "equal-wall-time"}
        ):
            raise FinalPipelineError(f"cycle {cycle} promotion history is incomplete")
        history.append(
            {
                "cycle": cycle,
                "collection": {
                    "games": values["corpus"].get("games"),
                    "training_entries": values["corpus"].get("training_entries"),
                    "canonical_sha256": values["corpus"].get("canonical_sha256"),
                },
                "teacher": {
                    "roots": values["teacher"].get("roots"),
                    "candidate_estimates": values["teacher"].get("candidate_estimates"),
                    "rollouts": values["teacher"].get("rollouts"),
                    "canonical_sha256": values["teacher"].get("canonical_sha256"),
                },
                "origins": [
                    _compact_candidate(item) for item in candidates
                ],
                "loss_curves": curves,
                "selected_origin": selection.get("selected"),
                "candidate": {
                    key: values["candidate"].get(key)
                    for key in (
                        "selected_origin",
                        "model_manifest_sha256",
                        "weights_sha256",
                        "parity_report_sha256",
                    )
                },
                "promotion": values["promotion"],
                "champion": values["champion"],
            }
        )
    return history


def _seed_manifest() -> dict[str, Any]:
    path = ROOT / "control/protected-seed-domains.json"
    if path.is_file():
        return _read(path)
    state = _read(STATE)
    if (
        state.get("phase") != "final_protected_comparison"
        or state.get("protected_seed_values_opened") is not True
    ):
        raise FinalPipelineError("protected seeds cannot open before the frozen final champion")
    value = {
        "schema_id": "cascadia-v3-protected-seed-domains-v1",
        "opened_after_cycle_10": True,
        "campaign_state_sha256": state["state_sha256"],
        "paired_key": secrets.token_hex(32),
        "all_v3_key": secrets.token_hex(32),
    }
    _write_atomic(path, value)
    return value


def _validate_protected(directory: Path, job: ContainerInput) -> dict[str, int]:
    value = _read(directory / "protected.json")
    records = value.get("records", [])
    if (
        value.get("schema_id") != "cascadia-v3-final-protected-pair-shard-v1"
        or value.get("passed") is not True
        or value.get("first_pair_index") != int(job.application_metadata["first_pair_index"])
        or value.get("pairs") != int(job.application_metadata["pairs"])
        or len(records) != int(job.application_metadata["pairs"])
        or any(record.get("integrity_passed") is not True for record in records)
    ):
        raise PipelineError(f"protected pair artifact is invalid for {job.key}")
    return {"pairs": len(records), "physical_games": len(records) * 2}


def _protected(image: str, champion: Path, seeds: dict[str, Any]) -> Path:
    output = FINAL / "protected.json"
    if output.is_file() and len(_read(output).get("pairs", [])) == 250:
        return output
    client = _client(ROOT / "phase2/control/cluster-client", FINAL / "protected-accepted")
    _validate_fabric(client.api.nodes())
    store = client.object_store
    assert store is not None
    state = _read(STATE)
    champion_stage = stage_model(store, champion, "champion")
    references = [
        store.stage_file(STATE, target="/inputs/control"),
        store.stage_file(V1, target="/inputs/v1"),
        *champion_stage.references,
    ]
    jobs = []
    for first in range(0, 250, 5):
        jobs.append(
            ContainerInput(
                key=f"protected-{first:03d}",
                args=(
                    "v3-campaign-worker",
                    "final-protected-pairs",
                    "--output",
                    "/outputs/protected.json",
                    "--treatment-model-dir",
                    champion_stage.materialized_directory,
                    "--v1-weights",
                    "/inputs/v1/qualified-v1.bin",
                    "--first-pair-index",
                    str(first),
                    "--pairs",
                    "5",
                    "--seed-domain-key",
                    seeds["paired_key"],
                    "--campaign-state",
                    "/inputs/control/campaign-state.json",
                    "--approved-readiness-sha256",
                    state["approved_readiness_sha256"],
                ),
                environment={
                    "RAYON_NUM_THREADS": "1",
                    "CASCADIA_MODEL_BUNDLES_JSON": bundle_environment(
                        [champion_stage]
                    ),
                },
                inputs=tuple(references),
                application_metadata={"first_pair_index": str(first), "pairs": "5"},
            )
        )
    completion_path = FINAL / "protected-completion.json"
    if not completion_path.is_file():
        completion = _monitor(
            client=client,
            image=image,
            jobs=jobs,
            resources=Resources(cpu=1, memory_gib=1.0, disk_gib=1),
            request_id="v3-final-protected-250",
            experiment_id="cascadia-v3-final-protected-250-pairs",
            artifact_directory=FINAL / "protected-accepted",
            progress=FINAL / "protected-progress.json",
            timeout_seconds=24 * 60 * 60,
            validate=_validate_protected,
        )
        _write_atomic(completion_path, completion)
    reclaim_completed_increment(
        completion_path,
        completion_path.with_name("protected-storage-reclaim.json"),
    )
    reclaim_remote_workers(
        completion_path,
        completion_path.with_name("protected-remote-worker-reclaim.json"),
    )
    records = []
    elapsed = 0.0
    for path in sorted(
        (FINAL / "protected-accepted/v3-final-protected-250").glob("*/protected.json")
    ):
        value = _read(path)
        records.extend(value["records"])
        elapsed += float(value["elapsed_seconds"])
    records.sort(key=lambda record: int(record["pair_index"]))
    if [int(record["pair_index"]) for record in records] != list(range(250)):
        raise FinalPipelineError("protected pair domain is not exactly 0..249")
    treatment_seconds = sum(float(row["treatment"]["focal_seconds"]) for row in records)
    control_seconds = sum(float(row["control"]["focal_seconds"]) for row in records)
    _write_atomic(
        output,
        {
            "schema_id": "cascadia-v3-final-protected-corpus-v1",
            "passed": True,
            "pairs": records,
            "resource_metrics": {
                "worker_elapsed_seconds": elapsed,
                "treatment_focal_seconds": treatment_seconds,
                "control_focal_seconds": control_seconds,
                "treatment_control_time_ratio": treatment_seconds / max(control_seconds, 1e-9),
            },
        },
    )
    return output


def _validate_all_v3(directory: Path, job: ContainerInput) -> dict[str, int]:
    value = _read(directory / "all-v3.json")
    records = value.get("records", [])
    if (
        value.get("schema_id") != "cascadia-v3-final-all-v3-shard-v1"
        or value.get("passed") is not True
        or value.get("first_game_index") != int(job.application_metadata["first_game_index"])
        or value.get("games") != int(job.application_metadata["games"])
        or len(records) != int(job.application_metadata["games"])
        or any(
            record.get("integrity_passed") is not True or len(record.get("seats", [])) != 4
            for record in records
        )
    ):
        raise PipelineError(f"all-V3 artifact is invalid for {job.key}")
    return {"games": len(records), "seat_games": len(records) * 4}


def _all_v3_increment(
    image: str,
    champion: Path,
    seeds: dict[str, Any],
    start: int,
    count: int,
) -> Path:
    request = f"v3-final-all-v3-{start:04d}-{start + count:04d}"
    completion_path = FINAL / f"{request}-completion.json"
    if not completion_path.is_file():
        client = _client(ROOT / "phase2/control/cluster-client", FINAL / "all-v3-accepted")
        _validate_fabric(client.api.nodes())
        store = client.object_store
        assert store is not None
        state = _read(STATE)
        champion_stage = stage_model(store, champion, "champion")
        references = [
            store.stage_file(STATE, target="/inputs/control"),
            *champion_stage.references,
        ]
        jobs = []
        for first in range(start, start + count, 2):
            jobs.append(
                ContainerInput(
                    key=f"all-v3-{first:04d}",
                    args=(
                        "v3-campaign-worker",
                        "final-all-v3",
                        "--output",
                        "/outputs/all-v3.json",
                        "--model-dir",
                        champion_stage.materialized_directory,
                        "--first-game-index",
                        str(first),
                        "--games",
                        "2",
                        "--seed-domain-key",
                        seeds["all_v3_key"],
                        "--campaign-state",
                        "/inputs/control/campaign-state.json",
                        "--approved-readiness-sha256",
                        state["approved_readiness_sha256"],
                    ),
                    environment={
                        "RAYON_NUM_THREADS": "1",
                        "CASCADIA_MODEL_BUNDLES_JSON": bundle_environment(
                            [champion_stage]
                        ),
                    },
                    inputs=tuple(references),
                    application_metadata={"first_game_index": str(first), "games": "2"},
                )
            )
        completion = _monitor(
            client=client,
            image=image,
            jobs=jobs,
            resources=Resources(cpu=1, memory_gib=1.0, disk_gib=1),
            request_id=request,
            experiment_id="cascadia-v3-final-all-v3-k32-r600",
            artifact_directory=FINAL / "all-v3-accepted",
            progress=FINAL / f"{request}-progress.json",
            timeout_seconds=48 * 60 * 60,
            validate=_validate_all_v3,
        )
        _write_atomic(completion_path, completion)
    reclaim_completed_increment(
        completion_path,
        completion_path.with_name(f"{request}-storage-reclaim.json"),
    )
    reclaim_remote_workers(
        completion_path,
        completion_path.with_name(f"{request}-remote-worker-reclaim.json"),
    )
    return completion_path


def _aggregate_all_v3(games: int) -> Path:
    records = []
    elapsed = 0.0
    for path in sorted((FINAL / "all-v3-accepted").rglob("all-v3.json")):
        value = _read(path)
        included = False
        for record in value["records"]:
            if int(record["game_index"]) < games:
                records.append(record)
                included = True
        if included:
            elapsed += float(value["elapsed_seconds"])
    records.sort(key=lambda record: int(record["game_index"]))
    if [int(record["game_index"]) for record in records] != list(range(games)):
        raise FinalPipelineError(f"all-V3 domain is not exactly 0..{games - 1}")
    decision_seconds = sum(
        float(seat["decision_seconds"]) for record in records for seat in record["seats"]
    )
    output = FINAL / f"all-v3-{games}.json"
    _write_atomic(
        output,
        {
            "schema_id": "cascadia-v3-final-all-v3-corpus-v1",
            "passed": True,
            "games": records,
            "resource_metrics": {
                "worker_elapsed_seconds": elapsed,
                "decision_seconds": decision_seconds,
            },
        },
    )
    return output


def _advance(destination: str, evidence: Path) -> None:
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
            hashlib.sha256(evidence.read_bytes()).hexdigest(),
        ],
        cwd=REPOSITORY,
        check=True,
    )


def run(image: str, champion: Path) -> dict[str, Any]:
    _validate_image(image)
    champion_id = model_identity(champion)
    before = _snapshot_once(FINAL / "resources-before.json", "before-final-evaluation")
    seeds = _seed_manifest()
    state = _read(STATE)
    protected = _protected(image, champion, seeds)
    if state.get("phase") == "final_protected_comparison":
        _advance("final_all_v3_evaluation", protected)
    for start in range(0, 1_000, 100):
        _all_v3_increment(image, champion, seeds, start, 100)
    all_v3 = _aggregate_all_v3(1_000)
    report = v3_final_report.build_report(_read(protected), _read(all_v3))
    if report["all_v3"]["requires_4000_game_extension"]:
        for start in range(1_000, 4_000, 100):
            _all_v3_increment(image, champion, seeds, start, 100)
        all_v3 = _aggregate_all_v3(4_000)
    after = _resource_snapshot("after-final-evaluation")
    _write_atomic(FINAL / "resources-after.json", after)
    resource_observations = {
        "before": before,
        "after": after,
        "history_7d": _http_json(CLUSTER_HISTORY_API),
        "swap_delta_bytes": {
            host: (
                int(after["swap"][host]["used_bytes"])
                - int(before["swap"][host]["used_bytes"])
                if after["swap"][host]["used_bytes"] is not None
                and before["swap"][host]["used_bytes"] is not None
                else None
            )
            for host in ("john1", "john2", "john3")
        },
    }
    final_state = _read(STATE)
    champion_manifest = _read(champion / "model.json")
    report = v3_final_report.build_report(
        _read(protected),
        _read(all_v3),
        campaign_history=_campaign_history(),
        resource_observations=resource_observations,
        champion={
            "model_id": champion_id,
            "directory": str(champion.resolve()),
            "model_manifest_sha256": _sha256(champion / "model.json"),
            "weights_sha256": _sha256(champion / "weights.v3q"),
            "manifest": champion_manifest,
        },
        provenance={
            "canonical_image": image,
            "approved_readiness_sha256": final_state["approved_readiness_sha256"],
            "campaign_state_sha256": final_state["state_sha256"],
            "protected_seed_manifest_sha256": _sha256(
                ROOT / "control/protected-seed-domains.json"
            ),
            "protected_seed_values_redacted": True,
            "research_spec": str(
                (REPOSITORY / "docs/v3/CASCADIA_V3_RESEARCH_SPEC.md").resolve()
            ),
        },
    )
    report_path = ROOT / "reports/final-v3-campaign.json"
    _write_atomic(report_path, report)
    _write_text_atomic(
        ROOT / "reports/final-v3-campaign.md",
        v3_final_report.render_markdown(report),
    )
    if _read(STATE).get("phase") == "final_all_v3_evaluation":
        _advance("complete", report_path)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--champion-model", type=Path, required=True)
    args = parser.parse_args()
    try:
        result = run(args.image, args.champion_model)
    except (
        FinalPipelineError,
        PipelineError,
        OSError,
        ValueError,
        json.JSONDecodeError,
        subprocess.CalledProcessError,
    ) as error:
        raise SystemExit(str(error)) from error
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
