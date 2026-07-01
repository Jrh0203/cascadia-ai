#!/usr/bin/env python3
"""Validate and analyze Full-Legal Public Oracle paired-game shards."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import statistics
from collections import defaultdict
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

EXPERIMENT_ID = "full-legal-public-oracle-v1"
RULES_PROTOCOL_ID = "cascadia-aaaaa-4p-base-v1"
BASELINE_ID = "exact-mlx-k32-r600-champion-v1"
TREATMENT_ID = "full-legal-public-oracle-v1"
EXPECTED_SOURCE_BLAKE3 = "3d8a378b8b3088141fbc30f3194a84681008c0c339263714b2e94d0ce4f3c40d"
EXPECTED_EXECUTABLE_SHA256 = (
    "b1dee74da6e2288c51358d1f146deb2c73b9d8d64ac88646413fdc4ec85bf7d3"
)
EXPECTED_EXECUTABLE_BLAKE3 = (
    "b666e499cc04d8d74236baedeb10761879d2818f59a1a83ea8d083056d05f0fd"
)
EXPECTED_MODEL_JSON_SHA256 = (
    "9dff120f4238497f0c9440f5d353ef00cf6cb8fdef513cc8c4fe5bbdfc490d3a"
)
EXPECTED_MODEL_JSON_BLAKE3 = (
    "dd3ea3bbbff0187107695132531a56c09a1da18b58fac4bacacf66960fd7ff0d"
)
EXPECTED_MODEL_SAFETENSORS_SHA256 = (
    "9fd11f704a5feb427aab324c19dc819213dda08f8a4b90331999df3726b11f89"
)
EXPECTED_MODEL_SAFETENSORS_BLAKE3 = (
    "3f8f2609b1440396720aa48adabf9561a4a172d006f77011a9516baa0b06ba65"
)


def frozen_config(screen_limit: int) -> dict[str, Any]:
    return {
        "protocol_id": "full-legal-decision-regret-audit-v1",
        "champion_rollouts": 600,
        "screen_limit": screen_limit,
        "sentinel_count": 16,
        "substantial_rollouts": 1_200,
        "high_confidence_limit": 8,
        "high_confidence_rollouts": 4_800,
        "audited_completed_turns": None,
        "realized_hidden_completed_turns": [],
        "paid_wipe_determinizations": 0,
        "paid_wipe_followup_determinizations": 2,
        "paid_wipe_followup_width": 3,
    }
TIME_REAL_RE = re.compile(
    r"^\s*(?:([0-9]+(?:\.[0-9]+)?) real|real ([0-9]+(?:\.[0-9]+)?))",
    re.MULTILINE,
)
TIME_RSS_RE = re.compile(r"^\s*([0-9]+)\s+maximum resident set size", re.MULTILINE)
TIME_SWAPS_RE = re.compile(r"^\s*([0-9]+)\s+swaps", re.MULTILINE)
SYSTEM_SWAP_USED_RE = re.compile(r"\bused = ([0-9]+(?:\.[0-9]+)?)([KMG])\b")


@dataclass(frozen=True)
class ExpectedShard:
    host: str
    first_seed: int
    games: int

    @property
    def seeds(self) -> range:
        return range(self.first_seed, self.first_seed + self.games)


@dataclass(frozen=True)
class OracleInput:
    path: Path
    host: str
    sha256: str
    report: dict[str, Any]


def timestamp() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json_atomic(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def parse_expected_shard(value: str) -> ExpectedShard:
    parts = value.split(":")
    if len(parts) != 3:
        raise argparse.ArgumentTypeError("expected HOST:FIRST_SEED:GAMES")
    host, first_seed, games = parts
    try:
        parsed = ExpectedShard(host=host, first_seed=int(first_seed), games=int(games))
    except ValueError as error:
        raise argparse.ArgumentTypeError("seed and games must be integers") from error
    if not parsed.host or parsed.games <= 0:
        raise argparse.ArgumentTypeError("host must be nonempty and games must be positive")
    return parsed


def parse_host_count(value: str) -> tuple[str, int]:
    host, separator, count = value.partition(":")
    if not separator or not host:
        raise argparse.ArgumentTypeError("expected HOST:COUNT")
    try:
        parsed = int(count)
    except ValueError as error:
        raise argparse.ArgumentTypeError("host count must be an integer") from error
    if parsed < 0:
        raise argparse.ArgumentTypeError("host count cannot be negative")
    return host, parsed


def positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("expected a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("expected a positive integer")
    return parsed


def parse_key_value_file(path: Path) -> dict[str, str]:
    if not path.is_file():
        raise ValueError(f"{path}: required metadata file is missing")
    values: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text().splitlines(), start=1):
        key, separator, value = line.partition("=")
        if not separator or not key or key in values:
            raise ValueError(f"{path}:{line_number}: malformed or duplicate metadata")
        values[key] = value
    return values


def verify_artifact_checksum_manifest(
    directory: Path,
    required_names: set[str],
) -> str:
    manifest = directory / "SHA256SUMS"
    if not manifest.is_file():
        raise ValueError(f"{manifest}: required artifact checksum manifest is missing")
    root = directory.resolve()
    observed: set[str] = set()
    for line_number, line in enumerate(manifest.read_text().splitlines(), start=1):
        digest, separator, raw_path = line.partition("  ")
        if (
            not separator
            or re.fullmatch(r"[0-9a-f]{64}", digest) is None
            or not raw_path
        ):
            raise ValueError(f"{manifest}:{line_number}: malformed checksum line")
        relative = Path(raw_path)
        if relative.is_absolute():
            raise ValueError(f"{manifest}:{line_number}: absolute artifact path")
        candidate = (directory / relative).resolve()
        try:
            normalized = candidate.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"{manifest}:{line_number}: artifact path escapes its shard"
            ) from error
        normalized_name = normalized.as_posix()
        if normalized_name in observed:
            raise ValueError(f"{manifest}:{line_number}: duplicate artifact path")
        if not candidate.is_file():
            raise ValueError(f"{manifest}:{line_number}: artifact is missing")
        actual = sha256_file(candidate)
        if actual != digest:
            raise ValueError(
                f"{manifest}:{line_number}: checksum mismatch for {normalized_name}"
            )
        observed.add(normalized_name)
    missing = required_names - observed
    if missing:
        raise ValueError(f"{manifest}: required artifacts are unchecksummed: {sorted(missing)}")
    return sha256_file(manifest)


def validate_system_metadata(
    directory: Path,
    expected: ExpectedShard,
    *,
    screen_limit: int,
) -> None:
    system = parse_key_value_file(directory / "system.txt")
    required = {
        "host_label": expected.host,
        "first_seed": str(expected.first_seed),
        "game_count": str(expected.games),
        "source_blake3": EXPECTED_SOURCE_BLAKE3,
        "binary_sha256": EXPECTED_EXECUTABLE_SHA256,
        "model_json_sha256": EXPECTED_MODEL_JSON_SHA256,
        "model_safetensors_sha256": EXPECTED_MODEL_SAFETENSORS_SHA256,
    }
    for key, value in required.items():
        if system.get(key) != value:
            raise ValueError(f"{directory / 'system.txt'}: {key} drifted")
    recorded_screen_limit = system.get("screen_limit")
    if recorded_screen_limit is None:
        if screen_limit != 64:
            raise ValueError(f"{directory / 'system.txt'}: screen_limit is missing")
    elif recorded_screen_limit != str(screen_limit):
        raise ValueError(f"{directory / 'system.txt'}: screen_limit drifted")


def validate_score(score: dict[str, Any], context: str) -> None:
    habitat = score.get("habitat")
    wildlife = score.get("wildlife")
    bonuses = score.get("habitat_bonus")
    if not isinstance(habitat, list) or len(habitat) != 5:
        raise ValueError(f"{context}: habitat score is not length five")
    if not isinstance(wildlife, list) or len(wildlife) != 5:
        raise ValueError(f"{context}: wildlife score is not length five")
    if not isinstance(bonuses, list) or len(bonuses) != 5 or any(bonuses):
        raise ValueError(f"{context}: habitat bonuses must be disabled")
    expected = sum(habitat) + sum(wildlife) + int(score["nature_tokens"])
    if int(score["base_total"]) != expected or int(score["total"]) != expected:
        raise ValueError(f"{context}: total does not match score decomposition")


def mean_game_score(match: dict[str, Any]) -> float:
    return statistics.fmean(float(score["base_total"]) for score in match["scores"])


def validate_report(
    item: OracleInput,
    expected: ExpectedShard,
    *,
    screen_limit: int = 64,
) -> None:
    report = item.report
    if report.get("schema_version") != 1:
        raise ValueError(f"{item.path}: unsupported schema")
    if report.get("experiment_id") != EXPERIMENT_ID or report.get("status") != "complete":
        raise ValueError(f"{item.path}: incomplete or wrong experiment")
    if report.get("worker") != expected.host or item.host != expected.host:
        raise ValueError(f"{item.path}: worker ownership mismatch")
    if int(report["first_seed"]) != expected.first_seed or int(report["games"]) != expected.games:
        raise ValueError(f"{item.path}: shard seed domain mismatch")
    if report.get("config") != frozen_config(screen_limit):
        raise ValueError(f"{item.path}: oracle configuration drifted")
    frozen_identities = {
        "source.v2_source_blake3": (
            report.get("source", {}).get("v2_source_blake3"),
            EXPECTED_SOURCE_BLAKE3,
        ),
        "executable_blake3": (
            report.get("executable_blake3"),
            EXPECTED_EXECUTABLE_BLAKE3,
        ),
        "model_json_blake3": (
            report.get("model_json_blake3"),
            EXPECTED_MODEL_JSON_BLAKE3,
        ),
        "model_safetensors_blake3": (
            report.get("model_safetensors_blake3"),
            EXPECTED_MODEL_SAFETENSORS_BLAKE3,
        ),
    }
    for label, (actual, expected_identity) in frozen_identities.items():
        if actual != expected_identity:
            raise ValueError(f"{item.path}: frozen {label} identity drifted")
    if report["comparison"]["protocol_id"] != RULES_PROTOCOL_ID:
        raise ValueError(f"{item.path}: rules protocol drifted")
    if report["comparison"]["baseline_id"] != BASELINE_ID:
        raise ValueError(f"{item.path}: baseline strategy drifted")
    if report["comparison"]["treatment_id"] != TREATMENT_ID:
        raise ValueError(f"{item.path}: treatment strategy drifted")
    if len(report.get("game_records", [])) != expected.games:
        raise ValueError(f"{item.path}: game-record count mismatch")
    if int(report["decision_summary"]["decisions"]) != expected.games * 80:
        raise ValueError(f"{item.path}: treatment decision count mismatch")
    if not report["baseline_clean_shutdown"] or not report["treatment_clean_shutdown"]:
        raise ValueError(f"{item.path}: MLX service did not shut down cleanly")
    for side in ("baseline", "treatment"):
        if int(report[f"{side}_diagnostics"]["fallbacks"]) != 0:
            raise ValueError(f"{item.path}: {side} bridge fallback occurred")
        if int(report[f"{side}_batch_diagnostics"]["policy_fallbacks"]) != 0:
            raise ValueError(f"{item.path}: {side} policy fallback occurred")
        if int(report[f"{side}_batch_diagnostics"]["bootstrapped_samples"]) != 0:
            raise ValueError(f"{item.path}: {side} bootstrapped a terminal sample")

    for offset, record in enumerate(report["game_records"]):
        raw_seed = expected.first_seed + offset
        if int(record["raw_seed"]) != raw_seed:
            raise ValueError(f"{item.path}: noncontiguous seed at game {offset}")
        baseline = record["baseline"]
        treatment = record["treatment"]
        if baseline["seed"] != treatment["seed"]:
            raise ValueError(f"{item.path}: paired game seeds differ at {raw_seed}")
        if int(baseline["turns"]) != 80 or int(treatment["turns"]) != 80:
            raise ValueError(f"{item.path}: incomplete game at {raw_seed}")
        if baseline["strategies"] != [BASELINE_ID] * 4:
            raise ValueError(f"{item.path}: baseline seat strategy drift at {raw_seed}")
        if treatment["strategies"] != [TREATMENT_ID] * 4:
            raise ValueError(f"{item.path}: treatment seat strategy drift at {raw_seed}")
        if len(baseline["scores"]) != 4 or len(treatment["scores"]) != 4:
            raise ValueError(f"{item.path}: paired game does not contain four seats")
        if baseline["replay"]["final_state_hash"] is None:
            raise ValueError(f"{item.path}: baseline final hash missing at {raw_seed}")
        if treatment["replay"]["final_state_hash"] is None:
            raise ValueError(f"{item.path}: treatment final hash missing at {raw_seed}")
        for seat, score in enumerate(baseline["scores"]):
            validate_score(score, f"{item.path}: baseline seed {raw_seed} seat {seat}")
        for seat, score in enumerate(treatment["scores"]):
            validate_score(score, f"{item.path}: treatment seed {raw_seed} seat {seat}")

    recomputed_baseline = statistics.fmean(
        mean_game_score(record["baseline"]) for record in report["game_records"]
    )
    recomputed_treatment = statistics.fmean(
        mean_game_score(record["treatment"]) for record in report["game_records"]
    )
    if not math.isclose(
        recomputed_baseline,
        float(report["comparison"]["baseline_mean"]),
        abs_tol=1e-12,
    ):
        raise ValueError(f"{item.path}: baseline summary disagrees with raw scores")
    if not math.isclose(
        recomputed_treatment,
        float(report["comparison"]["treatment_mean"]),
        abs_tol=1e-12,
    ):
        raise ValueError(f"{item.path}: treatment summary disagrees with raw scores")


def load_inputs(
    input_dirs: list[Path],
    expected: list[ExpectedShard],
    *,
    screen_limit: int = 64,
) -> list[OracleInput]:
    expected_by_host = {shard.host: shard for shard in expected}
    if len(expected_by_host) != len(expected):
        raise ValueError("duplicate expected host")
    inputs: list[OracleInput] = []
    for directory in input_dirs:
        host = directory.name
        if host not in expected_by_host:
            raise ValueError(f"unexpected oracle host directory {host}")
        paths = sorted(directory.glob("shard-*.json"))
        if len(paths) != 1:
            raise ValueError(f"{directory}: expected exactly one oracle shard, found {len(paths)}")
        path = paths[0]
        required_names = {
            "memory-pressure-after.txt",
            "memory-pressure-before.txt",
            path.name,
            path.with_suffix(".stdout").name,
            path.with_suffix(".time").name,
            path.with_suffix(".validate").name,
            "swap-after.txt",
            "swap-before.txt",
            "system.txt",
        }
        verify_artifact_checksum_manifest(directory, required_names)
        if not (directory / "SHA256SUMS.inputs").is_file():
            raise ValueError(f"{directory / 'SHA256SUMS.inputs'}: input manifest is missing")
        validate_system_metadata(
            directory,
            expected_by_host[host],
            screen_limit=screen_limit,
        )
        item = OracleInput(
            path=path,
            host=host,
            sha256=sha256_file(path),
            report=json.loads(path.read_text()),
        )
        validate_report(item, expected_by_host[host], screen_limit=screen_limit)
        inputs.append(item)
    if {item.host for item in inputs} != set(expected_by_host):
        raise ValueError("oracle input directories do not cover every expected host")
    return sorted(inputs, key=lambda item: expected_by_host[item.host].first_seed)


def bootstrap_metric(
    values: list[float],
    *,
    bootstrap_samples: int,
    seed: int,
) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    mean = float(np.mean(array))
    standard_error = (
        float(np.std(array, ddof=1) / math.sqrt(len(array))) if len(array) > 1 else 0.0
    )
    rng = np.random.default_rng(seed)
    samples: list[np.ndarray] = []
    remaining = bootstrap_samples
    while remaining:
        batch = min(remaining, 4_096)
        indices = rng.integers(0, len(array), size=(batch, len(array)))
        samples.append(np.mean(array[indices], axis=1))
        remaining -= batch
    bootstrap = np.concatenate(samples)
    low, high = np.quantile(bootstrap, [0.025, 0.975])
    return {
        "count": len(values),
        "mean": mean,
        "game_block_standard_error": standard_error,
        "normal_confidence_95": [
            mean - 1.96 * standard_error,
            mean + 1.96 * standard_error,
        ],
        "bootstrap_confidence_95": [float(low), float(high)],
    }


def score_breakdown(records: list[dict[str, Any]], side: str) -> dict[str, Any]:
    scores = [score for record in records for score in record[side]["scores"]]
    return {
        "habitat": [
            statistics.fmean(float(score["habitat"][index]) for score in scores)
            for index in range(5)
        ],
        "wildlife": [
            statistics.fmean(float(score["wildlife"][index]) for score in scores)
            for index in range(5)
        ],
        "nature_tokens": statistics.fmean(float(score["nature_tokens"]) for score in scores),
        "base_total": statistics.fmean(float(score["base_total"]) for score in scores),
    }


def subtract_breakdown(treatment: dict[str, Any], baseline: dict[str, Any]) -> dict[str, Any]:
    return {
        "habitat": [
            treatment["habitat"][index] - baseline["habitat"][index] for index in range(5)
        ],
        "wildlife": [
            treatment["wildlife"][index] - baseline["wildlife"][index]
            for index in range(5)
        ],
        "nature_tokens": treatment["nature_tokens"] - baseline["nature_tokens"],
        "base_total": treatment["base_total"] - baseline["base_total"],
    }


def parse_time_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace")
    real = TIME_REAL_RE.search(text)
    rss = TIME_RSS_RE.search(text)
    swaps = TIME_SWAPS_RE.search(text)
    return {
        "real_seconds": float(real.group(1) or real.group(2)) if real else None,
        "maximum_resident_bytes": int(rss.group(1)) if rss else None,
        "swaps": int(swaps.group(1)) if swaps else None,
        "sha256": sha256_file(path),
    }


def parse_swap_file(path: Path) -> dict[str, Any] | None:
    if not path.is_file():
        return None
    text = path.read_text(errors="replace").strip()
    match = SYSTEM_SWAP_USED_RE.search(text)
    if match is None:
        used_bytes = None
    else:
        scale = {"K": 1024, "M": 1024**2, "G": 1024**3}[match.group(2)]
        used_bytes = int(float(match.group(1)) * scale)
    return {"used_bytes": used_bytes, "raw": text, "sha256": sha256_file(path)}


def utilization(item: OracleInput, retries: int = 0) -> dict[str, Any]:
    first_seed = int(item.report["first_seed"])
    last_seed = first_seed + int(item.report["games"]) - 1
    timing = parse_time_file(item.path.with_name(f"shard-{first_seed}-{last_seed}.time"))
    before = parse_swap_file(item.path.parent / "swap-before.txt")
    after = parse_swap_file(item.path.parent / "swap-after.txt")
    swap_delta = None
    if (
        before is not None
        and before["used_bytes"] is not None
        and after is not None
        and after["used_bytes"] is not None
    ):
        swap_delta = after["used_bytes"] - before["used_bytes"]
    process_wall_seconds = timing["real_seconds"] if timing is not None else None
    productive_wall_seconds = float(item.report["elapsed_seconds"])
    runner_overhead_seconds = (
        max(process_wall_seconds - productive_wall_seconds, 0.0)
        if process_wall_seconds is not None
        else None
    )
    normalized_games_per_hour = (
        int(item.report["games"]) * 3_600.0 / process_wall_seconds
        if process_wall_seconds is not None and process_wall_seconds > 0.0
        else None
    )
    return {
        "assigned_games": int(item.report["games"]),
        "completed_games": len(item.report["game_records"]),
        "failures_or_retries_observed": retries,
        "idle_with_work_queued_seconds": 0.0,
        "productive_wall_seconds": productive_wall_seconds,
        "runner_overhead_seconds": runner_overhead_seconds,
        "normalized_games_per_hour": normalized_games_per_hour,
        "timing": timing,
        "system_swap_before": before,
        "system_swap_after": after,
        "system_swap_delta_bytes": swap_delta,
    }


def aggregate_decision_latency(
    inputs: list[OracleInput],
    side: str,
) -> dict[str, Any]:
    latencies = [
        item.report["comparison"][f"{side}_statistics"]["decision_latency"]
        for item in inputs
    ]
    decisions = sum(int(latency["decisions"]) for latency in latencies)
    if decisions == 0:
        raise ValueError(f"{side} decision latency contains no decisions")
    return {
        "decisions": decisions,
        "mean_milliseconds": (
            sum(
                int(latency["decisions"]) * float(latency["mean_milliseconds"])
                for latency in latencies
            )
            / decisions
        ),
        "maximum_milliseconds": max(float(latency["max_milliseconds"]) for latency in latencies),
        "per_host": {
            item.host: item.report["comparison"][f"{side}_statistics"][
                "decision_latency"
            ]
            for item in inputs
        },
    }


def analyze(
    inputs: list[OracleInput],
    expected: list[ExpectedShard],
    *,
    treatment_mean_minimum: float,
    paired_delta_minimum: float,
    bootstrap_samples: int,
    stage: str = "pilot",
    experiment_id: str = EXPERIMENT_ID,
    require_positive_delta_confidence: bool = False,
    require_positive_host_deltas: bool = False,
    host_retries: dict[str, int] | None = None,
) -> dict[str, Any]:
    if stage not in {"pilot", "confirmation"}:
        raise ValueError(f"unsupported oracle report stage {stage}")
    host_retries = host_retries or {}
    expected = sorted(expected, key=lambda shard: shard.first_seed)
    unknown_retry_hosts = set(host_retries) - {shard.host for shard in expected}
    if unknown_retry_hosts:
        raise ValueError(f"retry counts contain unexpected hosts: {sorted(unknown_retry_hosts)}")
    records = [
        {**record, "host": item.host}
        for item in inputs
        for record in item.report["game_records"]
    ]
    records.sort(key=lambda record: int(record["raw_seed"]))
    expected_seeds = [seed for shard in expected for seed in shard.seeds]
    actual_seeds = [int(record["raw_seed"]) for record in records]
    if actual_seeds != expected_seeds:
        raise ValueError(f"oracle seed coverage mismatch: {actual_seeds}")

    identities: dict[str, set[str]] = defaultdict(set)
    for item in inputs:
        report = item.report
        identities["model_json_blake3"].add(report["model_json_blake3"])
        identities["model_safetensors_blake3"].add(report["model_safetensors_blake3"])
        identities["executable_blake3"].add(report["executable_blake3"])
        identities["v2_source_blake3"].add(report["source"]["v2_source_blake3"])
    if any(len(values) != 1 for values in identities.values()):
        raise ValueError("oracle shards do not share one source, binary, and model identity")

    baseline_game_means = [mean_game_score(record["baseline"]) for record in records]
    treatment_game_means = [mean_game_score(record["treatment"]) for record in records]
    deltas = [
        treatment - baseline
        for baseline, treatment in zip(
            baseline_game_means,
            treatment_game_means,
            strict=True,
        )
    ]
    baseline_metric = bootstrap_metric(
        baseline_game_means,
        bootstrap_samples=bootstrap_samples,
        seed=20_260_615,
    )
    treatment_metric = bootstrap_metric(
        treatment_game_means,
        bootstrap_samples=bootstrap_samples,
        seed=20_260_616,
    )
    delta_metric = bootstrap_metric(
        deltas,
        bootstrap_samples=bootstrap_samples,
        seed=20_260_617,
    )

    host_reports: dict[str, Any] = {}
    for index, shard in enumerate(expected):
        host_records = [record for record in records if record["host"] == shard.host]
        baseline = [mean_game_score(record["baseline"]) for record in host_records]
        treatment = [mean_game_score(record["treatment"]) for record in host_records]
        host_deltas = [
            right - left for left, right in zip(baseline, treatment, strict=True)
        ]
        host_reports[shard.host] = {
            "first_seed": shard.first_seed,
            "games": shard.games,
            "baseline": bootstrap_metric(
                baseline,
                bootstrap_samples=bootstrap_samples,
                seed=20_261_000 + index * 3,
            ),
            "treatment": bootstrap_metric(
                treatment,
                bootstrap_samples=bootstrap_samples,
                seed=20_261_001 + index * 3,
            ),
            "paired_delta": bootstrap_metric(
                host_deltas,
                bootstrap_samples=bootstrap_samples,
                seed=20_261_002 + index * 3,
            ),
        }

    decision_summary = {
        key: sum(int(item.report["decision_summary"][key]) for item in inputs)
        for key in (
            "decisions",
            "changed_actions",
            "top_screen_recalled_winners",
            "actions_screened",
            "champion_frontier_actions",
            "substantial_actions",
            "high_confidence_actions",
        )
    }
    decision_summary["champion_regret_sum"] = sum(
        float(item.report["decision_summary"]["champion_regret_sum"]) for item in inputs
    )
    phase_decisions = [
        sum(int(item.report["decision_summary"]["phase_decisions"][phase]) for item in inputs)
        for phase in range(3)
    ]
    phase_changes = [
        sum(
            int(item.report["decision_summary"]["phase_changed_actions"][phase])
            for item in inputs
        )
        for phase in range(3)
    ]
    phase_regret = [
        sum(
            float(item.report["decision_summary"]["phase_champion_regret_sum"][phase])
            for item in inputs
        )
        for phase in range(3)
    ]
    phase = {
        label: {
            "decisions": phase_decisions[index],
            "mean_champion_regret": phase_regret[index] / phase_decisions[index],
            "action_change_rate": phase_changes[index] / phase_decisions[index],
        }
        for index, label in enumerate(("early", "middle", "late"))
    }

    baseline_breakdown = score_breakdown(records, "baseline")
    treatment_breakdown = score_breakdown(records, "treatment")
    every_host_delta_passed = all(
        float(host["paired_delta"]["mean"]) > 0.0
        if require_positive_host_deltas
        else float(host["paired_delta"]["mean"]) >= 0.0
        for host in host_reports.values()
    )
    expected_phase_counts = [len(records) * 28, len(records) * 28, len(records) * 24]
    complete_phase_coverage = phase_decisions == expected_phase_counts
    host_utilization = {
        item.host: utilization(item, host_retries.get(item.host, 0)) for item in inputs
    }
    host_telemetry_complete = all(
        metrics["timing"] is not None
        and metrics["timing"]["real_seconds"] is not None
        and metrics["timing"]["maximum_resident_bytes"] is not None
        and metrics["timing"]["swaps"] is not None
        and metrics["system_swap_before"] is not None
        and metrics["system_swap_after"] is not None
        for metrics in host_utilization.values()
    )
    process_swaps_zero = all(
        metrics["timing"] is not None and metrics["timing"]["swaps"] == 0
        for metrics in host_utilization.values()
    )
    process_walls = [
        float(metrics["timing"]["real_seconds"])
        for metrics in host_utilization.values()
        if metrics["timing"] is not None
        and metrics["timing"]["real_seconds"] is not None
    ]
    cluster_wall_seconds = max(process_walls) if len(process_walls) == len(inputs) else None
    cluster_utilization = {
        "cluster_wall_seconds": cluster_wall_seconds,
        "aggregate_games_per_hour": (
            len(records) * 3_600.0 / cluster_wall_seconds
            if cluster_wall_seconds is not None and cluster_wall_seconds > 0.0
            else None
        ),
        "normalized_per_node_games_per_hour": {
            host: metrics["normalized_games_per_hour"]
            for host, metrics in host_utilization.items()
        },
        "total_failures_or_retries_observed": sum(host_retries.values()),
    }
    gates = {
        "all_games_and_decisions_complete": decision_summary["decisions"] == len(records) * 80,
        "single_source_binary_model_identity": all(
            len(values) == 1 for values in identities.values()
        ),
        "all_integrity_checks_passed": True,
        "host_memory_and_swap_telemetry_complete": host_telemetry_complete,
        "process_swaps_zero": process_swaps_zero,
        "treatment_mean_at_least_threshold": (
            float(treatment_metric["mean"]) >= treatment_mean_minimum
        ),
        "paired_delta_at_least_threshold": (
            float(delta_metric["mean"]) >= paired_delta_minimum
        ),
        "paired_delta_bootstrap_lower_bound_positive": (
            float(delta_metric["bootstrap_confidence_95"][0]) > 0.0
            if require_positive_delta_confidence
            else True
        ),
        (
            "every_host_paired_delta_positive"
            if require_positive_host_deltas
            else "every_host_paired_delta_nonnegative"
        ): every_host_delta_passed,
        "complete_phase_coverage": complete_phase_coverage,
    }
    passed = all(gates.values())
    return {
        "schema_version": 1,
        "experiment_id": experiment_id,
        "stage": stage,
        "status": f"{stage}_passed" if passed else f"{stage}_failed",
        "generated_at": timestamp(),
        "rules_protocol_id": RULES_PROTOCOL_ID,
        "configuration": inputs[0].report["config"],
        "coverage": {
            "games": len(records),
            "seat_games_per_strategy": len(records) * 4,
            "first_seed": expected_seeds[0],
            "last_seed": expected_seeds[-1],
            "hosts": {shard.host: shard.games for shard in expected},
        },
        "identity": {
            key: next(iter(values)) for key, values in sorted(identities.items())
        },
        "baseline": baseline_metric,
        "treatment": treatment_metric,
        "paired_delta": delta_metric,
        "game_wins": sum(delta > 0.0 for delta in deltas),
        "game_ties": sum(delta == 0.0 for delta in deltas),
        "game_losses": sum(delta < 0.0 for delta in deltas),
        "baseline_breakdown": baseline_breakdown,
        "treatment_breakdown": treatment_breakdown,
        "breakdown_delta": subtract_breakdown(treatment_breakdown, baseline_breakdown),
        "decision_summary": {
            **decision_summary,
            "mean_champion_regret": (
                decision_summary["champion_regret_sum"] / decision_summary["decisions"]
            ),
            "action_change_rate": (
                decision_summary["changed_actions"] / decision_summary["decisions"]
            ),
            "top_screen_recall": (
                decision_summary["top_screen_recalled_winners"]
                / decision_summary["decisions"]
            ),
        },
        "phase": phase,
        "hosts": host_reports,
        "decision_latency": {
            "baseline": aggregate_decision_latency(inputs, "baseline"),
            "treatment": aggregate_decision_latency(inputs, "treatment"),
        },
        "host_utilization": host_utilization,
        "cluster_utilization": cluster_utilization,
        "pairs": [
            {
                "seed": int(record["raw_seed"]),
                "host": record["host"],
                "baseline_mean": baseline,
                "treatment_mean": treatment,
                "delta": delta,
            }
            for record, baseline, treatment, delta in zip(
                records,
                baseline_game_means,
                treatment_game_means,
                deltas,
                strict=True,
            )
        ],
        "thresholds": {
            "treatment_mean_minimum": treatment_mean_minimum,
            "paired_delta_minimum": paired_delta_minimum,
            "positive_delta_confidence_required": require_positive_delta_confidence,
            "positive_host_deltas_required": require_positive_host_deltas,
        },
        "gates": gates,
        "passed": passed,
    }


def format_metric(metric: dict[str, Any]) -> str:
    low, high = metric["bootstrap_confidence_95"]
    return f"{metric['mean']:.3f} [{low:.3f}, {high:.3f}]"


def count_label(count: int, singular: str, plural: str | None = None) -> str:
    label = singular if count == 1 else plural or f"{singular}s"
    return f"{count} {label}"


def render_markdown(report: dict[str, Any]) -> str:
    decision = report["decision_summary"]
    screen_limit = int(report["configuration"]["screen_limit"])
    display_name = (
        "Full-Legal Public Oracle K1024 V1"
        if report["experiment_id"] == "full-legal-public-oracle-k1024-v1"
        else "Full-Legal Public Oracle V1"
    )
    lines = [
        f"# {display_name} {report['stage'].title()}",
        "",
        f"- Status: **{report['status']}**",
        f"- Games: {report['coverage']['games']}",
        f"- Baseline mean: {format_metric(report['baseline'])}",
        f"- Treatment mean: {format_metric(report['treatment'])}",
        f"- Paired delta: **{format_metric(report['paired_delta'])}**",
        (
            f"- Record: {count_label(report['game_wins'], 'win')}, "
            f"{count_label(report['game_ties'], 'tie')}, "
            f"{count_label(report['game_losses'], 'loss', 'losses')}"
        ),
        f"- Mean local champion regret: {decision['mean_champion_regret']:.3f}",
        f"- Action change rate: {decision['action_change_rate'] * 100.0:.3f}%",
        (
            f"- Top-{screen_limit} winner rate: "
            f"{decision['top_screen_recall'] * 100.0:.3f}%"
        ),
        "",
        "## Gates",
        "",
        "| Gate | Passed |",
        "|---|---:|",
    ]
    lines.extend(f"| `{name}` | `{passed}` |" for name, passed in report["gates"].items())
    lines.extend(
        [
            "",
            "## Hosts",
            "",
            "| Host | Games | Baseline | Treatment | Delta |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for host, metrics in report["hosts"].items():
        lines.append(
            f"| {host} | {metrics['games']} | {metrics['baseline']['mean']:.3f} | "
            f"{metrics['treatment']['mean']:.3f} | "
            f"{metrics['paired_delta']['mean']:+.3f} |"
        )
    baseline_latency = report["decision_latency"]["baseline"]
    treatment_latency = report["decision_latency"]["treatment"]
    cluster = report["cluster_utilization"]
    lines.extend(
        [
            "",
            "## Runtime And Utilization",
            "",
            (
                f"- Baseline mean decision latency: "
                f"{baseline_latency['mean_milliseconds']:.3f} ms"
            ),
            (
                f"- Treatment mean decision latency: "
                f"{treatment_latency['mean_milliseconds']:.3f} ms"
            ),
            (
                f"- Aggregate cluster throughput: "
                f"{cluster['aggregate_games_per_hour']:.3f} games/hour"
                if cluster["aggregate_games_per_hour"] is not None
                else "- Aggregate cluster throughput: n/a"
            ),
            "",
            (
                "| Host | Wall s | Productive s | Games/hour | Max RSS MiB | "
                "Process swaps | System swap delta MiB | Retries |"
            ),
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for host, metrics in report["host_utilization"].items():
        timing = metrics["timing"]
        wall = timing["real_seconds"] if timing is not None else None
        rss = timing["maximum_resident_bytes"] if timing is not None else None
        swaps = timing["swaps"] if timing is not None else None
        swap_delta = metrics["system_swap_delta_bytes"]
        lines.append(
            f"| {host} | "
            f"{wall:.3f} | "
            f"{metrics['productive_wall_seconds']:.3f} | "
            f"{metrics['normalized_games_per_hour']:.3f} | "
            f"{rss / 1024**2:.1f} | "
            f"{swaps} | "
            f"{swap_delta / 1024**2:.1f} | "
            f"{metrics['failures_or_retries_observed']} |"
            if wall is not None
            and metrics["normalized_games_per_hour"] is not None
            and rss is not None
            and swaps is not None
            and swap_delta is not None
            else f"| {host} | n/a | n/a | n/a | n/a | n/a | n/a | "
            f"{metrics['failures_or_retries_observed']} |"
        )
    lines.extend(
        [
            "",
            "## Phase Diagnostics",
            "",
            "| Phase | Decisions | Mean regret | Action change |",
            "|---|---:|---:|---:|",
        ]
    )
    for phase, metrics in report["phase"].items():
        lines.append(
            f"| {phase.title()} | {metrics['decisions']} | "
            f"{metrics['mean_champion_regret']:.3f} | "
            f"{metrics['action_change_rate'] * 100.0:.3f}% |"
        )
    lines.extend(
        [
            "",
            "## Seed Pairs",
            "",
            "| Seed | Host | Baseline | Treatment | Delta |",
            "|---:|---|---:|---:|---:|",
        ]
    )
    for pair in report["pairs"]:
        lines.append(
            f"| {pair['seed']} | {pair['host']} | {pair['baseline_mean']:.3f} | "
            f"{pair['treatment_mean']:.3f} | {pair['delta']:+.3f} |"
        )
    return "\n".join(lines) + "\n"


def write_index(path: Path, inputs: list[OracleInput], report: dict[str, Any]) -> None:
    files = []
    for item in inputs:
        related = {}
        for suffix in ("stdout", "time", "validate"):
            candidate = item.path.with_suffix(f".{suffix}")
            related[f"{suffix}_sha256"] = (
                sha256_file(candidate) if candidate.is_file() else None
            )
        files.append(
            {
                "host": item.host,
                "path": str(item.path),
                "bytes": item.path.stat().st_size,
                "sha256": item.sha256,
                "artifact_manifest_sha256": sha256_file(
                    item.path.parent / "SHA256SUMS"
                ),
                "input_manifest_sha256": sha256_file(
                    item.path.parent / "SHA256SUMS.inputs"
                ),
                "system_sha256": sha256_file(item.path.parent / "system.txt"),
                **related,
            }
        )
    value = {
        "schema_version": 1,
        "experiment_id": report["experiment_id"],
        "status": report["status"],
        "generated_at": report["generated_at"],
        "coverage": report["coverage"],
        "identity": report["identity"],
        "files": files,
    }
    receipt_manifest = path.parent / "collection-receipts/SHA256SUMS"
    if receipt_manifest.is_file():
        value["collection_receipts"] = {
            "path": str(receipt_manifest),
            "sha256": sha256_file(receipt_manifest),
        }
        receipt_validation = receipt_manifest.with_name("SHA256SUMS.validate")
        if receipt_validation.is_file():
            value["collection_receipts"]["validation_sha256"] = sha256_file(
                receipt_validation
            )
    write_json_atomic(path, value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, action="append", required=True)
    parser.add_argument(
        "--expected-shard",
        type=parse_expected_shard,
        action="append",
        required=True,
    )
    parser.add_argument("--treatment-mean-minimum", type=float, default=100.0)
    parser.add_argument("--paired-delta-minimum", type=float, default=3.0)
    parser.add_argument("--screen-limit", type=positive_integer, default=64)
    parser.add_argument("--report-experiment-id", default=EXPERIMENT_ID)
    parser.add_argument("--stage", choices=("pilot", "confirmation"), default="pilot")
    parser.add_argument("--require-positive-delta-confidence", action="store_true")
    parser.add_argument("--require-positive-host-deltas", action="store_true")
    parser.add_argument("--host-retry", type=parse_host_count, action="append", default=[])
    parser.add_argument("--bootstrap-samples", type=int, default=50_000)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown-output", type=Path, required=True)
    parser.add_argument("--index-output", type=Path, required=True)
    args = parser.parse_args()

    inputs = load_inputs(
        [path.resolve() for path in args.input_dir],
        args.expected_shard,
        screen_limit=args.screen_limit,
    )
    host_retries = dict(args.host_retry)
    if len(host_retries) != len(args.host_retry):
        parser.error("--host-retry contains duplicate hosts")
    report = analyze(
        inputs,
        args.expected_shard,
        treatment_mean_minimum=args.treatment_mean_minimum,
        paired_delta_minimum=args.paired_delta_minimum,
        bootstrap_samples=args.bootstrap_samples,
        stage=args.stage,
        experiment_id=args.report_experiment_id,
        require_positive_delta_confidence=args.require_positive_delta_confidence,
        require_positive_host_deltas=args.require_positive_host_deltas,
        host_retries=host_retries,
    )
    write_json_atomic(args.output.resolve(), report)
    markdown = args.markdown_output.resolve()
    markdown.parent.mkdir(parents=True, exist_ok=True)
    markdown.write_text(render_markdown(report))
    write_index(args.index_output.resolve(), inputs, report)
    print(
        json.dumps(
            {
                "status": report["status"],
                "games": report["coverage"]["games"],
                "baseline_mean": report["baseline"]["mean"],
                "treatment_mean": report["treatment"]["mean"],
                "paired_delta": report["paired_delta"]["mean"],
                "passed": report["passed"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
