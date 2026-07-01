"""Integrated complete-action evaluation for the ADR 0175 pointer pilot."""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import resource
import socket
import time
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import numpy as np

from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    STAGE_BATCH_SIZES,
    STAGE_WIDTHS,
    STAGES,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    configure_mlx_memory,
    mlx_memory_snapshot,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_expected_rank import (
    NORMAL_95,
    build_expected_rank_target_mask,
)
from cascadia_mlx.p1_relational_pointer_data import (
    DEFAULT_FACTOR_CACHE,
    DEFAULT_R3_CACHE,
    RelationalPointerCorpus,
    validate_pointer_batch,
)
from cascadia_mlx.p1_relational_pointer_model import (
    RelationalPointerModelConfig,
    RelationalPointerRanker,
    parameter_tensor_blake3,
)
from cascadia_mlx.p1_relational_pointer_train import (
    ADR_ID,
    EXPECTED_PARENT_PARAMETER_BLAKE3,
    EXPERIMENT_ID,
    PROTOCOL_ID,
    PointerParentEncodingMemo,
    evaluate_pointer_stage,
)
from cascadia_mlx.run_manifest import source_provenance

SCHEMA_VERSION = 1
EVALUATION_PROTOCOL_ID = "integrated-p1-pointer-open-evaluation-v1"
CLASSIFICATION_PIPELINE_INVALID = "p1_pointer_pipeline_invalid"
CLASSIFICATION_TILE_INSUFFICIENT = "p1_pointer_tile_stage_insufficient"
CLASSIFICATION_PROPOSAL_INSUFFICIENT = "p1_pointer_proposal_insufficient"
CLASSIFICATION_SELECTOR_INSUFFICIENT = "p1_pointer_selector_insufficient"
CLASSIFICATION_SUFFICIENT = "p1_pointer_offline_sufficient"


@dataclass(frozen=True)
class PointerIntegrationConfig:
    """Selected stage runs and immutable open caches."""

    draft_run_dir: Path
    tile_run_dir: Path
    wildlife_run_dir: Path
    output: Path
    draft_replay: Path | None = None
    tile_replay: Path | None = None
    wildlife_replay: Path | None = None
    factor_cache: Path = DEFAULT_FACTOR_CACHE
    r3_cache: Path = DEFAULT_R3_CACHE
    max_shards: int | None = None

    @property
    def production(self) -> bool:
        return self.max_shards is None

    def run_dirs(self) -> dict[str, Path]:
        return {
            "draft": self.draft_run_dir,
            "tile": self.tile_run_dir,
            "wildlife": self.wildlife_run_dir,
        }

    def replay_paths(self) -> dict[str, Path | None]:
        return {
            "draft": self.draft_replay,
            "tile": self.tile_replay,
            "wildlife": self.wildlife_replay,
        }

    def validate(self) -> None:
        if len({path.resolve() for path in self.run_dirs().values()}) != 3:
            raise ValueError("pointer integration requires three distinct stage runs")
        if self.max_shards is not None and not 1 <= self.max_shards <= 2:
            raise ValueError("bounded integration may inspect only one or two shards")
        if self.production and any(
            path is None for path in self.replay_paths().values()
        ):
            raise ValueError(
                "production pointer integration requires all three replays"
            )


@dataclass
class _SelectionAccumulator:
    groups: int = 0
    target_slots: int = 0
    target_hits: int = 0
    exact_sets: int = 0
    winner_hits: int = 0
    confidence_hits: int = 0
    distinguishable_groups: int = 0
    distinguishable_winner_hits: int = 0
    regret: float = 0.0

    def add(
        self,
        *,
        retained: np.ndarray,
        target: np.ndarray,
        source_flags: np.ndarray,
        winner: int,
        r4800_mean: np.ndarray,
        r4800_stddev: np.ndarray,
        r4800_samples: np.ndarray,
        r4800_mask: np.ndarray,
        action_hashes: np.ndarray,
    ) -> None:
        retained = np.asarray(retained, dtype=np.int32)
        nonfrontier = retained[
            (
                source_flags[retained]
                & GRADED_SOURCE_CHAMPION_FRONTIER
            )
            == 0
        ]
        quota = int(np.sum(target))
        hits = int(np.sum(target[nonfrontier]))
        observation = _r4800_observation(
            retained=retained,
            winner=winner,
            r4800_mean=r4800_mean,
            r4800_stddev=r4800_stddev,
            r4800_samples=r4800_samples,
            r4800_mask=r4800_mask,
            action_hashes=action_hashes,
        )
        self.groups += 1
        self.target_slots += quota
        self.target_hits += hits
        self.exact_sets += int(hits == quota)
        self.winner_hits += int(observation["winner"])
        self.confidence_hits += int(observation["confidence"])
        self.regret += float(observation["regret"])
        if bool(observation["distinguishable"]):
            self.distinguishable_groups += 1
            self.distinguishable_winner_hits += int(observation["winner"])

    def report(self) -> dict[str, int | float | None]:
        if not self.groups:
            return {
                "groups": 0,
                "target_slots": 0,
                "target_hits": 0,
                "target_positive_recall": None,
                "target_set_exact_fraction": None,
                "r4800_winner_retention": None,
                "top64_confidence_set_coverage_95": None,
                "distinguishable_groups": 0,
                "distinguishable_winner_retention": None,
                "mean_retained_r4800_regret": None,
            }
        return {
            "groups": self.groups,
            "target_slots": self.target_slots,
            "target_hits": self.target_hits,
            "target_positive_recall": self.target_hits
            / max(self.target_slots, 1),
            "target_set_exact_fraction": self.exact_sets / self.groups,
            "r4800_winner_retention": self.winner_hits / self.groups,
            "top64_confidence_set_coverage_95": (
                self.confidence_hits / self.groups
            ),
            "distinguishable_groups": self.distinguishable_groups,
            "distinguishable_winner_retention": (
                self.distinguishable_winner_hits
                / self.distinguishable_groups
                if self.distinguishable_groups
                else None
            ),
            "mean_retained_r4800_regret": self.regret / self.groups,
        }


def run_pointer_integration(
    config: PointerIntegrationConfig,
) -> dict[str, Any]:
    """Load selected stages, score both open splits, and classify P1."""
    overall_started = time.perf_counter()
    config.validate()
    mx.set_default_device(mx.gpu)
    allocator = configure_mlx_memory()
    source = source_provenance(Path(__file__).resolve().parents[2])
    models: dict[str, RelationalPointerRanker] = {}
    stage_reports: dict[str, dict[str, Any]] = {}
    stage_checkpoints: dict[str, dict[str, Any]] = {}
    for stage, run_dir in config.run_dirs().items():
        model, report_identity, checkpoint_identity = load_selected_stage(
            stage=stage,
            run_dir=run_dir,
            require_production=config.production,
        )
        models[stage] = model
        stage_reports[stage] = report_identity
        stage_checkpoints[stage] = checkpoint_identity
    foundation_ids = {
        json.dumps(
            report["foundation"],
            sort_keys=True,
            separators=(",", ":"),
        )
        for report in stage_reports.values()
    }
    if config.production and len(foundation_ids) != 1:
        raise ValueError("pointer stages do not share one foundation authorization")
    bundle_ids = {report.get("bundle_id") for report in stage_reports.values()}
    if config.production and (
        len(bundle_ids) != 1 or None in bundle_ids
    ):
        raise ValueError("pointer stages do not share one source bundle")
    replay_reports = {
        stage: load_stage_replay(
            stage=stage,
            path=path,
            stage_report=stage_reports[stage],
            checkpoint=stage_checkpoints[stage],
        )
        for stage, path in config.replay_paths().items()
        if path is not None
    }

    train_corpus = RelationalPointerCorpus(
        split="train",
        factor_cache=config.factor_cache,
        r3_cache=config.r3_cache,
        verify_r3_checksums=not config.production,
        verify_r3_semantics=not config.production,
    )
    validation_corpus = RelationalPointerCorpus(
        split="validation",
        factor_cache=config.factor_cache,
        r3_cache=config.r3_cache,
        verify_r3_checksums=False,
        verify_r3_semantics=False,
    )
    train = evaluate_integrated_split(
        corpus=train_corpus,
        models=models,
        max_shards=config.max_shards,
    )
    validation = evaluate_integrated_split(
        corpus=validation_corpus,
        models=models,
        max_shards=config.max_shards,
    )
    gates = pointer_integration_gates(
        stage_reports=stage_reports,
        train=train,
        validation=validation,
        production=config.production,
        replay_reports=replay_reports,
    )
    classification = classify_pointer_integration(gates)
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": EVALUATION_PROTOCOL_ID,
        "adr": ADR_ID,
        "mode": "production" if config.production else "bounded-smoke",
        "bundle_id": next(iter(bundle_ids)) if len(bundle_ids) == 1 else None,
        "stage_checkpoints": stage_checkpoints,
        "stage_reports": {
            stage: {
                "scientific_blake3": _canonical_blake3(report),
                "train": report["train"],
                "validation": report["validation"],
                "foundation": report["foundation"],
            }
            for stage, report in stage_reports.items()
        },
        "stage_replays": replay_reports,
        "factor_cache": {
            "train_payload_blake3": train_corpus.factor.manifest[
                "payload_blake3"
            ],
            "validation_payload_blake3": validation_corpus.factor.manifest[
                "payload_blake3"
            ],
        },
        "r3_cache_id": train_corpus.r3.cache_id,
        "train": train,
        "validation": validation,
        "gates": gates,
        "classification": classification,
        "passed": classification == CLASSIFICATION_SUFFICIENT,
        "information_boundary": {
            "open_train_used": True,
            "open_validation_used": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
            "new_teacher_compute_used": False,
            "external_compute_used": False,
            "hidden_order_read": False,
            "future_refill_read": False,
        },
        "claim_boundary": (
            "Offline proposal and selector sufficiency only; no gameplay or "
            "100-point claim."
        ),
        "source": source,
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "scientific_identity": identity,
        "scientific_blake3": _canonical_blake3(identity),
        "runtime": {
            "host": socket.gethostname().split(".")[0],
            "elapsed_seconds": time.perf_counter() - overall_started,
            "resource_usage": _resource_usage(),
            "mlx_allocator": allocator,
            "mlx_memory": mlx_memory_snapshot(),
        },
    }
    _write_json_atomic(config.output, report)
    return report


def replay_pointer_stage(
    *,
    stage: str,
    run_dir: Path,
    factor_cache: Path,
    r3_cache: Path,
    output: Path,
) -> dict[str, Any]:
    """Replay one selected production stage on a distinct host."""
    started = time.perf_counter()
    mx.set_default_device(mx.gpu)
    allocator = configure_mlx_memory()
    model, selected_report, checkpoint = load_selected_stage(
        stage=stage,
        run_dir=run_dir,
        require_production=True,
    )
    train = RelationalPointerCorpus(
        split="train",
        factor_cache=factor_cache,
        r3_cache=r3_cache,
        verify_r3_checksums=False,
        verify_r3_semantics=False,
    )
    validation = RelationalPointerCorpus(
        split="validation",
        factor_cache=factor_cache,
        r3_cache=r3_cache,
        verify_r3_checksums=False,
        verify_r3_semantics=False,
    )
    actual = {
        "train": evaluate_pointer_stage(
            model=model,
            corpus=train,
            stage=stage,
            batch_size=STAGE_BATCH_SIZES[stage],
        ),
        "validation": evaluate_pointer_stage(
            model=model,
            corpus=validation,
            stage=stage,
            batch_size=STAGE_BATCH_SIZES[stage],
        ),
    }
    expected = {
        split: selected_report[split]
        for split in ("train", "validation")
    }
    matched = actual == expected
    identity = {
        "schema_version": SCHEMA_VERSION,
        "experiment_id": EXPERIMENT_ID,
        "protocol_id": EVALUATION_PROTOCOL_ID,
        "adr": ADR_ID,
        "kind": "cross-host-stage-replay",
        "stage": stage,
        "checkpoint": checkpoint,
        "source_stage_report_scientific_blake3": _canonical_blake3(
            selected_report
        ),
        "expected": expected,
        "actual": actual,
        "matched": matched,
        "information_boundary": {
            "open_train_used": True,
            "open_validation_used": True,
            "sealed_test_opened": False,
            "gameplay_run": False,
        },
    }
    report = {
        "schema_version": SCHEMA_VERSION,
        "scientific_identity": identity,
        "scientific_blake3": _canonical_blake3(identity),
        "runtime": {
            "host": socket.gethostname().split(".")[0],
            "elapsed_seconds": time.perf_counter() - started,
            "resource_usage": _resource_usage(),
            "mlx_allocator": allocator,
            "mlx_memory": mlx_memory_snapshot(),
        },
    }
    _write_json_atomic(output, report)
    if not matched:
        raise RuntimeError(f"{stage} pointer replay differs from selected report")
    return report


def load_selected_stage(
    *,
    stage: str,
    run_dir: Path,
    require_production: bool,
) -> tuple[RelationalPointerRanker, dict[str, Any], dict[str, Any]]:
    """Load one selected stage through its final report and checkpoint hashes."""
    if stage not in STAGES:
        raise ValueError("selected pointer stage is unknown")
    report = _read_json(run_dir / "final-report.json", "pointer stage report")
    identity = report.get("scientific_identity")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or not isinstance(identity, dict)
        or report.get("scientific_blake3") != _canonical_blake3(identity)
        or identity.get("experiment_id") != EXPERIMENT_ID
        or identity.get("protocol_id") != PROTOCOL_ID
        or identity.get("adr") != ADR_ID
        or identity.get("stage") != stage
        or identity.get("model", {})
        .get("frozen_parent_parameter_tensor_blake3")
        != EXPECTED_PARENT_PARAMETER_BLAKE3
    ):
        raise ValueError(f"{stage} pointer report identity drifted")
    if require_production and (
        identity.get("mode") != "production"
        or identity.get("claims", {}).get(
            "offline_stage_comparison_complete"
        )
        is not True
        or not _stage_coverage_complete(identity)
    ):
        raise ValueError(f"{stage} pointer stage is not production-complete")
    selected = identity["selected_checkpoint"]
    checkpoint = run_dir / "checkpoints" / selected["name"]
    if checkpoint.is_dir():
        manifest_path = checkpoint / "checkpoint.json"
        model_path = checkpoint / "model.safetensors"
    else:
        published = _read_json(
            run_dir / "selected/selection.json",
            f"{stage} published selection",
        )
        if (
            published.get("checkpoint") != selected["name"]
            or published.get("checkpoint_manifest_blake3")
            != selected["manifest_blake3"]
            or published.get("model_blake3") != selected["model_blake3"]
        ):
            raise ValueError(f"{stage} published pointer selection drifted")
        manifest_path = run_dir / "selected/checkpoint.json"
        model_path = run_dir / "selected/model.safetensors"
    if (
        _checksum(manifest_path) != selected["manifest_blake3"]
        or _checksum(model_path) != selected["model_blake3"]
    ):
        raise ValueError(f"{stage} selected pointer checkpoint drifted")
    manifest = _read_json(manifest_path, f"{stage} checkpoint manifest")
    model = RelationalPointerRanker(
        RelationalPointerModelConfig.from_dict(manifest["model_config"])
    )
    model.load_weights(str(model_path))
    mx.eval(model.parameters())
    model.freeze_parent_for_pointer_training()
    if (
        parameter_tensor_blake3(model, parent_only=True)
        != EXPECTED_PARENT_PARAMETER_BLAKE3
        or parameter_tensor_blake3(model, trainable_only=True)
        != identity["model"]["final_pointer_parameter_tensor_blake3"]
    ):
        raise ValueError(f"{stage} pointer parameter identity drifted")
    return (
        model,
        identity,
        {
            "name": selected["name"],
            "manifest_blake3": selected["manifest_blake3"],
            "model_blake3": selected["model_blake3"],
            "stage_report_scientific_blake3": report["scientific_blake3"],
            "training_host": report.get("runtime", {}).get("host"),
        },
    )


def load_stage_replay(
    *,
    stage: str,
    path: Path,
    stage_report: dict[str, Any],
    checkpoint: dict[str, Any],
) -> dict[str, Any]:
    """Verify one exact stage replay occurred on a distinct host."""
    report = _read_json(path, f"{stage} pointer replay")
    identity = report.get("scientific_identity")
    replay_host = report.get("runtime", {}).get("host")
    if (
        report.get("schema_version") != SCHEMA_VERSION
        or not isinstance(identity, dict)
        or report.get("scientific_blake3") != _canonical_blake3(identity)
        or identity.get("experiment_id") != EXPERIMENT_ID
        or identity.get("protocol_id") != EVALUATION_PROTOCOL_ID
        or identity.get("adr") != ADR_ID
        or identity.get("kind") != "cross-host-stage-replay"
        or identity.get("stage") != stage
        or identity.get("matched") is not True
        or identity.get("checkpoint", {}).get("model_blake3")
        != checkpoint["model_blake3"]
        or identity.get("source_stage_report_scientific_blake3")
        != _canonical_blake3(stage_report)
        or not isinstance(replay_host, str)
        or replay_host == checkpoint.get("training_host")
    ):
        raise ValueError(f"{stage} pointer replay identity drifted")
    return {
        "scientific_blake3": report["scientific_blake3"],
        "host": replay_host,
        "source_host": checkpoint.get("training_host"),
        "matched": True,
    }


def score_pointer_stage_items(
    *,
    model: RelationalPointerRanker,
    corpus: RelationalPointerCorpus,
    stage: str,
    max_shards: int | None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Score every stage item once, preserving immutable shard-local indices."""
    started = time.perf_counter()
    counts = [
        int(entry["items"][stage])
        for entry in corpus.factor.shards
    ]
    scores = [
        np.full(count, np.nan, dtype=np.float32)
        for count in counts
    ]
    seen = [np.zeros(count, dtype=np.bool_) for count in counts]
    memo = PointerParentEncodingMemo()
    queries = 0
    items = 0
    model.eval()
    for batch in corpus.iter_stage_batches(
        stage=stage,
        batch_size=STAGE_BATCH_SIZES[stage],
        shuffle=False,
        seed=0,
        epoch=0,
        d6_augment=False,
    ):
        if max_shards is not None and batch.shard_index >= max_shards:
            break
        validate_pointer_batch(batch, stage=stage)
        encoding = memo.encoding(model, batch)
        output = model(batch, parent_encoding=encoding)
        mx.eval(output)
        values = np.asarray(output)
        mask = np.asarray(batch.item_mask)
        for row in range(values.shape[0]):
            valid = np.flatnonzero(mask[row])
            indices = batch.source_item_indices[row, valid]
            if (
                np.any(indices < 0)
                or np.any(indices >= len(scores[batch.shard_index]))
                or np.any(seen[batch.shard_index][indices])
            ):
                raise ValueError("pointer stage item coverage drifted")
            scores[batch.shard_index][indices] = values[row, valid]
            seen[batch.shard_index][indices] = True
            items += len(valid)
            queries += 1
    shard_limit = len(scores) if max_shards is None else max_shards
    complete = all(bool(np.all(seen[index])) for index in range(shard_limit))
    finite = all(
        bool(np.all(np.isfinite(scores[index][seen[index]])))
        for index in range(shard_limit)
    )
    if not complete or not finite:
        raise ValueError(f"{stage} pointer scoring was incomplete or nonfinite")
    elapsed = time.perf_counter() - started
    return scores, {
        "queries": queries,
        "items": items,
        "elapsed_seconds": elapsed,
        "items_per_second": items / max(elapsed, 1e-9),
        "all_requested_items_scored_once": complete,
        "all_scores_finite": finite,
        "parent_encoding_memo": memo.stats.report(),
    }


def evaluate_integrated_split(
    *,
    corpus: RelationalPointerCorpus,
    models: dict[str, RelationalPointerRanker],
    max_shards: int | None,
) -> dict[str, Any]:
    """Reconstruct proposals and top-64 complete actions on one open split."""
    if set(models) != set(STAGES):
        raise ValueError("pointer integration requires exactly three stage models")
    stage_scores: dict[str, list[np.ndarray]] = {}
    stage_runtime: dict[str, Any] = {}
    for stage in STAGES:
        stage_scores[stage], stage_runtime[stage] = score_pointer_stage_items(
            model=models[stage],
            corpus=corpus,
            stage=stage,
            max_shards=max_shards,
        )
    learned = _SelectionAccumulator()
    oracle = _SelectionAccumulator()
    phases = {
        0: (_SelectionAccumulator(), _SelectionAccumulator()),
        1: (_SelectionAccumulator(), _SelectionAccumulator()),
        2: (_SelectionAccumulator(), _SelectionAccumulator()),
    }
    subset_names = (
        "nature_token_available",
        "independent_draft_winner",
        "paired_draft_winner",
        "wildlife_none_winner",
        "wildlife_placed_winner",
        "frontier_anchor_winner",
    )
    subsets = {
        name: (_SelectionAccumulator(), _SelectionAccumulator())
        for name in subset_names
    }
    proposal_counts: list[int] = []
    group_latencies: list[float] = []
    candidates = 0
    shards = 0
    for shard_index, arrays in enumerate(corpus.factor.iter_shards()):
        if max_shards is not None and shard_index >= max_shards:
            break
        shards += 1
        scores = {
            stage: stage_scores[stage][shard_index]
            for stage in STAGES
        }
        selected = {
            stage: _selected_stage_items(
                scores=scores[stage],
                offsets=arrays[f"{stage}_query_offsets"],
                width=STAGE_WIDTHS[stage],
            )
            for stage in STAGES
        }
        for group_index, (left, right) in enumerate(
            pairwise(arrays["group_action_offsets"])
        ):
            group_started = time.perf_counter()
            left = int(left)
            right = int(right)
            count = right - left
            flags = arrays["action_source_flags"][left:right]
            frontier = (
                flags & GRADED_SOURCE_CHAMPION_FRONTIER
            ) != 0
            maps = {
                stage: arrays[f"{stage}_action_item"][left:right]
                for stage in STAGES
            }
            passing = ~frontier
            for stage in STAGES:
                valid = maps[stage] >= 0
                passing &= valid
                passing[valid] &= selected[stage][maps[stage][valid]]
            proposal = np.flatnonzero(frontier | passing).astype(np.int32)
            proposal_counts.append(len(proposal))
            combined = np.zeros(count, dtype=np.float32)
            for stage in STAGES:
                valid = maps[stage] >= 0
                combined[valid] += scores[stage][maps[stage][valid]]
            learned_local = frontier_anchored_retained_indices(
                scores=combined[proposal],
                source_flags=flags[proposal],
                action_hashes=arrays["action_hash"][left:right][proposal],
            )
            learned_retained = proposal[learned_local]
            ranks = arrays["action_expected_rank"][left:right]
            rank_mask = arrays["action_expected_rank_mask"][left:right]
            oracle_local = frontier_anchored_retained_indices(
                scores=np.where(
                    rank_mask[proposal],
                    -ranks[proposal],
                    -1e9,
                ),
                source_flags=flags[proposal],
                action_hashes=arrays["action_hash"][left:right][proposal],
            )
            oracle_retained = proposal[oracle_local]
            target = _group_target(
                expected_rank=ranks,
                expected_rank_mask=rank_mask,
                source_flags=flags,
                action_hashes=arrays["action_hash"][left:right],
            )
            winner = int(arrays["selected_index"][group_index])
            kwargs = {
                "target": target,
                "source_flags": flags,
                "winner": winner,
                "r4800_mean": arrays["action_r4800_mean"][left:right],
                "r4800_stddev": arrays["action_r4800_stddev"][left:right],
                "r4800_samples": arrays["action_r4800_samples"][left:right],
                "r4800_mask": arrays["action_r4800_mask"][left:right],
                "action_hashes": arrays["action_hash"][left:right],
            }
            learned.add(retained=learned_retained, **kwargs)
            oracle.add(retained=oracle_retained, **kwargs)
            phase = int(arrays["phase"][group_index])
            phases[phase][0].add(retained=learned_retained, **kwargs)
            phases[phase][1].add(retained=oracle_retained, **kwargs)
            for name in _group_subsets(
                arrays=arrays,
                group_index=group_index,
                action_left=left,
                winner=winner,
                source_flags=flags,
            ):
                subsets[name][0].add(retained=learned_retained, **kwargs)
                subsets[name][1].add(retained=oracle_retained, **kwargs)
            candidates += count
            group_latencies.append(time.perf_counter() - group_started)
    counts = np.asarray(proposal_counts, dtype=np.float64)
    latencies = np.asarray(group_latencies, dtype=np.float64) * 1000.0
    phase_names = {0: "early", 1: "middle", 2: "late"}
    expected_groups = sum(
        int(entry["groups"])
        for entry in corpus.factor.shards[:shards]
    )
    expected_candidates = sum(
        int(entry["candidates"])
        for entry in corpus.factor.shards[:shards]
    )
    return {
        "groups": learned.groups,
        "candidates": candidates,
        "shards": shards,
        "all_groups_scored_once": learned.groups == expected_groups,
        "all_candidates_scored_once": candidates == expected_candidates,
        "all_scores_finite": all(
            values["all_scores_finite"]
            for values in stage_runtime.values()
        ),
        "proposal_count": _distribution(counts),
        "mean_proposal_count": float(np.mean(counts)),
        "target_mean_proposal_count_at_most_512": float(np.mean(counts)) <= 512,
        "learned_top64": learned.report(),
        "oracle_inside_learned_proposal": oracle.report(),
        "learned_phase": {
            phase_names[index]: values[0].report()
            for index, values in phases.items()
        },
        "proposal_phase": {
            phase_names[index]: values[1].report()
            for index, values in phases.items()
        },
        "learned_subsets": {
            name: values[0].report()
            for name, values in subsets.items()
        },
        "proposal_subsets": {
            name: values[1].report()
            for name, values in subsets.items()
        },
        "stage_scoring": stage_runtime,
        "complete_action_integration_latency_ms": _distribution(latencies),
    }


def pointer_integration_gates(
    *,
    stage_reports: dict[str, dict[str, Any]],
    train: dict[str, Any],
    validation: dict[str, Any],
    production: bool,
    replay_reports: dict[str, dict[str, Any]],
) -> dict[str, bool]:
    """Apply ADR 0175 pipeline, proposal, and top-64 gates mechanically."""
    pipeline = {
        "all_stage_reports_present": set(stage_reports) == set(STAGES),
        "all_stage_reports_production": (
            not production
            or all(report["mode"] == "production" for report in stage_reports.values())
        ),
        "all_stage_coverage_complete": (
            not production
            or all(_stage_coverage_complete(report) for report in stage_reports.values())
        ),
        "all_stage_scores_finite": all(
            report["train"]["all_scores_finite"]
            and report["validation"]["all_scores_finite"]
            for report in stage_reports.values()
        ),
        "all_stage_parents_exact": all(
            report["model"]["frozen_parent_parameter_tensor_blake3"]
            == EXPECTED_PARENT_PARAMETER_BLAKE3
            for report in stage_reports.values()
        ),
        "all_stage_replays_exact_and_distinct": (
            not production
            or (
                set(replay_reports) == set(STAGES)
                and all(
                    values["matched"]
                    and values["host"] != values["source_host"]
                    for values in replay_reports.values()
                )
            )
        ),
        "integration_coverage": all(
            values["all_groups_scored_once"]
            and values["all_candidates_scored_once"]
            and values["all_scores_finite"]
            for values in (train, validation)
        ),
    }
    tile_stage = {
        "tile_validation_recall_at_least_0_90": (
            float(stage_reports["tile"]["validation"]["target_factor_recall"])
            >= 0.90
        )
    }
    proposal = {
        "train_target_recall_above_0_98": _metric(
            train,
            "oracle_inside_learned_proposal",
            "target_positive_recall",
        )
        > 0.98,
        "validation_target_recall_above_0_98": _metric(
            validation,
            "oracle_inside_learned_proposal",
            "target_positive_recall",
        )
        > 0.98,
        "train_winner_retention_above_0_98": _metric(
            train,
            "oracle_inside_learned_proposal",
            "r4800_winner_retention",
        )
        > 0.98,
        "validation_winner_retention_above_0_98": _metric(
            validation,
            "oracle_inside_learned_proposal",
            "r4800_winner_retention",
        )
        > 0.98,
        "mean_proposals_at_most_1024": (
            float(train["mean_proposal_count"]) <= 1024
            and float(validation["mean_proposal_count"]) <= 1024
        ),
    }
    for name, values in validation["proposal_phase"].items():
        proposal[f"{name}_target_recall_at_least_0_97"] = (
            float(values["target_positive_recall"]) >= 0.97
        )
    for name, values in validation["proposal_subsets"].items():
        if int(values["groups"]) >= 20:
            proposal[f"{name}_target_recall_at_least_0_95"] = (
                float(values["target_positive_recall"]) >= 0.95
            )

    selector = {
        "train_target_recall_above_0_98": _metric(
            train,
            "learned_top64",
            "target_positive_recall",
        )
        > 0.98,
        "validation_target_recall_above_0_98": _metric(
            validation,
            "learned_top64",
            "target_positive_recall",
        )
        > 0.98,
        "train_winner_retention_above_0_98": _metric(
            train,
            "learned_top64",
            "r4800_winner_retention",
        )
        > 0.98,
        "validation_winner_retention_above_0_98": _metric(
            validation,
            "learned_top64",
            "r4800_winner_retention",
        )
        > 0.98,
        "train_confidence_coverage_at_least_0_99": _metric(
            train,
            "learned_top64",
            "top64_confidence_set_coverage_95",
        )
        >= 0.99,
        "validation_confidence_coverage_at_least_0_99": _metric(
            validation,
            "learned_top64",
            "top64_confidence_set_coverage_95",
        )
        >= 0.99,
        "train_regret_below_0_15": _metric(
            train,
            "learned_top64",
            "mean_retained_r4800_regret",
        )
        < 0.15,
        "validation_regret_below_0_15": _metric(
            validation,
            "learned_top64",
            "mean_retained_r4800_regret",
        )
        < 0.15,
    }
    for name, values in validation["learned_phase"].items():
        selector[f"{name}_winner_retention_at_least_0_97"] = (
            float(values["r4800_winner_retention"]) >= 0.97
        )
        selector[f"{name}_confidence_coverage_at_least_0_98"] = (
            float(values["top64_confidence_set_coverage_95"]) >= 0.98
        )
        selector[f"{name}_regret_below_0_20"] = (
            float(values["mean_retained_r4800_regret"]) < 0.20
        )
    for name, values in validation["learned_subsets"].items():
        if int(values["groups"]) >= 20:
            selector[f"{name}_winner_retention_at_least_0_95"] = (
                float(values["r4800_winner_retention"]) >= 0.95
            )
            selector[f"{name}_confidence_coverage_at_least_0_95"] = (
                float(values["top64_confidence_set_coverage_95"]) >= 0.95
            )
            selector[f"{name}_regret_below_0_25"] = (
                float(values["mean_retained_r4800_regret"]) < 0.25
            )
    return {
        **{f"pipeline_{name}": value for name, value in pipeline.items()},
        **tile_stage,
        **{f"proposal_{name}": value for name, value in proposal.items()},
        **{f"selector_{name}": value for name, value in selector.items()},
        "pipeline_passed": all(pipeline.values()),
        "tile_stage_passed": all(tile_stage.values()),
        "proposal_passed": all(proposal.values()),
        "selector_passed": all(selector.values()),
    }


def classify_pointer_integration(gates: dict[str, bool]) -> str:
    """Classify ADR 0175 in preregistered precedence order."""
    if not gates["pipeline_passed"]:
        return CLASSIFICATION_PIPELINE_INVALID
    if not gates["tile_stage_passed"]:
        return CLASSIFICATION_TILE_INSUFFICIENT
    if not gates["proposal_passed"]:
        return CLASSIFICATION_PROPOSAL_INSUFFICIENT
    if not gates["selector_passed"]:
        return CLASSIFICATION_SELECTOR_INSUFFICIENT
    return CLASSIFICATION_SUFFICIENT


def _selected_stage_items(
    *,
    scores: np.ndarray,
    offsets: np.ndarray,
    width: int,
) -> np.ndarray:
    selected = np.zeros(len(scores), dtype=np.bool_)
    for left, right in pairwise(offsets):
        left = int(left)
        right = int(right)
        ranking = sorted(
            range(left, right),
            key=lambda index: (-float(scores[index]), index),
        )
        selected[ranking[: min(width, right - left)]] = True
    return selected


def _group_target(
    *,
    expected_rank: np.ndarray,
    expected_rank_mask: np.ndarray,
    source_flags: np.ndarray,
    action_hashes: np.ndarray,
) -> np.ndarray:
    count = len(expected_rank)
    return build_expected_rank_target_mask(
        expected_rank=expected_rank.reshape(1, count),
        expected_rank_mask=expected_rank_mask.reshape(1, count),
        source_flags=source_flags.reshape(1, count),
        candidate_mask=np.ones((1, count), dtype=np.bool_),
        action_hashes=action_hashes.reshape(1, count, -1),
    )[0]


def _r4800_observation(
    *,
    retained: np.ndarray,
    winner: int,
    r4800_mean: np.ndarray,
    r4800_stddev: np.ndarray,
    r4800_samples: np.ndarray,
    r4800_mask: np.ndarray,
    action_hashes: np.ndarray,
) -> dict[str, bool | float]:
    labeled = np.flatnonzero(r4800_mask).astype(np.int32)
    if len(labeled) < 2:
        raise ValueError("pointer integration requires two R4800 actions")
    ranking = np.asarray(
        sorted(
            (int(index) for index in labeled),
            key=lambda index: (
                -float(r4800_mean[index]),
                bytes(action_hashes[index]),
            ),
        ),
        dtype=np.int32,
    )
    if int(ranking[0]) != winner:
        raise ValueError("stored selected action is not the stable R4800 winner")
    runner_up = int(ranking[1])
    errors = r4800_stddev / np.sqrt(np.maximum(r4800_samples, 1.0))
    distinguishable = r4800_mean[winner] - r4800_mean[runner_up] > (
        NORMAL_95 * np.hypot(errors[winner], errors[runner_up])
    )
    confidence = np.zeros(len(r4800_mean), dtype=np.bool_)
    confidence[labeled] = r4800_mean[winner] - r4800_mean[labeled] <= (
        NORMAL_95 * np.hypot(errors[winner], errors[labeled])
    )
    retained_labeled = retained[r4800_mask[retained]]
    regret = (
        float(r4800_mean[winner] - np.max(r4800_mean[retained_labeled]))
        if len(retained_labeled)
        else float(np.ptp(r4800_mean[labeled]))
    )
    return {
        "winner": bool(np.any(retained == winner)),
        "confidence": bool(np.any(confidence[retained])),
        "distinguishable": bool(distinguishable),
        "regret": regret,
    }


def _group_subsets(
    *,
    arrays: dict[str, np.ndarray],
    group_index: int,
    action_left: int,
    winner: int,
    source_flags: np.ndarray,
) -> tuple[str, ...]:
    names = []
    if int(arrays["nature_tokens"][group_index]) > 0:
        names.append("nature_token_available")
    if int(arrays["action_draft_kind"][action_left + winner]) == 1:
        names.append("independent_draft_winner")
    else:
        names.append("paired_draft_winner")
    if (
        int(source_flags[winner]) & GRADED_SOURCE_CHAMPION_FRONTIER
    ) != 0:
        names.append("frontier_anchor_winner")
    wildlife_item = int(
        arrays["wildlife_action_item"][action_left + winner]
    )
    if wildlife_item >= 0:
        present = (
            float(arrays["wildlife_item_features"][wildlife_item, 0])
            > 0.5
        )
        names.append(
            "wildlife_placed_winner" if present else "wildlife_none_winner"
        )
    return tuple(names)


def _stage_coverage_complete(report: dict[str, Any]) -> bool:
    return all(
        bool(report[split]["all_queries_scored_once"])
        and bool(report[split]["all_items_scored_once"])
        and bool(report[split]["all_scores_finite"])
        for split in ("train", "validation")
    )


def _metric(report: dict[str, Any], section: str, name: str) -> float:
    value = report[section][name]
    if value is None or not math.isfinite(float(value)):
        raise ValueError(f"pointer integration metric is absent: {section}.{name}")
    return float(value)


def _distribution(values: np.ndarray) -> dict[str, float | int]:
    if values.ndim != 1 or not len(values) or not np.all(np.isfinite(values)):
        raise ValueError("pointer distribution input is invalid")
    return {
        "count": len(values),
        "minimum": float(np.min(values)),
        "mean": float(np.mean(values)),
        "p50": float(np.quantile(values, 0.50)),
        "p90": float(np.quantile(values, 0.90)),
        "p99": float(np.quantile(values, 0.99)),
        "maximum": float(np.max(values)),
    }


def _canonical_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        ).encode()
    ).hexdigest()


def _checksum(path: Path) -> str:
    digest = blake3.blake3()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read {label}: {error}") from error
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _resource_usage() -> dict[str, float | int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    maximum_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        maximum_rss *= 1024
    return {
        "maximum_rss_bytes": maximum_rss,
        "user_cpu_seconds": float(usage.ru_utime),
        "system_cpu_seconds": float(usage.ru_stime),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    integrate = subparsers.add_parser("integrate")
    integrate.add_argument("--draft-run-dir", type=Path, required=True)
    integrate.add_argument("--tile-run-dir", type=Path, required=True)
    integrate.add_argument("--wildlife-run-dir", type=Path, required=True)
    integrate.add_argument(
        "--factor-cache",
        type=Path,
        default=DEFAULT_FACTOR_CACHE,
    )
    integrate.add_argument("--r3-cache", type=Path, default=DEFAULT_R3_CACHE)
    integrate.add_argument("--max-shards", type=int)
    integrate.add_argument("--draft-replay", type=Path)
    integrate.add_argument("--tile-replay", type=Path)
    integrate.add_argument("--wildlife-replay", type=Path)
    integrate.add_argument("--output", type=Path, required=True)
    replay = subparsers.add_parser("replay")
    replay.add_argument("--stage", choices=STAGES, required=True)
    replay.add_argument("--run-dir", type=Path, required=True)
    replay.add_argument(
        "--factor-cache",
        type=Path,
        default=DEFAULT_FACTOR_CACHE,
    )
    replay.add_argument("--r3-cache", type=Path, default=DEFAULT_R3_CACHE)
    replay.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = _parser().parse_args()
    if args.command == "integrate":
        report = run_pointer_integration(
            PointerIntegrationConfig(
                draft_run_dir=args.draft_run_dir,
                tile_run_dir=args.tile_run_dir,
                wildlife_run_dir=args.wildlife_run_dir,
                factor_cache=args.factor_cache,
                r3_cache=args.r3_cache,
                max_shards=args.max_shards,
                draft_replay=args.draft_replay,
                tile_replay=args.tile_replay,
                wildlife_replay=args.wildlife_replay,
                output=args.output,
            )
        )
    else:
        report = replay_pointer_stage(
            stage=args.stage,
            run_dir=args.run_dir,
            factor_cache=args.factor_cache,
            r3_cache=args.r3_cache,
            output=args.output,
        )
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
