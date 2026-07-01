"""Oracle-proposal complete-action selector feasibility for ADR 0126."""

from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import socket
import time
from collections.abc import Iterator, Mapping
from dataclasses import asdict, dataclass
from itertools import pairwise
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
from mlx.utils import tree_flatten

from cascadia_mlx.full_legal_hierarchical_factor_retrieval import (
    STAGES,
    HierarchicalFactorCache,
)
from cascadia_mlx.graded_oracle_factor_integration import (
    PROBE_KINDS,
    PROBE_SEEDS,
    FactorProbeConfig,
    FrozenFactorCache,
    balanced_factor_binary_loss,
    build_factor_probe,
    configure_mlx_memory,
    factor_payload_blake3,
    mlx_memory_snapshot,
)
from cascadia_mlx.graded_oracle_frontier_anchor import (
    GRADED_SOURCE_CHAMPION_FRONTIER,
    frontier_anchored_retained_indices,
)
from cascadia_mlx.graded_oracle_frontier_warm_start import checksum

EXPERIMENT_ID = "oracle-proposal-complete-action-selector-v1"
CACHE_SCHEMA = "graded-oracle-oracle-proposal-factors-v1"
CACHE_SCHEMA_VERSION = 1
MAXIMUM_PEAK_RSS_BYTES = 4 * 1024**3


def _json_default(value: object) -> object:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, default=_json_default, indent=2, sort_keys=True) + "\n"
    )
    os.replace(temporary, path)


def _append_json(path: Path, value: object) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, default=_json_default, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _scientific_blake3(value: object) -> str:
    return blake3.blake3(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
            default=_json_default,
        ).encode()
    ).hexdigest()


def oracle_proposal_mask(arrays: Mapping[str, np.ndarray], group_index: int) -> np.ndarray:
    """Return the exact ADR 0114 oracle proposal for one group."""
    left = int(arrays["group_action_offsets"][group_index])
    right = int(arrays["group_action_offsets"][group_index + 1])
    flags = arrays["action_source_flags"][left:right]
    frontier = (flags & GRADED_SOURCE_CHAMPION_FRONTIER) != 0
    passing = ~frontier
    for stage in STAGES:
        mapping = arrays[f"{stage}_action_item"][left:right]
        valid = mapping >= 0
        passing &= valid
        passing[valid] &= arrays[f"{stage}_item_target"][mapping[valid]]
    return frontier | passing


def _hierarchical_groups(
    cache: HierarchicalFactorCache,
) -> Iterator[dict[str, Any]]:
    for arrays in cache.iter_shards():
        for group_index in range(len(arrays["group_action_offsets"]) - 1):
            left = int(arrays["group_action_offsets"][group_index])
            right = int(arrays["group_action_offsets"][group_index + 1])
            selected = int(arrays["selected_index"][group_index])
            if selected < 0 or selected >= right - left:
                raise ValueError("hierarchy selected index is outside its group")
            yield {
                "action_hash": arrays["action_hash"][left:right],
                "proposal": oracle_proposal_mask(arrays, group_index),
                "phase": int(arrays["phase"][group_index]),
                "nature_token_available": (int(arrays["nature_tokens"][group_index]) > 0),
                "independent_draft_winner": (
                    int(arrays["action_draft_kind"][left + selected]) == 1
                ),
            }


def filter_factor_batch(
    factors: np.ndarray,
    metadata: Mapping[str, np.ndarray],
    hierarchy_groups: Iterator[Mapping[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, int]]:
    """Filter one factor batch while proving group/action alignment."""
    kept_factors: list[np.ndarray] = []
    kept: dict[str, list[np.ndarray]] = {
        "target": [],
        "source_flags": [],
        "screen_rank": [],
        "action_hash": [],
        "r4800_mean": [],
        "r4800_mask": [],
    }
    offsets = [0]
    selected: list[int] = []
    phases: list[int] = []
    nature_token_available: list[bool] = []
    independent_draft_winner: list[bool] = []
    original_actions = 0
    retained_actions = 0
    selected_outside = 0
    for group_index, (left, right) in enumerate(pairwise(metadata["group_offsets"])):
        left = int(left)
        right = int(right)
        hierarchy = next(hierarchy_groups)
        hierarchy_hashes = np.asarray(hierarchy["action_hash"])
        proposal = np.asarray(hierarchy["proposal"], dtype=np.bool_)
        factor_hashes = metadata["action_hash"][left:right]
        if not np.array_equal(factor_hashes, hierarchy_hashes):
            raise ValueError("factor and hierarchy action hashes are not aligned")
        if len(proposal) != right - left:
            raise ValueError("hierarchy proposal width does not match factor group")
        indices = np.flatnonzero(proposal)
        if not len(indices):
            raise ValueError("oracle proposal is empty")
        kept_factors.append(np.asarray(factors[left:right])[indices])
        for name in kept:
            kept[name].append(np.asarray(metadata[name][left:right])[indices])
        source_selected = int(metadata["selected_index"][group_index])
        matches = np.flatnonzero(indices == source_selected)
        if len(matches):
            selected.append(int(matches[0]))
        else:
            selected.append(-1)
            selected_outside += 1
        phase = int(hierarchy["phase"])
        if phase not in (0, 1, 2):
            raise ValueError("hierarchy group has an invalid phase")
        phases.append(phase)
        nature_token_available.append(bool(hierarchy["nature_token_available"]))
        independent_draft_winner.append(bool(hierarchy["independent_draft_winner"]))
        offsets.append(offsets[-1] + len(indices))
        original_actions += right - left
        retained_actions += len(indices)
    result = {name: np.concatenate(values, axis=0) for name, values in kept.items()}
    result["group_offsets"] = np.asarray(offsets, dtype=np.int64)
    result["selected_index"] = np.asarray(selected, dtype=np.int32)
    result["phase"] = np.asarray(phases, dtype=np.int8)
    result["nature_token_available"] = np.asarray(
        nature_token_available,
        dtype=np.bool_,
    )
    result["independent_draft_winner"] = np.asarray(
        independent_draft_winner,
        dtype=np.bool_,
    )
    return (
        np.concatenate(kept_factors, axis=0),
        result,
        {
            "groups": len(selected),
            "original_actions": original_actions,
            "retained_actions": retained_actions,
            "selected_outside_proposal": selected_outside,
        },
    )


def proposal_payload_blake3(manifest: dict[str, Any]) -> str:
    """Hash the portable scientific identity of a filtered cache."""
    portable = {
        "cache_schema": manifest["cache_schema"],
        "split": manifest["split"],
        "source_factor_payload_blake3": manifest["source_factor_payload_blake3"],
        "source_hierarchy_payload_blake3": manifest["source_hierarchy_payload_blake3"],
        "dataset_manifest_blake3": manifest["dataset_manifest_blake3"],
        "factor_names": manifest["factor_names"],
        "factor_count": manifest["factor_count"],
        "factor_dim": manifest["factor_dim"],
        "slice_metadata_fields": manifest["slice_metadata_fields"],
        "groups": manifest["groups"],
        "original_candidates": manifest["original_candidates"],
        "candidates": manifest["candidates"],
        "selected_outside_proposal": manifest["selected_outside_proposal"],
        "batches": [
            {
                key: entry[key]
                for key in (
                    "index",
                    "factors_file",
                    "metadata_file",
                    "factors_blake3",
                    "metadata_blake3",
                    "groups",
                    "original_candidates",
                    "candidates",
                    "selected_outside_proposal",
                )
            }
            for entry in manifest["batches"]
        ],
    }
    return _scientific_blake3(portable)


def filter_factor_cache(
    *,
    factor_cache_root: Path,
    hierarchy_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Create one immutable oracle-proposal factor cache."""
    if output_root.exists():
        raise ValueError("oracle-proposal cache output already exists")
    factor_cache = FrozenFactorCache(factor_cache_root)
    hierarchy_cache = HierarchicalFactorCache(hierarchy_cache_root)
    if factor_cache.split != hierarchy_cache.split:
        raise ValueError("factor and hierarchy cache splits differ")
    factor_manifest = factor_cache.manifest
    hierarchy_manifest = hierarchy_cache.manifest
    if factor_manifest["dataset_manifest_blake3"] != hierarchy_manifest["dataset_manifest_blake3"]:
        raise ValueError("factor and hierarchy dataset identities differ")
    if (
        factor_cache.group_count != hierarchy_cache.group_count
        or factor_cache.candidate_count != hierarchy_cache.candidate_count
    ):
        raise ValueError("factor and hierarchy cache coverage differs")

    temporary = output_root.with_name(output_root.name + ".tmp")
    if temporary.exists():
        shutil.rmtree(temporary)
    (temporary / "batches").mkdir(parents=True)
    hierarchy_groups = _hierarchical_groups(hierarchy_cache)
    entries: list[dict[str, Any]] = []
    groups = 0
    original_candidates = 0
    candidates = 0
    selected_outside = 0
    started = time.perf_counter()
    for batch_index, (factors, metadata) in enumerate(factor_cache.iter_batches()):
        filtered_factors, filtered_metadata, counts = filter_factor_batch(
            factors,
            metadata,
            hierarchy_groups,
        )
        factor_name = f"batches/batch-{batch_index:06d}-factors.npy"
        metadata_name = f"batches/batch-{batch_index:06d}-metadata.npz"
        np.save(temporary / factor_name, filtered_factors)
        np.savez_compressed(temporary / metadata_name, **filtered_metadata)
        entry = {
            "index": batch_index,
            "factors_file": factor_name,
            "metadata_file": metadata_name,
            "factors_blake3": checksum(temporary / factor_name),
            "metadata_blake3": checksum(temporary / metadata_name),
            "groups": counts["groups"],
            "original_candidates": counts["original_actions"],
            "candidates": counts["retained_actions"],
            "selected_outside_proposal": counts["selected_outside_proposal"],
        }
        entries.append(entry)
        groups += counts["groups"]
        original_candidates += counts["original_actions"]
        candidates += counts["retained_actions"]
        selected_outside += counts["selected_outside_proposal"]
    try:
        next(hierarchy_groups)
    except StopIteration:
        pass
    else:
        raise ValueError("hierarchy cache contains unmatched trailing groups")
    if groups != factor_cache.group_count or original_candidates != factor_cache.candidate_count:
        raise ValueError("filtered cache coverage drifted")

    manifest = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "cache_schema": CACHE_SCHEMA,
        "experiment_id": EXPERIMENT_ID,
        "split": factor_cache.split,
        "dataset_id": (f"{factor_manifest['dataset_id']}-oracle-proposal"),
        "dataset_manifest_blake3": factor_manifest["dataset_manifest_blake3"],
        "checkpoint": factor_manifest["checkpoint"],
        "checkpoint_manifest_blake3": factor_manifest["checkpoint_manifest_blake3"],
        "model_blake3": factor_manifest["model_blake3"],
        "factor_names": factor_manifest["factor_names"],
        "factor_count": factor_manifest["factor_count"],
        "factor_dim": factor_manifest["factor_dim"],
        "slice_metadata_fields": [
            "phase",
            "nature_token_available",
            "independent_draft_winner",
        ],
        "source_factor_payload_blake3": factor_cache.payload_blake3,
        "source_hierarchy_payload_blake3": hierarchy_manifest["payload_blake3"],
        "groups": groups,
        "original_candidates": original_candidates,
        "candidates": candidates,
        "selected_outside_proposal": selected_outside,
        "mean_proposal_count": candidates / groups,
        "batches": entries,
        "features_contain_targets_or_teacher_values": False,
        "oracle_proposal_membership_used_for_filter_only": True,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "host": socket.gethostname().split(".")[0],
        },
    }
    manifest["payload_blake3"] = proposal_payload_blake3(manifest)
    _write_json_atomic(temporary / "cache.json", manifest)
    os.replace(temporary, output_root)
    return manifest


class OracleProposalFactorCache(FrozenFactorCache):
    """ADR 0126 cache with the original factor-cache reader contract."""

    def __init__(self, root: str | Path, *, verify_checksums: bool = True):
        self.root = Path(root)
        oracle_manifest = json.loads((self.root / "cache.json").read_text())
        if (
            oracle_manifest.get("schema_version") != CACHE_SCHEMA_VERSION
            or oracle_manifest.get("cache_schema") != CACHE_SCHEMA
            or oracle_manifest.get("experiment_id") != EXPERIMENT_ID
            or tuple(oracle_manifest.get("slice_metadata_fields", ()))
            != (
                "phase",
                "nature_token_available",
                "independent_draft_winner",
            )
        ):
            raise ValueError("unsupported oracle-proposal factor cache")
        compatibility = dict(oracle_manifest)
        compatibility["cache_schema"] = "graded-oracle-frozen-candidate-factors-v1"
        compatibility["payload_blake3"] = factor_payload_blake3(compatibility)
        super().__init__(
            self.root,
            verify_checksums=verify_checksums,
            manifest=compatibility,
        )
        self.manifest = oracle_manifest
        self.payload_blake3 = str(oracle_manifest["payload_blake3"])


def _resource_usage() -> dict[str, int]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    peak_rss = int(usage.ru_maxrss)
    if platform.system() != "Darwin":
        peak_rss *= 1024
    return {
        "peak_process_rss_bytes": peak_rss,
        "process_swaps": int(getattr(usage, "ru_nswap", 0)),
    }


@dataclass
class _SelectorSlice:
    groups: int = 0
    target_positives: int = 0
    target_hits: int = 0
    exact_sets: int = 0
    winner_hits: int = 0
    regret: float = 0.0

    def add(
        self,
        *,
        target: np.ndarray,
        retained: np.ndarray,
        source_flags: np.ndarray,
        winner: int,
        r4800_mean: np.ndarray,
        r4800_mask: np.ndarray,
    ) -> None:
        retained_nonfrontier = retained[
            (source_flags[retained] & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
        ]
        quota = int(np.sum(target))
        recalled = int(np.sum(target[retained_nonfrontier]))
        retained_labeled = retained[r4800_mask[retained]]
        self.groups += 1
        self.target_positives += quota
        self.target_hits += recalled
        self.exact_sets += int(recalled == quota)
        self.winner_hits += int(winner in retained)
        if np.any(r4800_mask) and len(retained_labeled):
            self.regret += max(
                0.0,
                float(np.max(r4800_mean[r4800_mask])) - float(np.max(r4800_mean[retained_labeled])),
            )

    def report(self) -> dict[str, int | float | None]:
        if not self.groups:
            return {
                "groups": 0,
                "target_positives": 0,
                "target_hits": 0,
                "target_positive_recall": None,
                "target_set_exact_fraction": None,
                "top64_r4800_winner_recall": None,
                "mean_top64_retained_r4800_regret": None,
            }
        return {
            "groups": self.groups,
            "target_positives": self.target_positives,
            "target_hits": self.target_hits,
            "target_positive_recall": (self.target_hits / max(self.target_positives, 1)),
            "target_set_exact_fraction": self.exact_sets / self.groups,
            "top64_r4800_winner_recall": self.winner_hits / self.groups,
            "mean_top64_retained_r4800_regret": self.regret / self.groups,
        }


def evaluate_selector_probe(
    model: nn.Module,
    cache: OracleProposalFactorCache,
) -> dict[str, Any]:
    """Measure aggregate and preregistered slice metrics for one selector."""
    model.eval()
    overall = _SelectorSlice()
    phases = {
        0: _SelectorSlice(),
        1: _SelectorSlice(),
        2: _SelectorSlice(),
    }
    subsets = {
        "nature_token_available": _SelectorSlice(),
        "independent_draft_winner": _SelectorSlice(),
    }
    candidates = 0
    finite = True
    total_loss = 0.0
    for factor_values, metadata in cache.iter_batches():
        target = metadata["target"].astype(np.bool_, copy=False)
        source_flags = metadata["source_flags"].astype(
            np.int32,
            copy=False,
        )
        eligible = (source_flags & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
        offsets = tuple(int(value) for value in metadata["group_offsets"])
        factors = mx.array(np.asarray(factor_values))
        logits = model(
            factors,
            offsets,
            metadata["screen_rank"],
            metadata["action_hash"],
        )
        loss = balanced_factor_binary_loss(
            model,
            factors,
            mx.array(target),
            mx.array(eligible),
            offsets,
            metadata["screen_rank"],
            metadata["action_hash"],
        )
        mx.eval(logits, loss)
        values = np.asarray(logits)
        finite &= bool(np.all(np.isfinite(values)))
        total_loss += float(loss.item()) * (len(offsets) - 1)
        selected_indices = metadata["selected_index"]
        r4800_mean = metadata["r4800_mean"]
        r4800_mask = metadata["r4800_mask"]
        action_hash = metadata["action_hash"]
        for group_index, (start, end) in enumerate(pairwise(offsets)):
            retained = frontier_anchored_retained_indices(
                scores=values[start:end],
                source_flags=source_flags[start:end],
                action_hashes=action_hash[start:end],
            )
            kwargs = {
                "target": target[start:end],
                "retained": retained,
                "source_flags": source_flags[start:end],
                "winner": int(selected_indices[group_index]),
                "r4800_mean": r4800_mean[start:end],
                "r4800_mask": r4800_mask[start:end],
            }
            overall.add(**kwargs)
            phase = int(metadata["phase"][group_index])
            if phase not in phases:
                raise ValueError("selector cache contains an invalid phase")
            phases[phase].add(**kwargs)
            for name in subsets:
                if bool(metadata[name][group_index]):
                    subsets[name].add(**kwargs)
            candidates += end - start
    phase_names = {0: "early", 1: "middle", 2: "late"}
    return {
        **overall.report(),
        "candidates": candidates,
        "balanced_binary_loss": total_loss / max(overall.groups, 1),
        "all_scores_finite": finite,
        "all_groups_scored_once": overall.groups == cache.group_count,
        "all_candidates_scored_once": candidates == cache.candidate_count,
        "phase": {
            phase_names[index]: accumulator.report() for index, accumulator in phases.items()
        },
        "subsets": {name: accumulator.report() for name, accumulator in subsets.items()},
    }


def train_probe(
    *,
    kind: str,
    train_cache_root: Path,
    validation_cache_root: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Train one frozen architecture with train-only checkpoint selection."""
    config = FactorProbeConfig(kind=kind, seed=PROBE_SEEDS[kind])
    config.validate()
    if output_root.exists():
        raise ValueError("selector probe output already exists")
    allocator = configure_mlx_memory()
    train_cache = OracleProposalFactorCache(train_cache_root)
    validation_cache = OracleProposalFactorCache(validation_cache_root)
    if train_cache.split != "train" or validation_cache.split != "validation":
        raise ValueError("selector probe cache split mismatch")
    mx.random.seed(config.seed)
    model = build_factor_probe(kind)
    optimizer = optim.AdamW(
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    loss_and_grad = nn.value_and_grad(model, balanced_factor_binary_loss)
    output_root.mkdir(parents=True)
    metrics_path = output_root / "metrics.jsonl"
    started = time.perf_counter()
    best_key: tuple[float, float, float] | None = None
    best_epoch = 0
    finite_training = True
    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0
        batches = 0
        for factor_values, metadata in train_cache.iter_batches(
            shuffle=True,
            seed=config.seed + epoch,
        ):
            factors = mx.array(np.asarray(factor_values))
            target = mx.array(metadata["target"])
            source_flags = mx.array(metadata["source_flags"])
            eligible = (source_flags.astype(mx.int32) & GRADED_SOURCE_CHAMPION_FRONTIER) == 0
            offsets = tuple(int(value) for value in metadata["group_offsets"])
            loss, gradients = loss_and_grad(
                model,
                factors,
                target,
                eligible,
                offsets,
                metadata["screen_rank"],
                metadata["action_hash"],
            )
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            loss_value = float(loss.item())
            finite_training &= np.isfinite(loss_value)
            if not finite_training:
                raise RuntimeError("selector training became nonfinite")
            epoch_loss += loss_value
            batches += 1
        train_metrics = evaluate_selector_probe(model, train_cache)
        event = {
            "epoch": epoch,
            "train_loss": epoch_loss / max(batches, 1),
            "elapsed_seconds": time.perf_counter() - started,
            "train": train_metrics,
        }
        _append_json(metrics_path, event)
        print(json.dumps(event, default=_json_default, sort_keys=True), flush=True)
        key = (
            float(train_metrics["target_positive_recall"]),
            float(train_metrics["target_set_exact_fraction"]),
            -float(train_metrics["balanced_binary_loss"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_epoch = epoch
            mx.save_safetensors(
                str(output_root / "best.safetensors"),
                dict(tree_flatten(model.parameters())),
            )
            _write_json_atomic(output_root / "best.json", event)
        mx.clear_cache()
    model.load_weights(str(output_root / "best.safetensors"))
    mx.eval(model.parameters())
    train_metrics = evaluate_selector_probe(model, train_cache)
    validation_metrics = evaluate_selector_probe(model, validation_cache)
    report = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "host": socket.gethostname().split(".")[0],
        "probe": asdict(config),
        "best_epoch": best_epoch,
        "weights_blake3": checksum(output_root / "best.safetensors"),
        "train_cache_payload_blake3": train_cache.payload_blake3,
        "validation_cache_payload_blake3": validation_cache.payload_blake3,
        "train": train_metrics,
        "validation": validation_metrics,
        "finite_training": finite_training,
        "checkpoint_selection_uses_validation": False,
        "validation_evaluations": 1,
        "execution": {
            "elapsed_seconds": time.perf_counter() - started,
            "mlx_allocator": allocator,
            "mlx_memory": mlx_memory_snapshot(),
            **_resource_usage(),
        },
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    scientific = {
        key: report[key]
        for key in (
            "experiment_id",
            "probe",
            "best_epoch",
            "weights_blake3",
            "train_cache_payload_blake3",
            "validation_cache_payload_blake3",
            "train",
            "validation",
            "finite_training",
            "checkpoint_selection_uses_validation",
            "validation_evaluations",
            "test_split_opened",
            "gameplay_opened",
            "new_teacher_compute_used",
            "external_compute_used",
        )
    }
    report["scientific_blake3"] = _scientific_blake3(scientific)
    _write_json_atomic(output_root / "report.json", report)
    return report


def probe_gates(report: Mapping[str, Any]) -> dict[str, bool]:
    """Apply ADR 0126's per-arm pipeline and feasibility gates."""
    train = report["train"]
    validation = report["validation"]
    pipeline = {
        "experiment_identity": report.get("experiment_id") == EXPERIMENT_ID,
        "finite_training": bool(report.get("finite_training")),
        "train_coverage": bool(train["all_groups_scored_once"])
        and bool(train["all_candidates_scored_once"])
        and bool(train["all_scores_finite"]),
        "validation_coverage": bool(validation["all_groups_scored_once"])
        and bool(validation["all_candidates_scored_once"])
        and bool(validation["all_scores_finite"]),
        "train_only_selection": not bool(report.get("checkpoint_selection_uses_validation", True)),
        "single_validation_evaluation": int(report.get("validation_evaluations", -1)) == 1,
        "resources": int(report["execution"]["peak_process_rss_bytes"]) < MAXIMUM_PEAK_RSS_BYTES
        and int(report["execution"]["process_swaps"]) == 0,
        "closed_domains": all(
            report.get(name) is False
            for name in (
                "test_split_opened",
                "gameplay_opened",
                "new_teacher_compute_used",
                "external_compute_used",
            )
        ),
    }
    feasibility = {
        "train_target_recall_at_least_0_95": float(train["target_positive_recall"]) >= 0.95,
        "train_exact_sets_at_least_0_50": float(train["target_set_exact_fraction"]) >= 0.50,
        "validation_target_recall_at_least_0_90": float(validation["target_positive_recall"])
        >= 0.90,
        "validation_winner_recall_at_least_0_98": float(validation["top64_r4800_winner_recall"])
        >= 0.98,
        "validation_regret_below_0_15": float(validation["mean_top64_retained_r4800_regret"])
        < 0.15,
    }
    for name, values in validation["phase"].items():
        feasibility[f"{name}_winner_recall_at_least_0_97"] = (
            float(values["top64_r4800_winner_recall"]) >= 0.97
        )
        feasibility[f"{name}_retained_regret_below_0_20"] = (
            float(values["mean_top64_retained_r4800_regret"]) < 0.20
        )
    for name in (
        "nature_token_available",
        "independent_draft_winner",
    ):
        values = validation["subsets"][name]
        if int(values["groups"]) >= 20:
            feasibility[f"{name}_winner_recall_at_least_0_95"] = (
                float(values["top64_r4800_winner_recall"]) >= 0.95
            )
            feasibility[f"{name}_retained_regret_below_0_25"] = (
                float(values["mean_top64_retained_r4800_regret"]) < 0.25
            )
    return {
        **{f"pipeline_{key}": value for key, value in pipeline.items()},
        **feasibility,
        "pipeline_passed": all(pipeline.values()),
        "feasibility_passed": all(feasibility.values()),
    }


def classify_reports(
    reports: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Select the smallest feasible arm after all pipelines pass."""
    if set(reports) != set(PROBE_KINDS):
        raise ValueError("selector classification requires every frozen arm")
    gates = {kind: probe_gates(report) for kind, report in reports.items()}
    if not all(values["pipeline_passed"] for values in gates.values()):
        classification = "oracle_proposal_selector_pipeline_invalid"
        selected = None
    else:
        order = (
            "wide-concat",
            "pairwise-gated",
            "factor-attention",
            "screen-relative",
        )
        selected = next(
            (kind for kind in order if gates[kind]["feasibility_passed"]),
            None,
        )
        classification = (
            "oracle_proposal_selector_feasible"
            if selected is not None
            else "oracle_proposal_selector_representation_insufficient"
        )
    return {
        "classification": classification,
        "selected_kind": selected,
        "gates": gates,
    }


def combine(
    *,
    artifact_root: Path,
    markdown_path: Path,
) -> dict[str, Any]:
    """Combine all four frozen selector arms."""
    train_cache = json.loads((artifact_root / "cache" / "train" / "cache.json").read_text())
    validation_cache = json.loads(
        (artifact_root / "cache" / "validation" / "cache.json").read_text()
    )
    local_identity = json.loads(
        (artifact_root / "reports" / "source-identity-john1.json").read_text()
    )
    remote_identity = json.loads(
        (artifact_root / "reports" / "source-identity-john4.json").read_text()
    )
    reports = {
        kind: json.loads((artifact_root / "runs" / kind / "report.json").read_text())
        for kind in PROBE_KINDS
    }
    result = classify_reports(reports)
    campaign_pipeline = {
        "train_cache_identity": (
            train_cache.get("experiment_id") == EXPERIMENT_ID
            and train_cache.get("split") == "train"
            and train_cache.get("cache_schema") == CACHE_SCHEMA
            and proposal_payload_blake3(train_cache) == train_cache.get("payload_blake3")
        ),
        "validation_cache_identity": (
            validation_cache.get("experiment_id") == EXPERIMENT_ID
            and validation_cache.get("split") == "validation"
            and validation_cache.get("cache_schema") == CACHE_SCHEMA
            and proposal_payload_blake3(validation_cache) == validation_cache.get("payload_blake3")
        ),
        "source_identity_match": (
            local_identity.get("identity_kind") == "complete-mlx-runtime-source-v1"
            and remote_identity.get("identity_kind") == "complete-mlx-runtime-source-v1"
            and local_identity.get("bundle_sha256") == remote_identity.get("bundle_sha256")
        ),
        "all_arms_use_frozen_caches": all(
            report.get("train_cache_payload_blake3") == train_cache.get("payload_blake3")
            and report.get("validation_cache_payload_blake3")
            == validation_cache.get("payload_blake3")
            for report in reports.values()
        ),
        "all_frozen_arms_present": set(reports) == set(PROBE_KINDS),
    }
    campaign_pipeline["passed"] = all(campaign_pipeline.values())
    if not campaign_pipeline["passed"]:
        result["classification"] = "oracle_proposal_selector_pipeline_invalid"
        result["selected_kind"] = None
    scientific = {
        **result,
        "campaign_pipeline": campaign_pipeline,
        "cache_identity": {
            "train": train_cache["payload_blake3"],
            "validation": validation_cache["payload_blake3"],
        },
        "source_identity": {
            "john1": local_identity["bundle_sha256"],
            "john4": remote_identity["bundle_sha256"],
        },
        "reports": reports,
        "test_split_opened": False,
        "gameplay_opened": False,
        "new_teacher_compute_used": False,
        "external_compute_used": False,
    }
    combined = {
        "schema_version": 1,
        "experiment_id": EXPERIMENT_ID,
        "scientific": scientific,
        "scientific_blake3": _scientific_blake3(scientific),
    }
    _write_json_atomic(artifact_root / "reports" / "combined.json", combined)
    rows = "\n".join(
        (
            f"| `{kind}` | {report['train']['target_positive_recall']:.2%} | "
            f"{report['train']['target_set_exact_fraction']:.2%} | "
            f"{report['validation']['target_positive_recall']:.2%} | "
            f"{report['validation']['top64_r4800_winner_recall']:.2%} | "
            f"{result['gates'][kind]['feasibility_passed']} |"
        )
        for kind, report in reports.items()
    )
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(
        f"""# Oracle-Proposal Complete-Action Selector V1 Result

Date: 2026-06-16

Classification: **`{result["classification"]}`**

Selected architecture: `{result["selected_kind"]}`

Campaign pipeline passed: `{campaign_pipeline["passed"]}`

| Arm | Train recall | Train exact | Validation recall | Validation winner | Passed |
|---|---:|---:|---:|---:|---:|
{rows}

This diagnostic used only the open oracle-factor proposal. It did not alter
ADR 0120, promote a selector, or open sealed gameplay.
"""
    )
    return combined


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache_parser = subparsers.add_parser("filter-cache")
    cache_parser.add_argument("--factor-cache", type=Path, required=True)
    cache_parser.add_argument("--hierarchy-cache", type=Path, required=True)
    cache_parser.add_argument("--output", type=Path, required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--kind", choices=PROBE_KINDS, required=True)
    train_parser.add_argument("--train-cache", type=Path, required=True)
    train_parser.add_argument("--validation-cache", type=Path, required=True)
    train_parser.add_argument("--output", type=Path, required=True)

    combine_parser = subparsers.add_parser("combine")
    combine_parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("artifacts/experiments") / EXPERIMENT_ID,
    )
    combine_parser.add_argument(
        "--markdown",
        type=Path,
        default=Path("docs/v2/reports") / f"{EXPERIMENT_ID}-result.md",
    )
    args = parser.parse_args()

    if args.command == "filter-cache":
        report = filter_factor_cache(
            factor_cache_root=args.factor_cache,
            hierarchy_cache_root=args.hierarchy_cache,
            output_root=args.output,
        )
    elif args.command == "train":
        report = train_probe(
            kind=args.kind,
            train_cache_root=args.train_cache,
            validation_cache_root=args.validation_cache,
            output_root=args.output,
        )
    else:
        report = combine(
            artifact_root=args.artifact_root,
            markdown_path=args.markdown,
        )
    print(json.dumps(report, default=_json_default, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
