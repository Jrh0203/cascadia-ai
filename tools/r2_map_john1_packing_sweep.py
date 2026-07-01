#!/usr/bin/env python3
"""Measure and freeze production R2-MAP packing on John1 without local files."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

sys.dont_write_bytecode = True
REPOSITORY = Path(__file__).resolve().parents[1]
PYTHON_ROOT = REPOSITORY / "python"
if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

import mlx.core as mx  # noqa: E402
from cascadia_mlx.graded_oracle_dataset import (  # noqa: E402
    GRADED_ORACLE_MAX_WILDLIFE_WIPES,
)
from cascadia_mlx.r2_map_contracts import local_campaign_host_id  # noqa: E402
from cascadia_mlx.r2_map_dataset import (  # noqa: E402
    R2MapCompactDatasetAdapter,
    R2MapStreamReader,
    _training_dataset_contract,
    compact_packing_plan,
    validate_compact_index_value,
)
from cascadia_mlx.r2_map_local_write_guard import (  # noqa: E402
    require_no_local_write_sandbox,
)
from cascadia_mlx.r2_map_model import (  # noqa: E402
    R2MapBatch,
    R2MapPublicState,
)
from cascadia_mlx.r2_map_packing_sweep import (  # noqa: E402
    MAXIMUM_CANDIDATES_PER_BATCH,
    MAXIMUM_WIDTH_CANDIDATES,
    MINIMUM_TIMED_STEPS,
    PRODUCTION_MEASUREMENT_PROTOCOL,
    QUALIFYING_CAPS,
    QUALIFYING_EPOCHS,
    QUALIFYING_GAMES,
    QUALIFYING_SWEEP_SCHEMA,
    REPRESENTATIVE_MEASUREMENT_PROTOCOL,
    SELECTOR_ID,
    validate_qualifying_packing_report,
)
from cascadia_mlx.r2_map_remote_identity import (  # noqa: E402
    load_verified_bootstrap_phase_barrier,
    load_verified_remote_json,
    require_transaction_object,
    transaction_object_descriptor,
    validate_source_identity,
    validate_transaction_commit,
    validate_transaction_manifest,
)
from cascadia_mlx.r2_map_remote_storage import (  # noqa: E402
    RemoteStorageClient,
    SshTransport,
    canonical_json,
)
from cascadia_mlx.r2_map_remote_training import (  # noqa: E402
    John2RemoteCheckpointStore,
    John2RemoteWindowLoader,
)
from cascadia_mlx.r2_map_serve import (  # noqa: E402
    REFERENCE_MAX_CANDIDATES_PER_GROUP,
)
from cascadia_mlx.r2_map_tensor_contract import (  # noqa: E402
    BOARD_SLOTS,
    BOARD_TOKEN_CAPACITY,
    GLOBAL_FEATURES,
    MARKET_FEATURES,
    PLAYER_FEATURES,
    TOKEN_FEATURES,
)
from cascadia_mlx.r2_map_train import (  # noqa: E402
    R2MapAdapterStep,
    R2MapSupervisedBatch,
    R2MapTrainer,
    R2MapTrainerConfig,
)
from cascadia_mlx.r2_map_training_resources import (  # noqa: E402
    R2MapTrainingResourceMonitor,
    validate_training_resource_receipt,
)

WINDOW_EVIDENCE_SCHEMA = "cascadia.r2-map.window-read-evidence.v1"
SWEEP_ADAPTER_PROTOCOL = REPRESENTATIVE_MEASUREMENT_PROTOCOL


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--source-transaction-manifest-relative", required=True)
    result.add_argument("--source-transaction-commit-receipt-relative", required=True)
    result.add_argument("--dataset-transaction-manifest-relative", required=True)
    result.add_argument("--run-id", required=True)
    result.add_argument("--maximum-window-bytes", type=int, default=1 << 30)
    result.add_argument("--warmup-steps", type=int, default=1)
    result.add_argument("--timed-steps", type=int, default=MINIMUM_TIMED_STEPS)
    result.add_argument("--seed", type=int, default=20260618)
    return result


def _arguments() -> argparse.Namespace:
    arguments = parser().parse_args()
    if local_campaign_host_id() != "john1":
        raise SystemExit("R2-MAP packing measurement is authorized only on John1")
    if (
        not 1 <= arguments.maximum_window_bytes <= 1 << 30
        or arguments.warmup_steps < 1
        or arguments.timed_steps < MINIMUM_TIMED_STEPS
        or arguments.seed < 0
    ):
        raise SystemExit("qualifying packing-sweep limits are invalid")
    return arguments


@dataclass
class _FactoryBatchAdapter:
    batch_factory: Callable[[], R2MapSupervisedBatch]
    dataset_blake3: str
    dataset_contract: dict[str, Any]
    group_batch_size: int
    maximum_candidates_per_batch: int
    protocol_id: str = SWEEP_ADAPTER_PROTOCOL

    def __post_init__(self) -> None:
        self.produced_group_counts: list[int] = []

    def initial_state(self, seed: int) -> tuple[dict[str, Any], dict[str, Any]]:
        return {"batch_index": 0}, {"seed": seed}

    def training_batch(
        self, cursor: dict[str, Any], sampler_state: dict[str, Any]
    ) -> R2MapAdapterStep:
        index = int(cursor["batch_index"])
        batch = self.batch_factory()
        groups, _ = batch.validate()
        self.produced_group_counts.append(groups)
        return R2MapAdapterStep(
            batch=batch,
            next_cursor={"batch_index": index + 1},
            next_sampler_state=dict(sampler_state),
        )

    def validation_batches(self) -> tuple[R2MapSupervisedBatch, ...]:
        return (self.batch_factory(),)

    def fixed_prediction_batch(self, _panel_id: str):
        return self.batch_factory().inputs


class _RecordingAdapter:
    def __init__(self, delegate: R2MapCompactDatasetAdapter):
        self.delegate = delegate
        self.dataset_blake3 = delegate.dataset_blake3
        self.dataset_contract = delegate.dataset_contract
        self.group_batch_size = delegate.group_batch_size
        self.maximum_candidates_per_batch = delegate.maximum_candidates_per_batch
        self.protocol_id = delegate.protocol_id
        self.produced_group_counts: list[int] = []

    def initial_state(self, seed: int):
        return self.delegate.initial_state(seed)

    def training_batch(self, cursor: dict[str, Any], sampler_state: dict[str, Any]):
        step = self.delegate.training_batch(cursor, sampler_state)
        groups, _ = step.batch.validate()
        self.produced_group_counts.append(groups)
        return step

    def validation_batches(self):
        return self.delegate.validation_batches()

    def fixed_prediction_batch(self, panel_id: str):
        return self.delegate.fixed_prediction_batch(panel_id)


class _TimedWindowLoader:
    def __init__(self, delegate: John2RemoteWindowLoader):
        self.delegate = delegate
        self.durations_ns: list[int] = []

    def __call__(self, source: str, mode: str, epoch: int, sampler_seed: int):
        started = time.monotonic_ns()
        result = self.delegate(source, mode, epoch, sampler_seed)
        self.durations_ns.append(time.monotonic_ns() - started)
        return result


def _width_catalog(
    index: dict[str, Any],
    loader: John2RemoteWindowLoader,
    *,
    seed: int,
) -> dict[str, dict[str, tuple[int, ...]]]:
    train_sources = {
        game["source_file_name"] for game in index["games"] if game["split"] == "train"
    }
    catalog: dict[str, dict[str, tuple[int, ...]]] = {}
    for source in sorted(train_sources):
        manifest, stream = loader(source, "train", 0, seed)
        grouped: dict[str, list[int]] = {}
        with R2MapStreamReader(manifest, stream) as reader:
            for ref in reader.refs:
                grouped.setdefault(ref.game_id.hex(), []).append(ref.candidate_count)
        catalog[source] = {game_id: tuple(widths) for game_id, widths in grouped.items()}
        del stream
    return catalog


def _quantile_index(indices: list[int], widths: list[int], quantile: float) -> int:
    if not indices or not 0.0 <= quantile <= 1.0:
        raise ValueError("imitation quantile request is invalid")
    ordered = sorted(indices, key=lambda index: (widths[index], index))
    rank = math.ceil(quantile * len(ordered)) - 1
    return ordered[max(rank, 0)]


def _representative_indices(
    widths: list[int],
    *,
    group_batch_size: int,
    maximum_candidates_per_batch: int,
    imitation_quantile: float | None,
    imitation_width: int | None = None,
) -> list[int]:
    selected_only = [index for index, width in enumerate(widths) if width == 1]
    if imitation_quantile is None and imitation_width is None:
        if maximum_candidates_per_batch < group_batch_size:
            raise RuntimeError("selected-only representative exceeds the candidate budget")
        if len(selected_only) < group_batch_size:
            raise RuntimeError("selected-only representative cannot fill the exact group cap")
        return selected_only[:group_batch_size]
    imitation = [index for index, width in enumerate(widths) if width > 1]
    if imitation_quantile is not None and imitation_width is not None:
        raise ValueError("imitation representative must select one width rule")
    if imitation_width is None:
        assert imitation_quantile is not None
        wide_index = _quantile_index(imitation, widths, imitation_quantile)
    else:
        wide_index = next(
            (index for index in imitation if widths[index] == imitation_width),
            -1,
        )
        if wide_index < 0:
            raise RuntimeError("registered imitation width is absent from its source")
    wide = widths[wide_index]
    count = min(group_batch_size, maximum_candidates_per_batch // wide)
    if count == 0:
        raise RuntimeError("one imitation row exceeds the padded candidate budget")
    if len(selected_only) < count - 1:
        raise RuntimeError("imitation representative cannot fill the exact packed group count")
    return [wide_index, *selected_only[: count - 1]]


def _supports_representative_widths(
    widths: list[int], *, p50_width: int, maximum_width: int
) -> bool:
    try:
        for width in (None, p50_width, maximum_width):
            _representative_indices(
                widths,
                group_batch_size=max(QUALIFYING_CAPS),
                maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
                imitation_quantile=None,
                imitation_width=width,
            )
    except RuntimeError:
        return False
    return True


def _measure_adapter_factory(
    adapter_factory: Callable[[], Any],
    *,
    label: str,
    measurement_protocol: str,
    source_blake3: str,
    dataset_contract: dict[str, Any],
    seed: int,
    warmup_steps: int,
    timed_steps: int,
    run_id: str,
    group_batch_size: int,
    maximum_candidates_per_batch: int,
    remote_evidence_count: Callable[[], int] = lambda: 0,
    remote_window_durations: Callable[[], list[int]] = lambda: [],
    remote_window_required: bool = False,
    candidate_widths: list[int] | None = None,
    frame_indices: list[int] | None = None,
) -> dict[str, Any]:
    mx.clear_cache()
    mx.reset_peak_memory()
    forbidden = Path(f"/private/var/empty/r2-map-sweep-{run_id}-{label}")
    if forbidden.exists():
        raise RuntimeError("forbidden local packing-sweep path exists")

    def trainer_for(adapter: Any, branch: str) -> R2MapTrainer:
        return R2MapTrainer(
            R2MapTrainerConfig(
                run_dir=forbidden,
                run_id=run_id,
                branch_id=branch,
                source_blake3=source_blake3,
                dataset_blake3=dataset_contract["dataset_blake3"],
                adapter_protocol_id=adapter.protocol_id,
                group_batch_size=adapter.group_batch_size,
                maximum_candidates_per_batch=adapter.maximum_candidates_per_batch,
                warmup_steps=10,
                schedule_steps=1_000_000,
                loss_event_interval_steps=25,
                seed=seed,
            ),
            adapter,
            in_memory=True,
        )

    warm_delegate = adapter_factory()
    warm_adapter = (
        _RecordingAdapter(warm_delegate)
        if isinstance(warm_delegate, R2MapCompactDatasetAdapter)
        else warm_delegate
    )
    warm_trainer = trainer_for(warm_adapter, f"{label}-warmup")
    for _ in range(warmup_steps):
        warm_trainer.step()
        mx.synchronize()
    if isinstance(warm_delegate, R2MapCompactDatasetAdapter):
        warm_delegate.close()
    del warm_trainer, warm_adapter, warm_delegate

    timed_delegate = adapter_factory()
    adapter = (
        _RecordingAdapter(timed_delegate)
        if isinstance(timed_delegate, R2MapCompactDatasetAdapter)
        else timed_delegate
    )
    trainer = trainer_for(adapter, f"{label}-timed")
    monitor = R2MapTrainingResourceMonitor.start()
    monitor.sample()
    start_counters = dict(trainer.training_counters)
    durations_ns: list[int] = []
    observed_groups: list[int] = []
    evidence_before = remote_evidence_count()
    duration_count_before = len(remote_window_durations())
    window_duration_per_step: list[int] = []
    for _ in range(timed_steps):
        before_groups = trainer.training_counters["draft_groups"]
        before_window_durations = len(remote_window_durations())
        started = time.monotonic_ns()
        trainer.step()
        mx.synchronize()
        durations_ns.append(time.monotonic_ns() - started)
        window_duration_per_step.append(sum(remote_window_durations()[before_window_durations:]))
        observed_groups.append(trainer.training_counters["draft_groups"] - before_groups)
        monitor.sample()
    evidence_after = remote_evidence_count()
    window_durations = remote_window_durations()[duration_count_before:]
    elapsed_ns = sum(durations_ns)
    deltas = {
        name: trainer.training_counters[name] - start_counters[name]
        for name in trainer.training_counters
    }
    receipt = validate_training_resource_receipt(monitor.receipt())
    result = {
        "label": label,
        "measurement_protocol": measurement_protocol,
        "warmup_steps": warmup_steps,
        "warmup_synchronized": True,
        "timed_steps": timed_steps,
        "elapsed_ns": elapsed_ns,
        "step_durations_ns": durations_ns,
        "p50_step_duration_ns": statistics.median(durations_ns),
        "steps_per_second": timed_steps * 1_000_000_000 / elapsed_ns,
        "draft_groups_per_second": deltas["draft_groups"] * 1_000_000_000 / elapsed_ns,
        "draft_candidates_per_second": (deltas["draft_candidates"] * 1_000_000_000 / elapsed_ns),
        "training_counters": deltas,
        "resource_receipt": receipt,
        "mlx_memory": {
            "active_bytes": int(mx.get_active_memory()),
            "cache_bytes": int(mx.get_cache_memory()),
            "peak_active_bytes": int(mx.get_peak_memory()),
        },
        "expected_group_count_per_step": list(adapter.produced_group_counts),
        "observed_group_count_per_step": observed_groups,
        "decode_and_padding_inside_timed_step": True,
        "mlx_allocation_inside_timed_step": True,
        "remote_window_acquisition_inside_timed_interval": remote_window_required,
        "remote_windows_acquired": evidence_after - evidence_before,
        "remote_window_durations_ns": window_durations,
        "remote_window_duration_ns_per_step": window_duration_per_step,
        "candidate_widths": list(candidate_widths or ()),
        "frame_indices": list(frame_indices or ()),
    }
    if result["expected_group_count_per_step"] != observed_groups:
        raise RuntimeError("packing measurement group counts differ from decoded batches")
    if remote_window_required and result["remote_windows_acquired"] < 1:
        raise RuntimeError("production-path timing omitted remote window acquisition")
    if result["remote_windows_acquired"] != len(window_durations):
        raise RuntimeError("production-path timing/window evidence counts differ")
    if not remote_window_required and result["remote_windows_acquired"] != 0:
        raise RuntimeError("representative measurement unexpectedly acquired a remote window")
    if forbidden.exists():
        raise RuntimeError("packing sweep created a forbidden local run tree")
    if isinstance(timed_delegate, R2MapCompactDatasetAdapter):
        timed_delegate.close()
    del trainer
    mx.clear_cache()
    return result


def _synthetic_public_state(*, groups: int, candidates: int | None = None) -> R2MapPublicState:
    leading = (groups,) if candidates is None else (groups, candidates)
    token_shape = (*leading, BOARD_SLOTS, BOARD_TOKEN_CAPACITY)
    return R2MapPublicState(
        token_features=mx.zeros((*token_shape, TOKEN_FEATURES), dtype=mx.float32),
        token_types=mx.zeros(token_shape, dtype=mx.int32),
        token_mask=mx.zeros(token_shape, dtype=mx.bool_),
        market_features=mx.zeros((*leading, 4, MARKET_FEATURES), dtype=mx.float32),
        market_mask=mx.ones((*leading, 4), dtype=mx.bool_),
        player_features=mx.zeros((*leading, BOARD_SLOTS, PLAYER_FEATURES), dtype=mx.float32),
        player_mask=mx.ones((*leading, BOARD_SLOTS), dtype=mx.bool_),
        global_features=mx.zeros((*leading, GLOBAL_FEATURES), dtype=mx.float32),
    )


def _synthetic_maximum_width_batch(candidate_count: int, panel_sha256: str) -> R2MapSupervisedBatch:
    actions = np.zeros((1, candidate_count, 140), dtype=np.float32)
    actions[0, :, 0] = np.linspace(0.0, 1.0, candidate_count, dtype=np.float32)
    score_mask = np.zeros((1, candidate_count), dtype=np.bool_)
    score_mask[0, candidate_count - 1] = True
    wipe_shape = (1, 3, GRADED_ORACLE_MAX_WILDLIFE_WIPES)
    result = R2MapSupervisedBatch(
        inputs=R2MapBatch(
            parent=_synthetic_public_state(groups=1),
            candidates=_synthetic_public_state(groups=1, candidates=candidate_count),
            candidate_mask=mx.ones((1, candidate_count), dtype=mx.bool_),
            action_features=mx.array(actions),
            exact_afterstate_scores=mx.zeros((1, candidate_count), dtype=mx.float32),
        ),
        score_to_go_targets=mx.zeros((1, candidate_count), dtype=mx.float32),
        score_component_targets=mx.zeros((1, candidate_count, 11), dtype=mx.float32),
        score_target_mask=mx.array(score_mask),
        selected_action_index=mx.array([candidate_count - 1], dtype=mx.int32),
        bootstrap_policy_mask=mx.array([True]),
        opponent_tile_slot_targets=mx.zeros((1, 3), dtype=mx.int32),
        opponent_wildlife_slot_targets=mx.zeros((1, 3), dtype=mx.int32),
        opponent_draft_kind_targets=mx.zeros((1, 3), dtype=mx.int32),
        opponent_drafted_wildlife_targets=mx.zeros((1, 3), dtype=mx.int32),
        opponent_replace_three_targets=mx.zeros((1, 3), dtype=mx.int32),
        opponent_paid_wipe_count_targets=mx.zeros((1, 3), dtype=mx.int32),
        opponent_paid_wipe_mask_targets=mx.zeros(wipe_shape, dtype=mx.int32),
        opponent_paid_wipe_mask_valid=mx.zeros(wipe_shape, dtype=mx.bool_),
        opponent_valid_mask=mx.zeros((1, 3), dtype=mx.bool_),
        market_disposition_targets=mx.zeros((1, 4), dtype=mx.int32),
        market_pair_survival_targets=mx.zeros((1, 4), dtype=mx.int32),
        market_final_slot_targets=mx.zeros((1, 4), dtype=mx.int32),
        market_disposition_mask=mx.zeros((1, 4), dtype=mx.bool_),
        market_pair_survival_mask=mx.zeros((1, 4), dtype=mx.bool_),
        market_final_slot_mask=mx.zeros((1, 4), dtype=mx.bool_),
        batch_identity=(f"synthetic-maximum-width:{candidate_count}:{panel_sha256}"),
    )
    result.validate()
    return result


def _wall_projections(
    plans: list[dict[str, Any]],
    production_measurements: list[dict[str, Any]],
    representative_measurements: list[dict[str, Any]],
    *,
    remote_source_durations_ns: list[int],
) -> list[dict[str, Any]]:
    """Project exact compute steps plus one measured fetch of every source per epoch.

    The production-path measurement is the central estimator because it exercises
    the real compact adapter.  Its enclosing window-fetch time is subtracted so
    remote I/O is not counted twice.  The conservative compute estimator also
    covers every frozen representative corpus shape for the cap, while the remote
    estimator is based on the complete source census rather than whichever shard
    happened to be first in the production adapter.
    """
    if not remote_source_durations_ns or any(
        not isinstance(duration, int) or isinstance(duration, bool) or duration <= 0
        for duration in remote_source_durations_ns
    ):
        raise ValueError("production wall projection requires remote source windows")
    remote_windows_per_epoch = len(remote_source_durations_ns)
    production_by_label = {
        measurement["label"]: measurement for measurement in production_measurements
    }
    representative_by_label = {
        measurement["label"]: measurement for measurement in representative_measurements
    }
    remote_durations = [value / 1_000_000_000 for value in remote_source_durations_ns]
    remote_rates = {
        "optimistic": min(remote_durations),
        "central": statistics.median(remote_durations),
        "conservative": max(remote_durations),
    }
    projections = []
    for plan in plans:
        group_size = plan["group_batch_size"]
        measured = production_by_label[f"g{group_size}-production"]
        compute_durations = [
            (duration - remote_duration) / 1_000_000_000
            for duration, remote_duration in zip(
                measured["step_durations_ns"],
                measured["remote_window_duration_ns_per_step"],
                strict=True,
            )
        ]
        if any(duration <= 0 for duration in compute_durations):
            raise RuntimeError("remote window duration exceeds its enclosing optimizer step")
        representative_durations = [
            duration / 1_000_000_000
            for suffix in ("selected", "imitation-p50", "imitation-max")
            for duration in representative_by_label[f"g{group_size}-{suffix}"][
                "step_durations_ns"
            ]
        ]
        compute_rates = {
            "optimistic": min(compute_durations),
            "central": statistics.median(compute_durations),
            "conservative": max(*compute_durations, *representative_durations),
        }
        central_epochs = [
            epoch["steps"] * compute_rates["central"]
            + remote_windows_per_epoch * remote_rates["central"]
            for epoch in plan["epoch_plans"]
        ]
        optimistic_epochs = [
            epoch["steps"] * compute_rates["optimistic"]
            + remote_windows_per_epoch * remote_rates["optimistic"]
            for epoch in plan["epoch_plans"]
        ]
        conservative_epochs = [
            epoch["steps"] * compute_rates["conservative"]
            + remote_windows_per_epoch * remote_rates["conservative"]
            for epoch in plan["epoch_plans"]
        ]
        projections.append(
            {
                "group_batch_size": group_size,
                "method": "exact-plan-compute-plus-all-source-remote-window-rate-v4",
                "steps_per_epoch": [epoch["steps"] for epoch in plan["epoch_plans"]],
                "remote_windows_per_epoch": remote_windows_per_epoch,
                "central_seconds_per_epoch": central_epochs,
                "central_12_epoch_wall_seconds": sum(central_epochs),
                "optimistic_12_epoch_wall_seconds": sum(optimistic_epochs),
                "conservative_12_epoch_wall_seconds": sum(conservative_epochs),
                "compute_seconds_per_step": compute_rates,
                "remote_seconds_per_window": remote_rates,
                "includes_remote_window_acquisition": True,
            }
        )
    return projections


def _select_group_batch_size(
    plans: list[dict[str, Any]],
    measurements: list[dict[str, Any]],
    wall_projections: list[dict[str, Any]],
    *,
    maximum_candidates_per_batch: int,
    required_measurements: tuple[dict[str, Any], ...] = (),
) -> dict[str, Any]:
    by_label = {measurement["label"]: measurement for measurement in measurements}
    by_group = {projection["group_batch_size"]: projection for projection in wall_projections}
    candidates = []
    for plan in plans:
        group_size = plan["group_batch_size"]
        shape_measurements = [
            by_label[f"g{group_size}-{suffix}"]
            for suffix in (
                "selected",
                "imitation-p50",
                "imitation-max",
                "production",
            )
        ] + list(required_measurements)
        resource_pass = all(
            measured["resource_receipt"]["maximum_rss_bytes"] <= 4 * (1 << 30)
            and measured["resource_receipt"]["process_swaps"] == 0
            and measured["resource_receipt"]["system_swap_delta_bytes"] == 0
            and measured["mlx_memory"]["cache_bytes"] <= 1 << 30
            for measured in shape_measurements
        )
        candidate_budget_pass = plan[
            "maximum_candidate_width"
        ] <= maximum_candidates_per_batch and all(
            measured["training_counters"]["padded_draft_candidates"]
            <= maximum_candidates_per_batch * measured["timed_steps"]
            for measured in shape_measurements
        )
        projection = by_group[group_size]
        candidates.append(
            {
                "group_batch_size": group_size,
                "resource_pass": resource_pass,
                "candidate_budget_pass": candidate_budget_pass,
                "conservative_12_epoch_wall_seconds": projection[
                    "conservative_12_epoch_wall_seconds"
                ],
            }
        )
    eligible = [
        candidate
        for candidate in candidates
        if candidate["resource_pass"] and candidate["candidate_budget_pass"]
    ]
    if not eligible:
        raise RuntimeError("no packing-sweep group cap passes every frozen resource gate")
    selected = min(
        eligible,
        key=lambda candidate: (
            candidate["conservative_12_epoch_wall_seconds"],
            candidate["group_batch_size"],
        ),
    )
    return {
        "selector": SELECTOR_ID,
        "selected_group_batch_size": selected["group_batch_size"],
        "selected_schedule_steps": next(
            plan["totals"]["steps"]
            for plan in plans
            if plan["group_batch_size"] == selected["group_batch_size"]
        ),
        "selected_epochs": QUALIFYING_EPOCHS,
        "selected_conservative_12_epoch_wall_seconds": selected[
            "conservative_12_epoch_wall_seconds"
        ],
        "candidates": candidates,
        "rationale": (
            "minimum conservative 12-epoch optimizer-wall projection among caps "
            "passing RSS, MLX-cache, zero-swap, and padded-candidate gates; ties "
            "select the lower cap"
        ),
    }


def _load_receipt_bound_identities(
    client: RemoteStorageClient, arguments: argparse.Namespace
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    source_transaction = load_verified_remote_json(
        client, arguments.source_transaction_manifest_relative
    )
    source_transaction_value = validate_transaction_manifest(source_transaction)
    source_target = source_transaction_value["target_relative"]
    source_manifest = load_verified_remote_json(client, f"{source_target}/source-manifest.json")
    reference_manifest = load_verified_remote_json(
        client,
        f"{source_target}/docs/v2/reports/r2-map-w0-reference-panel-manifest-v1.1.json",
    )
    source_archive_verification = load_verified_remote_json(
        client,
        f"{source_target}/source-archive-verification.json",
    )
    source_commit = load_verified_remote_json(
        client,
        arguments.source_transaction_commit_receipt_relative,
        maximum_bytes=2 << 20,
    )
    source_identity = validate_source_identity(
        source_manifest=source_manifest,
        reference_manifest=reference_manifest,
        source_archive_verification=source_archive_verification,
        transaction_manifest=source_transaction,
        transaction_commit_receipt=source_commit,
    )
    exporter_relative = f"{source_target}/tools/r2_map_compact_dataset.py"
    exporter_descriptor = transaction_object_descriptor(
        source_transaction_value, exporter_relative
    )
    if exporter_descriptor.get("mode") != "0500":
        raise RuntimeError("compact exporter is not executable in the source transaction")

    dataset_transaction_hint = load_verified_remote_json(
        client, arguments.dataset_transaction_manifest_relative
    )
    (
        bootstrap_phase_barrier,
        index_document,
        dataset_transaction_document,
        dataset_commit,
    ) = load_verified_bootstrap_phase_barrier(
        client,
        dataset_transaction_hint=dataset_transaction_hint,
    )
    index = validate_compact_index_value(index_document.value)
    dataset_transaction = validate_transaction_manifest(dataset_transaction_document)
    if dataset_transaction["target_relative"].split("/", 1)[0] != "datasets":
        raise RuntimeError("bootstrap transaction is outside canonical datasets/")
    if dataset_transaction_document.evidence.relative != (
        arguments.dataset_transaction_manifest_relative
    ):
        raise RuntimeError("caller paths differ from the derived bootstrap phase barrier")
    require_transaction_object(dataset_transaction, index_document)
    validate_transaction_commit(dataset_commit, dataset_transaction)
    manifest = index["dataset_manifest"]
    for source in manifest["sources"]:
        descriptor = transaction_object_descriptor(
            dataset_transaction,
            f"{bootstrap_phase_barrier['shard_root_relative']}/{source['file_name']}",
        )
        if descriptor.get("size") != source["bytes"]:
            raise RuntimeError("dataset transaction shard size differs from compact index")
    dataset_identity = {
        "dataset_blake3": manifest["dataset_blake3"],
        "game_count": manifest["game_count"],
        "collection_kind": manifest["round"]["collection_kind"],
        "shard_root_relative": bootstrap_phase_barrier["shard_root_relative"],
        "exporter_relative": exporter_relative,
        "compact_index": index_document.to_dict(),
        "transaction_manifest": dataset_transaction_document.to_dict(),
        "transaction_manifest_sha256": dataset_transaction["manifest_sha256"],
        "transaction_commit_receipt": dataset_commit.to_dict(),
        "transaction_commit_receipt_sha256": dataset_commit.payload_sha256,
        "bootstrap_phase_barrier": bootstrap_phase_barrier,
    }
    return index, source_identity, dataset_identity


def main() -> int:
    arguments = _arguments()
    local_write_guard = require_no_local_write_sandbox(Path(__file__))
    sweep_monitor = R2MapTrainingResourceMonitor.start()
    sweep_monitor.sample()
    transport = SshTransport(compression=False)
    ssh_configuration = transport.verify_local_configuration()
    client = RemoteStorageClient(transport)
    preflight = client.preflight()
    index, source_identity, dataset_identity = _load_receipt_bound_identities(client, arguments)
    manifest = index["dataset_manifest"]
    if (
        manifest["game_count"] != QUALIFYING_GAMES
        or manifest["round"]["collection_kind"] != "bootstrap"
    ):
        raise SystemExit("qualifying packing sweep requires the exact 100,000-game bootstrap index")
    if REFERENCE_MAX_CANDIDATES_PER_GROUP != MAXIMUM_WIDTH_CANDIDATES:
        raise RuntimeError("live serving maximum width differs from the W0 contract")
    store = John2RemoteCheckpointStore(client, run_id=arguments.run_id)
    window_evidence: list[dict[str, Any]] = []
    window_evidence_bytes = 0

    def record_window_evidence(evidence: Any) -> None:
        nonlocal window_evidence_bytes
        value: dict[str, Any] = {
            "schema_version": 1,
            "schema_id": WINDOW_EVIDENCE_SCHEMA,
            **evidence.to_dict(),
        }
        value["evidence_sha256"] = hashlib.sha256(canonical_json(value)).hexdigest()
        window_evidence_bytes += len(canonical_json(value))
        if window_evidence_bytes > 48 << 20:
            raise RuntimeError("packing window evidence exceeds its in-memory bound")
        window_evidence.append(value)

    loader = John2RemoteWindowLoader(
        client,
        exporter_relative=dataset_identity["exporter_relative"],
        shard_root_relative=dataset_identity["shard_root_relative"],
        maximum_window_bytes=arguments.maximum_window_bytes,
        evidence_sink=record_window_evidence,
    )
    timed_loader = _TimedWindowLoader(loader)
    catalog_sources = sorted(
        {
            game["source_file_name"]
            for game in index["games"]
            if game["split"] == "train"
        }
    )
    catalog = _width_catalog(index, timed_loader, seed=arguments.seed)
    if (
        len(timed_loader.durations_ns) != len(catalog_sources)
        or len(window_evidence) != len(catalog_sources)
    ):
        raise RuntimeError("all-source width census timing is incomplete")
    source_window_timings = [
        {
            "source": source,
            "duration_ns": duration,
            "window_run_id": evidence["run_id"],
            "window_evidence_sha256": evidence["evidence_sha256"],
        }
        for source, duration, evidence in zip(
            catalog_sources,
            timed_loader.durations_ns,
            window_evidence,
            strict=True,
        )
    ]
    sweep_monitor.sample()
    plans = [
        compact_packing_plan(
            index,
            catalog,
            group_batch_size=group_size,
            maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
            seed=arguments.seed,
            epochs=QUALIFYING_EPOCHS,
        )
        for group_size in QUALIFYING_CAPS
    ]
    all_widths = [
        width for source in catalog.values() for game in source.values() for width in game
    ]
    imitation_widths = sorted(width for width in all_widths if width > 1)
    if not imitation_widths:
        raise RuntimeError("packing sweep requires real imitation rows")
    p50_width = imitation_widths[(len(imitation_widths) - 1) // 2]
    maximum_real_width = imitation_widths[-1]
    eligible_sources = []
    for source, games in catalog.items():
        source_widths = [width for game in games.values() for width in game]
        if _supports_representative_widths(
            source_widths,
            p50_width=p50_width,
            maximum_width=maximum_real_width,
        ):
            eligible_sources.append(source)
    if not eligible_sources:
        raise RuntimeError(
            "selected, global-p50, and global-maximum rows must share one source window"
        )
    representative_source = min(eligible_sources)
    manifest, stream = loader(representative_source, "train", 0, arguments.seed)
    measurements = []
    with R2MapStreamReader(manifest, stream) as reader:
        widths = [ref.candidate_count for ref in reader.refs]
        dataset_contract = _training_dataset_contract(index["dataset_manifest"])
        for group_size in QUALIFYING_CAPS:
            for suffix, target_width in (
                ("selected", None),
                ("imitation-p50", p50_width),
                ("imitation-max", maximum_real_width),
            ):
                indices = _representative_indices(
                    widths,
                    group_batch_size=group_size,
                    maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
                    imitation_quantile=None,
                    imitation_width=target_width,
                )
                label = f"g{group_size}-{suffix}"
                measured = _measure_adapter_factory(
                    lambda indices=tuple(indices), group_size=group_size: _FactoryBatchAdapter(
                        batch_factory=lambda: reader.batch(list(indices)),
                        dataset_blake3=dataset_contract["dataset_blake3"],
                        dataset_contract=dataset_contract,
                        group_batch_size=group_size,
                        maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
                    ),
                    label=label,
                    measurement_protocol=REPRESENTATIVE_MEASUREMENT_PROTOCOL,
                    source_blake3=source_identity["source_blake3"],
                    dataset_contract=dataset_contract,
                    seed=arguments.seed,
                    warmup_steps=arguments.warmup_steps,
                    timed_steps=arguments.timed_steps,
                    run_id=arguments.run_id,
                    group_batch_size=group_size,
                    maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
                    candidate_widths=[widths[index] for index in indices],
                    frame_indices=indices,
                )
                measurements.append(measured)
                sweep_monitor.sample()
    del stream

    production_measurements = []
    for group_size in QUALIFYING_CAPS:
        production_measurements.append(
            _measure_adapter_factory(
                lambda group_size=group_size: R2MapCompactDatasetAdapter(
                    index=index,
                    window_loader=timed_loader,
                    group_batch_size=group_size,
                    maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
                    maximum_window_bytes=arguments.maximum_window_bytes,
                    maximum_prefetch_windows=0,
                    fixed_panel_games=1,
                ),
                label=f"g{group_size}-production",
                measurement_protocol=PRODUCTION_MEASUREMENT_PROTOCOL,
                source_blake3=source_identity["source_blake3"],
                dataset_contract=_training_dataset_contract(manifest),
                seed=arguments.seed,
                warmup_steps=arguments.warmup_steps,
                timed_steps=arguments.timed_steps,
                run_id=arguments.run_id,
                group_batch_size=group_size,
                maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
                remote_evidence_count=lambda: len(window_evidence),
                remote_window_durations=lambda: timed_loader.durations_ns,
                remote_window_required=True,
            )
        )
        sweep_monitor.sample()

    panel_sha256 = source_identity["maximum_width_panel_sha256"]
    maximum_measurement = _measure_adapter_factory(
        lambda: _FactoryBatchAdapter(
            batch_factory=lambda: _synthetic_maximum_width_batch(
                MAXIMUM_WIDTH_CANDIDATES, panel_sha256
            ),
            dataset_blake3=manifest["dataset_blake3"],
            dataset_contract=_training_dataset_contract(manifest),
            group_batch_size=max(QUALIFYING_CAPS),
            maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
        ),
        label="synthetic-maximum-width",
        measurement_protocol=REPRESENTATIVE_MEASUREMENT_PROTOCOL,
        source_blake3=source_identity["source_blake3"],
        dataset_contract=_training_dataset_contract(manifest),
        seed=arguments.seed,
        warmup_steps=arguments.warmup_steps,
        timed_steps=arguments.timed_steps,
        run_id=arguments.run_id,
        group_batch_size=max(QUALIFYING_CAPS),
        maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
        candidate_widths=[MAXIMUM_WIDTH_CANDIDATES],
    )
    sweep_monitor.sample()

    wall_projections = _wall_projections(
        plans,
        production_measurements,
        measurements,
        remote_source_durations_ns=[item["duration_ns"] for item in source_window_timings],
    )
    selection = _select_group_batch_size(
        plans,
        [*measurements, *production_measurements],
        wall_projections,
        maximum_candidates_per_batch=MAXIMUM_CANDIDATES_PER_BATCH,
        required_measurements=(maximum_measurement,),
    )
    sweep_monitor.sample()
    sweep_resource_receipt = validate_training_resource_receipt(sweep_monitor.receipt())
    report: dict[str, Any] = {
        "schema_version": 3,
        "schema_id": QUALIFYING_SWEEP_SCHEMA,
        "qualification_status": "qualifying-exact-bootstrap",
        "run_id": arguments.run_id,
        "source_identity": source_identity,
        "dataset_identity": dataset_identity,
        "packing_contract": {
            "group_batch_sizes": list(QUALIFYING_CAPS),
            "maximum_candidates_per_batch": MAXIMUM_CANDIDATES_PER_BATCH,
            "maximum_window_bytes": arguments.maximum_window_bytes,
            "games": QUALIFYING_GAMES,
            "epochs": QUALIFYING_EPOCHS,
            "warmup_steps": arguments.warmup_steps,
            "timed_steps": arguments.timed_steps,
            "seed": arguments.seed,
            "production_measurement_protocol": PRODUCTION_MEASUREMENT_PROTOCOL,
            "representative_measurement_protocol": REPRESENTATIVE_MEASUREMENT_PROTOCOL,
            "coverage": [
                "selected-only",
                "imitation-p50",
                "imitation-maximum",
                "registered-maximum-width",
            ],
        },
        "registered_maximum_width": {
            "candidate_count": MAXIMUM_WIDTH_CANDIDATES,
            "panel_sha256": panel_sha256,
            "synthetic_resource_gate_only": True,
            "measurement": maximum_measurement,
        },
        "width_census": {
            "draft_groups": len(all_widths),
            "selected_only_groups": sum(width == 1 for width in all_widths),
            "imitation_groups": len(imitation_widths),
            "imitation_minimum": min(imitation_widths),
            "imitation_median": imitation_widths[(len(imitation_widths) - 1) // 2],
            "imitation_maximum": max(imitation_widths),
        },
        "packing_plans": plans,
        "representative_measurements": measurements,
        "production_path_measurements": production_measurements,
        "source_window_timings": source_window_timings,
        "wall_projections": wall_projections,
        "selection": selection,
        "sweep_resource_receipt": sweep_resource_receipt,
        "window_evidence_publications": window_evidence,
        "ssh_transport": ssh_configuration,
        "storage_preflight_receipt": {
            key: preflight[key] for key in ("storage_receipt_relative", "storage_receipt_sha256")
        },
        "local_write_guard": local_write_guard,
    }
    report["report_sha256"] = hashlib.sha256(canonical_json(report)).hexdigest()
    validate_qualifying_packing_report(report)
    relative = f"reports/w2-w3/{arguments.run_id}/packing-sweep.json"
    publication = store.publish_immutable_json(relative, report)
    print(
        json.dumps(
            {
                "schema_version": 1,
                "schema_id": "cascadia.r2-map.john1-packing-sweep-publication.v1",
                "report_relative": relative,
                "report_sha256": report["report_sha256"],
                "report_object_sha256": publication["sha256"],
                "report_publication_receipt_relative": publication["storage_receipt_relative"],
                "report_publication_receipt_sha256": publication["storage_receipt_sha256"],
                "local_write_attestation_relative": local_write_guard["attestation_relative"],
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
