"""Resumable MLX training for grouped search-teacher action ranking."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from importlib.metadata import version
from pathlib import Path
from typing import Any

import blake3
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from cascadia_mlx.checkpoint import (
    TrainerState,
    load_latest_checkpoint_with_factory,
    prune_checkpoints,
    save_checkpoint,
    set_checkpoint_pointer,
)
from cascadia_mlx.ranking_dataset import RankingDataset
from cascadia_mlx.ranking_model import EntitySetRanker, RankingModelConfig, ranking_loss
from cascadia_mlx.ranking_promote import load_promoted_ranking_model
from cascadia_mlx.run_manifest import source_provenance, validate_resume_manifest


@dataclass(frozen=True)
class GroupedRankingAdapter:
    """Architecture-specific hooks for the shared grouped-ranking trainer."""

    kind: str
    dataset_factory: Callable[[Path], Any]
    model_factory: Callable[[dict[str, object]], Any]
    new_model: Callable[[Any], Any]
    load_promoted: Callable[[Path], Any]
    loss: Callable[[Any, object], mx.array]
    score_batch: Callable[[Any, object], mx.array]
    augment_batch: Callable[[object, int], object] | None = None
    evaluate: Callable[[Any, Any, int], dict[str, Any]] | None = None
    selection_metric: str = "listwise_loss"
    accuracy_metric: str = "top1_accuracy"
    tertiary_metric: str | None = None
    batch_kwargs: dict[str, object] = field(default_factory=dict)
    init_manifest_name: str = "model.json"


def _entity_adapter() -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="entity-set-ranking",
        dataset_factory=RankingDataset,
        model_factory=lambda values: EntitySetRanker(RankingModelConfig.from_dict(values)),
        new_model=EntitySetRanker,
        load_promoted=load_promoted_ranking_model,
        loss=ranking_loss,
        score_batch=lambda model, batch: model(
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
        ),
    )


@dataclass(frozen=True)
class RankingTrainingConfig:
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    init_model_dir: Path | None = None
    epochs: int = 10
    group_batch_size: int = 16
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 20260610
    checkpoint_steps: int = 500
    validation_patience: int = 5
    resume: bool = False
    model: RankingModelConfig = field(default_factory=RankingModelConfig)

    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive")
        if self.group_batch_size <= 0:
            raise ValueError("group_batch_size must be positive")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be positive")
        if self.weight_decay < 0:
            raise ValueError("weight_decay cannot be negative")
        if self.checkpoint_steps <= 0:
            raise ValueError("checkpoint_steps must be positive")
        if self.validation_patience <= 0:
            raise ValueError("validation_patience must be positive")
        all_train = (self.train_dataset, *self.additional_train_datasets)
        all_validation = (
            self.validation_dataset,
            *self.regression_validation_datasets,
        )
        if len({path.resolve() for path in all_train}) != len(all_train):
            raise ValueError("ranking training datasets must be unique")
        if len({path.resolve() for path in all_validation}) != len(all_validation):
            raise ValueError("ranking validation datasets must be unique")
        self.model.validate()


def train_ranking(
    config: RankingTrainingConfig | Any,
    *,
    adapter: GroupedRankingAdapter | None = None,
) -> dict[str, Any]:
    """Train or resume a ranking model and return held-out metrics."""
    adapter = adapter or _entity_adapter()
    config.validate()
    train_datasets = (
        adapter.dataset_factory(config.train_dataset),
        *(adapter.dataset_factory(path) for path in config.additional_train_datasets),
    )
    validation_dataset = adapter.dataset_factory(config.validation_dataset)
    regression_datasets = tuple(
        adapter.dataset_factory(path) for path in config.regression_validation_datasets
    )
    if any(dataset.split != "train" for dataset in train_datasets):
        raise ValueError("ranking training datasets must use the train split")
    if validation_dataset.split not in {"validation", "test"}:
        raise ValueError("ranking validation dataset must use validation or test split")
    if any(dataset.split not in {"validation", "test"} for dataset in regression_datasets):
        raise ValueError(
            "ranking regression validation datasets must use validation or test splits"
        )
    teacher = train_datasets[0].manifest["teacher"]
    if any(dataset.manifest["teacher"] != teacher for dataset in train_datasets):
        raise ValueError("ranking training datasets must use the same frozen teacher")
    if validation_dataset.manifest["teacher"] != teacher or any(
        dataset.manifest["teacher"] != teacher for dataset in regression_datasets
    ):
        raise ValueError("ranking datasets must use the same frozen teacher")
    config.run_dir.mkdir(parents=True, exist_ok=True)
    run_manifest = _build_run_manifest(
        config,
        train_datasets,
        validation_dataset,
        regression_datasets,
        adapter,
    )

    if config.resume:
        validate_resume_manifest(
            config.run_dir,
            training=run_manifest["training"],
            datasets=run_manifest["datasets"],
            runtime=run_manifest["runtime"],
            source=run_manifest["source"],
        )
        model, optimizer, state, _ = load_latest_checkpoint_with_factory(
            config.run_dir,
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
            model_factory=adapter.model_factory,
        )
        if model.config != config.model:
            raise ValueError("resume ranking model configuration does not match")
        if config.epochs < state.epoch:
            raise ValueError("resume epoch budget is behind the checkpoint")
    else:
        if (config.run_dir / "latest.json").exists():
            raise ValueError("ranking run already has checkpoints; pass --resume")
        mx.random.seed(config.seed)
        model = (
            adapter.load_promoted(config.init_model_dir)
            if config.init_model_dir is not None
            else adapter.new_model(config.model)
        )
        if model.config != config.model:
            raise ValueError("initial ranking model configuration does not match")
        optimizer = optim.AdamW(
            learning_rate=config.learning_rate,
            weight_decay=config.weight_decay,
        )
        state = TrainerState()
        _write_json_atomic(config.run_dir / "run.json", run_manifest)

    initial_path = config.run_dir / "initial-validation.json"
    if initial_path.exists():
        initial_validation = json.loads(initial_path.read_text())
    else:
        initial_validation = {
            "validation": _evaluate_dataset(
                model, validation_dataset, config.group_batch_size, adapter
            ),
            "regression_validation": _evaluate_regressions(
                model,
                regression_datasets,
                config.group_batch_size,
                adapter=adapter,
            ),
        }
        initial_validation["selection_loss"] = _selection_loss(
            initial_validation["validation"],
            initial_validation["regression_validation"],
            adapter.selection_metric,
        )
        _write_json_atomic(initial_path, initial_validation)
    if not config.resume and state.best_ranking_loss is None:
        state.best_ranking_loss = initial_validation["selection_loss"]
        state.best_top1_accuracy = initial_validation["validation"][adapter.accuracy_metric]
        state.best_ranking_tiebreak_loss = _optional_metric(
            initial_validation["validation"],
            adapter.tertiary_metric,
        )
        checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
        set_checkpoint_pointer(
            config.run_dir,
            "best",
            checkpoint,
            initial_validation,
        )

    completed_report = _completed_resume_report(config, state)
    if completed_report is not None:
        return completed_report

    loss_and_grad = nn.value_and_grad(model, adapter.loss)
    metrics_path = config.run_dir / "metrics.jsonl"
    latest_validation: dict[str, Any] = {}
    latest_regression: dict[str, Any] = {}
    latest_selection_loss: float | None = None
    started = time.perf_counter()
    elapsed_before = state.elapsed_seconds
    stopped_early = state.ranking_epochs_without_improvement >= config.validation_patience

    for epoch in range(state.epoch, config.epochs):
        if stopped_early:
            break
        model.train()
        epoch_loss = 0.0
        trained_batches = 0
        resume_batch = state.batch_in_epoch if epoch == state.epoch else 0
        for batch_index, batch in enumerate(
            _training_batches(
                train_datasets,
                config.group_batch_size,
                seed=config.seed + epoch,
                adapter=adapter,
                batch_kwargs=adapter.batch_kwargs,
            )
        ):
            if batch_index < resume_batch:
                continue
            loss, gradients = loss_and_grad(model, batch)
            optimizer.update(model, gradients)
            mx.eval(model.parameters(), optimizer.state, loss)
            epoch_loss += float(loss.item())
            trained_batches += 1
            state.global_step += 1
            state.epoch = epoch
            state.batch_in_epoch = batch_index + 1
            if state.global_step % config.checkpoint_steps == 0:
                state.elapsed_seconds = elapsed_before + time.perf_counter() - started
                save_checkpoint(config.run_dir, model, optimizer, state)
                prune_checkpoints(config.run_dir)

        latest_validation = _evaluate_dataset(
            model, validation_dataset, config.group_batch_size, adapter
        )
        latest_regression = _evaluate_regressions(
            model,
            regression_datasets,
            config.group_batch_size,
            adapter=adapter,
        )
        latest_selection_loss = _selection_loss(
            latest_validation,
            latest_regression,
            adapter.selection_metric,
        )
        latest_accuracy = float(latest_validation[adapter.accuracy_metric])
        latest_tiebreak = _optional_metric(
            latest_validation,
            adapter.tertiary_metric,
        )
        improved = _ranking_improved(
            latest_selection_loss,
            latest_accuracy,
            latest_tiebreak,
            state,
        )
        if improved:
            state.best_ranking_loss = latest_selection_loss
            state.best_top1_accuracy = latest_accuracy
            state.best_ranking_tiebreak_loss = latest_tiebreak
            state.ranking_epochs_without_improvement = 0
        else:
            state.ranking_epochs_without_improvement += 1
        state.epoch = epoch + 1
        state.batch_in_epoch = 0
        event = {
            "epoch": epoch + 1,
            "global_step": state.global_step,
            "train_loss": epoch_loss / max(trained_batches, 1),
            "elapsed_seconds": elapsed_before + time.perf_counter() - started,
            "epochs_without_improvement": state.ranking_epochs_without_improvement,
            "selection_loss": latest_selection_loss,
            "validation": latest_validation,
            "regression_validation": latest_regression,
        }
        state.elapsed_seconds = event["elapsed_seconds"]
        _append_json(metrics_path, event)
        print(json.dumps(event, sort_keys=True), flush=True)
        checkpoint = save_checkpoint(config.run_dir, model, optimizer, state)
        if improved:
            set_checkpoint_pointer(
                config.run_dir,
                "best",
                checkpoint,
                {
                    "selection_loss": latest_selection_loss,
                    "validation": latest_validation,
                    "regression_validation": latest_regression,
                },
            )
        prune_checkpoints(config.run_dir)
        stopped_early = state.ranking_epochs_without_improvement >= config.validation_patience

    final_validation = latest_validation or _evaluate_dataset(
        model, validation_dataset, config.group_batch_size, adapter
    )
    final_regression = latest_regression or _evaluate_regressions(
        model,
        regression_datasets,
        config.group_batch_size,
        adapter=adapter,
    )
    final_selection_loss = (
        latest_selection_loss
        if latest_selection_loss is not None
        else _selection_loss(final_validation, final_regression, adapter.selection_metric)
    )
    final_report = {
        "schema_version": 1,
        "epochs": state.epoch,
        "global_step": state.global_step,
        "best_ranking_loss": state.best_ranking_loss,
        "best_top1_accuracy": state.best_top1_accuracy,
        "best_ranking_tiebreak_loss": state.best_ranking_tiebreak_loss,
        "initial_validation": initial_validation,
        "stopped_reason": (
            f"validation futility after {config.validation_patience} non-improving epochs"
            if stopped_early
            else "epoch budget completed"
        ),
        "validation": final_validation,
        "regression_validation": final_regression,
        "selection_loss": final_selection_loss,
        "elapsed_seconds": state.elapsed_seconds,
        "device": str(mx.default_device()),
    }
    _write_json_atomic(config.run_dir / "final-report.json", final_report)
    return final_report


def _completed_resume_report(
    config: RankingTrainingConfig | Any, state: TrainerState
) -> dict[str, Any] | None:
    if not config.resume:
        return None
    completed = (
        state.ranking_epochs_without_improvement >= config.validation_patience
        or state.epoch >= config.epochs
    )
    report_path = config.run_dir / "final-report.json"
    if not completed or not report_path.exists():
        return None
    try:
        report = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read completed ranking report: {error}") from error
    if (
        int(report.get("epochs", -1)) != state.epoch
        or int(report.get("global_step", -1)) != state.global_step
        or report.get("best_ranking_loss") != state.best_ranking_loss
        or report.get("best_top1_accuracy") != state.best_top1_accuracy
        or report.get("best_ranking_tiebreak_loss")
        != state.best_ranking_tiebreak_loss
    ):
        raise ValueError("completed ranking report disagrees with the latest checkpoint")
    return report


def evaluate_ranking(
    model: Any,
    dataset: Any,
    group_batch_size: int,
    *,
    adapter: GroupedRankingAdapter | None = None,
) -> dict[str, Any]:
    """Measure listwise loss, top-choice accuracy, regret, and rank fidelity."""
    adapter = adapter or _entity_adapter()
    model.eval()
    group_count = 0
    total_loss = 0.0
    top1_correct = 0
    strict_top1_correct = 0
    top1_regret = 0.0
    top5_recall = 0
    reciprocal_rank = 0.0
    pairwise_correct = 0
    pairwise_count = 0
    rank_correlations: list[float] = []
    teacher_differences: list[float] = []
    student_differences: list[float] = []

    for batch in dataset.batches(group_batch_size):
        scores = adapter.score_batch(model, batch)
        loss = adapter.loss(model, batch)
        mx.eval(scores, loss)
        score_values = np.asarray(scores)
        teacher_values = np.asarray(batch.teacher_mean)
        masks = np.asarray(batch.candidate_mask)
        total_loss += float(loss.item()) * len(score_values)
        group_count += len(score_values)

        for student, teacher, mask in zip(score_values, teacher_values, masks, strict=True):
            student = student[mask]
            teacher = teacher[mask]
            strict, value_recall, regret = _top1_metrics(student, teacher)
            strict_top1_correct += strict
            top1_correct += value_recall
            top1_regret += regret
            teacher_best_value = float(np.max(teacher))
            ranked_indices = np.argsort(-student, kind="stable")
            top5_recall += int(
                np.any(teacher[ranked_indices[: min(5, len(ranked_indices))]] == teacher_best_value)
            )
            first_best_rank = next(
                rank
                for rank, index in enumerate(ranked_indices, start=1)
                if float(teacher[index]) == teacher_best_value
            )
            reciprocal_rank += 1.0 / first_best_rank
            if len(student) > 1:
                rank_correlations.append(_rank_correlation(student, teacher))
            for left in range(len(student)):
                for right in range(left + 1, len(student)):
                    teacher_difference = float(teacher[left] - teacher[right])
                    if teacher_difference == 0.0:
                        continue
                    student_difference = float(student[left] - student[right])
                    pairwise_correct += int(
                        np.sign(teacher_difference) == np.sign(student_difference)
                    )
                    pairwise_count += 1
                    teacher_differences.append(teacher_difference)
                    student_differences.append(student_difference)

    if group_count == 0:
        raise ValueError("ranking evaluation dataset is empty")
    return {
        "groups": group_count,
        "candidates": dataset.candidate_count,
        "listwise_loss": total_loss / group_count,
        "top1_accuracy": top1_correct / group_count,
        "top1_value_recall": top1_correct / group_count,
        "strict_top1_accuracy": strict_top1_correct / group_count,
        "mean_top1_regret": top1_regret / group_count,
        "top5_recall": top5_recall / group_count,
        "mean_reciprocal_rank": reciprocal_rank / group_count,
        "pairwise_accuracy": pairwise_correct / max(pairwise_count, 1),
        "mean_rank_correlation": float(np.mean(rank_correlations)) if rank_correlations else 0.0,
        "value_difference_correlation": _correlation(
            student_differences,
            teacher_differences,
        ),
    }


def _top1_metrics(student: np.ndarray, teacher: np.ndarray) -> tuple[int, int, float]:
    teacher_best = int(np.argmax(teacher))
    student_best = int(np.argmax(student))
    best_value = float(teacher[teacher_best])
    chosen_value = float(teacher[student_best])
    return (
        int(student_best == teacher_best),
        int(chosen_value == best_value),
        best_value - chosen_value,
    )


def _training_batches(
    datasets: tuple[Any, ...],
    group_batch_size: int,
    *,
    seed: int,
    adapter: GroupedRankingAdapter,
    batch_kwargs: dict[str, object],
) -> Any:
    order = np.arange(len(datasets))
    rng = np.random.default_rng(seed)
    rng.shuffle(order)
    for order_index, dataset_index in enumerate(order):
        for batch in datasets[int(dataset_index)].batches(
            group_batch_size,
            shuffle=True,
            seed=seed + order_index + 1,
            **batch_kwargs,
        ):
            if adapter.augment_batch is not None:
                augmentation_seed = int(rng.integers(0, np.iinfo(np.int64).max, dtype=np.int64))
                batch = adapter.augment_batch(batch, augmentation_seed)
            yield batch


def _evaluate_regressions(
    model: Any,
    datasets: tuple[Any, ...],
    group_batch_size: int,
    *,
    adapter: GroupedRankingAdapter,
) -> dict[str, dict[str, Any]]:
    return {
        str(dataset.root.resolve()): _evaluate_dataset(model, dataset, group_batch_size, adapter)
        for dataset in datasets
    }


def _selection_loss(
    validation: dict[str, Any],
    regressions: dict[str, dict[str, Any]],
    metric: str = "listwise_loss",
) -> float:
    losses = [float(validation[metric])]
    losses.extend(float(metrics[metric]) for metrics in regressions.values())
    return float(np.mean(losses))


def _optional_metric(
    metrics: dict[str, Any],
    name: str | None,
) -> float | None:
    return None if name is None else float(metrics[name])


def _ranking_improved(
    selection_loss: float,
    accuracy: float,
    tiebreak_loss: float | None,
    state: TrainerState,
) -> bool:
    if state.best_ranking_loss is None:
        return True
    if selection_loss != state.best_ranking_loss:
        return selection_loss < state.best_ranking_loss
    best_accuracy = state.best_top1_accuracy
    if best_accuracy is None or accuracy != best_accuracy:
        return best_accuracy is None or accuracy > best_accuracy
    if tiebreak_loss is None:
        return False
    best_tiebreak = state.best_ranking_tiebreak_loss
    return best_tiebreak is None or tiebreak_loss < best_tiebreak


def _evaluate_dataset(
    model: Any,
    dataset: Any,
    group_batch_size: int,
    adapter: GroupedRankingAdapter,
) -> dict[str, Any]:
    if adapter.evaluate is not None:
        return adapter.evaluate(model, dataset, group_batch_size)
    return evaluate_ranking(model, dataset, group_batch_size, adapter=adapter)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--additional-train-dataset", type=Path, action="append", default=[])
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument(
        "--regression-validation-dataset",
        type=Path,
        action="append",
        default=[],
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-model-dir", type=Path)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--group-batch-size", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=20260610)
    parser.add_argument("--checkpoint-steps", type=int, default=500)
    parser.add_argument("--validation-patience", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--hidden-dim", type=int, default=96)
    parser.add_argument("--attention-heads", type=int, default=4)
    parser.add_argument("--board-blocks", type=int, default=2)
    parser.add_argument("--market-blocks", type=int, default=1)
    args = parser.parse_args()
    report = train_ranking(
        RankingTrainingConfig(
            train_dataset=args.train_dataset,
            additional_train_datasets=tuple(args.additional_train_dataset),
            validation_dataset=args.validation_dataset,
            regression_validation_datasets=tuple(args.regression_validation_dataset),
            run_dir=args.run_dir,
            init_model_dir=args.init_model_dir,
            epochs=args.epochs,
            group_batch_size=args.group_batch_size,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            seed=args.seed,
            checkpoint_steps=args.checkpoint_steps,
            validation_patience=args.validation_patience,
            resume=args.resume,
            model=RankingModelConfig(
                hidden_dim=args.hidden_dim,
                attention_heads=args.attention_heads,
                board_blocks=args.board_blocks,
                market_blocks=args.market_blocks,
            ),
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True))


def _rank_correlation(left: np.ndarray, right: np.ndarray) -> float:
    left_ranks = np.argsort(np.argsort(left)).astype(np.float64)
    right_ranks = np.argsort(np.argsort(right)).astype(np.float64)
    return _correlation(left_ranks.tolist(), right_ranks.tolist())


def _correlation(left: list[float], right: list[float]) -> float:
    if len(left) < 2:
        return 0.0
    value = np.corrcoef(left, right)[0, 1]
    return 0.0 if not np.isfinite(value) else float(value)


def _build_run_manifest(
    config: Any,
    train_datasets: tuple[Any, ...],
    validation_dataset: Any,
    regression_datasets: tuple[Any, ...],
    adapter: GroupedRankingAdapter,
) -> dict[str, Any]:
    training = asdict(config)
    training["train_dataset"] = str(config.train_dataset.resolve())
    training["additional_train_datasets"] = [
        str(path.resolve()) for path in config.additional_train_datasets
    ]
    training["validation_dataset"] = str(config.validation_dataset.resolve())
    training["regression_validation_datasets"] = [
        str(path.resolve()) for path in config.regression_validation_datasets
    ]
    training["run_dir"] = str(config.run_dir.resolve())
    training["init_model_dir"] = (
        str(config.init_model_dir.resolve()) if config.init_model_dir is not None else None
    )
    training["init_model_manifest_blake3"] = (
        _checksum(config.init_model_dir / adapter.init_manifest_name)
        if config.init_model_dir is not None
        else None
    )
    training["model"] = config.model.to_dict()
    return {
        "schema_version": 1,
        "kind": adapter.kind,
        "training": training,
        "datasets": {
            "teacher": train_datasets[0].manifest["teacher"],
            "train_manifest_blake3": _checksum(config.train_dataset / "dataset.json"),
            "validation_manifest_blake3": _checksum(config.validation_dataset / "dataset.json"),
            "train_groups": sum(dataset.group_count for dataset in train_datasets),
            "validation_groups": validation_dataset.group_count,
            "train_candidates": sum(dataset.candidate_count for dataset in train_datasets),
            "validation_candidates": validation_dataset.candidate_count,
            "train_datasets": [_dataset_identity(dataset) for dataset in train_datasets],
            "regression_validation_datasets": [
                _dataset_identity(dataset) for dataset in regression_datasets
            ],
        },
        "runtime": {
            "mlx_version": version("mlx"),
            "python_version": platform.python_version(),
            "machine": platform.machine(),
            "platform": platform.platform(),
            "device": str(mx.default_device()),
        },
        "source": source_provenance(Path(__file__).resolve().parents[2]),
    }


def _dataset_identity(dataset: Any) -> dict[str, Any]:
    return {
        "path": str(dataset.root.resolve()),
        "manifest_blake3": _checksum(dataset.root / "dataset.json"),
        "dataset_id": dataset.manifest["dataset_id"],
        "feature_schema": dataset.manifest["feature_schema"],
        "target_schema": dataset.manifest["target_schema"],
        "source": dataset.manifest.get("source"),
        "trajectory": dataset.manifest.get("trajectory"),
        "groups": dataset.group_count,
        "candidates": dataset.candidate_count,
    }


def _append_json(path: Path, value: Any) -> None:
    with path.open("a") as handle:
        handle.write(json.dumps(value, sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _write_json_atomic(path: Path, value: Any) -> None:
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True, default=str) + "\n")
    os.replace(temp, path)


def _checksum(path: Path) -> str:
    return blake3.blake3(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
