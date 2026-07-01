"""Train the preregistered exact-parent candidate-set residual in MLX."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from cascadia_mlx.checkpoint import load_checkpoint_pointer_with_factory
from cascadia_mlx.imitation_distribution_train import evaluate_imitation_evidence
from cascadia_mlx.imitation_model import (
    IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4,
    ImitationModelConfig,
    SharedStateActionRanker,
    distributional_imitation_loss,
    score_imitation_actions,
)
from cascadia_mlx.imitation_parent_prior_dataset import (
    ImitationParentEvidenceDataset,
)
from cascadia_mlx.ranking_train import GroupedRankingAdapter, train_ranking


@dataclass(frozen=True)
class ParentResidualTrainingConfig:
    train_dataset: Path
    validation_dataset: Path
    run_dir: Path
    epochs: int = 30
    group_batch_size: int = 8
    learning_rate: float = 5e-5
    weight_decay: float = 1e-4
    seed: int = 20260622
    checkpoint_steps: int = 500
    validation_patience: int = 6
    resume: bool = False
    init_model_dir: Path | None = None
    additional_train_datasets: tuple[Path, ...] = ()
    regression_validation_datasets: tuple[Path, ...] = ()
    model: ImitationModelConfig = field(
        default_factory=lambda: ImitationModelConfig(
            architecture=IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4,
            hidden_dim=192,
            attention_heads=8,
            board_blocks=3,
            market_blocks=1,
        )
    )

    def validate(self) -> None:
        if self.epochs <= 0 or self.group_batch_size <= 0 or self.checkpoint_steps <= 0:
            raise ValueError("parent-residual training counts must be positive")
        if self.learning_rate <= 0 or self.weight_decay < 0:
            raise ValueError("parent-residual optimizer configuration is invalid")
        if self.validation_patience <= 0:
            raise ValueError("validation patience must be positive")
        if self.init_model_dir is not None:
            raise ValueError("parent-residual training does not warm-start")
        if self.additional_train_datasets or self.regression_validation_datasets:
            raise ValueError("ADR 0069 accepts exactly one train and validation dataset")
        self.model.validate()
        if self.model.architecture != IMITATION_ARCHITECTURE_PARENT_SET_RESIDUAL_V4:
            raise ValueError("ADR 0069 freezes the exact-parent set-residual architecture")
        if (
            self.model
            != ParentResidualTrainingConfig(
                self.train_dataset,
                self.validation_dataset,
                self.run_dir,
            ).model
        ):
            raise ValueError("ADR 0069 model dimensions are frozen")


def parent_residual_adapter() -> GroupedRankingAdapter:
    return GroupedRankingAdapter(
        kind="exact-parent-candidate-set-residual",
        dataset_factory=ImitationParentEvidenceDataset,
        model_factory=lambda values: SharedStateActionRanker(
            ImitationModelConfig.from_dict(values)
        ),
        new_model=SharedStateActionRanker,
        load_promoted=_reject_warm_start,
        loss=distributional_imitation_loss,
        score_batch=lambda model, batch: score_imitation_actions(
            model,
            batch.board_entities,
            batch.board_mask,
            batch.market_entities,
            batch.market_mask,
            batch.global_features,
            batch.action_features,
            batch.candidate_mask,
            batch.parent_total,
            batch.parent_rank,
        ),
        evaluate=evaluate_parent_residual,
        selection_metric="distributional_loss",
        accuracy_metric="top1_accuracy",
    )


def _reject_warm_start(_path: Path) -> SharedStateActionRanker:
    raise ValueError("ADR 0069 does not warm-start the residual model")


def evaluate_parent_residual(
    model: SharedStateActionRanker,
    dataset: ImitationParentEvidenceDataset,
    group_batch_size: int,
) -> dict[str, object]:
    return evaluate_imitation_evidence(
        model,
        dataset,
        group_batch_size,
        loss_function=distributional_imitation_loss,
        loss_metric="distributional_loss",
    )


def train_parent_residual(config: ParentResidualTrainingConfig) -> dict[str, Any]:
    config.validate()
    train_dataset = ImitationParentEvidenceDataset(config.train_dataset)
    validation_dataset = ImitationParentEvidenceDataset(config.validation_dataset)
    if train_dataset.manifest["model"] != validation_dataset.manifest["model"]:
        raise ValueError("parent-residual train and validation parents differ")
    if (
        train_dataset.evidence.source.manifest["candidates"]
        != validation_dataset.evidence.source.manifest["candidates"]
    ):
        raise ValueError("parent-residual candidate contracts differ")

    adapter = parent_residual_adapter()
    training_report = train_ranking(config, adapter=adapter)
    selected, _, state, checkpoint = load_checkpoint_pointer_with_factory(
        config.run_dir,
        pointer="best",
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        model_factory=adapter.model_factory,
    )
    baseline = SharedStateActionRanker(config.model)
    baseline_train = evaluate_parent_residual(
        baseline,
        train_dataset,
        config.group_batch_size,
    )
    baseline_validation = evaluate_parent_residual(
        baseline,
        validation_dataset,
        config.group_batch_size,
    )
    selected_train = evaluate_parent_residual(
        selected,
        train_dataset,
        config.group_batch_size,
    )
    selected_validation = evaluate_parent_residual(
        selected,
        validation_dataset,
        config.group_batch_size,
    )
    gates = _validation_gates(
        baseline_train,
        selected_train,
        baseline_validation,
        selected_validation,
    )
    report = {
        "schema_version": 1,
        "experiment_id": "exact-parent-candidate-set-residual-v4-20260612",
        "status": "validation-passed" if all(gates.values()) else "rejected-on-validation",
        "training": asdict(config),
        "selected_checkpoint": str(checkpoint.resolve()),
        "selected_epoch": state.epoch,
        "selected_step": state.global_step,
        "baseline_train": baseline_train,
        "selected_train": selected_train,
        "baseline_validation": baseline_validation,
        "selected_validation": selected_validation,
        "gates": gates,
        "failed_gates": [name for name, passed in gates.items() if not passed],
        "training_report": training_report,
        "test_domain_opened": False,
        "gameplay_domain_opened": False,
    }
    path = config.run_dir / "adr69-report.json"
    _write_json_atomic(path, report)
    return report


def _validation_gates(
    baseline_train: dict[str, object],
    selected_train: dict[str, object],
    baseline: dict[str, object],
    selected: dict[str, object],
) -> dict[str, bool]:
    return {
        "distributional_loss": float(selected["distributional_loss"])
        < float(baseline["distributional_loss"]),
        "validation_top1": float(selected["top1_accuracy"])
        >= float(baseline["top1_accuracy"]) + 0.03,
        "validation_top5": float(selected["top5_recall"]) >= float(baseline["top5_recall"]) + 0.05,
        "validation_mrr": float(selected["mean_reciprocal_rank"])
        >= float(baseline["mean_reciprocal_rank"]) + 0.04,
        "validation_pairwise": float(selected["scored_pairwise_accuracy"])
        >= float(baseline["scored_pairwise_accuracy"]) + 0.02,
        "validation_value_correlation": float(selected["scored_value_difference_correlation"])
        >= float(baseline["scored_value_difference_correlation"]),
        "validation_regret": _optional_float(selected["conditional_mean_regret"])
        <= _optional_float(baseline["conditional_mean_regret"]) - 0.15,
        "validation_teacher_coverage": float(selected["predicted_teacher_coverage"])
        >= float(baseline["predicted_teacher_coverage"]),
        "train_top1": float(selected_train["top1_accuracy"])
        >= float(baseline_train["top1_accuracy"]) + 0.05,
    }


def _optional_float(value: object) -> float:
    if value is None:
        return float("inf")
    return float(value)


def _write_json_atomic(path: Path, value: dict[str, Any]) -> None:
    serializable = json.loads(json.dumps(value, default=str))
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(serializable, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dataset", type=Path, required=True)
    parser.add_argument("--validation-dataset", type=Path, required=True)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--validation-patience", type=int, default=6)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    report = train_parent_residual(
        ParentResidualTrainingConfig(
            train_dataset=args.train_dataset,
            validation_dataset=args.validation_dataset,
            run_dir=args.run_dir,
            epochs=args.epochs,
            validation_patience=args.validation_patience,
            resume=args.resume,
        )
    )
    print(json.dumps(report, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
