from __future__ import annotations

from frontier_supervision_identifiability_report import classify


def payload(*, hard: bool, soft: bool) -> dict[str, dict[str, object]]:
    hard_report = {
        "train": {"gate_passed": hard},
        "validation": {"gate_passed": hard},
    }
    return {
        "boundary-signal": hard_report,
        "cross-fidelity": hard_report,
        "teacher-resampling": hard_report,
        "expected-rank-ceiling": {
            "train": {"gate_passed": soft},
            "validation": {"gate_passed": soft},
        },
    }


def test_classification_prefers_sufficient_soft_supervision() -> None:
    assert classify(payload(hard=False, soft=True)) == "uncertainty_aware_supervision_sufficient"


def test_classification_distinguishes_stable_hard_target() -> None:
    assert (
        classify(payload(hard=True, soft=False))
        == "hard_target_stable_but_soft_ceiling_insufficient"
    )


def test_classification_rejects_unstable_existing_teacher() -> None:
    assert classify(payload(hard=False, soft=False)) == "existing_teacher_supervision_insufficient"
