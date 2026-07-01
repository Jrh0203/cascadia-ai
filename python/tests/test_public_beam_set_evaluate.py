from cascadia_mlx.public_beam_set_evaluate import validation_gates


def test_public_beam_set_validation_gates_require_every_metric() -> None:
    metrics = {
        "centered_advantage_correlation": 0.72,
        "top_value_recall": 0.45,
        "mean_top_action_regret": 0.30,
    }
    assert all(validation_gates(metrics).values())

    metrics["top_value_recall"] = 0.39
    gates = validation_gates(metrics)
    assert gates["top_value_recall"] is False
    assert not all(gates.values())
