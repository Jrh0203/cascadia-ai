from cascadia_mlx.public_beam_value_evaluate import validation_gates


def test_public_beam_value_validation_gates_require_every_metric() -> None:
    metrics = {
        "mean_absolute_error": 0.9,
        "value_correlation": 0.92,
        "centered_advantage_correlation": 0.7,
        "top_action_agreement": 0.55,
        "mean_top_action_regret": 0.3,
    }
    assert all(validation_gates(metrics).values())

    metrics["centered_advantage_correlation"] = 0.64
    gates = validation_gates(metrics)
    assert gates["centered_advantage_correlation"] is False
    assert not all(gates.values())
