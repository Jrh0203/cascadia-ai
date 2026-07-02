from __future__ import annotations

import json
import sys


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def eval_response(root: dict, *, model_fallback: bool) -> dict:
    action_ids = [action["action_id"] for action in root["legal_actions"]]
    weights = [float(index + 1) for index in range(len(action_ids))]
    total = sum(weights)
    exact = root.get("exact_afterstate_score_active")
    if isinstance(exact, list) and len(exact) == len(action_ids):
        score_to_go = [1.0 for _ in action_ids]
        q_values = [float(afterstate) + 1.0 for afterstate in exact]
    else:
        score_to_go = [0.0 for _ in action_ids]
        q_values = [0.0 for _ in action_ids]
    return {
        "type": "eval_response",
        "state_hash": root["state_hash"],
        "action_ids": action_ids,
        "priors": [weight / total for weight in weights],
        "q": q_values,
        "score_to_go": score_to_go,
        "uncertainty": [1.0 for _ in action_ids],
        "value": [80.0, 80.0, 80.0, 80.0],
        "model_fallback": model_fallback,
    }


def main() -> int:
    model_fallback = "--model-fallback" in sys.argv[1:]
    no_batch = "--no-batch" in sys.argv[1:]
    hello: dict = {"type": "hello", "protocol": "cascadiav3.mock_model_bridge.v1"}
    if not no_batch:
        hello["protocol_features"] = ["eval_batch", "value_vector"]
    emit(hello)
    for line in sys.stdin:
        message = json.loads(line)
        message_type = message.get("type")
        if message_type == "shutdown":
            emit({"type": "shutdown", "status": "ok"})
            return 0
        if message_type == "eval_request":
            emit(eval_response(message["root"], model_fallback=model_fallback))
            continue
        if message_type == "eval_batch_request" and not no_batch:
            results = [
                eval_response(root, model_fallback=model_fallback)
                for root in message["roots"]
            ]
            emit({"type": "eval_batch_response", "results": results})
            continue
        emit({"type": "error", "error": f"unknown message type {message_type!r}"})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
