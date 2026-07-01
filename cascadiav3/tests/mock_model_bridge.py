from __future__ import annotations

import json
import sys


def emit(payload: dict) -> None:
    sys.stdout.write(json.dumps(payload, sort_keys=True) + "\n")
    sys.stdout.flush()


def main() -> int:
    model_fallback = "--model-fallback" in sys.argv[1:]
    emit({"type": "hello", "protocol": "cascadiav3.mock_model_bridge.v1"})
    for line in sys.stdin:
        message = json.loads(line)
        if message.get("type") == "shutdown":
            emit({"type": "shutdown", "status": "ok"})
            return 0
        if message.get("type") != "eval_request":
            emit({"type": "error", "error": f"unknown message type {message.get('type')!r}"})
            continue
        root = message["root"]
        action_ids = [action["action_id"] for action in root["legal_actions"]]
        weights = [float(index + 1) for index in range(len(action_ids))]
        total = sum(weights)
        emit(
            {
                "type": "eval_response",
                "state_hash": root["state_hash"],
                "action_ids": action_ids,
                "priors": [weight / total for weight in weights],
                "q": [0.0 for _ in action_ids],
                "uncertainty": [1.0 for _ in action_ids],
                "model_fallback": model_fallback,
            }
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
