"""Validate D6 transform/inverse preservation for exported action coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .hex import coord_ref
from .replay import read_replay_jsonl


def _transform(q: int, r: int, index: int) -> tuple[int, int]:
    s = -q - r
    transforms = (
        (q, r),
        (-r, -s),
        (s, q),
        (-q, -r),
        (r, s),
        (-s, -q),
    )
    return transforms[index]


def _inverse_index(index: int) -> int:
    return (0, 5, 4, 3, 2, 1)[index]


def _roundtrip_coord(original: dict[str, Any], transform_index: int) -> bool:
    q = int(original["q"])
    r = int(original["r"])
    tq, tr = _transform(q, r, transform_index)
    iq, ir = _transform(tq, tr, _inverse_index(transform_index))
    if original["kind"] == "canonical":
        restored = coord_ref(iq, ir)
    else:
        restored = coord_ref(
            iq,
            ir,
            owner_seat=int(original["owner_seat"]),
            placement_id=int(original["placement_id"]),
        )
    return restored == original


def validate_roots(path: Path) -> dict[str, Any]:
    records = read_replay_jsonl(path)
    checked = 0
    failures: list[dict[str, Any]] = []
    for root_index, record in enumerate(records):
        for action in record["legal_actions"]:
            for field in ("target_coord_ref", "wildlife_coord_ref"):
                coord = action[field]
                for transform_index in range(6):
                    checked += 1
                    if not _roundtrip_coord(coord, transform_index):
                        failures.append(
                            {
                                "root_index": root_index,
                                "action_id": action["action_id"],
                                "field": field,
                                "transform": transform_index,
                            }
                        )
    return {
        "status": "pass" if not failures else "fail",
        "roots": len(records),
        "coordinate_roundtrips_checked": checked,
        "failures": failures[:20],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--roots", required=True)
    args = parser.parse_args()
    report = validate_roots(Path(args.roots))
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
