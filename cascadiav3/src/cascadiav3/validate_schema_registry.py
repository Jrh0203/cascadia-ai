"""Validate the additive Cascadia v3 schema registry."""

from __future__ import annotations

import argparse
import json

from .schema import registry_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--include-legacy", action="store_true")
    parser.add_argument("--include-expert", action="store_true")
    args = parser.parse_args()

    include_legacy = args.include_legacy or not args.include_expert
    include_expert = args.include_expert or not args.include_legacy
    report = registry_report(include_legacy=include_legacy, include_expert=include_expert)
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0 if report["status"] == "pass" else 1


if __name__ == "__main__":
    raise SystemExit(main())
