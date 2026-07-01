#!/usr/bin/env python3
"""Continuously publish the truthful RED-D0 dashboard heartbeat on John1."""

from __future__ import annotations

import argparse
import json
import sys
import time

from r2_d0.live_dashboard import LiveDashboardError, publish_red_heartbeat


def parser() -> argparse.ArgumentParser:
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--watch", action="store_true")
    result.add_argument("--interval-seconds", type=int, default=10, choices=range(5, 31))
    return result


def main() -> int:
    arguments = parser().parse_args()
    while True:
        try:
            result = publish_red_heartbeat()
        except LiveDashboardError as error:
            print(f"R2-MAP D0 dashboard heartbeat failed: {error}", file=sys.stderr, flush=True)
            if not arguments.watch:
                return 2
        else:
            print(json.dumps(result, sort_keys=True), flush=True)
            if not arguments.watch:
                return 0
        time.sleep(arguments.interval_seconds)


if __name__ == "__main__":
    raise SystemExit(main())
