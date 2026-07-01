#!/usr/bin/env -S python3 -I -S -B
"""Standalone entrypoint for the R2-MAP D0 infrastructure helper."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from r2_d0.cli import main

if __name__ == "__main__":
    raise SystemExit(main())
