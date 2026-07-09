#!/usr/bin/env bash
set -euo pipefail

# Rust and cc-rs require a single executable path for the linker. Zig exposes
# its C-driver compatibility as a subcommand, so this tiny checked-in adapter
# keeps the target build command explicit and reproducible.
exec "${ZIG:?set ZIG to the pinned Zig executable}" cc "$@"
