#!/usr/bin/env bash
set -euo pipefail

# Rust and cc-rs require a single executable path for the linker. Zig exposes
# its C-driver compatibility as a subcommand, so this checked-in adapter keeps
# the target build command explicit and reproducible. cc-rs spells the native
# Rust triple with the vendor component (`unknown`); Zig uses the equivalent
# vendor-free spelling.
args=()
for arg in "$@"; do
  case "$arg" in
    --target=x86_64-unknown-linux-gnu)
      args+=(--target=x86_64-linux-gnu)
      ;;
    *)
      args+=("$arg")
      ;;
  esac
done
exec "${ZIG:?set ZIG to the pinned Zig executable}" cc "${args[@]}"
