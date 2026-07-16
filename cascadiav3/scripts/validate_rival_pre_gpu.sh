#!/usr/bin/env bash
# Complete CPU-only validation gate for Cascadia Rival pre-GPU machinery.
#
# This script deliberately sets the device contract instead of discovering
# hardware. It must never contain or call an accelerator availability probe.

set -euo pipefail

if (($# != 0)); then
  echo "usage: $0" >&2
  exit 2
fi

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd -P)"
REPO_ROOT="$(cd -- "$SCRIPT_DIR/../.." && pwd -P)"
PYTHON="${PYTHON:-$REPO_ROOT/.venv/bin/python}"
RUFF="${RUFF:-$REPO_ROOT/.venv/bin/ruff}"
CARGO="${CARGO:-cargo}"

for executable in "$PYTHON" "$RUFF"; do
    if [[ ! -x "$executable" ]]; then
        echo "required executable is unavailable: $executable" >&2
        exit 2
    fi
done
if ! command -v "$CARGO" >/dev/null 2>&1; then
    echo "required Cargo executable is unavailable: $CARGO" >&2
    exit 2
fi

export CUDA_VISIBLE_DEVICES=""
export CASCADIA_DEVICE="cpu"
export CASCADIA_CPU_ONLY_TESTS="1"
export PYTHONDONTWRITEBYTECODE="1"
export PYTHONWARNINGS="error"
export PYTHONPATH="$REPO_ROOT/cascadiav3/src${PYTHONPATH:+:$PYTHONPATH}"

cd -- "$REPO_ROOT"

"$PYTHON" -c \
  'from cascadiav3.cpu_test_guard import assert_cpu_only_test_environment; assert_cpu_only_test_environment()'

"$PYTHON" -m cascadiav3.rival.preflight \
  --fixture cascadiav3/tests/fixtures/rival/preflight_fixture.json \
  --device cpu \
  --validate-only

"$CARGO" fmt --package cascadia-rival -- --check
"$CARGO" fmt --package cascadia-provenance -- --check

"$CARGO" check \
  --locked \
  -p cascadia-rival \
  --all-targets \
  --no-default-features \
  --features cpu-reference

"$CARGO" test \
  --locked \
  -p cascadia-rival \
  --no-default-features \
  --features cpu-reference

"$CARGO" clippy \
  --locked \
  -p cascadia-rival \
  --all-targets \
  --no-default-features \
  --features cpu-reference \
  --no-deps \
  -- -D warnings

"$CARGO" test --locked -p cascadia-provenance

"$CARGO" build \
  --locked \
  -p cascadia-rival \
  --no-default-features \
  --features cpu-reference \
  --bin rival-contract

export CASCADIA_RIVAL_CONTRACT_BIN="$REPO_ROOT/target/debug/rival-contract"

"$PYTHON" -m unittest discover \
  -s cascadiav3/tests \
  -p 'test_cpu_test_guard.py' \
  -v

# These existing suites exercise production entry points modified by the
# CPU-only guard.  Keep them in this gate so a test-only safety change cannot
# silently regress ordinary bridge or trainer behavior.
"$PYTHON" -m unittest discover \
  -s cascadiav3/tests \
  -p 'test_bridge_throughput_knobs.py' \
  -v

"$PYTHON" -m unittest discover \
  -s cascadiav3/tests \
  -p 'test_trainer_perf_knobs.py' \
  -v

"$PYTHON" -m unittest discover \
  -s cascadiav3/tests \
  -p 'test_rival_*.py' \
  -v

"$RUFF" check \
  cascadiav3/src/cascadiav3/cpu_test_guard.py \
  cascadiav3/src/cascadiav3/rival \
  cascadiav3/tests/test_cpu_test_guard.py \
  cascadiav3/tests/test_rival_*.py

"$RUFF" format --check \
  cascadiav3/src/cascadiav3/cpu_test_guard.py \
  cascadiav3/src/cascadiav3/rival \
  cascadiav3/tests/test_cpu_test_guard.py \
  cascadiav3/tests/test_rival_*.py

echo "Rival pre-GPU validation passed under the explicit CPU-only contract."
