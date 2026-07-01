#!/bin/zsh
set -euo pipefail
umask 077

readonly ROOT="${CASCADIA_CLUSTER_ROOT:?CASCADIA_CLUSTER_ROOT is required}"
readonly ROLE="${CASCADIA_BACALHAU_ROLE:?CASCADIA_BACALHAU_ROLE is required}"
readonly BIN="$ROOT/bin/bacalhau"
readonly CONFIG="$ROOT/config/bacalhau.yaml"
readonly SECRETS="$ROOT/config/secrets.env"
readonly DATA="$ROOT/state/bacalhau"

[[ -x "$BIN" && -r "$CONFIG" && -r "$SECRETS" ]] || {
  print -u2 "Bacalhau runtime is incomplete under $ROOT"
  exit 1
}

set -a
source "$SECRETS"
set +a

args=(
  serve
  --config "$CONFIG"
  --data-dir "$DATA"
  --config "Compute.Auth.Token=$BACALHAU_COMPUTE_TOKEN"
)
if [[ "$ROLE" == orchestrator ]]; then
  args+=(
    --config "Orchestrator.Auth.Token=$BACALHAU_COMPUTE_TOKEN"
    --config "Publishers.Types.S3Managed.Bucket=cascadia-results"
  )
fi

exec "$BIN" "${args[@]}"
