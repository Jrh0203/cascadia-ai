#!/bin/zsh
set -euo pipefail
umask 077

readonly ROOT="${CASCADIA_CLUSTER_ROOT:?CASCADIA_CLUSTER_ROOT is required}"
readonly DOCKER="${DOCKER_BIN:-/opt/homebrew/bin/docker}"
readonly TAILSCALE="${TAILSCALE_BIN:-/opt/homebrew/bin/tailscale}"
# Docker runs inside the dedicated Colima VM, which cannot bind John1's macOS
# Tailscale address. Publish only to loopback high ports, then let Tailscale's
# native TCP proxy expose the registered private-cluster ports. This also
# avoids macOS services that already own wildcard port 5000.
readonly REGISTRY_LOOPBACK_PORT="${CASCADIA_REGISTRY_LOOPBACK_PORT:-15000}"
readonly MINIO_API_LOOPBACK_PORT="${CASCADIA_MINIO_API_LOOPBACK_PORT:-19000}"
readonly MINIO_CONSOLE_LOOPBACK_PORT="${CASCADIA_MINIO_CONSOLE_LOOPBACK_PORT:-19001}"
readonly REGISTRY_IMAGE='registry@sha256:85347ed2ecde64161c7a4788a4d7d3dcc9d6f86f7be95834022e3c6a423a945a'
readonly MINIO_IMAGE='quay.io/minio/minio@sha256:14cea493d9a34af32f524e538b8346cf79f3321eff8e708c1e2960462bd8936e'

set -a
source "$ROOT/config/secrets.env"
set +a

mkdir -p "$ROOT/state/registry" "$ROOT/state/minio"

if ! "$DOCKER" container inspect cascadia-registry >/dev/null 2>&1; then
  "$DOCKER" run -d --name cascadia-registry --restart unless-stopped \
    -p "127.0.0.1:$REGISTRY_LOOPBACK_PORT:5000" \
    -v "$ROOT/state/registry:/var/lib/registry" \
    -v "$ROOT/config/registry.yml:/etc/distribution/config.yml:ro" \
    "$REGISTRY_IMAGE" >/dev/null
else
  "$DOCKER" start cascadia-registry >/dev/null 2>&1 || true
fi

if ! "$DOCKER" container inspect cascadia-minio >/dev/null 2>&1; then
  "$DOCKER" run -d --name cascadia-minio --restart unless-stopped \
    -p "127.0.0.1:$MINIO_API_LOOPBACK_PORT:9000" \
    -p "127.0.0.1:$MINIO_CONSOLE_LOOPBACK_PORT:9001" \
    -e MINIO_ROOT_USER -e MINIO_ROOT_PASSWORD \
    -v "$ROOT/state/minio:/data" \
    "$MINIO_IMAGE" server /data --console-address ':9001' >/dev/null
else
  "$DOCKER" start cascadia-minio >/dev/null 2>&1 || true
fi

"$TAILSCALE" serve --bg --tcp 5000 "tcp://127.0.0.1:$REGISTRY_LOOPBACK_PORT" >/dev/null
"$TAILSCALE" serve --bg --tcp 9000 "tcp://127.0.0.1:$MINIO_API_LOOPBACK_PORT" >/dev/null
"$TAILSCALE" serve --bg --tcp 9001 "tcp://127.0.0.1:$MINIO_CONSOLE_LOOPBACK_PORT" >/dev/null
