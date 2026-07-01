#!/bin/sh
set -eu

output_root="${CASCADIA_OUTPUT_ROOT:-/outputs}"
scratch_root="${CASCADIA_SCRATCH_ROOT:-/tmp/cascadia-scratch-$$}"
metadata="${CASCADIA_APPLICATION_METADATA_JSON:-}"
if [ -z "$metadata" ]; then
  metadata='{}'
fi
retryable=",${CASCADIA_RETRYABLE_EXIT_CODES:-125,126,127,137,143},"
mkdir -p "$output_root"
case "$scratch_root" in
  /*) ;;
  *) echo "CASCADIA_SCRATCH_ROOT must be absolute" >&2; exit 125 ;;
esac
rm -rf "$scratch_root"
mkdir -p "$scratch_root"
chmod 700 "$scratch_root"
export CASCADIA_SCRATCH_ROOT="$scratch_root"
cleanup_scratch() {
  rm -rf "$scratch_root" /tmp/cascadia-models
}
trap cleanup_scratch EXIT HUP INT TERM

# Bacalhau's S3 ChecksumSHA256 field refers to optional checksum metadata
# returned by HeadObject, not the raw object's hexadecimal digest. MinIO does
# not expose that metadata for these uploads, so validate the content-addressed
# digest inside the container before any user command can observe the input.
input_checks="${CASCADIA_INPUT_SHA256_JSON:-}"
if [ -z "$input_checks" ]; then
  input_checks='{}'
fi
python3 - "$input_checks" <<'PY'
import hashlib
import json
import os
import stat
import sys

checks = json.loads(sys.argv[1])
if not isinstance(checks, dict):
    raise SystemExit("cluster input checksum map is not an object")
for raw_path, expected in sorted(checks.items()):
    if not isinstance(raw_path, str) or not raw_path.startswith("/"):
        raise SystemExit("cluster input checksum path is not absolute")
    if not isinstance(expected, str) or len(expected) != 64:
        raise SystemExit("cluster input checksum is malformed")
    info = os.lstat(raw_path)
    if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
        raise SystemExit(f"cluster input is not a regular file: {raw_path}")
    digest = hashlib.sha256()
    with open(raw_path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if digest.hexdigest() != expected:
        raise SystemExit(f"cluster input checksum differs: {raw_path}")
PY
unset CASCADIA_INPUT_SHA256_JSON

# Bacalhau mounts every S3 object into a distinct directory. Materialize any
# explicitly declared two-file V3 serving bundles only after each source file
# has passed the content-addressed checksum gate above.
model_bundles="${CASCADIA_MODEL_BUNDLES_JSON:-}"
if [ -z "$model_bundles" ]; then
  model_bundles='{}'
fi
python3 - "$model_bundles" <<'PY'
import json
import os
import shutil
import stat
import sys

bundles = json.loads(sys.argv[1])
if not isinstance(bundles, dict):
    raise SystemExit("model bundle map is not an object")
for destination, sources in sorted(bundles.items()):
    if not isinstance(destination, str) or not destination.startswith("/tmp/cascadia-models/"):
        raise SystemExit("model bundle destination is outside /tmp/cascadia-models")
    if not isinstance(sources, dict) or set(sources) != {"manifest", "weights"}:
        raise SystemExit("model bundle sources must contain manifest and weights")
    manifest = sources["manifest"]
    weights = sources["weights"]
    for source in (manifest, weights):
        if not isinstance(source, str) or not source.startswith("/inputs/"):
            raise SystemExit("model bundle source is outside /inputs")
        info = os.lstat(source)
        if not stat.S_ISREG(info.st_mode) or stat.S_ISLNK(info.st_mode):
            raise SystemExit(f"model bundle source is not a regular file: {source}")
    with open(manifest, encoding="utf-8") as stream:
        model = json.load(stream)
    weights_name = model.get("weights_file")
    if not isinstance(weights_name, str) or os.path.basename(weights_name) != weights_name:
        raise SystemExit("model manifest weights_file is not a portable basename")
    os.makedirs(destination, mode=0o700, exist_ok=False)
    shutil.copyfile(manifest, os.path.join(destination, "model.json"))
    shutil.copyfile(weights, os.path.join(destination, weights_name))
PY
unset CASCADIA_MODEL_BUNDLES_JSON

set +e
"$@"
status=$?
set -e

if [ "$status" -ne 0 ]; then
  case "$retryable" in
    *,"$status",*) exit "$status" ;;
  esac
  printf '{"exit_code":%s,"status":"failed"}\n' "$status" > "$output_root/application-failure.json"
  metadata="$(python3 -c 'import json,sys; value=json.loads(sys.argv[1]); value["cascadia_application_status"]="failed"; value["cascadia_exit_code"]=int(sys.argv[2]); print(json.dumps(value,separators=(",",":")))' "$metadata" "$status")"
fi

python3 -m cascadia_cluster.manifest_writer \
  --root "$output_root" \
  --metadata-json "$metadata" \
  --protocol-version "${CASCADIA_PROTOCOL_VERSION:-cascadia-cluster-map-v1}" \
  -- "$@"

# A deterministic application failure is represented in the validated manifest,
# so Bacalhau does not churn it across every worker. The importer restores failure.
exit 0
