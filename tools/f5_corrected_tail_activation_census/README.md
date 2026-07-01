# F5 Corrected-Tail Activation Census

Standalone implementation for
`corrected-mid-tail-activation-census-v1`.

It generates deterministic public-state records, replays every record through
the actual legacy Rust extractor compiled with
`legacy-mid-v4-fixed-v1`, reports all 301 corrected channels by phase, seat,
policy, and overflow slice, and merges four disjoint modulo-owned shards.

Trajectory policy is record provenance, not part of the hashed public game
state. Artifact schema version 2 enforces that boundary.

All scientific JSON inputs pass through a recursive duplicate-key-rejecting
parser before typed deserialization.

The Git object ID is embedded by `build.rs`; runtime execution works from an
immutable source bundle with no `.git` directory.

The separately labeled reachable overflow witness is never included in
representative statistics.

## Verify

```bash
cargo fmt --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- --check
cargo test --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml
cargo clippy --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml \
  --all-targets --no-deps -- -D warnings
```

## Local Smoke

```bash
root=/tmp/corrected-mid-tail-activation-census-v1
rm -rf "$root"

for shard in 0 1 2 3; do
  cargo run --release \
    --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- \
    generate-shard \
    --output-root "$root/corpus/shard-$shard" \
    --shard-index "$shard" \
    --shard-count 4 \
    --first-game-index 0 \
    --total-games 4

  cargo run --release \
    --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- \
    census-shard \
    --corpus-root "$root/corpus/shard-$shard" \
    --output "$root/report-$shard.json"
done

cargo run --release \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- \
  aggregate \
  --report "$root/report-0.json" \
  --report "$root/report-1.json" \
  --report "$root/report-2.json" \
  --report "$root/report-3.json" \
  --require-shards 4 \
  --output "$root/aggregate-forward.json"

cargo run --release \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- \
  aggregate \
  --report "$root/report-3.json" \
  --report "$root/report-2.json" \
  --report "$root/report-1.json" \
  --report "$root/report-0.json" \
  --require-shards 4 \
  --output "$root/aggregate-reverse.json"

cargo run --release \
  --manifest-path tools/f5_corrected_tail_activation_census/Cargo.toml -- \
  verify-order \
  --left "$root/aggregate-forward.json" \
  --right "$root/aggregate-reverse.json"
```
