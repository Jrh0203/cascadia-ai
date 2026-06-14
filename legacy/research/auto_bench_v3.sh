#!/bin/bash
# Auto-bench v3 iter weights after they're FULLY trained.
#
# Strategy: when iter(N+1) appears, bench iter(N). This guarantees iter(N) is
# fully written (not mid-epoch checkpoint).
#
# Output: bench_results/v3_iterN_nnue_100g.log

cd /Users/johnherrick/cascadia
mkdir -p bench_results

state=$(mktemp)
# Pre-mark any iters that already have benches
for f in bench_results/v3_iter*_nnue_100g.log; do
    [ -f "$f" ] && grep -q "Mean:" "$f" 2>/dev/null && \
        basename "$f" .log | sed 's/v3_iter//; s/_nnue_100g//' >> "$state"
done

while true; do
    # Find max iter file present
    max_iter=0
    for f in nnue_weights_v3_iter*.bin; do
        [ -f "$f" ] || continue
        n=$(basename "$f" .bin | sed 's/nnue_weights_v3_iter//')
        if [ "$n" -gt "$max_iter" ] 2>/dev/null; then
            max_iter=$n
        fi
    done

    # Bench any iter < max_iter that hasn't been benched yet
    # (max_iter is the one currently being trained, so it's mid-checkpoint)
    for ((iter=1; iter < max_iter; iter++)); do
        wfile="nnue_weights_v3_iter${iter}.bin"
        [ -f "$wfile" ] || continue
        if ! grep -qFx "$iter" "$state" 2>/dev/null; then
            outfile="bench_results/v3_iter${iter}_nnue_100g.log"
            echo "[$(date +%H:%M:%S)] auto-benching v3 iter${iter}..."
            ./target/release/cascadia-cli 100 --nnue --weights "$wfile" > "$outfile" 2>&1
            if grep -q "Mean:" "$outfile"; then
                mean=$(grep -m1 "Mean:" "$outfile" | awk '{print $2}')
                echo "[$(date +%H:%M:%S)] v3 iter${iter} mean=$mean"
                echo "$iter" >> "$state"
            else
                echo "[$(date +%H:%M:%S)] v3 iter${iter} bench FAILED — will retry next loop"
            fi
        fi
    done
    sleep 30
done
