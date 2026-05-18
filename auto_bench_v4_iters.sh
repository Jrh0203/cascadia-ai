#!/bin/bash
# Watch for new v4 iterN weights and bench each as it appears.
# Runs in background, low priority (nice 15).
# Output: bench_results/v4_iter*_nnue_200g.log
# Stops when v4 training process exits AND all iterN weights are benched.

set -e
mkdir -p bench_results

bench_iter() {
    local n=$1
    local weights="nnue_weights_v4_iter${n}.bin"
    local log="bench_results/v4_iter${n}_nnue_200g.log"
    if [ -f "$log" ] && grep -q "Mean:" "$log"; then
        return 0
    fi
    if [ ! -f "$weights" ]; then
        return 1
    fi
    # Skip if a cascadia-cli is already benching this exact weight file
    if pgrep -f "cascadia-cli .* --weights ${weights}\$" > /dev/null 2>&1 || \
       pgrep -f "cascadia-cli .* --weights ${weights} " > /dev/null 2>&1; then
        return 0
    fi
    echo "[$(date +%H:%M:%S)] Benching v4 iter${n} NNUE 200g..."
    nice -n 15 ./target/release/cascadia-cli 200 --nnue --weights "$weights" \
        > "$log" 2>&1
    local mean=$(grep -m1 "Mean:" "$log" | awk '{print $2}')
    echo "[$(date +%H:%M:%S)] v4 iter${n} NNUE 200g: $mean"
}

echo "[$(date +%H:%M:%S)] auto_bench_v4_iters: watching for new iter weights"

while true; do
    # Bench any iter weights that exist but aren't benched yet
    for n in 1 2 3 4 5 6 7 8 9 10; do
        bench_iter "$n" || true
    done

    # Stop if training has exited and all existing iters are benched
    if ! pgrep -f "train_hybrid.py.*v4" > /dev/null; then
        # One last sweep
        for n in 1 2 3 4 5 6 7 8 9 10; do
            bench_iter "$n" || true
        done
        echo "[$(date +%H:%M:%S)] auto_bench_v4_iters: training done, exiting"
        break
    fi

    sleep 60
done
