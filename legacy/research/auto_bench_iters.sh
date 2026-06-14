#!/bin/bash
# Auto-bench new iter files as they appear from train_hybrid.py.
# Watches for nnue_weights_hybrid_iter21+.bin and runs NNUE-only 200g bench on each.

cd /Users/johnherrick/cascadia
mkdir -p iter_history

# Track which iters have been benched
state=$(mktemp)
for f in iter_history/nnue_iter*_200g.log; do
    [ -f "$f" ] && grep -q "Mean:" "$f" 2>/dev/null && basename "$f" .log | sed 's/nnue_iter//; s/_200g//' >> "$state"
done

while true; do
    for f in nnue_weights_hybrid_iter*.bin; do
        [ -f "$f" ] || continue
        iter=$(basename "$f" .bin | sed 's/nnue_weights_hybrid_iter//')
        # Only bench iter21+
        if [ "$iter" -ge 21 ] 2>/dev/null && ! grep -qFx "$iter" "$state"; then
            outfile="iter_history/nnue_iter${iter}_200g.log"
            echo "[$(date +%H:%M:%S)] auto-benching iter$iter..."
            ./target/release/cascadia-cli 200 --nnue --weights "$f" > "$outfile" 2>&1
            mean=$(grep -m1 "Mean:" "$outfile" | awk '{print $2}')
            echo "[$(date +%H:%M:%S)] iter$iter mean=$mean"
            echo "$iter" >> "$state"
        fi
    done
    sleep 30
done
