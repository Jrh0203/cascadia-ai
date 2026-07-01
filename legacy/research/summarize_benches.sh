#!/bin/bash
# Summarize bench results into a comparison table.
# Reads bench_results/*.log files and prints mean/median/p10/p90/time/wildlife.

cd "$(dirname "$0")"

printf "%-25s %6s %6s %6s %6s %12s %4s %4s %4s %4s %4s\n" \
    NAME MEAN MED P10 P90 TIME BR EL SA HK FX
echo "------------------------------------------------------------------------------------------------------"

for f in bench_results/*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    grep -q "Mean:" "$f" 2>/dev/null || { printf "%-25s [running...]\n" "$name"; continue; }
    mean=$(grep -m1 "Mean:" "$f" | awk '{print $2}')
    median=$(grep "Median:" "$f" | head -1 | awk '{print $2}')
    p10=$(grep -m1 "P10:" "$f" | awk '{print $2}')
    p90=$(grep -m1 "P90:" "$f" | awk '{print $2}')
    time=$(grep -m1 "Results" "$f" | sed 's/.*in //; s/,.*//')
    bear=$(grep "Bear" "$f" | head -1 | awk '{print $2}')
    elk=$(grep "Elk" "$f" | head -1 | awk '{print $2}')
    salmon=$(grep "Salmon" "$f" | head -1 | awk '{print $2}')
    hawk=$(grep "Hawk" "$f" | head -1 | awk '{print $2}')
    fox=$(grep "Fox" "$f" | head -1 | awk '{print $2}')
    printf "%-25s %6s %6s %6s %6s %12s %4s %4s %4s %4s %4s\n" \
        "$name" "$mean" "$median" "$p10" "$p90" "$time" "$bear" "$elk" "$salmon" "$hawk" "$fox"
done | sort -k2 -r
