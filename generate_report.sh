#!/bin/bash
# Generate the WAKEUP_REPORT.md from bench_results/ files.
# Reads each completed bench, builds a comparison table, sorts by mean.

cd "$(dirname "$0")"

REPORT=WAKEUP_REPORT.md

# Build a sortable list of completed benches
TMPFILE=$(mktemp)
for f in bench_results/*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    if grep -q "Mean:" "$f" 2>/dev/null; then
        mean=$(grep -m1 "Mean:" "$f" | awk '{print $2}')
        median=$(grep "Median:" "$f" | head -1 | awk '{print $2}')
        p10=$(grep -m1 "P10:" "$f" | awk '{print $2}')
        p90=$(grep -m1 "P90:" "$f" | awk '{print $2}')
        max=$(grep -m1 "Min/Max:" "$f" | awk '{print $2}' | tr / ' ' | awk '{print $2}')
        time=$(grep -m1 "Results" "$f" | sed 's/.*in //; s/,.*//')
        bear=$(grep -A1 "Bear" "$f" | head -1 | awk '{print $2}')
        elk=$(grep "Elk" "$f" | head -1 | awk '{print $2}')
        salmon=$(grep "Salmon" "$f" | head -1 | awk '{print $2}')
        hawk=$(grep "Hawk" "$f" | head -1 | awk '{print $2}')
        fox=$(grep "Fox" "$f" | head -1 | awk '{print $2}')
        n=$(grep -m1 "Results" "$f" | awk '{print $2}' | tr -d '(')
        echo "$mean|$name|$median|$p10|$p90|$max|$time|$n|$bear|$elk|$salmon|$hawk|$fox" >> "$TMPFILE"
    fi
done

# Sort by mean descending
sort -t'|' -k1 -nr "$TMPFILE" > "${TMPFILE}.sorted"

# Generate markdown table
echo "## Results table (sorted by mean)"
echo
echo "| Strategy | Mean | Med | P10 | P90 | Max | Time | N | Bear | Elk | Salm | Hawk | Fox |"
echo "|---|---|---|---|---|---|---|---|---|---|---|---|---|"
while IFS='|' read -r mean name median p10 p90 max time n bear elk salmon hawk fox; do
    echo "| $name | **$mean** | $median | $p10 | $p90 | $max | $time | $n | $bear | $elk | $salmon | $hawk | $fox |"
done < "${TMPFILE}.sorted"

# Show in-progress benches
echo
echo "## In-progress benches"
echo
for f in bench_results/*.log; do
    [ -f "$f" ] || continue
    name=$(basename "$f" .log)
    if ! grep -q "Mean:" "$f" 2>/dev/null; then
        echo "- $name"
    fi
done

rm -f "$TMPFILE" "${TMPFILE}.sorted"
