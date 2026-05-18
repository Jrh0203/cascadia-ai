#!/bin/bash
# Run this when you wake up to see what completed overnight.
# Usage: ./wakeup_status.sh

cd "$(dirname "$0")"

echo "=== Wakeup status — $(date) ==="
echo
echo "Running cascadia processes:"
ps aux | grep "target/release/cascadia-cli " | grep -v grep | awk '{printf "  PID %s  CPU %s%%  cmd %s\n", $2, $3, substr($0, index($0, $11))}'
echo

echo "=== Completed benches (sorted by mean) ==="
./generate_report.sh
echo

echo "=== Phase queue logs ==="
for log in bench_runner.log phase3_runner.log phase4_runner.log phase5_runner.log; do
    if [ -f "$log" ]; then
        echo "--- $log ---"
        cat "$log"
    fi
done
echo

echo "=== mce_policy_samples.bin ==="
ls -la mce_policy_samples.bin 2>/dev/null
echo

echo "=== Recommended next actions ==="
echo "1. Read WAKEUP_REPORT.md for full analysis"
echo "2. Read overnight_results.md for per-experiment details"
echo "3. Check git diff to see code changes"
echo "4. If LEAF_EXPECTIMAX is confirmed, enable it as default in run_overnight_benches.sh"
