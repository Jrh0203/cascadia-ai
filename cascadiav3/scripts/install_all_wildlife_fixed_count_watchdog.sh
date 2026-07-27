#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
LABEL="com.johnherrick.cascadia.all-wildlife-fixed-count-watchdog"
AGENT_DIR="${HOME}/Library/LaunchAgents"
PLIST="${AGENT_DIR}/${LABEL}.plist"
PYTHON="${ROOT}/.venv/bin/python"
WATCHDOG="${ROOT}/tools/all_wildlife_fixed_count_watchdog.py"
WATCHDOG_RUNNER="${ROOT}/cascadiav3/scripts/run_all_wildlife_fixed_count_watchdog.sh"
CONFIG="${ROOT}/cascadiav3/fleet/all_wildlife_fixed_count_pipeline_20260726.json"
STDOUT_LOG="${ROOT}/cascadiav3/logs/all_wildlife_fixed_count_watchdog.launchd.log"
STDERR_LOG="${ROOT}/cascadiav3/logs/all_wildlife_fixed_count_watchdog.launchd.err"

test -x "$PYTHON"
test -f "$WATCHDOG"
test -x "$WATCHDOG_RUNNER"
test -f "$CONFIG"
mkdir -p "$AGENT_DIR" "${ROOT}/cascadiav3/logs"

temporary="$(mktemp "${AGENT_DIR}/.${LABEL}.XXXXXX")"
cleanup() {
  rm -f "$temporary"
}
trap cleanup EXIT

/usr/bin/plutil -create xml1 "$temporary"
/usr/libexec/PlistBuddy -c "Add :Label string $LABEL" "$temporary"
/usr/libexec/PlistBuddy -c "Add :ProgramArguments array" "$temporary"
/usr/libexec/PlistBuddy -c "Add :ProgramArguments:0 string $PYTHON" "$temporary"
/usr/libexec/PlistBuddy -c "Add :ProgramArguments:1 string $WATCHDOG" "$temporary"
/usr/libexec/PlistBuddy -c "Add :ProgramArguments:2 string --config" "$temporary"
/usr/libexec/PlistBuddy -c "Add :ProgramArguments:3 string $CONFIG" "$temporary"
/usr/libexec/PlistBuddy -c "Add :WorkingDirectory string $ROOT" "$temporary"
/usr/libexec/PlistBuddy -c "Add :StartInterval integer 3600" "$temporary"
/usr/libexec/PlistBuddy -c "Add :RunAtLoad bool true" "$temporary"
/usr/libexec/PlistBuddy -c "Add :ProcessType string Background" "$temporary"
/usr/libexec/PlistBuddy -c "Add :ThrottleInterval integer 60" "$temporary"
/usr/libexec/PlistBuddy -c "Add :StandardOutPath string $STDOUT_LOG" "$temporary"
/usr/libexec/PlistBuddy -c "Add :StandardErrorPath string $STDERR_LOG" "$temporary"
/usr/bin/plutil -lint "$temporary"
mv "$temporary" "$PLIST"
trap - EXIT

uid="$(id -u)"
if /bin/launchctl print "gui/${uid}" >/dev/null 2>&1; then
  domain="gui/${uid}"
else
  domain="user/${uid}"
fi
/bin/launchctl bootout "${domain}/${LABEL}" >/dev/null 2>&1 || true
if /bin/launchctl bootstrap "$domain" "$PLIST" >/dev/null 2>&1; then
  /bin/launchctl enable "${domain}/${LABEL}"
  /bin/launchctl kickstart -k "${domain}/${LABEL}"
  printf 'installed %s in %s with a 3600-second interval\n' "$LABEL" "$domain"
  exit 0
fi

# Headless/background macOS sessions do not always expose a bootstrap domain
# that accepts per-user LaunchAgents. Cron is managed by the system launchd
# daemon and remains available in that case.
cron_marker="cascadia-all-wildlife-fixed-count-watchdog"
cron_temporary="$(mktemp "${TMPDIR:-/tmp}/${cron_marker}.XXXXXX")"
cleanup_cron() {
  rm -f "$cron_temporary"
}
trap cleanup_cron EXIT
(crontab -l 2>/dev/null || true) \
  | awk -v marker="$cron_marker" 'index($0, marker) == 0' \
  > "$cron_temporary"
printf '7 * * * * %s >> %s 2>&1 # %s\n' \
  "$WATCHDOG_RUNNER" "$STDOUT_LOG" "$cron_marker" >> "$cron_temporary"
crontab "$cron_temporary"
trap - EXIT
rm -f "$cron_temporary"
printf 'installed %s in crontab with an hourly interval\n' "$LABEL"
