#!/usr/bin/env bash
#
# The way out that does not depend on the browser, the page, or the keyboard.
#
# Run it from a terminal, over SSH, or from a text console reached with
# Ctrl+Alt+F2. Every escape hatch that lives inside the kiosk assumes the
# browser is alive and the page is loaded; this one assumes nothing.
#
#   ./deploy/stop-kiosk.sh            close the display, keep detecting
#   ./deploy/stop-kiosk.sh --all      stop everything, sensor included

set -uo pipefail

RUN_DIR="${XDG_RUNTIME_DIR:-/tmp}/astravigil"
PORT="${ASTRAVIGIL_PORT:-8000}"
ALL=0
[ "${1:-}" = "--all" ] && ALL=1

say() { printf '%s\n' "$*"; }

# --- the display
killed=0
if [ -f "$RUN_DIR/browser.pid" ]; then
    pid="$(cat "$RUN_DIR/browser.pid" 2>/dev/null)"
    if [ -n "$pid" ] && kill "$pid" 2>/dev/null; then
        say "closed the kiosk browser (pid $pid)"
        killed=1
    fi
    rm -f "$RUN_DIR/browser.pid"
fi
if [ "$killed" -eq 0 ]; then
    # No pid file, or a stale one. Fall back to the profile directory, which
    # is unique to this kiosk - so this cannot hit somebody's own browsing.
    if pkill -f "user-data-dir=${RUN_DIR}/chrome-profile" 2>/dev/null; then
        say "closed the kiosk browser by profile match"
    else
        say "no kiosk browser was running"
    fi
fi

# --- the sensor
if [ "$ALL" -eq 1 ]; then
    if [ -f "$RUN_DIR/dashboard.pid" ]; then
        pid="$(cat "$RUN_DIR/dashboard.pid" 2>/dev/null)"
        [ -n "$pid" ] && kill "$pid" 2>/dev/null && say "stopped the supervisor (pid $pid)"
        rm -f "$RUN_DIR/dashboard.pid"
    fi
    pkill -f "scripts/run_dashboard.py" 2>/dev/null && say "stopped the pipeline"
    say ""
    say "Everything stopped. The site is no longer being watched."
else
    if curl -fsS --max-time 2 "http://localhost:${PORT}/api/state" >/dev/null 2>&1; then
        say ""
        say "The sensor is STILL RUNNING and still detecting."
        say "  console : http://localhost:${PORT}"
        say "  stop it : $0 --all"
    fi
fi
