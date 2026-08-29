#!/usr/bin/env bash
#
# Stop AstraVigil completely: processes, port, camera, lock.
#
# Run it from a terminal, over SSH, or from a text console reached with
# Ctrl+Alt+F2. Every escape hatch inside the kiosk assumes the browser is
# alive and the page is loaded; this one assumes nothing.
#
#   ./deploy/stop-kiosk.sh              stop everything
#   ./deploy/stop-kiosk.sh --display    close the console, keep detecting
#
# Default is a full stop. The console escape does the same thing, so after
# either one the icon starts a fresh run with nothing left over.

set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
RUN_DIR="${XDG_RUNTIME_DIR:-/tmp}/astravigil"
PORT="${ASTRAVIGIL_PORT:-8000}"
PY="${ASTRAVIGIL_PY:-python3}"

# shellcheck source=_teardown.sh
. "$REPO/deploy/_teardown.sh"

if [ "${1:-}" = "--display" ]; then
    if [ -f "$RUN_DIR/browser.pid" ]; then
        kill -TERM "$(cat "$RUN_DIR/browser.pid" 2>/dev/null)" 2>/dev/null
        rm -f "$RUN_DIR/browser.pid"
    fi
    pkill -f "user-data-dir=${RUN_DIR}/chrome-profile" 2>/dev/null
    echo "console closed - the sensor is still running on port $PORT"
    exit 0
fi

astravigil_teardown
