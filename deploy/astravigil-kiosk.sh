#!/usr/bin/env bash
#
# AstraVigil operator console - start the sensor, then take over the screen.
#
# Double-clicked from the desktop icon. Starts the detection pipeline, waits
# for it to actually answer, and only then goes fullscreen. That order matters:
# a kiosk browser that launches into a dead page is a black rectangle with no
# address bar and no obvious way out, which is exactly the situation you do not
# want to discover in front of an audience.
#
# ESCAPING
#   Ctrl+Shift+Esc three times within three seconds  (on-screen counter)
#   deploy/stop-kiosk.sh                             (over SSH, or a terminal)
#   Ctrl+Alt+F2                                      (switch to a text console)
#
# The escape sequence releases the DISPLAY only. The sensor keeps detecting -
# a perimeter system should never go blind because somebody wanted the desktop.

set -uo pipefail

REPO="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/.." && pwd)"
PORT="${ASTRAVIGIL_PORT:-8000}"
URL="http://localhost:${PORT}/?kiosk=1"
RESTART_EXIT_CODE=42
RUN_DIR="${XDG_RUNTIME_DIR:-/tmp}/astravigil"
LOG="${HOME}/.astravigil-kiosk.log"
LOCK="${RUN_DIR}/kiosk.lock"
BROWSER_PID_FILE="${RUN_DIR}/browser.pid"
DASH_PID_FILE="${RUN_DIR}/dashboard.pid"

# Extra arguments for run_dashboard.py. Override in the .desktop file or the
# environment, e.g. ASTRAVIGIL_ARGS="--source hardware --site-model data/baseline/site.npz"
ARGS="${ASTRAVIGIL_ARGS:---source hardware --calibration data/calibration/H.json --site-model data/baseline/site.npz --save-site}"

log() { printf '%s  %s\n' "$(date '+%F %T')" "$*" | tee -a "$LOG"; }

# Modal. Use for things the operator must actually read - failures, and the
# note that the sensor is still running after an escape.
notify() {
    if command -v zenity >/dev/null 2>&1; then
        zenity --info --no-wrap --title=AstraVigil --text="$1" 2>/dev/null
        return 0
    fi
    if command -v notify-send >/dev/null 2>&1; then
        notify-send "AstraVigil" "$1" 2>/dev/null
        return 0
    fi
    log "$1"
}

# Non-blocking. MUST NOT be zenity --info, which waits for someone to click OK
# and would stall the launch behind a dialog nobody is watching.
splash() {
    if command -v notify-send >/dev/null 2>&1; then
        notify-send -t "${2:-8000}" "AstraVigil" "$1" 2>/dev/null
    elif command -v zenity >/dev/null 2>&1; then
        (zenity --notification --text="AstraVigil: $1" 2>/dev/null &)
    fi
    log "$1"
}

# Single instance, checked after the helpers above because it uses them.
# Double-clicking the icon twice must not race two pipelines onto one camera.
mkdir -p "$RUN_DIR"
exec 9>"$LOCK"
if ! flock -n 9; then
    log "already running - not starting a second instance"
    notify "AstraVigil is already running."
    exit 0
fi

die() {
    log "FAILED: $1"
    notify "AstraVigil could not start.

$1

Log: $LOG"
    exit 1
}

# ---------------------------------------------------------------- python
PY=""
for cand in "$REPO/.venv/bin/python" "$REPO/venv/bin/python" python3 python; do
    if command -v "$cand" >/dev/null 2>&1 || [ -x "$cand" ]; then PY="$cand"; break; fi
done
[ -n "$PY" ] || die "No python interpreter found."

# --------------------------------------------------------------- browser
BROWSER=""
for cand in chromium-browser chromium google-chrome chromium-browser-v7; do
    command -v "$cand" >/dev/null 2>&1 && { BROWSER="$cand"; break; }
done
[ -n "$BROWSER" ] || die "No Chromium found. Install it with:
    sudo apt install -y chromium-browser"

log "repo=$REPO python=$PY browser=$BROWSER port=$PORT"

# Say something immediately. Starting the pipeline and warming the camera can
# take tens of seconds on a Pi 4, and a desktop icon that produces no visible
# response gets double-clicked again - which the lock then refuses with a
# confusing "already running".
splash "Starting the sensor - the console will open fullscreen shortly." 12000

# ------------------------------------------------------------- dashboard
# Supervised in a subshell: if the console asks to restart, the pipeline exits
# with RESTART_EXIT_CODE and this brings it straight back up.
start_dashboard() {
    (
        cd "$REPO" || exit 1
        while true; do
            log "starting pipeline: $PY scripts/run_dashboard.py --port $PORT $ARGS"
            # shellcheck disable=SC2086
            "$PY" scripts/run_dashboard.py --port "$PORT" $ARGS >>"$LOG" 2>&1
            code=$?
            if [ "$code" -eq "$RESTART_EXIT_CODE" ]; then
                log "pipeline asked to restart"
                sleep 1
                continue
            fi
            log "pipeline exited with $code"
            break
        done
    ) &
    echo $! >"$DASH_PID_FILE"
}

already_up() { curl -fsS --max-time 2 "http://localhost:${PORT}/api/state" >/dev/null 2>&1; }

if already_up; then
    log "pipeline already listening on $PORT - attaching to it"
else
    start_dashboard
fi

# Wait for it to actually answer. NOT a fixed sleep: on a cold Pi 4 the first
# frame can take a while, and going fullscreen early is how you end up staring
# at a connection error with no address bar.
log "waiting for the pipeline to answer on $PORT"
for _ in $(seq 1 90); do
    already_up && break
    sleep 1
done
already_up || die "The pipeline did not start listening on port $PORT within 90 s.

Check the log, then try running it by hand:
    cd $REPO
    $PY scripts/run_dashboard.py --port $PORT $ARGS"

log "pipeline is up"

# --------------------------------------------------------- screen comfort
# All optional; failures are ignored because the display server may be X11 or
# Wayland and these tools may simply not be installed.
xset s off        2>/dev/null
xset -dpms        2>/dev/null
xset s noblank    2>/dev/null
command -v unclutter >/dev/null 2>&1 && (unclutter -idle 0 >/dev/null 2>&1 &)

# ---------------------------------------------------------------- kiosk
PROFILE="${RUN_DIR}/chrome-profile"
mkdir -p "$PROFILE"
# Clear crash flags, or Chromium opens a "restore pages?" bar that a kiosk
# user cannot dismiss and that covers the top of the console.
sed -i 's/"exit_type":"Crashed"/"exit_type":"Normal"/' \
    "$PROFILE/Default/Preferences" 2>/dev/null

log "opening $URL"
"$BROWSER" \
    --kiosk "$URL" \
    --user-data-dir="$PROFILE" \
    --start-fullscreen \
    --noerrdialogs \
    --disable-infobars \
    --disable-session-crashed-bubble \
    --disable-features=TranslateUI,Translate \
    --disable-translate \
    --disable-pinch \
    --overscroll-history-navigation=0 \
    --autoplay-policy=no-user-gesture-required \
    --check-for-update-interval=31536000 \
    --password-store=basic \
    >>"$LOG" 2>&1 &
BROWSER_PID=$!
echo "$BROWSER_PID" >"$BROWSER_PID_FILE"
log "browser pid $BROWSER_PID"

cleanup() {
    kill "$BROWSER_PID" 2>/dev/null
    rm -f "$BROWSER_PID_FILE"
}
trap cleanup EXIT INT TERM

# ------------------------------------------------------------- watchdog
# Poll for the escape sequence. The page cannot close a kiosk window itself -
# window.close() is refused for windows the script did not open - so it raises
# a flag and this loop does the closing.
while kill -0 "$BROWSER_PID" 2>/dev/null; do
    if curl -fsS --max-time 2 "http://localhost:${PORT}/api/kiosk/status" 2>/dev/null \
        | grep -q '"exit_requested"[[:space:]]*:[[:space:]]*true'; then
        log "escape sequence accepted - closing the display"
        curl -fsS -X POST --max-time 2 "http://localhost:${PORT}/api/kiosk/ack" >/dev/null 2>&1
        kill "$BROWSER_PID" 2>/dev/null
        sleep 1
        notify "Console released.

The sensor is STILL RUNNING and still detecting.
  Reopen it:  double-click the AstraVigil icon
  Stop it:    $REPO/deploy/stop-kiosk.sh"
        break
    fi
    sleep 1
done

log "kiosk finished"
