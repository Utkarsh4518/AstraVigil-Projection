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

# Machine-local settings, read before anything below is decided so this file
# can set any of them: FEATHERLESS_API_KEY, ASTRAVIGIL_ARGS, ASTRAVIGIL_PORT,
# ASTRAVIGIL_THERMAL_ROT.
#
# A desktop launcher inherits the session's environment and nothing else. It
# does not read ~/.bashrc - that is for interactive shells - and ~/.profile is
# only sourced for the graphical session by some display managers, so "export
# it in your shell" is not an answer that survives a double-click. Secrets do
# not belong in the .desktop file either: install_kiosk.sh rewrites it, and it
# is world-readable.
#
#   printf 'FEATHERLESS_API_KEY=sk-...\n' > ~/.astravigil.env
#   chmod 600 ~/.astravigil.env
ENV_FILE="${ASTRAVIGIL_ENV:-$HOME/.astravigil.env}"
if [ -f "$ENV_FILE" ]; then
    # set -a exports every assignment, so the pipeline inherits them.
    set -a
    # shellcheck source=/dev/null
    . "$ENV_FILE"
    set +a
fi

PORT="${ASTRAVIGIL_PORT:-8000}"
URL="http://localhost:${PORT}/?kiosk=1"
RESTART_EXIT_CODE=42
RUN_DIR="${XDG_RUNTIME_DIR:-/tmp}/astravigil"
LOG="${HOME}/.astravigil-kiosk.log"
# shellcheck source=_teardown.sh
. "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")/_teardown.sh"
LOCK="${RUN_DIR}/kiosk.lock"
BROWSER_PID_FILE="${RUN_DIR}/browser.pid"
DASH_PID_FILE="${RUN_DIR}/dashboard.pid"
LAUNCHER_PID_FILE="${RUN_DIR}/launcher.pid"

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

# Single instance: double-clicking the icon twice must not race two pipelines
# onto one camera.
#
# But "another instance holds the lock" is not the same as "there is nothing
# to do". The normal state after an escape is pipeline still running - on
# purpose, a perimeter sensor should not go blind because somebody wanted the
# desktop back - with the console closed. Double-clicking the icon then means
# "give me the console back", and answering "already running" is useless: it
# names the state instead of doing the obvious thing about it.
already_up() { curl -fsS --max-time 2 "http://localhost:${PORT}/api/state" >/dev/null 2>&1; }
browser_alive() {
    local pid
    pid="$(cat "$BROWSER_PID_FILE" 2>/dev/null)" || return 1
    [ -n "$pid" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    # kill -0 only proves *something* owns that number. PIDs are recycled, and
    # a stale file that happens to name a live process is how the launcher ends
    # up insisting a console is open that is not on the screen.
    [ -r "/proc/$pid/cmdline" ] || return 0
    grep -qa "chrome-profile" "/proc/$pid/cmdline" 2>/dev/null
}

# Is another *launcher* actually alive? Neither the lock nor a pid file can
# answer that alone: the lock says only that some descriptor somewhere holds
# it, and a pid file outlives the process that wrote it.
launcher_alive() {
    local pid
    pid="$(cat "$LAUNCHER_PID_FILE" 2>/dev/null)" || return 1
    [ -n "$pid" ] && [ "$pid" != "$$" ] || return 1
    kill -0 "$pid" 2>/dev/null || return 1
    [ -r "/proc/$pid/cmdline" ] || return 0
    grep -qa "astravigil-kiosk" "/proc/$pid/cmdline" 2>/dev/null
}

mkdir -p "$RUN_DIR"

# The guard has to fail OPEN. Refusing to start is the expensive answer: from
# the desktop it is indistinguishable from a dead icon, and the operator has
# nothing to act on. So it is only ever given when another launcher is
# demonstrably alive - never because a tool is missing, and never because a
# lock outlived the process that took it.
HELD_LOCK=0
release_lock() {
    if [ "$HELD_LOCK" -eq 1 ]; then
        HELD_LOCK=0
        # Closing the descriptor is what releases the flock. The file stays -
        # see the note in _teardown.sh about why unlinking it is worse.
        exec 9>&-
    fi
    [ "$(cat "$LAUNCHER_PID_FILE" 2>/dev/null)" = "$$" ] && rm -f "$LAUNCHER_PID_FILE"
    return 0
}

exec 9>"$LOCK"
CONTENDED=0
if command -v flock >/dev/null 2>&1; then
    if flock -n 9; then HELD_LOCK=1; else CONTENDED=1; fi
else
    # util-linux is not installed. Without this branch `! flock -n 9` is a
    # command-not-found - non-zero, and therefore read as "the lock is held" -
    # which sends every launch, including the very first one on a fresh
    # machine, down the "still starting" path. The icon then never opens and
    # never explains itself.
    log "flock is not installed - falling back to a pid check"
    if launcher_alive; then CONTENDED=1; else HELD_LOCK=1; fi
fi
[ "$HELD_LOCK" -eq 1 ] && echo $$ >"$LAUNCHER_PID_FILE"
trap release_lock EXIT

REATTACH=0
if [ "$CONTENDED" -eq 1 ]; then
    if browser_alive; then
        log "console already open - nothing to do"
        notify "The AstraVigil console is already open.

If you cannot see it, it may be on another virtual console - try Ctrl+Alt+F7.
To close it:  $REPO/deploy/stop-kiosk.sh"
        exit 0
    fi
    if already_up; then
        # Sensor up, console closed. Reopen the console and nothing else: the
        # lock exists to stop two pipelines fighting over one camera, and
        # opening a browser does not do that.
        log "pipeline already running - reopening the console only"
        REATTACH=1
    elif launcher_alive; then
        HOLDER="$(cat "$LAUNCHER_PID_FILE" 2>/dev/null)"
        log "launcher $HOLDER is mid-startup - leaving it to finish"
        notify "AstraVigil is still starting (pid $HOLDER).

Give it up to 90 seconds. If the console still does not appear:
    $REPO/deploy/stop-kiosk.sh
and then double-click the icon again.

Log: $LOG"
        exit 0
    else
        # Nothing listening, no console, and no launcher to wait for: the lock
        # outlived whatever held it. This is the state that used to answer
        # "still starting" on every double-click, for ever - an unverified
        # claim about a process that no longer existed.
        log "stale lock, no live launcher - taking it over"
        rm -f "$LAUNCHER_PID_FILE"
        if command -v flock >/dev/null 2>&1; then
            exec 9>&-
            exec 9>"$LOCK"
            if flock -n 9; then HELD_LOCK=1; else
                log "lock still held by an orphaned descriptor - proceeding anyway"
            fi
        else
            HELD_LOCK=1
        fi
        echo $$ >"$LAUNCHER_PID_FILE"
    fi
fi

# Set by start_dashboard, so a failed start can take back what it started.
DASH_SUPERVISOR_PID=""
stop_dashboard() {
    [ -n "$DASH_SUPERVISOR_PID" ] || return 0
    # Supervisor first, or it will helpfully restart the pipeline we are about
    # to stop. TERM only, never -9: the pipeline is holding the camera.
    kill "$DASH_SUPERVISOR_PID" 2>/dev/null
    DASH_SUPERVISOR_PID=""
    pkill -TERM -f "scripts/run_dashboard.py --port $PORT" 2>/dev/null
    rm -f "$DASH_PID_FILE"
    return 0
}

die() {
    log "FAILED: $1"
    # Give back the pipeline and the lock BEFORE the dialog. notify() is modal
    # and blocks until somebody clicks OK - and it used to block while still
    # holding the lock, so one failed start turned every later double-click
    # into "AstraVigil is still starting", permanently, with the dialog that
    # explained why sitting unread behind the fullscreen splash.
    stop_dashboard
    release_lock
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

# Every readiness check in this script is a curl. Without it the pipeline is
# never seen to come up, however well it is running, and the launcher waits
# the full 90 s before failing for a reason that has nothing to do with the
# sensor.
command -v curl >/dev/null 2>&1 || die "curl is not installed. Install it with:
    sudo apt install -y curl"

log "repo=$REPO python=$PY browser=$BROWSER port=$PORT"

# Say something immediately. Starting the pipeline and warming the camera can
# take tens of seconds on a Pi 4, and a desktop icon that produces no visible
# response gets double-clicked again - which the lock then refuses with a
# confusing "already running".
if [ "$REATTACH" -eq 1 ]; then
    splash "Reopening the console - the sensor never stopped." 6000
else
    splash "Starting the sensor - the console will open fullscreen shortly." 12000
fi

# ------------------------------------------------------------- dashboard
# Supervised in a subshell: if the console asks to restart, the pipeline exits
# with RESTART_EXIT_CODE and this brings it straight back up.
start_dashboard() {
    (
        # Do NOT inherit the lock. A background subshell holding fd 9 keeps the
        # single-instance lock alive long after the launcher has exited, so the
        # next double-click finds a locked file with nothing behind it.
        exec 9>&-
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
    DASH_SUPERVISOR_PID=$!
    echo "$DASH_SUPERVISOR_PID" >"$DASH_PID_FILE"
}

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
    # Teardown has usually already run by here; this only catches the paths
    # that did not go through it (a crash, or the browser being closed some
    # other way). Never -9 the pipeline here: it is holding the camera.
    kill "$BROWSER_PID" 2>/dev/null
    rm -f "$BROWSER_PID_FILE"
    release_lock
}
trap cleanup EXIT
# A bare `trap cleanup TERM` runs cleanup and then carries on round the
# watchdog loop, so stop-kiosk.sh had to escalate to SIGKILL to be rid of a
# launcher that had already tidied up after itself.
trap 'cleanup; exit 0' INT TERM

# ------------------------------------------------------------- watchdog
# Poll for the escape sequence. The page cannot close a kiosk window itself -
# window.close() is refused for windows the script did not open - so it raises
# a flag and this loop does the closing.
while kill -0 "$BROWSER_PID" 2>/dev/null; do
    if curl -fsS --max-time 2 "http://localhost:${PORT}/api/kiosk/status" 2>/dev/null \
        | grep -q '"exit_requested"[[:space:]]*:[[:space:]]*true'; then
        log "escape sequence accepted - stopping everything"
        curl -fsS -X POST --max-time 2 "http://localhost:${PORT}/api/kiosk/ack" >/dev/null 2>&1
        # Full stop, not just the display. Leaving the sensor running is the
        # safer default for a fixed installation, but it left this rig in a
        # half-state that was hard to reason about and hard to restart. A
        # demo rig wants the icon to be the only control there is.
        astravigil_teardown keep-launcher 2>&1 | while IFS= read -r l; do log "$l"; done
        # Before the modal, not after: it blocks until somebody clicks OK, and
        # the whole point of the message is that the icon is ready to go again.
        release_lock
        notify "AstraVigil stopped.

Everything has been shut down and the camera released.
Double-click the AstraVigil icon to start a fresh run."
        break
    fi
    sleep 1
done

log "kiosk finished"
