# Shared full teardown. Sourced by astravigil-kiosk.sh and stop-kiosk.sh, so
# escaping the console and stopping from a terminal do exactly the same thing.
#
# The requirement is that after this runs, the next double-click starts from
# nothing: no processes, port free, camera claimable, no stale lock.
#
# WHY THIS IS NOT JUST `pkill -9`
#
# The thermal driver claims the HIKMICRO by detaching uvcvideo and taking the
# USB endpoint itself. It hands all of that back in close_camera(). SIGKILL
# skips cleanup handlers, so a -9 leaves the interfaces detached and the
# kernel believing a dead process owns them - and the next launch fails with
# "Device or resource busy" on hardware that is fine. Killing hardest is the
# thing that breaks "ready to use next time".
#
# So: SIGTERM and wait, SIGKILL only what refuses, and reset the USB device
# afterwards if force was needed.
#
# Expects REPO, RUN_DIR and PORT to be set by the caller.

TEARDOWN_GRACE_S="${TEARDOWN_GRACE_S:-8}"

_td_say() { printf '%s\n' "$*"; }

_td_pids() {
    # Full command line matching: the pipeline and launcher are started with
    # paths, so a bare process-name match misses them.
    pgrep -f "$1" 2>/dev/null | grep -v "^$$\$"
}

# TERM, wait, then KILL. Returns 0 if graceful, 1 if force was needed.
_td_stop_pattern() {
    local pattern="$1" label="$2" forced=0 pids waited
    pids="$(_td_pids "$pattern")"
    [ -z "$pids" ] && return 0

    # shellcheck disable=SC2086
    kill -TERM $pids 2>/dev/null

    waited=0
    while [ "$waited" -lt "$TEARDOWN_GRACE_S" ]; do
        pids="$(_td_pids "$pattern")"
        [ -z "$pids" ] && break
        sleep 1
        waited=$((waited + 1))
    done

    pids="$(_td_pids "$pattern")"
    if [ -n "$pids" ]; then
        # shellcheck disable=SC2086
        kill -KILL $pids 2>/dev/null
        forced=1
        _td_say "  $label did not exit in ${TEARDOWN_GRACE_S}s - forced"
    else
        _td_say "  $label stopped"
    fi
    return "$forced"
}

_td_port_free() {
    ! (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) \
        | grep -q "[:.]${PORT}[[:space:]]"
}

# Full stop. $1 = "keep-launcher" when called from the launcher itself, which
# must not kill the process running this function.
astravigil_teardown() {
    local keep_launcher="${1:-}" forced=0

    _td_say "stopping AstraVigil..."

    # Browser first: it is the only piece with nothing to clean up, and
    # closing it stops the console showing a frozen last frame while the rest
    # shuts down.
    if [ -f "$RUN_DIR/browser.pid" ]; then
        kill -TERM "$(cat "$RUN_DIR/browser.pid" 2>/dev/null)" 2>/dev/null
        rm -f "$RUN_DIR/browser.pid"
    fi
    _td_stop_pattern "user-data-dir=${RUN_DIR}/chrome-profile" "browser" || true

    # The pipeline. This is the one that must exit gracefully, because it is
    # holding the camera.
    _td_stop_pattern "scripts/run_dashboard.py" "pipeline" || forced=1
    rm -f "$RUN_DIR/dashboard.pid"

    if [ "$keep_launcher" != "keep-launcher" ]; then
        _td_stop_pattern "astravigil-kiosk.sh" "launcher" || true
        rm -f "$RUN_DIR/launcher.pid"
    fi

    # The lock lives on the launcher's file descriptor and is released when
    # that process exits - which the stop above has just ensured.
    #
    # The lock FILE is deliberately left alone. Unlinking it frees nothing: it
    # only detaches the name, so the next launcher creates a fresh inode and
    # takes an exclusive lock on that instead. Two launchers would then each
    # hold a perfectly valid lock on a different file and race two pipelines
    # onto one camera, which is the single thing the lock exists to prevent.

    # Hand the camera back if the pipeline was killed before it could.
    if [ "$forced" -eq 1 ]; then
        _td_say "  pipeline was forced - resetting the thermal camera"
        "${PY:-python3}" "$REPO/scripts/release_camera.py" 2>&1 | sed 's/^/  /'
    fi

    # Confirm rather than assume. A port still held is the failure that makes
    # the next launch time out with a message about the pipeline not starting,
    # which sends you looking in entirely the wrong place.
    local waited=0
    while [ "$waited" -lt 5 ] && ! _td_port_free; do
        sleep 1
        waited=$((waited + 1))
    done
    if _td_port_free; then
        _td_say "  port $PORT is free"
    else
        _td_say "  WARNING: something is still listening on port $PORT"
        (ss -ltnp 2>/dev/null || netstat -ltnp 2>/dev/null) \
            | grep "[:.]${PORT}[[:space:]]" | sed 's/^/    /'
    fi

    _td_say "stopped - the icon will start a fresh run"
}
