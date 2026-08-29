# Desktop launcher and kiosk console

A double-clickable icon on the Raspberry Pi desktop that starts the sensor and
takes over the screen — no window chrome, no address bar, no close button.

```bash
./deploy/install_kiosk.sh
```

That writes `AstraVigil.desktop` to the desktop folder and to the applications
menu, makes the scripts executable, and marks the launcher trusted (without
which Raspberry Pi OS shows *"Untrusted application launcher"* instead of
running it).

## Read this before you rely on it

**Test the way out before you need it.** A fullscreen kiosk with no exit button
is a machine you can be locked out of. The escape sequence lives in the web
page, so it only works while the page is loaded and focused — if Chromium
crashes or the page fails, that hatch is gone with it. That is why the launcher
refuses to go fullscreen until the pipeline actually answers on its port, and
why there are three independent ways out rather than one.

```
0.  restart button                     stays visible in kiosk mode
1.  "close console" button            top right of the page, two clicks
2.  Ctrl+Shift+Esc  ×3 within 3 s      in-page, with an on-screen counter
3.  deploy/stop-kiosk.sh               terminal or SSH - needs no browser
4.  Ctrl+Alt+F2                        text console, then run (3)
```

Escaping performs a **full stop**: browser, pipeline, port and camera all
released, so the next double-click starts a fresh run with nothing left over.
`stop-kiosk.sh --display` closes the console but keeps detecting, if that is
what you want.

This was the other way round originally — escape released only the display,
on the reasoning that a perimeter system should not go blind because somebody
wanted the desktop back. That is the right default for a fixed installation
and the wrong one for a rig you are carrying to a demo: it leaves a half-state
that is hard to reason about, and the next launch refuses with "already
running" while appearing to have nothing running at all.

**Shutdown is by SIGTERM, never SIGKILL, and that is load-bearing.** The
thermal driver detaches `uvcvideo` to claim the USB endpoint and hands it back
in its cleanup path. Python does not run `finally` blocks on SIGTERM by
default — the default disposition kills the process outright — so
`run_dashboard.py` installs a handler that turns SIGTERM into the same
`KeyboardInterrupt` the Ctrl+C path already handles. Without it, even a polite
`kill` leaves the camera claimed by a dead process and the next run fails with
"Device or resource busy" on hardware that is perfectly fine.

Anything that does have to be forced triggers `scripts/release_camera.py`,
which resets the USB device so the claim is dropped. Killing hardest is the
thing that breaks "ready to use next time".

## The close button

Top right of the console, and the way out to reach for first. Click it once to
arm — it turns red and reads "click again to stop" — then click again within
four seconds.

Two clicks rather than one because a single-click kill on an always-on console
is one sleeve-brush away from taking the sensor down mid-demo. It is the same
"you meant this" test the keyboard sequence applies, without depending on a
shortcut that may never arrive.

**Prefer it to the key sequence.** Linux desktops commonly grab Ctrl+Shift+Esc
for their own task manager before the page ever sees it, which leaves an
operator in a fullscreen window with no address bar and a shortcut that does
nothing. The button cannot be intercepted.

## The escape sequence

Press **Ctrl+Shift+Esc** three times within three seconds. A counter appears at
the bottom of the screen after the first press:

```
Exit kiosk: 1 of 3 — press Ctrl+Shift+Esc again
```

The counter is deliberate. An escape hatch you cannot tell is registering is
one you will assume is broken, and the person finding that out is standing in
front of a fullscreen window with no other way out.

It is armed only when the page is opened with `?kiosk=1`, which is what the
launcher does — so pressing it during normal development does nothing.

**If Chromium intercepts the combination**, change it in
[page.py](../src/astravigil/dashboard/page.py) — the handler checks
`e.ctrlKey && e.shiftKey` and `Escape`. Chromium binds Shift+Esc to its own
task manager on some builds; `--kiosk` normally suppresses that, but it has not
been verified on your Pi's exact Chromium version. **Verify it on the hardware
before the demo**, and if it does not fire, `stop-kiosk.sh` is the fallback.

## Restart

The console keeps one control in kiosk mode: **restart**. It exits the pipeline
with code `42`, which the launcher's supervisor loop treats as *"start me
again"*, then reloads the page. Verified: the process really does exit 42.

## How it fits together

```
AstraVigil.desktop
   └─ astravigil-kiosk.sh
        ├─ supervises  run_dashboard.py     (restart on exit code 42)
        ├─ waits for   /api/state           (up to 90 s - no fixed sleep)
        ├─ launches    chromium --kiosk http://localhost:8000/?kiosk=1
        └─ polls       /api/kiosk/status    (closes the browser on the flag)
```

The page cannot close the window itself — browsers refuse `window.close()` for
windows a script did not open — so it raises a flag and the launcher does the
closing.

One double-click does **both halves**: the detection pipeline and the GUI. The
GUI is the same browser console as `run_dashboard.py` serves, opened fullscreen
with `?kiosk=1` so the escape handler arms itself and the developer-only
controls hide.

The URL is passed to Chromium as a **positional argument to `--kiosk`, not via
`--app=`**. Both forms appear in Pi kiosk guides and they are not equivalent:
on some Chromium builds `--app` takes precedence and opens an app *window* -
with a title bar and a close button - which is exactly what a kiosk must not
have. `--kiosk <URL>` is the well-tested form.

**A fixed sleep would have been the obvious way to wait for the pipeline, and
it is wrong.** On a cold Pi 4 the first frame can take a while; going
fullscreen early leaves you looking at a connection error with no address bar.
The launcher polls, and if the pipeline never comes up it shows a dialog with
the failure and the command to reproduce it by hand — and never goes fullscreen
at all.

Single-instance is enforced with `flock`, so double-clicking the icon twice
does not race two pipelines onto the same USB camera. Because that wait can
run to tens of seconds on a cold Pi, the launcher posts a desktop notification
the moment it starts - an icon that produces no visible response for half a
minute gets double-clicked again, and the second click would be met with a
confusing "already running".

**The guard fails open, and that is deliberate.** Refusing to start is the
expensive answer here: from the desktop it looks exactly like a dead icon, and
it hands the operator nothing to act on. So *"still starting"* is only ever
said when another launcher is demonstrably alive — it names that process's pid,
and points at `stop-kiosk.sh`. A held lock on its own is not evidence: if
nothing is listening on the port, no console is open and no launcher process
exists, the lock outlived whatever took it and the launcher takes it over
rather than deferring to a process that is not there.

Three things follow from that, all of which were bugs:

- `flock` missing is not the same as the lock being held. `! flock -n 9` is
  also true when the command does not exist, which sent *every* launch —
  including the first on a fresh machine — down the "still starting" path.
  Without util-linux the launcher now falls back to a pid check.
- **Nothing blocks while holding the lock.** `notify()` is `zenity --info`,
  which waits for someone to click OK. The failure dialog used to be raised
  while the lock was still held, so a single failed start made every later
  double-click answer "still starting" — permanently — with the dialog that
  explained why sitting unread behind the splash. The lock is released, and
  the half-started pipeline stopped, *before* any dialog is raised.
- The lock file is never unlinked. Removing it frees nothing — it detaches the
  name, so the next launcher creates a fresh inode and locks that instead, and
  two launchers each hold a valid exclusive lock on a different file. The
  supervisor subshell also explicitly closes the descriptor (`exec 9>&-`), or
  it keeps the lock alive after the launcher it belongs to has exited.

That notification is deliberately `notify-send`, never `zenity --info`: the
modal dialog blocks until somebody clicks OK, which would stall the launch
behind a box nobody is watching.

## Kiosk control is local-only

The dashboard binds `0.0.0.0` so you can watch the console from a laptop, which
is the point of a rooftop unit. Without a guard, anyone who could reach the page
could also close or restart the operator console remotely. The four
`/api/kiosk/*` endpoints therefore accept loopback only — the kiosk browser is
on localhost, so this costs nothing. Verified:

```
kiosk exit endpoint:
  from the sensor itself           ALLOWED -> {'exiting': True}
  from across the network          REFUSED 403 -> kiosk control is local-only

normal dashboard viewing over the network still works:
  frame 8, fps 24.7 - readable remotely as intended
```

## The icon does nothing, or says "still starting"

`~/.astravigil-kiosk.log` is the first place to look — every branch above logs
which one it took.

| What you see | What it means |
|---|---|
| `AstraVigil is still starting (pid N)` | A launcher really is mid-startup. Give it 90 s; if the console never appears, `deploy/stop-kiosk.sh` and click again. |
| `stale lock, no live launcher - taking it over` | A previous run left the lock behind. Handled; the launch continues. |
| `flock is not installed - falling back to a pid check` | `sudo apt install -y util-linux` to restore the real lock. |
| `AstraVigil could not start` + a reason | The pipeline never answered. The dialog carries the command to reproduce it by hand. |
| Nothing at all in the log | The `.desktop` entry never ran the script — check it is marked trusted, and re-run `install_kiosk.sh`. |

A full stop from anywhere clears every one of these:

```bash
./deploy/stop-kiosk.sh
```

## Configuring what it launches

The launcher's defaults assume the real rig:

```
--source hardware --calibration data/calibration/H.json
--site-model data/baseline/site.npz --save-site
```

Override at install time, which bakes it into the `.desktop` entry:

```bash
ASTRAVIGIL_ARGS='--source synthetic --scenario intrusion' ./deploy/install_kiosk.sh
ASTRAVIGIL_ARGS='--source hardware --view-fps 6' ./deploy/install_kiosk.sh
```

`ASTRAVIGIL_PORT` changes the port (default 8000). Log goes to
`~/.astravigil-kiosk.log`, which is the first place to look if the icon appears
to do nothing.

### Secrets and per-rig settings

The launcher sources `~/.astravigil.env` before it decides anything, so that
file can set any of `FEATHERLESS_API_KEY`, `ASTRAVIGIL_ARGS`, `ASTRAVIGIL_PORT`
and `ASTRAVIGIL_THERMAL_ROT`:

```bash
printf 'FEATHERLESS_API_KEY=sk-...\n' > ~/.astravigil.env
chmod 600 ~/.astravigil.env
```

It exists because there is nowhere else that works. A desktop launcher inherits
the graphical session's environment and nothing more: it does not read
`~/.bashrc`, which is for interactive shells, and `~/.profile` is only sourced
for the session by *some* display managers — so "export it in your shell" is
not an answer that survives a double-click. The `.desktop` file is not the
place either, because `install_kiosk.sh` rewrites it and it is world-readable.

Escalation stays **off** unless `--featherless` is in `ASTRAVIGIL_ARGS` as
well. The key alone does not enable it: sending cropped imagery of the
protected site to a third party should take a deliberate second step, and
nothing in the system depends on it.

To check all of that at once:

```bash
python3 scripts/check_featherless.py          # placement and shape
python3 scripts/check_featherless.py --live   # spend one call, prove it works
```

The dashboard's `featherless: on` line is not proof the key is good -
`FeatherlessClient.configured` is `bool(api_key)` and nothing more. Only
`--live` distinguishes a working key from a typo.

## Starting on boot instead

If the unit should come up watching without anyone logging in, a systemd user
service is the better tool than a desktop icon:

```bash
mkdir -p ~/.config/systemd/user
cat > ~/.config/systemd/user/astravigil.service <<'EOF'
[Unit]
Description=AstraVigil kiosk console
After=graphical-session.target
PartOf=graphical-session.target

[Service]
ExecStart=%h/astravigil-cuas/deploy/astravigil-kiosk.sh
Restart=on-failure
RestartSec=5

[Install]
WantedBy=graphical-session.target
EOF
systemctl --user daemon-reload
systemctl --user enable --now astravigil.service
sudo loginctl enable-linger "$USER"     # start without an interactive login
```

`systemctl --user stop astravigil` then becomes a fourth way out.

## Not tested on hardware

All of this was written and syntax-checked on Windows. What **has** been
verified: the escape handshake end to end, including the exact pattern the
shell watchdog greps for; that the pipeline keeps detecting after an escape;
that restart exits with code 42; that the kiosk endpoints refuse non-local
callers; that the page carries the handler; and that all three scripts parse
under `bash -n` with Unix line endings.

The single-instance guard was then exercised against a stubbed launcher in all
four of its states — no `flock` on `PATH`, a lock file left behind by a dead
process, a launcher genuinely mid-startup, and a sensor already answering on
its port. The first three used to end in "still starting"; they now start, and
the fourth still reattaches to the running sensor instead of racing it.

What has **not** been verified, because it needs the Pi: whether Chromium on
your build swallows Ctrl+Shift+Esc, whether the desktop file manager accepts
the trusted flag, and how the cursor-hiding and screen-blanking calls behave
under Wayland versus X11. Run `install_kiosk.sh`, double-click the icon, and
test the escape sequence — before the demo, not during it.
