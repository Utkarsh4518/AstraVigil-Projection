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
1.  Ctrl+Shift+Esc  ×3 within 3 s      in-page, with an on-screen counter
2.  deploy/stop-kiosk.sh               terminal or SSH - needs no browser
3.  Ctrl+Alt+F2                        text console, then run (2)
```

Escaping releases **the display only**. The sensor keeps running and keeps
detecting — a perimeter system should never go blind because somebody wanted
the desktop. `stop-kiosk.sh --all` stops everything, and says so plainly.

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

What has **not** been verified, because it needs the Pi: whether Chromium on
your build swallows Ctrl+Shift+Esc, whether the desktop file manager accepts
the trusted flag, and how the cursor-hiding and screen-blanking calls behave
under Wayland versus X11. Run `install_kiosk.sh`, double-click the icon, and
test the escape sequence — before the demo, not during it.
