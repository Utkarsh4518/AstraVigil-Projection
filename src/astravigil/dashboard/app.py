"""Flask dashboard: four live views, the alert list, and the track table.

Web rather than an OpenCV window, for a practical reason: the Pi will be on a
roof with no display, and a browser page works over the network without
X-forwarding. The same page then runs identically on Windows during
development, which is the whole point of being able to test before deploying.

One capture thread owns the pipeline and publishes the latest rendered frames;
HTTP handlers only ever read them. Viewers therefore cost almost nothing and
a slow client cannot stall the pipeline.

DRAWING IS THE EXPENSIVE PART, AND IT IS NOT THE JOB.

Measured on an x86 laptop: detect, learn and assess costs 1.1 ms per frame.
Rendering the four views and encoding them costs 7.5 ms - nearly seven times
as much. On a Pi 4 that inverts the whole budget: the detector fits inside
40 ms at any plausible ARM multiplier, while rendering alone would push the
loop over it. A rooftop sensor that drops to 15 Hz because it is drawing
pictures nobody is looking at has failed at its actual job.

So two things are decoupled here:

  Detection runs every frame, at the camera's rate. It is the safety-critical
  path and it is cheap.

  Rendering runs only for views a browser is currently subscribed to, and at a
  much lower rate - nobody needs a 25 Hz dashboard. With no viewer the render
  cost is zero.
"""
import os
import threading
import time

import signal

from flask import Flask, Response, jsonify, render_template_string, request

from . import render
from .page import PAGE

BOUNDARY = "frame"
DEFAULT_VIEW_FPS = 6.0

# Exit code the launcher script watches for to mean "start me again".
RESTART_EXIT_CODE = 42

# Loopback addresses. The kiosk control endpoints refuse anything else - see
# _from_this_machine below.
LOOPBACK = ("127.0.0.1", "::1", "localhost")

# name -> how to draw it. Kept as a table so capture_loop can render exactly
# the subset that someone is watching, rather than all of them unconditionally.
RENDERERS = {
    "thermal": lambda res, pipe: render.thermal_view(res),
    "site": lambda res, pipe: render.site_view(res, pipe.site),
    "optical": lambda res, pipe: render.optical_view(res, pipe.H),
    "overlay": lambda res, pipe: render.overlay_view(res, pipe.H),
}


class DashboardState:
    """Latest rendered views, guarded so HTTP threads never touch the pipeline."""

    def __init__(self):
        self.lock = threading.Lock()
        self.views = {}
        self.detections = []
        self.assessments = []
        self.alerts = []
        self.stats = {}
        # How many browsers are currently pulling each stream. This is what
        # decides whether a view is worth drawing at all.
        self.viewers = {}
        self.stop = threading.Event()

    # ---------------------------------------------------------- subscribers
    def subscribe(self, name):
        with self.lock:
            self.viewers[name] = self.viewers.get(name, 0) + 1

    def unsubscribe(self, name):
        with self.lock:
            n = self.viewers.get(name, 0) - 1
            if n > 0:
                self.viewers[name] = n
            else:
                self.viewers.pop(name, None)

    def wanted(self):
        with self.lock:
            return [n for n in RENDERERS if self.viewers.get(n)]

    # ------------------------------------------------------------- publish
    def publish_data(self, detections, assessments, alerts, stats):
        with self.lock:
            self.detections = detections
            self.assessments = assessments
            self.alerts = alerts
            self.stats = stats

    def publish_views(self, views):
        with self.lock:
            self.views.update(views)

    def get(self, name):
        with self.lock:
            return self.views.get(name)

    def snapshot(self):
        with self.lock:
            return (list(self.detections), list(self.assessments),
                    list(self.alerts), dict(self.stats))


def capture_loop(pipeline, state, target_fps=25.0, view_fps=DEFAULT_VIEW_FPS):
    period = 1.0 / target_fps
    view_period = 1.0 / view_fps if view_fps > 0 else None
    ticks = []
    last = None
    last_render = 0.0
    render_ms = 0.0
    rendered = []

    while not state.stop.is_set():
        t0 = time.monotonic()
        # Actual delivered rate, measured wall clock to wall clock. This is
        # what the viewer sees; pipeline.proc_ms is how long the work took.
        if last is not None:
            ticks.append(t0 - last)
            if len(ticks) > 30:
                ticks.pop(0)
        last = t0

        try:
            with state.lock:
                result = pipeline.step()
        except Exception as exc:            # keep the page up on a bad frame
            print(f"pipeline error: {exc}")
            time.sleep(0.25)
            continue

        # --- draw, but only what someone is looking at and only every so often
        wanted = state.wanted()
        if wanted and view_period is not None and t0 - last_render >= view_period:
            t_render = time.monotonic()
            views = {}
            for name in wanted:
                try:
                    views[name] = render.encode_jpeg(
                        RENDERERS[name](result, pipeline))
                except Exception as exc:
                    print(f"render error on {name}: {exc}")
            state.publish_views(views)
            render_ms = 1000.0 * (time.monotonic() - t_render)
            rendered = wanted
            last_render = t0
        elif not wanted:
            render_ms = 0.0
            rendered = []

        dets = [d.as_dict() for d in result.detections]
        seen = render.assessment_map(result)
        for d, det in zip(dets, result.detections):
            tr = result.tracks.get(det.track_id)
            a = seen.get(det.track_id)
            d["hits"] = tr.hits if tr else 0
            d["speed_px"] = round(tr.speed_px, 2) if tr else 0.0
            d["straightness"] = round(tr.straightness, 2) if tr else 0.0
            d["threat"] = round(a.threat, 3) if a else 0.0
            d["level"] = a.level if a else "nominal"
            d["site_risk"] = round(a.site_risk, 3) if a else 0.0
            d["dwell_s"] = round(a.dwell_s, 1) if a else 0.0
            d["key"] = a.key if a else None

        mean_tick = sum(ticks) / len(ticks) if ticks else 0.0
        stats = {
            "fps": round(1.0 / mean_tick, 1) if mean_tick > 0 else 0.0,
            "proc_ms": round(result.proc_ms, 2),
            "capture_ms": round(result.capture_ms, 1),
            # Headroom on the PROCESSING stage only. Capture is excluded
            # because in simulation it renders a whole world and would
            # make this look far worse than the Pi will actually be.
            "headroom": (round(period / (result.proc_ms / 1000.0))
                         if result.proc_ms > 0 else None),
            "frame": result.frame_index,
            "tracks": len(result.tracks),
            "detections": len(dets),
            "calibrated": pipeline.calibrated,
            "source": pipeline.source.name,
            "warmed_up": pipeline.detector.ready,
            "site": result.site_stats,
            "cross": result.cross,
            "learning": result.learning,
            "escalation": result.escalation,
            "alerts": len(result.alerts),
            # What drawing currently costs, and what it is drawing. Zero when
            # nobody has the page open, which is the point.
            "render_ms": round(render_ms, 2),
            "rendering": rendered,
            "view_fps": view_fps,
        }
        # The scan grid is 768 numbers and only means anything while a
        # learning run is in progress, so it rides along then and not
        # otherwise - no point pushing it to the browser 25 times a second
        # for a model that has already finished learning.
        if result.learning.get("active"):
            stats["grid"] = pipeline.site.coverage()
            stats["grid_w"] = pipeline.site.gw
            stats["grid_h"] = pipeline.site.gh

        alerts = [a.as_dict() for a in result.alerts]
        for a in alerts:
            ident = result.identifications.get(a["key"])
            if ident is not None:
                a["identification"] = ident
        state.publish_data(dets, [a.as_dict() for a in result.assessments],
                           alerts, stats)

        rest = period - (time.monotonic() - t0)
        if rest > 0:
            time.sleep(rest)


def _from_this_machine():
    """Kiosk controls are for the browser running on the sensor itself.

    The dashboard binds 0.0.0.0 so the console can be watched from a laptop
    over the network, which is the whole point of a rooftop unit. That also
    means that without this check, anyone who can reach the page could close
    or restart the operator console remotely. The kiosk browser is on
    localhost, so restricting these four endpoints to loopback costs nothing
    and closes that off.
    """
    return request.remote_addr in LOOPBACK


def create_app(pipeline, state, site_path=None):
    app = Flask(__name__)
    # Set by the escape sequence; the launcher script polls for it and closes
    # the browser. Deliberately does NOT stop the pipeline - escaping the
    # display should never blind the site.
    kiosk = {"exit_requested": False}

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    def stream(name):
        def gen():
            # Registering here and clearing in the finally is what tells the
            # capture loop this view is worth drawing. The disconnect is
            # noticed on the next failed write, which is why the loop keeps
            # drawing until the count actually reaches zero.
            state.subscribe(name)
            try:
                last = None
                while not state.stop.is_set():
                    buf = state.get(name)
                    # Re-sending an unchanged frame just burns bandwidth.
                    if buf is None or buf is last:
                        time.sleep(0.01)
                        continue
                    last = buf
                    yield (b"--" + BOUNDARY.encode() + b"\r\n"
                           b"Content-Type: image/jpeg\r\n"
                           b"Content-Length: " + str(len(buf)).encode() +
                           b"\r\n\r\n" + buf + b"\r\n")
            finally:
                state.unsubscribe(name)
        return Response(gen(), mimetype=
                        f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.route("/stream/<name>")
    def s_view(name):
        if name not in RENDERERS:
            return jsonify({"error": f"no such view {name!r}"}), 404
        return stream(name)

    @app.route("/api/state")
    def api_state():
        dets, assessments, alerts, stats = state.snapshot()
        return jsonify({"detections": dets, "assessments": assessments,
                        "alerts": alerts, "stats": stats})

    @app.route("/api/accept", methods=["POST"])
    def api_accept():
        """Operator: this object is normal here, fold it into the baseline."""
        key = (request.get_json(silent=True) or {}).get("key")
        if not key:
            return jsonify({"error": "no key"}), 400
        with state.lock:
            cells = pipeline.accept_assessment(key)
        return jsonify({"key": key, "cells": cells})

    @app.route("/api/ack", methods=["POST"])
    def api_ack():
        alert_id = (request.get_json(silent=True) or {}).get("id")
        with state.lock:
            alert = pipeline.alerts.ack(alert_id)
        return jsonify({"acked": alert is not None})

    @app.route("/api/site/learn", methods=["POST"])
    def api_site_learn():
        """Operator commissioning the site: start, stop, or check a scan."""
        body = request.get_json(silent=True) or {}
        action = body.get("action", "status")
        if action == "start":
            return jsonify(pipeline.start_learning(
                seconds=float(body.get("seconds", 90)),
                reset=bool(body.get("reset", True))))
        if action == "stop":
            path = body.get("path") or site_path or "data/baseline/site.npz"
            with state.lock:
                return jsonify(pipeline.stop_learning(save_path=path))
        return jsonify(pipeline.learning_status())

    @app.route("/api/site/grid")
    def api_site_grid():
        return jsonify({"w": pipeline.site.gw, "h": pipeline.site.gh,
                        "coverage": pipeline.site.coverage(),
                        "learning": pipeline.learning_status()})

    # ------------------------------------------------------------- kiosk
    @app.route("/api/kiosk/status")
    def kiosk_status():
        if not _from_this_machine():
            return jsonify({"error": "kiosk control is local-only"}), 403
        # Only the launcher polls this, so it doubles as proof that a
        # supervisor is alive and will act on exit_requested.
        kiosk["last_poll"] = time.time()
        return jsonify(kiosk)

    def _supervised(within_s=6.0):
        last = kiosk.get("last_poll", 0.0)
        return last and (time.time() - last) < within_s

    def _self_stop():
        # Raise the interrupt the shutdown path already handles, rather than
        # exiting outright: that is what runs source.close() and hands the
        # thermal camera back. Killing the process instead leaves uvcvideo
        # detached and the next run fails on hardware that is fine.
        time.sleep(1.0)
        try:
            os.kill(os.getpid(), signal.SIGINT)
        except (AttributeError, OSError, ValueError):
            # Windows has no usable SIGINT to self. Dev only - there is no
            # thermal camera on this path, so nothing is left claimed.
            os._exit(0)

    @app.route("/api/kiosk/exit", methods=["POST"])
    def kiosk_exit():
        """Stop request from the page - the close button or the key sequence.

        Two ways this gets honoured, and which one applies depends on whether
        anything is supervising:

          LAUNCHED FROM THE ICON  the launcher is polling /api/kiosk/status,
            sees the flag, and runs the full teardown - browser, pipeline,
            port, lock, camera. Setting the flag is all that is needed.

          RUN FROM A TERMMINAL  nothing polls, so the flag alone would do
            nothing at all and the button would look broken. Stop ourselves
            instead.
        """
        if not _from_this_machine():
            return jsonify({"error": "kiosk control is local-only"}), 403
        kiosk["exit_requested"] = True
        if _supervised():
            print("stop requested - the launcher will tear everything down")
            return jsonify({"exiting": True, "by": "launcher"})
        print("stop requested - no launcher supervising, stopping this process")
        threading.Thread(target=_self_stop, daemon=True).start()
        return jsonify({"exiting": True, "by": "self"})

    @app.route("/api/kiosk/ack", methods=["POST"])
    def kiosk_ack():
        """Launcher confirming it has closed the browser."""
        if not _from_this_machine():
            return jsonify({"error": "kiosk control is local-only"}), 403
        kiosk["exit_requested"] = False
        return jsonify({"ok": True})

    @app.route("/api/kiosk/restart", methods=["POST"])
    def kiosk_restart():
        """Exit with the code the launcher treats as 'run me again'."""
        if not _from_this_machine():
            return jsonify({"error": "kiosk control is local-only"}), 403
        print("kiosk: restart requested")

        def bye():
            # Long enough for this response to reach the browser, then a hard
            # exit - Flask's dev server has no clean programmatic shutdown and
            # the launcher is going to start a fresh process anyway.
            time.sleep(0.4)
            os._exit(RESTART_EXIT_CODE)

        threading.Thread(target=bye, daemon=True).start()
        return jsonify({"restarting": True})

    @app.route("/api/save_site", methods=["POST"])
    def api_save_site():
        path = site_path or "data/baseline/site.npz"
        with state.lock:
            pipeline.save_site(path)
        return jsonify({"saved": path,
                        "frames": pipeline.site.frames})

    return app
