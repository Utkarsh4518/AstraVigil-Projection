"""Flask dashboard: three live views plus the detection table.

Web rather than an OpenCV window, for a practical reason: the Pi will be on a
roof with no display, and a browser page works over the network without
X-forwarding. The same page then runs identically on Windows during
development, which is the whole point of being able to test before deploying.

One capture thread owns the pipeline and publishes the latest rendered frames;
HTTP handlers only ever read them. Viewers therefore cost almost nothing and
a slow client cannot stall the pipeline.
"""
import threading
import time

from flask import Flask, Response, jsonify, render_template_string

from . import render
from .page import PAGE

BOUNDARY = "frame"


class DashboardState:
    """Latest rendered views, guarded so HTTP threads never touch the pipeline."""

    def __init__(self):
        self.lock = threading.Lock()
        self.views = {}
        self.detections = []
        self.stats = {}
        self.stop = threading.Event()

    def publish(self, views, detections, stats):
        with self.lock:
            self.views = views
            self.detections = detections
            self.stats = stats

    def get(self, name):
        with self.lock:
            return self.views.get(name)

    def snapshot(self):
        with self.lock:
            return list(self.detections), dict(self.stats)


def capture_loop(pipeline, state, target_fps=25.0):
    period = 1.0 / target_fps
    ticks = []
    last = None
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
            result = pipeline.step()
        except Exception as exc:            # keep the page up on a bad frame
            print(f"pipeline error: {exc}")
            time.sleep(0.25)
            continue

        views = {
            "thermal": render.encode_jpeg(render.thermal_view(result)),
            "optical": render.encode_jpeg(
                render.optical_view(result, pipeline.H)),
            "overlay": render.encode_jpeg(
                render.overlay_view(result, pipeline.H)),
        }
        dets = [d.as_dict() for d in result.detections]
        for d, det in zip(dets, result.detections):
            tr = result.tracks.get(det.track_id)
            d["hits"] = tr.hits if tr else 0
            d["speed_px"] = round(tr.speed_px, 2) if tr else 0.0
            d["straightness"] = round(tr.straightness, 2) if tr else 0.0

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
        }
        state.publish(views, dets, stats)

        rest = period - (time.monotonic() - t0)
        if rest > 0:
            time.sleep(rest)


def create_app(pipeline, state):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    def stream(name):
        def gen():
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
        return Response(gen(), mimetype=
                        f"multipart/x-mixed-replace; boundary={BOUNDARY}")

    @app.route("/stream/thermal")
    def s_thermal():
        return stream("thermal")

    @app.route("/stream/optical")
    def s_optical():
        return stream("optical")

    @app.route("/stream/overlay")
    def s_overlay():
        return stream("overlay")

    @app.route("/api/state")
    def api_state():
        dets, stats = state.snapshot()
        return jsonify({"detections": dets, "stats": stats})

    return app
