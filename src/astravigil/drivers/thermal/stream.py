# Background thermal capture, so the detection pipeline never blocks on USB.
#
# The reader runs in a daemon thread and keeps only the most recent frame.
# That is the right trade for this system: if detection falls behind, we want
# the newest view of the apron, not a backlog of stale ones. A queue would
# give us frames in order at the cost of latency, and latency is the whole
# point of the pitch.
#
# Adapted from the Pi 5 bench reader, which published a colourised BMP to
# /dev/shm for a GUI to pick up. That is not what this project needs - the
# detection stage wants the raw 16-bit values, and the false-colour map is
# cosmetic. So this keeps the raw frame and leaves colourising to whoever is
# displaying it.

import threading
import time

from . import calibration
from .constants import ERRORS_BEFORE_WARNING, ERRORS_BETWEEN_WARNINGS, STREAM_ENDPOINT
from .frame_reader import FrameStats, frames
from .usb_device import close_camera, open_camera


class ThermalStream:
    # Starts a capture thread, hands out the latest frame on request.
    #
    #   stream = ThermalStream()
    #   stream.start()
    #   raw16, ts = stream.latest()
    #   stream.stop()

    def __init__(self, endpoint=STREAM_ENDPOINT, warn=print, log=print):
        self._endpoint = endpoint
        self._warn = warn
        self._log = log

        self._lock = threading.Lock()
        self._frame = None
        self._frame_at = 0.0
        self._frame_index = 0

        self._stop_event = threading.Event()
        self._thread = None
        self._ready = threading.Event()
        self.stats = FrameStats()

    # --- lifecycle -------------------------------------------------------

    def start(self, wait_s=3.0):
        # Launch the reader. Returns True once a first frame has landed, so
        # callers can fail fast instead of polling a stream that never starts.
        if self._thread is not None and self._thread.is_alive():
            # Two threads fighting over one USB endpoint gets you neither.
            return True

        self._stop_event.clear()
        self._ready.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        return self._ready.wait(timeout=wait_s)

    def stop(self, wait_s=3.0):
        self._stop_event.set()
        if self._thread is not None and self._thread.is_alive():
            self._thread.join(timeout=wait_s)
            if self._thread.is_alive():
                self._warn("thermal capture thread did not stop in time - "
                           "the camera may still be busy")
        self._thread = None

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        self.stop()
        return False

    # --- reading ---------------------------------------------------------

    def latest(self):
        # (raw16, captured_at) for the most recent frame, or (None, 0.0).
        with self._lock:
            return self._frame, self._frame_at

    def latest_celsius(self):
        # Most recent frame as a 192x256 float32 array of degrees C.
        # Costs a full-frame pass through the correction curve, so call it
        # once per frame you actually process, not once per question you ask.
        with self._lock:
            frame = self._frame
        return None if frame is None else calibration.calibrate_frame(frame)

    def max_temp(self):
        # Hottest pixel in degrees C, or None. Much cheaper than
        # latest_celsius() - use it for health checks and status lines.
        with self._lock:
            frame = self._frame
        return None if frame is None else calibration.calibrate_max(frame)

    def frame_index(self):
        # Total frames captured. Compare across calls to detect a stalled feed.
        with self._lock:
            return self._frame_index

    def seconds_since_frame(self):
        # How long since the last good frame, or None if there never was one.
        with self._lock:
            if self._frame_at == 0.0:
                return None
            return time.monotonic() - self._frame_at

    def is_healthy(self, stale_after_s=2.0):
        # Whether the feed is currently delivering. A stalled camera reads as
        # "no detections" otherwise, which is the dangerous failure mode for
        # anything claiming to watch a perimeter.
        age = self.seconds_since_frame()
        return age is not None and age < stale_after_s

    # --- worker ----------------------------------------------------------

    def _run(self):
        try:
            dev, detached = open_camera()
        except Exception as exc:
            self._warn(f"thermal camera init failed: {exc}")
            return

        if dev is None:
            self._warn("HIKMICRO thermal camera not found - thermal channel "
                       "is offline")
            return

        self._log("thermal camera streaming started")

        try:
            self._capture(dev)
        finally:
            # However the reader ends, hand the camera back.
            close_camera(dev, detached, warn=self._warn)

    def _capture(self, dev):
        stream = frames(dev, endpoint=self._endpoint,
                        stop_event=self._stop_event, stats=self.stats,
                        on_read_error=self._on_read_error)

        for raw16 in stream:
            now = time.monotonic()
            with self._lock:
                self._frame = raw16
                self._frame_at = now
                self._frame_index += 1

            if not self._ready.is_set():
                self._ready.set()

    def _on_read_error(self, consecutive):
        # Called from inside the read loop, so this still fires when the
        # camera has stopped producing frames entirely. Most read failures are
        # ordinary timeouts, so judge on the consecutive run, not the lifetime
        # total, and stay quiet until it clearly runs away.
        if consecutive == ERRORS_BEFORE_WARNING:
            self._warn("thermal camera reads are failing - the feed has "
                       "stopped updating")
        elif (consecutive > ERRORS_BEFORE_WARNING
              and consecutive % ERRORS_BETWEEN_WARNINGS == 0):
            self._warn(f"thermal camera still not responding after "
                       f"{consecutive} consecutive failed reads")
