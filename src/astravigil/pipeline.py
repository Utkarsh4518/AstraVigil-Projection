"""One frame in, one situational picture out.

Wires the stages together: capture -> calibrate -> detect -> track -> classify
-> map into the optical frame. Everything the dashboard draws comes from the
Result this returns, so the UI holds no pipeline logic and the pipeline can be
run headless for testing.
"""
import time

import cv2
import numpy as np

from .calibration import homography
from .classification.rules import classify
from .detection.thermal import ThermalDetector
from .drivers.thermal.calibration import calibrate_frame
from .tracking.tracker import Tracker


class Result:
    __slots__ = ("thermal_raw", "thermal_c", "optical", "detections",
                 "tracks", "mask", "proc_ms", "capture_ms", "frame_index",
                 "healthy")

    def __init__(self):
        self.thermal_raw = None
        self.thermal_c = None
        self.optical = None
        self.detections = []
        self.tracks = {}
        self.mask = None
        # Time to process one frame, NOT the frame rate - the capture loop
        # paces itself to the camera, so low numbers mean headroom rather than
        # throughput.
        #
        # Capture is timed separately because it is the one stage that lies
        # about the target platform: in simulation it renders an entire world
        # (~19 ms) while on real hardware it is a USB read. Folding the two
        # together made the pipeline look ~40x more expensive than it is.
        # proc_ms is the part that carries over to the Pi.
        self.proc_ms = 0.0
        self.capture_ms = 0.0
        self.frame_index = 0
        self.healthy = True


class Pipeline:
    def __init__(self, source, H=None, threshold_c=1.5):
        self.source = source
        self.H = H
        self.detector = ThermalDetector(threshold_c=threshold_c)
        self.tracker = Tracker()
        self.frame_index = 0
        self._times = []
        self.result = Result()

    @property
    def calibrated(self):
        return self.H is not None

    def set_homography(self, H):
        self.H = H

    def step(self):
        t_cap = time.monotonic()
        raw, optical = self.source.frames()
        capture_s = time.monotonic() - t_cap

        t0 = time.monotonic()
        res = Result()
        res.thermal_raw = raw
        res.optical = optical
        res.thermal_c = calibrate_frame(raw)

        detections = self.detector.update(res.thermal_c)
        tracks = self.tracker.update(detections)

        for det in detections:
            tr = tracks.get(det.track_id)
            det.label, det.confidence = classify(det, tr)
            if tr is not None:
                tr.label, tr.confidence = det.label, det.confidence

        res.detections = detections
        res.tracks = tracks
        res.mask = self.detector.last_mask

        self.frame_index += 1
        res.frame_index = self.frame_index

        self._times.append((time.monotonic() - t0, capture_s))
        if len(self._times) > 30:
            self._times.pop(0)
        n = len(self._times)
        res.proc_ms = 1000.0 * sum(a for a, _ in self._times) / n
        res.capture_ms = 1000.0 * sum(b for _, b in self._times) / n

        self.result = res
        return res

    # ------------------------------------------------------------ views
    def optical_box(self, det):
        """Where a thermal detection lands in the optical frame."""
        if self.H is None:
            return None
        return homography.map_box(self.H, det.box)

    def crop_optical(self, det, margin=8):
        """Optical patch for a thermal detection - the fusion handoff.

        This is what a shape classifier or a VLM would be handed. It is also
        the thing that silently breaks when the homography drifts: the crop
        still returns an image, just of the wrong place.
        """
        if self.H is None or self.result.optical is None:
            return None
        x, y, w, h = self.optical_box(det)
        H_img, W_img = self.result.optical.shape[:2]
        x0 = max(0, x - margin)
        y0 = max(0, y - margin)
        x1 = min(W_img, x + w + margin)
        y1 = min(H_img, y + h + margin)
        if x1 <= x0 or y1 <= y0:
            return None
        return self.result.optical[y0:y1, x0:x1].copy()
