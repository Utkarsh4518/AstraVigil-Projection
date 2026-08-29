"""Thermal anomaly detection: find warm movers against a cold background.

Deliberately classical - background subtraction, contours, hand-built features
- rather than a learned detector. Three reasons, in order of weight:

  1. There is no labelled drone/bird thermal data to train on, and there will
     not be before the deadline.
  2. It runs in a few ms on a Pi 4 CPU with no accelerator.
  3. It is explainable. When a judge asks why it fired, "the blob was 4x the
     noise floor, 40 px, and its area barely changed over a second" is an
     answer. A confidence score from a net is not.

The background model is a running average rather than MOG2 because the scene
is mostly static sky and the camera does its own temporal filtering already;
a simple model is easier to reason about when a detection looks wrong.
"""
import numpy as np

import cv2

# Detection threshold is expressed in degrees above the local background
# rather than raw counts, so it stays meaningful if the sensor scaling is ever
# revised.
DEFAULT_THRESHOLD_C = 1.5
DEFAULT_MIN_AREA_PX = 4        # a drone at 70 m is only a couple of pixels
DEFAULT_MAX_AREA_PX = 4000     # bigger than this is scenery, not a target
BACKGROUND_ALPHA = 0.02        # ~2 s adaptation at 25 fps
WARMUP_FRAMES = 20

# A quadcopter close enough to resolve does not appear as one blob: the four
# motors are far hotter than the airframe, so it breaks into a body plus four
# hot dots. Reported raw, that is five alerts for one aircraft - exactly the
# "wall of separate alarms" the brief asks us not to produce. Blobs whose
# boxes lie within this gap are therefore merged into one object.
#
# The cost is that two genuinely separate targets flying within a few pixels
# of each other merge too. At these ranges that is the rarer error, and the
# merged detection still raises an alert - it just under-counts.
MERGE_GAP_PX = 6.0


class Detection:
    """One warm blob in one frame, in thermal pixel coordinates."""

    __slots__ = ("box", "centroid", "area", "peak_c", "mean_c", "contrast_c",
                 "aspect", "extent", "solidity", "track_id", "label",
                 "confidence", "flap_score", "parts")

    def __init__(self, box, centroid, area, peak_c, mean_c, contrast_c,
                 aspect, extent, solidity, parts=1):
        self.box = box                 # (x, y, w, h)
        self.centroid = centroid       # (cx, cy) float
        self.area = area
        self.peak_c = peak_c
        self.mean_c = mean_c
        self.contrast_c = contrast_c   # how far above local background
        self.aspect = aspect
        self.extent = extent           # area / bounding box area
        self.solidity = solidity       # area / convex hull area
        self.parts = parts             # blobs merged into this object
        self.track_id = None
        self.label = "unknown"
        self.confidence = 0.0
        self.flap_score = 0.0

    @property
    def hotspot_c(self):
        """Peak temperature above the object's OWN mean.

        This is the motor signature. Contrast against the background is not -
        a 31 C bird against -5 C sky is just as bright as a drone, so grading
        on background contrast discriminates nothing. What separates a quad is
        that it carries small cores far hotter than the rest of itself.
        """
        return self.peak_c - self.mean_c

    def as_dict(self):
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(float(self.confidence), 3),
            "box": [int(v) for v in self.box],
            "centroid": [round(float(v), 1) for v in self.centroid],
            "area_px": int(self.area),
            "parts": int(self.parts),
            "peak_c": round(float(self.peak_c), 1),
            "mean_c": round(float(self.mean_c), 1),
            "hotspot_c": round(float(self.hotspot_c), 2),
            "contrast_c": round(float(self.contrast_c), 2),
            "aspect": round(float(self.aspect), 2),
            "extent": round(float(self.extent), 2),
            "solidity": round(float(self.solidity), 2),
            "flap_score": round(float(self.flap_score), 3),
        }


class ThermalDetector:
    def __init__(self, threshold_c=DEFAULT_THRESHOLD_C,
                 min_area=DEFAULT_MIN_AREA_PX, max_area=DEFAULT_MAX_AREA_PX,
                 alpha=BACKGROUND_ALPHA, merge_gap_px=MERGE_GAP_PX):
        self.threshold_c = threshold_c
        self.min_area = min_area
        self.max_area = max_area
        self.alpha = alpha
        self.merge_gap_px = merge_gap_px
        self.background = None
        self.frames_seen = 0
        self.last_mask = None

    @property
    def ready(self):
        # Before the background has settled everything looks like a detection.
        return self.frames_seen >= WARMUP_FRAMES

    def reset(self):
        self.background = None
        self.frames_seen = 0

    def update(self, celsius):
        """Feed one calibrated frame; return the detections in it."""
        frame = np.asarray(celsius, np.float32)

        if self.background is None:
            self.background = frame.copy()
            self.frames_seen = 1
            return []

        diff = frame - self.background

        # Update the background *before* returning, but only where nothing was
        # detected - otherwise a target that hovers slowly dissolves into the
        # model and the system goes quiet on the one thing it should not.
        mask = (diff > self.threshold_c).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.last_mask = mask

        quiet = mask == 0
        self.background[quiet] = ((1 - self.alpha) * self.background[quiet]
                                 + self.alpha * frame[quiet])
        self.frames_seen += 1
        if not self.ready:
            return []

        return self._contours_to_detections(mask, frame, diff)

    def _contours_to_detections(self, mask, frame, diff):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        out = []
        for c in contours:
            area = float(cv2.contourArea(c))
            # contourArea returns 0 for 1-2 px blobs that are still real at
            # range, so fall back to the pixel count.
            if area < 1.0:
                area = float(len(c))
            if not (self.min_area <= area <= self.max_area):
                continue

            x, y, w, h = cv2.boundingRect(c)
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            pix = blob > 0
            if not pix.any():
                continue

            hull_area = float(cv2.contourArea(cv2.convexHull(c))) or area
            m = cv2.moments(c)
            if m["m00"] > 0:
                cx, cy = m["m10"] / m["m00"], m["m01"] / m["m00"]
            else:
                cx, cy = x + w / 2.0, y + h / 2.0

            out.append(Detection(
                box=(x, y, w, h),
                centroid=(cx, cy),
                area=area,
                peak_c=float(frame[pix].max()),
                mean_c=float(frame[pix].mean()),
                contrast_c=float(diff[pix].mean()),
                aspect=float(w) / max(h, 1),
                extent=area / max(float(w * h), 1.0),
                solidity=area / max(hull_area, 1.0),
            ))
        out = merge_nearby(out, self.merge_gap_px)
        out.sort(key=lambda d: d.area, reverse=True)
        return out


def _gap(a, b):
    """Edge-to-edge gap between two boxes; 0 when they touch or overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(bx - (ax + aw), ax - (bx + bw), 0)
    dy = max(by - (ay + ah), ay - (by + bh), 0)
    return max(dx, dy)


def merge_nearby(detections, gap_px):
    """Group blobs that belong to one physical object into one detection.

    Union-find over "boxes closer than gap_px", so a chain of blobs - body to
    motor to motor - collapses into a single object rather than pairing up
    arbitrarily.
    """
    if gap_px <= 0 or len(detections) < 2:
        return detections

    n = len(detections)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(n):
        for j in range(i + 1, n):
            if _gap(detections[i].box, detections[j].box) <= gap_px:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[ri] = rj

    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(detections[i])

    merged = []
    for members in groups.values():
        if len(members) == 1:
            merged.append(members[0])
            continue
        merged.append(_combine(members))
    return merged


def _combine(members):
    x0 = min(d.box[0] for d in members)
    y0 = min(d.box[1] for d in members)
    x1 = max(d.box[0] + d.box[2] for d in members)
    y1 = max(d.box[1] + d.box[3] for d in members)
    w, h = x1 - x0, y1 - y0

    area = sum(d.area for d in members)
    total = area if area > 0 else 1.0
    cx = sum(d.centroid[0] * d.area for d in members) / total
    cy = sum(d.centroid[1] * d.area for d in members) / total

    return Detection(
        box=(x0, y0, w, h),
        centroid=(cx, cy),
        area=area,
        peak_c=max(d.peak_c for d in members),
        mean_c=sum(d.mean_c * d.area for d in members) / total,
        contrast_c=sum(d.contrast_c * d.area for d in members) / total,
        aspect=float(w) / max(h, 1),
        extent=area / max(float(w * h), 1.0),
        # Solidity of the group is not the mean of its parts - a body plus
        # four separated motors fills its hull loosely, and that looseness is
        # itself informative, so derive it from the union box.
        solidity=min(1.0, area / max(float(w * h), 1.0) * 1.15),
        parts=sum(d.parts for d in members),
    )
