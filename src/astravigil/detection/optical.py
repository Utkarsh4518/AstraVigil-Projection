"""Optical detection: the other half of the cross-cue.

The thermal camera leads because warm-against-cold-sky is the strongest signal
available for a small airframe. But it is not the only signal, and there are
two situations where leading with thermal alone fails outright:

  THERMAL CROSSOVER   twice a day, at dawn and dusk, the background temperature
                      sweeps through the object temperature and the contrast
                      that the whole detector rests on goes to zero. During
                      those windows thermal is blind and optical is the only
                      sensor still working.

  COLD-SOAKED TARGET  a drone that has been sitting on the apron for hours is
                      at ambient. It has no thermal signature left to find,
                      only a shape.

So optical detects independently, and can cue thermal, rather than only ever
being asked. That is the point of a two-way architecture: each sensor covers
the other's blind spot instead of merely confirming its hits.

Two deliberate differences from the thermal detector:

  ABSOLUTE difference, not signed. A warm object is brighter than cold sky, so
  thermal only ever looks for positive excursions. An airframe against a bright
  sky is DARKER than its background, so looking for bright blobs here would
  miss every target in daylight.

  Restricted to the thermal footprint by default. The optical camera sees a
  62 degree field, the thermal one 25 degrees - the thermal frame covers only
  about 15% of the optical frame's area. Outside that overlap an optical
  detection cannot be cross-checked against anything, and on a fixed mount
  there is no way to slew to look. Detections out there are reportable but not
  verifiable, and pretending otherwise would flood the operator with
  unconfirmable clutter.
"""
import cv2
import numpy as np

from ..utils.env import env_float

DEFAULT_THRESHOLD = 18.0        # grey levels away from the background model
DEFAULT_MIN_AREA_PX = 12
DEFAULT_MAX_AREA_PX = 30000

# Shape sanity, all on the BOUNDING BOX rather than the contour area.
#
# The area limits above cannot catch what an indoor rig actually produces. A
# lighting change along a wall, or a camera that shifts a pixel, leaves a long
# thin L-shaped contour tracing an edge: a few hundred pixels of area, well
# inside every area limit, inside a bounding box that spans most of the frame.
# The console then draws a box round half the room and calls it a contact,
# which is what an operator photographed and reasonably called nonsense.
#
# Three tests, each aimed at one way that goes wrong:

# How much of the searched frame one contact may claim. Something occupying a
# third of the view is the view changing, not an object arriving.
MAX_BOX_FRAC = env_float("ASTRAVIGIL_OPTICAL_MAX_BOX_FRAC", 0.22)

# Longest side over shortest. Real objects are not 8:1 slivers; wall edges,
# door frames, curtain lines and skirting boards are exactly that.
MAX_BOX_ASPECT = env_float("ASTRAVIGIL_OPTICAL_MAX_ASPECT", 6.0)

# How much of its own bounding box the blob has to fill. This is the test that
# does most of the work: an L along two walls has a huge box and fills almost
# none of it, while a person, a bag or an airframe fills a good half.
MIN_BOX_EXTENT = env_float("ASTRAVIGIL_OPTICAL_MIN_EXTENT", 0.14)
BACKGROUND_ALPHA = 0.02
WARMUP_FRAMES = 20
MERGE_GAP_PX = 8.0


class OpticalDetection:
    """One candidate in the optical frame, in optical pixel coordinates."""

    __slots__ = ("box", "centroid", "area", "aspect", "extent", "solidity",
                 "contrast", "darker", "track_id", "thermal_match")

    def __init__(self, box, centroid, area, aspect, extent, solidity,
                 contrast, darker):
        self.box = box
        self.centroid = centroid
        self.area = area
        self.aspect = aspect
        self.extent = extent
        self.solidity = solidity
        self.contrast = contrast       # mean |difference| from background
        # Which way it deviates. An airframe against sky is darker; a person
        # or a vehicle against shaded ground is usually brighter. Not
        # diagnostic on its own, but it is free and it is context.
        self.darker = darker
        self.track_id = None
        self.thermal_match = None      # set when a thermal detection claims it

    def as_dict(self):
        return {"box": [int(v) for v in self.box],
                "centroid": [round(float(v), 1) for v in self.centroid],
                "area_px": int(self.area),
                "aspect": round(float(self.aspect), 2),
                "extent": round(float(self.extent), 2),
                "solidity": round(float(self.solidity), 2),
                "contrast": round(float(self.contrast), 1),
                "darker": bool(self.darker),
                "matched_thermal": self.thermal_match}


class OpticalDetector:
    def __init__(self, threshold=DEFAULT_THRESHOLD,
                 min_area=DEFAULT_MIN_AREA_PX, max_area=DEFAULT_MAX_AREA_PX,
                 alpha=BACKGROUND_ALPHA, roi=None, every_n=1):
        self.threshold = threshold
        self.min_area = min_area
        self.max_area = max_area
        self.alpha = alpha
        # (x, y, w, h) in optical pixels, normally the thermal footprint.
        self.roi = roi
        # Optical frames are 6x the pixel count of thermal ones and carry far
        # more clutter, so on a Pi this is the stage worth running at a lower
        # rate. Thermal keeps the full frame rate; it is the primary sensor.
        self.every_n = max(1, int(every_n))
        self.background = None
        self.frames_seen = 0
        self.calls = 0
        self.last_mask = None
        self._last = []
        # Contacts thrown out for being the wrong shape to be an object.
        # Counted rather than silently dropped: a filter nobody can see the
        # effect of is one nobody can tell is set wrong.
        self.rejected_shape = 0

    @property
    def ready(self):
        return self.frames_seen >= WARMUP_FRAMES

    def set_roi_from_homography(self, H, thermal_shape, optical_shape,
                                pad=0):
        """Confine the search to where the thermal camera can verify."""
        from ..calibration import homography
        h, w = thermal_shape[:2]
        pts = homography.map_points(H, [[0, 0], [w, 0], [w, h], [0, h]])
        x0 = max(0, int(pts[:, 0].min()) - pad)
        y0 = max(0, int(pts[:, 1].min()) - pad)
        x1 = min(optical_shape[1], int(pts[:, 0].max()) + pad)
        y1 = min(optical_shape[0], int(pts[:, 1].max()) + pad)
        if x1 <= x0 or y1 <= y0:
            self.roi = None
        else:
            self.roi = (x0, y0, x1 - x0, y1 - y0)
        return self.roi

    def reset(self):
        self.background = None
        self.frames_seen = 0

    def update(self, bgr):
        """Feed one optical frame; return the candidates in it.

        Returns the previous result on frames it skips, so a caller running
        this at a reduced rate still sees a stable candidate list rather than
        objects blinking in and out.
        """
        self.calls += 1
        if (self.calls - 1) % self.every_n:
            return self._last

        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if self.roi is not None:
            x, y, w, h = self.roi
            grey = grey[y:y + h, x:x + w]
        grey = grey.astype(np.float32)

        if self.background is None or self.background.shape != grey.shape:
            self.background = grey.copy()
            self.frames_seen = 1
            self._last = []
            return self._last

        signed = grey - self.background
        diff = np.abs(signed)
        mask = (diff > self.threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
        self.last_mask = mask

        # Same discipline as the thermal side: do not learn where something
        # was detected, or an object that stops dissolves into the model.
        quiet = mask == 0
        self.background[quiet] = ((1 - self.alpha) * self.background[quiet]
                                  + self.alpha * grey[quiet])
        self.frames_seen += 1
        if not self.ready:
            self._last = []
            return self._last

        self._last = self._contours(mask, signed, diff)
        return self._last

    @staticmethod
    def _plausible(area, w, h, frame_shape):
        """Could a blob this shape be an object, rather than a wall edge?

        Cheap, and applied before the expensive per-blob work below - which
        is most of the cost of this stage on a cluttered indoor frame.
        """
        fh, fw = frame_shape[:2]
        box = float(w * h)
        if box <= 0:
            return False
        if box > MAX_BOX_FRAC * fw * fh:
            return False
        if max(w, h) / max(min(w, h), 1) > MAX_BOX_ASPECT:
            return False
        return area / box >= MIN_BOX_EXTENT

    def _contours(self, mask, signed, diff):
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)
        ox, oy = (self.roi[0], self.roi[1]) if self.roi else (0, 0)
        out = []
        for c in contours:
            area = float(cv2.contourArea(c))
            if area < 1.0:
                area = float(len(c))
            if not (self.min_area <= area <= self.max_area):
                continue
            x, y, w, h = cv2.boundingRect(c)
            if not self._plausible(area, w, h, mask.shape):
                self.rejected_shape += 1
                continue
            blob = np.zeros(mask.shape, np.uint8)
            cv2.drawContours(blob, [c], -1, 255, -1)
            pix = blob > 0
            if not pix.any():
                continue
            hull = float(cv2.contourArea(cv2.convexHull(c))) or area
            m = cv2.moments(c)
            cx = m["m10"] / m["m00"] if m["m00"] else x + w / 2.0
            cy = m["m01"] / m["m00"] if m["m00"] else y + h / 2.0
            out.append(OpticalDetection(
                box=(x + ox, y + oy, w, h),
                centroid=(cx + ox, cy + oy),
                area=area,
                aspect=float(w) / max(h, 1),
                extent=area / max(float(w * h), 1.0),
                solidity=area / max(hull, 1.0),
                contrast=float(diff[pix].mean()),
                darker=bool(signed[pix].mean() < 0),
            ))
        out.sort(key=lambda d: d.area, reverse=True)
        return out
