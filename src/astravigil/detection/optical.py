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

# How many robust sigma of frame noise a pixel must exceed, on top of the
# fixed threshold above.
#
# The fixed threshold is a statement about the SIGNAL - eighteen grey levels
# is a real change in the scene - and it says nothing about the sensor. In
# good light a camera's frame-to-frame noise is well under a grey level and
# this changes nothing. In a dark room the same sensor runs at high gain and
# its noise floor climbs several levels, so eighteen stops being a threshold
# and becomes a sieve: an operator gets forty boxes over an empty room, each
# numbered, each labelled as a contact.
#
# So the floor rises with the measured noise. The median absolute difference
# from the background model is a robust estimate of that noise - robust
# because the real moving object, whatever it is, occupies a small fraction
# of the frame and cannot drag a median. Six sigma over it is conservative:
# on pure Gaussian noise that is one false pixel in half a billion, and a
# single pixel is far below the minimum area anyway.
NOISE_SIGMA = env_float("ASTRAVIGIL_OPTICAL_NOISE_SIGMA", 6.0)
BACKGROUND_ALPHA = 0.02
WARMUP_FRAMES = 20
# Fragments closer than this are one object.
#
# This constant has been sitting here unused since the file was written, and
# its absence is what an operator sees as "when a person passes, it detects
# other things too". Background subtraction on a person does not return a
# person: it returns a head, an arm, a leg and a shadow, as four disconnected
# blobs, because the middle of a plain shirt differs from the plain wall
# behind it by less than the threshold. Each fragment then gets its own box,
# its own number and its own contact.
#
# Merging is done on the BOXES, after contours, and not by closing the mask.
# Closing wide enough to bridge a person's limbs - twenty or thirty pixels -
# smears every shape in the frame and inflates every box; the gaps that need
# bridging are between parts of one object, and that is a question about
# objects rather than about pixels.
#
# And the threshold is relative to size, because that is what actually
# distinguishes the two cases. Parts of one object are close compared to how
# big the object is: a person two hundred pixels tall has limbs a fair way
# apart, while two separate objects twenty pixels across are separate at a
# fraction of that distance. A fixed pixel gap has to choose which of those
# to get wrong.
# Fraction of full resolution the detector actually works at.
#
# DEFAULT 1.0, deliberately.
#
# This was written at 0.5 to halve each side and quarter the per-pixel work,
# on a measurement that turned out to be of the wrong thing: a benchmark loop
# that spent most of itself generating synthetic frames, not detecting in
# them. Under a profiler this whole call is about 1.2 ms, and the reduction
# bought a fraction of a millisecond in exchange for raising the smallest
# visible object from twelve pixels of area to fifty.
#
# Kept as an escape hatch for a machine slow enough to need it, and left off,
# because a behaviour change that buys nothing measurable is not a trade -
# it is just a change.
WORK_SCALE = env_float("ASTRAVIGIL_OPTICAL_WORK_SCALE", 1.0)

MERGE_GAP_PX = 8.0          # floor, for objects too small for the fraction
MERGE_GAP_FRAC = env_float("ASTRAVIGIL_OPTICAL_MERGE_FRAC", 0.35)


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


def _box_gap(a, b):
    """Pixels between two boxes; 0 if they touch or overlap."""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    dx = max(0, max(ax, bx) - min(ax + aw, bx + bw))
    dy = max(0, max(ay, by) - min(ay + ah, by + bh))
    return max(dx, dy)


def _merge_fragments(dets, frame_shape):
    """Join detections that are parts of one object.

    Background subtraction does not return a person; it returns a head, an
    arm, a leg and a shadow, because the middle of a plain shirt differs from
    the plain wall behind it by less than the threshold. Each fragment
    otherwise gets its own box, its own number and its own contact - which is
    what somebody sees as "when a person passes, it detects other things too".
    """
    if len(dets) < 2:
        return dets
    boxes = [list(d.box) for d in dets]
    groups = [[i] for i in range(len(dets))]
    merged = True
    while merged and len(groups) > 1:
        merged = False
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                a, b = boxes[i], boxes[j]
                # The LARGER of the two, because the object is at least as
                # big as its biggest part: a head fifteen pixels above a
                # torso is part of the person, and judging that gap against
                # the head alone gets it wrong in the one direction that
                # matters. The union still has to survive the shape test, so
                # this cannot run away across a frame.
                span = max(a[2], a[3], b[2], b[3])
                limit = max(MERGE_GAP_PX, MERGE_GAP_FRAC * span)
                if _box_gap(a, b) > limit:
                    continue
                x0 = min(a[0], b[0])
                y0 = min(a[1], b[1])
                x1 = max(a[0] + a[2], b[0] + b[2])
                y1 = max(a[1] + a[3], b[1] + b[3])
                if not OpticalDetector._plausible(
                        sum(dets[k].area for k in groups[i] + groups[j]),
                        x1 - x0, y1 - y0, frame_shape):
                    continue        # the union would not be an object either
                boxes[i] = [x0, y0, x1 - x0, y1 - y0]
                groups[i] += groups[j]
                del groups[j], boxes[j]
                merged = True
                break
            if merged:
                break

    out = []
    for box, members in zip(boxes, groups):
        if len(members) == 1:
            out.append(dets[members[0]])
            continue
        # The largest fragment carries the measurements - it is the one with
        # the most pixels behind its contrast and shape numbers - and takes
        # the union box, which is the extent of the whole object.
        lead = max(members, key=lambda k: dets[k].area)
        d = dets[lead]
        d.box = tuple(int(v) for v in box)
        d.area = float(sum(dets[k].area for k in members))
        d.centroid = (box[0] + box[2] / 2.0, box[1] + box[3] / 2.0)
        d.aspect = float(box[2]) / max(box[3], 1)
        d.extent = d.area / max(float(box[2] * box[3]), 1.0)
        out.append(d)
    return out


class OpticalDetector:
    def __init__(self, threshold=DEFAULT_THRESHOLD,
                 min_area=DEFAULT_MIN_AREA_PX, max_area=DEFAULT_MAX_AREA_PX,
                 alpha=BACKGROUND_ALPHA, roi=None, every_n=1,
                 work_scale=None):
        self.threshold = threshold
        # Areas are quoted in FULL-resolution pixels by every caller, so they
        # are converted once here rather than at each comparison.
        self.work_scale = (WORK_SCALE if work_scale is None
                           else float(work_scale))
        self.work_scale = min(max(self.work_scale, 0.1), 1.0)
        area_scale = self.work_scale ** 2
        self.min_area = max(3.0, min_area * area_scale)
        self.max_area = max_area * area_scale
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
        # The measured noise floor and the threshold it produced, both
        # reported, so "the pane is empty" and "the camera is too noisy to
        # see anything" are distinguishable from the console.
        self.noise = 0.0
        self.effective_threshold = float(threshold)

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
        if self.work_scale != 1.0:
            grey = cv2.resize(grey, None, fx=self.work_scale,
                              fy=self.work_scale,
                              interpolation=cv2.INTER_AREA)
        grey = grey.astype(np.float32)

        if self.background is None or self.background.shape != grey.shape:
            self.background = grey.copy()
            self.frames_seen = 1
            self._last = []
            return self._last

        signed = grey - self.background
        diff = np.abs(signed)
        # 1.4826 * MAD is the usual robust stand-in for a standard deviation.
        # Measured against the background model rather than against the frame,
        # so this is the noise in what CHANGED, which is what the threshold is
        # applied to.
        # Every sixteenth pixel. A median is a sort, and sorting three
        # hundred thousand floats every frame to estimate a noise floor that
        # moves with the room lighting is paying a great deal for a digit
        # that does not change between frames.
        self.noise = 1.4826 * float(np.median(diff[::4, ::4]))
        self.effective_threshold = max(self.threshold,
                                       NOISE_SIGMA * self.noise)
        mask = (diff > self.effective_threshold).astype(np.uint8) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE,
                                np.ones((5, 5), np.uint8))
        self.last_mask = mask

        # Same discipline as the thermal side: do not learn where something
        # was detected, or an object that stops dissolves into the model.
        # cv2 with a mask, rather than numpy boolean indexing. The numpy
        # form builds two temporary arrays the size of the frame and does a
        # masked scatter back; this is the same arithmetic in one C pass.
        quiet = cv2.bitwise_not(mask)
        cv2.accumulateWeighted(grey, self.background, self.alpha, mask=quiet)
        self.frames_seen += 1
        if not self.ready:
            self._last = []
            return self._last

        self._last = self._contours(mask, signed, diff)
        return self._last

    @staticmethod
    def _plausible(area, w, h, frame_shape):
        # Everything here is a ratio, so it is scale free and works on the
        # reduced frame exactly as it did on the full one.
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
        inv = 1.0 / self.work_scale
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
                box=(int(x * inv) + ox, int(y * inv) + oy,
                     max(1, int(w * inv)), max(1, int(h * inv))),
                centroid=(cx * inv + ox, cy * inv + oy),
                area=area * inv * inv,
                aspect=float(w) / max(h, 1),
                extent=area / max(float(w * h), 1.0),
                solidity=area / max(hull, 1.0),
                contrast=float(diff[pix].mean()),
                darker=bool(signed[pix].mean() < 0),
            ))
        out = _merge_fragments(out, mask.shape)
        out.sort(key=lambda d: d.area, reverse=True)
        return out
