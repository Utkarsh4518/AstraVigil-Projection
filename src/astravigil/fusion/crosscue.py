"""Each sensor asking the other about what it just found.

Both cameras detect on their own, and either can put a question to the other
through the homography. The two questions are not the same question:

  THERMAL -> OPTICAL   "I have a warm object at this thermal box. What shape
                       is it?" Thermal is good at finding small things and
                       nearly blind to what they are, because a bird and a
                       quadcopter are both a handful of warm pixels.

  OPTICAL -> THERMAL   "I have a moving object at this optical box. Is it
                       warm?" Optical is good at shape and hopeless at
                       separating an aircraft from a cloud edge, a shadow or a
                       waving branch. A heat signature settles it instantly.

The second direction is the one that earns the architecture. Thermal-led
detection has two hard blind spots - thermal crossover at dawn and dusk, and a
cold-soaked airframe that has been parked for hours - and in both, optical is
the only sensor still producing a signal. A one-way pipeline has no way to act
on that.

TWO RULES KEEP THIS FROM BEING WORSE THAN NO FUSION AT ALL.

Associate before assessing. If both sensors independently find the same
aircraft and both raise it, the operator gets two alarms for one object, which
is precisely the "wall of separate alarms" the brief exists to complain about.
Detections are paired through the homography first; the pair is judged once.

Keep the evidence independent. Each sensor reports its own raw measurement and
they are combined exactly once. If optical's answer were derived from a region
thermal chose, and thermal's confidence then rose because optical agreed, the
system would be confirming its own guess. Disagreement therefore has to be
allowed to LOWER confidence - a fusion that can only ever increase certainty is
not fusing anything.
"""
import cv2
import numpy as np

from ..calibration import homography

# Below this many pixels on target there is no shape to measure and any
# verdict is noise dressed up as evidence.
MIN_SHAPE_PX = 60
GOOD_SHAPE_PX = 400

# Shape thresholds, applied to the segmented crop.
DRONE_ASPECT = (0.55, 1.80)
DRONE_SOLIDITY = 0.62
BIRD_ASPECT = 2.10
BIRD_SOLIDITY = 0.52

# A thermal query counts as "warm" at this much above the local surround.
WARM_C = 1.2
HOT_CORE_C = 5.0


class OpticalEvidence:
    """Optical's answer to a thermal cue."""

    __slots__ = ("found", "pixels", "aspect", "extent", "solidity", "label",
                 "confidence", "reason", "box")

    def __init__(self, found=False, pixels=0, aspect=0.0, extent=0.0,
                 solidity=0.0, label="unknown", confidence=0.0, reason="",
                 box=None):
        self.found = found
        self.pixels = pixels
        self.aspect = aspect
        self.extent = extent
        self.solidity = solidity
        self.label = label
        self.confidence = confidence
        self.reason = reason
        self.box = box

    @property
    def usable(self):
        return self.found and self.confidence > 0.0

    def as_dict(self):
        return {"found": self.found, "pixels": int(self.pixels),
                "aspect": round(self.aspect, 2),
                "solidity": round(self.solidity, 2),
                "label": self.label,
                "confidence": round(self.confidence, 3),
                "reason": self.reason}


class ThermalEvidence:
    """Thermal's answer to an optical cue."""

    __slots__ = ("found", "peak_c", "mean_c", "contrast_c", "hotspot_c",
                 "warm", "confidence", "reason", "box")

    def __init__(self, found=False, peak_c=0.0, mean_c=0.0, contrast_c=0.0,
                 hotspot_c=0.0, warm=False, confidence=0.0, reason="",
                 box=None):
        self.found = found
        self.peak_c = peak_c
        self.mean_c = mean_c
        self.contrast_c = contrast_c
        self.hotspot_c = hotspot_c
        self.warm = warm
        self.confidence = confidence
        self.reason = reason
        self.box = box

    def as_dict(self):
        return {"found": self.found, "warm": self.warm,
                "peak_c": round(self.peak_c, 1),
                "contrast_c": round(self.contrast_c, 2),
                "hotspot_c": round(self.hotspot_c, 2),
                "confidence": round(self.confidence, 3),
                "reason": self.reason}


def _clip_box(box, shape, margin=0):
    x, y, w, h = box
    H_img, W_img = shape[:2]
    x0 = max(0, int(x) - margin)
    y0 = max(0, int(y) - margin)
    x1 = min(W_img, int(x + w) + margin)
    y1 = min(H_img, int(y + h) + margin)
    if x1 <= x0 or y1 <= y0:
        return None
    return x0, y0, x1, y1


# ------------------------------------------------- thermal asks optical
def verify_optical(optical_bgr, H, thermal_box, margin=10):
    """Thermal has something here. What does it look like?

    Segments the crop against its own local background rather than a global
    threshold, because a drone against bright sky is dark and the same drone
    against dark trees is bright, and a fixed polarity gets one of those
    wrong.
    """
    if optical_bgr is None or H is None:
        return OpticalEvidence(reason="no optical frame or no calibration")

    box = homography.map_box(H, thermal_box)
    clipped = _clip_box(box, optical_bgr.shape, margin)
    if clipped is None:
        return OpticalEvidence(
            reason="thermal detection maps outside the optical frame")
    x0, y0, x1, y1 = clipped
    crop = optical_bgr[y0:y1, x0:x1]
    if crop.size == 0:
        return OpticalEvidence(reason="empty crop")

    grey = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    # Otsu picks the split between object and surround per crop, so it works
    # whether the target is the bright thing or the dark thing.
    _, m1 = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Whichever class is the minority is the object; the crop is mostly
    # background by construction.
    mask = m1 if (m1 > 0).sum() <= m1.size / 2 else cv2.bitwise_not(m1)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return OpticalEvidence(reason="nothing segmented in the crop",
                               box=(x0, y0, x1 - x0, y1 - y0))
    c = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(c)) or float(len(c))
    x, y, w, h = cv2.boundingRect(c)
    hull = float(cv2.contourArea(cv2.convexHull(c))) or area
    aspect = float(w) / max(h, 1)
    extent = area / max(float(w * h), 1.0)
    solidity = area / max(hull, 1.0)

    # Confidence is governed by how many pixels are actually on the target.
    # This is the honest part of the whole module: at the Pi camera's default
    # 640x480 the optical IFOV equals the thermal IFOV, so a distant target
    # lands on just as few pixels here as it did there and optical adds
    # essentially nothing. The number below says so out loud rather than
    # inventing a shape verdict from twelve pixels.
    if area < MIN_SHAPE_PX:
        return OpticalEvidence(
            found=True, pixels=area, aspect=aspect, extent=extent,
            solidity=solidity, label="unknown", confidence=0.0,
            reason=f"only {area:.0f} px on target - too few to judge shape",
            box=(x0, y0, x1 - x0, y1 - y0))

    scale = min(1.0, (area - MIN_SHAPE_PX) / (GOOD_SHAPE_PX - MIN_SHAPE_PX))

    label, strength, why = "unknown", 0.0, "shape is not distinctive"
    if aspect >= BIRD_ASPECT or solidity <= BIRD_SOLIDITY:
        label = "bird"
        strength = 0.55 + 0.35 * min(1.0, (aspect - BIRD_ASPECT) / 1.5
                                     if aspect >= BIRD_ASPECT else
                                     (BIRD_SOLIDITY - solidity) / 0.3)
        why = (f"elongated/ragged outline (aspect {aspect:.2f}, "
               f"solidity {solidity:.2f}) - wings")
    elif (DRONE_ASPECT[0] <= aspect <= DRONE_ASPECT[1]
          and solidity >= DRONE_SOLIDITY):
        label = "drone"
        strength = 0.55 + 0.35 * min(1.0, (solidity - DRONE_SOLIDITY) / 0.3)
        why = (f"compact solid outline (aspect {aspect:.2f}, "
               f"solidity {solidity:.2f}) - rigid airframe")

    return OpticalEvidence(
        found=True, pixels=area, aspect=aspect, extent=extent,
        solidity=solidity, label=label,
        confidence=round(strength * scale, 3),
        reason=f"{why}; {area:.0f} px on target",
        box=(x0, y0, x1 - x0, y1 - y0))


# ------------------------------------------------- optical asks thermal
def verify_thermal(thermal_c, H, optical_box, margin=2, surround=6):
    """Optical has something here. Is it warm?

    Compares the mapped patch against a ring around it rather than against the
    frame as a whole, because the question is local: warm relative to the sky
    behind it, not warm relative to the tarmac at the bottom of the frame.
    """
    if thermal_c is None or H is None:
        return ThermalEvidence(reason="no thermal frame or no calibration")

    try:
        H_inv = np.linalg.inv(np.asarray(H, np.float64))
    except np.linalg.LinAlgError:
        return ThermalEvidence(reason="homography is not invertible")

    box = homography.map_box(H_inv, optical_box)
    clipped = _clip_box(box, thermal_c.shape, margin)
    if clipped is None:
        return ThermalEvidence(
            found=False,
            reason="outside the thermal field of view - cannot be verified")
    x0, y0, x1, y1 = clipped
    patch = thermal_c[y0:y1, x0:x1]
    if patch.size == 0:
        return ThermalEvidence(reason="empty thermal patch")

    ring = _clip_box((x0, y0, x1 - x0, y1 - y0), thermal_c.shape, surround)
    rx0, ry0, rx1, ry1 = ring
    outer = thermal_c[ry0:ry1, rx0:rx1]
    # Median of the surround, with the patch itself still in it. The patch is
    # the minority of that area, so the median is the background.
    background = float(np.median(outer))

    peak = float(patch.max())
    mean = float(patch.mean())
    contrast = mean - background
    hotspot = peak - mean
    warm = contrast >= WARM_C

    if warm and hotspot >= HOT_CORE_C:
        conf, why = 0.85, (f"warm ({contrast:+.1f} C over surround) with a "
                           f"{hotspot:.1f} C core - powered")
    elif warm:
        conf, why = 0.6, f"warm, {contrast:+.1f} C over its surround"
    else:
        conf, why = 0.25, (f"no heat signature ({contrast:+.1f} C) - "
                           f"clutter, or cold-soaked")

    return ThermalEvidence(found=True, peak_c=peak, mean_c=mean,
                           contrast_c=contrast, hotspot_c=hotspot, warm=warm,
                           confidence=conf, reason=why,
                           box=(x0, y0, x1 - x0, y1 - y0))


# ------------------------------------------------------------ association
def associate(thermal_dets, optical_dets, H, tolerance_px=28.0):
    """Pair detections that are the same physical object.

    Runs before anything is assessed, so one aircraft seen by both sensors
    produces one object rather than two alarms.
    """
    pairs, unmatched_optical = {}, list(optical_dets)
    if H is None or not thermal_dets or not optical_dets:
        return pairs, unmatched_optical

    for det in thermal_dets:
        ox, oy, ow, oh = homography.map_box(H, det.box)
        centre = (ox + ow / 2.0, oy + oh / 2.0)
        best, best_d = None, tolerance_px + max(ow, oh) / 2.0
        for o in unmatched_optical:
            d = float(np.hypot(o.centroid[0] - centre[0],
                               o.centroid[1] - centre[1]))
            if d < best_d:
                best, best_d = o, d
        if best is not None:
            pairs[det.track_id] = best
            best.thermal_match = det.track_id
            unmatched_optical.remove(best)
    return pairs, unmatched_optical


class OpticalContactLog:
    """How long each optical-only contact has been sitting where it is.

    The thermal side gets dwell from the tracker and the site baseline. An
    optical-only contact has neither - there is no thermal track to follow and
    no thermal deviation for the site model to persist - so without this it is
    judged fresh on every single frame and can never escalate.

    That would lose exactly the case this whole reverse path exists for: a
    cold-soaked airframe parked on the apron is visible only to the optical
    camera, has no heat signature to confirm, and is distinguished from a
    passing shadow by one thing - it is still there a minute later.

    Keyed on quantised position rather than by tracking, because a stationary
    object does not need a tracker and a moving one is the thermal side's job.
    """

    def __init__(self, cell_px=24, grace_s=2.0):
        self.cell_px = cell_px
        self.grace_s = grace_s
        self.seen = {}          # key -> [first_seen, last_seen]

    def key_for(self, centroid):
        return (f"optical:{int(centroid[1]) // self.cell_px}:"
                f"{int(centroid[0]) // self.cell_px}")

    def update(self, contacts, now):
        """Returns {key: dwell_s} for this frame's contacts."""
        out = {}
        for c in contacts:
            k = self.key_for(c.centroid)
            rec = self.seen.get(k)
            if rec is None or now - rec[1] > self.grace_s:
                # New, or gone long enough that this is a fresh arrival.
                rec = [now, now]
                self.seen[k] = rec
            rec[1] = now
            out[k] = now - rec[0]
        for k, rec in list(self.seen.items()):
            if now - rec[1] > max(self.grace_s * 4, 10.0):
                del self.seen[k]
        return out


# --------------------------------------------------------------- identity
def fuse_identity(label, confidence, optical):
    """Combine the thermal classifier's verdict with optical's shape verdict.

    Returns (label, confidence, note). Agreement raises confidence via a
    noisy-OR; disagreement lowers it and says so. A fusion that could only
    ever increase certainty would be a rubber stamp.
    """
    if optical is None or not optical.usable:
        why = optical.reason if optical is not None else "no optical check"
        return label, confidence, f"thermal only ({why})"

    if optical.label == label:
        fused = 1.0 - (1.0 - confidence) * (1.0 - optical.confidence)
        return label, fused, f"optical agrees: {optical.reason}"

    if optical.label == "unknown":
        return label, confidence, f"optical inconclusive: {optical.reason}"

    if label == "unknown":
        # Thermal could not commit and optical could. Take optical's verdict,
        # but not at full strength - it is one sensor, unsupported.
        return optical.label, optical.confidence * 0.8, \
            f"optical led: {optical.reason}"

    # Genuine disagreement. Keep whichever is stronger, cut the confidence
    # hard, and surface it - two sensors contradicting each other is
    # information, not something to average away.
    if optical.confidence > confidence:
        return optical.label, optical.confidence * 0.5, \
            f"SENSORS DISAGREE - thermal said {label}, optical says " \
            f"{optical.label}"
    return label, confidence * 0.5, \
        f"SENSORS DISAGREE - optical said {optical.label}, thermal holds " \
        f"{label}"
