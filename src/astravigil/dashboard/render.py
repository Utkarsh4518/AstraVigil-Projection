"""Turning a pipeline Result into the four views the dashboard shows.

Two of them are diagnostics for things that fail silently, which is why they
exist at all:

  OVERLAY warps the thermal frame into optical coordinates, so you can see
  directly whether the frame matching is right - hot regions should sit on
  the objects that are actually hot. That is the check that catches a drifted
  homography before it quietly corrupts every optical crop downstream.

  SITE draws what the baseline has learned: where traffic normally passes and
  which cells are currently off-baseline. An adaptive system that cannot show
  you its own model is a system nobody will trust in a control room, and it is
  also the only way to tell "it has learned the scene" apart from "it has
  learned nothing and is silent for that reason".

Boxes are coloured by threat level rather than by class. The class is still
written in the label, but what an operator needs at a glance is which of the
things on screen matters.
"""
import cv2
import numpy as np

from ..calibration import homography
from ..site_intelligence.optical_baseline import (
    ACTIVITY_FLOOR, ACTIVITY_FULL, Z_ANOMALOUS)
from ..utils.env import env_int

# How the thermal camera is mounted, in 90-degree anticlockwise steps applied
# to the two thermal-space panes before they are shown.
#
# 3 on this rig: the sensor is on its side AND inverted, so a single
# anticlockwise quarter turn left the scene upside down. Three of them - a
# quarter turn clockwise - is what puts it the right way up. Measured on the
# hardware, not derived: which way a camera ends up facing on a bracket is not
# something the code can know.
#
# This is a VIEW setting, not a pipeline one. Detection, tracking, the site
# model and the homography all keep working in the sensor's own coordinates,
# which is what keeps a site baseline learned before the mount was described
# still valid, and what keeps calibrate_homography.py - which fits H in sensor
# space - agreeing with the overlay. Only the pixels an operator looks at are
# turned, along with the boxes drawn on them.
THERMAL_VIEW_ROT = env_int("ASTRAVIGIL_THERMAL_ROT", 3) % 4

# The same idea for the two optical-space panes. 2 on this rig: the Pi camera
# is mounted inverted, so the scene needs a half turn to come up the right way.
#
# Also a VIEW setting. The homography is fitted in the optical camera's own
# coordinates and detections are mapped into them, so nothing upstream of the
# draw call knows or cares which way up the pane is shown.
OPTICAL_VIEW_ROT = env_int("ASTRAVIGIL_OPTICAL_ROT", 2) % 4

# Whether nominal detections get a box drawn on them.
#
# The detector is built for warm movers against cold sky and deliberately
# freezes anything it has detected out of its background model, so a hovering
# target cannot dissolve into it. Indoors that assumption inverts: a room is
# full of things warmer than the wall behind them, every one of them gets
# frozen out on the first frame it crosses the threshold, and the pane fills
# with boxes that never leave.
#
# The threat fusion already handles them correctly - they score nominal and
# raise nothing. What they do is bury the one detection that matters in a
# wall of labels, which is the failure mode the brief asks us not to produce.
# So they are counted rather than drawn. Set ASTRAVIGIL_SHOW_NOMINAL=1 to see
# every box again, which is what you want while tuning a threshold.
SHOW_NOMINAL = env_int("ASTRAVIGIL_SHOW_NOMINAL", 0) != 0

COL_DRONE = (60, 60, 235)      # BGR - red
COL_BIRD = (235, 170, 60)      # blue
COL_UNKNOWN = (150, 150, 160)

COL_ALERT = (55, 55, 240)      # red
COL_WATCH = (40, 175, 235)     # amber
COL_NOMINAL = (120, 200, 130)  # muted green
COL_SETTLED = (220, 90, 220)   # magenta - the site channel, not the detector
COL_OPTICAL = (235, 235, 60)   # cyan - found by the optical camera alone
COL_LEARNT = (205, 195, 70)    # the learning map, same cyan as the site scan
COL_TRAFFIC = (80, 210, 100)   # where change normally happens
FONT = cv2.FONT_HERSHEY_SIMPLEX

# How many cue numbers the site panes will draw at once.
#
# These panes do NOT filter by threat level the way the picture panes do.
# While a site is being learned every object scores nominal by construction,
# so a level filter would empty the pane at exactly the moment somebody is
# watching it to see what the model is being fed. A cap bounds the clutter
# instead, highest threat first, which degrades sensibly in a busy room.
SITE_BADGES = 8


def colour_for(label):
    return {"drone": COL_DRONE, "bird": COL_BIRD}.get(label, COL_UNKNOWN)


def colour_for_level(level):
    return {"alert": COL_ALERT, "watch": COL_WATCH}.get(level, COL_NOMINAL)


def assessment_map(result):
    """track_id -> Assessment, for the views that draw detections."""
    return {a.track_id: a for a in result.assessments if a.kind == "track"}


def colourise(celsius, lo=None, hi=None):
    """Degrees C to an INFERNO image. Percentile scaling keeps a single hot
    pixel from crushing the rest of the scene to black."""
    c = np.asarray(celsius, np.float32)
    lo = float(np.percentile(c, 1)) if lo is None else lo
    hi = float(np.percentile(c, 99.5)) if hi is None else hi
    span = max(hi - lo, 0.5)
    u8 = np.clip((c - lo) * (255.0 / span), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def _rot_image(img, k=None):
    # np.rot90 returns a view with negative strides, which several cv2 calls
    # reject outright. Make it contiguous here rather than at each use.
    k = THERMAL_VIEW_ROT if k is None else k
    return img if not k else np.ascontiguousarray(np.rot90(img, k))


def _rot_point(x, y, w, h, k=None):
    """Where (x, y) lands after _rot_image on a frame of size (h, w).

    One anticlockwise quarter turn sends column x to row w-1-x and row y to
    column y, and swaps the frame's own dimensions - so the step has to carry
    w and h along with it to compose correctly for k of 2 and 3.
    """
    k = THERMAL_VIEW_ROT if k is None else k
    for _ in range(k):
        x, y, w, h = y, w - 1 - x, h, w
    return x, y


def _rot_box(box, w, h, k=None):
    # Two opposite corners survive rotation; the rectangle is whatever they
    # bound afterwards. Rotating the origin alone would put the box in the
    # right place with the wrong extent for odd k.
    #
    # Both corners must be INCLUSIVE. _rot_point mirrors about w - 1, so
    # handing it the exclusive far corner lands the box one pixel out on every
    # odd quarter turn - right for k of 0 and 2, and quietly wrong for the one
    # rotation this rig actually uses.
    x, y, bw, bh = box
    ax, ay = _rot_point(x, y, w, h, k)
    bx, by = _rot_point(x + max(bw - 1, 0), y + max(bh - 1, 0), w, h, k)
    return min(ax, bx), min(ay, by), abs(bx - ax) + 1, abs(by - ay) + 1


def _cues(result):
    """key -> the small number every pane draws on that object."""
    return getattr(result, "cue_numbers", None) or {}


def _mark(img, n, text, x, y, col, scale=0.42):
    """The object's cue number, and its label to the right of it.

    One number, on every pane, for one object. The keys underneath are the
    right thing for code and hopeless on a screen - `optical:12:31` in the
    cross-cue trail and `#214` on the picture are the same contact, and
    nothing visible says so. A number an operator can say out loud is what
    makes five panes, a trail and a table describe one scene.

    y is the TOP of the badge, not a text baseline, because the caller is
    positioning it against the top edge of a box.
    """
    bh = 17
    x = int(max(0, min(x, img.shape[1] - 10)))
    y = int(max(0, min(y, img.shape[0] - bh - 2)))
    if n is not None:
        t = str(n)
        (tw, _), _ = cv2.getTextSize(t, FONT, 0.48, 1)
        bw = tw + 9
        cv2.rectangle(img, (x, y), (x + bw, y + bh), (14, 14, 18), -1)
        cv2.rectangle(img, (x, y), (x + bw, y + bh), col, 1)
        cv2.putText(img, t, (x + 5, y + bh - 5), FONT, 0.48, col, 1,
                    cv2.LINE_AA)
        x += bw + 4
    if text:
        _tag(img, text, x, y + bh - 5, col, scale)


def _tag(img, text, x, y, col, scale=0.4):
    cv2.putText(img, text, (x, max(12, y)), FONT, scale, (0, 0, 0), 3,
                cv2.LINE_AA)
    cv2.putText(img, text, (x, max(12, y)), FONT, scale, col, 1, cv2.LINE_AA)


def thermal_view(result, scale=2):
    img = colourise(result.thermal_c)
    seen = assessment_map(result)
    fh, fw = result.thermal_c.shape

    hidden = 0
    for det in result.detections:
        a = seen.get(det.track_id)
        if not _worth_drawing(a):
            hidden += 1
            continue
        x, y, w, h = det.box
        cv2.rectangle(img, (x, y), (x + w, y + h),
                      colour_for_level(a.level if a else "nominal"), 1)
    for an in result.static_anomalies:
        x, y, w, h = an.box
        cv2.rectangle(img, (x, y), (x + w, y + h), COL_SETTLED, 1)

    # Turn the picture and its boxes together, then scale. The labels are
    # placed afterwards, in rotated coordinates, so that the text itself stays
    # the right way up - a rotated frame with sideways writing on it is worse
    # to read than an unrotated one.
    img = _rot_image(img)
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                     interpolation=cv2.INTER_NEAREST)

    cues = _cues(result)
    for det in result.detections:
        a = seen.get(det.track_id)
        if not _worth_drawing(a):
            continue
        rx, ry, _, _ = _rot_box(det.box, fw, fh)
        x, y = rx * scale, ry * scale
        col = colour_for_level(a.level if a else "nominal")
        tag = det.label
        if a is not None:
            tag += f" t{a.threat:.2f}"
            if a.dwell_s >= 3:
                tag += f" [{a.dwell_s:.0f}s still]"
        _mark(img, cues.get(f"track:{det.track_id}"), tag, x, y - 19, col)
    for an in result.static_anomalies:
        rx, ry, _, _ = _rot_box(an.box, fw, fh)
        _mark(img, cues.get(an.key), f"SETTLED {an.dwell_s:.0f}s",
              rx * scale, ry * scale - 19, COL_SETTLED)

    note = f"  +{hidden} nominal" if hidden else ""
    _banner(img,
            f"THERMAL {fw}x{fh}  {result.proc_ms:.2f} ms detect{note}")
    return img


def _worth_drawing(assessment):
    """A box on the picture is a claim on the operator's attention.

    Nominal detections have already been judged as belonging here. Drawing
    them anyway does not add information - it spends the one thing the pane
    is for, which is showing what does not belong.
    """
    if SHOW_NOMINAL or assessment is None:
        return True
    return assessment.level != "nominal"


def _uncalibrated_banner(result, what):
    """What to say on a pane that has no homography yet.

    "not calibrated" is a true statement that gives the operator nothing to do.
    While the automatic calibrator is running there is something far more
    useful to show: whether it is making progress, and what would help it -
    which is almost always "walk something across more of the view".
    """
    auto = (getattr(result, "cross", None) or {}).get("auto_calibration")
    if not auto:
        return f"{what} - not calibrated"
    return f"{what} - auto-calibrating: {auto['reason']}"


def optical_view(result, H):
    img = result.optical.copy()
    fh, fw = img.shape[:2]          # before rotation - _rot_box needs these
    if H is None:
        img = _rot_image(img, OPTICAL_VIEW_ROT)
        _banner(img, _uncalibrated_banner(result, "OPTICAL"))
        return img

    seen = assessment_map(result)

    # Pass one: rectangles, drawn in the camera's own coordinates.
    roi = getattr(result, "optical_roi", None)
    if roi is not None:
        rx, ry, rw, rh = roi
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), (90, 90, 100), 1)

    optical_only = [od for od in result.optical_detections
                    if od.thermal_match is None]
    for od in optical_only:
        x, y, w, h = od.box
        cv2.rectangle(img, (x, y), (x + w, y + h), COL_OPTICAL, 2)

    mapped = []
    for det in result.detections:
        box = homography.map_box(H, det.box)
        x, y, w, h = box
        a = seen.get(det.track_id)
        col = colour_for_level(a.level if a else "nominal")
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        mapped.append((box, det, a, col))

    statics = []
    for an in result.static_anomalies:
        box = homography.map_box(H, an.box)
        x, y, w, h = box
        cv2.rectangle(img, (x, y), (x + w, y + h), COL_SETTLED, 2)
        statics.append((box, an))

    # Turn picture and boxes together, then label in rotated coordinates so
    # the writing stays the right way up.
    img = _rot_image(img, OPTICAL_VIEW_ROT)

    cues = _cues(result)
    ocues = getattr(result, "optical_cues", None) or {}
    for od in optical_only:
        rx, ry, _, _ = _rot_box(od.box, fw, fh, OPTICAL_VIEW_ROT)
        _mark(img, ocues.get(tuple(od.box)), "OPTICAL ONLY", rx, ry - 21,
              COL_OPTICAL, 0.45)

    for box, det, a, col in mapped:
        rx, ry, _, _ = _rot_box(box, fw, fh, OPTICAL_VIEW_ROT)
        tag = f"{det.label} {det.confidence:.2f}"
        if a is not None:
            tag += f"  threat {a.threat:.2f}"
        ev = result.optical_evidence.get(det.track_id)
        if ev is not None and ev.usable:
            tag += f"  | optical: {ev.label} {ev.confidence:.2f}"
        elif ev is not None:
            tag += "  | optical: no shape"
        _mark(img, cues.get(f"track:{det.track_id}"), tag, rx, ry - 21, col,
              0.45)

    for box, an in statics:
        rx, ry, _, _ = _rot_box(box, fw, fh, OPTICAL_VIEW_ROT)
        _mark(img, cues.get(an.key), f"SETTLED OBJECT {an.dwell_s:.0f}s",
              rx, ry - 21, COL_SETTLED, 0.45)

    c = result.cross or {}
    _banner(img, f"OPTICAL - {c.get('paired_with_thermal', 0)} paired with "
                 f"thermal, {c.get('optical_only', 0)} optical-only, "
                 f"{c.get('shape_usable', 0)}/{c.get('shape_checks', 0)} "
                 f"shape checks usable")
    return img


def overlay_view(result, H, alpha=0.6):
    """Thermal warped into optical space. The frame-matching sanity check.

    Only the WARM part of the thermal frame is drawn. Blending the whole of it
    put a large flat orange rectangle over most of the optical image: the cold
    background of a thermal frame is most of its area, it carries no
    information here, and tinting it hid the very thing the pane exists to
    check. What tells you the homography is right is a hot patch landing on
    the object that is actually hot - so the hot patch is drawn, at an opacity
    that rises with temperature, and everything below the warm threshold is
    left as clean optical picture.

    The white outline is the edge of that warm region. Registration is a
    judgement about whether two edges coincide, and an edge is far easier to
    judge than the middle of a gradient.
    """
    img = result.optical.copy()
    if H is None:
        img = _rot_image(img, OPTICAL_VIEW_ROT)
        _banner(img, _uncalibrated_banner(result, "OVERLAY"))
        return img

    c = result.thermal_c
    # Warm relative to THIS scene, not to an absolute temperature: indoors
    # everything is warm, outdoors nothing is, and a fixed threshold is wrong
    # in one of those two places every time.
    lo = float(np.percentile(c, 75))
    hi = float(np.percentile(c, 99.5))
    weight = np.clip((c - lo) / max(hi - lo, 0.5), 0.0, 1.0)
    warped = homography.warp(colourise(c, lo, hi), H, img.shape)
    wmap = homography.warp((weight * 255).astype(np.uint8), H, img.shape)

    # Per-pixel blend in uint8 rather than float32. The same arithmetic, on
    # SIMD paths, over three quarters of a megapixel every rendered frame -
    # the float version measured four times slower for an identical picture,
    # and this is a Pi.
    w3 = cv2.merge([cv2.convertScaleAbs(wmap, alpha=alpha)] * 3)
    img = cv2.add(cv2.multiply(img, cv2.bitwise_not(w3), scale=1.0 / 255),
                  cv2.multiply(warped, w3, scale=1.0 / 255))

    contours, _ = cv2.findContours((wmap > 130).astype(np.uint8),
                                   cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, (255, 255, 255), 1)

    h, w = result.thermal_c.shape
    corners = homography.map_points(H, [[0, 0], [w, 0], [w, h], [0, h]])

    fh, fw = img.shape[:2]          # before rotation
    boxes = [(homography.map_box(H, d.box), d) for d in result.detections]
    statics = [(homography.map_box(H, an.box), an)
               for an in result.static_anomalies]

    img = _rot_image(img, OPTICAL_VIEW_ROT)
    corners = np.array(
        [_rot_point(int(px), int(py), fw, fh, OPTICAL_VIEW_ROT)
         for px, py in corners], np.int32)
    cv2.polylines(img, [corners], True, (0, 200, 200), 1)

    # The numbered boxes are the check itself: box 3 should be drawn around
    # the warm patch that is object 3, and if it is not, the homography has
    # drifted and every optical crop downstream is being taken from the wrong
    # place.
    cues = _cues(result)
    seen = assessment_map(result)
    for box, det in boxes:
        a = seen.get(det.track_id)
        if not _worth_drawing(a):
            continue
        rx, ry, rw, rh = _rot_box(box, fw, fh, OPTICAL_VIEW_ROT)
        col = colour_for_level(a.level if a else "nominal")
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), col, 1)
        _mark(img, cues.get(f"track:{det.track_id}"), "", rx, ry - 19, col)
    for box, an in statics:
        rx, ry, rw, rh = _rot_box(box, fw, fh, OPTICAL_VIEW_ROT)
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), COL_SETTLED, 1)
        _mark(img, cues.get(an.key), "", rx, ry - 19, COL_SETTLED)

    tl = corners[np.argmin(corners.sum(axis=1))]
    cv2.putText(img, "thermal FOV", tuple(tl + np.array([4, 18])),
                FONT, 0.45, (0, 200, 200), 1, cv2.LINE_AA)
    _banner(img, "OVERLAY - warm parts of the thermal frame, in optical "
                 "space. They should sit on what is actually hot")
    return img


def site_view(result, site, scale=2):
    """What the site model has learned, and where it currently disagrees.

    Green is learned traffic: the brighter the cell, the more often something
    has moved through it. Magenta is a cell that has been off its learned
    temperature long enough to be an object rather than a fly-past. An empty
    green map means the model has seen nothing yet and its silence means
    nothing either - which is exactly why this view is here.
    """
    img = colourise(result.thermal_c)

    rate = cv2.resize(site._rate, (img.shape[1], img.shape[0]),
                      interpolation=cv2.INTER_NEAREST)
    # Square root so a rarely-used corner is still visible next to a corridor
    # that carries most of the traffic.
    norm = np.sqrt(np.clip(rate / max(rate.max(), 1e-6), 0, 1))
    green = np.zeros_like(img)
    green[:, :, 1] = (norm * 200).astype(np.uint8)
    img = cv2.addWeighted(img, 1.0, green, 0.55, 0)

    settled = cv2.resize(
        (site.persist >= site.persist_frames).astype(np.uint8),
        (img.shape[1], img.shape[0]), interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(settled, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, COL_SETTLED, 1)

    # Same turn as the thermal pane. These two are read side by side, and a
    # site map at ninety degrees to the picture it explains is worse than no
    # site map at all.
    fh, fw = result.thermal_c.shape
    img = _rot_image(img)
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                     interpolation=cv2.INTER_NEAREST)
    cues = _cues(result)
    for an in result.static_anomalies:
        x, y, w, h = [v * scale for v in _rot_box(an.box, fw, fh)]
        cv2.rectangle(img, (x, y), (x + w, y + h), COL_SETTLED, 1)
        _mark(img, cues.get(an.key),
              f"{an.dwell_s:.0f}s  {an.peak_dev_c:+.1f}C", x, y - 19,
              COL_SETTLED)

    # Number whatever the model is currently being fed, highest threat first.
    # This is the pane an operator watches while a site is being learned, and
    # "which of these is number 4" is the question the trail below leaves them
    # holding.
    seen = assessment_map(result)
    ranked = sorted(result.detections, reverse=True,
                    key=lambda d: (seen[d.track_id].threat
                                   if d.track_id in seen else 0.0))
    for det in ranked[:SITE_BADGES]:
        a = seen.get(det.track_id)
        x, y, w, h = [v * scale for v in _rot_box(det.box, fw, fh)]
        col = colour_for_level(a.level if a else "nominal")
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)
        _mark(img, cues.get(f"track:{det.track_id}"), "", x, y - 19, col)

    s = site.stats()
    state = ("LEARNING" if s["learning"] else "LEARNED")
    _banner(img, f"SITE {state}  scene {s['scene_maturity']*100:.0f}%  "
                 f"activity {s['activity_maturity']*100:.0f}%  "
                 f"off-baseline cells {s['anomalous_cells']}")
    return img


def optical_site_view(result, site, scale=1):
    """What the OPTICAL camera has learned about this place.

    The mirror of site_view. That pane shows which cells are at a temperature
    the thermal camera did not expect; this one shows which cells do not look
    the way the optical camera has learned they look. It is the half that sees
    the object with no heat signature at all, and a baseline nobody can
    inspect is one nobody has reason to trust.

    Drawn as the grid it actually is, one rectangle per cell, rather than as a
    smooth tint over the picture. Two earlier versions both came out as a flat
    green wash over the whole frame and told an operator nothing:

      COVERAGE was uniform by construction. Every cell saw every frame, so
      every cell had identical history - the map could not vary. (It can now:
      history accrues at the rate a cell is really learning, so a cell with
      something standing in it visibly lags.)

      ACTIVITY was normalised against the busiest cell, and dividing by the
      maximum guarantees something is at full brightness. A view where nothing
      whatsoever happens rendered exactly as green as a doorway.

    So: while learning, green is per-cell history and the outlined cells have
    no model yet. Once learned, green is where change normally happens, on an
    absolute scale - if nothing has moved through, the map is legitimately
    empty and says so. Magenta has looked wrong long enough to be an object;
    amber is off baseline right now.
    """
    if result.optical is None:
        return None
    img = result.optical.copy()
    if site is None:
        img = _rot_image(img, OPTICAL_VIEW_ROT)
        _banner(img, "OPTICAL SITE - no optical frame learned yet")
        return img

    fh, fw = img.shape[:2]
    cell = site.cell_px
    learning = site.learning
    if learning:
        field, col = site.coverage(), COL_LEARNT
    else:
        field = np.clip((site.activity - ACTIVITY_FLOOR)
                        / max(ACTIVITY_FULL - ACTIVITY_FLOOR, 1e-6), 0, 1)
        col = COL_TRAFFIC

    over = img.copy()
    lit = 0
    for gy, gx in zip(*np.nonzero(field > 0.02)):
        v = float(field[gy, gx])
        x0, y0 = int(gx) * cell, int(gy) * cell
        # A floor under the brightness so a cell that only just clears the
        # threshold is still visible, and the gap between rectangles so the
        # result reads as a grid of measurements rather than a stain.
        shade = tuple(int(round(ch * (0.3 + 0.7 * v))) for ch in col)
        cv2.rectangle(over, (x0 + 1, y0 + 1),
                      (x0 + cell - 2, y0 + cell - 2), shade, -1)
        lit += 1
    img = cv2.addWeighted(img, 0.5, over, 0.5, 0)

    if learning:
        # The holes. A cell with no history has no opinion, and an operator
        # needs to know which part of the frame is not being watched yet
        # rather than reading its silence as quiet.
        for gy, gx in zip(*np.nonzero(field < 0.05)):
            x0, y0 = int(gx) * cell, int(gy) * cell
            cv2.rectangle(img, (x0 + 1, y0 + 1),
                          (x0 + cell - 2, y0 + cell - 2), (72, 72, 84), 1)

    # Magenta outline: off baseline long enough to be an object. Outlined
    # rather than filled, so what is underneath stays readable - the operator
    # needs to see WHAT is sitting there.
    settled = cv2.resize(
        (site.persist >= site.persist_frames).astype(np.uint8), (fw, fh),
        interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(settled, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, COL_SETTLED, 2)

    # Amber outline: off baseline right now, but not for long enough to
    # count. Movement in progress rather than something that has arrived.
    live = cv2.resize((site.z_map() > Z_ANOMALOUS).astype(np.uint8), (fw, fh),
                      interpolation=cv2.INTER_NEAREST)
    live = cv2.subtract(live, settled)
    contours, _ = cv2.findContours(live, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, COL_WATCH, 1)

    img = _rot_image(img, OPTICAL_VIEW_ROT)
    if scale != 1:
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)

    # The optical camera's own contacts, numbered the same as everywhere else.
    ocues = getattr(result, "optical_cues", None) or {}
    for od in result.optical_detections:
        if od.thermal_match is not None:
            continue
        rx, ry, rw, rh = [v * scale for v in
                          _rot_box(od.box, fw, fh, OPTICAL_VIEW_ROT)]
        cv2.rectangle(img, (rx, ry), (rx + rw, ry + rh), COL_OPTICAL, 1)
        _mark(img, ocues.get(tuple(od.box)), "", rx, ry - 19, COL_OPTICAL)

    st = site.stats()
    if learning:
        _banner(img, f"OPTICAL SITE LEARNING {st['maturity'] * 100:.0f}%  "
                     f"green = history learned, outlined = no model yet "
                     f"({st['blind_cells']} cells)")
    elif not lit:
        _banner(img, "OPTICAL SITE LEARNED - nothing has moved through this "
                     "view yet, so there is no traffic to draw")
    else:
        _banner(img, f"OPTICAL SITE LEARNED  green = where change normally "
                     f"happens ({lit} cells)  off-baseline "
                     f"{st['anomalous_cells']}  settled "
                     f"{st['settled_cells']}")
    return img


def _banner(img, text):
    h = img.shape[0]
    cv2.rectangle(img, (0, h - 22), (img.shape[1], h), (18, 18, 22), -1)
    cv2.putText(img, text, (8, h - 7), FONT, 0.45, (215, 215, 225), 1,
                cv2.LINE_AA)


def encode_jpeg(img, quality=80):
    ok, buf = cv2.imencode(".jpg", img,
                           [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return buf.tobytes() if ok else None
