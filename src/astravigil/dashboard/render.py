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
    MIN_LEARNED_FRAMES, Z_ANOMALOUS)
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
FONT = cv2.FONT_HERSHEY_SIMPLEX


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

    for det in result.detections:
        a = seen.get(det.track_id)
        if not _worth_drawing(a):
            continue
        rx, ry, _, _ = _rot_box(det.box, fw, fh)
        x, y = rx * scale, ry * scale
        col = colour_for_level(a.level if a else "nominal")
        tag = f"#{det.track_id} {det.label}"
        if a is not None:
            tag += f" t{a.threat:.2f}"
            if a.dwell_s >= 3:
                tag += f" [{a.dwell_s:.0f}s still]"
        _tag(img, tag, x, y - 4, col)
    for an in result.static_anomalies:
        rx, ry, _, _ = _rot_box(an.box, fw, fh)
        _tag(img, f"SETTLED {an.dwell_s:.0f}s", rx * scale, ry * scale - 4,
             COL_SETTLED)

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

    for od in optical_only:
        rx, ry, _, _ = _rot_box(od.box, fw, fh, OPTICAL_VIEW_ROT)
        _tag(img, "OPTICAL ONLY", rx, ry - 6, COL_OPTICAL, 0.45)

    for box, det, a, col in mapped:
        rx, ry, _, _ = _rot_box(box, fw, fh, OPTICAL_VIEW_ROT)
        tag = f"#{det.track_id} {det.label} {det.confidence:.2f}"
        if a is not None:
            tag += f"  threat {a.threat:.2f}"
        ev = result.optical_evidence.get(det.track_id)
        if ev is not None and ev.usable:
            tag += f"  | optical: {ev.label} {ev.confidence:.2f}"
        elif ev is not None:
            tag += "  | optical: no shape"
        _tag(img, tag, rx, ry - 6, col, 0.5)

    for box, an in statics:
        rx, ry, _, _ = _rot_box(box, fw, fh, OPTICAL_VIEW_ROT)
        _tag(img, f"SETTLED OBJECT {an.dwell_s:.0f}s", rx, ry - 6,
             COL_SETTLED, 0.5)

    c = result.cross or {}
    _banner(img, f"OPTICAL - {c.get('paired_with_thermal', 0)} paired with "
                 f"thermal, {c.get('optical_only', 0)} optical-only, "
                 f"{c.get('shape_usable', 0)}/{c.get('shape_checks', 0)} "
                 f"shape checks usable")
    return img


def overlay_view(result, H, alpha=0.45):
    """Thermal warped into optical space. The frame-matching sanity check."""
    img = result.optical.copy()
    if H is None:
        img = _rot_image(img, OPTICAL_VIEW_ROT)
        _banner(img, _uncalibrated_banner(result, "OVERLAY"))
        return img

    hot = colourise(result.thermal_c)
    warped = homography.warp(hot, H, img.shape)

    # Only blend where the thermal frame actually lands, so the rest of the
    # optical image stays untinted and the registration is easy to judge.
    coverage = homography.warp(
        np.full(result.thermal_c.shape, 255, np.uint8), H, img.shape)
    m = (coverage > 0)[:, :, None]
    img = np.where(m, cv2.addWeighted(img, 1 - alpha, warped, alpha, 0), img)

    h, w = result.thermal_c.shape
    corners = homography.map_points(H, [[0, 0], [w, 0], [w, h], [0, h]])

    fh, fw = img.shape[:2]          # before rotation
    img = _rot_image(img, OPTICAL_VIEW_ROT)
    corners = np.array(
        [_rot_point(int(px), int(py), fw, fh, OPTICAL_VIEW_ROT)
         for px, py in corners], np.int32)

    cv2.polylines(img, [corners], True, (0, 235, 235), 2)
    # Label the corner that is now top-left, or the caption ends up in the
    # middle of the frame after an odd turn.
    tl = corners[np.argmin(corners.sum(axis=1))]
    cv2.putText(img, "thermal FOV", tuple(tl + np.array([4, 18])),
                FONT, 0.45, (0, 235, 235), 1, cv2.LINE_AA)
    _banner(img, f"OVERLAY - thermal warped into optical  alpha={alpha:.2f}")
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
    for an in result.static_anomalies:
        x, y, w, h = [v * scale for v in _rot_box(an.box, fw, fh)]
        cv2.rectangle(img, (x, y), (x + w, y + h), COL_SETTLED, 1)
        _tag(img, f"{an.dwell_s:.0f}s  {an.peak_dev_c:+.1f}C", x, y - 4,
             COL_SETTLED)

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
    the way the optical camera has learned they look.

    It exists because the model was working invisibly. A baseline nobody can
    inspect is one nobody has reason to trust - the same argument that put the
    thermal site pane on the screen - and this half is the one that sees the
    object with no heat signature at all.

    Green is where change normally happens - the optical equivalent of the
    learned traffic on the thermal pane. Coverage was the obvious thing to
    draw and it is useless: once the model matures every cell has full
    history, so the map is uniform and washes the whole frame flat green.
    What an operator needs is the sparse thing, not the saturated one.

    Magenta has looked wrong long enough to be an object; amber is off
    baseline right now.
    """
    if result.optical is None:
        return None
    img = result.optical.copy()
    if site is None:
        img = _rot_image(img, OPTICAL_VIEW_ROT)
        _banner(img, "OPTICAL SITE - no optical frame learned yet")
        return img

    h, w = img.shape[:2]

    # While it is still learning, coverage IS the interesting map - it shows
    # the model filling in, and an empty corner is the operator's cue that
    # part of the frame has not been modelled yet.
    if site.learning:
        field = np.clip(site.ref_n / max(MIN_LEARNED_FRAMES, 1), 0, 1)
    else:
        # Once mature, coverage is uniform and tells you nothing. Switch to
        # where things actually happen, normalised against the busiest cell so
        # a quiet corner is still visible next to a doorway.
        field = site.activity / max(float(site.activity.max()), 1e-6)
    field = cv2.resize(np.sqrt(np.clip(field, 0, 1)), (w, h),
                       interpolation=cv2.INTER_NEAREST)
    green = np.zeros_like(img)
    green[:, :, 1] = (field * 190).astype(np.uint8)
    img = cv2.addWeighted(img, 1.0, green, 0.40, 0)

    # Magenta outline: cells that have been off baseline long enough to be an
    # object. Outlined rather than filled, so what is underneath stays
    # readable - the operator needs to see WHAT is sitting there.
    settled = cv2.resize(
        (site.persist >= site.persist_frames).astype(np.uint8), (w, h),
        interpolation=cv2.INTER_NEAREST)
    contours, _ = cv2.findContours(settled, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, COL_SETTLED, 2)

    # Amber outline: off baseline right now, but not for long enough to
    # count. Movement in progress rather than something that has arrived.
    live = cv2.resize((site.z_map() > Z_ANOMALOUS).astype(np.uint8), (w, h),
                      interpolation=cv2.INTER_NEAREST)
    live = cv2.subtract(live, settled)
    contours, _ = cv2.findContours(live, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img, contours, -1, COL_WATCH, 1)

    img = _rot_image(img, OPTICAL_VIEW_ROT)
    if scale != 1:
        img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                         interpolation=cv2.INTER_NEAREST)

    st = site.stats()
    if st["learning"]:
        _banner(img, f"OPTICAL SITE LEARNING  coverage "
                     f"{st['maturity'] * 100:.0f}%  (green fills in as each "
                     f"cell gains history)")
    else:
        _banner(img, f"OPTICAL SITE LEARNED  green = where change normally "
                     f"happens ({st['active_cells']} cells)  off-baseline "
                     f"{st['anomalous_cells']}  settled {st['settled_cells']}")
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
