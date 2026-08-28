"""Turning a pipeline Result into the three views the dashboard shows.

The overlay view is the one that matters. It warps the thermal frame into
optical coordinates so you can see, directly, whether the frame matching is
right - hot regions should sit on the objects that are actually hot. That is
the check that catches a drifted homography before it quietly corrupts every
optical crop downstream.
"""
import cv2
import numpy as np

from ..calibration import homography

COL_DRONE = (60, 60, 235)      # BGR - red
COL_BIRD = (235, 170, 60)      # blue
COL_UNKNOWN = (150, 150, 160)
FONT = cv2.FONT_HERSHEY_SIMPLEX


def colour_for(label):
    return {"drone": COL_DRONE, "bird": COL_BIRD}.get(label, COL_UNKNOWN)


def colourise(celsius, lo=None, hi=None):
    """Degrees C to an INFERNO image. Percentile scaling keeps a single hot
    pixel from crushing the rest of the scene to black."""
    c = np.asarray(celsius, np.float32)
    lo = float(np.percentile(c, 1)) if lo is None else lo
    hi = float(np.percentile(c, 99.5)) if hi is None else hi
    span = max(hi - lo, 0.5)
    u8 = np.clip((c - lo) * (255.0 / span), 0, 255).astype(np.uint8)
    return cv2.applyColorMap(u8, cv2.COLORMAP_INFERNO)


def thermal_view(result, scale=2):
    img = colourise(result.thermal_c)
    for det in result.detections:
        x, y, w, h = det.box
        col = colour_for(det.label)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 1)
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale),
                     interpolation=cv2.INTER_NEAREST)
    for det in result.detections:
        x, y, w, h = [v * scale for v in det.box]
        col = colour_for(det.label)
        tag = f"#{det.track_id} {det.label}"
        if det.confidence:
            tag += f" {det.confidence:.2f}"
        cv2.putText(img, tag, (x, max(12, y - 4)), FONT, 0.4, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, tag, (x, max(12, y - 4)), FONT, 0.4, col, 1,
                    cv2.LINE_AA)
    _banner(img, f"THERMAL 256x192  {result.proc_ms:.2f} ms detect")
    return img


def optical_view(result, H):
    img = result.optical.copy()
    if H is None:
        _banner(img, "OPTICAL - not calibrated")
        return img
    for det in result.detections:
        x, y, w, h = homography.map_box(H, det.box)
        col = colour_for(det.label)
        cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        tag = f"#{det.track_id} {det.label} {det.confidence:.2f}"
        cv2.putText(img, tag, (x, max(14, y - 6)), FONT, 0.5, (0, 0, 0), 3,
                    cv2.LINE_AA)
        cv2.putText(img, tag, (x, max(14, y - 6)), FONT, 0.5, col, 1,
                    cv2.LINE_AA)
    _banner(img, "OPTICAL - thermal detections mapped through H")
    return img


def overlay_view(result, H, alpha=0.45):
    """Thermal warped into optical space. The frame-matching sanity check."""
    img = result.optical.copy()
    if H is None:
        _banner(img, "OVERLAY - calibrate to enable")
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
    corners = homography.map_points(
        H, [[0, 0], [w, 0], [w, h], [0, h]]).astype(np.int32)
    cv2.polylines(img, [corners], True, (0, 235, 235), 2)
    cv2.putText(img, "thermal FOV", tuple(corners[0] + np.array([4, 18])),
                FONT, 0.45, (0, 235, 235), 1, cv2.LINE_AA)
    _banner(img, f"OVERLAY - thermal warped into optical  alpha={alpha:.2f}")
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
