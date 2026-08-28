"""Thermal <-> optical frame matching.

A homography is a 3x3 that maps one image plane onto another. It is exact when
the scene is planar, or when everything in view is far enough away that depth
differences do not matter. Neither is quite true for an apron watched from a
roof, so the residual error is parallax and it grows with the gap between your
calibration distance and your observing distance - see hardware/README.md.

The important thing this module provides is a way to *score* a calibration.
A homography that is subtly wrong looks fine on screen and quietly puts your
optical crops a few pixels off, which then reads downstream as a classifier
problem. reprojection_error() is what tells you which one you have.
"""
import json
import os

import cv2
import numpy as np

MIN_POINTS = 4


class CalibrationError(RuntimeError):
    pass


def compute(thermal_pts, optical_pts, ransac_px=3.0):
    """Least-squares homography from matched points, thermal -> optical.

    Returns (H, inlier_mask). Four points is the algebraic minimum; use more
    and spread them across the frame, because points clustered in the middle
    leave the corners unconstrained and the corners are where the error shows.
    """
    t = np.asarray(thermal_pts, np.float32).reshape(-1, 1, 2)
    o = np.asarray(optical_pts, np.float32).reshape(-1, 1, 2)
    if len(t) < MIN_POINTS or len(t) != len(o):
        raise CalibrationError(
            f"need at least {MIN_POINTS} matched pairs, got "
            f"{len(t)} thermal and {len(o)} optical")

    if len(t) == MIN_POINTS:
        H = cv2.getPerspectiveTransform(t.reshape(4, 2), o.reshape(4, 2))
        mask = np.ones((4, 1), np.uint8)
    else:
        H, mask = cv2.findHomography(t, o, cv2.RANSAC, ransac_px)

    if H is None:
        raise CalibrationError(
            "homography could not be fitted - points are probably collinear "
            "or mismatched between the two views")
    return H / H[2, 2], mask


def reprojection_error(H, thermal_pts, optical_pts):
    """Per-point error in optical pixels: how far each mapped point misses.

    This is the number to judge a calibration by. Under ~2 px is good; above
    ~5 px something is wrong with the correspondences, not with the maths.
    """
    t = np.asarray(thermal_pts, np.float32).reshape(-1, 1, 2)
    o = np.asarray(optical_pts, np.float32).reshape(-1, 2)
    mapped = cv2.perspectiveTransform(t, H).reshape(-1, 2)
    return np.linalg.norm(mapped - o, axis=1)


def compare(H, H_reference, shape):
    """Mean corner disagreement (px) between two homographies.

    Only meaningful when a reference is known, i.e. in simulation. Comparing
    matrices entry by entry is misleading because they are only defined up to
    scale; comparing where they send the frame corners is not.
    """
    h, w = shape
    corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
    a = cv2.perspectiveTransform(corners, H).reshape(-1, 2)
    b = cv2.perspectiveTransform(corners, H_reference).reshape(-1, 2)
    return float(np.linalg.norm(a - b, axis=1).mean())


def map_points(H, pts):
    """Thermal pixel coordinates -> optical pixel coordinates."""
    p = np.asarray(pts, np.float32).reshape(-1, 1, 2)
    return cv2.perspectiveTransform(p, H).reshape(-1, 2)


def map_box(H, box):
    """Thermal (x, y, w, h) -> axis-aligned optical box.

    The mapped rectangle is generally not axis aligned, so this returns the
    bounding box of the four mapped corners. It is therefore slightly loose,
    which is the right direction to err for a crop.
    """
    x, y, w, h = box
    corners = np.float32([[x, y], [x + w, y], [x + w, y + h], [x, y + h]])
    m = map_points(H, corners)
    x0, y0 = m[:, 0].min(), m[:, 1].min()
    x1, y1 = m[:, 0].max(), m[:, 1].max()
    return int(round(x0)), int(round(y0)), int(round(x1 - x0)), int(round(y1 - y0))


def warp(thermal_img, H, optical_shape):
    """Warp a whole thermal frame into optical coordinates, for the overlay."""
    h, w = optical_shape[:2]
    return cv2.warpPerspective(thermal_img, H, (w, h), flags=cv2.INTER_LINEAR)


def save(path, H, meta=None):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    payload = {"H": np.asarray(H, float).tolist()}
    if meta:
        payload.update(meta)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=2)


def load(path):
    with open(path, encoding="utf-8") as fh:
        return np.array(json.load(fh)["H"], dtype=np.float64)
