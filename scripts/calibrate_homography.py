#!/usr/bin/env python3
"""Pick matching points in the thermal and optical views, solve for H, save it.

    python scripts/calibrate_homography.py --source synthetic     # rehearse
    python3 scripts/calibrate_homography.py --source hardware     # on the Pi

Click the SAME physical point in the left (thermal) then right (optical) pane,
alternating. Keys: u undo, s save, f freeze/unfreeze, q quit.

An OpenCV window rather than the web UI, because this is a setup task done at
the rig with a display, and pixel-accurate clicking through MJPEG is worse.

Two things decide whether a calibration is any good:

  WHAT YOU AIM AT   paper targets are invisible in thermal. Use point heat
                    sources - a mug, a lamp, a hand held still - that are
                    unambiguous in BOTH views.

  HOW FAR AWAY      a homography is exact only at its calibration depth.
                    Calibrating at 3 m and then watching at 50 m builds in
                    about 9 px of error at this baseline, before any noise.
                    Calibrate at roughly the range you intend to watch.

Spread the points across the frame and into the corners. Points clustered in
the middle leave the corners unconstrained, and the corners are where the
error shows up.
"""
import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil import sources                          # noqa: E402
from astravigil.calibration import homography           # noqa: E402
from astravigil.dashboard import render                 # noqa: E402
from astravigil.drivers.thermal.calibration import calibrate_frame  # noqa: E402

OUT = "data/calibration/H.json"
SCALE = 2                      # thermal upscale, for clickable pixels
WIN = "AstraVigil calibration  -  click thermal, then optical"


class Picker:
    def __init__(self):
        self.thermal_pts = []
        self.optical_pts = []
        self.pending = None    # a thermal click waiting for its optical pair
        self.split = 0

    def on_mouse(self, event, x, y, flags, _):
        if event != cv2.EVENT_LBUTTONDOWN:
            return
        if x < self.split:
            self.pending = (x / SCALE, y / SCALE)
        elif self.pending is not None:
            self.thermal_pts.append(self.pending)
            self.optical_pts.append((x - self.split, y))
            self.pending = None

    def undo(self):
        if self.pending is not None:
            self.pending = None
        elif self.thermal_pts:
            self.thermal_pts.pop()
            self.optical_pts.pop()


def draw(thermal_bgr, optical, picker, msg):
    t = cv2.resize(thermal_bgr, (thermal_bgr.shape[1] * SCALE,
                                 thermal_bgr.shape[0] * SCALE),
                   interpolation=cv2.INTER_NEAREST)
    h = max(t.shape[0], optical.shape[0])
    canvas = np.zeros((h + 34, t.shape[1] + optical.shape[1], 3), np.uint8)
    canvas[:t.shape[0], :t.shape[1]] = t
    canvas[:optical.shape[0], t.shape[1]:] = optical
    picker.split = t.shape[1]

    for i, (tp, op) in enumerate(zip(picker.thermal_pts, picker.optical_pts)):
        col = ((37 * i) % 255, (91 * i + 80) % 255, (53 * i + 160) % 255)
        a = (int(tp[0] * SCALE), int(tp[1] * SCALE))
        b = (int(op[0]) + picker.split, int(op[1]))
        for p in (a, b):
            cv2.drawMarker(canvas, p, col, cv2.MARKER_CROSS, 13, 2)
            cv2.circle(canvas, p, 9, col, 1)
        cv2.putText(canvas, str(i + 1), (a[0] + 9, a[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)
        cv2.putText(canvas, str(i + 1), (b[0] + 9, b[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, col, 1, cv2.LINE_AA)

    if picker.pending is not None:
        p = (int(picker.pending[0] * SCALE), int(picker.pending[1] * SCALE))
        cv2.drawMarker(canvas, p, (0, 255, 255), cv2.MARKER_TILTED_CROSS, 15, 2)
        cv2.putText(canvas, "now click the same point on the right",
                    (p[0] + 12, p[1] + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                    (0, 255, 255), 1, cv2.LINE_AA)

    cv2.line(canvas, (picker.split, 0), (picker.split, h), (60, 60, 70), 1)
    cv2.rectangle(canvas, (0, h), (canvas.shape[1], h + 34), (18, 18, 22), -1)
    cv2.putText(canvas, msg, (10, h + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                (215, 215, 225), 1, cv2.LINE_AA)
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="synthetic",
                    choices=["synthetic", "replay", "hardware"])
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--optical", default=None)
    ap.add_argument("--swap-rb", action="store_true",
                    help="flip red/blue on the optical feed if colours look "
                         "inverted")
    args = ap.parse_args()

    kw = {}
    if args.source == "replay":
        kw["optical"] = args.optical
    elif args.source == "hardware":
        kw["optical_index"] = int(args.optical or 0)
        kw["swap_rb"] = args.swap_rb
    src = sources.create(args.source, **kw)

    picker = Picker()
    cv2.namedWindow(WIN, cv2.WINDOW_AUTOSIZE)
    cv2.setMouseCallback(WIN, picker.on_mouse)
    print(__doc__)

    frozen = None
    H = None
    try:
        while True:
            if frozen is None:
                raw, optical = src.frames()
                thermal_bgr = render.colourise(calibrate_frame(raw))
            else:
                thermal_bgr, optical = frozen

            n = len(picker.thermal_pts)
            msg = f"{n} pairs"
            if n >= 4:
                try:
                    H, _ = homography.compute(picker.thermal_pts,
                                              picker.optical_pts)
                    err = homography.reprojection_error(
                        H, picker.thermal_pts, picker.optical_pts)
                    msg += (f"   reprojection mean {err.mean():.2f} px, "
                            f"max {err.max():.2f} px")
                    msg += ("   [s] save" if err.mean() < 5
                            else "   -- too high, check your pairs")
                except Exception as exc:
                    msg += f"   {exc}"
            else:
                msg += "   need 4+ (6-8 spread wide is better)"
            msg += "   |  u undo   f freeze   s save   q quit"
            if frozen is not None:
                msg = "[FROZEN]  " + msg

            cv2.imshow(WIN, draw(thermal_bgr, optical, picker, msg))
            k = cv2.waitKey(30) & 0xFF
            if k == ord("q"):
                break
            if k == ord("u"):
                picker.undo()
            elif k == ord("f"):
                # Freezing helps a lot: clicking a moving scene guarantees the
                # two views are of different instants, which corrupts the fit.
                frozen = None if frozen is not None else (thermal_bgr, optical)
            elif k == ord("s") and H is not None:
                err = homography.reprojection_error(
                    H, picker.thermal_pts, picker.optical_pts)
                homography.save(args.out, H, {
                    "source": args.source,
                    "n_points": len(picker.thermal_pts),
                    "reprojection_mean_px": float(err.mean()),
                    "reprojection_max_px": float(err.max()),
                    "thermal_points": [list(map(float, p))
                                       for p in picker.thermal_pts],
                    "optical_points": [list(map(float, p))
                                       for p in picker.optical_pts],
                })
                print(f"\nsaved {args.out}")
                print(f"  {len(picker.thermal_pts)} pairs, "
                      f"mean {err.mean():.2f} px, max {err.max():.2f} px")
                if src.truth_homography is not None:
                    d = homography.compare(H, src.truth_homography, (192, 256))
                    print(f"  vs simulated ground truth: {d:.2f} px "
                          f"mean corner error")
                break
    finally:
        cv2.destroyAllWindows()
        src.close()


if __name__ == "__main__":
    main()
