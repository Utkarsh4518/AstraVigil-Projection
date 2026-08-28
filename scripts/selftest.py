#!/usr/bin/env python3
"""Headless check of the whole pipeline. Run this before the dashboard.

Scores frame matching against simulated ground truth, which is the one check
that catches a homography that is subtly wrong - the failure mode that looks
fine on screen and quietly puts every optical crop a few pixels off target.

    python scripts/selftest.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil import sources                      # noqa: E402
from astravigil.calibration import homography       # noqa: E402
from astravigil.pipeline import Pipeline            # noqa: E402

FAILURES = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def section(t):
    print("\n" + "=" * 66 + f"\n{t}\n" + "=" * 66)


def main():
    section("1. SOURCE")
    src = sources.create("synthetic")
    raw, optical = src.frames()
    check("thermal frame shape", raw.shape == (344, 256), str(raw.shape))
    check("thermal dtype uint16", raw.dtype == np.uint16, str(raw.dtype))
    check("optical frame shape", optical.shape == (480, 640, 3), str(optical.shape))
    check("metadata carries geometry",
          raw[192, 8] == 192 and raw[192, 9] == 256)

    section("2. FRAME MATCHING vs GROUND TRUTH")
    shape = (192, 256)
    # Perfect correspondences: this isolates the maths from clicking error.
    t_pts, o_pts = src.calibration_points(n=8, jitter_px=0.0)
    H, _ = homography.compute(t_pts, o_pts)
    err = homography.reprojection_error(H, t_pts, o_pts)
    corner = homography.compare(H, src.truth_homography, shape)
    check("recovers ground truth from clean points", corner < 0.5,
          f"mean corner error {corner:.3f} px")
    check("reprojection error near zero", err.max() < 0.5,
          f"max {err.max():.3f} px")

    # Realistic case: a human picking points is good to a pixel or two.
    t_j, o_j = src.calibration_points(n=8, jitter_px=1.5)
    Hj, _ = homography.compute(t_j, o_j)
    corner_j = homography.compare(Hj, src.truth_homography, shape)
    check("tolerates 1.5 px clicking jitter", corner_j < 6.0,
          f"mean corner error {corner_j:.2f} px")

    # Four points is the algebraic minimum and has no redundancy to average
    # errors out, so it should be visibly worse. Worth showing rather than
    # asserting a threshold.
    t4, o4 = src.calibration_points(n=4, jitter_px=1.5)
    H4, _ = homography.compute(t4, o4)
    print(f"         (4 points, same jitter: "
          f"{homography.compare(H4, src.truth_homography, shape):.2f} px "
          f"- use 6-8 spread across the frame)")

    section("3. DETECTION AND TRACKING")
    pipe = Pipeline(sources.create("synthetic"), H=src.truth_homography)
    seen, labels = 0, {}
    for _ in range(140):
        res = pipe.step()
        seen += len(res.detections)
        for d in res.detections:
            labels[d.label] = labels.get(d.label, 0) + 1

    check("background model warmed up", pipe.detector.ready)
    check("detections produced", seen > 0, f"{seen} across 140 frames")
    check("tracks established", len(pipe.tracker.tracks) > 0,
          f"{len(pipe.tracker.tracks)} live")
    print(f"         labels: {labels}")

    section("4. FUSION - thermal detection into the optical frame")
    res = pipe.result
    if res.detections:
        det = res.detections[0]
        box = pipe.optical_box(det)
        crop = pipe.crop_optical(det)
        check("thermal box maps into optical", box is not None, str(box))
        check("optical crop non-empty",
              crop is not None and crop.size > 0,
              str(None if crop is None else crop.shape))
        inside = (0 <= box[0] < 640 and 0 <= box[1] < 480)
        check("mapped box lands inside the optical frame", inside)
    else:
        check("had a detection to fuse", False, "none in the final frame")

    section("5. RENDER")
    from astravigil.dashboard import render
    for name, fn in (("thermal", lambda: render.thermal_view(res)),
                     ("optical", lambda: render.optical_view(res, pipe.H)),
                     ("overlay", lambda: render.overlay_view(res, pipe.H))):
        img = fn()
        jpg = render.encode_jpeg(img)
        check(f"{name} view renders and encodes",
              img is not None and jpg is not None and len(jpg) > 500,
              f"{img.shape[1]}x{img.shape[0]}, {len(jpg) // 1024} KB")

    section("RESULT")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all checks passed - safe to run scripts/run_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
