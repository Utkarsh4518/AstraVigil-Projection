#!/usr/bin/env python3
"""Score detection and classification against simulated ground truth.

    python scripts/evaluate.py --frames 600

Produces the confusion matrix. Read it for what it is: performance against a
SIMULATION whose targets were drawn by the same person who wrote the
classifier rules. It demonstrates the pipeline is wired correctly end to end.
It says nothing about how the system behaves on a real bird, and it must not
be presented as though it does - the honest headline is "validated in
simulation, real footage pending".
"""
import argparse
import os
import sys
from collections import Counter

import cv2
import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil import sources                # noqa: E402
from astravigil.pipeline import Pipeline      # noqa: E402

MATCH_RADIUS_PX = 22.0
CLASSES = ["drone", "bird", "unknown"]


def truth_positions(scene):
    """True target centres in thermal pixel coordinates, per class."""
    out = []
    for tg in scene.targets:
        p = cv2.perspectiveTransform(
            np.float32([[[tg.x, tg.y]]]), scene.H_world_to_thermal)[0][0]
        if 0 <= p[0] < 256 and 0 <= p[1] < 192:
            out.append((tg.kind, (float(p[0]), float(p[1]))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", type=int, default=600)
    ap.add_argument("--warmup", type=int, default=40)
    ap.add_argument("--threshold", type=float, default=1.5)
    args = ap.parse_args()

    src = sources.create("synthetic")
    pipe = Pipeline(src, H=src.truth_homography, threshold_c=args.threshold)

    confusion = Counter()
    missed = Counter()
    spurious = 0
    truth_seen = Counter()
    frames_scored = 0

    for i in range(args.frames):
        res = pipe.step()
        if i < args.warmup:
            continue
        frames_scored += 1

        truth = truth_positions(src.scene)
        for kind, _ in truth:
            truth_seen[kind] += 1

        unmatched = list(res.detections)
        for kind, pos in truth:
            best, best_d = None, MATCH_RADIUS_PX
            for det in unmatched:
                d = float(np.hypot(det.centroid[0] - pos[0],
                                   det.centroid[1] - pos[1]))
                if d < best_d:
                    best, best_d = det, d
            if best is None:
                missed[kind] += 1
            else:
                confusion[(kind, best.label)] += 1
                unmatched.remove(best)
        spurious += len(unmatched)

    print(f"\nframes scored : {frames_scored}")
    print(f"true instances: {dict(truth_seen)}")
    print(f"missed        : {dict(missed)}")
    print(f"unmatched detections (clutter / split blobs): {spurious}")

    print("\nCONFUSION MATRIX   rows = truth, cols = predicted")
    print(f"{'':10}" + "".join(f"{c:>10}" for c in CLASSES) + f"{'recall':>10}")
    for t in ("drone", "bird"):
        row = [confusion[(t, p)] for p in CLASSES]
        tot = sum(row) + missed[t]
        rec = row[CLASSES.index(t)] / tot if tot else 0.0
        print(f"{t:10}" + "".join(f"{v:>10}" for v in row) + f"{rec:>10.2f}")

    det_total = sum(confusion.values())
    correct = confusion[("drone", "drone")] + confusion[("bird", "bird")]
    print(f"\naccuracy over matched detections: "
          f"{correct / det_total:.2%}" if det_total else "no matches")

    # The number that actually matters for a perimeter system.
    dr_total = sum(confusion[("drone", p)] for p in CLASSES) + missed["drone"]
    dr_as_bird = confusion[("drone", "bird")]
    print(f"\ndrones called BIRD (the dangerous error): "
          f"{dr_as_bird} / {dr_total}"
          + (f" = {dr_as_bird / dr_total:.2%}" if dr_total else ""))
    print(f"drones missed entirely                 : "
          f"{missed['drone']} / {dr_total}"
          + (f" = {missed['drone'] / dr_total:.2%}" if dr_total else ""))
    print(f"birds called DRONE (nuisance alarm)    : "
          f"{confusion[('bird', 'drone')]}")


if __name__ == "__main__":
    main()
