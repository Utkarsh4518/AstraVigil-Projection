#!/usr/bin/env python3
"""Where the frame budget actually goes, measured on the machine you run it on.

    python3 scripts/benchmark.py                      # simulated frames
    python3 scripts/benchmark.py --source hardware    # on the Pi, real camera

Run this on the Pi rather than scaling laptop numbers by a guessed ARM
multiplier. The multiplier is not one number: OpenCV's warps and the JPEG
encoder are SIMD-optimised on both x86 and ARM and scale reasonably, while
plain numpy arithmetic and Python-level loops fall off much harder on a
Cortex-A72. Guessing "8x" gets the shape of the answer wrong.

What the output is for: the DETECT block must fit in the frame period, because
it is the safety-critical path and it runs every frame. The RENDER block does
not - the dashboard only draws views a browser is subscribed to, at a lower
rate - so it is reported separately and should be read as "what one viewer
costs", not as part of the steady-state load.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil import sources                          # noqa: E402
from astravigil.alerting import AlertManager            # noqa: E402
from astravigil.dashboard import render                 # noqa: E402
from astravigil.dashboard.app import RENDERERS          # noqa: E402
from astravigil.drivers.thermal.calibration import calibrate_frame  # noqa: E402
from astravigil.fusion import verify_optical, verify_thermal    # noqa: E402
from astravigil.pipeline import Pipeline                # noqa: E402
from astravigil.site_intelligence import SiteBaseline   # noqa: E402


def timed(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return 1000.0 * (time.perf_counter() - t0) / n


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "replay", "hardware"])
    p.add_argument("--scenario", default="intrusion",
                   choices=["clean", "patrol", "intrusion", "resident", "crossover"])
    p.add_argument("--warmup-frames", type=int, default=450,
                   help="frames to run before measuring, so the background "
                        "model and site baseline are settled")
    p.add_argument("--repeats", type=int, default=100)
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--site-model", default=None)
    p.add_argument("--optical", default=None)
    args = p.parse_args()

    if args.source == "synthetic":
        src = sources.create("synthetic", scenario=args.scenario)
    elif args.source == "replay":
        src = sources.create("replay", optical=args.optical)
    else:
        src = sources.create("hardware", optical_index=int(args.optical or 0))

    site = (SiteBaseline.load(args.site_model)
            if args.site_model and os.path.exists(args.site_model) else None)
    pipe = Pipeline(src, H=src.truth_homography, site=site, fps=args.fps,
                    clock="frames", alerts=AlertManager(log_path=None))

    print(f"source        : {src.name}")
    print(f"settling      : {args.warmup_frames} frames...")
    for _ in range(args.warmup_frames):
        pipe.step()
    res = pipe.result
    print(f"detections    : {len(res.detections)}  "
          f"tracks: {len(res.tracks)}  alerts: {len(res.alerts)}")

    raw, _ = src.frames()
    cel = calibrate_frame(raw)
    tracks = pipe.tracker.tracks
    dets = res.detections
    n = args.repeats

    print("\n" + "=" * 58)
    print("DETECT + LEARN + ASSESS   runs every frame - must fit the budget")
    print("=" * 58)
    detect = []
    detect.append(("calibrate_frame",
                   timed(lambda: calibrate_frame(raw), n)))
    detect.append(("detection (bg sub, morph, contours)",
                   timed(lambda: pipe.detector.update(cel), n)))
    detect.append(("site baseline: observe/learn",
                   timed(lambda: pipe.site.observe(cel, tracks), n)))
    detect.append(("site baseline: score objects",
                   timed(lambda: [pipe.site.score(d, tracks.get(d.track_id),
                                                  3.0) for d in dets], n)))
    detect.append(("site baseline: settled regions",
                   timed(lambda: pipe.site.static_anomalies(), n)))

    # The optical half. Detection here is ROI-limited to the thermal
    # footprint and runs every Nth frame, so the per-frame figure is the
    # measured cost divided by that stride - which is what the row shows.
    opt_frame = res.optical
    if opt_frame is not None and pipe.cross_cue:
        stride = pipe.optical.every_n
        raw_opt = timed(lambda: pipe.optical.update(opt_frame), n)
        detect.append((f"optical detection (1 frame in {stride})",
                       raw_opt / stride))
        if dets:
            detect.append(("cross-cue: thermal asks optical (shape)",
                           timed(lambda: [verify_optical(opt_frame, pipe.H,
                                                         d.box) for d in dets],
                                 n)))
        detect.append(("cross-cue: optical asks thermal (heat)",
                       timed(lambda: verify_thermal(
                           cel, pipe.H, (100, 100, 40, 40)), n)))
    for name, ms in detect:
        print(f"  {name:44} {ms:7.2f} ms")
    detect_total = sum(ms for _, ms in detect)
    print(f"  {'-' * 44} {'-' * 7}")
    print(f"  {'TOTAL':44} {detect_total:7.2f} ms")

    print("\n" + "=" * 58)
    print("CAPTURE                   what one frame costs to acquire")
    print("=" * 58)
    cap = timed(lambda: src.frames(), max(10, n // 5))
    print(f"  {src.name:44} {cap:7.2f} ms")
    if args.source == "synthetic":
        print("  (simulation renders a whole world here; on real hardware "
              "this is a USB read)")

    # The itemised rows above are the stages worth naming, not every stage -
    # tracking, classification and alert bookkeeping are not separately timed
    # because none of them can be called twice without mutating their own
    # state. So take the honest total by timing the real thing and
    # subtracting capture, and report the difference rather than pretending
    # the parts sum to the whole.
    step = timed(lambda: pipe.step(), n)
    end_to_end = max(step - cap, 0.0)
    print("\n" + "=" * 58)
    print("END TO END                one real frame, capture removed")
    print("=" * 58)
    print(f"  {'pipeline.step() minus capture':44} {end_to_end:7.2f} ms")
    print(f"  {'  of which itemised above':44} {detect_total:7.2f} ms")
    print(f"  {'  tracking, classification, alerting':44} "
          f"{max(end_to_end - detect_total, 0.0):7.2f} ms")

    print("\n" + "=" * 58)
    print("RENDER                    only when a browser is watching")
    print("=" * 58)
    rend = []
    for name in RENDERERS:
        rend.append((name, timed(lambda nm=name: RENDERERS[nm](res, pipe), n)))
    imgs = [RENDERERS[nm](res, pipe) for nm in RENDERERS]
    rend.append(("jpeg encode x%d" % len(imgs),
                 timed(lambda: [render.encode_jpeg(i) for i in imgs], n)))
    for name, ms in rend:
        print(f"  {name:44} {ms:7.2f} ms")
    render_total = sum(ms for _, ms in rend)
    print(f"  {'-' * 44} {'-' * 7}")
    print(f"  {'TOTAL, all four views':44} {render_total:7.2f} ms")

    budget = 1000.0 / args.fps
    print("\n" + "=" * 58)
    print(f"VERDICT at {args.fps:.0f} Hz   budget {budget:.1f} ms per frame")
    print("=" * 58)
    headroom = budget / end_to_end if end_to_end > 0 else float("inf")
    print(f"  detect + learn + assess      {end_to_end:7.2f} ms   "
          f"{headroom:5.1f}x headroom   "
          f"{'OK' if end_to_end < budget else 'OVER BUDGET'}")

    for view_fps in (0, 3, 6, 12):
        amortised = end_to_end + (render_total * view_fps / args.fps)
        label = ("no viewer" if view_fps == 0
                 else f"1 viewer, views at {view_fps} Hz")
        print(f"  {label:28} {amortised:7.2f} ms   "
              f"{budget / amortised:5.1f}x headroom   "
              f"{'OK' if amortised < budget else 'OVER BUDGET'}")

    print("\n  Rendering every view every frame - what the dashboard did")
    print(f"  before views became subscription-gated - would be "
          f"{end_to_end + render_total:.2f} ms.")
    if end_to_end + render_total > budget:
        print("  That is over budget on this machine. Gating is not an "
              "optimisation here, it is the difference between holding the "
              "frame rate and not.")
    src.close()


if __name__ == "__main__":
    main()
