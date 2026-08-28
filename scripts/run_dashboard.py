#!/usr/bin/env python3
"""Launch the AstraVigil live dashboard.

Windows (no hardware needed):
    python scripts/run_dashboard.py --source synthetic

Raspberry Pi, real cameras:
    python3 scripts/run_dashboard.py --source hardware --calibration data/calibration/H.json

Replay captured thermal frames, optionally with a webcam alongside:
    python scripts/run_dashboard.py --source replay --optical 0

Then open http://localhost:8000 (or the Pi's address from another machine).
"""
import argparse
import os
import sys
import threading

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil import sources                      # noqa: E402
from astravigil.calibration import homography       # noqa: E402
from astravigil.dashboard.app import (              # noqa: E402
    DashboardState, capture_loop, create_app)
from astravigil.pipeline import Pipeline            # noqa: E402

DEFAULT_CALIBRATION = "data/calibration/H.json"


def build_source(args):
    if args.source == "synthetic":
        return sources.create("synthetic", seed=args.seed)
    if args.source == "replay":
        return sources.create("replay", thermal_dir=args.thermal_dir,
                              optical=args.optical)
    return sources.create("hardware", optical_index=int(args.optical or 0),
                          swap_rb=args.swap_rb)


def resolve_homography(args, source):
    """Calibration file if there is one, ground truth in simulation, else none."""
    path = args.calibration or DEFAULT_CALIBRATION
    if os.path.exists(path):
        print(f"calibration      : {path}")
        return homography.load(path)

    if source.truth_homography is not None and not args.uncalibrated:
        # Simulation knows the exact mapping. Using it means the dashboard
        # shows frame matching working from the first frame, which is what
        # you want when developing detection. Run a real calibration with
        # scripts/calibrate_homography.py to exercise the real path.
        print("calibration      : simulated ground truth "
              "(no H.json found; run scripts/calibrate_homography.py)")
        return source.truth_homography

    print("calibration      : NONE - overlay and optical mapping disabled")
    return None


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "replay", "hardware"])
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--calibration", default=None,
                   help=f"homography JSON (default {DEFAULT_CALIBRATION})")
    p.add_argument("--uncalibrated", action="store_true",
                   help="ignore ground truth in simulation, to see the "
                        "uncalibrated state")
    p.add_argument("--threshold", type=float, default=1.5,
                   help="detection threshold in degrees C above background")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thermal-dir", default="data/raw/thermal")
    p.add_argument("--optical", default=None,
                   help="webcam index or video path for replay/hardware")
    p.add_argument("--swap-rb", action="store_true",
                   help="flip red/blue on the optical feed, if colours look "
                        "inverted (skin appearing blue)")
    args = p.parse_args()

    source = build_source(args)
    H = resolve_homography(args, source)
    pipeline = Pipeline(source, H=H, threshold_c=args.threshold)

    state = DashboardState()
    worker = threading.Thread(target=capture_loop,
                              args=(pipeline, state, args.fps), daemon=True)
    worker.start()

    app = create_app(pipeline, state)
    shown = "localhost" if args.host in ("0.0.0.0", "") else args.host
    print(f"source           : {source.name}")
    print(f"detect threshold : {args.threshold} C above background")
    print(f"\n  http://{shown}:{args.port}\n")
    try:
        app.run(host=args.host, port=args.port, threaded=True,
                debug=False, use_reloader=False)
    finally:
        state.stop.set()
        source.close()


if __name__ == "__main__":
    main()
