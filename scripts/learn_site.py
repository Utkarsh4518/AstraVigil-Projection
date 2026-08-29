#!/usr/bin/env python3
"""Watch a site until the baseline knows it, then write the model to disk.

Point the cameras at what you are protecting, leave this running while the
scene is behaving normally, and it records what "normal" looks like there:
where the warm things are, where traffic passes, how big and how fast the
things that pass usually are.

    python scripts/learn_site.py --seconds 90                 # simulated
    python3 scripts/learn_site.py --source hardware --minutes 20   # at the rig

The saved model is what lets a later run catch something that never moved.
An object already sitting on the ground when the system boots is invisible to
motion detection - it is in the very first background frame - so the only
thing that can see it is a memory of the site from before it arrived. That
memory is this file.

Learn a clean scene. Anything present and stationary while this runs becomes
part of the definition of normal, which is the correct behaviour and also the
obvious way to blind the system on purpose.
"""
import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil import sources                          # noqa: E402
from astravigil.alerting import AlertManager            # noqa: E402
from astravigil.pipeline import Pipeline                # noqa: E402
from astravigil.site_intelligence import SiteBaseline   # noqa: E402

DEFAULT_OUT = "data/baseline/site.npz"


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "replay", "hardware"])
    p.add_argument("--scenario", default="clean",
                   choices=["clean", "patrol", "intrusion", "resident", "crossover"],
                   help="synthetic only; 'clean' is the scene as it should "
                        "be, which is the only honest thing to learn from")
    p.add_argument("--seconds", type=float, default=None)
    p.add_argument("--minutes", type=float, default=None)
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--append", action="store_true",
                   help="keep learning on top of an existing model rather "
                        "than starting fresh")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--realtime", action="store_true",
                   help="pace to --fps; off by default so simulated learning "
                        "runs as fast as the machine allows")
    p.add_argument("--optical", default=None)
    p.add_argument("--thermal-dir", default="data/raw/thermal")
    args = p.parse_args()

    seconds = args.seconds if args.seconds is not None else (
        args.minutes * 60 if args.minutes is not None else 60.0)
    frames = int(seconds * args.fps)

    if args.source == "synthetic":
        source = sources.create("synthetic", scenario=args.scenario)
    elif args.source == "replay":
        source = sources.create("replay", thermal_dir=args.thermal_dir,
                                optical=args.optical)
    else:
        source = sources.create("hardware",
                                optical_index=int(args.optical or 0))

    site = None
    if args.append and os.path.exists(args.out):
        site = SiteBaseline.load(args.out)
        print(f"continuing from {args.out} ({site.frames} frames)")

    pipe = Pipeline(source, H=source.truth_homography, site=site,
                    fps=args.fps, clock="frames",
                    # No alert log while learning: everything seen here is
                    # meant to be normal, and writing it to the incident
                    # trail would be noise in the record.
                    alerts=AlertManager(log_path=None))

    print(f"learning {source.name} for {seconds:.0f} s ({frames} frames)")
    period = 1.0 / args.fps
    t_start = time.monotonic()
    try:
        for i in range(frames):
            t0 = time.monotonic()
            pipe.step()
            if i % max(1, frames // 20) == 0:
                s = pipe.site.stats()
                print(f"  {i / args.fps:6.1f}s  scene "
                      f"{s['scene_maturity']*100:3.0f}%  traffic "
                      f"{s['activity_maturity']*100:3.0f}%  "
                      f"off-baseline cells {s['anomalous_cells']}")
            if args.realtime:
                rest = period - (time.monotonic() - t0)
                if rest > 0:
                    time.sleep(rest)
    except KeyboardInterrupt:
        print("\ninterrupted - saving what has been learned so far")
    finally:
        source.close()

    s = pipe.site.stats()
    pipe.save_site(args.out)
    print(f"\nwall clock       : {time.monotonic() - t_start:.1f} s")
    print(f"frames learned   : {s['frames']}")
    print(f"scene maturity   : {s['scene_maturity']*100:.0f}%")
    print(f"traffic maturity : {s['activity_maturity']*100:.0f}%  "
          f"({s['activity_obs']} track observations)")
    print(f"cells off baseline at the end: {s['anomalous_cells']} "
          f"of {s['cells']}")
    if s["anomalous_cells"]:
        print("  ^ something was still reading as out of place when this "
              "stopped. If it belongs there, learn for longer or leave the "
              "scene clear; it is not in the baseline.")
    print(f"\nsaved: {args.out}")
    print(f"use it with:\n  python scripts/run_dashboard.py "
          f"--site-model {args.out}")


if __name__ == "__main__":
    main()
