#!/usr/bin/env python3
"""Launch the AstraVigil live dashboard.

Windows (no hardware needed):
    python scripts/run_dashboard.py --source synthetic

The scenario the challenge brief actually describes - a drone flies in, lands
on the apron, and sits there while its motors cool:
    python scripts/run_dashboard.py --scenario intrusion

The harder half of it - the drone was already on the ground before the system
was switched on, so nothing ever moved and motion detection is blind to it.
Learn a clean site first, then run against it:
    python scripts/learn_site.py --seconds 90
    python scripts/run_dashboard.py --scenario resident --site-model data/baseline/site.npz

Raspberry Pi, real cameras:
    python3 scripts/run_dashboard.py --source hardware \
            --calibration data/calibration/H.json \
            --site-model data/baseline/site.npz

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
from astravigil.alerting import AlertManager        # noqa: E402
from astravigil.calibration import homography       # noqa: E402
from astravigil.dashboard.app import (              # noqa: E402
    DashboardState, capture_loop, create_app)
from astravigil.llm import Escalator, FeatherlessClient   # noqa: E402
from astravigil.pipeline import Pipeline            # noqa: E402
from astravigil.site_intelligence import SiteBaseline   # noqa: E402

DEFAULT_CALIBRATION = "data/calibration/H.json"
DEFAULT_SITE = "data/baseline/site.npz"


def build_source(args):
    if args.source == "synthetic":
        return sources.create("synthetic", seed=args.seed,
                              scenario=args.scenario)
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


def resolve_site(args):
    """A site model learned earlier, or a fresh one that starts from nothing.

    Loading one matters more than it looks. A baseline built in a previous
    session is the only thing that can see an object which was already sitting
    there when the system booted - it never moved, so motion detection has
    nothing to work with.
    """
    path = args.site_model
    if path and os.path.exists(path):
        site = SiteBaseline.load(path)
        print(f"site model       : {path} ({site.frames} frames learned)")
        return site
    if path:
        print(f"site model       : {path} not found - starting from nothing")
    else:
        print("site model       : fresh - learning this site from scratch")
    return SiteBaseline(fps=args.fps)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", default="synthetic",
                   choices=["synthetic", "replay", "hardware"])
    p.add_argument("--scenario", default="patrol",
                   choices=["clean", "patrol", "intrusion", "resident", "crossover"],
                   help="synthetic only: clean is routine traffic with no "
                        "drone at all; patrol adds one flying past; intrusion "
                        "lands one on the apron and leaves it there; resident "
                        "starts with one already on the ground")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--calibration", default=None,
                   help=f"homography JSON (default {DEFAULT_CALIBRATION})")
    p.add_argument("--site-model", default=None,
                   help=f"learned site baseline .npz (e.g. {DEFAULT_SITE}); "
                        f"omit to learn from scratch")
    p.add_argument("--save-site", action="store_true",
                   help="write the site model back out on exit")
    p.add_argument("--no-learn", action="store_true",
                   help="freeze the site model - score against it but do not "
                        "update it")
    p.add_argument("--uncalibrated", action="store_true",
                   help="ignore ground truth in simulation, to see the "
                        "uncalibrated state")
    p.add_argument("--threshold", type=float, default=1.5,
                   help="detection threshold in degrees C above background")
    p.add_argument("--alert-threshold", type=float, default=0.75,
                   help="threat score at which an alert opens")
    p.add_argument("--featherless", action="store_true",
                   help="escalate high-threat detections to a hosted model "
                        "for identification. Needs FEATHERLESS_API_KEY. OFF "
                        "by default: it sends cropped imagery of the "
                        "protected site to a third party, and nothing in the "
                        "system depends on it")
    p.add_argument("--featherless-model", default=None,
                   help="model id on Featherless (default: "
                        "$FEATHERLESS_MODEL, else a Qwen2.5-VL build)")
    p.add_argument("--escalate-at", type=float, default=0.75,
                   help="threat score above which identification is asked")
    p.add_argument("--fps", type=float, default=25.0)
    p.add_argument("--view-fps", type=float, default=6.0,
                   help="how often the browser views are redrawn. Detection "
                        "always runs at --fps; drawing is far more expensive "
                        "than detecting and nobody needs a 25 Hz dashboard. "
                        "0 disables the views entirely (headless sensor).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--thermal-dir", default="data/raw/thermal")
    p.add_argument("--optical", default=None,
                   help="webcam index or video path for replay/hardware")
    p.add_argument("--policy", default=None,
                   help="site policy JSON from scripts/compile_policy.py - "
                        "what is ALLOWED here, as opposed to what the "
                        "baseline learns is normal here")
    p.add_argument("--swap-rb", action="store_true",
                   help="flip red/blue on the optical feed, if colours look "
                        "inverted (skin appearing blue)")
    args = p.parse_args()

    policy = None
    if args.policy:
        from astravigil.policy import SitePolicy, validate
        policy = SitePolicy.load(args.policy)
        problems = validate(policy)
        print(f"policy           : {args.policy} "
              f"({len(policy.rules)} rules, {len(policy.zones) - 1} zones)")
        for pr in problems:
            print(f"  ! {pr}")
    else:
        print("policy           : none - judging on learned statistics alone")

    source = build_source(args)
    H = resolve_homography(args, source)
    site = resolve_site(args)
    alerts = AlertManager(threshold=args.alert_threshold)

    client = FeatherlessClient(
        model=args.featherless_model) if args.featherless_model \
        else FeatherlessClient()
    escalator = Escalator(client=client, enabled=args.featherless)
    if args.featherless and not escalator.enabled:
        print("featherless      : REQUESTED BUT DISABLED - "
              "FEATHERLESS_API_KEY is not set")
    elif escalator.enabled:
        print(f"featherless      : on, model {client.model}, "
              f"escalating above {args.escalate_at} threat")
    else:
        print("featherless      : off (local classification only)")

    pipeline = Pipeline(source, H=H, threshold_c=args.threshold, policy=policy, site=site,
                        alerts=alerts, fps=args.fps,
                        learn=not args.no_learn, escalator=escalator,
                        escalate_at=args.escalate_at)

    state = DashboardState()
    worker = threading.Thread(target=capture_loop,
                              args=(pipeline, state, args.fps, args.view_fps),
                              daemon=True)
    worker.start()

    site_path = args.site_model or DEFAULT_SITE
    app = create_app(pipeline, state, site_path=site_path)
    shown = "localhost" if args.host in ("0.0.0.0", "") else args.host
    print(f"source           : {source.name}")
    print(f"detect threshold : {args.threshold} C above background")
    print(f"alert threshold  : {args.alert_threshold} threat")
    print(f"site learning    : {'frozen' if args.no_learn else 'on'}")
    print(f"detect / views   : {args.fps:.0f} Hz / {args.view_fps:.0f} Hz "
          f"(views drawn only while a browser is subscribed)")
    print(f"alert log        : {alerts.log_path}")
    print(f"\n  http://{shown}:{args.port}\n")
    try:
        app.run(host=args.host, port=args.port, threaded=True,
                debug=False, use_reloader=False)
    finally:
        state.stop.set()
        if args.save_site:
            print(f"\nsaving site model to {site_path} "
                  f"({pipeline.site.frames} frames)")
            pipeline.save_site(site_path)
        source.close()


if __name__ == "__main__":
    main()
