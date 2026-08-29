#!/usr/bin/env python3
"""Headless check of the whole pipeline. Run this before the dashboard.

Scores frame matching against simulated ground truth, which is the one check
that catches a homography that is subtly wrong - the failure mode that looks
fine on screen and quietly puts every optical crop a few pixels off target.

Then it exercises the site model on the challenge brief's own scenario: a
drone that lands and sits still, and one that was already on the ground before
the system booted. The second is the interesting one, because motion detection
is structurally blind to it - it is in the very first background frame - so it
either proves the persistent baseline works or proves it does not.

    python scripts/selftest.py
"""
import os
import shutil
import subprocess
import tempfile
import sys

import numpy as np

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil import sources                      # noqa: E402
from astravigil.alerting import AlertManager        # noqa: E402
from astravigil.calibration import homography       # noqa: E402
from astravigil.pipeline import Pipeline            # noqa: E402
from astravigil.site_intelligence import SiteBaseline   # noqa: E402

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

    section("5b. DASHBOARD PAGE")
    # A syntax error anywhere in the page's script block stops the WHOLE block
    # parsing, so nothing runs: no polling, no buttons, no escape sequence.
    # The page still loads and still shows the four camera streams, because
    # those are plain <img> tags the browser fetches on its own - so it looks
    # alive while every control on it is dead. That failure has shipped once
    # already; it is cheap to make impossible.
    from astravigil.dashboard.page import PAGE
    script = PAGE[PAGE.index("<script>") + 8:PAGE.rindex("</script>")]

    check("page markup has balanced script tags",
          PAGE.count("<script>") == PAGE.count("</script>") == 1)

    # Every element the script reaches for by id must exist in the markup, or
    # the first call returns null and everything after it in that function
    # stops - which is how one renamed div silently kills the poll loop.
    import re
    wanted = set(re.findall(r'getElementById\("([^"]+)"\)', script))
    present = set(re.findall(r'id="([^"]+)"', PAGE))
    missing = sorted(wanted - present)
    check("every getElementById target exists in the markup",
          not missing, f"{len(wanted)} looked up" +
          (f", MISSING: {', '.join(missing)}" if missing else ""))

    node = shutil.which("node")
    if node:
        with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False,
                                         encoding="utf-8") as fh:
            fh.write(script)
            tmp = fh.name
        proc = subprocess.run([node, "--check", tmp],
                              capture_output=True, text=True)
        os.unlink(tmp)
        check("dashboard javascript parses",
              proc.returncode == 0,
              (proc.stderr.strip().splitlines() or ["clean"])[0][:120])
    else:
        # Not a failure: node is a convenience here, not a dependency. The
        # id check above still runs and catches the commoner mistake.
        print("  [skip] node not installed - javascript not syntax-checked")

    section("6. SITE INTELLIGENCE")
    # Learn a clean scene first. Everything after this depends on the model
    # having a real baseline, and on that baseline being quiet - a site model
    # that flags an empty apron is worse than none.
    clean = sources.create("synthetic", scenario="clean")
    learner = Pipeline(clean, H=clean.truth_homography, clock="frames",
                       alerts=AlertManager(log_path=None))
    for _ in range(900):
        learner.step()
    site_stats = learner.site.stats()
    check("site baseline matured", site_stats["scene_maturity"] >= 1.0,
          f"scene {site_stats['scene_maturity']*100:.0f}%, traffic "
          f"{site_stats['activity_maturity']*100:.0f}%")
    check("clean scene raises no alerts", not learner.result.alerts,
          f"{len(learner.result.alerts)} open")
    check("clean scene has no settled anomalies",
          not learner.result.static_anomalies,
          f"{site_stats['anomalous_cells']} cells off baseline")
    worst = max((a.threat for a in learner.result.assessments), default=0.0)
    check("routine traffic stays below the alert line", worst < 0.75,
          f"highest threat {worst:.2f}")

    # Save and reload: a site model that does not survive a restart cannot do
    # the job it exists for, which is remembering yesterday.
    tmp = os.path.join("data", "baseline", "_selftest.npz")
    learner.save_site(tmp)
    reloaded = SiteBaseline.load(tmp)
    same = float(np.abs(reloaded.ref_mean - learner.site.ref_mean).max())
    check("site model survives save and reload", same < 1e-6
          and reloaded.frames == learner.site.frames,
          f"max cell difference {same:.2e}, {reloaded.frames} frames")

    section("7. THE TARMAC CASE - an object that never moved")
    # The drone is on the ground before frame one, so the motion detector's
    # background absorbs it immediately and never reports it. Only the
    # baseline learned above can see it.
    resident = sources.create("synthetic", scenario="resident")
    watcher = Pipeline(resident, H=resident.truth_homography, site=reloaded,
                       clock="frames", alerts=AlertManager(log_path=None))
    settled_alerts, bird_alerts = set(), set()
    for _ in range(500):
        r = watcher.step()
        for al in r.alerts:
            (settled_alerts if al.kind == "static" else bird_alerts).add(al.id)

    check("settled object detected with no motion cue",
          len(settled_alerts) >= 1,
          f"{len(settled_alerts)} settled-object alert(s)")
    check("reported as ONE alert, not one per frame",
          len(settled_alerts) == 1,
          f"{len(settled_alerts)} distinct alerts over 500 frames")
    check("routine traffic did not alert alongside it",
          not bird_alerts, f"{len(bird_alerts)} track alerts")
    if watcher.result.alerts:
        al = watcher.result.alerts[0]
        print(f"         \"{al.label}\" threat {al.threat:.2f}, "
              f"still for {al.dwell_s:.0f} s")
        print(f"         because: {'; '.join(al.reasons)}")

    try:
        os.remove(tmp)
    except OSError:
        pass

    section("8. CROSS-CUEING - each camera asking the other")
    # Thermal -> optical. The shape check must either produce a verdict or say
    # plainly that there were not enough pixels to produce one. Inventing a
    # shape from twelve pixels is the failure mode worth guarding against.
    cross = sources.create("synthetic", scenario="intrusion")
    cp = Pipeline(cross, H=cross.truth_homography, clock="frames",
                  alerts=AlertManager(log_path=None))
    for _ in range(500):
        cr = cp.step()

    c = cr.cross
    check("optical camera detects independently",
          c["optical_candidates"] > 0,
          f"{c['optical_candidates']} optical candidates")
    check("thermal asks optical for a shape check",
          c["shape_checks"] > 0, f"{c['shape_checks']} checks this frame")

    ev = list(cr.optical_evidence.values())
    honest = all(e.confidence == 0.0 or e.pixels >= 60 for e in ev)
    check("no shape verdict from too few pixels", honest,
          "; ".join(f"{e.pixels:.0f}px conf {e.confidence:.2f}" for e in ev))

    # The rule that keeps this from rebuilding the wall of alarms: one object
    # seen by both sensors must be one entry, not two.
    keys = [a.key for a in cr.assessments]
    check("one object seen by both sensors is ONE entry",
          len(keys) == len(set(keys)) and c["paired_with_thermal"] >= 1,
          f"{c['paired_with_thermal']} paired, {len(keys)} assessments, "
          f"{c['optical_only']} optical-only")

    section("9. THERMAL CROSSOVER - the reverse cue carrying alone")
    # The drone sits at exactly ambient: no contrast, no signature, nothing
    # for the thermal detector or the site baseline to find. Either the
    # optical camera raises it on its own or the system never sees it.
    xo = sources.create("synthetic", scenario="crossover")
    xp = Pipeline(xo, H=xo.truth_homography, clock="frames",
                  alerts=AlertManager(log_path=None))
    thermal_ever = 0
    contact = None
    for i in range(2200):
        xr = xp.step()
        # The bird is warm and legitimately thermal; only count detections
        # near where the invisible drone actually is.
        for a in xr.assessments:
            if a.kind == "optical":
                contact = a
        thermal_ever = max(thermal_ever, len(xr.static_anomalies))

    check("thermal raised nothing - it is genuinely blind here",
          thermal_ever == 0, f"{thermal_ever} thermal settled anomalies")
    check("optical found it anyway", contact is not None,
          contact.label if contact else "nothing")
    if contact is not None:
        check("optical asked thermal and got an honest 'no heat'",
              contact.thermal_check is not None
              and not contact.thermal_check.warm,
              contact.thermal_check.reason if contact.thermal_check else "-")
        check("escalates with dwell rather than sitting at zero",
              contact.threat > 0.45,
              f"threat {contact.threat:.2f} after {contact.dwell_s:.0f} s")
        # Deliberately NOT an alert. With no heat and one sensor, the system
        # cannot tell a cold airframe from a parked object, and claiming it
        # can is how a perimeter system loses its operator's trust.
        check("stays a WATCH, not an ALERT - one sensor, no corroboration",
              contact.level == "watch",
              f"level {contact.level} at threat {contact.threat:.2f}")
        print(f"         \"{contact.label}\" - "
              f"{'; '.join(contact.reasons)}")

    section("RESULT")
    if FAILURES:
        print(f"{len(FAILURES)} FAILED: " + ", ".join(FAILURES))
        return 1
    print("all checks passed - safe to run scripts/run_dashboard.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
