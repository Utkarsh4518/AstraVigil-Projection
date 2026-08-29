# Live dashboard — frame matching and detection

A browser page showing the thermal feed, the site model, the optical feed and
the thermal warped into optical space, with live detections, threat scores and
the alert list.

Two halves are documented separately:
[site_intelligence.md](site_intelligence.md) — what the site baseline learns and
how it catches an object that never moved. [cross_cueing.md](cross_cueing.md) —
each camera detecting on its own and questioning the other, and the resolution
finding that governs how much optical can contribute. [kiosk.md](kiosk.md) —
the desktop launcher, fullscreen operator console, and the three ways out of it.

## Run it

```bash
pip install -r requirements.txt          # opencv, flask, numpy (pyusb on Linux)

python scripts/selftest.py               # always run this first
python scripts/run_dashboard.py --source synthetic
# open http://localhost:8000

# the challenge brief's own scenario: a drone lands on the apron and sits there
python scripts/run_dashboard.py --scenario intrusion

# the harder half: learn a clean site, then meet a drone that was already
# on the ground before boot - nothing ever moved, so motion detection is blind
python scripts/learn_site.py --seconds 90
python scripts/run_dashboard.py --scenario resident \
       --site-model data/baseline/site.npz
```

On the Pi, with real cameras:

```bash
python3 scripts/run_dashboard.py --source hardware \
        --calibration data/calibration/H.json \
        --site-model data/baseline/site.npz --save-site
# open http://<pi-address>:8000 from your laptop
```

## Why a web page and not an OpenCV window

The Pi will be on a roof with no display. A browser page works over the
network with no X-forwarding, renders the same on Windows and Linux, and puts
the demo on any screen in the room. The one place an OpenCV window is right is
calibration, which is a hands-on task done at the rig — so that tool uses one.

## Developing on Windows

The thermal camera cannot be opened on Windows without replacing its driver
with WinUSB via Zadig, which then breaks it for every other Windows app. So
rather than ask for that, the source is abstracted and `hardware` is
Linux-only by design:

| `--source` | Thermal | Optical | Runs on |
|---|---|---|---|
| `synthetic` | simulated | simulated | any OS |
| `replay` | saved `.npy` frames | webcam or video file | any OS |
| `hardware` | HIKMICRO over USB | Pi Camera or webcam | Linux |

Every backend returns the same `(thermal_raw16, optical_bgr)` pair, so the
pipeline never learns where frames came from. Code developed against
`synthetic` on Windows runs unchanged on the Pi.

### The simulation is checkable, not just decorative

Both synthetic views are rendered from one flat world by two different
homographies, so **the true thermal→optical mapping is known exactly**. That
turns frame matching from something you eyeball into something you score:

```
recovers ground truth from clean points   mean corner error 0.000 px
tolerates 1.5 px clicking jitter          mean corner error 3.61 px
   (4 points, same jitter: 4.65 px — use 6–8 spread across the frame)
```

This matters because a subtly wrong homography looks fine on screen. It
quietly puts every optical crop a few pixels off, and downstream that reads as
a classifier problem rather than a calibration one.

## What the four views are for

**Thermal** — where detection actually runs. Warm movers against cold sky.
Boxes are coloured by threat, not by class: what an operator needs at a glance
is which of the things on screen matters.

**Site** — what the baseline has learned. Green is traffic it has seen before,
magenta is a patch that has been off its learned temperature long enough to be
an object rather than a fly-past. This is the view that distinguishes "the
model knows this scene and there is nothing in it" from "the model knows
nothing and is silent for that reason".

**Optical** — thermal detections mapped through H. Boxes should sit on the
real objects.

**Overlay** — thermal warped into optical space. This is the frame-matching
sanity check: hot regions must sit on things that are actually hot. If they
drift apart, recalibrate before trusting anything downstream.

## Reading the numbers

`proc` is detection cost per frame; `capture` is timed separately because in
simulation it renders an entire world (~18 ms) while on real hardware it is
just a USB read. Folding them together made the pipeline look about 40× more
expensive than it is.

### Drawing costs more than detecting

Run `python3 scripts/benchmark.py` **on the Pi**. Do not scale laptop numbers
by a guessed ARM multiplier — it is not one number. OpenCV's warps and the JPEG
encoder are SIMD-optimised on both architectures and scale reasonably; plain
numpy arithmetic and Python-level loops fall off much harder on a Cortex-A72.

Measured on an x86 laptop:

| Stage | ms/frame | runs |
|---|---|---|
| detect + learn + assess, end to end | **1.30** | every frame |
| &nbsp;&nbsp;— calibrate_frame | 0.30 | |
| &nbsp;&nbsp;— detection: bg sub, morphology, contours | 0.15 | |
| &nbsp;&nbsp;— site model: learn, score, settled regions | 0.09 | |
| &nbsp;&nbsp;— tracking, classification, alerting | 0.76 | |
| render, all four views + JPEG | 6.35 | only while watched |

**Rendering is roughly 5× the cost of the work the box exists to do.** Which is
why the capture loop does not render unconditionally: it draws only views a
browser is currently subscribed to, at `--view-fps` (default 6 Hz) rather than
at the detection rate. With nobody watching, the render cost is zero —
confirmed live, `render_ms` reads `0.00` and `rendering` is empty.

That matters specifically on a Pi 4. The detector fits inside a 40 ms budget at
any plausible ARM multiplier; drawing all four views every frame would not. A
rooftop sensor that drops to 15 Hz because it is rendering pictures nobody is
looking at has failed at its actual job. `--view-fps 0` disables the views
entirely for a genuinely headless deployment.

The dashboard header shows both numbers live: `detect` is the safety-critical
path, `draw` is what the views currently cost.

## Features, and why these ones

At 40 m a drone and a bird are both a handful of warm pixels. Silhouette,
aspect and solidity are essentially noise there. Two features survive:

**Flap** — coefficient of variation of blob area over ~1.6 s. A bird modulates
its silhouette several times a second; a rigid airframe does not. Measured in
simulation: bird ~0.55, quad ~0.06.

**Hotspot** — peak temperature above the object's **own** mean. A quad carries
motor cores far hotter than its airframe. Note this is *not* contrast against
the sky: a 31 °C bird against −5 °C sky is exactly as bright as a drone, so
grading on background contrast discriminates nothing. Getting that wrong was
what originally made this classifier call every bird a drone.

**Parts** — how many blobs were merged into one object. A resolved quad breaks
into a body plus four motor dots; reporting those separately would be five
alarms for one aircraft, which is precisely the "wall of separate alarms" the
challenge brief asks us to avoid.

## Performance in simulation

`python scripts/evaluate.py --frames 500`

```
CONFUSION MATRIX   rows = truth, cols = predicted
               drone      bird   unknown    recall
drone            460         0         0      1.00
bird               0       398        61      0.87

drones called BIRD (the dangerous error): 0 / 460 = 0.00%
drones missed entirely                  : 0 / 460 = 0.00%
birds called DRONE (nuisance alarm)     : 0
```

**Read this for exactly what it is.** It is performance against a simulation
whose targets were drawn by the same person who wrote the classifier rules. It
demonstrates the pipeline is wired correctly end to end and that the features
separate the two classes. It says **nothing** about how the system behaves on a
real bird in real weather, and must not be presented as though it does. The
honest headline is *"validated in simulation; real footage pending"* — which is
also why capturing negatives is still on the critical path.

The 61 birds called "unknown" are the classifier declining to commit rather
than guessing. That is the intended behaviour.

## Calibration

```bash
python scripts/calibrate_homography.py --source synthetic   # rehearse the flow
python3 scripts/calibrate_homography.py --source hardware   # at the rig
```

Click the same physical point in the thermal pane then the optical pane,
alternating. `f` freezes both feeds — worth using, because clicking a moving
scene means the two views are of different instants, which corrupts the fit.

Two things decide whether the result is any good:

- **What you aim at.** Paper targets are invisible in thermal. Use point heat
  sources — a mug, a lamp, a hand held still — unambiguous in *both* views.
- **How far away.** A homography is exact only at its calibration depth. At the
  mount's 50 mm baseline, calibrating at 3 m and then watching at 50 m builds
  in ~9 px of error before any noise. Calibrate at roughly your operating
  range.

Spread points into the corners. Points clustered in the middle leave the
corners unconstrained, and the corners are where error shows.

## Layout

```
src/astravigil/
  sim.py                     synthetic world, known ground-truth homography
  sources.py                 synthetic | replay | hardware, one interface
  pipeline.py                capture -> detect -> track -> classify
                             -> learn -> assess -> alert
  calibration/homography.py  compute, score, save, warp
  detection/thermal.py       background subtraction, contours, blob merging
  detection/optical.py       independent optical detection, ROI-limited
  tracking/tracker.py        IDs and the temporal features
  classification/rules.py    interpretable drone/bird rules
  site_intelligence/
    baseline.py              the learned site model, and what is out of place
    dwell.py                 how long each track has been sitting still
  fusion/crosscue.py         each camera questioning the other
  fusion/threat.py           identity + site anomaly -> one threat score
  alerting/alerts.py         one stateful alert per object, JSONL log
  dashboard/                 Flask app, page, renderers
scripts/
  selftest.py                end-to-end check, scored against ground truth
  evaluate.py                confusion matrix
  learn_site.py              watch a site, save the baseline
  calibrate_homography.py    point picker
  benchmark.py               where the frame budget goes, on this machine
  run_dashboard.py           launch
```
