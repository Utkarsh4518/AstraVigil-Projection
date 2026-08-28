# Live dashboard — frame matching and detection

A browser page showing the thermal feed, the optical feed, and the thermal
warped into optical space, with live detections and their features.

## Run it

```bash
pip install -r requirements.txt          # opencv, flask, numpy (pyusb on Linux)

python scripts/selftest.py               # always run this first
python scripts/run_dashboard.py --source synthetic
# open http://localhost:8000
```

On the Pi, with real cameras:

```bash
python3 scripts/run_dashboard.py --source hardware \
        --calibration data/calibration/H.json
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

## What the three views are for

**Thermal** — where detection actually runs. Warm movers against cold sky.

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

Measured on an x86 laptop:

| Stage | ms/frame |
|---|---|
| calibrate_frame | 0.10 |
| detection + morphology + contours | 0.38 |
| render overlay (2× warpPerspective) | 4.18 |
| JPEG encode, 3 streams | 1.81 |

Detection is **~0.5 ms**. Even assuming a Pi 4 is 8× slower that is ~4 ms, far
inside the 40 ms budget at 25 Hz. The expensive part is *rendering the views*,
and that only happens when someone is watching.

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
  sim.py                    synthetic world, known ground-truth homography
  sources.py                synthetic | replay | hardware, one interface
  pipeline.py               capture -> detect -> track -> classify -> fuse
  calibration/homography.py compute, score, save, warp
  detection/thermal.py      background subtraction, contours, blob merging
  tracking/tracker.py       IDs and the temporal features
  classification/rules.py   interpretable drone/bird rules
  dashboard/                Flask app, page, renderers
scripts/
  selftest.py               end-to-end check, scored against ground truth
  evaluate.py               confusion matrix
  calibrate_homography.py   point picker
  run_dashboard.py          launch
```
