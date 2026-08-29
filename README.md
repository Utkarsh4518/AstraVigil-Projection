# AstraVigil

Thermal and optical sensor fusion for spotting small drones over a fixed site.
Built for Challenge 04 of the European Defense Tech Hackathon, Hamburg.

The brief describes a quadcopter that crossed a cargo airport undetected and
then sat on the tarmac for hours. Two different problems hide in that sentence:
seeing a small low-signature aircraft at all, and still seeing it once it has
landed and stopped moving. Motion detection solves the first and is completely
blind to the second.

Work in progress — roughly half built. See [Status](#status).

## Hardware

- Raspberry Pi 4
- HIKMICRO Mini2Plus V2 thermal camera (256x192, 25 Hz, USB-C)
- Raspberry Pi Camera Module 2

Under £400 of parts. Commercial thermal/EO perimeter systems start around
$50k.

## Running it

You do not need the cameras to run this. The simulated source renders both
views from one flat world, so the whole pipeline works on a laptop.

```bash
pip install -r requirements.txt
python scripts/selftest.py            # 16 checks, no hardware
python scripts/run_dashboard.py       # http://localhost:8000
```

Scenarios worth looking at:

```bash
python scripts/run_dashboard.py --scenario intrusion   # flies in, lands, cools
python scripts/run_dashboard.py --scenario crossover   # no thermal signature at all
```

On the Pi, with the cameras attached:

```bash
python3 scripts/thermal_probe.py 20            # check the thermal camera first
python3 scripts/calibrate_homography.py --source hardware
python3 scripts/run_dashboard.py --source hardware --calibration data/calibration/H.json
```

Bring-up notes, including the USB driver situation, are in
[docs/pi4_thermal_bringup.md](docs/pi4_thermal_bringup.md).

## How it works

Thermal leads because a warm object against cold sky is an easy signal and it
works at night. Three detection paths run alongside each other:

- **thermal motion** — background subtraction over about two seconds. Fast,
  and blind to anything that stops.
- **site baseline** — a model of the scene built over ninety seconds. Slow,
  and the only thing that sees a drone that has already landed.
- **optical motion** — independent detection in the visible band. Covers
  thermal crossover at dawn and dusk, and cold-soaked airframes.

Each camera can put a question to the other through the homography. Thermal
asks "what shape is this", because it finds small things well and cannot tell
a bird from a quadcopter. Optical asks "is this warm", because it reads shape
well and cannot tell an aircraft from a cloud edge. Everything is associated
before assessment, so one object seen by both sensors is one alert rather than
four.

Classification is rules over measured features, not a learned model. There is
no labelled thermal drone data to train on, it has to run on a Pi CPU, and
when someone asks why it fired the answer should be a sentence rather than a
number. The two features that survive at range are wingbeat — a bird modulates
its silhouette several times a second, a rigid airframe does not — and motor
hotspots measured against the object's own mean rather than against the sky,
since a 31°C bird against −5°C sky is exactly as bright as a drone.

Threat is a noisy-OR over three independent signals: what it is, whether the
site has seen anything like it, and whether policy permits it there. Any one
of them is sufficient on its own and none can veto the others.

Longer notes: [site intelligence](docs/site_intelligence.md),
[cross-cueing](docs/cross_cueing.md), [policy](docs/policy.md),
[dashboard](docs/dashboard.md).

## Site policy

The baseline learns what is normal. Policy states what is allowed, which is a
different thing and cannot be learned — anything present while the model is
learning becomes part of its definition of normal, which is also the obvious
way to blind the system deliberately.

Policy is written in English and compiled once into JSON rules that a person
reads before they govern anything. Runtime is arithmetic. A language model is
allowed to write the file and never to run in the frame loop.

```bash
python scripts/compile_policy.py --example
python scripts/run_dashboard.py --policy configs/policy.json
```

## Range

The camera is specified 0.1 m to 50 m, and its 6.9 mm f/1.0 lens is hyperfocal
at about 4 m, so focus is not the limit. Angular resolution is: 1.70 mrad per
pixel works out at roughly 70 m to detect a Mini-class drone as a couple of
warm pixels and roughly 18 m to have enough pixels to judge shape.

These are geometric numbers. They say nothing about thermal contrast falling
off with distance, atmospheric attenuation, or the camera's own temporal
filtering, none of which have been measured outdoors yet.

## Status

Working and verified against real hardware:

- thermal driver, 250 consecutive frames at 25 Hz with none dropped
- temperature calibration, matching the reference implementation to 6e-06 °C
- detection, tracking, classification, alerting, dashboard
- site baseline, cross-cueing, policy
- kiosk deployment with a desktop launcher

Not done:

- no real drone footage yet, so every accuracy number quoted anywhere in this
  repo comes from simulation and should be treated as such
- classifier thresholds are tuned against that simulation, not against birds
- at 640x480 the optical camera has the same angular resolution as the
  thermal one, so it adds no shape information at range. Capturing a
  high-resolution crop on cue would take shape classification from about 18 m
  to about 94 m, and is the most valuable thing left to build
- the camera applies internal temporal filtering, which may smear fast
  crossing targets. Untested
- absolute temperature accuracy is unverified against any reference. Detection
  uses contrast so this does not matter yet, but it would if thermal magnitude
  became a classification feature

## Layout

```
src/astravigil/     drivers, detection, tracking, fusion, site intelligence, policy
scripts/            entry points - dashboard, calibration, selftest, benchmarks
deploy/             kiosk launcher and teardown
docs/               longer explanations of each subsystem
configs/            udev rule, example policy
```

## Testing without hardware

`scripts/selftest.py` runs the pipeline end to end against the simulator and
checks sixteen behaviours, including the ones that are easy to get quietly
wrong: that calibration recovers a known homography, that a clean scene
produces no alerts, that a settled object is still found with no motion cue,
and that an object seen by both cameras produces one entry rather than two.

The simulator exists to make frame matching checkable rather than to look
convincing. Both views are rendered from one flat world by two known
homographies, so calibration can be scored against ground truth instead of
eyeballed.

## Author

Utkarsh Maurya, MSc Information and Communication Systems, TUHH Hamburg.
