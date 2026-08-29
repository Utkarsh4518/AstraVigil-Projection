# Cross-cueing — each camera asking the other

Both cameras detect independently, and either can put a question to the other
through the homography. They are not asking the same question, because they are
not good at the same things.

```
        THERMAL                                   OPTICAL
   finds small things well              reads shape well
   cannot tell what they are            cannot tell aircraft from clutter
        │                                         │
        │  "warm object here.                     │  "moving object here.
        │   what shape is it?"  ──────────────►   │   is it warm?"
        │                       ◄──────────────   │
        └──────────────► one object, one verdict ─┘
```

The reverse direction is what earns the architecture. A thermal-led pipeline
has two hard blind spots, and in both of them optical is the only sensor still
producing a signal:

- **Thermal crossover.** Twice a day, at dawn and dusk, the background
  temperature sweeps through ambient and the contrast the whole detector rests
  on goes to zero.
- **A cold-soaked airframe.** A drone parked for hours is at ambient. There is
  no heat signature left to find — only a shape.

Without a reverse path the system simply never hears about either.

## Two rules that keep fusion from making things worse

**Associate before assessing.** If both sensors find the same aircraft and both
raise it, the operator gets two alarms for one object — precisely the "wall of
separate alarms" the brief exists to complain about. Detections are paired
through the homography first, and the pair is judged once.

**Disagreement must be allowed to lower confidence.** Each sensor reports its
own raw measurement and they are combined exactly once. If optical's answer
were derived from a region thermal chose, and thermal's confidence then rose
because optical agreed, the system would be confirming its own guess. A fusion
that can only ever increase certainty is not fusing anything.

```
agree        →  noisy-OR, confidence rises
optical mute →  thermal's verdict stands unchanged
thermal mute →  optical leads, at 80% strength (one sensor, unsupported)
disagree     →  stronger side wins at HALF confidence, flagged
                "SENSORS DISAGREE - thermal said drone, optical says bird"
```

## The finding that shapes this whole design

**As configured, the optical camera has the same angular resolution as the
thermal camera, so its shape verdict is worthless at range.**

| | IFOV | 0.25 m drone spans 8 px at |
|---|---|---|
| thermal, 256 px over 25° | 1.704 mrad/px | 18.3 m |
| optical @ 640×480, 62.2° | 1.696 mrad/px | 18.4 m |
| optical @ 1640 (2×2 binned) | 0.662 mrad/px | 47.2 m |
| optical @ 3280 (IMX219 native) | 0.331 mrad/px | **94.4 m** |

`sources.py` configures the Pi Camera at 640×480 and throws away **5× the
resolution the sensor actually has**. At that setting optical adds no shape
information over thermal — a target that was four pixels in thermal is four
pixels in optical.

The code says so out loud rather than inventing a verdict: `verify_optical()`
scales its confidence by pixels on target and returns confidence `0.00` with
`"only 14 px on target - too few to judge shape"` below 60 px. In the current
simulation that is what it reports most of the time, which is the honest
answer.

**Cross-cueing is what makes fixing this affordable.** You do not need a
3280-wide frame every frame — you need a high-resolution crop only where
thermal is pointing. picamera2 supports a `lores` stream for detection
alongside a full-resolution `main` for on-demand capture. That is the
single highest-value change left, and it would extend shape classification from
~18 m to ~94 m, directly attacking the project's stated weakest point.

**Not built yet, and it needs the hardware to test.**

## What runs now

| Stage | File | Cost (x86) |
|---|---|---|
| optical detection, ROI-limited, 1 frame in 3 | [detection/optical.py](../src/astravigil/detection/optical.py) | 0.04 ms |
| thermal → optical shape check | [fusion/crosscue.py](../src/astravigil/fusion/crosscue.py) | 0.04 ms |
| optical → thermal heat check | same | 0.01 ms |
| **cross-cue total** | | **0.09 ms** |

Two deliberate choices in the optical detector:

**Absolute difference, not signed.** Thermal only ever looks for positive
excursions, because warm is bright. An airframe against a bright sky is
*darker* than its background, so looking for bright blobs would miss every
daylight target.

**Restricted to the thermal footprint.** The thermal frame covers about **15%
of the optical frame's area** (25°×18.8° inside 62.2°×48.8°). Outside that
overlap an optical detection cannot be cross-checked against anything, and on a
fixed mount there is no way to slew and look. Those detections are still
reported, labelled *"outside the thermal field of view — cannot be
cross-checked on a fixed mount"*, rather than silently treated as cleared.

## Optical-only contacts and why they stay at WATCH

An optical-only contact has no thermal track and no thermal deviation, so it
gets neither the tracker's dwell nor the site baseline's persistence. Without
help it would be judged fresh on every frame and could never escalate — losing
exactly the cold-soaked-drone case this path exists for.

`OpticalContactLog` gives it dwell, keyed on quantised position rather than by
tracking, because a stationary object does not need a tracker and a moving one
is thermal's job. The score then ramps over the first minute:

```
t=  5s  unverified contact    threat 0.17  nominal
t= 21s  cold contact          threat 0.32  nominal
t= 37s  cold contact          threat 0.48  watch
t= 69s  cold contact          threat 0.70  watch
```

It stops at 0.70, below the 0.75 alert line, and that is deliberate. With no
heat signature and a single sensor, the system genuinely cannot separate a cold
airframe from a parked bin or a shadow that stopped moving. It says *look at
this* without claiming to know. Lower `--alert-threshold` if you want it to
shout; the score is not capped, the operator's threshold governs.

## Scenarios

| `--scenario` | What it exercises |
|---|---|
| `clean` | routine traffic only — the baseline to learn from |
| `patrol` | a drone and a bird on routine tracks |
| `intrusion` | drone lands on the apron, motors cool over ~25 s |
| `resident` | drone already on the ground at boot — site baseline only |
| `crossover` | **drone at exactly ambient — thermal is blind, optical carries it alone** |

`crossover` is the one that tests this module. The target is drawn into a
throwaway array so it leaves no thermal signature of any kind. Matching the
background by sampling a single pixel does *not* work — the sky has a vertical
gradient across the target's own footprint and the residual edge contrast is
enough for the detector to find it. Blind has to mean blind.

## Verified

`scripts/selftest.py`, sections 8 and 9:

```
8. CROSS-CUEING - each camera asking the other
  [PASS] optical camera detects independently          1 optical candidate
  [PASS] thermal asks optical for a shape check        2 checks this frame
  [PASS] no shape verdict from too few pixels          100px conf 0.10; 14px conf 0.00
  [PASS] one object seen by both sensors is ONE entry  1 paired, 2 assessments

9. THERMAL CROSSOVER - the reverse cue carrying alone
  [PASS] thermal raised nothing - it is genuinely blind here
  [PASS] optical found it anyway                       cold contact
  [PASS] optical asked thermal and got an honest 'no heat'
  [PASS] escalates with dwell rather than sitting at zero   0.70 after 77 s
  [PASS] stays a WATCH, not an ALERT - one sensor, no corroboration
```

## Honest limitations

- **The two cameras are not time-synchronised.** `sources.frames()` takes the
  newest thermal frame the driver has and grabs an optical frame now — 0–40 ms
  of skew, ~6 px of misregistration for a drone crossing at 10 m/s at 20 m. It
  is zero for a stationary target, so it does not affect the settled-object
  cases, and it is worst exactly where cross-cueing matters least.
- **Optical shape classification is currently near-useless at range**, for the
  resolution reason above. It is wired, measured, and honest about it.
- **Optical detection will produce clutter outdoors** — cloud edges, shadows,
  foliage, rain. The ROI restriction and the thermal heat check suppress most
  of it in principle; none of that has been tested against real weather.
- Everything here is validated **in simulation only**.
