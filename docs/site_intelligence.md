# Adaptive Site Intelligence — learning what belongs here

> "The drone sat on the tarmac for hours."

That sentence is the challenge brief, and it is also a precise description of
the case a motion detector cannot handle. AstraVigil's thermal detector answers
*what moved in the last two seconds*. A drone that lands and stops moving is
absorbed by a two-second background model in two seconds. A drone that was
already on the ground when the system booted is written into the very first
background frame and is invisible for as long as the process runs.

This layer answers a different question — *does this belong here* — on a
different timescale, and persists what it learns to disk.

```
                Thermal + Optical
                        |
                 Object detection
                        |
                  Object tracking
                        |
        +---------------+---------------+
        |                               |
  Object classifier              Site baseline
  "what is it?"                  "does it belong?"
  stable, hand-tuned             learns continuously
        |                               |
        +---------------+---------------+
                        |
                  Threat fusion            noisy-OR: either
                        |                  signal alone can
                  One alert per object     raise the alarm
```

## The two models, and why they are separate

| | Motion detector | Site baseline |
|---|---|---|
| file | `detection/thermal.py` | `site_intelligence/baseline.py` |
| timescale | ~2 s | ~90 s, persisted between runs |
| resolution | per pixel | 8×8 px cells (24 × 32 grid) |
| sees | things that are moving now | things that are where they should not be |
| blind to | anything stationary | anything that leaves before ~3 s |

The site baseline holds two things:

**Scene reference** — per cell, a running mean and variance of temperature.
Stored as an offset from the frame's own median rather than as an absolute
temperature, so the model survives the scene warming through the day without
needing separate time-of-day buckets that would take days of observation to
fill.

**Activity model** — per cell, how many times traffic has been seen there,
plus running statistics of how big and how fast the things that pass usually
are.

## What makes an object suspicious

Four independent channels, combined so the strongest one dominates with a
small bonus when several agree:

| Channel | Question | Notes |
|---|---|---|
| `position` | has traffic ever used this space? | averaged over the track's recent path, not its current pixel |
| `appearance` | is this patch off its learned temperature? | gated on *how long* it has been off, so it does not merely restate the motion detector |
| `feature` | is it an unusual size, speed or heat for this site? | z-scores against the running statistics |
| `dwell` | has it stopped somewhere nothing stops? | wall-clock seconds, not frames |

Each is scaled by how much the model actually knows — a baseline with thirty
frames in it reports nothing, and the dashboard says `learning` rather than
going quietly green.

## Fusing identity with site anomaly

```python
threat = 1 - (1 - identity_risk) * (1 - site_risk)
```

A noisy-OR, not a weighted average, and that choice is the design. An average
lets a confident "bird" drag down a screaming site anomaly, and lets a quiet
site drag down a confident "drone". The noisy-OR says either signal alone is
sufficient cause — which is exactly what the project's own safety note demands:
`known object + abnormal behaviour` and `unknown object + anomalous behaviour`
must both stay flaggable.

`identity_risk` is the classifier's drone confidence, `0` for a confident bird,
and a floor of `0.15` for an unclassified mover — because "we could not tell"
is not the same as "harmless", and at 40 m most objects are four warm pixels
and will be unclassified.

## One alert per object

An alert is a stateful object, not an event. It opens once, updates in place
while the thing is still there, and closes when it leaves — so the operator
sees one row saying *drone, on the apron, 4 min 12 s*, which is the sentence
the airport in the brief needed and did not get. Three mechanisms hold that
line: a debounce before opening, a stable identity key, and a grace period so
a track that blinks does not reopen as a second incident. Everything is also
appended to `data/alerts/alerts.jsonl`.

When a landed drone is caught by both paths at once — the tracker never let go
of it *and* the site model watched it become part of the scene — the track wins
and the settled region is suppressed. One aircraft, one alarm.

## Three failure modes this had, and how each is closed

These are worth knowing about because each one looked like it worked until it
was measured.

**The anomaly erased itself.** A cell that keeps learning while an object sits
on it does not just drag its mean towards the object — it inflates its own
variance by the square of the deviation. A few dozen frames of that pushes the
z-score back under the threshold, and the intruder vanishes before the
persistence counter ever reaches the reporting level. Fixed by gating the
reference update on the *instantaneous* anomaly test. Anomalous cells still
learn, at 1/40th the rate, so a genuine slow change — sun coming round onto a
wall — settles out within about an hour instead of being flagged for ever.

**The intruder normalised itself.** The activity model counts where traffic is
seen. A drone that lands and is counted teaches the model that its own hiding
place is busy, and within a minute its position stops being novel. Stationary
objects are therefore not counted as traffic, and neither are objects currently
under alert — something the system is already calling anomalous does not get to
vote on what normal looks like.

**Novelty grew with runtime.** Familiarity was first written as a *rate* —
sightings per frame. The denominator grows for ever and the numerator only
grows when something is actually there, so every cell drifts towards "novel"
the longer the system runs. A bird that had patrolled the same perimeter for a
minute read as more anomalous than one that had just arrived. It is a count
now, not a rate.

## Running it

Learn the site first. Anything present and stationary while this runs becomes
part of the definition of normal — which is correct behaviour, and also the
obvious way to blind the system on purpose.

```bash
python scripts/learn_site.py --seconds 90              # simulated clean scene
python3 scripts/learn_site.py --source hardware --minutes 20   # at the rig
```

Then watch against it:

```bash
# a drone flies in, lands, and sits there while its motors cool
python scripts/run_dashboard.py --scenario intrusion

# the harder half: it was already on the ground before boot, so nothing
# ever moved and motion detection is structurally blind to it
python scripts/run_dashboard.py --scenario resident \
       --site-model data/baseline/site.npz
```

Useful flags: `--no-learn` freezes the model, `--save-site` writes it back on
exit, `--alert-threshold` moves the line.

The dashboard's **site** view draws the model itself: green is learned traffic,
magenta is a patch that has been off its learned temperature long enough to be
an object. An adaptive system that cannot show you its own model is one nobody
will trust in a control room — and it is the only way to tell "it has learned
the scene and there is nothing there" apart from "it has learned nothing and is
silent for that reason".

**Normal here** on an alert folds that object into the baseline. It writes the
current *scene* at that location, not the object's identity, so a different
object arriving at the same spot tomorrow is still anomalous. That distinction
is the difference between a site baseline and a mute button.

## Simulated scenarios

| `--scenario` | What happens |
|---|---|
| `clean` | one bird on routine patrol, no drone at all — the only honest thing to learn a baseline from |
| `patrol` | a drone and a bird crossing on routine tracks |
| `intrusion` | a drone flies in, lands on the apron, and sits there while its motors cool towards its airframe |
| `resident` | a drone already on the ground before frame one, fully cold-soaked |

`intrusion` is worth watching for a full minute. The motor hotspot — the
strongest cue the classifier has — decays with a 25 s time constant, and the
airframe follows with a 70 s one. The thermal signature fades while the object
is still sitting on the apron. What is left is a small warm patch that has not
moved and was not there yesterday, and the site model is the only thing that
can still see it.

## Measured

Verified in simulation by `scripts/selftest.py`, sections 6 and 7:

```
6. SITE INTELLIGENCE
  [PASS] site baseline matured                    scene 100%, traffic 59%
  [PASS] clean scene raises no alerts             0 open
  [PASS] clean scene has no settled anomalies     0 cells off baseline
  [PASS] routine traffic stays below the alert line   highest threat 0.23
  [PASS] site model survives save and reload      900 frames

7. THE TARMAC CASE - an object that never moved
  [PASS] settled object detected with no motion cue     1 settled-object alert
  [PASS] reported as ONE alert, not one per frame       1 over 500 frames
  [PASS] routine traffic did not alert alongside it     0 track alerts
         "settled object" threat 0.84, still for 20 s
         because: object at rest 20 s where the learned site is empty;
                  +4.0 C off baseline over 6 cells
```

Cost per frame, x86 laptop:

| Stage | ms |
|---|---|
| `site.observe` (learn) | 0.31 |
| `site.score` (2 objects) | 0.22 |
| `site.static_anomalies` | 0.10 |
| **site intelligence total** | **0.63** |
| whole pipeline, capture excluded | 3.4 |

The site model runs on a 24 × 32 grid rather than 49k pixels, which is why it
is cheap enough to be free next to detection.

## Honest limitations

- **Everything above is simulation.** The scenarios were written by the same
  person who wrote the thresholds. It demonstrates the mechanism is sound and
  the failure modes are closed; it says nothing about a real apron in real
  weather. Real footage is still on the critical path.
- **The scene reference assumes the camera does not move.** A bumped mount
  invalidates the whole baseline, exactly as it invalidates the homography.
  There is no automatic detection of this yet — a large-area change trips the
  global-change guard and the model re-learns rather than alarming, which is
  the safe direction but not the same as noticing.
- **Sun, rain and thermal crossover will produce false settled objects.** The
  region-size cap and the global-change guard catch the big ones. A patch the
  size of a drone that changes for real reasons will be reported, and an
  operator has to accept it.
- **Time-of-day is handled by working in relative temperature, not by
  bucketing.** That covers the scene warming uniformly. It does not cover a
  wall that is only sunlit in the afternoon. Bucketing is the right answer and
  it needs days of observation to fill the buckets.
- **The activity model has no concept of a restricted zone.** It learns where
  traffic goes; it does not know that the apron is off-limits and the perimeter
  road is not. Drawing zones on the frame is straightforward and is not built.
