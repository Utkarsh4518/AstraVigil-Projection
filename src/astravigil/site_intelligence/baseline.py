"""Adaptive Site Intelligence: learn what this site normally looks like.

The detector in `detection/thermal.py` answers "what moved in the last two
seconds". That is a different and much shorter question than "does this belong
here", and the gap between them is the whole scenario in the challenge brief:
a quadcopter that lands and sits still stops moving, and a two-second
background model absorbs it in two seconds. Worse, anything already on the
ground when the system boots is baked into the very first background frame and
is invisible for ever after.

So this module keeps a second, much slower model of the site, on a completely
different timescale, and persists it to disk:

  SCENE REFERENCE   per-cell thermal statistics, adapting over ~90 s and
                    saved between runs. Answers "is this patch of ground
                    where it usually is, thermally?" - which catches a settled
                    object with no motion cue at all, including one that was
                    already there at startup.

  ACTIVITY MODEL    per-cell counts of where movers are normally seen, plus
                    running statistics of how big and how fast they normally
                    are. Answers "do things normally move here, and do they
                    normally look like this?"

Two design rules are carried over from the project's own safety note, and both
are load-bearing:

  1. This layer never overrides the object classifier. It produces a separate
     score that gets fused with it, so `known object + abnormal behaviour` and
     `unknown object + anomalous behaviour` both stay flaggable.

  2. It learns a baseline of expected *behaviour*, not a whitelist of objects.
     Concretely: cells the model currently considers anomalous are excluded
     from the scene update, so a drone that sits on the apron does not quietly
     teach the system that drones on the apron are fine. The cost is that a
     genuine slow scene change - sun creeping across a wall - stays flagged
     until an operator accepts it. That trade is deliberate; accept() is the
     release valve.

Relative, not absolute, temperature is what gets stored. Each cell is recorded
as its offset from the frame's own median, so the whole model survives the
scene warming through the day without needing separate time-of-day buckets it
would take days of observation to fill.
"""
import json
import math
import os
import time

import cv2
import numpy as np

CELL_PX = 8                 # 8 px cells -> a 24 x 32 grid over 192 x 256

# --- scene reference -----------------------------------------------------
NOISE_FLOOR_C = 0.35        # below this a "deviation" is just sensor noise
Z_ENTER = 4.5               # sigma above the learned cell before it counts
MIN_DEV_C = 1.0             # ...and it must also be this many degrees off
PERSIST_S = 3.0             # how long a cell must stay off before reporting
PERSIST_DECAY = 3           # counters fall 3x faster than they rise
ADAPT_S = 90.0              # slowest adaptation rate of the reference
MAX_ANOM_FRACTION = 0.10    # more than this is a global change, not intruders
MAX_REGION_CELLS = 80       # bigger than this is scenery, not an aircraft
# How much slower a cell learns while it is reading as anomalous. Not zero:
# see _update_reference for why neither zero nor one works.
ANOMALY_SLOWDOWN = 40.0
# A cell may not be called anomalous until it has this many frames behind it.
# The first frames carry almost the whole weight of the mean - alpha is 1/n -
# so anything that happens to be sitting in a cell at startup writes itself
# into the reference, and then reads as a glaring anomaly the moment it moves
# away. Five seconds of history is enough for that to wash out.
MIN_LEARNED_FRAMES = 125

# --- maturity ------------------------------------------------------------
# A baseline with thirty frames in it knows nothing, and a system that is
# confidently wrong in front of judges is worse than one that says "learning".
SCENE_MATURITY_FRAMES = 400     # ~16 s at 25 fps
ACTIVITY_MATURITY_OBS = 1500    # ~60 s of one continuous track

# --- activity ------------------------------------------------------------
# A cell counts as a known corridor once traffic has been seen in it about
# this many times. Deliberately a COUNT, not a rate: expressed as a fraction
# of frames, every cell drifts towards "novel" the longer the system runs,
# because the denominator grows and the numerator only grows when something
# is actually there. That made a bird that had patrolled the same perimeter
# for a minute read as more anomalous than one that had just arrived.
FAMILIAR_COUNT = 20.0
MIN_TRAFFIC_SPEED_PX = 0.5  # slower than this is a parked object, not traffic
DWELL_WARN_S = 5.0          # settling starts to count against an object here
DWELL_FULL_S = 20.0         # ...and is fully anomalous by here


class _Running:
    """Incremental mean and variance with a floor on the adaptation rate.

    Plain Welford converges and then stops moving, which is wrong for a site
    that changes; a plain EWMA never settles, which is wrong for the first
    minute. This is Welford until the count reaches 1/alpha_floor and an EWMA
    after, so it is accurate early and stays adaptive later.
    """

    __slots__ = ("n", "mean", "var", "alpha_floor")

    def __init__(self, alpha_floor=1.0 / 3000.0):
        self.n = 0
        self.mean = 0.0
        self.var = 0.0
        self.alpha_floor = alpha_floor

    def update(self, x):
        self.n += 1
        a = max(1.0 / self.n, self.alpha_floor)
        d = float(x) - self.mean
        self.mean += a * d
        self.var = (1.0 - a) * (self.var + a * d * d)

    def z(self, x, floor):
        s = max(self.var ** 0.5, floor)
        return abs(float(x) - self.mean) / s

    def as_dict(self):
        return {"n": self.n, "mean": self.mean, "var": self.var}

    def load(self, d):
        self.n = int(d.get("n", 0))
        self.mean = float(d.get("mean", 0.0))
        self.var = float(d.get("var", 0.0))
        return self


class Novelty:
    """Why - and how strongly - the site model thinks something is out of place."""

    __slots__ = ("position", "appearance", "feature", "dwell", "overall",
                 "reasons")

    def __init__(self, position=0.0, appearance=0.0, feature=0.0, dwell=0.0,
                 reasons=None):
        self.position = position
        self.appearance = appearance
        self.feature = feature
        self.dwell = dwell
        self.reasons = reasons or []
        parts = (position, appearance, feature, dwell)
        # Dominated by the strongest single cue, with a small bonus when
        # several agree. A plain noisy-OR over four mild signals reads as
        # near-certainty, which is how an anomaly detector earns a reputation
        # for crying wolf.
        self.overall = 0.8 * max(parts) + 0.2 * (sum(parts) / len(parts))

    def as_dict(self):
        return {"position": round(self.position, 3),
                "appearance": round(self.appearance, 3),
                "feature": round(self.feature, 3),
                "dwell": round(self.dwell, 3),
                "overall": round(self.overall, 3),
                "reasons": list(self.reasons)}


class StaticAnomaly:
    """A patch of scene that has been wrong for a while and is not moving.

    This is the tarmac case: something is sitting there that was not there
    when the site was learned. There is no motion cue and there may never have
    been one - the object can have arrived while the system was switched off.
    """

    __slots__ = ("box", "cells", "dwell_s", "peak_dev_c", "mean_dev_c", "key",
                 "centroid")

    def __init__(self, box, cells, dwell_s, peak_dev_c, mean_dev_c, centroid):
        self.box = box                  # (x, y, w, h) in thermal pixels
        self.cells = cells
        self.dwell_s = dwell_s
        self.peak_dev_c = peak_dev_c
        self.mean_dev_c = mean_dev_c
        self.centroid = centroid
        # Quantised so a stationary object keeps one identity frame to frame
        # and produces one alert rather than a new one every time its extent
        # wobbles by a cell.
        self.key = f"static:{int(centroid[1]) // 16}:{int(centroid[0]) // 16}"

    def as_dict(self):
        return {"key": self.key,
                "box": [int(v) for v in self.box],
                "cells": int(self.cells),
                "dwell_s": round(float(self.dwell_s), 1),
                "peak_dev_c": round(float(self.peak_dev_c), 2),
                "mean_dev_c": round(float(self.mean_dev_c), 2)}


class SiteBaseline:
    def __init__(self, shape=(192, 256), cell_px=CELL_PX, fps=25.0,
                 adapt_s=ADAPT_S, persist_s=PERSIST_S):
        self.shape = tuple(shape)
        self.cell_px = int(cell_px)
        self.fps = float(fps)
        self.gh = max(1, self.shape[0] // self.cell_px)
        self.gw = max(1, self.shape[1] // self.cell_px)
        self.persist_frames = max(1, int(persist_s * fps))
        self.alpha_floor = 1.0 / max(1.0, adapt_s * fps)
        self.adapt_s = adapt_s

        g = (self.gh, self.gw)
        self.ref_mean = np.zeros(g, np.float32)
        self.ref_var = np.zeros(g, np.float32)
        self.ref_n = np.zeros(g, np.float32)
        self.persist = np.zeros(g, np.int32)
        self.activity = np.zeros(g, np.float32)

        self.frames = 0
        self.activity_obs = 0
        self.feat = {"area": _Running(), "speed": _Running(),
                     "hotspot": _Running()}
        self.created = time.time()

        self._z = np.zeros(g, np.float32)
        self._dev = np.zeros(g, np.float32)
        self._rel = np.zeros(g, np.float32)
        self._rate = np.zeros(g, np.float32)
        self._count = np.zeros(g, np.float32)
        self._suppressed = False        # global change this frame

    # -------------------------------------------------------------- maturity
    @property
    def scene_maturity(self):
        return min(1.0, self.frames / float(SCENE_MATURITY_FRAMES))

    @property
    def activity_maturity(self):
        return min(1.0, self.activity_obs / float(ACTIVITY_MATURITY_OBS))

    @property
    def learning(self):
        return self.scene_maturity < 1.0

    # --------------------------------------------------------------- observe
    def _cells(self, celsius):
        """Frame to the cell grid. INTER_AREA is a box mean, which is exactly
        what a cell is, in one optimised pass rather than a reshape and a
        reduce."""
        return cv2.resize(np.asarray(celsius, np.float32),
                          (self.gw, self.gh), interpolation=cv2.INTER_AREA)

    def observe(self, celsius, tracks=None, exclude_ids=()):
        """Feed one calibrated frame plus the live tracks. Updates the model."""
        cells = self._cells(celsius)
        # Median of the grid, not of the full frame: 768 values instead of
        # 49k for the same robustness, and this runs every frame.
        level = float(np.median(cells))
        rel = cells - level
        self._rel = rel

        std = np.maximum(np.sqrt(self.ref_var), NOISE_FLOOR_C)
        dev = rel - self.ref_mean
        # Until a cell has real history its statistics are fiction.
        seen = self.ref_n >= MIN_LEARNED_FRAMES
        z = np.where(seen, np.abs(dev) / std, 0.0).astype(np.float32)
        self._dev, self._z = dev.astype(np.float32), z

        anom = seen & (z > Z_ENTER) & (np.abs(dev) > MIN_DEV_C)
        # A change across most of the frame is weather, a gain shift or the
        # sun, not a field full of intruders. Stand down and re-learn instead
        # of producing the wall of alarms the brief asks us to avoid.
        self._suppressed = bool(anom.mean() > MAX_ANOM_FRACTION)
        if self._suppressed:
            anom = np.zeros_like(anom)

        self.persist = np.where(anom, self.persist + 1,
                                np.maximum(self.persist - PERSIST_DECAY, 0))

        self._update_reference(rel, ~anom)

        self.frames += 1
        if tracks:
            self._update_activity(tracks, exclude_ids)
        if self.frames % 25 == 1:
            self._refresh_rate()
        return self

    def _update_reference(self, rel, learn):
        """Fold this frame into the reference, slowly, and not where it is wrong.

        The gate is on the *instantaneous* anomaly test rather than on the
        persistence counter, and that detail is the difference between this
        working and not. A cell that keeps learning while an object sits on it
        does not just drag its mean towards the object - it inflates its own
        variance by the square of the deviation, and a few dozen frames of
        that is enough to push the z-score back under the threshold. The
        anomaly quietly erases itself before the counter ever reaches the
        reporting level.

        Neither extreme works, though. Freezing anomalous cells outright means
        a genuine slow change - the sun coming round onto a wall - stays
        flagged for ever. So they learn, at 1/40th the rate: an unexplained
        object holds its anomaly for roughly an hour before being absorbed,
        which is far longer than it takes an operator to see the alert, and
        real scene changes still settle out on their own.
        """
        n = np.where(learn, self.ref_n + 1.0, self.ref_n)
        fast = np.maximum(1.0 / np.maximum(n, 1.0), self.alpha_floor)
        a = np.where(learn, fast,
                     self.alpha_floor / ANOMALY_SLOWDOWN).astype(np.float32)
        d = rel - self.ref_mean
        self.ref_mean += a * d
        self.ref_var = (1.0 - a) * (self.ref_var + a * d * d)
        self.ref_n = n

    def _update_activity(self, tracks, exclude_ids=()):
        """Learn where traffic normally passes.

        Two exclusions, both closing a loop that would otherwise let an
        intruder normalise itself:

          A stationary object is not traffic. Counting one would mean a drone
          that lands teaches the model that its own hiding place is busy, and
          within a minute its position stops being novel.

          Objects already under alert are not counted either. The project's
          safety note is explicit that the system must not learn that every
          repeatedly-observed object is friendly - so something we are already
          calling anomalous does not get to vote on what normal looks like.
        """
        for tr in tracks.values():
            if tr.misses > 0 or tr.hits < 2:
                continue
            if tr.id in exclude_ids or tr.speed_px < MIN_TRAFFIC_SPEED_PX:
                continue
            cx, cy = tr.centroid
            gx = int(np.clip(cx // self.cell_px, 0, self.gw - 1))
            gy = int(np.clip(cy // self.cell_px, 0, self.gh - 1))
            self.activity[gy, gx] += 1.0
            self.activity_obs += 1
            # log area: a 4 px and a 400 px blob differ by two orders of
            # magnitude, and a linear mean would be owned by the big one.
            area = tr.areas[-1] if tr.areas else 1.0
            self.feat["area"].update(math.log10(max(area, 1.0)))
            self.feat["speed"].update(tr.speed_px)

    def _refresh_rate(self):
        # Neighbourhood SUM, not average: a track passing one cell over should
        # count towards this cell being familiar. The grid is an
        # implementation detail, not a real boundary in the world.
        self._count = cv2.boxFilter(self.activity, -1, (5, 5), normalize=False)
        # Kept for the dashboard's learned-traffic overlay, where a rate is
        # the more readable quantity.
        self._rate = cv2.blur(self.activity / max(self.frames, 1), (3, 3))

    def note_hotspot(self, hotspot_c):
        self.feat["hotspot"].update(hotspot_c)

    # ------------------------------------------------------------- scoring
    def _cell_of(self, centroid):
        # Plain integer arithmetic, not np.clip. This is called once per
        # position in a track's history, and np.clip on a Python scalar costs
        # several microseconds - enough to make scoring the most expensive
        # stage in the pipeline, ahead of detection itself.
        gx = int(centroid[0]) // self.cell_px
        gy = int(centroid[1]) // self.cell_px
        return (min(max(gy, 0), self.gh - 1), min(max(gx, 0), self.gw - 1))

    def _path_count(self, det, track):
        """Mean traffic count over the track's recent positions."""
        points = list(track.positions) if track is not None else []
        if not points:
            points = [det.centroid]
        p = np.asarray(points, np.float32)
        gx = np.clip(p[:, 0].astype(np.int32) // self.cell_px, 0, self.gw - 1)
        gy = np.clip(p[:, 1].astype(np.int32) // self.cell_px, 0, self.gh - 1)
        return float(self._count[gy, gx].mean())

    def score(self, det, track, dwell_s=0.0):
        """How out of place is this object, and why."""
        reasons = []
        gy, gx = self._cell_of(det.centroid)

        # --- does traffic normally pass through here
        #
        # Averaged over the track's recent path rather than read off its
        # current pixel. A wandering bird will always eventually clip a cell
        # nothing has used before, and judging it on that one cell made it
        # spike to full novelty for a few frames at a time. What actually
        # distinguishes an intruder is a whole approach through space traffic
        # never uses, and that is what the history shows.
        count = self._path_count(det, track)
        familiar = 1.0 - float(np.exp(-count / FAMILIAR_COUNT))
        position = (1.0 - familiar) * self.activity_maturity
        if position > 0.6:
            reasons.append("approach through space traffic never uses")

        # --- is this patch of scene where it should be, and has it been
        # wrong long enough to be an object rather than a fly-past.
        #
        # The persist gate is what stops this channel from simply restating
        # what the motion detector already said: any mover puts its own cell
        # far off baseline for the moment it is there. Only something that
        # stays wrong is telling us something new.
        z = float(self._z[gy, gx])
        settled = min(1.0, self.persist[gy, gx] / float(self.persist_frames))
        appearance = min(1.0, z / 8.0) * self.scene_maturity * settled
        if appearance > 0.5:
            reasons.append(f"{z:.0f} sigma off the learned scene here")

        # --- does it look like the things normally seen here
        feature = 0.0
        if self.activity_maturity > 0.25:
            zs = {
                "size": self.feat["area"].z(math.log10(max(det.area, 1.0)), 0.15),
                "speed": self.feat["speed"].z(
                    track.speed_px if track else 0.0, 0.4),
                "heat": self.feat["hotspot"].z(det.hotspot_c, 0.8),
            }
            worst = max(zs, key=zs.get)
            feature = min(1.0, zs[worst] / 4.0) * self.activity_maturity
            if feature > 0.5:
                reasons.append(f"{worst} unlike anything seen here "
                               f"({zs[worst]:.1f} sigma)")

        # --- has it settled somewhere nothing settles
        dwell = 0.0
        if dwell_s > DWELL_WARN_S:
            ramp = min(1.0, (dwell_s - DWELL_WARN_S)
                       / max(DWELL_FULL_S - DWELL_WARN_S, 1e-6))
            # A familiar resting place damps this but never cancels it: a
            # drone landing where a tug normally parks is still a drone.
            dwell = ramp * (0.35 + 0.65 * position)
            if dwell > 0.3:
                reasons.append(f"stationary for {dwell_s:.0f} s")

        return Novelty(position, appearance, feature, dwell, reasons)

    # ---------------------------------------------------- static anomalies
    def static_anomalies(self):
        """Regions that have been off-baseline long enough to be real objects."""
        if self.frames < SCENE_MATURITY_FRAMES // 2:
            return []
        settled = (self.persist >= self.persist_frames).astype(np.uint8)
        if not settled.any():
            return []

        n, labels, stats, cents = cv2.connectedComponentsWithStats(settled, 8)
        out = []
        for i in range(1, n):
            cells = int(stats[i, cv2.CC_STAT_AREA])
            if cells > MAX_REGION_CELLS:
                continue                        # scenery-scale, not a target
            m = labels == i
            gx, gy = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            gw, gh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            dwell_frames = int(self.persist[m].max())
            out.append(StaticAnomaly(
                box=(int(gx * self.cell_px), int(gy * self.cell_px),
                     int(gw * self.cell_px), int(gh * self.cell_px)),
                cells=cells,
                dwell_s=dwell_frames / self.fps,
                peak_dev_c=float(np.abs(self._dev[m]).max()),
                mean_dev_c=float(self._dev[m].mean()),
                centroid=(float(cents[i][0] * self.cell_px),
                          float(cents[i][1] * self.cell_px)),
            ))
        out.sort(key=lambda a: a.dwell_s, reverse=True)
        return out

    def reset(self):
        """Forget this site completely and start again.

        The honest answer to a camera that has been moved. The scene reference
        adapts on its own within a few minutes, but the activity map does not -
        it only ever accumulates, so traffic corridors learned at the old
        aiming point would keep reading as familiar at the new one, quietly
        LOWERING sensitivity exactly where the model is now wrong.
        """
        g = (self.gh, self.gw)
        self.ref_mean = np.zeros(g, np.float32)
        self.ref_var = np.zeros(g, np.float32)
        self.ref_n = np.zeros(g, np.float32)
        self.persist = np.zeros(g, np.int32)
        self.activity = np.zeros(g, np.float32)
        self.frames = 0
        self.activity_obs = 0
        self.feat = {"area": _Running(), "speed": _Running(),
                     "hotspot": _Running()}
        self.created = time.time()
        self._z = np.zeros(g, np.float32)
        self._dev = np.zeros(g, np.float32)
        self._rel = np.zeros(g, np.float32)
        self._rate = np.zeros(g, np.float32)
        self._count = np.zeros(g, np.float32)
        self._suppressed = False
        return self

    def coverage(self):
        """Per-cell learned-ness, 0-100, as a flat list.

        This is what the console's scan animation draws. It is not decoration:
        each number is how much real history that cell actually has, so a
        patch of the scene the camera cannot see properly - glare, an occluded
        corner - stays visibly unfilled instead of being quietly assumed fine.
        """
        frac = np.clip(self.ref_n / float(MIN_LEARNED_FRAMES), 0.0, 1.0)
        return (frac * 100).astype(np.uint8).flatten().tolist()

    def accept(self, box=None):
        """Operator says this is normal - fold it into the reference now.

        Not a whitelist of objects: it writes the current *scene* into the
        model at that location, so a different object arriving at the same
        spot tomorrow is still anomalous. That distinction is the difference
        between a site baseline and a mute button.
        """
        if box is None:
            m = np.ones_like(self.persist, bool)
        else:
            x, y, w, h = box
            m = np.zeros_like(self.persist, bool)
            y0 = int(np.clip(y // self.cell_px, 0, self.gh - 1))
            x0 = int(np.clip(x // self.cell_px, 0, self.gw - 1))
            y1 = int(np.clip((y + h) // self.cell_px + 1, 1, self.gh))
            x1 = int(np.clip((x + w) // self.cell_px + 1, 1, self.gw))
            m[y0:y1, x0:x1] = True
        self.ref_mean[m] = self._rel[m]
        self.ref_var[m] = np.maximum(self.ref_var[m], NOISE_FLOOR_C ** 2)
        self.ref_n[m] = np.maximum(self.ref_n[m], 10.0)
        self.persist[m] = 0
        return int(m.sum())

    # ---------------------------------------------------------------- stats
    def stats(self):
        return {
            "frames": self.frames,
            "scene_maturity": round(self.scene_maturity, 3),
            "activity_maturity": round(self.activity_maturity, 3),
            "learning": self.learning,
            "cells": int(self.gh * self.gw),
            "anomalous_cells": int((self.persist >= self.persist_frames).sum()),
            "activity_obs": self.activity_obs,
            "adapt_s": self.adapt_s,
            "global_change": self._suppressed,
        }

    def novelty_image(self):
        """Cell z-map upsampled to frame size, for the dashboard."""
        return cv2.resize(self._z, (self.shape[1], self.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    # ------------------------------------------------------------ persistence
    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        meta = {
            "shape": list(self.shape), "cell_px": self.cell_px,
            "fps": self.fps, "adapt_s": self.adapt_s,
            "frames": self.frames, "activity_obs": self.activity_obs,
            "created": self.created, "saved": time.time(),
            "feat": {k: v.as_dict() for k, v in self.feat.items()},
        }
        np.savez_compressed(path, ref_mean=self.ref_mean, ref_var=self.ref_var,
                            ref_n=self.ref_n, activity=self.activity,
                            meta=np.array(json.dumps(meta)))
        return path

    @classmethod
    def load(cls, path):
        z = np.load(path, allow_pickle=False)
        meta = json.loads(str(z["meta"]))
        self = cls(shape=tuple(meta["shape"]), cell_px=meta["cell_px"],
                   fps=meta.get("fps", 25.0),
                   adapt_s=meta.get("adapt_s", ADAPT_S))
        self.ref_mean = z["ref_mean"].astype(np.float32)
        self.ref_var = z["ref_var"].astype(np.float32)
        self.ref_n = z["ref_n"].astype(np.float32)
        self.activity = z["activity"].astype(np.float32)
        self.frames = int(meta["frames"])
        self.activity_obs = int(meta["activity_obs"])
        self.created = meta.get("created", time.time())
        for k, d in meta.get("feat", {}).items():
            if k in self.feat:
                self.feat[k].load(d)
        self._refresh_rate()
        return self
