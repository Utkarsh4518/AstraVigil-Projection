"""What the optical camera has learned about this place.

The thermal baseline in baseline.py answers "is this patch at the temperature
it normally is". This answers the same question for the other sensor: is this
patch of the scene SHAPED the way it normally is. Until now the site half of
every verdict came from thermal alone - the docstring on assess_track said so
outright - and optical only ever answered questions thermal thought to ask.

WHY NOT JUST LEARN THE PIXELS

Raw colour and brightness track the sun, the clouds and the room lights far
more strongly than they track anything arriving on the ground. A model built
on them spends its whole adaptation window chasing the time of day, and its
variance inflates until nothing is ever anomalous - the classic way an
adaptive system learns its way into blindness.

So each cell carries two statistics chosen to survive a change in lighting:

  STRUCTURE   the cell's mean gradient magnitude, DIVIDED by the frame's
              median gradient. Edges belong to what is there rather than to
              how brightly it is lit - but only as a ratio. Dimming the room
              scales every gradient down together, so the absolute value moves
              and the ratio does not.

  TONE        the cell's brightness minus the frame median, divided by the
              frame's median absolute deviation. Subtracting the median
              cancels a light that gets brighter; dividing by the spread
              cancels one that gets dimmer. The first version did only the
              subtraction and a 45% dimming lit up 162 cells out of 300 -
              multiplicative changes survive an offset correction untouched.

Both are therefore dimensionless, and both are ratios against the frame's own
state that frame, which is what makes a change in the light a change in
nothing.

Two statistics rather than one because they fail on different things. A dark
object on a dark surface barely moves structure; a smooth pale object on a
textured floor barely moves tone. Something has to catch each.

WHAT IT IS FOR

Two jobs. It gives the thermal site model a second opinion on any detection
that maps into the optical frame, so "out of place" is a judgement both
cameras contribute to. And it holds a persistence counter per cell, which is
the only way to see the object this whole reverse path exists for: a
cold-soaked airframe, invisible to thermal, that was not there an hour ago.
"""
import time

import cv2
import numpy as np

# Optical frames are 640x480, six times the pixel count of thermal ones. A
# 32 px cell gives a 15x20 grid - coarse enough to be cheap at frame rate,
# fine enough that a person occupies a couple of cells rather than a fifth of
# the frame.
CELL_PX = 32

# NOT reduced. 1.0, and the failed attempt is worth recording.
#
# Halving each side before the Sobel looked free - this model only wants one
# mean gradient per 32-pixel cell, so the reasoning went, and half-resolution
# gradients average to the same thing. They do not. INTER_AREA before a
# Sobel changes the gradient distribution enough to break the property this
# whole file is built on: a 45% dimming of the room, which lights zero cells
# at full resolution, lit nineteen of three hundred.
#
# And it was bought for nothing. Under a profiler this call is under a
# millisecond; the wall-clock measurement that motivated it was of a
# benchmark loop that spent its time generating synthetic frames rather than
# processing them. A property this model depends on, traded away for a
# fraction of a millisecond that was never there.
WORK_SCALE = 1.0

# Frames of history before a cell's statistics mean anything. Below this the
# model reports its own immaturity rather than pretending to a baseline, for
# the same reason the thermal one does: silence from a model that has learned
# nothing is not evidence of quiet.
MIN_LEARNED_FRAMES = 90

# How fast a cell forgets. Slower than thermal: an optical scene has more
# short-lived clutter - shadows, reflections, someone walking past - and
# adapting quickly to those is how a model learns to accept an intruder.
ADAPT_S = 240.0

# Seconds a cell must stay off its baseline before it is an object rather
# than something passing through.
PERSIST_S = 6.0

# Floors, in the normalised units above, below which a deviation is noise
# rather than signal. Without them an untouched cell divides by its own tiny
# variance and reports enormous z-scores for nothing at all.
STRUCTURE_FLOOR = 0.10
TONE_FLOOR = 0.15

# z-score at which a cell counts as off-baseline.
Z_ANOMALOUS = 3.5

# Fraction of cells that must have a full history before the model will vote.
#
# Not 1.0, and it cannot be. A cell with something permanently in it learns at
# a twentieth of the normal rate by design, so on any real scene the mean
# never reaches the top - and a model that waits for it stays silent forever.
MATURE_AT = 0.9

# How often a cell has to be off its baseline before it is drawn as somewhere
# things happen, and how often before it is drawn at full strength. ABSOLUTE
# fractions, deliberately.
#
# The first version normalised against the busiest cell, which cannot fail to
# produce a bright map: divide by the maximum and something is always at 1.0,
# so a view where nothing whatsoever moves renders exactly as green as a
# doorway onto a corridor. That is the flat wash an operator reported as "just
# green, nothing useful". A floor in real units is what makes an empty map
# legitimately empty.
ACTIVITY_FLOOR = 0.02
ACTIVITY_FULL = 0.25

# The largest deviation, in sigma, that is allowed to enter a cell's VARIANCE.
#
# Slowing the mean down on an anomalous cell is not enough on its own, and the
# arithmetic says why. The variance update moves by alpha * d squared, so a
# 35-sigma object contributes 1225 times the current variance; even at the
# damped rate that is roughly a doubling of the variance every frame. Ten
# frames later sigma has grown thirty-fold, the object measures 1 sigma, and
# the cell has decided a stationary intruder is what normal looks like.
#
# Measured, not theorised: a block parked in one cell went from 35 sigma to
# 1.2 sigma in about 140 frames, resetting the persistence counter before it
# could reach the six seconds that mark something as settled. The one thing
# this model exists to catch - an object that arrives and stays - could not
# fire. Winsorising the update to 4 sigma keeps the variance responsive to
# ordinary drift and stops a single anomaly from teaching the cell to ignore
# it.
MAX_DEV_SIGMA = 4.0

# Above this many cells a settled patch is scenery changing - a door left
# open, a curtain drawn, the sun moving across a wall - rather than an object
# that has arrived. Set to the same fraction of frame area the thermal model
# uses: 80 cells of 8 px is a tenth of a 192x256 frame, and 30 cells of 32 px
# is a tenth of a 480x640 one.
MAX_REGION_CELLS = 30

# The smallest patch worth calling an object, and how much of its own
# bounding box it has to fill.
#
# Without these, every edge in the scene becomes a settled object the moment
# the baseline is a little stale - a camera nudged a millimetre, or a light
# changed after learning, leaves thin off-baseline traces along every
# boundary in the room. They pass the cell-count limit easily, because a
# trace two cells wide and ten long is twenty cells, and they arrive by the
# dozen. The extent test is what separates them: an object that has been put
# down fills its own box, and an outline does not.
MIN_REGION_CELLS = 3
MIN_REGION_EXTENT = 0.45

# Most settled patches to report at once.
#
# Past this, the honest reading is not "eleven objects have been placed in
# the room" but "the baseline no longer matches what this camera sees" -
# which is a different problem with a different fix, and the console should
# say so rather than draw eleven boxes.
MAX_REGIONS = 6


class OpticalRegion:
    """A patch of the optical scene that has looked wrong long enough to be
    an object. The optical mirror of the thermal model's StaticAnomaly.

    This is the ONLY optical channel that can see something stationary. The
    optical detector is motion-based, so an object that arrives and then stops
    fades out of it within seconds - a mug of hot water put down on a shelf is
    gone from the detector long before anyone asks about it. The baseline
    keeps reporting that patch for as long as it does not look the way this
    camera learned it looks, which is exactly as long as the object is there.
    """

    __slots__ = ("box", "cells", "dwell_s", "peak_z", "centroid", "key")

    def __init__(self, box, cells, dwell_s, peak_z, centroid, key):
        self.box = box                  # (x, y, w, h) in OPTICAL pixels
        self.cells = cells
        self.dwell_s = dwell_s
        self.peak_z = peak_z
        self.centroid = centroid
        self.key = key                  # assigned by OpticalBaseline

    def as_dict(self):
        return {"key": self.key,
                "box": [int(v) for v in self.box],
                "cells": int(self.cells),
                "dwell_s": round(float(self.dwell_s), 1),
                "peak_z": round(float(self.peak_z), 2)}


class OpticalBaseline:
    def __init__(self, shape=(480, 640), cell_px=CELL_PX, fps=25.0,
                 adapt_s=ADAPT_S, persist_s=PERSIST_S):
        self.shape = tuple(shape[:2])
        self.cell_px = int(cell_px)
        self.fps = float(fps)
        self.gh = max(1, self.shape[0] // self.cell_px)
        self.gw = max(1, self.shape[1] // self.cell_px)
        self.persist_frames = max(1, int(persist_s * fps))
        self.alpha_floor = 1.0 / max(1.0, adapt_s * fps)

        g = (self.gh, self.gw)
        self.struct_mean = np.zeros(g, np.float32)
        self.struct_var = np.zeros(g, np.float32)
        self.tone_mean = np.zeros(g, np.float32)
        self.tone_var = np.zeros(g, np.float32)
        self.ref_n = np.zeros(g, np.float32)
        self.persist = np.zeros(g, np.int32)
        # How often each cell is off its baseline: the optical equivalent of
        # learned traffic. This, not coverage, is the interesting map. Once
        # the model matures every cell has full history, so a coverage map is
        # uniform - it draws as a flat wash over the whole frame and says
        # nothing. Where change actually happens is sparse, and sparse is
        # what an operator can read.
        self.activity = np.zeros(g, np.float32)

        self._z = np.zeros(g, np.float32)
        self._last_cells = None
        # Last frame's settled patches, as (box, key). Identity for a patch
        # comes from overlapping one of these, not from where its centroid
        # happens to round to - see _region_key.
        self._prev_regions = []
        self.regions_found = 0
        self.frames = 0
        self.created = time.time()

    # ------------------------------------------------------------ maturity
    @property
    def maturity(self):
        return float(np.clip(self.ref_n / MIN_LEARNED_FRAMES, 0, 1).mean())

    @property
    def learning(self):
        return self.maturity < MATURE_AT

    def coverage(self):
        """Per-cell history, 0..1. What the learning map draws."""
        return np.clip(self.ref_n / max(MIN_LEARNED_FRAMES, 1), 0, 1)

    def stats(self):
        return {
            "frames": int(self.frames),
            "maturity": round(self.maturity, 3),
            "learning": bool(self.learning),
            "anomalous_cells": int((self._z > Z_ANOMALOUS).sum()),
            "settled_cells": int((self.persist >= self.persist_frames).sum()),
            "active_cells": int((self.activity >= ACTIVITY_FLOOR).sum()),
            "blind_cells": int((self.ref_n < MIN_LEARNED_FRAMES).sum()),
            "regions_found": int(self.regions_found),
            "cells": int(self.gh * self.gw),
        }

    # ------------------------------------------------------------- observe
    def _cells(self, bgr):
        """One frame to the two per-cell statistics."""
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        if WORK_SCALE != 1.0:
            grey = cv2.resize(grey, None, fx=WORK_SCALE, fy=WORK_SCALE,
                              interpolation=cv2.INTER_AREA)
        grey = grey.astype(np.float32)

        # Structure. Sobel magnitude, then a box mean per cell - INTER_AREA is
        # exactly that mean in one optimised pass.
        gx = cv2.Sobel(grey, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(grey, cv2.CV_32F, 0, 1, ksize=3)
        mag = cv2.magnitude(gx, gy)
        structure = cv2.resize(mag, (self.gw, self.gh),
                               interpolation=cv2.INTER_AREA)
        # As a ratio to this frame's own median gradient, so turning the
        # lights down moves every cell's absolute edge strength and no cell's
        # relative one.
        structure = structure / max(float(np.median(structure)), 1e-3)

        # Tone, centred on the frame median and scaled by its spread. Both
        # steps matter: the first cancels an additive lighting change, the
        # second a multiplicative one.
        #
        # Statistics of the GRID, not the full frame: 300 values instead of
        # 300k for the same robustness, and this runs every frame.
        tone = cv2.resize(grey, (self.gw, self.gh),
                          interpolation=cv2.INTER_AREA)
        med = float(np.median(tone))
        # 1.4826 * MAD is the usual robust stand-in for a standard deviation,
        # and unlike one it is not dragged around by the object we are looking
        # for.
        mad = 1.4826 * float(np.median(np.abs(tone - med)))
        tone = (tone - med) / max(mad, 1e-3)
        return structure, tone

    def observe(self, bgr, learn=True):
        """Feed one optical frame. Returns the per-cell z-score map.

        Reading and updating are separate jobs and `learn` only switches off
        the second. The z-scores and the persistence counters have to advance
        every frame regardless, or a frozen model stops seeing the settled
        object it exists to catch.
        """
        if bgr is None:
            return self._z
        if bgr.shape[:2] != self.shape:
            # The camera changed resolution under us. Rebuild rather than
            # silently squashing every frame into a grid of the wrong shape.
            self.__init__(shape=bgr.shape[:2], cell_px=self.cell_px,
                          fps=self.fps)

        structure, tone = self._cells(bgr)
        # Kept so accept() can adopt what the cell looks like RIGHT NOW as
        # its new baseline, which is what the operator is pointing at.
        self._last_cells = (structure, tone)
        self.frames += 1

        s_std = np.maximum(np.sqrt(self.struct_var), STRUCTURE_FLOOR)
        t_std = np.maximum(np.sqrt(self.tone_var), TONE_FLOOR)
        s_z = np.abs(structure - self.struct_mean) / s_std
        t_z = np.abs(tone - self.tone_mean) / t_std

        # The stronger of the two channels. They catch different things, so
        # requiring both to agree would forfeit whichever one is doing the
        # work on this particular object.
        z = np.maximum(s_z, t_z)
        # Until a cell has real history its statistics are fiction.
        z = np.where(self.ref_n >= 8, z, 0.0)
        self._z = z.astype(np.float32)

        hot = z > Z_ANOMALOUS
        self.persist = np.where(hot, self.persist + 1, 0)
        # Slower than the reference update, so a corridor used once a minute
        # still shows up next to one used constantly.
        self.activity += 0.01 * (hot.astype(np.float32) - self.activity)

        if learn:
            # Exponential update with a floor, so a cell that has been quiet
            # for hours still adapts to a slow change. Cells currently off
            # baseline are updated far more slowly - otherwise an object that
            # sits still is absorbed into the definition of normal, which is
            # precisely the failure this model exists to avoid.
            alpha = np.maximum(1.0 / np.maximum(self.ref_n + 1.0, 1.0),
                               self.alpha_floor)
            alpha = np.where(hot, alpha * 0.05, alpha).astype(np.float32)
            # History counts at the rate the cell is actually learning, not at
            # one frame per frame. Before this, ref_n rose by 1 everywhere on
            # every frame - so coverage was IDENTICAL in every cell and the
            # map that was meant to show the model filling in could only ever
            # be a uniform wash. Counting effective samples makes a cell with
            # somebody standing in it visibly lag its neighbours, which is
            # exactly the thing worth seeing.
            gain = np.where(hot, 0.05, 1.0).astype(np.float32)

            d_s = structure - self.struct_mean
            self.struct_mean += alpha * d_s
            self.struct_var += alpha * (
                np.minimum(d_s * d_s, (MAX_DEV_SIGMA * s_std) ** 2)
                - self.struct_var)
            d_t = tone - self.tone_mean
            self.tone_mean += alpha * d_t
            self.tone_var += alpha * (
                np.minimum(d_t * d_t, (MAX_DEV_SIGMA * t_std) ** 2)
                - self.tone_var)
            self.ref_n += gain

        return self._z

    # --------------------------------------------------------------- score
    def score(self, box):
        """How unlike itself does the optical scene look inside this box?

        Returns (0..1, reasons). The box is in OPTICAL pixels - map a thermal
        detection through the homography first.
        """
        reasons = []
        if self.learning:
            # An immature model must not vote. Reporting 0 here is not the
            # same as reporting "normal": it means "no opinion", and the
            # caller treats a missing opinion as no evidence either way.
            return 0.0, reasons

        x, y, w, h = box
        gx0 = max(0, int(x) // self.cell_px)
        gy0 = max(0, int(y) // self.cell_px)
        gx1 = min(self.gw, int(x + w) // self.cell_px + 1)
        gy1 = min(self.gh, int(y + h) // self.cell_px + 1)
        if gx1 <= gx0 or gy1 <= gy0:
            return 0.0, reasons

        patch_z = self._z[gy0:gy1, gx0:gx1]
        patch_p = self.persist[gy0:gy1, gx0:gx1]
        if patch_z.size == 0:
            return 0.0, reasons

        peak = float(patch_z.max())
        # z of 3.5 is the threshold; 9 is saturated. Anything between scales.
        novelty = float(np.clip((peak - Z_ANOMALOUS) / (9.0 - Z_ANOMALOUS),
                                0.0, 1.0))
        if novelty > 0:
            reasons.append(f"optical scene here is {peak:.1f} sigma off what "
                           f"this camera has learned")

        settled = int((patch_p >= self.persist_frames).sum())
        if settled:
            secs = float(patch_p.max()) / max(self.fps, 1e-6)
            novelty = max(novelty, 0.75)
            reasons.append(f"optically changed for {secs:.0f} s - something "
                           f"is sitting there")
        return novelty, reasons

    def settled_regions(self):
        """Patches that have looked wrong long enough to be objects.

        Guarded on maturity for the same reason score() is: a model that has
        learned nothing has no grounds to call anything unusual, and every
        cell of an empty model is off its own non-existent baseline.
        """
        settled = ((self.persist >= self.persist_frames).astype(np.uint8)
                   if not self.learning else None)
        if settled is None or not settled.any():
            # Nothing there - so nothing to carry an identity forward from
            # either. A patch that comes back after this is genuinely new.
            self._prev_regions = []
            return []

        n, labels, stats, cents = cv2.connectedComponentsWithStats(settled, 8)
        out = []
        for i in range(1, n):
            cells = int(stats[i, cv2.CC_STAT_AREA])
            if not MIN_REGION_CELLS <= cells <= MAX_REGION_CELLS:
                continue                    # noise, or scenery-scale
            gx, gy = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            gw, gh = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
            if cells < MIN_REGION_EXTENT * gw * gh:
                # An outline, not an object. A stale baseline traces every
                # edge in the room and each trace passes the cell count.
                continue
            m = labels == i
            box = (int(gx * self.cell_px), int(gy * self.cell_px),
                   int(gw * self.cell_px), int(gh * self.cell_px))
            centroid = (float(cents[i][0] * self.cell_px),
                        float(cents[i][1] * self.cell_px))
            out.append(OpticalRegion(
                box=box, cells=cells,
                dwell_s=float(self.persist[m].max()) / max(self.fps, 1e-6),
                peak_z=float(self._z[m].max()),
                centroid=centroid,
                key=self._region_key(box, centroid)))
        out.sort(key=lambda r: r.dwell_s, reverse=True)
        # Longest-settled first, then capped. Past the cap this is a stale
        # baseline rather than a room filling with objects, and the count is
        # reported so an operator can tell those two apart.
        self.regions_found = len(out)
        out = out[:MAX_REGIONS]
        self._prev_regions = [(r.box, r.key) for r in out]
        return out

    def _region_key(self, box, centroid):
        """A name for this patch that survives the patch changing shape.

        A quantised centroid on its own is not an identity. A patch that grows
        by one cell moves its centroid by half a cell, and a centroid that
        crosses a quantisation boundary renames the object - which renumbers
        its badge on every pane at once, which destroys the one thing the
        number exists to do.

        Overlapping the previous frame's patch is what actually means "this is
        the same thing", so that wins. The quantised centroid is only used to
        mint a name for a patch that is genuinely new.
        """
        ax, ay, aw, ah = box
        for (bx, by, bw, bh), key in self._prev_regions:
            if (ax < bx + bw and bx < ax + aw
                    and ay < by + bh and by < ay + ah):
                return key
        return (f"optical-static:{int(centroid[1]) // 64}:"
                f"{int(centroid[0]) // 64}")

    def accept(self, box):
        """An operator says this patch is normal here. Believe them.

        Adopts the cells under the box as their own new baseline in one step,
        rather than waiting out the adaptation window - which for a settled
        object is deliberately slow, because the whole point of this model is
        that a thing which arrives and stays does not get absorbed. An
        operator saying "that is my chair" is the one piece of evidence that
        should override that, and it should take effect immediately or they
        will click it again.
        """
        x, y, w, h = box
        gx0 = max(0, int(x) // self.cell_px)
        gy0 = max(0, int(y) // self.cell_px)
        gx1 = min(self.gw, int(x + w) // self.cell_px + 1)
        gy1 = min(self.gh, int(y + h) // self.cell_px + 1)
        if gx1 <= gx0 or gy1 <= gy0:
            return 0
        sl = (slice(gy0, gy1), slice(gx0, gx1))
        if self._last_cells is None:
            return 0
        structure, tone = self._last_cells
        self.struct_mean[sl] = structure[sl]
        self.tone_mean[sl] = tone[sl]
        # Variance is left alone: what it is normal for this cell to VARY by
        # has not changed just because what it looks like has.
        self.persist[sl] = 0
        self._z[sl] = 0.0
        self.ref_n[sl] = np.maximum(self.ref_n[sl], MIN_LEARNED_FRAMES)
        return int((gy1 - gy0) * (gx1 - gx0))

    def z_map(self):
        """The z-scores at frame resolution, for drawing."""
        return cv2.resize(self._z, (self.shape[1], self.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    def reset(self):
        self.__init__(shape=self.shape, cell_px=self.cell_px, fps=self.fps)
