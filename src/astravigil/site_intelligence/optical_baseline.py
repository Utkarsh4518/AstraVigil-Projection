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

        self._z = np.zeros(g, np.float32)
        self.frames = 0
        self.created = time.time()

    # ------------------------------------------------------------ maturity
    @property
    def maturity(self):
        return float(np.clip(self.ref_n / MIN_LEARNED_FRAMES, 0, 1).mean())

    @property
    def learning(self):
        return self.maturity < 1.0

    def stats(self):
        return {
            "frames": int(self.frames),
            "maturity": round(self.maturity, 3),
            "learning": bool(self.learning),
            "anomalous_cells": int((self._z > Z_ANOMALOUS).sum()),
            "settled_cells": int((self.persist >= self.persist_frames).sum()),
            "cells": int(self.gh * self.gw),
        }

    # ------------------------------------------------------------- observe
    def _cells(self, bgr):
        """One frame to the two per-cell statistics."""
        grey = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)

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

        if learn:
            # Exponential update with a floor, so a cell that has been quiet
            # for hours still adapts to a slow change. Cells currently off
            # baseline are updated far more slowly - otherwise an object that
            # sits still is absorbed into the definition of normal, which is
            # precisely the failure this model exists to avoid.
            alpha = np.maximum(1.0 / np.maximum(self.ref_n + 1.0, 1.0),
                               self.alpha_floor)
            alpha = np.where(hot, alpha * 0.05, alpha).astype(np.float32)

            d_s = structure - self.struct_mean
            self.struct_mean += alpha * d_s
            self.struct_var += alpha * (d_s * d_s - self.struct_var)
            d_t = tone - self.tone_mean
            self.tone_mean += alpha * d_t
            self.tone_var += alpha * (d_t * d_t - self.tone_var)
            self.ref_n += 1.0

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

    def z_map(self):
        """The z-scores at frame resolution, for drawing."""
        return cv2.resize(self._z, (self.shape[1], self.shape[0]),
                          interpolation=cv2.INTER_NEAREST)

    def reset(self):
        self.__init__(shape=self.shape, cell_px=self.cell_px, fps=self.fps)
