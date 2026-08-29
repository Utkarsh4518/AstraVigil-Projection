"""Learning the thermal->optical homography from the scene itself.

scripts/calibrate_homography.py is the accurate way to do this: a person
clicks the same physical point in both views, at watch range, with the frame
frozen. It is also a job somebody has to stand at the rig and do, and until
they have, the optical pane, the overlay and the whole cross-cue path are
switched off - `self.H is not None` gates all three.

This closes that gap without replacing the manual tool. The scene provides
correspondences for free: a warm mover in the thermal frame and a mover in the
optical frame are, often enough, the same object, and a pair of centroids is a
calibration point.

WHY NOT WAIT FOR FRAMES WITH EXACTLY ONE OBJECT IN EACH VIEW

That was the first design, and the simulator killed it: 881 frames out of 900
had more than one candidate on one side or the other. Optical motion
detection on a real scene is not tidy, and a rule that only fires on a tidy
frame does not fire.

So ambiguity is passed to RANSAC instead of avoided. Every thermal centroid is
paired with every optical centroid in the frame, most of those pairs are
wrong, and that is precisely the input RANSAC is built for: the correct pairs
all agree with one homography, and the wrong ones agree with nothing. What
matters is keeping the fraction of correct pairs high enough to find - hence
the caps on how many candidates a single frame may contribute, and the raised
iteration count, since the inlier ratio here is far below what findHomography
assumes by default.

WHAT THE ACCEPTANCE TESTS ARE ACTUALLY FOR

A homography can be fitted to almost anything. These decide whether to believe
it, and they are all measured on the INLIER set, because the outliers are
supposed to be wrong:

  spread        Four points in a line still produce a matrix - one that maps
                those four beautifully and sends the corners to nonsense. Low
                error on clustered input is the most convincing wrong answer
                this system can produce.

  distinct      Twelve inliers from two frames is one object seen twice, not
  frames        twelve observations. Consensus has to be built over time or it
                is not consensus.

  reprojection  Looser than the manual tool's 5 px, because a blob centroid is
                noisier than a person's clicking - and still tight enough that
                a mistaken pairing cannot pass.

This never overwrites a calibration that already exists. A manual fit made at
watch range beats an automatic one made from whatever happened to walk past,
and silently replacing it would be helpfulness nobody asked for.
"""
import numpy as np

from . import homography

# Per-frame caps on candidate pairs. A frame with five movers on each side
# would contribute twenty-five pairs of which at most five are right, and
# drowning the inlier ratio to chase a busy frame is a bad trade - there will
# be a quieter one along shortly.
MAX_THERMAL_PER_FRAME = 2
MAX_OPTICAL_PER_FRAME = 3

# Inliers required, and how many separate frames they must come from.
MIN_INLIERS = 12
MIN_DISTINCT_FRAMES = 8

# Candidate pairs held. Enough for RANSAC to find a minority consensus,
# bounded so a long run does not grow without limit.
MAX_CANDIDATES = 1200

# The primary thermal blob must move this far, in thermal pixels, before the
# frame is worth sampling. At 25 Hz a walking person barely moves between
# frames, and without this the buffer fills with the same point many times -
# a fit exact at one spot and meaningless everywhere else.
MIN_STEP_PX = 4.0

# Fractions of the thermal frame the inlier cloud must cover: its bounding box
# on each axis, and the narrower axis of its principal spread. The second is
# what rejects a straight line, which has an excellent bounding box.
MIN_BBOX_FRAC = 0.25
MIN_MINOR_FRAC = 0.05

# Mean reprojection error over the inliers, in optical pixels.
ACCEPT_PX = 8.0

# RANSAC iterations. The default 2000 assumes a healthy inlier ratio; here it
# can be one in six, where four-point samples land all-inlier about once in
# 1300 tries. This is cheap - it runs on at most a few hundred points, only
# while uncalibrated, and only on frames that added something.
RANSAC_ITERS = 40000
RANSAC_PX = 3.0


class AutoCalibrator:
    """Accumulates candidate centroid pairs and fits when they justify one."""

    def __init__(self, min_inliers=MIN_INLIERS, accept_px=ACCEPT_PX,
                 min_step_px=MIN_STEP_PX, max_candidates=MAX_CANDIDATES,
                 min_distinct_frames=MIN_DISTINCT_FRAMES):
        self.min_inliers = int(min_inliers)
        self.accept_px = float(accept_px)
        self.min_step_px = float(min_step_px)
        self.max_candidates = int(max_candidates)
        self.min_distinct_frames = int(min_distinct_frames)

        self.thermal_pts = []
        self.optical_pts = []
        self.frame_of = []           # which frame each candidate came from
        self._last_primary = None
        self._frame = 0

        self.H = None
        self.error_px = None
        self.inliers = 0
        self.frames_seen = 0
        self.frames_sampled = 0
        self.rejected_fits = 0
        self.reason = "waiting for movement in both cameras"

    # ------------------------------------------------------------- collect
    @staticmethod
    def _primary(dets):
        """The detection most likely to be the real object: the biggest."""
        def area(d):
            box = getattr(d, "box", None)
            return box[2] * box[3] if box else 0
        return max(dets, key=area)

    def observe(self, thermal_dets, optical_dets, thermal_shape):
        """Feed one frame. Returns a homography the moment one is trusted."""
        self.frames_seen += 1
        if self.H is not None:
            return self.H
        if not thermal_dets or not optical_dets:
            return None

        # Busy frames are skipped rather than sampled thinly: they are the
        # ones whose pairs are most likely to be wrong.
        if (len(thermal_dets) > MAX_THERMAL_PER_FRAME
                or len(optical_dets) > MAX_OPTICAL_PER_FRAME):
            return None

        primary = self._primary(thermal_dets).centroid
        if self._last_primary is not None:
            dx = primary[0] - self._last_primary[0]
            dy = primary[1] - self._last_primary[1]
            if np.hypot(dx, dy) < self.min_step_px:
                return None
        self._last_primary = primary

        self._frame += 1
        self.frames_sampled += 1
        for t in thermal_dets:
            for o in optical_dets:
                self.thermal_pts.append(tuple(float(v) for v in t.centroid))
                self.optical_pts.append(tuple(float(v) for v in o.centroid))
                self.frame_of.append(self._frame)

        excess = len(self.thermal_pts) - self.max_candidates
        if excess > 0:
            del self.thermal_pts[:excess]
            del self.optical_pts[:excess]
            del self.frame_of[:excess]

        if len(self.thermal_pts) < max(self.min_inliers, 8):
            self.reason = (f"{self.frames_sampled} usable frames, "
                           f"{len(self.thermal_pts)} candidate pairs")
            return None

        return self._try_fit(thermal_shape)

    # ----------------------------------------------------------------- fit
    def _try_fit(self, thermal_shape):
        try:
            H, mask = homography.compute(
                self.thermal_pts, self.optical_pts,
                ransac_px=RANSAC_PX, max_iters=RANSAC_ITERS)
        except Exception as exc:
            self.rejected_fits += 1
            self.reason = f"fit failed ({exc})"
            return None

        keep = np.asarray(mask, bool).ravel()
        n = int(keep.sum())
        if n < self.min_inliers:
            self.reason = (f"{len(self.thermal_pts)} candidate pairs, only "
                           f"{n} agree - need {self.min_inliers}")
            return None

        t_in = [p for p, k in zip(self.thermal_pts, keep) if k]
        o_in = [p for p, k in zip(self.optical_pts, keep) if k]
        frames = {f for f, k in zip(self.frame_of, keep) if k}

        if len(frames) < self.min_distinct_frames:
            self.reason = (f"{n} agreeing pairs but from only {len(frames)} "
                           f"frames - one object seen repeatedly is not "
                           f"consensus")
            return None

        spread_ok, note = self.spread(t_in, thermal_shape)
        if not spread_ok:
            self.reason = note
            return None

        err = float(homography.reprojection_error(H, t_in, o_in).mean())
        if err > self.accept_px:
            self.rejected_fits += 1
            self.reason = (f"{n} agreeing pairs, but reprojection is "
                           f"{err:.1f} px - still watching")
            return None

        self.H = H
        self.error_px = err
        self.inliers = n
        self.reason = (f"calibrated from {n} agreeing pairs across "
                       f"{len(frames)} frames, reprojection {err:.1f} px")
        return H

    def spread(self, pts, thermal_shape):
        """Is the inlier cloud wide enough to constrain a homography?

        Two independent tests, because they fail differently: the bounding box
        catches a cluster, the minor axis catches a line.
        """
        h, w = thermal_shape[:2]
        pts = np.asarray(pts, np.float64)
        if len(pts) < 4:
            return False, "not enough agreeing points"

        bw = (pts[:, 0].max() - pts[:, 0].min()) / max(w, 1)
        bh = (pts[:, 1].max() - pts[:, 1].min()) / max(h, 1)
        if bw < MIN_BBOX_FRAC or bh < MIN_BBOX_FRAC:
            return False, (f"agreeing points cover {bw * 100:.0f}% x "
                           f"{bh * 100:.0f}% of frame - need "
                           f"{MIN_BBOX_FRAC * 100:.0f}% each way, move "
                           f"something through more of the view")

        centred = pts - pts.mean(axis=0)
        eig = np.linalg.eigvalsh(np.cov(centred.T))
        minor = float(np.sqrt(max(eig.min(), 0.0))) / max(min(h, w), 1)
        if minor < MIN_MINOR_FRAC:
            return False, ("agreeing points lie along a line - a homography "
                           "from those is exact on the line and wrong "
                           "everywhere else")

        return True, "spread is sufficient"

    # -------------------------------------------------------------- report
    def status(self):
        return {
            "calibrated": self.H is not None,
            "candidates": len(self.thermal_pts),
            "inliers": self.inliers,
            "needed": self.min_inliers,
            "error_px": (round(self.error_px, 2)
                         if self.error_px is not None else None),
            "frames_seen": self.frames_seen,
            "frames_sampled": self.frames_sampled,
            "rejected_fits": self.rejected_fits,
            "reason": self.reason,
        }

    def reset(self):
        self.__init__(min_inliers=self.min_inliers, accept_px=self.accept_px,
                      min_step_px=self.min_step_px,
                      max_candidates=self.max_candidates,
                      min_distinct_frames=self.min_distinct_frames)
