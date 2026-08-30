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
# bounded so a long run does not grow without limit - and kept modest because
# the cost of a fit grows with it.
MAX_CANDIDATES = 600

# Sampled frames between fit attempts. A fit is expensive and this runs inside
# the capture loop, so trying one every time a frame contributed a pair is how
# the dashboard ended up at 3.6 FPS with 250 ms in detect. There is nothing to
# gain from it either: a fit that failed on 40 candidates is not going to
# succeed on 46.
FIT_EVERY_SAMPLES = 12

# How far a detection must be from anything in the previous frame to count as
# having moved. Everything else is treated as scenery.
#
# Only movers are paired, and that is the difference between working and not.
# A room contains plenty of things warmer than the wall behind them - a desk
# edge, a laptop, a radiator - and the detector deliberately freezes anything
# it has seen out of its background model, so they are detected for ever
# without going anywhere. They cannot be calibration points: they never move,
# so they carry no information about the mapping, and pairing them adds
# outliers and drags the whole point cloud onto whatever line they happen to
# lie along.
#
# The first version gated on the LARGEST blob instead, on the theory that the
# biggest thing is the real object. In an office the biggest thing is the desk
# edge, which never moves, so the gate rejected every frame and collection
# stopped dead after the first few.
# Scenery is recognised by staying put, not by moving slowly.
#
# The first version compared each detection with the previous frame and called
# anything that had shifted less than a few pixels static. That cannot work:
# measured on the simulator, real movers shift a median of 1.1 px per frame,
# which is the same order as the centroid jitter of something that is not
# moving at all. A threshold high enough to reject the jitter rejects the
# object too - it took usable frames from 900 down to 21.
#
# What actually separates a desk edge from a hand is duration. A cell holding
# a detection in most of the last few seconds is furniture; a cell holding one
# now and not a moment ago is something happening. So occupancy is counted
# over a window and the busy cells are excluded.
#
# The threshold has to sit close to 1.0, not near the middle. A desk edge is
# in its cell on essentially every frame; somebody waving deliberately in one
# part of the view is in a given cell well over half of them. At 0.55 the
# operator's own hand was being classified as furniture and discarded, so
# moving MORE made calibration collect LESS - the exact opposite of the
# instruction the pane was giving them.
SCENERY_WINDOW = 90          # frames of occupancy history
SCENERY_FRACTION = 0.85      # occupied at least this often -> scenery
SCENERY_WARMUP = 30          # frames before any of it is trusted

# Ceiling on correspondences contributed by things that have STOPPED.
#
# A moving object offers a new pair every time it moves; a parked one sits in
# the same place for as long as it is there and would offer the same pair on
# every frame forever. RANSAC decides by counting agreement, so a single
# WRONG stationary pair - a warm desk edge lined up with an unrelated bag -
# repeated a thousand times would out-vote every correct pair in the buffer
# and the fit would converge confidently on nonsense.
#
# So a given (thermal object, optical object) combination contributes exactly
# once, ever, and the total is capped. That bounds their influence to roughly
# what a few seconds of movement provides, which is the right weight for
# evidence that cannot be independently repeated.
MAX_SETTLED_PAIRS = 40
THERMAL_CELL_PX = 8
OPTICAL_CELL_PX = 24

# Nearer than this to the last points stored and a frame adds nothing worth
# keeping. Redundancy, not scenery: a genuinely slow mover is still a real
# observation, it just does not need recording forty times on its way across
# one cell. Without this a crawl fills the buffer with near-copies of itself
# and crowds out the spread that makes the fit trustworthy.
DEDUPE_PX = 3.0

# Fractions of the thermal frame the inlier cloud must cover: its bounding box
# on each axis, and the narrower axis of its principal spread. The second is
# what rejects a straight line, which has an excellent bounding box.
MIN_BBOX_FRAC = 0.25
MIN_MINOR_FRAC = 0.05

# Mean reprojection error over the inliers, in optical pixels.
ACCEPT_PX = 8.0

# RANSAC iterations, sized against the worst inlier ratio the per-frame caps
# allow. Two thermal by three optical is one correct pair in six, so a
# four-point sample is all-inlier about once in 1200 tries and ~6000 iterations
# gives 99% confidence of finding one. 8000 leaves headroom.
#
# It was 40000, which is not free: 607 ms per fit against 120 ms at this value,
# measured. Buying confidence far past the point of diminishing returns is
# what made the capture loop stall.
RANSAC_ITERS = 8000
RANSAC_PX = 3.0


class AutoCalibrator:
    """Accumulates candidate centroid pairs and fits when they justify one."""

    def __init__(self, min_inliers=MIN_INLIERS, accept_px=ACCEPT_PX,
                 max_candidates=MAX_CANDIDATES,
                 min_distinct_frames=MIN_DISTINCT_FRAMES):
        self.min_inliers = int(min_inliers)
        self.accept_px = float(accept_px)
        self.max_candidates = int(max_candidates)
        self.min_distinct_frames = int(min_distinct_frames)

        self.thermal_pts = []
        self.optical_pts = []
        self.frame_of = []           # which frame each candidate came from
        self._t_hist = []
        self._o_hist = []
        self._last_added = []
        self._prev_optical_key = None
        # (thermal key, optical key) combinations already contributed by the
        # stationary path, so none of them can be counted twice.
        self._settled_pairs = set()
        self._frame = 0
        # Primed, so the first frame that reaches the minimum fits at once
        # rather than waiting out a throttle interval it has not used. The
        # throttle is there to stop repeated retries, not to delay the first
        # answer.
        self._since_fit = FIT_EVERY_SAMPLES

        self.H = None
        self.error_px = None
        self.inliers = 0
        self.frames_seen = 0
        self.frames_sampled = 0
        self.rejected_fits = 0
        self.reason = "waiting for movement in both cameras"

    # ------------------------------------------------------------- collect
    @staticmethod
    def _cell(centroid, cell_px):
        return (int(centroid[0]) // cell_px, int(centroid[1]) // cell_px)

    def _not_scenery(self, dets, history, cell_px):
        """Drop detections sitting in cells that are nearly always occupied.

        History is a list of the recent frames' occupied cell sets. Until
        there are enough of them nothing is excluded, because a cell seen
        three times out of three is not yet evidence of furniture.
        """
        if len(history) < SCENERY_WARMUP:
            return list(dets)
        counts = {}
        for occupied in history:
            for c in occupied:
                counts[c] = counts.get(c, 0) + 1
        limit = SCENERY_FRACTION * len(history)
        return [d for d in dets
                if counts.get(self._cell(d.centroid, cell_px), 0) < limit]

    def observe_settled(self, thermal_statics, optical_regions,
                        thermal_shape):
        """Correspondences from objects that have stopped moving.

        The motion path needs somebody to walk about, and an operator who has
        been told to keep moving and is watching a counter refuse to climb has
        no other lever to pull. Two things that have arrived and settled - a
        thermal static anomaly and an optical settled patch - are a usable
        correspondence that costs nobody any walking.

        Deliberately a SUPPLEMENT and not a replacement. Stationary objects
        cluster wherever they happen to be, so a scene with two of them in one
        corner still fails the spread test and still needs movement across the
        view; what this removes is the case where somebody has moved enough,
        stopped, and the last few pairs never arrive.
        """
        if self.H is not None or not thermal_statics or not optical_regions:
            return None
        # Busy frames are skipped rather than sampled thinly, exactly as the
        # motion path does. Every static is paired with every region, so three
        # of each means nine pairs of which at most three can be right; the
        # inlier ratio RANSAC has to find falls off a cliff as this grows.
        if (len(thermal_statics) > MAX_THERMAL_PER_FRAME
                or len(optical_regions) > MAX_OPTICAL_PER_FRAME):
            return None

        added = 0
        for t in thermal_statics:
            for o in optical_regions:
                if len(self._settled_pairs) >= MAX_SETTLED_PAIRS:
                    break
                pair = (t.key, o.key)
                if pair in self._settled_pairs:
                    continue
                self._settled_pairs.add(pair)
                self.thermal_pts.append(
                    tuple(float(v) for v in t.centroid))
                self.optical_pts.append(
                    tuple(float(v) for v in o.centroid))
                added += 1
        if not added:
            return None

        # One frame index for the whole call, not one per pair. Counting each
        # pair as its own frame would let a single instant satisfy the
        # distinct-frames test, which exists precisely to stop that.
        self._frame += 1
        self.frame_of.extend([self._frame] * added)
        self._trim()

        if len(self.thermal_pts) < max(self.min_inliers, 8):
            return None
        spread_ok, note = self.spread(self.thermal_pts, thermal_shape)
        if not spread_ok:
            self.reason = note
            return None
        self._since_fit = 0
        return self._try_fit(thermal_shape)

    def _trim(self):
        excess = len(self.thermal_pts) - self.max_candidates
        if excess > 0:
            del self.thermal_pts[:excess]
            del self.optical_pts[:excess]
            del self.frame_of[:excess]

    def observe(self, thermal_dets, optical_dets, thermal_shape):
        """Feed one frame. Returns a homography the moment one is trusted."""
        self.frames_seen += 1
        if self.H is not None:
            return self.H
        if not thermal_dets or not optical_dets:
            return None

        # The optical detector runs at a fraction of the frame rate and hands
        # back its previous answer in between. Those repeats carry nothing new
        # and would read as "nothing moved", so they are skipped outright.
        opt_key = tuple(tuple(o.centroid) for o in optical_dets)
        if opt_key == self._prev_optical_key:
            return None

        self._prev_optical_key = opt_key
        self._t_hist.append({self._cell(d.centroid, THERMAL_CELL_PX)
                             for d in thermal_dets})
        self._o_hist.append({self._cell(d.centroid, OPTICAL_CELL_PX)
                             for d in optical_dets})
        del self._t_hist[:-SCENERY_WINDOW]
        del self._o_hist[:-SCENERY_WINDOW]

        thermal_dets = self._not_scenery(thermal_dets, self._t_hist,
                                         THERMAL_CELL_PX)
        optical_dets = self._not_scenery(optical_dets, self._o_hist,
                                         OPTICAL_CELL_PX)
        if not thermal_dets or not optical_dets:
            # Only speak while there is nothing better to say. Once enough
            # candidates exist the last fit diagnostic is the useful message -
            # it tells the operator which way to move - and overwriting it
            # every quiet frame replaced guidance with a status nobody can
            # act on.
            if len(self.thermal_pts) < max(self.min_inliers, 8):
                self.reason = (f"{self.frames_sampled} usable frames - "
                               f"waiting for something to MOVE in both "
                               f"cameras (things that never move are read "
                               f"as scenery)")
            return None

        # Busy frames are skipped rather than sampled thinly: they are the
        # ones whose pairs are most likely to be wrong.
        if (len(thermal_dets) > MAX_THERMAL_PER_FRAME
                or len(optical_dets) > MAX_OPTICAL_PER_FRAME):
            return None

        if self._last_added:
            fresh = [d for d in thermal_dets
                     if min(np.hypot(d.centroid[0] - px, d.centroid[1] - py)
                            for px, py in self._last_added) >= DEDUPE_PX]
            if not fresh:
                return None
        self._last_added = [tuple(d.centroid) for d in thermal_dets]

        self._frame += 1
        self.frames_sampled += 1
        for t in thermal_dets:
            for o in optical_dets:
                self.thermal_pts.append(tuple(float(v) for v in t.centroid))
                self.optical_pts.append(tuple(float(v) for v in o.centroid))
                self.frame_of.append(self._frame)

        self._trim()

        if len(self.thermal_pts) < max(self.min_inliers, 8):
            self.reason = (f"{self.frames_sampled} usable frames, "
                           f"{len(self.thermal_pts)} candidate pairs")
            return None

        # Throttled, and only after a cheap look at whether a fit could
        # possibly pass. Both guards exist because this runs inside the
        # capture loop and a fit is the most expensive thing in it.
        self._since_fit += 1
        if self._since_fit < FIT_EVERY_SAMPLES:
            return None
        self._since_fit = 0

        spread_ok, note = self.spread(self.thermal_pts, thermal_shape)
        if not spread_ok:
            # No fit can pass a spread test its own input already fails, and
            # this costs microseconds where the fit costs a tenth of a second.
            self.reason = note
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
            short = "wider" if bw < bh else "higher and lower"
            return False, (f"covered {bw * 100:.0f}% across x {bh * 100:.0f}% "
                           f"down, need {MIN_BBOX_FRAC * 100:.0f}% each - "
                           f"move {short}")

        centred = pts - pts.mean(axis=0)
        eig = np.linalg.eigvalsh(np.cov(centred.T))
        minor = float(np.sqrt(max(eig.min(), 0.0))) / max(min(h, w), 1)
        if minor < MIN_MINOR_FRAC:
            return False, (f"points are {minor * 100:.1f}% off a straight "
                           f"line, need {MIN_MINOR_FRAC * 100:.0f}% - move "
                           f"through DIFFERENT HEIGHTS as well as across")

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
                      max_candidates=self.max_candidates,
                      min_distinct_frames=self.min_distinct_frames)
