"""How long has each track been sitting still, and where.

The tracker keeps ~1.6 s of position history, which is the right length for
the wingbeat feature and far too short for the question this answers. "It has
not moved for four minutes" is the single most useful thing you can say about
a small object on an apron, and it needs a memory measured in wall-clock
seconds rather than in frames.

Kept out of Track deliberately. The tracker's history exists to serve the
classifier's temporal features and is sized for that; bolting an unbounded
timer onto it would couple two things that want to change independently.
"""
import math
import time

SETTLE_RADIUS_PX = 3.0      # stay inside this of the anchor and you are still
MIN_HITS = 4                # ignore brand-new tracks; they have no position yet


class DwellRecord:
    __slots__ = ("track_id", "first_seen", "anchor", "settled_since",
                 "last_moved", "peak_dwell_s")

    def __init__(self, track_id, centroid, now):
        self.track_id = track_id
        self.first_seen = now
        self.anchor = centroid
        self.settled_since = None
        self.last_moved = now
        self.peak_dwell_s = 0.0

    def dwell_s(self, now):
        if self.settled_since is None:
            return 0.0
        return now - self.settled_since

    def age_s(self, now):
        return now - self.first_seen


class DwellMonitor:
    """Per-track settling state, keyed on track id."""

    def __init__(self, radius_px=SETTLE_RADIUS_PX):
        self.radius_px = radius_px
        self.records = {}

    def update(self, tracks, now=None):
        now = time.time() if now is None else now

        for tid, tr in tracks.items():
            if tr.hits < MIN_HITS or tr.misses > 0:
                continue
            rec = self.records.get(tid)
            if rec is None:
                rec = self.records[tid] = DwellRecord(tid, tr.centroid, now)
                continue
            if math.dist(tr.centroid, rec.anchor) > self.radius_px:
                # Broke out of the circle: re-anchor here and start again.
                # A slow drift therefore resets every few seconds rather than
                # accumulating dwell while creeping across the frame.
                rec.anchor = tr.centroid
                rec.settled_since = None
                rec.last_moved = now
            elif rec.settled_since is None:
                rec.settled_since = now
            rec.peak_dwell_s = max(rec.peak_dwell_s, rec.dwell_s(now))

        for tid in [t for t in self.records if t not in tracks]:
            del self.records[tid]
        return self.records

    def dwell_s(self, track_id, now=None):
        rec = self.records.get(track_id)
        if rec is None:
            return 0.0
        return rec.dwell_s(time.time() if now is None else now)

    def age_s(self, track_id, now=None):
        rec = self.records.get(track_id)
        if rec is None:
            return 0.0
        return rec.age_s(time.time() if now is None else now)
