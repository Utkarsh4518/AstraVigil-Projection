"""Nearest-neighbour tracker, enough to give blobs a stable identity.

Tracking exists here for one reason beyond tidy IDs: the feature that actually
separates a bird from a quadcopter is temporal. A bird flaps, so its silhouette
area oscillates several times a second; a quad is a rigid body and its area is
steady. You cannot see that in a single frame, so something has to hold a
history - that is this.

Greedy nearest-neighbour association is crude and will swap IDs when two
targets cross. That is an acceptable trade at this scale: a Kalman filter with
proper gating is the right answer for many targets, and overkill for the two
or three a site perimeter sees at once.
"""
import math
from collections import deque

MAX_MATCH_DIST_PX = 28.0    # thermal pixels between frames
MAX_MISSES = 12             # ~0.5 s at 25 fps before a track is dropped
HISTORY = 40                # ~1.6 s, long enough to see a wingbeat


class Track:
    def __init__(self, track_id, detection):
        self.id = track_id
        self.centroid = detection.centroid
        self.box = detection.box
        self.misses = 0
        self.hits = 1
        self.areas = deque([detection.area], maxlen=HISTORY)
        self.positions = deque([detection.centroid], maxlen=HISTORY)
        self.label = "unknown"
        self.confidence = 0.0

    def update(self, detection):
        self.centroid = detection.centroid
        self.box = detection.box
        self.areas.append(detection.area)
        self.positions.append(detection.centroid)
        self.misses = 0
        self.hits += 1

    @property
    def speed_px(self):
        """Mean speed over the recent history, thermal px per frame."""
        if len(self.positions) < 2:
            return 0.0
        p = list(self.positions)
        d = [math.dist(p[i], p[i - 1]) for i in range(1, len(p))]
        return sum(d) / len(d)

    @property
    def straightness(self):
        """Net displacement over path length: 1.0 is a straight line.

        Quads under command fly straight lines and smooth arcs; birds wander.
        """
        p = list(self.positions)
        if len(p) < 3:
            return 0.0
        path = sum(math.dist(p[i], p[i - 1]) for i in range(1, len(p)))
        if path < 1e-6:
            return 0.0
        return math.dist(p[0], p[-1]) / path

    @property
    def flap_score(self):
        """Normalised oscillation of blob area - the wingbeat signal.

        Coefficient of variation of area over the history. A flapping bird
        modulates its silhouette strongly; a rigid airframe does not. Returns
        0 until there is enough history to mean anything, so a brand new track
        is never called a bird on one frame.
        """
        if len(self.areas) < 8:
            return 0.0
        a = list(self.areas)
        mean = sum(a) / len(a)
        if mean <= 0:
            return 0.0
        var = sum((v - mean) ** 2 for v in a) / len(a)
        return math.sqrt(var) / mean


class Tracker:
    def __init__(self, max_dist=MAX_MATCH_DIST_PX, max_misses=MAX_MISSES):
        self.max_dist = max_dist
        self.max_misses = max_misses
        self.tracks = {}
        self._next_id = 1

    def update(self, detections):
        """Associate detections to tracks; returns the live tracks."""
        unmatched = list(detections)

        # Greedy: closest pair first, so an obvious match is not stolen by a
        # marginal one earlier in the list.
        pairs = []
        for tid, tr in self.tracks.items():
            for det in unmatched:
                d = math.dist(tr.centroid, det.centroid)
                if d <= self.max_dist:
                    pairs.append((d, tid, det))
        pairs.sort(key=lambda p: p[0])

        used_tracks, used_dets = set(), set()
        for _, tid, det in pairs:
            if tid in used_tracks or id(det) in used_dets:
                continue
            self.tracks[tid].update(det)
            det.track_id = tid
            used_tracks.add(tid)
            used_dets.add(id(det))

        for det in unmatched:
            if id(det) in used_dets:
                continue
            tid = self._next_id
            self._next_id += 1
            self.tracks[tid] = Track(tid, det)
            det.track_id = tid

        for tid, tr in list(self.tracks.items()):
            if tid not in used_tracks:
                tr.misses += 1
                if tr.misses > self.max_misses:
                    del self.tracks[tid]

        for det in detections:
            tr = self.tracks.get(det.track_id)
            if tr is not None:
                det.flap_score = tr.flap_score

        return self.tracks
