"""One alert per object, for as long as that object is there.

The brief's actual complaint about existing perimeter systems is not that they
miss things - it is that they produce "a wall of separate alarms". A detector
that fires 25 times a second for 90 seconds has technically detected the
drone and has practically told the operator nothing.

So an alert here is a *stateful object*, not an event. It opens once, updates
in place while the thing is still there, and closes when it leaves. The
operator sees one row that says "drone, on the apron, 4 min 12 s" - which is
the sentence the airport in the brief needed and did not get.

Three mechanisms keep it to one row:

  DEBOUNCE   the threat has to hold above the line for several consecutive
             frames before an alert opens, so a noise spike is not an alarm.
  IDENTITY   alerts are keyed on the assessment key - a track id, or a
             quantised grid location for a settled object - so the same
             physical thing maps to the same alert across frames.
  GRACE      a track that blinks out for a moment does not close and reopen
             the alert as a second incident.

Everything is also appended to a JSONL log, because "when did it arrive" is a
question asked after the fact and the dashboard's memory dies with the process.
"""
import json
import os
import time

OPEN_FRAMES = 8         # consecutive frames above the line before opening
CLOSE_GRACE_S = 5.0     # gone this long before the alert is closed
KEEP_CLOSED = 20        # closed alerts retained for the dashboard


class Alert:
    __slots__ = ("id", "key", "kind", "opened_at", "updated_at", "closed_at",
                 "label", "threat", "peak_threat", "level", "reasons", "box",
                 "optical_box", "dwell_s", "track_id", "frames", "acked",
                 "sensors")

    def __init__(self, alert_id, assessment, now):
        self.id = alert_id
        self.key = assessment.key
        self.kind = assessment.kind
        self.opened_at = now
        self.updated_at = now
        self.closed_at = None
        self.acked = False
        self.frames = 1
        self.peak_threat = 0.0
        self.update(assessment, now)

    def update(self, a, now):
        self.updated_at = now
        self.label = a.label
        self.threat = a.threat
        self.level = a.level
        self.reasons = list(a.reasons)
        self.box = [int(v) for v in a.box]
        self.dwell_s = a.dwell_s
        self.track_id = a.track_id
        # Which cameras backed this call. "thermal+optical" means the two
        # sensors were cross-checked against each other, which is a different
        # quality of evidence from either one alone.
        self.sensors = getattr(a, "sensors", "thermal")
        self.frames += 1
        self.peak_threat = max(self.peak_threat, a.threat)

    @property
    def active(self):
        return self.closed_at is None

    def duration_s(self, now=None):
        end = self.closed_at or (time.time() if now is None else now)
        return end - self.opened_at

    def as_dict(self, now=None):
        return {
            "id": self.id, "key": self.key, "kind": self.kind,
            "label": self.label, "level": self.level,
            "threat": round(self.threat, 3),
            "peak_threat": round(self.peak_threat, 3),
            "opened_at": self.opened_at,
            "opened_hms": time.strftime("%H:%M:%S", time.localtime(self.opened_at)),
            "duration_s": round(self.duration_s(now), 1),
            "dwell_s": round(self.dwell_s, 1),
            "sensors": self.sensors,
            "track_id": self.track_id,
            "box": list(self.box),
            "reasons": list(self.reasons),
            "active": self.active,
            "acked": self.acked,
            "frames": self.frames,
        }


class AlertManager:
    def __init__(self, log_path="data/alerts/alerts.jsonl",
                 open_frames=OPEN_FRAMES, close_grace_s=CLOSE_GRACE_S,
                 threshold=0.75):
        self.log_path = log_path
        self.open_frames = open_frames
        self.close_grace_s = close_grace_s
        self.threshold = threshold
        self.alerts = {}            # key -> Alert (active only)
        self.closed = []
        self._streak = {}           # key -> consecutive frames above the line
        self._next_id = 1

    def update(self, assessments, now=None):
        """Feed this frame's verdicts; returns the currently active alerts."""
        now = time.time() if now is None else now
        seen = set()

        for a in assessments:
            over = a.threat >= self.threshold
            n = self._streak.get(a.key, 0)
            self._streak[a.key] = n + 1 if over else 0

            existing = self.alerts.get(a.key)
            if existing is not None:
                seen.add(a.key)
                existing.update(a, now)
            elif over and self._streak[a.key] >= self.open_frames:
                seen.add(a.key)
                alert = Alert(self._next_id, a, now)
                self._next_id += 1
                self.alerts[a.key] = alert
                self._log("open", alert)

        for key, alert in list(self.alerts.items()):
            if key in seen:
                continue
            if now - alert.updated_at >= self.close_grace_s:
                alert.closed_at = now
                self._log("close", alert)
                self.closed.append(alert)
                del self.alerts[key]
                self._streak.pop(key, None)

        del self.closed[:-KEEP_CLOSED]
        return list(self.alerts.values())

    def clear(self, key):
        """Drop any alert and debounce state for one key.

        Used when an operator accepts an object as normal: the alert should
        go away immediately rather than linger for the close grace period,
        and its debounce counter must reset or it reopens on the next frame.
        """
        self.alerts.pop(key, None)
        self._streak.pop(key, None)

    def ack(self, alert_id):
        for a in list(self.alerts.values()) + self.closed:
            if a.id == int(alert_id):
                a.acked = True
                return a
        return None

    def active(self, now=None):
        out = [a.as_dict(now) for a in self.alerts.values()]
        out.sort(key=lambda d: d["threat"], reverse=True)
        return out

    def history(self, now=None):
        return [a.as_dict(now) for a in reversed(self.closed)]

    def _log(self, event, alert):
        if not self.log_path:
            return
        try:
            os.makedirs(os.path.dirname(self.log_path) or ".", exist_ok=True)
            with open(self.log_path, "a", encoding="utf-8") as fh:
                rec = alert.as_dict()
                rec["event"] = event
                fh.write(json.dumps(rec) + "\n")
        except OSError as exc:
            # A read-only or full disk must not take the detector down with
            # it - the live picture matters more than the audit trail.
            print(f"alert log write failed: {exc}")
