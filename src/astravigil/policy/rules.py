"""Site policy: what is ALLOWED here, as opposed to what is normal here.

These are different questions and the system needs both.

  NORMAL   is descriptive and learned. site_intelligence/baseline.py watches
           the site and records what usually happens. It is measured, so it
           is honest about this site specifically.

  ALLOWED  is prescriptive and authored. A person states what should happen.
           Nothing can learn this, because it is a decision, not a fact.

They diverge in exactly the case that matters. A learned baseline cannot tell
you an object should not be there if the object was there while it learned -
scripts/learn_site.py says as much in its own help text, and that is the
obvious way to blind this system deliberately. A policy is immune to that: it
was written by someone who knows what the site is for, before the sensor ever
saw it.

They also diverge the other way, which is where the false-positive relief
comes from. A ground vehicle in the service lane is thermally novel every
single time, and alarming on it every time is how an operator learns to ignore
the system. Policy is what lets you say "that is expected there" WITHOUT
teaching the baseline to ignore that patch of ground.

Scoped permission, never blanket whitelist
------------------------------------------
A permission is a claim about a class in a place at a time behaving a certain
way - not about an object. A vehicle permitted in the service lane during
daylight that sits motionless for an hour at 03:00 has left the terms of its
permission and is judged on the site evidence again. That follows the safety
principle in the project doc: known object plus abnormal behaviour must stay
flaggable. Whitelisting objects is what makes a perimeter system safe to walk
past.

Evaluation is pure arithmetic and set membership. No model runs here. An LLM
may have WRITTEN this file (see compile.py) but nothing learned is in the loop
at runtime, and the compiled rules are reviewable before they go live.
"""
import json
import os
import time

ANY = "*"

# A detection that matches no rule contributes no policy risk. Default-deny
# would alarm on everything unlisted, which is the "wall of separate alarms"
# the brief explicitly asks us not to build; default-allow would silently bless
# the unlisted. Neutral leaves the statistical baseline to do its job and lets
# policy speak only where someone actually stated something.
NEUTRAL_RISK = 0.0


class Zone:
    """A named region, as a polygon in thermal pixel coordinates.

    Thermal is the reference frame because that is where detection happens and
    where boxes are measured; the homography maps it to optical when needed.
    """

    __slots__ = ("name", "polygon", "description")

    def __init__(self, name, polygon=None, description=""):
        self.name = name
        self.polygon = [(float(x), float(y)) for x, y in (polygon or [])]
        self.description = description

    @property
    def everywhere(self):
        return self.name == ANY or not self.polygon

    def contains(self, point):
        """Even-odd ray casting. Small polygons, called a few times a frame."""
        if self.everywhere:
            return True
        x, y = point
        inside = False
        n = len(self.polygon)
        for i in range(n):
            x0, y0 = self.polygon[i]
            x1, y1 = self.polygon[(i + 1) % n]
            if (y0 > y) != (y1 > y):
                t = (y - y0) / (y1 - y0) if y1 != y0 else 0.0
                if x < x0 + t * (x1 - x0):
                    inside = not inside
        return inside

    def as_dict(self):
        return {"name": self.name, "polygon": [list(p) for p in self.polygon],
                "description": self.description}


class Rule:
    """One authored statement about what may happen where.

    verdict:
      permitted   - expected here; suppresses site novelty while in scope
      prohibited  - must not be here at all
      restricted  - allowed only within the stated limits; exceeding them is
                    a violation (this is how "no loitering" is expressed)
    """

    __slots__ = ("id", "zone", "classes", "verdict", "severity", "hours",
                 "days", "max_dwell_s", "min_speed_px", "reason")

    def __init__(self, id, zone=ANY, classes=None, verdict="prohibited",
                 severity=0.8, hours=None, days=None, max_dwell_s=None,
                 min_speed_px=None, reason=""):
        self.id = id
        self.zone = zone
        self.classes = [c.lower() for c in (classes or [ANY])]
        self.verdict = verdict
        self.severity = float(severity)
        self.hours = tuple(hours) if hours else None      # (start, end) local
        self.days = list(days) if days else None          # 0=Mon .. 6=Sun
        self.max_dwell_s = max_dwell_s
        self.min_speed_px = min_speed_px
        self.reason = reason

    def matches_class(self, label):
        return ANY in self.classes or (label or "unknown").lower() in self.classes

    def in_time_window(self, when):
        """Whether the rule's schedule covers this moment.

        Windows that wrap midnight (22 to 06) are the normal case for a site
        that is quiet overnight, so they have to work.
        """
        if self.hours is None and self.days is None:
            return True
        lt = time.localtime(when)
        if self.days is not None and lt.tm_wday not in self.days:
            return False
        if self.hours is None:
            return True
        start, end = self.hours
        h = lt.tm_hour + lt.tm_min / 60.0
        if start <= end:
            return start <= h < end
        return h >= start or h < end

    def as_dict(self):
        d = {"id": self.id, "zone": self.zone, "classes": self.classes,
             "verdict": self.verdict, "severity": self.severity,
             "reason": self.reason}
        for k in ("hours", "days", "max_dwell_s", "min_speed_px"):
            v = getattr(self, k)
            if v is not None:
                d[k] = list(v) if isinstance(v, tuple) else v
        return d


class Judgement:
    """What policy has to say about one object, this frame."""

    __slots__ = ("risk", "verdict", "reasons", "rule_ids", "zone",
                 "suppresses_novelty")

    def __init__(self, risk=NEUTRAL_RISK, verdict="unstated", reasons=None,
                 rule_ids=None, zone=None, suppresses_novelty=False):
        self.risk = risk
        self.verdict = verdict          # permitted | prohibited | unstated
        self.reasons = reasons or []
        self.rule_ids = rule_ids or []
        self.zone = zone
        self.suppresses_novelty = suppresses_novelty

    def as_dict(self):
        return {"risk": round(float(self.risk), 3), "verdict": self.verdict,
                "reasons": self.reasons, "rules": self.rule_ids,
                "zone": self.zone, "suppresses_novelty": self.suppresses_novelty}


class SitePolicy:
    def __init__(self, site="", zones=None, rules=None, source_text=""):
        self.site = site
        self.zones = {z.name: z for z in (zones or [])}
        self.zones.setdefault(ANY, Zone(ANY))
        self.rules = list(rules or [])
        # The original prose, kept so an operator can see what the compiled
        # rules were meant to say and check the compiler did not misread it.
        self.source_text = source_text

    # ------------------------------------------------------------ evaluate
    def zone_of(self, point):
        """Innermost named zone containing the point, or None.

        Named zones win over the implicit everywhere-zone so that a rule about
        the apron is not shadowed by a site-wide one.
        """
        for name, z in self.zones.items():
            if name != ANY and z.contains(point):
                return name
        return None

    def judge(self, label, centroid, dwell_s=0.0, speed_px=0.0, when=None):
        """Deterministic. Same inputs, same output, no model, no network."""
        when = time.time() if when is None else when
        zone_name = self.zone_of(centroid)

        risk = NEUTRAL_RISK
        verdict = "unstated"
        reasons, ids = [], []
        suppress = False

        for rule in self.rules:
            if not rule.matches_class(label):
                continue
            if rule.zone != ANY and rule.zone != zone_name:
                continue
            if rule.zone == ANY and not self.zones[ANY].contains(centroid):
                continue

            in_window = rule.in_time_window(when)
            where = f" in {zone_name}" if zone_name else ""

            if rule.verdict == "prohibited":
                if in_window:
                    risk = max(risk, rule.severity)
                    verdict = "prohibited"
                    reasons.append(rule.reason or
                                   f"{label} not permitted{where}")
                    ids.append(rule.id)

            elif rule.verdict == "permitted":
                if in_window and self._within_limits(rule, dwell_s, speed_px):
                    # In scope: this is expected here, so stop the baseline
                    # shouting about it. Deliberately does NOT drive risk
                    # negative - permission removes noise, it never argues
                    # against other evidence.
                    verdict = "permitted" if verdict == "unstated" else verdict
                    suppress = True
                    ids.append(rule.id)
                    reasons.append(rule.reason or
                                   f"{label} expected{where}")
                else:
                    # Out of scope. The permission simply does not apply, and
                    # saying why is the useful part for an operator.
                    reasons.append(
                        self._out_of_scope_reason(rule, label, where,
                                                  dwell_s, speed_px, in_window))
                    ids.append(rule.id)
                    risk = max(risk, rule.severity * 0.5)
                    verdict = "prohibited" if verdict != "prohibited" else verdict

            elif rule.verdict == "restricted":
                if in_window and not self._within_limits(rule, dwell_s, speed_px):
                    risk = max(risk, rule.severity)
                    verdict = "prohibited"
                    reasons.append(
                        self._out_of_scope_reason(rule, label, where,
                                                  dwell_s, speed_px, True))
                    ids.append(rule.id)

        # A permission and a prohibition can both fire. Prohibition wins, and
        # the permission must not go on suppressing the baseline underneath it.
        if verdict == "prohibited":
            suppress = False

        return Judgement(risk=risk, verdict=verdict, reasons=reasons,
                         rule_ids=ids, zone=zone_name,
                         suppresses_novelty=suppress)

    @staticmethod
    def _within_limits(rule, dwell_s, speed_px):
        if rule.max_dwell_s is not None and dwell_s > rule.max_dwell_s:
            return False
        if rule.min_speed_px is not None and speed_px < rule.min_speed_px:
            return False
        return True

    @staticmethod
    def _out_of_scope_reason(rule, label, where, dwell_s, speed_px, in_window):
        if not in_window:
            if rule.hours:
                return (f"{label}{where} outside permitted hours "
                        f"{rule.hours[0]:02.0f}:00-{rule.hours[1]:02.0f}:00")
            return f"{label}{where} outside permitted schedule"
        if rule.max_dwell_s is not None and dwell_s > rule.max_dwell_s:
            return (f"{label}{where} stationary {dwell_s:.0f} s, over the "
                    f"{rule.max_dwell_s:.0f} s limit")
        if rule.min_speed_px is not None and speed_px < rule.min_speed_px:
            return f"{label}{where} not moving as expected"
        return f"{label}{where} outside what is permitted"

    # ---------------------------------------------------------------- io
    def as_dict(self):
        return {"site": self.site,
                "zones": [z.as_dict() for z in self.zones.values()
                          if z.name != ANY],
                "rules": [r.as_dict() for r in self.rules],
                "source_text": self.source_text}

    def save(self, path):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.as_dict(), fh, indent=2)
        return path

    @classmethod
    def from_dict(cls, d):
        return cls(site=d.get("site", ""),
                   zones=[Zone(**z) for z in d.get("zones", [])],
                   rules=[Rule(**r) for r in d.get("rules", [])],
                   source_text=d.get("source_text", ""))

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def describe(self):
        """Plain-text rendering, for reviewing what will actually be enforced."""
        out = [f"site: {self.site or '(unnamed)'}",
               f"zones: {', '.join(n for n in self.zones if n != ANY) or 'none'}",
               f"rules: {len(self.rules)}"]
        for r in self.rules:
            bits = [f"  [{r.id}] {r.verdict:<10}",
                    f"{'/'.join(r.classes)} in {r.zone}"]
            if r.hours:
                bits.append(f"hours {r.hours[0]}-{r.hours[1]}")
            if r.max_dwell_s is not None:
                bits.append(f"max dwell {r.max_dwell_s}s")
            if r.severity:
                bits.append(f"severity {r.severity}")
            out.append(" ".join(bits))
            if r.reason:
                out.append(f"       \"{r.reason}\"")
        return "\n".join(out)


def validate(policy):
    """Problems a human should see before this governs anything.

    Returns a list of warnings. Deliberately warnings and not exceptions: a
    partly-wrong policy that still enforces its good rules is better than no
    policy, and refusing to load leaves the site unprotected over a typo.
    """
    problems = []
    names = set(policy.zones)
    for r in policy.rules:
        if r.zone not in names:
            problems.append(f"rule '{r.id}' references unknown zone "
                            f"'{r.zone}' - it will never match")
        if r.verdict not in ("permitted", "prohibited", "restricted"):
            problems.append(f"rule '{r.id}' has unknown verdict "
                            f"'{r.verdict}' - it will be ignored")
        if not 0.0 <= r.severity <= 1.0:
            problems.append(f"rule '{r.id}' severity {r.severity} outside 0-1")
        if r.verdict == "restricted" and r.max_dwell_s is None \
                and r.min_speed_px is None:
            problems.append(f"rule '{r.id}' is 'restricted' but states no "
                            f"limit, so nothing can exceed it")
    for z in policy.zones.values():
        if not z.everywhere and len(z.polygon) < 3:
            problems.append(f"zone '{z.name}' has fewer than 3 points")
    return problems
