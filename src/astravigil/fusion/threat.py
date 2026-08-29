"""Fusing identity and site anomaly into one threat number.

Two independent questions get answered upstream and neither is allowed to
overrule the other:

  WHAT IS IT      the rule classifier: drone / bird / unknown, with a
                  confidence. Stable, hand-tuned, does not learn.
  DOES IT BELONG  the site baseline: position, appearance, feature and dwell
                  novelty. Learns continuously, knows nothing about classes.

They are combined with a noisy-OR rather than a weighted average, and that
choice is the whole design:

    threat = 1 - (1 - identity_risk) * (1 - site_risk)

An average lets a confident "bird" drag down a screaming site anomaly, and
lets a quiet site drag down a confident "drone". A noisy-OR says either
signal alone is sufficient cause, which is what the project's own safety note
demands - `known object + abnormal behaviour` and `unknown object + anomalous
behaviour` must both stay flaggable.

Note the deliberate asymmetry in identity risk. An unclassified mover carries
a small floor of risk rather than zero, because "we could not tell" is not
the same as "harmless" - and at 40 m, where a drone is four warm pixels, most
of them will be unclassified. A confidently-classified bird carries none.
"""
from .crosscue import fuse_identity

# How much of the site-novelty signal survives when policy says this activity
# is expected here. Not zero, deliberately: policy states that a CLASS belongs
# in a PLACE, not that any particular object is harmless. Leaving a quarter of
# the statistical signal in means a permitted vehicle behaving like nothing
# ever seen in that lane can still climb back to an alert on its own.
SITE_SUPPRESSION = 0.25

# Identity risk by label. Bird is zero on purpose: the site channel is what
# catches a bird behaving impossibly, and double-counting it here would just
# make every gull a suspect.
UNKNOWN_FLOOR = 0.15

# Hotspot above which an optical-cued contact reads as powered rather than
# merely warm. Mirrors HOT_CORE_C in crosscue.
HOT_CORE_HINT = 8.0

WATCH = 0.45
ALERT = 0.75

LEVELS = ("nominal", "watch", "alert")


class Assessment:
    """One object, one verdict, and the reasons behind it."""

    __slots__ = ("key", "kind", "track_id", "label", "confidence", "threat",
                 "level", "identity_risk", "site_risk", "policy", "novelty", "dwell_s",
                 "reasons", "box", "centroid", "optical", "thermal_check",
                 "sensors")

    def __init__(self, key, kind, threat, level, identity_risk, site_risk,
                 reasons, box, centroid, track_id=None, label="unknown",
                 confidence=0.0, novelty=None, dwell_s=0.0, optical=None,
                 thermal_check=None, sensors="thermal", policy=None):
        self.key = key
        self.kind = kind                # "track" | "static"
        self.track_id = track_id
        self.label = label
        self.confidence = confidence
        self.threat = threat
        self.level = level
        self.identity_risk = identity_risk
        self.policy = policy
        self.site_risk = site_risk
        self.novelty = novelty
        self.dwell_s = dwell_s
        self.reasons = reasons
        self.box = box
        self.centroid = centroid
        self.optical = optical              # optical's answer to a thermal cue
        self.thermal_check = thermal_check  # thermal's answer to an optical cue
        # Which sensors contributed: "thermal", "optical", or "thermal+optical".
        # An operator asking "why should I believe this" starts here.
        self.sensors = sensors

    def as_dict(self):
        d = {"key": self.key, "kind": self.kind, "track_id": self.track_id,
             "label": self.label, "confidence": round(self.confidence, 3),
             "threat": round(self.threat, 3), "level": self.level,
             "identity_risk": round(self.identity_risk, 3),
             "site_risk": round(self.site_risk, 3),
             "dwell_s": round(self.dwell_s, 1),
             "reasons": list(self.reasons),
             "sensors": self.sensors,
             "box": [int(v) for v in self.box]}
        if self.novelty is not None:
            d["novelty"] = self.novelty.as_dict()
        if self.policy is not None:
            d["policy"] = self.policy.as_dict()
        if self.optical is not None:
            d["optical"] = self.optical.as_dict()
        if self.thermal_check is not None:
            d["thermal_check"] = self.thermal_check.as_dict()
        return d


def _level(threat):
    if threat >= ALERT:
        return "alert"
    if threat >= WATCH:
        return "watch"
    return "nominal"


def identity_risk(label, confidence):
    if label == "drone":
        return float(confidence)
    if label == "bird":
        return 0.0
    return UNKNOWN_FLOOR


def assess_track(det, track, novelty, dwell_s=0.0, optical=None,
                 judgement=None):
    """Verdict on a tracked mover, with optical's shape check folded in.

    The identity half may now come from two sensors; the site half never
    does. They still meet at the same noisy-OR, so a cross-confirmed drone
    and a behaviourally impossible unknown both still reach the operator.

    `judgement` adds a third independent term: what a human said is ALLOWED
    here, from site policy. It joins the same noisy-OR because it is evidence
    of the same kind - sufficient on its own, never able to veto the others.
    A policy that could cancel a detection would be a way to switch the sensor
    off by editing a config file.

    The one thing policy may do downward is suppress SITE novelty, and only
    for activity explicitly stated as expected. A ground vehicle in its lane
    is thermally novel every time it passes; alarming every time is how an
    operator learns to ignore the system. Note this suppresses the statistical
    surprise, not the identity term - a drone in a lane where vehicles are
    permitted still carries its full identity risk.
    """
    label, confidence = det.label, det.confidence
    note = None
    if optical is not None:
        label, confidence, note = fuse_identity(label, confidence, optical)

    ident = identity_risk(label, confidence)
    site = novelty.overall if novelty is not None else 0.0
    policy_risk = 0.0

    if judgement is not None:
        policy_risk = judgement.risk
        if judgement.suppresses_novelty:
            site *= SITE_SUPPRESSION
    threat = 1.0 - (1.0 - ident) * (1.0 - site) * (1.0 - policy_risk)

    reasons = []
    if label == "drone" and confidence > 0.5:
        reasons.append(f"classified drone ({confidence:.2f})")
    elif label == "unknown":
        reasons.append("unclassified mover")
    if note:
        reasons.append(note)
    # Policy reasons lead: "no aircraft over the apron" is what an operator
    # needs first, ahead of the statistics that also happened to fire.
    if judgement is not None and judgement.verdict == "prohibited":
        reasons = judgement.reasons + reasons
    if novelty is not None:
        reasons.extend(novelty.reasons)
    if judgement is not None and judgement.verdict == "permitted":
        reasons.extend(judgement.reasons)

    sensors = "thermal+optical" if (optical is not None
                                    and optical.usable) else "thermal"
    return Assessment(
        key=f"track:{det.track_id}", kind="track", threat=threat,
        level=_level(threat), identity_risk=ident, site_risk=site,
        reasons=reasons, box=det.box, centroid=det.centroid,
        track_id=det.track_id, label=label, confidence=confidence,
        novelty=novelty, dwell_s=dwell_s, optical=optical, sensors=sensors,
        policy=judgement)


def assess_optical_only(odet, thermal_check, key, dwell_s=0.0):
    """Verdict on something only the optical camera found.

    This is the reverse cue, and it exists for the two cases thermal cannot
    cover: crossover at dawn and dusk, when contrast against the background
    briefly vanishes, and an airframe that has been parked long enough to sit
    at ambient. In both, optical is the only sensor still producing a signal,
    and a thermal-led pipeline with no reverse path simply never hears about
    it.

    A confirmed heat signature makes this a real object thermal's own
    threshold missed. No heat signature usually means clutter - a cloud edge,
    a shadow, a branch - so it is reported quietly rather than suppressed,
    because "usually" is not "always" and a cold-soaked drone lands here too.
    """
    warm = thermal_check is not None and thermal_check.warm
    verifiable = thermal_check is not None and thermal_check.found
    # Ramps over the first minute. A shadow crossing a wall is gone in
    # seconds; something still in the same place a minute later is an object.
    settled = min(1.0, dwell_s / 60.0)

    if warm:
        ident = 0.55 + 0.25 * min(1.0, thermal_check.hotspot_c / HOT_CORE_HINT)
        reasons = ["optical found it first, thermal confirmed heat",
                   thermal_check.reason]
    elif verifiable:
        # No heat, and thermal did look. Most of the time that means clutter:
        # a cloud edge, a shadow, foliage. Sometimes it means a cold-soaked
        # airframe, which is the one case a thermal-led system cannot see at
        # all. Persistence is the only thing that separates them, so this
        # starts low and climbs with how long the thing stays put.
        ident = 0.12 + 0.58 * settled
        reasons = ["optical-only contact, no heat where thermal looked",
                   thermal_check.reason]
        if settled > 0.15:
            reasons.append(f"still in the same place after {dwell_s:.0f} s - "
                           f"clutter usually is not")
    else:
        # Outside the thermal footprint entirely. Honest about being
        # unconfirmable rather than silently treating it as cleared.
        ident = 0.20 + 0.45 * settled
        reasons = ["optical-only contact, outside the thermal field of view",
                   "cannot be cross-checked on a fixed mount"]

    label = "cold contact" if (verifiable and not warm and settled > 0.25) \
        else "unverified contact"
    return Assessment(
        key=key, kind="optical", threat=ident, level=_level(ident),
        identity_risk=ident, site_risk=0.0, reasons=reasons,
        box=odet.box, centroid=odet.centroid, label=label,
        thermal_check=thermal_check, sensors="optical", dwell_s=dwell_s)


def assess_static(anomaly):
    """Verdict on a settled patch of scene with no motion cue at all.

    There is no classifier output here - nothing moved, so there are no
    temporal features and nothing to classify on. The site channel carries
    the whole decision, which is exactly the situation the challenge brief
    describes: the drone that landed and sat there.
    """
    # The floor is high on purpose. Reaching here already means the region
    # survived every gate the site model has: several seconds off baseline,
    # a mature scene reference, small enough to be an object rather than
    # weather, and no live track claiming it. Something is sitting where the
    # site says nothing sits, and that is the brief's scenario verbatim.
    #
    # It then ramps with how long it has been there and how hard it stands
    # out, so a few seconds is a watch item and half a minute is an alert.
    dwell_term = min(1.0, anomaly.dwell_s / 30.0)
    heat_term = min(1.0, abs(anomaly.peak_dev_c) / 8.0)
    site = 0.60 + 0.25 * dwell_term + 0.15 * heat_term
    threat = min(1.0, site)

    reasons = [f"object at rest {anomaly.dwell_s:.0f} s where the learned "
               f"site is empty",
               f"{anomaly.peak_dev_c:+.1f} C off baseline over "
               f"{anomaly.cells} cells"]

    return Assessment(
        key=anomaly.key, kind="static", threat=threat, level=_level(threat),
        identity_risk=0.0, site_risk=site, reasons=reasons,
        box=anomaly.box, centroid=anomaly.centroid, label="settled object",
        dwell_s=anomaly.dwell_s)
