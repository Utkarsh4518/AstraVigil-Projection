"""Rule-based drone / bird / unknown classifier.

Hand-built rules over interpretable features, not a learned model. There is no
labelled thermal drone/bird data to train on, and a model trained on nothing is
worse than rules you can defend out loud.

Weighting reflects what actually survives at range. A drone and a bird at 40 m
are both a handful of warm pixels: silhouette, aspect and solidity are all
essentially noise there. Two things still separate them:

  WINGBEAT  a bird modulates its silhouette several times a second; a rigid
            airframe does not. Measured on the simulated scene, a flapping
            bird scores ~0.5 and a quad ~0.12, which is a wide margin.

  HOT CORES a quad carries motor hotspots far above its own body temperature.
            Note this is peak-above-OWN-MEAN, not contrast against the sky -
            grading on background contrast discriminates nothing, because a
            31 C bird against -5 C sky is exactly as bright as a drone.

Straightness is deliberately weighted low. Birds glide in straight lines often
enough that it is weak evidence, and leaning on it was what previously made
this classifier call every bird a drone.

The numbers below are tuned against SIMULATION, not real footage. Retune once
you have captured negatives, and do not quote the confidences as accuracy.
"""

W_FLAP = 0.45           # wingbeat - the strongest cue that survives at range
W_HOTSPOT = 0.30        # motor cores above the object's own mean
W_SHAPE = 0.15          # only meaningful on a well-resolved target
W_STRAIGHT = 0.10       # weak evidence; birds fly straight too

FLAP_BIRD = 0.30        # area CoV at or above this reads as flapping
FLAP_RIGID = 0.15       # below this the silhouette is essentially rigid
HOTSPOT_DRONE_C = 6.0   # peak this far above own mean implies motors
HOTSPOT_FLAT_C = 2.0    # a near-uniform blob is a body, not an airframe
STRAIGHT_DRONE = 0.90
MIN_HITS = 8            # frames of history before committing to a label
DECISION = 0.58         # below this margin, stay honest and say unknown


def classify(detection, track):
    """Return (label, confidence in 0..1).

    Stays "unknown" until there is enough history for the temporal features to
    mean anything. Calling a two-frame blob a drone is how you get a confident
    wrong answer in front of judges.
    """
    if track is None or track.hits < MIN_HITS:
        return "unknown", 0.0

    drone = bird = 0.0

    # --- wingbeat
    flap = track.flap_score
    if flap >= FLAP_BIRD:
        bird += W_FLAP
    elif flap <= FLAP_RIGID:
        drone += W_FLAP
    else:
        # Between the two: split the weight proportionally rather than
        # forcing a side.
        f = (flap - FLAP_RIGID) / (FLAP_BIRD - FLAP_RIGID)
        bird += W_FLAP * f
        drone += W_FLAP * (1.0 - f)

    # --- motor hotspots
    hot = detection.hotspot_c
    if hot >= HOTSPOT_DRONE_C:
        drone += W_HOTSPOT
    elif hot <= HOTSPOT_FLAT_C:
        bird += W_HOTSPOT
    else:
        f = (hot - HOTSPOT_FLAT_C) / (HOTSPOT_DRONE_C - HOTSPOT_FLAT_C)
        drone += W_HOTSPOT * f
        bird += W_HOTSPOT * (1.0 - f)

    # --- silhouette, only trusted on a target big enough to have one
    if detection.area >= 25:
        if detection.aspect > 2.0:
            bird += W_SHAPE
        elif 0.6 <= detection.aspect <= 1.8 and detection.extent > 0.4:
            drone += W_SHAPE
    # A drone resolved into body-plus-motors fills its bounding box loosely
    # while staying compact overall - a signature a bird does not produce.
    if detection.parts >= 3:
        drone += W_SHAPE * 0.5

    # --- path regularity
    if track.straightness >= STRAIGHT_DRONE:
        drone += W_STRAIGHT
    else:
        bird += W_STRAIGHT * min(1.0, (STRAIGHT_DRONE - track.straightness)
                                 / STRAIGHT_DRONE)

    total = drone + bird
    if total <= 0:
        return "unknown", 0.0

    if drone >= bird:
        conf = drone / total
        return ("drone", conf) if conf >= DECISION else ("unknown", conf)
    conf = bird / total
    return ("bird", conf) if conf >= DECISION else ("unknown", conf)
