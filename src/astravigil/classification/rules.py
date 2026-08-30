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

  HOT CORE  a quad carries a small very hot region far above its own mean.
            Measured on the real camera, that region is the CENTRE of the
            airframe - battery, ESCs and video transmitter - not the motors
            at the arm tips. This was assumed the other way round originally
            and the assumption was wrong; the feature works either way,
            because it only asks how far the peak sits above the object's own
            mean, but the simulation it was tuned against had to be corrected.

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
W_HOTSPOT = 0.30        # hot core above the object's own mean
W_SHAPE = 0.15          # only meaningful on a well-resolved target
W_STRAIGHT = 0.10       # weak evidence; birds fly straight too

FLAP_BIRD = 0.30        # area CoV at or above this reads as flapping
FLAP_RIGID = 0.15       # below this the silhouette is essentially rigid
# Both thresholds are validated against the two real captures, which is more
# than the rest of this file can claim:
#   close range  peak 45.4 C, blob mean ~31 C  ->  hotspot ~14 C
#   mid range    peak 35.0 C, blob mean ~27 C  ->  hotspot ~8 C
# So a real quadcopter clears HOTSPOT_DRONE_C with margin at both ranges, and
# the margin shrinks with distance exactly as expected - apparent peak falls
# as the target covers fewer pixels and each one averages in more background.
HOTSPOT_DRONE_C = 6.0   # peak this far above own mean implies a powered core
HOTSPOT_FLAT_C = 2.0    # a near-uniform blob is a body, not an airframe
STRAIGHT_DRONE = 0.90
MIN_HITS = 8            # frames of history before committing to a label
DECISION = 0.58         # below this margin, stay honest and say unknown

# --- is this an aircraft at all?
#
# Everything above discriminates a drone FROM A BIRD, and that is all it does.
# It has no null hypothesis, so anything that is not bird-like comes out the
# other side as a drone - and a radiator, a laptop vent or somebody's hands
# are about as un-birdlike as an object can be. Rigid silhouette scores the
# full wingbeat weight for "drone", a hot core above its own mean scores the
# hotspot weight for "drone", a compact blob scores the shape weight, and the
# console announces a 0.9-confidence quadcopter sitting motionless on a desk.
#
# That was defensible while the only things in frame were airborne: the
# detector was built for warm movers against cold sky, where everything it
# can see is already flying. It is wrong the moment the camera points at a
# room, and it is wrong in the most damaging direction, because a false
# "drone" at high confidence is exactly the alarm nobody can afford to
# ignore and exactly the one they will learn to.
#
# So two gates, both asking a question the discriminator never asks.

# Net displacement over the track history before "drone" is on the table.
#
# A drone that has flown into a perimeter camera's view has moved; a hot
# object in a room has not. Eight pixels over ~1.6 s is a very low bar - far
# below anything under command - and a stationary object clears none of it,
# because this is net displacement and jitter does not accumulate into it.
#
# A genuinely hovering airframe is downgraded to "unknown" by this, and that
# is the intended trade. It does NOT go quiet: threat is a noisy-OR over
# identity and site novelty, so something hovering where the site model has
# never seen traffic still raises on the site channel. What is lost is a
# confident NAME, which was never earned by a stationary blob anyway.
MIN_DISPLACEMENT_PX = 8.0

# Blob area above which this is not an airframe at any useful range.
#
# A 0.25 m quad spans about 8 px at 18 m and 2 px at 73 m on this sensor, so
# a target of even a few hundred pixels is either very close or very large.
# Two thousand is about four percent of a 256x192 frame: a person at indoor
# range clears it comfortably, an aircraft at watch range does not come near.
MAX_AIRFRAME_AREA_PX = 2000.0

# THE LANDED AIRFRAME EXEMPTION.
#
# The displacement gate above says a thing that has not moved is not an
# aircraft, and it is wrong about the one aircraft that matters most: a drone
# sitting on a surface with its battery and ESCs still warm has not moved
# either. Measured on the rig - a clear hot signature in the thermal frame,
# and the classifier answering "unknown" because the object was stationary.
# The gate added to stop a radiator being called a drone was stopping a drone
# being called a drone.
#
# What separates them is structure. A radiator, a mug or a laptop vent is one
# warm region. A quadcopter is several - motors, ESCs and a battery - inside
# one compact rigid outline, and the detector already counts those as parts.
# So a stationary object is exempt from the movement test when it looks like
# an airframe rather than like a warm surface:
#
#   a hot core well above its own mean, which a warm surface does not have;
#   two or more distinct hot parts, which a single warm object does not have;
#   and a compact outline, which a radiator does not have.
#
# All three, or the movement test stands.
# And the "hot core" half of that has to allow for the case the rig is
# actually looking at: a drone on a desk in front of a warm wall is COLDER
# than its background, not warmer. Its own peak-above-own-mean is small,
# because a cold-soaked airframe is uniformly cold - so requiring a hot core
# rules out exactly the cold-soaked target the rest of this system is built
# to catch. What it does have is contrast, in whichever direction, against
# the surface behind it. Either qualifies.
LANDED_MIN_PARTS = 2
LANDED_MAX_ASPECT = 2.6
LANDED_MIN_EXTENT = 0.25
LANDED_MIN_CONTRAST_C = 3.0


def looks_landed(detection):
    """Does this stationary warm thing look like an airframe, or like a
    warm surface?

    See LANDED_MIN_PARTS. The point is to let a drone that has landed
    keep its name without letting every warm object in a room acquire
    one.
    """
    aspect = detection.aspect
    if aspect < 1.0:
        aspect = 1.0 / max(aspect, 1e-6)
    stands_out = (detection.hotspot_c >= HOTSPOT_DRONE_C
                  or abs(detection.contrast_c) >= LANDED_MIN_CONTRAST_C)
    return (stands_out
            and detection.parts >= LANDED_MIN_PARTS
            and aspect <= LANDED_MAX_ASPECT
            and detection.extent >= LANDED_MIN_EXTENT)


def not_airborne(detection, track):
    """Why this cannot be an aircraft, or None if it might be.

    Returned as a sentence rather than a boolean so the reason can be put in
    front of an operator. "unknown" with no explanation is the same silence
    that made the previous answer untrustworthy.
    """
    if (track is not None and track.hits >= MIN_HITS
            and track.displacement_px < MIN_DISPLACEMENT_PX
            and not looks_landed(detection)):
        return (f"has not moved - {track.displacement_px:.1f} px of net "
                f"travel over its whole history")
    if detection.area > MAX_AIRFRAME_AREA_PX:
        return (f"far too large for an airframe at watch range "
                f"({int(detection.area)} px)")
    return None


def classify(detection, track):
    """Return (label, confidence in 0..1).

    Stays "unknown" until there is enough history for the temporal features to
    mean anything. Calling a two-frame blob a drone is how you get a confident
    wrong answer in front of judges.
    """
    if track is None or track.hits < MIN_HITS:
        return "unknown", 0.0
    if not_airborne(detection, track):
        # Not "bird", which would be a different confident wrong answer.
        # There is no evidence here about what it is, only about what it is
        # not, and the honest label for that is unknown.
        return "unknown", 0.0

    drone = bird = 0.0

    # A landed airframe has no wingbeat to measure and never will, so the
    # 0.45 the flap test carries has to come from somewhere or a stationary
    # drone can never clear the decision threshold. Its structure is the
    # evidence: several hot parts in a compact rigid outline is the thing a
    # warm surface cannot produce, and it is why the exemption let this
    # object past the movement gate in the first place.
    if (track.displacement_px < MIN_DISPLACEMENT_PX
            and looks_landed(detection)):
        return "drone", min(0.95, 0.62 + 0.06 * detection.parts)

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

    # --- hot core (battery/ESC/VTX stack), not the motors
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
