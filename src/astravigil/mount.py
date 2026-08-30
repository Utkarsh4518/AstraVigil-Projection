"""How the two cameras are physically bolted on.

Quarter turns, anticlockwise, that take a sensor frame to the way a person
standing at the site would see it. Which way a camera ends up facing on a
bracket is not something code can derive; these are measured on the hardware
and set by whoever built the rig.

This lives on its own, away from both the dashboard and the detectors,
because it is neither a display preference nor a detection parameter - it is
a fact about the world that several unrelated parts of the system need to
agree on. They did not, and the disagreement cost real capability:

  The dashboard rotated the picture for the operator and the object
  recogniser was handed the raw sensor frame. On a camera mounted upside
  down that means a detector trained entirely on upright photographs was
  being shown an inverted room. Measured on one scene: four objects found
  upright, and one at a third of the confidence when inverted. A chair
  filling the frame was reported as nothing at all.

So anything that needs to know which way up the world is imports it from
here, and there is one answer.
"""
from .utils.env import env_int

# 3 on this rig: the thermal sensor is on its side AND inverted, so a single
# anticlockwise quarter turn left the scene upside down. Three of them - a
# quarter turn clockwise - is what puts it the right way up.
THERMAL_ROT = env_int("ASTRAVIGIL_THERMAL_ROT", 3) % 4

# 2 on this rig: the Pi camera is mounted inverted, so the scene needs a half
# turn to come up the right way.
OPTICAL_ROT = env_int("ASTRAVIGIL_OPTICAL_ROT", 2) % 4


def turn_point(x, y, w, h, k):
    """Where (x, y) lands after k anticlockwise turns of a (h, w) frame.

    One anticlockwise quarter turn sends column x to row w-1-x and row y to
    column y, and swaps the frame's own dimensions - so the step has to carry
    w and h along with it to compose correctly for k of 2 and 3.
    """
    for _ in range(k % 4):
        x, y, w, h = y, w - 1 - x, h, w
    return x, y


def turn_box(box, w, h, k):
    """The same for an (x, y, w, h) box on a (h, w) frame.

    Two opposite corners survive rotation and the rectangle is whatever they
    bound afterwards; rotating the origin alone would put the box in the
    right place with the wrong extent for odd k. Both corners must be
    INCLUSIVE, because turn_point mirrors about w - 1.
    """
    x, y, bw, bh = box
    ax, ay = turn_point(x, y, w, h, k)
    bx, by = turn_point(x + max(bw - 1, 0), y + max(bh - 1, 0), w, h, k)
    return min(ax, bx), min(ay, by), abs(bx - ax) + 1, abs(by - ay) + 1
