"""Small, stable numbers for the things the two cameras are discussing.

Every object in the system already has a key - `track:7`, `optical:12:31`,
`static:4:9`. Keys are excellent for code and useless on a screen: they are
long, they are three different shapes, and the number inside a track key
climbs into the hundreds within a few minutes of running, so the one warm
mover in the room ends up labelled #214 while the cross-cue trail talks about
`optical:12:31` and nothing visibly connects the two.

What an operator needs while the site is being learned is the ability to point
at something on one pane and find the same thing on the other four, and in the
trail, and in the table. That wants ONE number, small enough to read at a
glance on a quarter-height pane, and the same number everywhere.

On screen these are called OBJECT numbers, never cue numbers. "Cross-cue" is
already this dashboard's word for one camera asking the other a question, and
a table column headed CUE read as a claim that a question had been asked about
that row - which is exactly the thing it does not mean. The internal names
stay `cue` because within the code the number is a visual cue and nothing else.

So numbers are handed out here, on first sight, and recycled once an object
has been gone long enough that reusing its number cannot confuse anybody.
Recycling is the whole point: a board that only ever counts upwards produces
exactly the three-digit labels this exists to replace. The cost is that #3 an
hour from now is not #3 today, which is what the log and the keys are for -
these numbers are for the picture in front of you right now.
"""


# How long a number stays reserved after its object was last seen. Long enough
# to survive a few dropped frames or a contact that flickers at the edge of a
# threshold; short enough that a room with four things in it uses the numbers
# 1 to 4 rather than 1 to 40.
FORGET_S = 8.0


class CueBoard:
    def __init__(self, forget_s=FORGET_S):
        self.forget_s = float(forget_s)
        self._n = {}        # key -> number
        self._seen = {}     # key -> last time it was asked about

    def number(self, key, now):
        """The number for this key, allocating the lowest free one if new."""
        if key is None:
            return None
        self._seen[key] = now
        n = self._n.get(key)
        if n is not None:
            return n
        taken = set(self._n.values())
        n = 1
        while n in taken:
            n += 1
        self._n[key] = n
        return n

    def prune(self, now):
        """Release numbers whose objects have been gone a while."""
        for key, last in list(self._seen.items()):
            if now - last > self.forget_s:
                del self._seen[key]
                self._n.pop(key, None)

    def snapshot(self):
        return dict(self._n)
