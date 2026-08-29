"""Reading tuning knobs out of the environment without dying on a typo.

Every one of these is parsed at import time, and an import-time exception in
this project does not present as a configuration error. It presents as the
GUI never opening: run_dashboard fails before app.run() binds the port, the
kiosk launcher polls /api/state for ninety seconds, finds nothing, and shows
"the pipeline did not start listening on port 8000". The actual cause -
ASTRAVIGIL_THERMAL_ROT=3.0, or a value with a trailing comment the shell kept
- is one line in a log nobody has reason to open yet.

A perimeter sensor should not be stoppable by a mistyped display setting. So
a bad value warns on stderr and the default is used: the operator gets a rig
that runs and a message naming the variable, instead of a black screen.

Nothing here validates ranges. A value that parses but is silly is the
caller's business, and callers already clamp what needs clamping.
"""
import os
import sys


def _warn(name, raw, default, exc):
    print(f"warning: {name}={raw!r} is not usable ({exc}); "
          f"falling back to {default!r}", file=sys.stderr)


def env_int(name, default):
    """int from the environment, or `default` if it is missing or malformed."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw.strip())
    except ValueError as exc:
        _warn(name, raw, default, exc)
        return default


def env_float(name, default):
    """float from the environment, or `default` if missing or malformed."""
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw.strip())
    except ValueError as exc:
        _warn(name, raw, default, exc)
        return default
