# Reassembling USB packets into thermal frames.
#
# The camera sends each frame as a run of packets, every one prefixed with a
# short UVC payload header. We strip the headers, concatenate the payloads,
# and cut a frame when the end-of-frame bit appears.
#
# Two pieces of hardening over the Pi 5 reference, both aimed at the Pi 4's
# busier USB path (VL805 controller, more chance of a dropped packet):
#
#   - the error bit is honoured, so a payload the camera itself flagged as bad
#     does not silently corrupt a frame
#   - the frame-id bit is tracked, so a *missed* end-of-frame packet is caught
#     instead of welding two half-frames into one plausible-looking frame that
#     passes the length check and produces nonsense temperatures
#
# Without the frame-id check a dropped EOF gives you a frame that is the right
# size and completely wrong, which is far worse for detection than a dropped
# frame - a torn frame looks like a scene that changed violently, which is
# exactly what the anomaly detector is watching for.

import time

import numpy as np

from .constants import (
    EXPECTED_FRAME_BYTES,
    SIZE_SLACK_BYTES,
    FRAME_H,
    FRAME_W,
    HEADER_EOF,
    HEADER_ERR,
    HEADER_FID,
    READ_CHUNK_BYTES,
    READ_TIMEOUT_MS,
    STREAM_ENDPOINT,
)


class FrameStats:
    # Running counts, so callers can tell "quiet" from "broken".
    def __init__(self):
        self.frames = 0
        self.short_frames = 0     # packet loss inside a frame
        self.torn_frames = 0      # a missed end-of-frame packet
        self.error_payloads = 0   # camera flagged the payload itself
        # A complete frame that is not the size the driver negotiated.
        #
        # These used to be truncated and reshaped anyway, which is the worst
        # of the available options. If the camera is sending rows of a
        # different width, cutting the buffer at the expected length starts
        # every row partway through the one before it, and the frame draws as
        # dense vertical stripes with a seam where the drift wraps. It looks
        # like a broken sensor and it is a broken assumption.
        self.mismatched_frames = 0
        self.last_frame_bytes = 0
        self.read_errors = 0      # timeouts and USB errors, lifetime total
        # Reset by every successful read. This is the one to judge health on:
        # the lifetime total climbs slowly even on a perfectly healthy feed,
        # because ordinary timeouts land in it too.
        self.consecutive_read_errors = 0

    def __repr__(self):
        return (f"FrameStats(frames={self.frames}, short={self.short_frames}, "
                f"torn={self.torn_frames}, errors={self.error_payloads}, "
                f"mismatched={self.mismatched_frames}, "
                f"reads={self.read_errors})")


def frames(dev, endpoint=STREAM_ENDPOINT, stop_event=None, stats=None,
           chunk=READ_CHUNK_BYTES, timeout_ms=READ_TIMEOUT_MS,
           on_read_error=None):
    # Yield raw uint16 frames of shape (FRAME_H, FRAME_W) until stop_event is
    # set, or forever if there is no stop_event.
    #
    # Frames are yielded as views over a fresh buffer each time, so callers can
    # hold on to one without it being overwritten by the next read.
    #
    # on_read_error is called with the running error count each time a read
    # fails. It exists because a camera that has died produces no frames at
    # all: anything watching only the yielded frames would sit silent forever,
    # which is the one failure this system must not be quiet about.
    stats = stats if stats is not None else FrameStats()

    payload = bytearray()
    current_fid = None
    frame_damaged = False

    while stop_event is None or not stop_event.is_set():
        try:
            data = dev.read(endpoint, chunk, timeout=timeout_ms)
        except Exception:
            # Timeouts are ordinary in normal running - the camera simply has
            # nothing ready yet. The caller decides when a run of these is
            # worth complaining about.
            stats.read_errors += 1
            stats.consecutive_read_errors += 1
            if on_read_error is not None:
                on_read_error(stats.consecutive_read_errors)
            continue

        stats.consecutive_read_errors = 0

        if len(data) < 2:
            continue

        header_len = data[0]
        flags = data[1]

        # A frame-id flip without an end-of-frame packet means we missed the
        # EOF. Whatever we have accumulated belongs to the previous frame and
        # is incomplete - drop it rather than merging it into this one.
        fid = flags & HEADER_FID
        if current_fid is not None and fid != current_fid and payload:
            stats.torn_frames += 1
            payload = bytearray()
            frame_damaged = False
        current_fid = fid

        if flags & HEADER_ERR:
            stats.error_payloads += 1
            frame_damaged = True

        if header_len < len(data):
            payload.extend(data[header_len:])

        if not (flags & HEADER_EOF):
            continue

        # End of frame.
        if frame_damaged:
            payload = bytearray()
            frame_damaged = False
            continue

        stats.last_frame_bytes = len(payload)
        if len(payload) < EXPECTED_FRAME_BYTES:
            # Lost packets mid-frame. Common enough on a loaded bus; just skip.
            stats.short_frames += 1
            payload = bytearray()
            continue

        # A complete frame that is too BIG is not a frame with extra on the
        # end - it is a frame of a different shape, which means the format
        # negotiation did not take. Truncating it produces a picture, and the
        # picture is a lie: rows sliced at the wrong stride, drawn as vertical
        # stripes. Refuse it and say what arrived instead.
        extra = len(payload) - EXPECTED_FRAME_BYTES
        if extra > SIZE_SLACK_BYTES:
            stats.mismatched_frames += 1
            if stats.mismatched_frames == 1:
                print(f"thermal: camera sent {len(payload)} bytes, expected "
                      f"{EXPECTED_FRAME_BYTES} for {FRAME_W}x{FRAME_H} - "
                      f"the stream format is not what was negotiated, so "
                      f"frames are being dropped rather than drawn wrong")
            payload = bytearray()
            continue

        raw16 = np.frombuffer(
            bytes(payload[:EXPECTED_FRAME_BYTES]), dtype="<u2",
        ).reshape((FRAME_H, FRAME_W))

        payload = bytearray()
        stats.frames += 1
        yield raw16


class _Deadline:
    # Minimal stop_event lookalike, so frames() can be given a time limit
    # without it needing to know anything about clocks.
    def __init__(self, seconds):
        self._expires_at = time.monotonic() + seconds

    def is_set(self):
        return time.monotonic() >= self._expires_at


def grab_frame(dev, endpoint=STREAM_ENDPOINT, wait_s=5.0, stats=None):
    # One frame, or None if nothing valid arrives inside wait_s.
    #
    # Convenience for the probe tool and for calibration captures, where a
    # single still frame is all that is wanted. The deadline goes in as a
    # stop_event because frames() blocks inside dev.read - checking the clock
    # after next() returns would never fire while the camera is silent.
    stream = frames(dev, endpoint=endpoint, stats=stats,
                    stop_event=_Deadline(wait_s))
    for raw16 in stream:
        return raw16
    return None
