# HIKMICRO Mini2 Plus V2 - device identity, frame geometry, and Pi 4 tuning.
# Everything the driver needs to know about the hardware lives here, so the
# tuning knobs that differ between boards sit in one file.

import os

# USB identity. The same on every board - the driver finds the camera by
# VID/PID, never by /dev/video* path, so port order and enumeration order
# do not matter.
VENDOR_ID = 0x2BDF
PRODUCT_ID = 0x0102

# The camera enumerates as a standard UVC composite device: interface 0 is
# the control interface (class 0x0e/0x01), interface 1 carries the video
# stream (class 0x0e/0x02).
#
# Endpoint 0x81 is BULK, not isochronous, confirmed off the hardware:
#   /sys/bus/usb/devices/1-1:1.1/ep_81/type -> Bulk    (bmAttributes 0x02)
#   wMaxPacketSize 512
#
# That is the fact this whole driver rests on, and it is unusual - most UVC
# cameras stream over isochronous endpoints. It matters twice over:
# pyusb's plain dev.read() handles bulk properly but has no real support for
# isochronous transfers, and bulk survives USB/IP, which is what makes it
# possible to test this driver over WSL without a Pi in front of you.
STREAM_INTERFACE = 1
STREAM_ENDPOINT = 0x81

# UVC control requests, used to negotiate the streaming format at open time.
UVC_SET_CUR = 0x01
UVC_GET_CUR = 0x81
UVC_REQ_TYPE_SET = 0x21
UVC_REQ_TYPE_GET = 0xA1
PROBE_CONTROL = 0x0100
COMMIT_CONTROL = 0x0200
PROBE_LENGTH = 34

# The mode we ask for, by the indices the camera advertises in its descriptors:
#
#   format 1, frame 1   256 x 344, 16 bits/pixel   the raw thermal stream
#   format 2, frame 2   256 x 192, 12 bits/pixel   NV12, what V4L2 shows
#
# The camera powers up on the NV12 mode, so the one we want has to be asked
# for by index. Overridable because these are per-unit descriptor numbers, and
# checking a different pair should not need a code change:
#
#   lsusb -v -d 2bdf:0102     # look for VideoStreaming Interface Descriptors
STREAM_FORMAT_INDEX = int(os.environ.get("ASTRAVIGIL_FORMAT_INDEX", "1"))
STREAM_FRAME_INDEX = int(os.environ.get("ASTRAVIGIL_FRAME_INDEX", "1"))

# Byte offsets into the 34-byte probe/commit control block (UVC 1.1).
PROBE_FORMAT_INDEX = 2       # bFormatIndex
PROBE_FRAME_INDEX = 3        # bFrameIndex
PROBE_MAX_FRAME_SIZE = 18    # dwMaxVideoFrameSize, 4 bytes little-endian

# Frame geometry. The sensor is 256x192, but every frame carries 344 rows of
# 16-bit little-endian values: 192 rows of thermal data, then a metadata row,
# then trailing rows we do not use.
#
# The camera confirms this itself. A real metadata row reads:
#   [1612, 1572, 0, 36, 7, 1, 0, 0, 192, 256, 67, 0, 0, 0, 0, 5]
#             ^ambient 16.12 C              ^^^  ^^^ height, width
# so cells 8 and 9 carry the thermal dimensions, matching THERMAL_ROWS and
# FRAME_W below.
FRAME_W = 256
FRAME_H = 344
THERMAL_ROWS = 192
METADATA_ROW = 192
BYTES_PER_PIXEL = 2
EXPECTED_FRAME_BYTES = FRAME_W * FRAME_H * BYTES_PER_PIXEL  # 176128

# Ambient temperature sits in the first cell of the metadata row, in
# hundredths of a degree.
AMBIENT_SCALE = 100.0

# Raw sensor counts to a linear temperature, before the correction curve.
TEMP_BASELINE = 4850
TEMP_SCALE = 31.0

# UVC payload header bits (byte 1 of each USB packet).
HEADER_FID = 0x01    # frame id, toggles between consecutive frames
HEADER_EOF = 0x02    # end of frame
HEADER_ERR = 0x40    # camera flagged this payload as bad

# --- Pi 4 tuning ---------------------------------------------------------
# The Pi 4's Cortex-A72 is meaningfully slower than the Pi 5's A76, and its
# USB sits behind the VL805 PCIe controller. Both push in the same direction:
# do fewer, larger reads so there is less Python-level work per frame.
#
# A frame is 176 KB, so at 16 KB per read that is 11+ round trips through
# pyusb per frame, 25 times a second. Doubling the chunk halves that.
# If your kernel returns short reads or the stream turns unreliable, drop
# READ_CHUNK_BYTES back to 16384 - that is the value the Pi 5 rig used.
#
# REVERTED to 16384. Doubling it was an untested guess that measured fine over
# USB/IP on a laptop and does not belong on the value the Pi 5 rig actually
# proved. The Pi's own USB stack is not the one that was measured, and a
# streaming driver is the wrong place to carry an optimisation nobody has run
# on the target board. Override to experiment; do not change the default
# without a Pi in front of you.
READ_CHUNK_BYTES = int(os.environ.get("ASTRAVIGIL_READ_CHUNK", "16384"))
READ_TIMEOUT_MS = int(os.environ.get("ASTRAVIGIL_READ_TIMEOUT_MS", "200"))

# Reads time out constantly in normal running, so a handful of failures in a
# row mean nothing. Only complain once it is clearly stuck.
ERRORS_BEFORE_WARNING = 25
ERRORS_BETWEEN_WARNINGS = 150
