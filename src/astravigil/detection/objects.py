"""Naming what the optical camera is looking at.

Everything else in this system measures. The thermal detector measures
temperature and motion, the site models measure how far a patch is from what
it has learned, and the classifier reads a track's shape and flap rate. None
of them can say the word "chair".

That gap costs twice, in both directions:

  THERMAL ASKS OPTICAL "what is it?" and optical answered with geometry -
  solidity 0.71, aspect 1.30, 240 px. That is real evidence and it is nearly
  unreadable, and it cannot distinguish the two cases an operator most needs
  separated: a quadcopter and a chair back are both compact blobs.

  OPTICAL ASKS THERMAL "is this warm?" about a patch that has been off its
  baseline for a minute. If something can say the patch is a handbag left on
  a chair, the answer stops being a mystery.

So this wraps a COCO object detector - eighty everyday classes, the ones in
an indoor scene as well as the ones on an apron - and hands back names.

WHY OPENCV'S DNN AND NOT A FRAMEWORK

The Pi runs opencv from apt. Adding onnxruntime or torch to that machine
means a long build, a larger image, and another thing to break on an
upgrade, in exchange for nothing this needs: cv2.dnn already runs both of the
model formats worth using, on the CPU, at the rate this is throttled to.

NOTHING HERE IS REQUIRED

If no weights file is present the recogniser reports `available` False and
every caller carries on exactly as before. A perimeter sensor that will not
start because a 12 MB download is missing would be a worse system than one
that cannot name a chair.
"""
import glob
import os
import time
import zlib

import cv2
import numpy as np

from ..utils.env import env_float, env_int

# Where a model is looked for, and what is accepted. ONNX first: it is one
# self-contained file, which is the format a person can actually obtain and
# copy onto a Pi without also finding a matching text config.
MODEL_DIR = os.environ.get("ASTRAVIGIL_MODEL_DIR", "data/models")
MODEL_PATH = os.environ.get("ASTRAVIGIL_OBJECT_MODEL", "")

# Square input the image is letterboxed into, PREFERRED rather than fixed.
#
# Most YOLO ONNX exports bake their input resolution into the graph, and
# feeding a different one fails deep inside a reshape with an assertion that
# names neither the model nor the size. Measured on the yolov5n export this
# ships by default: a 320 blob into a 640 graph dies in Reshape2Layer. So
# this is where the probe starts, and the loader falls back to whatever the
# file actually wants.
#
# Small is preferred because cost scales with the square of it and this runs
# on a Pi; 0 means "ask the model and use whatever it says".
INPUT_SIZE = env_int("ASTRAVIGIL_OBJECT_INPUT", 320)

# Tried in order when the preferred size is rejected. These are the sizes
# YOLO exports are actually produced at.
SIZE_CANDIDATES = (320, 640, 416, 512, 256, 288, 480, 960, 1280)

# Keep a detection above this, and merge boxes overlapping more than this.
# 0.30 rather than a safer 0.45. Measured on a real room photo with the
# 4 MB yolov5n this ships by default: 0.40 named four objects, 0.30 named
# five, 0.20 named seven and started inventing a television out of a laptop
# screen. Naming is context rather than an alarm - nothing escalates because
# of it - so the cost of one wrong label is far below the cost of a silent
# pane, which is what an operator has already reported once.
CONF_MIN = env_float("ASTRAVIGIL_OBJECT_CONF", 0.30)
NMS_IOU = env_float("ASTRAVIGIL_OBJECT_NMS", 0.45)

# Frames between runs. This is the expensive thing in the loop by an order of
# magnitude, and what it reports - the identity of the furniture - changes on
# the timescale of somebody walking in and putting a bag down, not on the
# timescale of a frame. Throttling is not a compromise here; running it every
# frame would spend the whole budget re-deciding that the chair is a chair.
EVERY_N = env_int("ASTRAVIGIL_OBJECT_EVERY_N", 25)

# The 80 COCO classes, in the order every COCO-trained model emits them.
# Embedded rather than read from a file: it is fixed, it is small, and a
# missing labels file is one more way for this to half-work and mislabel
# everything by an offset of one.
COCO = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush")

# Classes that are furniture in any room this is likely to sit in. Named here
# so a settled patch that turns out to be a chair can say so - the single
# largest source of the false settled objects an indoor rig produces.
FURNITURE = frozenset({
    "chair", "couch", "bed", "dining table", "tv", "refrigerator", "oven",
    "microwave", "sink", "toilet", "potted plant", "bench", "book", "clock",
    "vase", "laptop", "keyboard", "mouse", "bowl", "cup", "bottle",
    "wine glass"})

# Classes worth saying out loud on a counter-UAS console.
AIRBORNE = frozenset({"airplane", "bird", "kite", "frisbee", "sports ball"})


class Recognition:
    """One named object in the optical frame."""

    __slots__ = ("label", "confidence", "box", "centroid")

    def __init__(self, label, confidence, box):
        self.label = label
        self.confidence = float(confidence)
        self.box = tuple(int(v) for v in box)      # x, y, w, h
        self.centroid = (self.box[0] + self.box[2] / 2.0,
                         self.box[1] + self.box[3] / 2.0)

    @property
    def furniture(self):
        return self.label in FURNITURE

    def as_dict(self):
        return {"label": self.label,
                "confidence": round(self.confidence, 3),
                "box": list(self.box)}

    def __repr__(self):
        return f"<{self.label} {self.confidence:.2f} {self.box}>"


def find_model():
    """The weights file to use, or None.

    An explicit path wins. Otherwise the first model-shaped file in the model
    directory, ONNX before TensorFlow, so dropping one file in is the whole
    installation procedure.
    """
    if MODEL_PATH:
        return MODEL_PATH if os.path.exists(MODEL_PATH) else None
    for pattern in ("*.onnx", "*.pb"):
        found = sorted(glob.glob(os.path.join(MODEL_DIR, pattern)))
        if found:
            return found[0]
    return None


class ObjectRecogniser:
    """COCO object names for the optical frame, or nothing at all.

    Two model formats, chosen by extension, because the two are what cv2.dnn
    reads without help and they suit different machines:

      .onnx   a YOLO export. One file, easy to obtain, and the output layout
              is worked out from its own shape rather than from a flag
              somebody has to set correctly.
      .pb     a TensorFlow SSD, with its .pbtxt beside it. Faster on a Pi,
              at the cost of needing two files that must match.
    """

    def __init__(self, path=None, conf=CONF_MIN, nms=NMS_IOU,
                 size=INPUT_SIZE, every_n=EVERY_N):
        self.path = path if path is not None else find_model()
        self.conf = float(conf)
        self.nms = float(nms)
        self.size = int(size)
        self.every_n = max(1, int(every_n))
        self.net = None
        self.kind = None
        self.error = None
        self.runs = 0
        self.last_ms = 0.0
        self._tick = 0
        self._last = []
        if self.path:
            self._load()
        else:
            self.error = (f"no model file in {MODEL_DIR} - "
                          f"run scripts/fetch_object_model.py")

    # ---------------------------------------------------------------- load
    def _load(self):
        try:
            if self.path.endswith(".onnx"):
                self.net = cv2.dnn.readNetFromONNX(self.path)
                self.kind = "onnx"
            else:
                cfg = os.path.splitext(self.path)[0] + ".pbtxt"
                if not os.path.exists(cfg):
                    raise FileNotFoundError(
                        f"{cfg} is required beside {self.path}")
                self.net = cv2.dnn.readNetFromTensorflow(self.path, cfg)
                self.kind = "tensorflow"
            # One thread would be wrong on a quad-core Pi and all of them
            # would starve the capture loop. Two leaves headroom for it.
            self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
            self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            try:
                cv2.setNumThreads(max(1, (os.cpu_count() or 2) - 2))
            except Exception:
                pass
            if self.kind == "onnx":
                self.size = self._probe_size()
        except Exception as exc:
            # A broken or half-downloaded model must not stop the sensor.
            self.net = None
            self.error = f"{type(exc).__name__}: {exc}"

    def _probe_size(self):
        """Find an input size this graph will actually accept.

        Asking the file directly means parsing ONNX, and a graph can declare a
        dynamic axis and still reject one - so this runs the thing instead.
        One forward pass per candidate at startup, worst case a few hundred
        milliseconds once, in exchange for never having to tell an operator
        which number to put in an environment variable.
        """
        tried = []
        for size in (self.size,) + SIZE_CANDIDATES:
            if size in tried or size <= 0:
                continue
            tried.append(size)
            blob = cv2.dnn.blobFromImage(
                np.zeros((size, size, 3), np.uint8), 1 / 255.0, (size, size),
                swapRB=True, crop=False)
            try:
                self.net.setInput(blob)
                out = np.squeeze(self.net.forward())
            except cv2.error:
                continue
            if out.ndim != 2:
                continue
            wide = min(out.shape)
            if wide not in (len(COCO) + 4, len(COCO) + 5):
                # Runs, but is not a COCO YOLO. Say so once here rather than
                # letting every frame fail the same way in the decoder.
                self.error = (f"model output is {out.shape}, which is not a "
                              f"COCO YOLO ({len(COCO) + 4} or "
                              f"{len(COCO) + 5} attributes)")
                self.net = None
                return size
            return size
        self.error = (f"no input size worked - tried {tried}. "
                      f"Set ASTRAVIGIL_OBJECT_INPUT if you know it.")
        self.net = None
        return self.size

    @property
    def available(self):
        return self.net is not None

    def stats(self):
        return {"available": self.available,
                "model": os.path.basename(self.path) if self.path else None,
                "kind": self.kind,
                "error": self.error,
                "runs": self.runs,
                "last_ms": round(self.last_ms, 1),
                "objects": len(self._last),
                "every_n": self.every_n}

    # -------------------------------------------------------------- detect
    def update(self, bgr):
        """Run on every Nth frame, and hand back the last answer between.

        The same contract the optical detector already uses. A caller must be
        able to read a name on every frame without paying for one on every
        frame, and furniture does not move between ticks.
        """
        if not self.available or bgr is None:
            return []
        if self._tick % self.every_n == 0:
            self._last = self.detect(bgr)
        self._tick += 1
        return self._last

    def detect(self, bgr):
        """One pass. Returns a list of Recognition, highest confidence first."""
        if not self.available or bgr is None:
            return []
        t0 = time.monotonic()
        try:
            out = (self._detect_onnx(bgr) if self.kind == "onnx"
                   else self._detect_tf(bgr))
        except Exception as exc:
            # A model that throws once will throw every time; say so and stop
            # rather than filling the log at frame rate.
            self.error = f"{type(exc).__name__}: {exc}"
            self.net = None
            return []
        self.last_ms = 1000.0 * (time.monotonic() - t0)
        self.runs += 1
        out.sort(key=lambda r: r.confidence, reverse=True)
        return out

    def _detect_tf(self, bgr):
        model = cv2.dnn.DetectionModel(self.net)
        model.setInputParams(size=(self.size, self.size), scale=1.0 / 127.5,
                             mean=(127.5, 127.5, 127.5), swapRB=True)
        ids, scores, boxes = model.detect(bgr, self.conf, self.nms)
        out = []
        for cid, score, box in zip(np.array(ids).flatten(),
                                   np.array(scores).flatten(), boxes):
            # TensorFlow's COCO ids are 1-based and skip numbers; the usual
            # mapping for this family is id-1 into the 80-class list.
            idx = int(cid) - 1
            if 0 <= idx < len(COCO):
                out.append(Recognition(COCO[idx], score, box))
        return out

    def _detect_onnx(self, bgr):
        h, w = bgr.shape[:2]
        # Letterbox rather than squash. A stretched frame moves every aspect
        # ratio the model was trained on, and aspect is most of what
        # separates a person from a chair.
        scale = min(self.size / w, self.size / h)
        nw, nh = int(round(w * scale)), int(round(h * scale))
        canvas = np.full((self.size, self.size, 3), 114, np.uint8)
        dx, dy = (self.size - nw) // 2, (self.size - nh) // 2
        canvas[dy:dy + nh, dx:dx + nw] = cv2.resize(bgr, (nw, nh))

        blob = cv2.dnn.blobFromImage(canvas, 1 / 255.0,
                                     (self.size, self.size), swapRB=True,
                                     crop=False)
        self.net.setInput(blob)
        pred = self.net.forward()
        pred = np.squeeze(pred)
        if pred.ndim != 2:
            raise ValueError(f"unexpected model output shape {pred.shape}")
        # v8 emits (84, 8400) and v5 emits (25200, 85). Whichever axis is
        # short is the attribute axis - no flag to set wrong.
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T
        wide = pred.shape[1]

        if wide == len(COCO) + 5:            # v5: cx cy w h obj cls...
            objectness = pred[:, 4]
            cls = pred[:, 5:]
            conf = objectness * cls.max(axis=1)
        elif wide == len(COCO) + 4:          # v8: cx cy w h cls...
            cls = pred[:, 4:]
            conf = cls.max(axis=1)
        else:
            raise ValueError(f"output has {wide} columns, which is neither "
                             f"{len(COCO) + 4} nor {len(COCO) + 5}")

        keep = conf >= self.conf
        if not keep.any():
            return []
        pred, conf, cls = pred[keep], conf[keep], cls[keep]
        ids = cls.argmax(axis=1)

        # Back out of the letterbox, in that order: the padding was added
        # after the resize, so it comes off before the scale.
        cx, cy = pred[:, 0] - dx, pred[:, 1] - dy
        bw, bh = pred[:, 2], pred[:, 3]
        boxes = np.stack([(cx - bw / 2) / scale, (cy - bh / 2) / scale,
                          bw / scale, bh / scale], axis=1)

        idx = cv2.dnn.NMSBoxes(boxes.tolist(), conf.tolist(), self.conf,
                               self.nms)
        out = []
        for i in np.array(idx).flatten():
            i = int(i)
            x, y, bw_, bh_ = boxes[i]
            x, y = max(0.0, x), max(0.0, y)
            bw_, bh_ = min(bw_, w - x), min(bh_, h - y)
            if bw_ < 2 or bh_ < 2:
                continue
            out.append(Recognition(COCO[int(ids[i])], conf[i],
                                   (x, y, bw_, bh_)))
        return out


def best_overlap(recognitions, box, min_iou=0.15):
    """The named object that best covers this box, or None.

    Intersection over the SMALLER area rather than over the union. A thermal
    detection mapped into optical space is often a fraction of the object it
    sits on - a warm hand on a chair back - and true IoU scores that pairing
    near zero for two things that are obviously the same place.
    """
    x, y, w, h = box
    if w <= 0 or h <= 0:
        return None
    best, best_score = None, min_iou
    for r in recognitions:
        rx, ry, rw, rh = r.box
        ix = max(0, min(x + w, rx + rw) - max(x, rx))
        iy = max(0, min(y + h, ry + rh) - max(y, ry))
        inter = ix * iy
        if not inter:
            continue
        score = inter / max(1.0, float(min(w * h, rw * rh)))
        if score > best_score:
            best, best_score = r, score
    return best


def colour_for(label):
    """A stable colour per class, so one class reads the same every run.

    crc32 and not hash(): Python randomises string hashing per process, so
    the obvious version would have repainted every class on every restart.
    """
    hue = zlib.crc32(label.encode()) % 180
    hsv = np.uint8([[[hue, 200, 245]]])
    b, g, r = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)[0][0]
    return int(b), int(g), int(r)
