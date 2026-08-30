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
import threading
import time
import urllib.error
import urllib.request
import zlib

import cv2
import numpy as np

from ..mount import OPTICAL_ROT, turn_box
from ..utils.env import env_float, env_int

# Where a model is looked for, and what is accepted. ONNX first: it is one
# self-contained file, which is the format a person can actually obtain and
# copy onto a Pi without also finding a matching text config.
MODEL_DIR = os.environ.get("ASTRAVIGIL_MODEL_DIR", "data/models")
MODEL_PATH = os.environ.get("ASTRAVIGIL_OBJECT_MODEL", "")

# Where the default weights come from, and whether to go and get them.
#
# Fetching automatically rather than printing an instruction. The alternative
# was tried: the pane said "run scripts/fetch_object_model.py", and the rig
# ran for two more sessions with the naming switched off and an operator
# asking why the objects had no names. A 4 MB download that happens once, in
# the background, on a machine that is already on the network is not a
# decision worth interrupting somebody for.
#
# Set ASTRAVIGIL_OBJECT_AUTOFETCH=0 on an air-gapped rig, or where a
# particular model has been chosen deliberately.
RELEASE = "https://github.com/ultralytics/yolov5/releases/download/v7.0/"
SIZES = {
    "n": ("yolov5n.onnx", "4 MB, fastest, fewest names"),
    "s": ("yolov5s.onnx", "15 MB, noticeably better - try this first on a PC"),
    "m": ("yolov5m.onnx", "43 MB, best, slow on a Pi"),
}
# `s` and not `n`, which was the wrong call and cost three rounds of "why is
# the chair not detected".
#
# `n` is the smallest model in the family and it was chosen for a Pi, back
# when inference ran on the capture thread and every millisecond was coming
# out of the frame rate. It no longer does, so the only cost of a bigger
# model is how often the answer refreshes - and the difference in what comes
# back is not marginal. Measured on one room photograph, cropped progressively
# tighter until a chair filled the view the way it fills the rig's:
#
#   whole room          n: 4 objects        s: 9 objects, including the chair
#   chair, some room    n: nothing          s: chair 0.35
#   chair fills frame   n: nothing          s: chair 0.36
#
# `n` finds nothing at all once one object dominates the frame, which is the
# ordinary case for a camera watching a room rather than a sky.
DEFAULT_MODEL = SIZES[os.environ.get("ASTRAVIGIL_OBJECT_SIZE", "s")][0]

# Best first. find_model() used to take whatever sorted() handed back, which
# is alphabetical, which means an install that already had yolov5n.onnx would
# keep using it after fetching a better one - a silent downgrade at the worst
# possible moment, right after somebody went and got the bigger model
# specifically because the smaller one was not working.
# Best first, where "best" means best ON THE TARGET, not most accurate.
#
# `m` used to lead and it is the wrong choice for a Pi 4 by a wide margin:
# measured on the rig, 5299 ms per pass against 1273 for `s`. Four times the
# time for a marginal gain in what it names, on a machine that is also trying
# to keep a camera at frame rate. `s` leads now; `m` is still used if it is
# the only one present, and ASTRAVIGIL_OBJECT_MODEL forces either.
MODEL_PREFERENCE = ("yolov5s.onnx", "yolov5m.onnx", "yolov5n.onnx")

# A second model, run alternately with the first.
#
# COCO knows furniture and has never seen a quadcopter; a drone model knows
# quadcopters and nothing else. Neither answers the other's question, so the
# useful thing is both - and running them on alternate passes costs exactly
# one inference per pass, the same as running one. Each set of names is held
# for twelve seconds either way, so both stay on screen between their turns.
MODEL2_PATH = os.environ.get("ASTRAVIGIL_OBJECT_MODEL2", "")
DRONE_MODEL = "drone.onnx"

# A YOLOv5 trained on drones: 416px input, one class, 28 MB. It is the only
# model in this system that can produce the word "drone" - COCO cannot, and
# guesses "bowl", "dog" or "laptop" instead - so it is fetched automatically
# rather than left as a step to remember.
DRONE_URL = ("https://huggingface.co/engdarwish/drone-detection-yolov5/"
             "resolve/main/best.onnx")
DRONE_PATH = os.path.join(MODEL_DIR, DRONE_MODEL)

# The drone model gets a much lower bar than the COCO one, and the reason is
# structural rather than a fudge.
#
# A confidence threshold is protection against the model picking the wrong
# CLASS. With eighty classes that is a real risk and 0.30 is the right guard.
# With one class there is no wrong class to pick: the model either found a
# drone-shaped thing or it did not, and the score is only how sure it is. The
# cost of a false positive here is one extra box on a pane; the cost of a
# false negative is missing the object the whole system is for.
#
# It also matters that this model was trained on drones IN FLIGHT, against
# sky. A quadcopter sitting on a desk, side-on, at two metres is outside that
# distribution and will score low even when it is plainly right - which is
# what a rig with the model correctly loaded and naming nothing looks like.
DRONE_CONF = env_float("ASTRAVIGIL_DRONE_CONF", 0.12)

# Class names that are direct evidence of an aircraft rather than context.
# A model trained on drones saying "drone" is worth something; a COCO model
# cannot produce this word at all, which is the point.
AIRCRAFT_NAMES = frozenset({"drone", "uav", "quadcopter", "quadrotor"})
AUTOFETCH = env_int("ASTRAVIGIL_OBJECT_AUTOFETCH", 1) != 0
USER_AGENT = "AstraVigil/1.0"

# Square input the image is letterboxed into.
#
# 0 means "ask the model", which is the right default and now the only one
# most people should use: every YOLO ONNX export bakes its input resolution
# into the graph, and feeding a different one fails deep inside a reshape.
# Set it only to force a specific size on a model that accepts several.
INPUT_SIZE = env_int("ASTRAVIGIL_OBJECT_INPUT", 0)

# Tried in order. 640 first because that is what essentially every YOLO
# export off the shelf is built at - including all three this ships.
#
# The order matters for more than speed. A rejected size does not raise
# quietly: OpenCV prints a four-line assertion failure from C++ straight to
# stderr, naming a reshape layer and a tensor shape, and an operator who has
# just run the install command reasonably reads that as a broken install and
# stops. Getting it right on the first attempt means there is nothing to
# print. The probe also silences OpenCV's own logging while it works, for
# the models where the first guess is still wrong.
SIZE_CANDIDATES = (640, 320, 416, 512, 480, 288, 256, 960, 1280)

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

# Seconds a name survives a pass that did not repeat it.
#
# A chair sitting still scored 0.35 one run and nothing the next, so it
# appeared and vanished on a pane where nothing had moved. That is not the
# model changing its mind about the room; it is a confidence sitting on a
# threshold, sampled every couple of seconds. Furniture does not leave
# between passes, so a name is held for a few of them and only forgotten if
# it stops being reported for longer than a person would take to carry it
# out of the room.
HOLD_S = env_float("ASTRAVIGIL_OBJECT_HOLD_S", 12.0)

# Contrast stretching before inference was tried here and MEASURED WORSE, so
# there is none. It seemed obvious: a detector trained on daylight cannot see
# a frame whose histogram sits in the bottom fifth of the range, and CLAHE
# spreads it back out. On the same room photographed and then dimmed, with
# the boost against without:
#
#   mean 37   laptop                vs   laptop, cup, mouse
#   mean 12   nothing               vs   laptop
#
# It loses objects at every light level and loses them completely at the
# bottom. The network normalises its own input and copes with a dark frame
# better than it copes with one whose local statistics have been rewritten;
# CLAHE amplifies sensor noise, which in a dark room is most of what is
# there. Recorded because it is a plausible idea that a later reader will
# have again.
#
# What does help in the dark is a bigger model: --size s.

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

# COCO HAS NO DRONE CLASS. Eighty everyday objects, and a quadcopter is not
# one of them, so a COCO model shown a drone cannot answer "drone" - it
# answers with whichever of its eighty it finds least unlike, and on a real
# rig that came back as "bowl 0.56".
#
# That is not a bad model. It is a model being asked a question outside the
# set it was given, and it has no way to say so. Which means a COCO name is
# evidence about SOME objects and no evidence at all about others, and the
# system has to know which is which.
#
# These are the names allowed to overrule the thermal classifier. Every one
# is large, static, and structurally impossible to confuse with a small
# airframe: if the optical camera says the warm blob is sitting on a sofa,
# the blob is not a quadcopter. Everything else - bowl, cup, frisbee, sports
# ball, clock, vase, mouse, kite, bird - is exactly the shape a drone gets
# read as, and is reported as optical's opinion rather than as the object's
# identity.
CAN_OVERRULE = frozenset({
    "person", "chair", "couch", "bed", "dining table", "tv", "laptop",
    "refrigerator", "oven", "microwave", "sink", "toilet", "bench",
    "potted plant", "car", "truck", "bus", "motorcycle", "bicycle",
    "train", "boat", "horse", "cow", "elephant", "bear", "zebra",
    "giraffe", "dog", "cat"})


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


def _hush_opencv():
    """Silence OpenCV's C++ logging; returns a callable to restore it.

    Only ever wrapped around the size probe. A rejected input size is an
    expected, handled outcome there - the loop exists precisely to try sizes
    until one fits - but OpenCV reports it as a multi-line ERROR on stderr
    before the exception is even raised, which reads as a failed install to
    anybody who has just typed the install command.
    """
    try:
        log = cv2.utils.logging
        before = log.getLogLevel()
        log.setLogLevel(log.LOG_LEVEL_SILENT)
        return lambda: log.setLogLevel(before)
    except Exception:
        # Not every build exposes this. Noisy is better than broken.
        return lambda: None


def download_model(url, dest, timeout=60, progress=None):
    """Fetch one weights file, reporting progress, leaving no half-file.

    Downloads to a .part and renames, so an interrupted fetch cannot leave
    something that looks like a model and fails at load - which on a sensor
    that starts unattended is a much worse outcome than no model at all.
    """
    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    tmp = dest + ".part"
    got = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length") or 0)
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if progress:
                    progress(got, total)
    if got < 100000:
        os.remove(tmp)
        raise OSError(f"only {got} bytes - that is not a model")
    os.replace(tmp, dest)
    return got


def _load_labels(model_path):
    """Class names for a model: its own if it has any, COCO otherwise.

    Looked for as <model>.names and <model>.txt. A model trained on something
    this system actually cares about - drones - will not have eighty everyday
    classes, and hard-coding COCO would either crash on it or mislabel every
    detection by an offset.
    """
    if model_path:
        base = os.path.splitext(model_path)[0]
        for ext in (".names", ".txt"):
            try:
                with open(base + ext, encoding="utf-8") as fh:
                    names = [ln.strip() for ln in fh if ln.strip()]
                if names:
                    return tuple(names)
            except OSError:
                pass
    return COCO


def find_second_model(primary=None):
    """The other model to run on alternate passes, or None.

    A drone model in the model directory is picked up on sight. It used to
    need an environment variable pointing at it, which is one more step
    between fetching the thing and it working - and a step that, got wrong,
    produces a rig naming a quadcopter "dog" with no indication that the
    model which knows the word "drone" was never loaded. Installing it is
    now the whole of installing it.
    """
    if MODEL2_PATH:
        return MODEL2_PATH if os.path.exists(MODEL2_PATH) else None
    path = os.path.join(MODEL_DIR, DRONE_MODEL)
    if not os.path.exists(path):
        return None
    if primary and os.path.abspath(path) == os.path.abspath(primary):
        return None            # it is already the primary; do not run twice
    return path


def find_model():
    """The weights file to use, or None.

    An explicit path wins. Otherwise the first model-shaped file in the model
    directory, ONNX before TensorFlow, so dropping one file in is the whole
    installation procedure.
    """
    if MODEL_PATH:
        return MODEL_PATH if os.path.exists(MODEL_PATH) else None
    for name in MODEL_PREFERENCE:
        path = os.path.join(MODEL_DIR, name)
        if os.path.exists(path):
            return path
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
                 size=INPUT_SIZE, every_n=EVERY_N, autofetch=None,
                 rot=None):
        self.path = path if path is not None else find_model()
        self.conf = float(conf)
        self.nms = float(nms)
        self.size = int(size)
        self.every_n = max(1, int(every_n))
        # Quarter turns to put the sensor frame the right way up before
        # inference. Every COCO model is trained on upright photographs and
        # falls apart on an inverted one - measured on a real room, four
        # objects upright against one at a third of the confidence upside
        # down. Boxes are turned back into sensor coordinates afterwards, so
        # nothing downstream needs to know this happened.
        self.rot = (OPTICAL_ROT if rot is None else rot) % 4
        # Class names, from a sidecar file if there is one. This is what
        # makes a model somebody trained on drones usable: drop the .onnx in
        # with a .names beside it, one class per line, and nothing else here
        # needs to change.
        self.labels = _load_labels(self.path)
        self.net = None
        self.kind = None
        self.error = None
        self.runs = 0
        self.last_ms = 0.0
        self.fetch_pct = None
        self._tick = 0
        self._last = []
        # Inference runs on its own thread. Measured on the rig: 750 ms per
        # pass on a Pi 4 - which, called inline, is 750 ms the capture loop
        # spends not capturing, every twenty-fifth frame, as a visible stall.
        # Nothing downstream needs this frame's answer on this frame: what it
        # reports is the identity of the furniture.
        self._busy = False
        self._lock = threading.Lock()
        self.stale_ms = 0.0
        # label -> (Recognition, last time it was reported). Keyed by label
        # and position so two chairs stay two chairs.
        self._held = []
        if self.path:
            self._load()
            return
        self.error = f"no model file in {MODEL_DIR}"
        # Only when nothing was named explicitly. Someone who passed a path
        # meant that path, and quietly substituting a different model for one
        # they chose would be worse than failing.
        want = AUTOFETCH if autofetch is None else autofetch
        if want and path is None and not MODEL_PATH:
            self._start_fetch()

    # --------------------------------------------------------------- fetch
    def fetch_drone_model(self):
        """Fetch the drone model in the background if it is not there.

        Called for the SECOND slot, which is empty on a rig that has only
        ever fetched COCO. It was a command somebody had to run, and it did
        not get run - three sessions of a quadcopter being called a bowl, a
        dog and a laptop, because the only model that can say the word
        "drone" was one manual step away and nothing on screen said so.
        A 28 MB download that happens once is not worth a step.
        """
        dest = os.path.join(MODEL_DIR, DRONE_MODEL)
        # This recogniser IS the drone slot, whatever it was pointed at
        # before. Saying so here rather than at the call site keeps the two
        # from ever disagreeing about which model this object holds.
        self.path = dest
        self.labels = _load_labels(dest)
        if os.path.exists(dest):
            self.net, self.error = None, None
            self._load()
            return
        self.net = None
        self.fetch_pct = 0.0

        def run():
            def progress(got, total):
                self.fetch_pct = (100.0 * got / total) if total else None
            try:
                download_model(DRONE_URL, dest, progress=progress)
                with open(os.path.splitext(dest)[0] + ".names", "w",
                          encoding="utf-8") as fh:
                    fh.write("drone" + chr(10))
            except (urllib.error.URLError, OSError) as exc:
                self.fetch_pct = None
                self.error = f"could not fetch {DRONE_MODEL} ({exc})"
                return
            self.fetch_pct = None
            self.path = dest
            self.labels = _load_labels(dest)
            # Clear the failure from before the file existed, or a model that
            # loaded perfectly reports the error it had while it was still
            # downloading.
            self.error = None
            self._load()
            print(f"object naming: fetched {dest} "
                  f"({'ready' if self.available else self.error})")

        threading.Thread(target=run, daemon=True,
                         name="astravigil-drone-fetch").start()

    def _start_fetch(self):
        dest = os.path.join(MODEL_DIR, DEFAULT_MODEL)
        self.fetch_pct = 0.0
        self.error = f"downloading {DEFAULT_MODEL}"

        def run():
            def progress(got, total):
                self.fetch_pct = (100.0 * got / total) if total else None

            try:
                download_model(RELEASE + DEFAULT_MODEL, dest,
                               progress=progress)
            except (urllib.error.URLError, OSError) as exc:
                # Never fatal. The sensor's job does not depend on knowing
                # the word "chair", and a rig with no route to the internet
                # must still run exactly as it did before.
                self.fetch_pct = None
                self.error = (f"could not fetch {DEFAULT_MODEL} "
                              f"({getattr(exc, 'reason', exc)}) - copy any "
                              f"COCO YOLO .onnx into {MODEL_DIR}/")
                return
            self.fetch_pct = None
            self.path = dest
            self.error = None
            self._load()
            print(f"object naming: fetched {dest}"
                  if self.available else
                  f"object naming: fetched {dest} but {self.error}")

        threading.Thread(target=run, daemon=True,
                         name="astravigil-model-fetch").start()

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
            # Both are already the defaults, and newer OpenCV logs a
            # warning for setting them at all. Asked for explicitly because
            # older builds do not default the same way, and hushed because a
            # warning about a no-op is one more line an operator has to read
            # past to find out whether the install worked.
            quiet = _hush_opencv()
            try:
                self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
            finally:
                quiet()
            # cv2.setNumThreads used to be called here to keep the DNN off
            # every core. It is a PROCESS-WIDE setting, not a per-network
            # one, so it also throttled the thermal detector, every resize,
            # every morphology - the work that happens on every single frame
            # - in order to limit something that runs once every few seconds.
            # That is the wrong trade by two orders of magnitude in duty
            # cycle, and it is the sort of thing that quietly halves a frame
            # rate. Left at the default now; the inference thread is one
            # thread among many and the scheduler can do its job.
            if self.kind == "onnx":
                self.size = self._probe_size()
            elif self.size <= 0:
                # A TensorFlow SSD takes whatever size it is given, so
                # nothing is probed and "ask the model" has no answer.
                self.size = 320
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
        quiet = _hush_opencv()
        try:
            return self._probe_sizes(tried)
        finally:
            quiet()

    def _probe_sizes(self, tried):
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
            if wide not in (len(self.labels) + 4, len(self.labels) + 5):
                # Runs, but is not a COCO YOLO. Say so once here rather than
                # letting every frame fail the same way in the decoder.
                self.error = (
                    f"model output is {out.shape}, which does not fit "
                    f"{len(self.labels)} class names - put a .names file "
                    f"with one class per line beside the model")
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
                "rot": self.rot,
                "labels": len(self.labels),
                "coco": self.labels is COCO,
                "error": self.error,
                "fetch_pct": (round(self.fetch_pct, 1)
                              if self.fetch_pct is not None else None),
                "runs": self.runs,
                "last_ms": round(self.last_ms, 1),
                "busy": bool(self._busy),
                "objects": len(self._last),
                "held": len(self._held),
                "every_n": self.every_n}

    # -------------------------------------------------------------- detect
    def update(self, bgr):
        """Hand back the latest answer, and start a new one if it is time.

        NEVER blocks. The caller is a capture loop with a frame budget of a
        few tens of milliseconds and this takes the better part of a second
        on the target hardware; the only correct amount of that to spend on
        the hot path is none. One pass is in flight at a time, so a slow
        machine degrades by answering less often rather than by queueing.
        """
        if not self.available or bgr is None:
            return self._last
        if self._tick % self.every_n == 0:
            self._submit(bgr)
        self._tick += 1
        return self._last

    def _submit(self, bgr):
        with self._lock:
            if self._busy:
                # Still working on the previous one. Skipping is right:
                # a queue of stale frames would answer questions about a room
                # as it was several seconds ago.
                return
            self._busy = True
        frame = bgr.copy()          # the caller reuses its buffer

        def run():
            try:
                self._last = self._remember(self.detect(frame))
            finally:
                self._busy = False

        threading.Thread(target=run, daemon=True,
                         name="astravigil-recognise").start()

    def forget(self):
        """Drop every held name and re-run on the next frame."""
        self._held = []
        self._last = []
        self._tick = 0

    def _remember(self, found, now=None):
        """Merge this pass into the held set and drop what has gone quiet.

        A name that repeats refreshes its entry and takes the newer box; one
        that does not is kept until HOLD_S has passed. The result is what the
        pane draws, so an object on the confidence threshold reads as present
        rather than as flickering.
        """
        now = time.monotonic() if now is None else now
        kept = []
        for r, seen in self._held:
            if now - seen <= HOLD_S:
                kept.append([r, seen])
        for f in found:
            for entry in kept:
                if entry[0].label == f.label and _overlaps(entry[0].box,
                                                          f.box):
                    entry[0], entry[1] = f, now
                    break
            else:
                kept.append([f, now])
        self._held = [(r, seen) for r, seen in kept]
        out = [r for r, _ in self._held]
        out.sort(key=lambda r: r.confidence, reverse=True)
        return out

    def detect(self, bgr):
        """One pass. Returns a list of Recognition, highest confidence first.

        Boxes come back in SENSOR coordinates, the same frame everything else
        in the pipeline works in, however the camera happens to be mounted.
        """
        if not self.available or bgr is None:
            return []
        upright = (np.ascontiguousarray(np.rot90(bgr, self.rot))
                   if self.rot else bgr)
        t0 = time.monotonic()
        try:
            out = (self._detect_onnx(upright) if self.kind == "onnx"
                   else self._detect_tf(upright))
        except Exception as exc:
            # A model that throws once will throw every time; say so and stop
            # rather than filling the log at frame rate.
            self.error = f"{type(exc).__name__}: {exc}"
            self.net = None
            return []
        self.last_ms = 1000.0 * (time.monotonic() - t0)
        self.runs += 1
        if self.rot:
            # Undo the turn: the same rotation applied (4 - k) more times
            # brings a point back, measured on the ROTATED frame's shape.
            h, w = upright.shape[:2]
            out = [Recognition(r.label, r.confidence,
                               turn_box(r.box, w, h, 4 - self.rot))
                   for r in out]
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
            if 0 <= idx < len(self.labels):
                out.append(Recognition(self.labels[idx], score, box))
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

        # v5 carries an objectness column, v8 does not. Told apart by which
        # one makes the class count match the label list - and failing that,
        # by assuming v5, which is what every export of an unusual class
        # count in the wild has turned out to be.
        n = len(self.labels)
        if wide == n + 5:                    # v5: cx cy w h obj cls...
            cls = pred[:, 5:]
            conf = pred[:, 4] * cls.max(axis=1)
        elif wide == n + 4:                  # v8: cx cy w h cls...
            cls = pred[:, 4:]
            conf = cls.max(axis=1)
        else:
            raise ValueError(
                f"output has {wide} columns, which fits neither {n + 4} nor "
                f"{n + 5} for {n} labels - put a .names file with one class "
                f"per line beside the model")

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
            out.append(Recognition(self.labels[int(ids[i])], conf[i],
                                   (x, y, bw_, bh_)))
        return out


def _overlaps(a, b, min_frac=0.4):
    """Do these two boxes describe the same thing in the same place?"""
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter >= min_frac * min(aw * ah, bw * bh)


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
