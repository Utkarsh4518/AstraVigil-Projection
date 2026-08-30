#!/usr/bin/env python3
"""Fetch a COCO object-detection model for the optical camera.

The recogniser needs one weights file in data/models/ and works with any
COCO-trained YOLO exported to ONNX - v5 or v8, any input size. It is entirely
optional: without it the system behaves exactly as it did before, minus the
object names on the optical pane.

    python scripts/fetch_object_model.py                  # try the defaults
    python scripts/fetch_object_model.py --url <onnx url> # somewhere else
    python scripts/fetch_object_model.py --check          # what is installed

If your network will not reach any of these, this step is a file copy and
nothing more: put any COCO YOLO .onnx into data/models/ by whatever means you
like. There is no manifest, no checksum file and no naming convention - the
recogniser takes the first .onnx it finds.

To export one yourself on a machine that has torch:

    pip install ultralytics
    yolo export model=yolov8n.pt format=onnx imgsz=640 opset=12

A MODEL THAT KNOWS WHAT A DRONE IS

The models above are COCO, and COCO has eighty everyday objects with no
quadcopter among them. Shown a drone, they answer with whichever class they
find least unlike it - "bowl" and "frisbee" are the usual ones - so their
names are treated as evidence about furniture and as no evidence at all about
aircraft. Nothing here will ever announce a drone, and that is deliberate.

To use one trained on drones, put the .onnx in data/models/ and a .names file
beside it with the same stem, one class per line:

    data/models/drone.onnx
    data/models/drone.names     ->  drone
                                    bird
                                    balloon

Any class count works; the decoder reads the label file rather than assuming
COCO. Then point at it explicitly so the COCO models do not win the
preference order:

    ASTRAVIGIL_OBJECT_MODEL=data/models/drone.onnx
"""
import argparse
import os
import sys
import time
import urllib.error

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEST_DIR = "data/models"

# The addresses and the downloader both live in the module, because the rig
# fetches this by itself on first run and there must be exactly one answer to
# "where does the model come from". This script is for choosing a different
# size, re-fetching, and testing what is installed.
from astravigil.detection.objects import (       # noqa: E402
    RELEASE, SIZES, download_model)

# A model that has actually seen a quadcopter. YOLOv5, 416px input, ONE
# class - so it can say "drone", which no COCO model can. Verified reachable
# and verified to load and decode through cv2.dnn.
DRONE_URL = ("https://huggingface.co/engdarwish/drone-detection-yolov5/"
             "resolve/main/best.onnx")
DRONE_CLASSES = ["drone"]

# Tried in order. Mirrors move and repositories are renamed, so this is a list
# of candidates rather than one address, and a failure here is inconvenient
# rather than fatal - see the module docstring.
CANDIDATES = [
    ("yolov5s.onnx", RELEASE + "yolov5s.onnx"),
    ("yolov5n.onnx", RELEASE + "yolov5n.onnx"),
    ("yolov8n.onnx",
     "https://huggingface.co/Xenova/yolov8n/resolve/main/onnx/model.onnx"),
]

def human(n):
    return f"{n / 1e6:.1f} MB" if n else "unknown size"


def download(url, dest, timeout=60):
    def progress(got, total):
        if total:
            print(f"\r  {100.0 * got / total:5.1f}%  {human(got)} of "
                  f"{human(total)}", end="", flush=True)

    t0 = time.monotonic()
    got = download_model(url, dest, timeout=timeout, progress=progress)
    print()
    return got, time.monotonic() - t0


def check():
    from astravigil.detection.objects import ObjectRecogniser
    r = ObjectRecogniser()
    st = r.stats()
    if not st["available"]:
        print(f"NOT INSTALLED  {st['error']}")
        return 1
    print(f"INSTALLED      {st['model']}  ({st['kind']})")
    try:
        import numpy as np
        from astravigil.detection import objects as O
        img = np.full((480, 640, 3), 90, np.uint8)
        t0 = time.monotonic()
        found = r.detect(img)
        ms = 1000 * (time.monotonic() - t0)
        if r.net is None:
            print(f"BROKEN         {r.error}")
            return 1
        print(f"RUNS           {ms:.0f} ms on a {640}x{480} frame, "
              f"{len(found)} objects on a blank one")
        print(f"CLASSES        {len(O.COCO)} COCO labels")
        print(f"INPUT          {r.size}px, probed from the model itself")
        print(f"\nIt will run every {r.every_n} optical frames "
              f"on the live rig.")
        if "yolov5n" in (st["model"] or ""):
            print("This is the SMALL model. It finds nothing at all once one "
                  "object fills the frame -")
            print("fetch the better one:  "
                  "python scripts/fetch_object_model.py --size s --force")
    except Exception as exc:
        print(f"BROKEN         {type(exc).__name__}: {exc}")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--drone", action="store_true",
                    help="also fetch a drone-trained model and run it "
                         "alternately with the COCO one")
    ap.add_argument("--size", choices=sorted(SIZES),
                    help="which model to fetch: "
                         + "; ".join(f"{k} = {v[1]}"
                                     for k, v in sorted(SIZES.items())))
    ap.add_argument("--url", help="download this instead of the defaults")
    ap.add_argument("--name", default="yolov8n.onnx",
                    help="filename to save --url as")
    ap.add_argument("--dest", default=DEST_DIR)
    ap.add_argument("--check", action="store_true",
                    help="report what is installed and whether it runs")
    ap.add_argument("--force", action="store_true",
                    help="download even if a model is already present")
    args = ap.parse_args()

    if args.check:
        return check()

    if args.drone:
        os.makedirs(args.dest, exist_ok=True)
        dest = os.path.join(args.dest, "drone.onnx")
        if os.path.exists(dest) and not args.force:
            print(f"already installed: {dest}")
        else:
            print(f"trying {DRONE_URL}")
            try:
                size, secs = download(DRONE_URL, dest)
            except (urllib.error.URLError, OSError) as exc:
                print(f"  failed: {getattr(exc, 'reason', exc)}")
                return 1
            print(f"saved {dest}  ({human(size)} in {secs:.0f} s)")
        names = os.path.splitext(dest)[0] + ".names"
        with open(names, "w", encoding="utf-8") as fh:
            fh.write("\n".join(DRONE_CLASSES) + "\n")
        print(f"wrote {names}: {', '.join(DRONE_CLASSES)}")
        print()
        print("Nothing else to set. It is picked up on sight and runs on "
              "alternate")
        print("passes with the COCO model, so it costs no extra time per "
              "pass.")
        print("Restart the dashboard and the caption will name both.")
        return 0

    os.makedirs(args.dest, exist_ok=True)
    existing = [f for f in sorted(os.listdir(args.dest))
                if f.endswith((".onnx", ".pb"))]
    if existing and not args.force:
        print(f"already installed: {', '.join(existing)}")
        print("use --force to download anyway, or --check to test it")
        return 0

    if args.size:
        name = SIZES[args.size][0]
        todo = [(name, RELEASE + name)]
    elif args.url:
        todo = [(args.name, args.url)]
    else:
        todo = CANDIDATES
    for name, url in todo:
        dest = os.path.join(args.dest, name)
        print(f"trying {url}")
        try:
            size, secs = download(url, dest)
        except (urllib.error.URLError, OSError) as exc:
            print(f"  failed: {getattr(exc, 'reason', exc)}")
            continue
        print(f"saved {dest}  ({human(size)} in {secs:.0f} s)")
        return check()

    print("\nNone of the candidates could be fetched.")
    print("This is optional - the system runs without it.")
    print(f"To install by hand, copy any COCO YOLO .onnx into {args.dest}/")
    return 1


if __name__ == "__main__":
    sys.exit(main())
