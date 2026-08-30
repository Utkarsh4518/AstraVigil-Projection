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
    yolo export model=yolov8n.pt format=onnx imgsz=320 opset=12
"""
import argparse
import os
import sys
import time
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

DEST_DIR = "data/models"

# The yolov5 v7.0 release ships ready-made ONNX at three sizes, all verified
# reachable and all decoded by the recogniser without configuration. Larger
# is more accurate and proportionally slower, and since this runs on a
# throttle rather than per frame, a Pi with headroom can afford `s`.
RELEASE = "https://github.com/ultralytics/yolov5/releases/download/v7.0/"
SIZES = {
    "n": ("yolov5n.onnx", "4 MB, fastest, fewest names"),
    "s": ("yolov5s.onnx", "15 MB, noticeably better - try this first on a PC"),
    "m": ("yolov5m.onnx", "43 MB, best, slow on a Pi"),
}

# Tried in order. Mirrors move and repositories are renamed, so this is a list
# of candidates rather than one address, and a failure here is inconvenient
# rather than fatal - see the module docstring.
CANDIDATES = [
    ("yolov5n.onnx", RELEASE + "yolov5n.onnx"),
    ("yolov8n.onnx",
     "https://huggingface.co/Xenova/yolov8n/resolve/main/onnx/model.onnx"),
]

UA = {"User-Agent": "AstraVigil/1.0"}


def human(n):
    return f"{n / 1e6:.1f} MB" if n else "unknown size"


def download(url, dest, timeout=60):
    req = urllib.request.Request(url, headers=UA)
    tmp = dest + ".part"
    t0 = time.monotonic()
    with urllib.request.urlopen(req, timeout=timeout) as r:
        total = int(r.headers.get("Content-Length") or 0)
        got = 0
        with open(tmp, "wb") as f:
            while True:
                chunk = r.read(65536)
                if not chunk:
                    break
                f.write(chunk)
                got += len(chunk)
                if total:
                    pct = 100.0 * got / total
                    print(f"\r  {pct:5.1f}%  {human(got)} of {human(total)}",
                          end="", flush=True)
    print()
    if got < 100000:
        os.remove(tmp)
        raise OSError(f"only {got} bytes - that is not a model")
    os.replace(tmp, dest)
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
        print("For more names, fetch a bigger one: --size s --force")
    except Exception as exc:
        print(f"BROKEN         {type(exc).__name__}: {exc}")
        return 1
    return 0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
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
