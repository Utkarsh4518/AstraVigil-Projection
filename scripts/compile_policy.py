#!/usr/bin/env python3
"""Write your site policy in English; get reviewable rules out.

    # local model, nothing leaves the machine (preferred)
    ollama serve && ollama pull qwen2.5:3b-instruct
    python3 scripts/compile_policy.py --from policy.txt

    # hosted, if you have no local model
    FEATHERLESS_API_KEY=... python3 scripts/compile_policy.py \
        --from policy.txt --backend hosted

    # no model at all - check and review a hand-written file
    python3 scripts/compile_policy.py --review configs/policy.json

The model runs HERE, once, and never again. What it produces is a JSON file
you read before it governs anything, and the runtime evaluator is arithmetic.
That is the entire reason a language model is allowed near this system: a
hallucination is a line in a file a human reviews, not a decision taken at
03:00 on a live perimeter.

Prefer --backend local. A site security policy naming your restricted areas
and their hours is exactly the document that should not be posted to a
third-party API.
"""
import argparse
import json
import os
import sys

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil.policy import SitePolicy, validate            # noqa: E402
from astravigil.policy.compile import CompileError, compile_policy  # noqa: E402

DEFAULT_OUT = "configs/policy.json"

EXAMPLE = """\
This is the northern apron of a cargo airport. The apron itself is the
protected area. A service lane runs along its western edge.

No unmanned aircraft of any kind are permitted over the apron at any time.
Ground vehicles are expected in the service lane between 06:00 and 22:00.
Birds are common here and are not a concern.
Nothing should remain stationary on the apron for more than two minutes.
Personnel are permitted on the apron only during shift hours, 06:00 to 22:00.
"""


def show(policy, problems):
    print(policy.describe())
    if problems:
        print("\nreview these before trusting it:")
        for p in problems:
            print(f"  ! {p}")
    else:
        print("\nno structural problems found")
    print("\nZONES STILL NEED COORDINATES. A model can name the apron; it "
          "cannot know\nwhere the apron is in your camera's pixels. Add a "
          "polygon (thermal pixel\ncoordinates) to each zone in the JSON, or "
          "rules scoped to it will never match.")


def main():
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--from", dest="src", help="text file of the policy prose")
    ap.add_argument("--text", help="policy prose inline")
    ap.add_argument("--example", action="store_true",
                    help="compile the built-in example, to see the shape")
    ap.add_argument("--review", help="validate and print an existing JSON "
                                     "policy; runs no model")
    ap.add_argument("--backend", default="local", choices=["local", "hosted"])
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=DEFAULT_OUT)
    args = ap.parse_args()

    if args.review:
        policy = SitePolicy.load(args.review)
        show(policy, validate(policy))
        return 0

    if args.example:
        text = EXAMPLE
    elif args.text:
        text = args.text
    elif args.src:
        with open(args.src, encoding="utf-8") as fh:
            text = fh.read()
    else:
        ap.error("give --from, --text, --example or --review")

    print(f"compiling with the {args.backend} backend...\n")
    try:
        policy, problems = compile_policy(text, backend=args.backend,
                                          model=args.model)
    except CompileError as exc:
        print(f"could not compile: {exc}", file=sys.stderr)
        print("\nYou can also write configs/policy.json by hand and check it "
              "with --review;\nthe model is a convenience for authoring, not "
              "a dependency.", file=sys.stderr)
        return 1

    show(policy, problems)
    policy.save(args.out)
    print(f"\nwritten: {args.out}")
    print(f"use it with:\n  python3 scripts/run_dashboard.py "
          f"--policy {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
