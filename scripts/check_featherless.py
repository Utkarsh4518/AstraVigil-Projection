#!/usr/bin/env python3
"""Is the Featherless key present, in the place the kiosk reads, and valid?

    python3 scripts/check_featherless.py

Three separate questions, and they fail in different ways:

  IN THE RIGHT PLACE   A key exported in your shell works when you run the
                       pipeline by hand and does nothing when the icon is
                       double-clicked. A desktop launcher inherits the
                       graphical session's environment and reads neither
                       ~/.bashrc nor, reliably, ~/.profile. So this checks
                       ~/.astravigil.env - the file the launcher sources -
                       separately from the environment this script runs in.

  ENABLED              The key alone does not turn escalation on. run_dashboard
                       also needs --featherless, or it stays off by design:
                       escalation sends cropped imagery of the protected site
                       to a third party.

  VALID                FeatherlessClient.configured is bool(api_key) and
                       nothing more, so the dashboard's "featherless: on" line
                       proves only that a non-empty string was found. The one
                       way to know the key works is to spend a call on it,
                       which is what --live does.

Nothing here ever prints the key.
"""
import argparse
import json
import os
import sys
import urllib.error
import urllib.request

sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from astravigil.llm.featherless import (  # noqa: E402
    BASE_URL, DEFAULT_MODEL, RETRY_STATUSES, USER_AGENT, _headers)

ENV_FILE = os.environ.get("ASTRAVIGIL_ENV",
                          os.path.expanduser("~/.astravigil.env"))
OK, BAD, MEH = "  OK  ", " FAIL ", " WARN "


def mask(key):
    # Enough to tell two keys apart in a log, not enough to use.
    if not key:
        return "(none)"
    return f"{key[:6]}...{key[-4:]}  ({len(key)} chars)"


def read_env_file(path):
    """Parse the launcher's env file the way `set -a; . file` would.

    Deliberately not a shell: this must report what is in the file even when
    the file is malformed, which a shell would refuse to do.
    """
    found = {}
    if not os.path.exists(path):
        return None, found
    with open(path, encoding="utf-8", errors="replace") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line[7:].lstrip()
            if "=" not in line:
                found.setdefault("_problems", []).append(
                    f"line {n}: no '=' - the shell would set nothing here")
                continue
            k, v = line.split("=", 1)
            k = k.strip()
            v = v.strip()
            if len(v) >= 2 and v[0] == v[-1] and v[0] in "\"'":
                v = v[1:-1]
            found[k] = v
    return True, found


def check_key_string(key):
    """Shapes that give a confusing 401 rather than an obvious one."""
    problems = []
    if key != key.strip():
        problems.append("has leading or trailing whitespace")
    if key.startswith(("'", '"')) or key.endswith(("'", '"')):
        problems.append("still has quote characters in the value")
    if "\\n" in key or "\\r" in key:
        problems.append("contains a literal backslash-n - written with the "
                        "wrong printf quoting?")
    if key.lower().startswith("bearer "):
        problems.append("includes the word 'Bearer' - the client adds that")
    if " " in key:
        problems.append("contains a space")
    return problems


def _cloudflare_code(body):
    """Cloudflare's own error number, if this is Cloudflare talking.

    Worth separating out because these arrive as ordinary 403s and read like
    an answer about the key when they are not an answer from the API at all.
    """
    low = body.lower()
    if "error code:" not in low and "cloudflare" not in low:
        return None
    for code, what in (
            ("1010", "the request was refused on its client signature "
                     "(User-Agent), before it reached the API"),
            ("1020", "an access rule refused the request"),
            ("1015", "rate limited by the edge, not by the API"),
            ("1006", "the client IP is banned at the edge")):
        if code in body:
            return code, what
    return "?", "the request was stopped at the edge, not by the API"


def _request(url, key, payload=None, timeout=30.0):
    """Returns (status, body_text). Never raises for an HTTP status."""
    data = json.dumps(payload).encode() if payload is not None else None
    req = urllib.request.Request(
        url, data=data, headers=_headers(key),
        method="POST" if data else "GET")
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.status, r.read().decode("utf-8", "replace")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode("utf-8", "replace")
    except urllib.error.URLError as exc:
        return None, str(exc.reason)


def _explain(status, body, model):
    cf = _cloudflare_code(body)
    if cf is not None:
        code, what = cf
        return (f"HTTP {status}, but this is CLOUDFLARE error {code}, not "
                f"Featherless:\n         {what}.\n         The key was never "
                f"judged. Try a different FEATHERLESS_USER_AGENT "
                f"(currently {USER_AGENT!r}).")
    meaning = {
        400: "bad request - usually the model id, not the key",
        401: "the key was rejected - wrong, revoked, or not yet active",
        402: "the key is valid but the account cannot pay for this call",
        403: "the key is valid but not permitted to use this model",
        404: f"no such model: {model}",
        429: "rate limited - the key IS valid",
        503: "the model is busy on the provider's side - nothing is wrong "
             "with your setup",
    }.get(status, "")
    return f"HTTP {status}  {meaning}\n         {body[:300]}"


# Publishers whose vision models are the base instruction-tuned builds rather
# than someone's merge of them.
KNOWN_ORGS = ("qwen/", "meta-llama/", "google/", "mistralai/", "opengvlab/",
              "openbmb/", "llava-hf/", "microsoft/", "deepseek-ai/",
              "allenai/", "thudm/", "internlm/")

# Featherless carries a great many community fine-tunes, and a large share of
# the vision ones are roleplay merges. They will answer, and they are the wrong
# thing to hand a perimeter identification to: tuned for character voice, often
# with the refusal behaviour trained out. Rank them last rather than hide them,
# because on a given day they may be all that is free.
ROLEPLAY = ("uncensored", "heretic", "abliterated", "roleplay", "rp_",
            "eris", "nsfw", "erotic", "waifu", "smut", "horny", "chaotic")


def _score(model_id):
    low = model_id.lower()
    score = 0
    if low.startswith(KNOWN_ORGS):
        score += 4
    if "instruct" in low:
        score += 2
    if any(t in low for t in ROLEPLAY):
        score -= 8
    if any(t in low for t in ("finetuned", "merge", "-lora", "experimental")):
        score -= 2
    # Prefer the smaller builds: on a shared endpoint they are likelier to have
    # capacity, and quicker to answer when they do.
    for size, bonus in (("-3b", 2), ("-4b", 2), ("-7b", 2), ("-8b", 2),
                        ("-11b", 1), ("-12b", 1)):
        if size in low:
            score += bonus
            break
    return score


def vision_models(catalogue, limit=12):
    """Vision-language ids, best candidates first.

    Sorted rather than alphabetical because the alphabet puts community
    roleplay merges at the top, and an operator reading a suggestion list
    reasonably assumes the first entry is the recommendation.
    """
    try:
        rows = json.loads(catalogue).get("data", [])
    except ValueError:
        return []
    ids = [r.get("id", "") for r in rows if isinstance(r, dict)]
    hits = [i for i in ids
            if any(t in i.lower() for t in ("-vl", "vl-", "vision", "llava",
                                            "-vision", "internvl", "minicpm-v",
                                            "pixtral"))]
    return sorted(set(hits), key=lambda i: (-_score(i), len(i), i))[:limit]


def live_call(key, model, timeout=30.0):
    """Prove the key authenticates, then that the model is usable.

    Two calls, because one cannot separate them. A chat completion that fails
    with 403 might mean the key is bad, the model is not on the plan, or the
    request never arrived - and the operator needs to know which.
    """
    base = BASE_URL.rstrip("/")

    status, body = _request(f"{base}/models", key, timeout=timeout)
    if status is None:
        return False, f"could not reach {BASE_URL} ({body})"
    if status != 200:
        return False, "listing models: " + _explain(status, body, model)
    print(f"{OK} the key authenticates (GET /models -> 200)")
    catalogue = body

    payload = {
        "model": model,
        "messages": [{"role": "user", "content": "reply with the word ok"}],
        "max_tokens": 5,
        "temperature": 0.0,
    }
    status, body = _request(f"{base}/chat/completions", key, payload, timeout)
    if status is None:
        return False, f"could not reach {BASE_URL} ({body})"
    if status != 200:
        detail = f"calling {model}: " + _explain(status, body, model)
        if status in RETRY_STATUSES:
            # Transient, and upstream. The key, the file and the args are all
            # confirmed good by this point - reporting a fault here would send
            # the operator back to re-check settings that are already right.
            alts = vision_models(catalogue)
            if alts:
                detail += ("\n         Other vision models this key can see:"
                           + "".join(f"\n           {a}" for a in alts))
                detail += ("\n         Set one as a standby with:"
                           "\n           FEATHERLESS_FALLBACK_MODELS=<id>")
            return None, detail
        return False, detail
    try:
        reply = json.loads(body)["choices"][0]["message"]["content"].strip()
    except (KeyError, IndexError, ValueError) as exc:
        return False, f"unexpected reply shape ({exc})"
    return True, f"HTTP 200, model answered {reply!r}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--live", action="store_true",
                    help="actually call the API - the only real proof the key "
                         "works. Costs one tiny request.")
    ap.add_argument("--model", default=None,
                    help=f"model to test with (default {DEFAULT_MODEL})")
    ap.add_argument("--list-models", action="store_true",
                    help="print the vision models this key can see, then stop")
    ap.add_argument("--probe", type=int, nargs="?", const=6, default=0,
                    metavar="N",
                    help="try the top N vision models (default 6) and report "
                         "which actually answer right now, then stop. Costs "
                         "one tiny call each")
    args = ap.parse_args()

    fails = warns = 0

    if args.probe:
        key = (read_env_file(ENV_FILE)[1].get("FEATHERLESS_API_KEY")
               or os.environ.get("FEATHERLESS_API_KEY"))
        if not key:
            print("no key found")
            return 1
        base = BASE_URL.rstrip("/")
        status, body = _request(f"{base}/models", key)
        if status != 200:
            print(_explain(status, body, "-"))
            return 1

        wanted = args.model or os.environ.get("FEATHERLESS_MODEL",
                                              DEFAULT_MODEL)
        cands = vision_models(body, limit=args.probe)
        if wanted not in cands:
            cands.insert(0, wanted)

        print(f"\ntrying {len(cands)} model(s) - one small call each\n")
        working = []
        for m in cands:
            st, bd = _request(f"{base}/chat/completions", key, {
                "model": m,
                "messages": [{"role": "user", "content": "say ok"}],
                "max_tokens": 5, "temperature": 0.0})
            if st == 200:
                working.append(m)
                print(f"{OK} {m}")
            elif st in RETRY_STATUSES:
                print(f"{MEH} {m}  busy ({st})")
            else:
                short = (bd or "")[:90].replace("\n", " ")
                print(f"{BAD} {m}  HTTP {st}  {short}")

        print()
        if not working:
            print("nothing answered. Every candidate is busy or refused - "
                  "wait and try again; this is capacity, not configuration.")
            return 1
        print(f"{len(working)} model(s) answering. Put the first in "
              "FEATHERLESS_MODEL and keep the rest as standbys:\n")
        print(f"  FEATHERLESS_MODEL={working[0]}")
        if len(working) > 1:
            print("  FEATHERLESS_FALLBACK_MODELS="
                  + ",".join(working[1:]))
        print("\nAdd those to ~/.astravigil.env, then restart the kiosk.")
        return 0

    if args.list_models:
        key = (read_env_file(ENV_FILE)[1].get("FEATHERLESS_API_KEY")
               or os.environ.get("FEATHERLESS_API_KEY"))
        if not key:
            print("no key found")
            return 1
        status, body = _request(f"{BASE_URL.rstrip('/')}/models", key)
        if status != 200:
            print(_explain(status, body, "-"))
            return 1
        found = vision_models(body, limit=60)
        print(f"\n{len(found)} vision-capable model id(s) visible:\n")
        for m in found:
            print(f"  {m}")
        print()
        return 0

    # ---------------------------------------------------------- the file
    print(f"\nenv file   : {ENV_FILE}")
    exists, parsed = read_env_file(ENV_FILE)
    file_key = parsed.get("FEATHERLESS_API_KEY")

    if not exists:
        print(f"{MEH} not found - the kiosk icon will NOT see a key.")
        print("       Create it with:")
        print("           printf 'FEATHERLESS_API_KEY=sk-...\\n' "
              "> ~/.astravigil.env")
        print("           chmod 600 ~/.astravigil.env")
        warns += 1
    else:
        mode = os.stat(ENV_FILE).st_mode & 0o777
        print(f"{OK} exists, mode {mode:o}")
        if mode & 0o077:
            print(f"{MEH} readable by other users - chmod 600 {ENV_FILE}")
            warns += 1
        for p in parsed.get("_problems", []):
            print(f"{MEH} {p}")
            warns += 1
        if file_key:
            print(f"{OK} FEATHERLESS_API_KEY = {mask(file_key)}")
            for p in check_key_string(file_key):
                print(f"{BAD} the key {p}")
                fails += 1
        else:
            print(f"{BAD} no FEATHERLESS_API_KEY line in the file")
            fails += 1

        file_args = parsed.get("ASTRAVIGIL_ARGS", "")
        if "--featherless" in file_args:
            print(f"{OK} ASTRAVIGIL_ARGS carries --featherless")
        elif file_args:
            print(f"{BAD} ASTRAVIGIL_ARGS is set but has no --featherless, "
                  "so escalation stays off")
            fails += 1
        else:
            print(f"{BAD} no ASTRAVIGIL_ARGS line, so the launcher default is "
                  "used - and it has no --featherless")
            fails += 1

    # --------------------------------------------------- this environment
    env_key = os.environ.get("FEATHERLESS_API_KEY")
    print(f"\nthis shell : FEATHERLESS_API_KEY = {mask(env_key)}")
    if env_key and not file_key:
        print(f"{MEH} set here but not in {os.path.basename(ENV_FILE)} - "
              "running by hand will work and the ICON WILL NOT")
        warns += 1
    if env_key and file_key and env_key != file_key:
        print(f"{MEH} the two differ; the launcher uses the file, "
              "this shell uses its own")
        warns += 1

    # ---------------------------------------------------------- the call
    key = file_key or env_key
    model = args.model or os.environ.get("FEATHERLESS_MODEL", DEFAULT_MODEL)
    print(f"\nendpoint   : {BASE_URL}")
    print(f"model      : {model}")
    print(f"user-agent : {USER_AGENT}")

    if not args.live:
        print(f"\n{MEH} not checked against the API. Presence is not validity "
              "- re-run with --live to spend one call and know.")
        warns += 1
    elif not key:
        print(f"\n{BAD} no key anywhere to test")
        fails += 1
    else:
        good, detail = live_call(key, model)
        if good is None:
            print(f"\n{MEH} {detail}")
            print(f"\n{OK} your key, its placement and your args are all "
                  "confirmed good - this is the provider, not you. Retry, or "
                  "set a fallback model.")
            warns += 1
        else:
            print(f"\n{OK if good else BAD} {detail}")
            if not good:
                fails += 1

    print()
    if fails:
        print(f"{fails} problem(s) to fix"
              + (f", {warns} warning(s)" if warns else ""))
    elif warns:
        print(f"no failures, {warns} warning(s)")
    else:
        print("all good - the key is present, in the file the launcher reads, "
              "and accepted by the API")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
