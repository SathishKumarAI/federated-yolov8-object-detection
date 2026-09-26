"""Decode random shard images and check their labels against BDD100K's own JSON.

``pipeline/validate.py`` checks the fleet's *structure* — every image has a label,
slices are disjoint, the holdout is intact — and says so in its own docstring:
"Cheap: no image decoding." So four ways a shard can be wrong are outside it, and
every one of them produces a run that completes:

* a truncated JPEG that decodes to a grey band, or one that is not 1280×720
* a label whose boxes are normalised against different dimensions than the image has
* a label belonging to a *different* image — the name-keyed link is the only thing
  tying the two together, and nothing has ever checked it against the source
* a shard image that is a full copy rather than a hardlink, quietly costing disk

This samples instead of sweeping, because decoding 16 800 JPEGs to learn the same
thing is the expensive way to be sure. A seed makes any sample reproducible.

    python -m pipeline.spotcheck              # 40 images, seed 0
    python -m pipeline.spotcheck -n 200 --seed 3 --json report.json

Read-only, like the validator: it names what is wrong and repairs nothing.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

import ijson
from PIL import Image

from . import holdout, paths

#: The 13 classes in the order the data yaml lists them. BDD's own category strings
#: differ for two of them, and `lane` / `drivable area` carry poly2d rather than
#: box2d and so never become a box. Getting this map wrong reads as "the shard has
#: one box too many", which is exactly how this file's first run was wrong.
NAMES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle",
         "traffic light", "traffic sign", "trailer", "other person", "other vehicle"]
CID = {n: i for i, n in enumerate(NAMES)}
CID.update({"bike": CID["bicycle"], "motor": CID["motorcycle"]})

WIDTH, HEIGHT = 1280, 720       # every BDD100K frame; anything else is a resized copy
TOL = 1e-3                      # a label writes ~6 decimals; this is far looser


def _sample(root: Path, n: int, seed: int) -> list[tuple[str, str, str]]:
    pool = []
    for shard in sorted(root.glob("batch_*")):
        for split in ("train", "val"):
            listing = shard / f"{split}.txt"
            if listing.exists():
                pool += [(shard.name, split, name) for name in listing.read_text().split()]
    return random.Random(seed).sample(pool, min(n, len(pool))), len(pool)


def _cache_index() -> dict[str, Path]:
    """basename -> the kagglehub original.

    Indexed recursively: the Kaggle mirror nests ``train/`` into ``trainA/``,
    ``trainB/``, ... so a flat lookup finds 1 160 of 70 000 and reports every train
    image as missing from the cache. ``stages.py`` documents the same nesting.
    """
    pool = paths.find_pool()
    if pool is None:
        return {}
    out: dict[str, Path] = {}
    for f in pool.rglob("*.jpg"):
        out.setdefault(f.name, f)
    return out


def _truth(wanted: set[str]) -> dict[str, dict]:
    """The BDD annotation record for exactly these images, streamed.

    The train label JSON is 1.45 GB; ijson keeps the memory flat and the scan stops
    as soon as every sampled name has been seen.
    """
    found: dict[str, dict] = {}
    for js in paths.find_label_jsons():
        if not wanted - found.keys():
            break
        with open(js, "rb") as fh:
            for rec in ijson.items(fh, "item"):
                if rec.get("name") in wanted and rec["name"] not in found:
                    found[rec["name"]] = rec
                    if not wanted - found.keys():
                        break
    return found


def _boxes_from_bdd(rec: dict, w: int, h: int) -> list[tuple]:
    out = []
    for lab in rec.get("labels", []):
        b, cat = lab.get("box2d"), lab.get("category")
        if not b or cat not in CID:
            continue
        x1, y1, x2, y2 = (float(b[k]) for k in ("x1", "y1", "x2", "y2"))
        out.append((CID[cat], ((x1 + x2) / 2) / w, ((y1 + y2) / 2) / h,
                    (x2 - x1) / w, (y2 - y1) / h))
    return out


def check(name: str, shard: str, split: str, root: Path,
          cache: dict[str, Path], truth: dict[str, dict], held: set[str]) -> dict:
    """Everything checkable about one image. ``problems`` empty means it is sound."""
    r: dict = {"image": f"{shard}/{split}/{name}", "problems": []}
    say = r["problems"].append
    img_p = root / shard / "images" / split / name
    lab_p = root / shard / "labels" / split / f"{name.rsplit('.', 1)[0]}.txt"

    try:
        with Image.open(img_p) as im:
            im.load()                      # full decode: a truncated JPEG fails here
            w, h, mode = im.width, im.height, im.mode
    except Exception as e:                 # noqa: BLE001 - any decode failure is the finding
        say(f"undecodable: {e.__class__.__name__}: {e}")
        return r
    r.update(size=f"{w}x{h}", mode=mode, bytes=img_p.stat().st_size)
    if (w, h) != (WIDTH, HEIGHT):
        say(f"not {WIDTH}x{HEIGHT}: {w}x{h}")
    if mode != "RGB":
        say(f"mode {mode}, not RGB")

    st = img_p.stat()
    r["nlink"] = st.st_nlink
    orig = cache.get(name)
    if orig is None:
        say("no original in the kagglehub cache")
    else:
        o = orig.stat()
        r["hardlinked"] = o.st_ino == st.st_ino and o.st_dev == st.st_dev
        if not r["hardlinked"]:
            say(f"a copy, not a hardlink ({o.st_size} vs {st.st_size} bytes)")

    rows = []
    for i, line in enumerate(lab_p.read_text().splitlines(), 1):
        parts = line.split()
        if not parts:
            continue
        if len(parts) != 5:
            say(f"label line {i}: {len(parts)} fields, not 5")
            continue
        c, x, y, bw, bh = int(float(parts[0])), *map(float, parts[1:])
        rows.append((c, x, y, bw, bh))
        if not 0 <= c < len(NAMES):
            say(f"line {i}: class id {c} outside 0..{len(NAMES) - 1}")
        if not all(0.0 <= v <= 1.0 for v in (x, y, bw, bh)):
            say(f"line {i}: coordinates not normalised: {(x, y, bw, bh)}")
        if bw <= 0 or bh <= 0:
            say(f"line {i}: zero-area box")
        if min(x - bw / 2, y - bh / 2) < -TOL or max(x + bw / 2, y + bh / 2) > 1 + TOL:
            say(f"line {i}: box leaves the frame")
    r["boxes"] = len(rows)

    rec = truth.get(name)
    if rec is None:
        say("not found in the BDD label JSON")
    else:
        expected = _boxes_from_bdd(rec, w, h)
        r["bdd_boxes"] = len(expected)
        if len(expected) != len(rows):
            say(f"{len(rows)} boxes in the shard, {len(expected)} in BDD's own annotation")
        matched = sum(
            1 for e in expected
            if any(g[0] == e[0] and max(abs(a - b) for a, b in zip(e[1:], g[1:])) <= TOL
                   for g in rows))
        r["matched"] = f"{matched}/{len(expected)}"
        if matched != len(expected):
            say(f"only {matched} of {len(expected)} BDD boxes reproduce in the label — "
                "this label may belong to another image")
        r["attrs"] = {k: str(v) for k, v in rec.get("attributes", {}).items()}

    if name in held:
        say("IN THE HOLDOUT — every holdout metric measured against it is self-referential")
    return r


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", type=int, default=40, help="images to sample (default 40)")
    ap.add_argument("--seed", type=int, default=0, help="makes the sample reproducible")
    ap.add_argument("--json", type=Path, help="write the full per-image record here")
    args = ap.parse_args(argv)

    root = paths.VEHICLE_BATCHES
    if not root.is_dir():
        print(f"no fleet on disk at {root}", file=sys.stderr)
        return 1
    sample, total = _sample(root, args.n, args.seed)
    print(f"{total} images in the fleet; spot-checking {len(sample)} at seed {args.seed}")

    cache = _cache_index()
    print(f"kagglehub pool: {len(cache)} jpgs")
    t0 = time.time()
    truth = _truth({name for _, _, name in sample})
    print(f"BDD ground truth: {len(truth)}/{len(sample)} in {time.time() - t0:.1f}s\n")

    held = holdout.names()
    results = [check(name, shard, split, root, cache, truth, held)
               for shard, split, name in sample]

    for r in results:
        print(f"{'FAIL' if r['problems'] else 'ok  '} {r['image']}  "
              f"{r.get('size', '-')} {r.get('mode', '')} nlink={r.get('nlink', '?')} "
              f"hardlink={r.get('hardlinked', '?')} "
              f"boxes={r.get('boxes', '?')}/bdd={r.get('bdd_boxes', '?')} "
              f"matched={r.get('matched', '-')}")
        for p in r["problems"]:
            print(f"      ! {p}")

    bad = [r for r in results if r["problems"]]
    boxes = sum(r.get("boxes", 0) for r in results)
    print(f"\n{len(results) - len(bad)}/{len(results)} clean, {boxes} boxes checked "
          f"against BDD's own annotation")
    if args.json:
        args.json.write_text(json.dumps(
            {"seed": args.seed, "n": len(results), "fleet": total, "results": results},
            indent=1))
        print(f"wrote {args.json}")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
