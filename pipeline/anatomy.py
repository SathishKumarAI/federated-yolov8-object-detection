"""What actually travels between the server and the vehicles, read off a real checkpoint.

The dashboard can prove a federation *moved*; nothing on it says what moved. This module
opens the newest global checkpoint and reports the thing itself: how many tensors, how
many numbers, which of them are convolution kernels and which are BatchNorm bookkeeping,
what the first convolution looks like as 32 little RGB pictures, and what the detection
head's per-class bias is for each of the thirteen classes.

Three rules, each of which is the reason this file exists rather than a paragraph of prose:

1. **Every number is computed from a file on disk, at request time.** No constants, no
   transcribed table. The payload carries the checkpoint's name, size and mtime so the
   reader can open the same file.
2. **No checkpoint means no numbers.** `payload()` returns `available: False` and the
   reason. It never returns zeros, because a zero here reads as a measurement -- the
   failure mode this whole repo is a catalogue of.
3. **torch is imported inside the function that needs it.** The pipeline test job installs
   `pytest` and `pyyaml` only; a module-scope `import torch` fails the entire suite on a
   bare interpreter. That is not hypothetical -- it broke CI on 2026-09-26.

What it does NOT do: aggregate arithmetic. Whether the published aggregate really is the
weighted mean of what the clients sent is `pipeline/roundtrip.py`'s question, and this
module forwards that verdict rather than computing its own. Two implementations of one
identity is one implementation too many, and the second one is always the one that is
wrong without anybody noticing.

    python -m pipeline.anatomy                       # the newest global checkpoint
    python -m pipeline.anatomy --json
    python -m pipeline.anatomy --checkpoint PATH      # any .pt built from yolov8s-13
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import measurements, paths

#: How many kernels of the first convolution are rendered, and how many bins a
#: distribution is reduced to. The first conv is 32x3x3x3 = 864 numbers, so it travels
#: whole; every other convolution is up to 2.4 M numbers and is summarised, never sent.
#: A panel that shipped a megabyte of floats to draw 200 px of picture would be a
#: different kind of dishonest.
HIST_BINS = 14

#: state_dict key suffix -> what that tensor is, in words a reader who does not know what
#: a tensor is can act on. Order matters: the first match wins, and the integer counter
#: has to be tested before the two float statistics it sits beside.
KINDS = [
    ("bn_counter", "BatchNorm step counters",
     "One integer per normalisation layer, counting the batches it has seen. Not learned "
     "-- it climbs on its own, which is why the checksum this project trusts excludes it."),
    ("bn_stats", "BatchNorm running statistics",
     "The mean and variance of what this layer has seen, measured during training rather "
     "than learned. They encode the conditions a vehicle drove in, which is exactly what "
     "averaging across a weather-partitioned fleet blurs."),
    ("bn_affine", "BatchNorm scale and shift",
     "Two learned numbers per channel, applied after normalising. Tiny in count, and the "
     "part of normalisation that gradient descent actually moves."),
    ("conv", "Convolution kernels",
     "The learned filters. Each one slides over its input looking for one pattern -- an "
     "edge, a texture, later a wheel or a tail light. This is where nearly every number "
     "in the model lives."),
    ("head_cls_bias", "Detection head class biases",
     "One number per class per detection scale: how ready the model is to call something "
     "that class before it has looked at any pixels. Thirteen numbers, three times over."),
    ("head_box_bias", "Detection head box biases",
     "The same, for the 64 numbers each scale uses to place a box rather than name it."),
    ("other", "Everything else",
     "Tensors that match none of the patterns above. An empty row here is the expected "
     "state; a non-empty one means the architecture changed and this table needs a look."),
]
LABELS = {k: (label, what) for k, label, what in KINDS}


def classify(key: str, ndim: int) -> str:
    """Which `KINDS` row a state_dict key belongs to.

    Keyed on the name rather than on the shape, because the shapes collide: a BatchNorm
    scale and a head bias are both 1-D vectors of a few dozen floats, and calling the
    second one normalisation would hide the one place in this model where a weight has a
    name a person recognises.
    """
    if key.endswith("num_batches_tracked"):
        return "bn_counter"
    if key.endswith(("running_mean", "running_var")):
        return "bn_stats"
    if ".bn." in key:
        return "bn_affine"
    if ndim == 4:
        return "conv"
    if key.endswith(".bias"):
        # `cv3` is the classification branch of YOLOv8's head and `cv2` the box branch
        # (ultralytics/nn/modules/head.py). The class biases are the 13 numbers the
        # warm start could only fill nine of, so they are worth their own row.
        return "head_cls_bias" if ".cv3." in key else "head_box_bias"
    return "other"


def newest_checkpoint() -> Path | None:
    """The most recently written ``global_*.pt``, or None.

    Newest by mtime rather than by round number: `global_last.pt` and
    `global_round_N.pt` are written in the same breath, and a previous longer run leaves
    higher round numbers behind -- the trap `holdout.checkpoints` documents at length.
    """
    d = paths.PROJECT / "checkpoints"
    files = [p for p in d.glob("global_*.pt") if p.is_file()] if d.is_dir() else []
    return max(files, key=lambda p: p.stat().st_mtime) if files else None


def _histogram(values, bins: int) -> dict:
    """Counts per bin, plus the real edges. `values` is a 1-D float tensor."""
    lo, hi = float(values.min()), float(values.max())
    if hi <= lo:
        return {"lo": lo, "hi": hi, "counts": [int(values.numel())], "edges": [lo, hi]}
    counts = [int(c) for c in values.histc(bins=bins, min=lo, max=hi)]
    step = (hi - lo) / bins
    return {"lo": lo, "hi": hi, "counts": counts,
            "edges": [lo + i * step for i in range(bins + 1)]}


def _state_dict(path: Path):
    """The 355 tensors in a global checkpoint, and which slot held them.

    Returns ``(state_dict, meta)``. The slot matters and is reported rather than
    smoothed over: ultralytics' loader takes ``ckpt["ema"] or ckpt["model"]``, and on
    the global checkpoints this server writes the ema slot is **None** -- measured, not
    assumed. So "the fp16 EMA" is true of a client's ``best.pt`` and only half true of
    these: they are fp16, and they are the model slot.
    """
    import torch                                    # see rule 3 in the module docstring

    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    slot = "ema" if ckpt.get("ema") is not None else "model"
    module = ckpt.get(slot)
    if module is None:
        raise ValueError(f"{path.name} holds neither an 'ema' nor a 'model' entry")
    sd = module.state_dict() if hasattr(module, "state_dict") else module
    return sd, {
        "slot": slot,
        "ema_slot_empty": ckpt.get("ema") is None,
        "ultralytics_version": str(ckpt.get("version") or "unknown"),
        "written": str(ckpt.get("date") or "unknown"),
        # The class names the file itself carries. yolov8s-13.yaml declares none, so
        # ultralytics defaults them to "0".."12" and any table named from the
        # checkpoint comes out labelled by index -- the trap holdout.py documents.
        "names_are_indices": list((getattr(module, "names", None) or {}).values())
                             == [str(i) for i in range(13)],
    }


#: (path, mtime) -> inventory. Loading 22 MB through torch costs about a second, and the
#: page re-asks whenever a round lands. Keyed on the mtime so a checkpoint REWRITTEN in
#: place -- which is exactly what `global_last.pt` is, every round -- invalidates itself
#: rather than freezing the page on round 1 while the run continues.
_CACHE: dict[tuple[str, float], dict] = {}


def inventory(path: Path) -> dict:
    """Everything the page shows about one checkpoint. Computed, never quoted."""
    key = (str(path), path.stat().st_mtime)
    if key in _CACHE:
        return _CACHE[key]
    inv = _inventory(path)
    _CACHE.clear()                  # one entry: nothing here needs a history
    _CACHE[key] = inv
    return inv


def _inventory(path: Path) -> dict:
    import torch

    sd, meta = _state_dict(path)
    kinds: dict[str, dict] = {}
    file_bytes = wire_bytes = values = 0
    for key, t in sd.items():
        row = kinds.setdefault(classify(key, t.ndim),
                               {"tensors": 0, "values": 0, "file_bytes": 0,
                                "wire_bytes": 0, "example": key, "shape": list(t.shape)})
        row["tensors"] += 1
        row["values"] += t.numel()
        row["file_bytes"] += t.numel() * t.element_size()
        # What get_weights actually puts on the wire is `t.cpu().numpy()` of the LIVE
        # model, whose floats are fp32. The file's own fp16 is the checkpoint's doing,
        # not the transport's, so both are reported and neither is called the other.
        row["wire_bytes"] += t.numel() * (4 if t.dtype.is_floating_point else t.element_size())
    for row in kinds.values():
        values += row["values"]
        file_bytes += row["file_bytes"]
        wire_bytes += row["wire_bytes"]

    order = [k for k, _, _ in KINDS if k in kinds]
    rows = [{"key": k, "label": LABELS[k][0], "what": LABELS[k][1], **kinds[k]}
            for k in order]

    conv0 = next((k for k, t in sd.items() if t.ndim == 4), None)
    stats = [t.float().flatten() for k, t in sd.items() if k.endswith("running_mean")]
    var = [t.float().flatten() for k, t in sd.items() if k.endswith("running_var")]
    counters = sorted({int(t) for k, t in sd.items() if k.endswith("num_batches_tracked")})

    return {
        "tensors": len(sd),
        "values": values,
        "file_tensor_bytes": file_bytes,
        "wire_bytes": wire_bytes,
        "dtypes": {str(d): n for d, n in
                   sorted(((d, sum(1 for t in sd.values() if t.dtype == d))
                           for d in {t.dtype for t in sd.values()}),
                          key=lambda x: -x[1])},
        "kinds": rows,
        "first_conv": _first_conv(sd, conv0) if conv0 else None,
        "batchnorm": {
            "layers": sum(1 for k in sd if k.endswith("num_batches_tracked")),
            "running_mean": _histogram(torch.cat(stats), HIST_BINS) if stats else None,
            "running_var": _histogram(torch.cat(var), HIST_BINS) if var else None,
            "counter_values": counters[:8],
            "counters_all_zero": counters == [0],
        },
        "head": _head(sd),
        **meta,
    }


def _first_conv(sd, key: str) -> dict:
    """The first convolution as pictures: 32 kernels of 3x3x3, one RGB pixel per tap.

    This is the only layer whose weights are literally visible. Its three input channels
    *are* red, green and blue, so a kernel is a 3x3 colour image and a person can see
    that the model has learned edges and colour opponents without being told what a
    tensor is. Every deeper layer's input channels are other channels, and painting
    three of 512 as RGB would be a decoration rather than a measurement -- which is why
    only this one is sent.

    Each kernel is scaled to its own min..max for display, and the raw range travels
    beside it so the page can print the actual numbers. Scaling per kernel is a display
    choice with a real consequence -- a kernel whose weights are all tiny looks as
    strong as one whose weights are large -- so the page says so.
    """
    w = sd[key].float()
    out, cin, kh, kw = w.shape
    kernels = []
    for i in range(out):
        k = w[i]
        lo, hi = float(k.min()), float(k.max())
        span = (hi - lo) or 1.0
        rgb = []
        for y in range(kh):
            for x in range(kw):
                px = [round(float((k[c, y, x] - lo) / span), 3) if c < cin else 0.0
                      for c in range(3)]
                rgb.append(px)
        kernels.append({"min": round(lo, 5), "max": round(hi, 5), "rgb": rgb})
    return {"key": key, "shape": [out, cin, kh, kw], "kernels": kernels,
            "values": int(w.numel())}


def _head(sd) -> dict:
    """The detection head's per-class bias, one row per scale, named from the label set.

    Named from `measurements.BDD_CLASSES`, **not** from the checkpoint: yolov8s-13.yaml
    declares no `names:`, so the file's own list is "0".."12" and a table named from it
    is labelled by index. `holdout.class_names_on_disk` documents the same trap.

    `cold_values` is the part worth looking at. `warm_start_head` copied nine of the
    thirteen rows out of COCO; the other four began from one shared value, so classes
    still holding the *identical* number have not moved off their initialisation at this
    file's precision. The page groups them; the exact values are what travel, because
    "identical" is a bit-level question rather than a tolerance and deciding it here
    would hide the numbers that decide it.

    The branch is selected by name (`cv3` classifies, `cv2` places boxes) and then by
    length: `model.22.cv3.N.0.bn.bias` also ends in `.bias`, and it is excluded by having
    128 entries rather than 13. No BatchNorm layer in yolov8s has 13 channels.
    """
    names = measurements.BDD_CLASSES
    warm = set(measurements.WARM_STARTED_CLASSES)
    cold = [n for n in names if n not in warm]
    scales = []
    for key in [k for k in sd if ".cv3." in k and k.endswith(".bias")]:
        # NOT rounded. These 39 numbers are compared for exact equality by the page, and
        # rounding to 6 places turned -9.7265625 into -9.726562 -- which invented an
        # inequality between two classes that hold the identical stored value.
        vals = [float(v) for v in sd[key].float()]
        if len(vals) != len(names):
            continue
        scales.append({
            "key": key,
            "values": vals,
            "cold_values": {n: vals[names.index(n)] for n in cold},
        })
    return {"classes": names, "cold_start": cold,
            "scales": scales}


def exchange(inv: dict, vehicles: int, rounds: int) -> dict:
    """Bytes on the wire for a fleet, from the tensor sizes just measured.

    Tensor payload only. gRPC framing, the protobuf field headers around each of the 355
    arrays and TLS are all excluded and said to be excluded: they are real and they are
    small, and an estimate that quietly folded a guess at them in would stop being a
    measurement. `num_examples` -- the one integer that also travels up, and the weight
    FedAvg divides by -- is 8 bytes and is why the up direction is not identical to down.
    """
    one = inv["wire_bytes"]
    return {
        "down_per_vehicle": one,
        "up_per_vehicle": one + 8,
        "per_round": vehicles * (2 * one + 8),
        "total": vehicles * rounds * (2 * one + 8),
        "vehicles": vehicles,
        "rounds": rounds,
        "arrays": inv["tensors"],
        "images_moved": 0,
        "excluded": "gRPC framing, per-array protobuf headers and TLS. Tensor payload only.",
    }


def roundtrip_verdict() -> dict:
    """`pipeline/roundtrip.py`'s answer, or why there is not one.

    Imported here rather than at module scope, and its absence handled rather than
    raised: this module ships the panel that renders the identity, and a checkout where
    the check does not exist must say that instead of showing an empty table that reads
    as a failed federation.
    """
    try:
        from . import roundtrip
    except ImportError as e:
        return {"available": False,
                "reason": f"pipeline/roundtrip.py is not in this checkout ({e}), so the "
                          f"weighted-mean identity cannot be checked here"}
    rounds = roundtrip.collect()
    if not rounds:
        return {"available": False,
                "reason": "no run log on disk records an aggregation, so there is no "
                          "round to check"}
    if not roundtrip.checkable(rounds):
        return {"available": False,
                "reason": "this run's client logs carry no num_examples, so FedAvg's own "
                          "weights are unknown and the identity cannot be tested. "
                          "Re-run to check it."}
    ok, _ = roundtrip.check(rounds)
    return {"available": True, "ok": ok, "tolerance": roundtrip.TOLERANCE,
            "rounds": [{"round": r.number, "aggregate": r.aggregate,
                        "weighted_mean": r.weighted_mean, "deviation": r.deviation,
                        "clients": len(r.sent), "num_examples": [n for _, n in r.sent],
                        "sent": [c for c, _ in r.sent], "received": r.received}
                       for r in rounds]}


def payload(path: Path | None = None, vehicles: int = 6, rounds: int = 6) -> dict:
    """What `/api/anatomy` serves.

    `available: False` with a reason is a first-class answer, not an error: a clean
    checkout has no checkpoint, and the honest page says so rather than rendering a
    model made of zeros.
    """
    ckpt = path or newest_checkpoint()
    if ckpt is None:
        return {"available": False,
                "reason": f"no global_*.pt in {(paths.PROJECT / 'checkpoints').name}/ -- "
                          f"run the federate stage and one is written every round",
                "roundtrip": roundtrip_verdict()}
    try:
        inv = inventory(ckpt)
    except Exception as e:                      # a truncated or foreign .pt, loudly
        return {"available": False,
                "reason": f"{ckpt.name} could not be read as a YOLO checkpoint: "
                          f"{type(e).__name__}: {e}",
                "roundtrip": roundtrip_verdict()}
    stat = ckpt.stat()
    # Relative to the repo when it is inside it, which is the normal case and the only
    # form worth putting on a page. An explicit --checkpoint elsewhere keeps its own path.
    where = ckpt.parent
    shown = (where.relative_to(paths.REPO) if where.is_relative_to(paths.REPO)
             else where).as_posix()
    return {
        "available": True,
        "checkpoint": {"name": ckpt.name, "bytes": stat.st_size, "mtime": stat.st_mtime,
                       "dir": shown},
        "model": inv,
        "exchange": exchange(inv, vehicles, rounds),
        "roundtrip": roundtrip_verdict(),
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--checkpoint", type=Path, default=None,
                    help="read this file instead of the newest global_*.pt")
    ap.add_argument("--vehicles", type=int, default=6)
    ap.add_argument("--rounds", type=int, default=6)
    args = ap.parse_args(argv)

    p = payload(args.checkpoint, args.vehicles, args.rounds)
    if args.json:
        print(json.dumps(p, indent=1))
        return 0
    if not p["available"]:
        print(f"no anatomy: {p['reason']}")
        return 1
    m, c, x = p["model"], p["checkpoint"], p["exchange"]
    print(f"{c['dir']}/{c['name']}  {c['bytes'] / 1e6:.1f} MB on disk, "
          f"slot '{m['slot']}', written by ultralytics {m['ultralytics_version']}")
    print(f"{m['tensors']} state tensors, {m['values']:,} numbers, dtypes {m['dtypes']}")
    print(f"{'kind':<30} {'tensors':>8} {'%':>6} {'values':>12} {'%':>7}")
    for r in m["kinds"]:
        print(f"{r['label']:<30} {r['tensors']:>8} {100 * r['tensors'] / m['tensors']:>5.1f}% "
              f"{r['values']:>12,} {100 * r['values'] / m['values']:>6.3f}%")
    bn = [r for r in m["kinds"] if r["key"].startswith("bn_")]
    print(f"\nBatchNorm is {sum(r['tensors'] for r in bn)} of {m['tensors']} tensors "
          f"({100 * sum(r['tensors'] for r in bn) / m['tensors']:.1f}%) and "
          f"{100 * sum(r['values'] for r in bn) / m['values']:.3f}% of the numbers")
    print(f"first convolution {m['first_conv']['key']} {m['first_conv']['shape']}")
    for s in m["head"]["scales"]:
        print(f"class bias {s['key']}: "
              + ", ".join(f"{n} {v:+.2f}" for n, v in zip(m["head"]["classes"], s["values"])))
    print(f"\none direction is {x['down_per_vehicle']:,} bytes of tensor "
          f"({x['down_per_vehicle'] / 2**20:.1f} MiB); {x['vehicles']} vehicles x "
          f"{x['rounds']} rounds is {x['total'] / 2**30:.2f} GiB. Images moved: "
          f"{x['images_moved']}")
    rt = p["roundtrip"]
    print("round trip: " + ("PASS" if rt.get("ok") else "FAIL") if rt.get("available")
          else f"round trip: not checkable -- {rt['reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
