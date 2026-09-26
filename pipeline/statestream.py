"""State pushed as a sequence of diffs, instead of polled whole.

The dashboard used to GET `/api/state` every two seconds and throw almost all of it
away: the snapshot carries the fleet, the stage table, every metrics row, the GPU
history and the holdout curve, and between two ticks of an idle server exactly none of
it changes. Two seconds is also the floor on how stale anything on screen can be.

This module owns the wire format and nothing else. It does not know about HTTP, the
event bus, or what a snapshot contains.

    op ::= ["=", value]        replace this key outright (also: a new key)
         | ["-"]               this key is gone
         | ["~", {key: op}]    recurse; only the keys that moved are present

`diff` returns `None` when nothing moved, so an unchanged tick costs a keep-alive
comment rather than a payload. The encoding is explicitly tagged because the
alternative -- a bare nested dict meaning "recurse" -- cannot tell a sub-patch from a
replacement whose value happens to be a dict, and `config` is a dict.

Measured against an idle server on 2026-09-26, six seconds of `/api/stream`: one
3 984-byte snapshot, one 867-byte patch (the GPU sampler's history, appended), two
keep-alive comments. The old two-second poll of the same idle state moved 11 952 bytes
over the same six seconds. **What dominates a patch during a run is the GPU history
list**, because a list is replaced whole -- deliberately, see below -- so the win is
largest exactly where the page spends most of its life: open and idle.

The browser half is `static/js/stream.js`; `pipeline/tests/js/state_patch.mjs` feeds
patches produced *here* into the apply written *there*, so the two cannot drift.

    python -m pipeline.statestream          # self-check
"""
from __future__ import annotations

import json


def diff(old, new):
    """Smallest op taking ``old`` to ``new``, or None if they are already equal."""
    if isinstance(old, dict) and isinstance(new, dict):
        sub: dict = {}
        for key, value in new.items():
            if key not in old:
                sub[key] = ["=", value]
            else:
                op = diff(old[key], value)
                if op is not None:
                    sub[key] = op
        for key in old:
            if key not in new:
                sub[key] = ["-"]
        return ["~", sub] if sub else None
    # Lists are replaced whole. A metrics table gains rows at the end and an
    # element-wise diff would be smaller, but "smaller" is not the point: a
    # positional list patch is the kind of code that drops a row and looks fine.
    return None if old == new else ["=", new]


def apply(old, op):
    """``old`` with ``op`` applied. The inverse of `diff`, and the browser's twin."""
    if op is None:
        return old
    if op[0] == "=":
        return op[1]
    if op[0] == "-":
        return None
    out = dict(old) if isinstance(old, dict) else {}
    for key, sub in op[1].items():
        if sub[0] == "-":
            out.pop(key, None)
        else:
            out[key] = apply(out.get(key), sub)
    return out


def frame(event: str, payload: dict) -> bytes:
    """One SSE frame. Newlines inside the JSON would split the frame, so it is compact."""
    return f"event: {event}\ndata: {json.dumps(payload, separators=(',', ':'))}\n\n".encode()


class Sequence:
    """One browser's view of the state, and the sequence number it is at.

    Per connection on purpose. A server-wide patch log would need every connection's
    position, an eviction policy and a replay path; a reconnect is cheap here because
    the snapshot is read off disk anyway. So a gap in the sequence a client receives
    is not something it has to recover from -- it is a bug, and the client refetching
    `/api/state` on one is a check, not a mechanism.
    """

    def __init__(self):
        self.seq = 0
        self.last = None

    def open(self, snapshot: dict) -> bytes:
        self.seq, self.last = 0, snapshot
        return frame("snapshot", {"seq": 0, "state": snapshot})

    def update(self, snapshot: dict) -> bytes | None:
        """The next frame, or None when the snapshot is byte-for-byte what they have."""
        if snapshot is self.last:
            return None
        op = diff(self.last, snapshot)
        self.last = snapshot
        if op is None:
            return None
        self.seq += 1
        return frame("patch", {"seq": self.seq, "patch": op})


def demo() -> None:
    """Self-check: a run's worth of snapshots, replayed through the patches alone.

    The assertion that matters is the one inside the loop. A diff that drops a change
    still produces a plausible stream -- the client just quietly shows a stale number,
    which is this project's whole catalogue of bugs in one sentence.
    """
    steps = [
        {"busy": False, "live": {"checksums": [], "rounds_done": 0}, "fleet": []},
        {"busy": True, "live": {"checksums": [], "rounds_done": 0}, "fleet": [{"vid": 1}]},
        {"busy": True, "live": {"checksums": [-1032.5], "rounds_done": 1}, "fleet": [{"vid": 1}]},
        {"busy": True, "live": {"checksums": [-1032.5, -2646.9], "rounds_done": 2},
         "fleet": [{"vid": 1}], "gpu": {"util_pct": 27}},
        {"busy": False, "live": {"checksums": [-1032.5, -2646.9], "rounds_done": 2},
         "fleet": [{"vid": 1}]},                      # gpu key removed again
    ]
    seq = Sequence()
    opened = json.loads(seq.open(steps[0]).decode().split("data: ", 1)[1])
    assert opened["seq"] == 0
    client = opened["state"]                            # what the browser holds
    for want in steps[1:]:
        raw = seq.update(want)
        if raw is not None:
            payload = json.loads(raw.decode().split("data: ", 1)[1])
            client = apply(client, payload["patch"])
        assert client == want, f"patch stream diverged: {client} != {want}"

    # Nothing changed -> nothing on the wire.
    assert seq.update(json.loads(json.dumps(steps[-1]))) is None

    # A key whose value changes shape, in both directions.
    assert apply({"k": {"a": 1}}, diff({"k": {"a": 1}}, {"k": [1, 2]})) == {"k": [1, 2]}
    assert apply({"k": [1, 2]}, diff({"k": [1, 2]}, {"k": {"a": 1}})) == {"k": {"a": 1}}
    assert diff({"a": {"b": 1}}, {"a": {"b": 1}}) is None
    assert diff({"a": {"b": 1}}, {"a": {"b": 2}}) == ["~", {"a": ["~", {"b": ["=", 2]}]}]
    assert apply({"a": {"b": 1, "c": 2}}, ["~", {"a": ["~", {"c": ["-"]}]}]) == {"a": {"b": 1}}
    print("statestream self-check OK")


if __name__ == "__main__":
    demo()
