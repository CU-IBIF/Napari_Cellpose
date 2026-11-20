"""Running a processing pipeline on `ibif-napari-server` (a facility GPU server, or a
cluster job, which runs the same server).

A request is one `.npz` upload: every channel as an array (`channel_0`, `channel_1`, ...)
and a JSON `manifest` with the channel names, axes, voxel size, reference channel and
the pipeline. The server checks the pipeline against its own library of steps (unknown
steps, steps coming soon and settings that do not fit are refused), runs it in its
worker process (so Cancel stops it at once) and returns the processed channels and the
steps' records. Nothing but the library's own steps can run.

Images with positions (or other axes that are neither T nor spatial) are sent one
position at a time, so each upload stays small and a registration along T still sees a
whole time series.
"""
from __future__ import annotations

import io
import json

import numpy as np

from .base import Stack
from .pipeline import Pipeline

OP = "process"
SPATIAL = "ZYX"
AXIS_LETTERS = set("TPCZYX") | {"?"}


def _json_array(obj):
    return np.frombuffer(json.dumps(obj).encode(), np.uint8)


def _from_json_array(a):
    return json.loads(np.asarray(a, np.uint8).tobytes().decode())


def encode_request(stack, pipeline):
    """The upload for running `pipeline` (a Pipeline or its dict) on `stack`."""
    pipe = pipeline.to_dict() if isinstance(pipeline, Pipeline) else dict(pipeline)
    names = list(stack.channels)
    manifest = {"op": OP, "channels": names, "axes": stack.axes, "spacing": dict(stack.spacing),
                "reference": stack.reference, "pipeline": pipe}
    arrays = {f"channel_{i}": np.ascontiguousarray(stack.channels[n]) for i, n in enumerate(names)}
    buf = io.BytesIO()
    np.savez_compressed(buf, manifest=_json_array(manifest), **arrays)
    return buf.getvalue()


def is_request(arrays):
    return "manifest" in arrays


def check_request(arrays, max_bytes):
    """(Stack, Pipeline) from a decoded upload, or ValueError naming what is wrong
    (ValueError's `too_large` attribute is set when it is only too big)."""
    try:
        manifest = _from_json_array(arrays["manifest"])
    except Exception:
        raise ValueError("the manifest is not JSON") from None
    if not isinstance(manifest, dict) or manifest.get("op") != OP:
        raise ValueError("not a processing request")
    names = manifest.get("channels")
    axes = manifest.get("axes")
    spacing = manifest.get("spacing") or {}
    if not isinstance(names, list) or not names or not all(isinstance(n, str) for n in names):
        raise ValueError("the request names no channels")
    if not isinstance(axes, str) or not set(axes) <= AXIS_LETTERS or "Y" not in axes or "X" not in axes:
        raise ValueError(f"bad axes {axes!r}")
    if not isinstance(spacing, dict) or not all(
            k in SPATIAL and isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
            for k, v in spacing.items()):
        raise ValueError("bad voxel size")
    channels, total = {}, 0
    for i, name in enumerate(names):
        a = arrays.get(f"channel_{i}")
        if a is None:
            raise ValueError(f"channel {name!r} is missing")
        if a.ndim != len(axes) or a.ndim > 6 or a.dtype.kind not in "uif":
            raise ValueError(f"channel {name!r} must be a numeric array with axes {axes}")
        total += a.nbytes
        channels[name] = a
    if total > max_bytes:
        err = ValueError(f"the image ({total / 2**20:.0f} MB) is larger than this server accepts "
                         f"({max_bytes / 2**20:.0f} MB)")
        err.too_large = True
        raise err
    pipeline = Pipeline.from_dict(manifest.get("pipeline"))  # refuses unknown steps and stubs
    if not len(pipeline) or not any(s.enabled for s in pipeline):
        raise ValueError("the pipeline has no enabled steps")
    stack = Stack(channels, axes, {k: float(v) for k, v in spacing.items()}, manifest.get("reference"))
    return stack, pipeline


def run_payload(stack_dict, pipeline_dict):
    """Run in the server's worker process: (processed channels, steps)."""
    stack = Stack(**stack_dict)
    result = Pipeline.from_dict(pipeline_dict).run(stack)
    steps = [{**s, "records": _plain(s["records"])} for s in result.steps]
    return result.stack.channels, steps


def payload(stack, pipeline):
    """What the server hands its worker process (plain, picklable)."""
    return ({"channels": stack.channels, "axes": stack.axes, "spacing": stack.spacing,
             "reference": stack.reference}, pipeline.to_dict())


def encode_result(channels, steps):
    names = list(channels)
    buf = io.BytesIO()
    np.savez_compressed(buf, manifest=_json_array({"channels": names, "steps": steps}),
                        **{f"channel_{i}": np.asarray(channels[n]) for i, n in enumerate(names)})
    return buf.getvalue()


def decode_result(data):
    with np.load(io.BytesIO(data), allow_pickle=False) as npz:
        manifest = _from_json_array(npz["manifest"])
        channels = {n: npz[f"channel_{i}"] for i, n in enumerate(manifest["channels"])}
    return channels, manifest["steps"]


def _plain(records):
    if records is None:
        return None
    if hasattr(records, "to_dict"):
        records = records.to_dict(orient="records")
    return [{str(k): (v.item() if hasattr(v, "item") else v) for k, v in dict(r).items()} for r in records]


# ---- the client side -----------------------------------------------------------------------


def series(stack):
    """[(index into the axes, label e.g. {"P": 1}), ...]: the parts of `stack` sent one at a
    time (one per position, or other axis that is neither T nor spatial)."""
    outer = [i for i, a in enumerate(stack.axes) if a not in SPATIAL and a != "T"]
    shape = [stack.shape[i] for i in outer]
    out = []
    for pos in np.ndindex(*shape):
        index = [slice(None)] * len(stack.axes)
        for i, p in zip(outer, pos):
            index[i] = p
        out.append((tuple(index), {stack.axes[i]: int(p) for i, p in zip(outer, pos)}))
    return out


def sub_stack(stack, index):
    axes = "".join(a for a, ix in zip(stack.axes, index) if isinstance(ix, slice))
    return Stack({n: np.asarray(a[index]) for n, a in stack.channels.items()}, axes, dict(stack.spacing),
                 stack.reference)


def merge(stack, parts):
    """The whole stack again from [(index, label, channels, steps)], with each step's
    records labelled with the part they came from."""
    channels = {n: None for n in stack.channels}
    steps, summaries = None, None
    for index, label, part_channels, part_steps in parts:
        for name, data in part_channels.items():
            if channels[name] is None:
                channels[name] = np.empty(stack.shape, data.dtype)
            channels[name][index] = data
        if steps is None:
            steps = [{**s, "records": [] if s.get("records") is not None else None} for s in part_steps]
            summaries = [[] for _ in part_steps]
        for s, ps, said in zip(steps, part_steps, summaries):
            if ps.get("records") is not None:
                s["records"] += [{**label, **r} for r in ps["records"]]
            if ps.get("summary"):
                said.append(ps["summary"])
    for s, said in zip(steps or [], summaries or []):
        s["summary"] = said[0] if len(said) == 1 else (
            f"{len(said)} positions: " + "; ".join(said[:2]) + ("…" if len(said) > 2 else "") if said else "")
    return stack.with_channels(channels), steps or []


__all__ = ["encode_request", "check_request", "run_payload", "encode_result", "decode_result",
           "series", "sub_stack", "merge", "OP"]
