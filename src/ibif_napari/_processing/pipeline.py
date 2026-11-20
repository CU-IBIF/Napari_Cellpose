"""An ordered list of processing steps, with their settings, that runs on a `Stack`."""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from .base import Cancelled, Stack

FORMAT = "ibif-napari-processing"
VERSION = 1


@dataclass
class PipelineStep:
    key: str
    params: dict = field(default_factory=dict)
    enabled: bool = True

    @property
    def spec(self):
        from . import get

        return get(self.key)

    def to_dict(self):
        return {"key": self.key, "params": dict(self.params), "enabled": bool(self.enabled)}


@dataclass
class Result:
    """What a run made: the processed stack, and per step what it recorded (e.g. the
    shifts of a registration, one row per frame) and a short summary."""

    stack: Stack
    steps: list  # [{"key", "title", "params", "summary", "records"}]


class Pipeline:
    """Steps run one after the other, each on the previous one's result."""

    def __init__(self, steps=()):
        self.steps = []
        for s in steps:
            self.add(s.key, s.params, s.enabled) if isinstance(s, PipelineStep) else self.add(s)

    def __len__(self):
        return len(self.steps)

    def __iter__(self):
        return iter(self.steps)

    def add(self, key, params=None, enabled=True, index=None):
        """Add a step from the library (with its defaults for settings not given). Steps
        that are coming soon cannot be added."""
        from . import get

        spec = get(key)
        if not spec.available:
            raise ValueError(f"{spec.title} is coming soon and cannot be added to a pipeline yet")
        step = PipelineStep(key, spec.check_params(dict(params or {})), bool(enabled))
        if index is None:
            self.steps.append(step)
        else:
            self.steps.insert(index, step)
        return step

    def remove(self, index):
        return self.steps.pop(index)

    def move(self, index, to):
        """Move the step at `index` to position `to` (both clipped to the list)."""
        if not self.steps:
            return
        to = max(0, min(len(self.steps) - 1, to))
        self.steps.insert(to, self.steps.pop(index))

    def set_params(self, index, **params):
        step = self.steps[index]
        step.params = step.spec.check_params({**step.params, **params})
        return step

    # ---- saving --------------------------------------------------------------------

    def to_dict(self):
        from . import REGISTRY_VERSION

        return {"format": FORMAT, "version": VERSION, "registry": REGISTRY_VERSION,
                "steps": [s.to_dict() for s in self.steps]}

    def to_json(self, indent=2):
        return json.dumps(self.to_dict(), indent=indent)

    @classmethod
    def from_dict(cls, data):
        """A pipeline from `to_dict` (or hand-written: {"steps": [{"key": ...}]}). Unknown
        steps, steps coming soon and settings that do not fit are refused, naming them."""
        if not isinstance(data, dict) or not isinstance(data.get("steps"), list):
            raise ValueError("not a processing pipeline (no list of steps)")
        if data.get("format", FORMAT) != FORMAT:
            raise ValueError(f"not a processing pipeline (format {data.get('format')!r})")
        pipe = cls()
        for n, entry in enumerate(data["steps"], start=1):
            if not isinstance(entry, dict) or "key" not in entry:
                raise ValueError(f"step {n} has no key")
            params = entry.get("params") or {}
            if not isinstance(params, dict):
                raise ValueError(f"step {n}: settings must be an object")
            try:
                pipe.add(entry["key"], params, bool(entry.get("enabled", True)))
            except ValueError as e:
                raise ValueError(f"step {n}: {e}") from None
        return pipe

    @classmethod
    def from_json(cls, text):
        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            raise ValueError(f"not a processing pipeline (invalid JSON: {e})") from None
        return cls.from_dict(data)

    def copy(self):
        return Pipeline.from_dict(self.to_dict())

    # ---- running -------------------------------------------------------------------

    def run(self, stack, progress=None):
        """Run the enabled steps in order on `stack`. `progress(fraction, message)` is
        called as it goes and may raise `Cancelled`. Returns a `Result`."""
        active = [s for s in self.steps if s.enabled]
        if not active:
            raise ValueError("the pipeline has no enabled steps")
        done = []
        for n, step in enumerate(active):
            spec = step.spec

            def report(fraction, message="", n=n, spec=spec):
                if progress is not None:
                    progress((n + max(0.0, min(1.0, fraction))) / len(active),
                             f"{spec.title} ({n + 1}/{len(active)}){': ' + message if message else ''}")

            report(0.0)
            out = spec.run(stack, dict(step.params), report)
            records, summary = None, ""
            if isinstance(out, tuple):
                out, records, summary = (tuple(out) + (None, ""))[:3]
            if not isinstance(out, Stack):
                raise TypeError(f"{spec.title} did not return a stack")
            if out.shape != stack.shape or out.axes != stack.axes or set(out.channels) != set(stack.channels):
                raise ValueError(f"{spec.title} changed the image's size, axes or channels")
            stack = out
            done.append({"key": spec.key, "title": spec.title, "params": dict(step.params),
                         "summary": summary or "", "records": records})
        if progress is not None:
            progress(1.0, "done")
        return Result(stack, done)


__all__ = ["Pipeline", "PipelineStep", "Result", "Cancelled"]
