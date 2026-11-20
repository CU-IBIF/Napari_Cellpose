"""What a processing step is: its parameters (a schema the settings pane is built from)
and the function that runs it.

A step works on a `Stack`: one or more channels of the same image, each a NumPy array
with the same axes (e.g. "TZYX"), and the voxel size. It returns the processed stack,
and may add `records` (one row per frame or plane, e.g. a registration's shifts) that are
kept with the result.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace

from .._process import Cancelled  # noqa: F401  (raised by a progress callback to stop a run)

AVAILABLE = "available"
COMING_SOON = "coming_soon"
KINDS = ("float", "int", "choice", "bool")


@dataclass(frozen=True)
class Param:
    """One setting of a step. `kind` is float / int / choice / bool; `options` lists the
    choices; `unit` is shown after the value (µm, px, %...)."""

    name: str
    label: str
    kind: str
    default: object
    tip: str = ""
    minimum: float | None = None
    maximum: float | None = None
    step: float | None = None
    decimals: int = 2
    unit: str = ""
    options: tuple = ()

    def __post_init__(self):
        if self.kind not in KINDS:
            raise ValueError(f"unknown parameter kind {self.kind!r}")
        if self.kind == "choice" and self.default not in self.options:
            raise ValueError(f"{self.name}: default {self.default!r} is not one of the options")

    def check(self, value):
        """`value` converted to this parameter's type, or ValueError if it does not fit."""
        if self.kind == "bool":
            if not isinstance(value, bool):
                raise ValueError(f"{self.label}: expected yes / no, got {value!r}")
            return value
        if self.kind == "choice":
            if value not in self.options:
                raise ValueError(f"{self.label}: {value!r} is not one of {', '.join(map(str, self.options))}")
            return value
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{self.label}: expected a number, got {value!r}")
        if self.kind == "int":
            if int(value) != value:
                raise ValueError(f"{self.label}: expected a whole number, got {value!r}")
            value = int(value)
        else:
            value = float(value)
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{self.label}: {value} is below {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{self.label}: {value} is above {self.maximum}")
        return value


@dataclass(frozen=True)
class StepSpec:
    """A routine in the library. `run(stack, params, progress)` returns the processed
    stack; `progress(fraction 0-1, message)` may be called, and raises `Cancelled` when
    the run is cancelled. Steps with status COMING_SOON have no `run` and cannot be added
    to a pipeline."""

    key: str
    title: str
    category: str
    description: str
    params: tuple = ()
    run: object = None
    status: str = AVAILABLE
    needs_reference: bool = False  # uses the stack's reference channel

    @property
    def available(self):
        return self.status == AVAILABLE and self.run is not None

    def defaults(self):
        return {p.name: p.default for p in self.params}

    def check_params(self, params):
        """A full, checked set of parameters: defaults filled in, unknown names refused."""
        known = {p.name: p for p in self.params}
        unknown = set(params) - set(known)
        if unknown:
            raise ValueError(f"{self.title}: unknown setting(s) {', '.join(sorted(unknown))}")
        return {name: p.check(params[name]) if name in params else p.default for name, p in known.items()}


@dataclass
class Stack:
    """Channels of one image (name -> array, all the same shape and axes), the axes
    (e.g. "TZYX"), the voxel size per axis in µm ({"Z": 2.0, "Y": 0.5, "X": 0.5}) and the
    channel the steps that need one use as their reference."""

    channels: dict
    axes: str
    spacing: dict = field(default_factory=dict)
    reference: str | None = None

    def __post_init__(self):
        shapes = {tuple(a.shape) for a in self.channels.values()}
        if not self.channels:
            raise ValueError("no channels to process")
        if len(shapes) != 1:
            raise ValueError("the channels differ in size: " + ", ".join(
                f"{n} {tuple(a.shape)}" for n, a in self.channels.items()))
        if len(self.axes) != len(next(iter(shapes))):
            raise ValueError(f"axes {self.axes!r} do not match the data's {len(next(iter(shapes)))} dimensions")
        if self.reference is None:
            self.reference = next(iter(self.channels))
        if self.reference not in self.channels:
            raise ValueError(f"reference channel {self.reference!r} is not among the channels")

    @property
    def shape(self):
        return next(iter(self.channels.values())).shape

    def with_channels(self, channels):
        return replace(self, channels=dict(channels))

