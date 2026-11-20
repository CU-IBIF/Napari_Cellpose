"""Processing: ordered corrections of the image data before segmentation.

The library of steps lives in `STEPS` (key -> StepSpec). A step is added with `register`;
the Processing section lists the library, builds each step's settings from its
parameters, and runs a `Pipeline` of them. Steps marked "coming soon" are listed but
cannot be added to a pipeline.

    from ibif_napari import _processing
    pipe = _processing.Pipeline.from_json('{"steps": [{"key": "registration"}]}')
    result = pipe.run(_processing.Stack({"GFP": data}, "TZYX", {"Z": 2, "Y": .5, "X": .5}))
"""
from __future__ import annotations

from .base import AVAILABLE, COMING_SOON, Cancelled, Param, Stack, StepSpec  # noqa: F401

STEPS = {}  # key -> StepSpec, in library order
REGISTRY_VERSION = 1  # bumped when steps or their parameters change incompatibly


def register(spec):
    """Add (or replace) a step in the library; returns it."""
    if not isinstance(spec, StepSpec):
        raise TypeError("register a StepSpec")
    STEPS[spec.key] = spec
    return spec


def unregister(key):
    STEPS.pop(key, None)


def get(key):
    try:
        return STEPS[key]
    except KeyError:
        raise ValueError(f"unknown processing step {key!r}") from None


def library():
    """The steps in library order: available first, then those coming soon."""
    return sorted(STEPS.values(), key=lambda s: (not s.available,))


def _load_builtin():
    from . import stubs  # noqa: F401  (they register themselves)


from .pipeline import Pipeline, PipelineStep, Result  # noqa: E402,F401

_load_builtin()
