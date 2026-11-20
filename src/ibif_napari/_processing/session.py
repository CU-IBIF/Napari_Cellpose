"""The Processing section's state for one viewer (its pipeline), how image layers become
a `Stack` and results become layers again, and saving the pipeline with the workspace."""
from __future__ import annotations

import numpy as np

from .base import Stack
from .pipeline import Pipeline

SPATIAL = "ZYX"
SUFFIX = " [processed]"
# Metadata that ties a layer to its microscope file: a processed copy must not keep it,
# or a reopened workspace would read the unprocessed data from the file again.
SOURCE_KEYS = ("ibif_source", "ibif_scenes", "ibif_stacked", "ibif_tiff_layout")
SCOPE_ALL = "Every time point and position"
SCOPE_POSITION = "Current position (every time point)"

_pipelines = {}  # viewer -> Pipeline
_listeners = {}  # viewer -> [callback()] told when a workspace brings a pipeline back


def pipeline_of(viewer):
    if viewer not in _pipelines:
        _pipelines[viewer] = Pipeline()
    return _pipelines[viewer]


def set_pipeline(viewer, pipeline, notify=True):
    _pipelines[viewer] = pipeline
    if notify:
        for callback in list(_listeners.get(viewer, [])):
            callback()


def listen(viewer, callback):
    _listeners.setdefault(viewer, []).append(callback)


def _axes(layer):
    from .._util import layer_axes

    return layer_axes(layer)


def stack_from_layers(viewer, layers, reference=None, scope=SCOPE_ALL):
    """A `Stack` of image layers on the same grid (the channels of one image). With
    SCOPE_POSITION, only the viewer's current position (P and other non-T, non-spatial
    axes) is taken; time points are kept. Returns (stack, index) where index is the data
    index taken from each layer."""
    if not layers:
        raise ValueError("choose at least one channel")
    first = layers[0]
    axes = _axes(first)
    for layer in layers[1:]:
        if tuple(layer.data.shape) != tuple(first.data.shape) or _axes(layer) != axes:
            raise ValueError(f"{layer.name} is not on the same grid as {first.name}; process them separately")
        if not np.allclose(layer.scale, first.scale):
            raise ValueError(f"{layer.name} and {first.name} have different voxel sizes")
    index = [slice(None)] * len(axes)
    if scope == SCOPE_POSITION:
        from .._util import current_data_index

        current = current_data_index(viewer, first)
        for i, a in enumerate(axes):
            if a not in SPATIAL and a != "T":
                index[i] = current[i]
    kept = "".join(a for a, ix in zip(axes, index) if isinstance(ix, slice))
    spacing = {a: float(s) for a, s, ix in zip(axes, first.scale, index) if isinstance(ix, slice) and a in SPATIAL}
    channels = {}
    for layer in layers:
        data = layer.data[0] if isinstance(layer.data, list) else layer.data  # multiscale: full size
        channels[layer.name] = np.asarray(data[tuple(index)])
    return Stack(channels, kept, spacing, reference or first.name), tuple(index)


def add_result_layers(viewer, layers, result, index):
    """Add the processed channels next to their originals, named '<name> [processed]',
    with the steps that made them in their metadata. Returns the new layers."""
    stack = result.stack
    steps = [{k: (_records_jsonable(v) if k == "records" else v) for k, v in s.items()} for s in result.steps]
    added = []
    for layer in layers:
        kept = [i for i, ix in enumerate(index) if isinstance(ix, slice)]
        meta = {k: v for k, v in dict(layer.metadata or {}).items() if k not in SOURCE_KEYS}
        meta.update(ibif_axes=stack.axes, ibif_processed_from=layer.name, ibif_processing=steps)
        kwargs = dict(
            name=layer.name + SUFFIX, metadata=meta,
            scale=[layer.scale[i] for i in kept], translate=[layer.translate[i] for i in kept],
            colormap=layer.colormap.name, blending=layer.blending, gamma=layer.gamma,
        )
        data = stack.channels[layer.name]
        if data.size and np.isfinite(data).any():
            lo, hi = layer.contrast_limits
            kwargs["contrast_limits"] = (lo, hi) if hi > lo else None
        added.append(viewer.add_image(data, **{k: v for k, v in kwargs.items() if v is not None}))
    return added


def _records_jsonable(records):
    """Step records (a DataFrame, list of dicts or None) as a list of plain dicts."""
    if records is None:
        return None
    if hasattr(records, "to_dict"):
        records = records.to_dict(orient="records")
    out = []
    for row in records:
        out.append({str(k): (v.item() if hasattr(v, "item") else v) for k, v in dict(row).items()})
    return out


# ---- workspace -------------------------------------------------------------------------


def _save(viewer, folder):
    pipe = _pipelines.get(viewer)
    return pipe.to_dict() if pipe is not None and len(pipe) else None


def _load(viewer, folder, data):
    try:
        pipe = Pipeline.from_dict(data)
    except ValueError as e:
        return [f"processing: the pipeline could not be restored ({e})"]
    set_pipeline(viewer, pipe)
    return []


def _register():
    from .. import _workspace

    _workspace.register_extension("processing", _save, _load)


_register()
