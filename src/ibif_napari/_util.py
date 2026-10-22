"""Helpers shared by the panels: which axis is which, and the volume at the current position."""
from __future__ import annotations

import numpy as np

from ._reader import AXIS_NAMES


def layer_axes(layer):
    """Axis letters of a layer, e.g. "TZYX". ND2 layers record them; others are guessed."""
    axes = layer.metadata.get("ibif_axes") if layer.metadata else None
    ndim = layer.ndim
    if axes and len(axes) == ndim:
        return axes
    guesses = {1: "X", 2: "YX", 3: "ZYX", 4: "TZYX", 5: "TPZYX"}
    return guesses.get(ndim) or "?" * (ndim - 5) + "TPZYX"


def axis_labels_for(viewer):
    """World axis labels from the first ND2 layer that spans every viewer dimension."""
    for layer in viewer.layers:
        axes = layer.metadata.get("ibif_axes") if layer.metadata else None
        if axes and len(axes) == viewer.dims.ndim:
            return [AXIS_NAMES.get(a, a) for a in axes]
    return None


def current_data_index(viewer, layer):
    """Integer data index of `layer` at the viewer's current position."""
    coords = layer.world_to_data(viewer.dims.point)
    coords = np.asarray(coords)[-layer.ndim :]
    shape = layer.data.shape[: layer.ndim]
    return [int(np.clip(round(c), 0, s - 1)) for c, s in zip(coords, shape)]


def current_volume(viewer, layer, keep="ZYX", crop=None):
    """The slice of `layer` at the current time/position, keeping the axes in `keep`.

    `crop` optionally limits kept axes to {axis: (start, stop)}. Returns
    (array, axes_kept, scale_kept, index) where index is the full data index used.
    """
    axes = layer_axes(layer)
    idx = current_data_index(viewer, layer)
    crop = crop or {}
    index, kept, scale = [], "", []
    for a, i, s in zip(axes, idx, layer.scale):
        if a in keep:
            index.append(slice(*crop[a]) if a in crop else slice(None))
            kept += a
            scale.append(s)
        else:
            index.append(i)
    data = layer.data
    if isinstance(data, list):  # multiscale: use full resolution
        data = data[0]
    return np.asarray(data[tuple(index)]), kept, scale, tuple(index)
