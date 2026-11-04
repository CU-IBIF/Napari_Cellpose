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


def spatial_ndim(layer):
    axes = (layer.metadata or {}).get("ibif_axes")
    if axes and len(axes) == layer.ndim:
        return sum(a in "ZYX" for a in axes)
    return 3 if layer.ndim >= 3 else 2


def aligned_image(viewer, image, labels, lead_index):
    """The pixels of `image` under one block of `labels` (its spatial volume at
    `lead_index` of its time/position axes). Image axes the labels do not have follow
    the viewer's current position. The two must share pixel size."""
    n_sp = spatial_ndim(labels)
    current = current_data_index(viewer, image)
    index = list(current)
    for k in range(1, image.ndim + 1):
        ia, la = image.ndim - k, labels.ndim - k
        if la < 0:
            continue
        ls, isc = float(labels.scale[la]), float(image.scale[ia])
        if abs(ls - isc) > 1e-6 * max(abs(ls), abs(isc), 1e-12):
            raise ValueError(f"{image.name} and {labels.name} have different pixel sizes")
        spatial = k <= n_sp
        start = float(labels.translate[la]) + (0 if spatial else lead_index[la] * ls)
        i0 = int(round((start - float(image.translate[ia])) / isc))
        if spatial:
            stop = i0 + labels.data.shape[la]
            if i0 < 0 or stop > image.data.shape[ia]:
                raise ValueError(f"{labels.name} reaches outside {image.name}")
            index[ia] = slice(i0, stop)
        else:
            if not 0 <= i0 < image.data.shape[ia]:
                raise ValueError(f"{labels.name} reaches outside {image.name}")
            index[ia] = i0
    data = image.data[0] if isinstance(image.data, list) else image.data
    return np.asarray(data[tuple(index)])
