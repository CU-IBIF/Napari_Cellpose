"""Shared reading of multi-image formats through bioio readers (Zeiss CZI, Leica LIF).

Produces the same layers as the ND2 reader. Images ("scenes") of equal size can be
stacked as the position axis ("P"); tiled (mosaic) images are stitched lazily; axes of
length 1 are dropped so a single-Z image opens as 2D.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import dask.array as da
import numpy as np

from ._reader import layer_data

# Opening nothing (e.g. the image chooser was cancelled): npe2's "no layers" value.
NO_LAYERS = [(None,)]


@dataclass
class Scene:
    index: int
    name: str
    data: object  # dask array
    axes: str
    channel_names: list
    zyx_um: tuple
    time_interval_s: float | None
    emissions: list = field(default_factory=list)
    colormaps: list = field(default_factory=list)

    @property
    def summary(self):
        """e.g. 'T 3 · C 2 · Z 12 · 512×512'"""
        sizes = dict(zip(self.axes, self.data.shape))
        parts = [f"{a} {sizes[a]}" for a in "TCZ" if sizes.get(a, 1) > 1]
        parts.append(f"{sizes.get('Y', '?')}×{sizes.get('X', '?')}")
        if "S" in sizes:
            parts.append("RGB")
        return " · ".join(parts)


def import_reader(module, package, file_type):
    """The bioio reader class for a format, or an ImportError that says how to install it."""
    import importlib

    try:
        return importlib.import_module(module).Reader
    except ImportError as e:
        raise ImportError(
            f"Opening {file_type} files needs the '{package}' package, which is not installed "
            f"in this environment ({e}). Install it with:  pip install {package}"
        ) from e


def read_scene(reader, index):
    """Lazy data and metadata of one scene (does not read pixels)."""
    reader.set_scene(index)
    xa = reader.mosaic_xarray_dask_data if "M" in reader.dims.order else reader.xarray_dask_data
    pps = reader.physical_pixel_sizes
    return Scene(
        index=index,
        name=str(reader.scenes[index]),
        data=xa.data,
        axes="".join(xa.dims),
        channel_names=[str(n) for n in (reader.channel_names or [])],
        zyx_um=(pps.Z, pps.Y, pps.X),
        time_interval_s=_interval_s(reader),
    )


def scene_layers(scenes, stem, path, stack):
    """LayerData for the given scenes: stacked as positions when `stack` and all scenes
    have the same shape, otherwise one set of channel layers per scene."""
    if not scenes:
        return NO_LAYERS
    first = scenes[0]
    same = len({(s.data.shape, s.axes) for s in scenes}) == 1
    if len(scenes) > 1 and stack and same:
        stacked = da.stack([s.data for s in scenes])
        data, axes = _squeeze(*_positions_after_time(stacked, "P" + first.axes))
        return _layers(first, data, axes, stem, path)
    layers = []
    for s in scenes:
        data, axes = _squeeze(s.data, s.axes)
        layers += _layers(s, data, axes, stem if len(scenes) == 1 else f"{stem} {s.name}", path)
    return layers


def _layers(scene, data, axes, name, path):
    out = layer_data(
        data,
        dict(zip(axes, data.shape)),
        scene.channel_names,
        scene.emissions or [None] * len(scene.channel_names),
        scene.zyx_um,
        scene.time_interval_s,
        name,
        path,
    )
    if scene.colormaps:
        for _, kwargs, _ in out:
            cmaps = kwargs.get("colormap")
            if isinstance(cmaps, list):
                kwargs["colormap"] = [c or m for c, m in zip(scene.colormaps, cmaps)]
            elif cmaps is not None and scene.colormaps[0]:
                kwargs["colormap"] = scene.colormaps[0]
    return out


def stem_of(path):
    return Path(str(path)).stem


def _positions_after_time(data, axes):
    """Move a leading "P" behind "T", matching the ND2 axis order (T, P, Z, ...)."""
    if "T" in axes:
        order = [axes.index("T"), axes.index("P")]
        order += [i for i, a in enumerate(axes) if a not in "TP"]
        data = data.transpose(order)
        axes = "".join(axes[i] for i in order)
    return data, axes


def _squeeze(data, axes):
    """Drop length-1 time/position/Z axes (and any other singleton non-image axis)."""
    keep = [i for i, (a, n) in enumerate(zip(axes, data.shape)) if n > 1 or a in "CYXS"]
    index = tuple(slice(None) if i in keep else 0 for i in range(data.ndim))
    return data[index], "".join(axes[i] for i in keep)


def _interval_s(reader):
    try:
        interval = reader.time_interval
    except Exception:
        return None
    if interval is None:
        return None
    seconds = interval.total_seconds() if hasattr(interval, "total_seconds") else float(interval)
    return seconds if np.isfinite(seconds) and seconds > 0 else None
