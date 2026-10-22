"""Zeiss CZI reader (via bioio-czi), producing the same layers as the ND2 reader.

Scenes of equal size become the position axis ("P"); tiled (mosaic) scenes are stitched
lazily; axes of length 1 are dropped so a single-Z image opens as 2D.
"""
from __future__ import annotations

from pathlib import Path

import dask.array as da
import numpy as np

from ._reader import layer_data


def read_czi(path, reader=None):
    """Read a CZI file. `reader` (a bioio-czi Reader) can be passed for testing."""
    if reader is None:
        from bioio_czi import Reader

        reader = Reader(str(path))
    path = str(path)
    stem = Path(path).stem
    scenes = list(reader.scenes)

    arrays = []
    for i in range(len(scenes)):
        reader.set_scene(i)
        xa = reader.mosaic_xarray_dask_data if "M" in reader.dims.order else reader.xarray_dask_data
        arrays.append((xa.data, "".join(xa.dims)))
    reader.set_scene(0)
    names = [str(n) for n in (reader.channel_names or [])]
    emissions = _emissions(reader.metadata, len(names))
    pps = reader.physical_pixel_sizes
    zyx = (pps.Z, pps.Y, pps.X)
    interval = _interval_s(reader)

    same = len({(a.shape, axes) for a, axes in arrays}) == 1
    if len(arrays) > 1 and same:
        stacked = da.stack([a for a, _ in arrays])
        data, axes = _squeeze(*_positions_after_time(stacked, "P" + arrays[0][1]))
        return layer_data(data, dict(zip(axes, data.shape)), names, emissions, zyx, interval, stem, path)

    layers = []
    for (data, axes), scene in zip(arrays, scenes):
        data, axes = _squeeze(data, axes)
        name = stem if len(arrays) == 1 else f"{stem} {scene}"
        layers += layer_data(data, dict(zip(axes, data.shape)), names, emissions, zyx, interval, name, path)
    return layers


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


def _emissions(xml, n):
    out = []
    if xml is not None:
        for ch in xml.findall(".//Image/Dimensions/Channels/Channel")[:n]:
            text = ch.findtext("EmissionWavelength")
            try:
                out.append(float(text) if text else None)
            except ValueError:
                out.append(None)
    return out + [None] * (n - len(out))


def _interval_s(reader):
    try:
        interval = reader.time_interval
    except Exception:
        return None
    if interval is None:
        return None
    seconds = interval.total_seconds() if hasattr(interval, "total_seconds") else float(interval)
    return seconds if np.isfinite(seconds) and seconds > 0 else None
