"""Nikon ND2 reader: one image layer per channel, lazily loaded, with physical scale.

`layer_data` also builds the layers for the CZI reader (`_czi.py`).

Each layer's metadata records the axes it keeps (e.g. "TPZYX") so the Dimensions and
Cellpose panels know which axis is time, stage position or Z.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

# Axes the reader keeps as layer dimensions, outermost first. "C" is split into layers and
# "S" (RGB samples) is shown as an RGB layer.
AXIS_NAMES = {"T": "time", "P": "position", "Z": "z", "Y": "y", "X": "x"}


def napari_get_reader(path):
    if isinstance(path, list):
        if len(path) != 1:
            return None
        path = path[0]
    suffix = str(path).lower().rsplit(".", 1)[-1]
    if suffix == "nd2":
        return read_nd2
    if suffix == "czi":
        from ._czi import read_czi

        return read_czi
    return None


def read_nd2(path):
    import nd2

    path = str(path)
    with nd2.ND2File(path) as f:
        data = f.to_dask()
        sizes = dict(f.sizes)
        names, emissions = _channel_info(f)
        voxel = f.voxel_size()
        interval = _time_interval_s(f)
    return layer_data(
        data, sizes, names, emissions, (voxel.z, voxel.y, voxel.x), interval, Path(path).stem, path
    )


def layer_data(data, sizes, channel_names, emissions, zyx_um, time_interval_s, stem, source):
    """Build napari LayerData tuples from an ND2 array and its metadata (pure, testable)."""
    axes = "".join(sizes)
    if tuple(sizes.values()) != tuple(data.shape):
        raise ValueError(f"ND2 sizes {sizes} do not match array shape {data.shape}")

    rgb = "S" in axes
    if rgb and axes[-1] != "S":
        data = np.moveaxis(data, axes.index("S"), -1)
        axes = axes.replace("S", "") + "S"
    kept = axes.replace("C", "").replace("S", "")

    z, y, x = zyx_um
    spatial = {"Z": z or 1.0, "Y": y or 1.0, "X": x or 1.0}
    scale = [spatial.get(a, 1.0) for a in kept]

    meta = {
        "ibif_axes": kept,
        "ibif_source": source,
        "ibif_voxel_um": {"Z": z, "Y": y, "X": x},
        "ibif_time_interval_s": time_interval_s,
    }
    kwargs = {"scale": scale, "metadata": meta}
    kwargs.update(_axis_kwargs(kept))

    if rgb:
        return [(data, {**kwargs, "name": stem, "rgb": True}, "image")]

    if "C" not in axes:
        channel = channel_names[0] if channel_names else None
        kwargs.update(
            name=f"{stem} {channel}" if channel else stem,
            colormap=_colormap(emissions[0] if emissions else None, 0, 1),
            contrast_limits=_contrast_limits(data, axes, None),
            metadata={**meta, "ibif_channel": channel},
        )
        return [(data, kwargs, "image")]

    c_axis = axes.index("C")
    n = data.shape[c_axis]
    channel_names = list(channel_names)[:n] + [f"C{i}" for i in range(len(channel_names), n)]
    emissions = list(emissions)[:n] + [None] * (n - len(emissions))
    kwargs.update(
        channel_axis=c_axis,
        name=[f"{stem} {c}" for c in channel_names],
        colormap=[_colormap(e, i, n) for i, e in enumerate(emissions)],
        contrast_limits=[_contrast_limits(data, axes, i) for i in range(n)],
        blending="additive",
        metadata=[{**meta, "ibif_channel": c} for c in channel_names],
    )
    return [(data, kwargs, "image")]


def _axis_kwargs(kept):
    """Axis labels/units for napari versions whose layers accept them (0.6+)."""
    try:
        import inspect

        from napari.layers import Image

        params = inspect.signature(Image).parameters
    except Exception:  # napari not importable (e.g. headless tests)
        return {}
    out = {}
    if "axis_labels" in params:
        out["axis_labels"] = [AXIS_NAMES.get(a, a) for a in kept]
    if "units" in params:
        out["units"] = ["um" if a in "ZYX" else "pixel" for a in kept]
    return out


def _channel_info(f):
    names, emissions = [], []
    for ch in getattr(f.metadata, "channels", None) or []:
        names.append(ch.channel.name or f"C{len(names)}")
        emissions.append(getattr(ch.channel, "emissionLambdaNm", None))
    return names, emissions


def _time_interval_s(f):
    try:
        for loop in f.experiment:
            if loop.type in ("TimeLoop", "NETimeLoop"):
                params = loop.parameters
                period = getattr(params, "periodMs", None)
                if period is None and getattr(params, "periods", None):
                    period = params.periods[0].periodMs
                if period:
                    return float(period) / 1000.0
    except Exception:
        pass
    return None


def _colormap(emission_nm, index, n):
    if emission_nm:
        if emission_nm < 500:
            return "blue"
        if emission_nm < 560:
            return "green"
        if emission_nm < 650:
            return "magenta"
        return "red"
    if n == 1:
        return "gray"
    return ["blue", "green", "magenta", "red", "cyan", "yellow"][index % 6]


def _contrast_limits(data, axes, channel):
    """Percentile limits from the middle plane, so opening a large file stays fast."""
    index = []
    for a, size in zip(axes, data.shape):
        if a in "YXS":
            index.append(slice(None))
        elif a == "C":
            index.append(channel)
        else:
            index.append(size // 2)
    plane = np.asarray(data[tuple(index)])
    lo, hi = np.percentile(plane, (0.1, 99.9))
    if hi <= lo:
        hi = lo + 1
    return [float(lo), float(hi)]
