"""Leica LIF reader (via bioio-lif). A project with several images asks which to open."""
from __future__ import annotations

import numpy as np

from ._bioio import NO_LAYERS, import_reader, read_scene, scene_layers, stem_of

# Leica LUT names that are also napari colormaps.
_LUTS = {"red", "green", "blue", "cyan", "magenta", "yellow", "gray", "grey"}


def read_lif(path, reader=None, choose=None):
    """Read a LIF file. `reader` (a bioio-lif Reader) and `choose` (the image chooser,
    see `_chooser.choose_images`) can be passed for testing."""
    if reader is None:
        reader = import_reader("bioio_lif", "bioio-lif", "Leica LIF")(str(path))
    if choose is None:
        from ._chooser import choose_images as choose
    stem = stem_of(path)
    scenes = [read_scene(reader, i) for i in range(len(reader.scenes))]
    stack = False
    if len(scenes) > 1:
        picked = choose(stem, scenes)
        if picked is None:  # cancelled
            return NO_LAYERS
        indices, stack = picked
        scenes = [scenes[i] for i in indices]
    luts = _lut_names(path, [s.index for s in scenes])
    intervals = _intervals(path, [s.index for s in scenes if s.time_interval_s is None])
    for s in scenes:
        if s.time_interval_s is None:
            s.time_interval_s = intervals.get(s.index)
    for s in scenes:
        s.colormaps = [
            ("gray" if lut.lower() == "grey" else lut.lower()) if lut and lut.lower() in _LUTS else None
            for lut in luts.get(s.index, [])
        ]
    return scene_layers(scenes, stem, str(path), stack=stack)


def _lut_names(path, indices):
    """Each image's channel colours (Leica LUT names), ordered like the channels."""
    try:
        from liffile import LifFile
    except ImportError:
        return {}
    out = {}
    try:
        with LifFile(str(path)) as lif:
            for i in indices:
                elements = lif.images[i].xml_element.findall(
                    "./Data/Image/ImageDescription/Channels/ChannelDescription"
                )
                elements.sort(key=lambda el: int(el.attrib.get("BytesInc", 0)))
                out[i] = [el.attrib.get("LUTName", "") for el in elements]
    except Exception:
        return {}
    return out


def _intervals(path, indices):
    """Seconds between time points of each image, from its T dimension (bioio-lif only
    reports it for plate scans), or {} if unknown."""
    if not indices:
        return {}
    try:
        from liffile import LifFile
    except ImportError:
        return {}
    out = {}
    try:
        with LifFile(str(path)) as lif:
            for i in indices:
                times = lif.images[i].coords.get("T")
                if times is not None and len(times) > 1:
                    step = float(np.mean(np.diff(np.asarray(times, float))))
                    if np.isfinite(step) and step > 0:
                        out[i] = step
    except Exception:
        return {}
    return out
