"""TIFF reader: multipage / hyperstack / OME TIFFs in any order of dimensions.

A TIFF is a list of 2D pages (planes). How those planes make up Z, channels, time and
positions is written in the file's metadata, when there is any: ImageJ hyperstacks,
OME-TIFF and files written by tifffile describe it; a plain multipage TIFF does not.

The layout of the pixels is described like Bio-Formats' *DimensionOrder*: the axes from
the **fastest** changing to the slowest, e.g. ``XYCZT`` (planes go C1 Z1, C2 Z1, ... C1 Z2,
...) or ``XYZCT`` (all Z of channel 1, then all Z of channel 2). Any order works, also
with X/Y not first (``CXYZT``: channels interleaved pixel by pixel). Letters: X, Y, Z, C
(channel), T (time), P (stage position / series) and S (RGB samples).

When the file says how it is arranged it opens as such (channels as separate layers,
with names, colours and pixel size). When it does not, or with the reader "TIFF, choose
the dimension order", a dialog asks for the order and the sizes. Large files are read
lazily, plane by plane.
"""
from __future__ import annotations

import math
import re
import threading
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field

import numpy as np

from ._bioio import NO_LAYERS, Scene, scene_layers, stem_of

LETTERS = "TPCZYXS"  # the order layers keep (C is split into layers, S is RGB)
PRESETS = ("XYCZT", "XYZCT", "XYCTZ", "XYZTC", "XYTCZ", "XYTZC")
LAZY_BYTES = 1 << 30  # files larger than this are read plane by plane when shown
_UM = {"micron": 1.0, "microns": 1.0, "um": 1.0, "µm": 1.0, "\\u00b5m": 1.0, "micrometer": 1.0,
       "nm": 1e-3, "nanometer": 1e-3, "mm": 1e3, "millimeter": 1e3, "cm": 1e4, "centimeter": 1e4,
       "m": 1e6, "meter": 1e6, "inch": 25400.0, "in": 25400.0}
_SECONDS = {"s": 1.0, "sec": 1.0, "second": 1.0, "ms": 1e-3, "millisecond": 1e-3, "min": 60.0,
            "minute": 60.0, "h": 3600.0, "hour": 3600.0}


@dataclass
class Layout:
    """How the pixels of a file are arranged: `order` from the fastest axis to the slowest
    (e.g. "XYCZT") and the size of each axis."""

    order: str
    sizes: dict

    @classmethod
    def parse(cls, order, sizes):
        order = clean_order(order)
        return cls(order, {a: int(sizes[a]) for a in order})

    @property
    def total(self):
        return math.prod(self.sizes[a] for a in self.order)

    def describe(self):
        return " · ".join(f"{a} {self.sizes[a]}" for a in reversed(self.order))

    def to_dict(self):
        return {"order": self.order, "sizes": dict(self.sizes)}


def clean_order(text):
    """'XY-Z-C-T', 'xy z c t' or 'XYZCT' -> 'XYZCT'; checks the letters."""
    order = re.sub(r"[^A-Za-z]", "", str(text)).upper()
    bad = sorted(set(order) - set(LETTERS))
    if bad:
        raise ValueError(f"unknown dimension {', '.join(bad)} (use X, Y, Z, C, T, P or S)")
    twice = sorted({a for a in order if order.count(a) > 1})
    if twice:
        raise ValueError(f"{', '.join(twice)} appears more than once")
    if "X" not in order or "Y" not in order:
        raise ValueError("the order needs X and Y")
    return order


@dataclass
class SeriesInfo:
    """One image (series) of a TIFF and what its metadata says."""

    index: int
    name: str
    shape: tuple  # as stored, tifffile's axes
    axes: str
    dtype: object
    page_elems: int  # values per plane
    layout: Layout  # read from the metadata, or a guess
    known: bool  # the metadata says how the planes are arranged
    source: str  # "ImageJ hyperstack", "OME-TIFF", ...
    zyx_um: tuple = (None, None, None)
    time_interval_s: float | None = None
    channel_names: list = field(default_factory=list)
    emissions: list = field(default_factory=list)
    colormaps: list = field(default_factory=list)
    ranges: list = field(default_factory=list)  # display range per channel (ImageJ)
    rgb: bool = False

    @property
    def total(self):
        return math.prod(self.shape)

    @property
    def n_planes(self):
        return self.total // self.page_elems

    @property
    def plane(self):
        """(Y, X) size of a plane as stored."""
        sizes = dict(zip(self.axes, self.shape))
        return sizes.get("Y", 1), sizes.get("X", 1)

    def summary(self):
        y, x = self.plane
        text = f"{self.n_planes:,} planes of {x} × {y} ({np.dtype(self.dtype).name})"
        return text + (f"; {self.source} says {self.layout.describe()}" if self.known
                       else "; the file does not say how the planes are arranged")


def napari_get_reader(path):
    """Open TIFFs; ask for the dimension order only when the file does not say it."""
    path = _one(path)
    return read_tiff if path and _is_tiff(path) else None


def napari_get_reader_ask(path):
    """Open TIFFs, always choosing the dimension order (for files whose metadata is
    missing or wrong)."""
    path = _one(path)
    if not path or not _is_tiff(path):
        return None
    return lambda p: read_tiff(p, ask_always=True)


def _one(path):
    if isinstance(path, list):
        return path[0] if len(path) == 1 else None
    return path


def _is_tiff(path):
    return str(path).lower().endswith((".tif", ".tiff", ".btf", ".tf8", ".ome.tif", ".ome.tiff"))


# ---- reading --------------------------------------------------------------------------------


def read_tiff(path, layout=None, ask=None, choose=None, ask_always=False, selection=None):
    """LayerData for a TIFF.

    layout: a `Layout` (or its dict) to use instead of the file's; ask(info) -> Layout or
    None (cancelled) is the dimension-order dialog (default: `ask_layout`, only shown
    with a running Qt application); choose: the image chooser for files with several
    images (see `_chooser.choose_images`); selection: (indices, stacked) to reopen.
    """
    infos = inspect(path)
    if not infos:
        raise ValueError(f"{path}: no images in this TIFF")
    stem = stem_of(path)
    if stem.lower().endswith(".ome"):
        stem = stem[:-4]
    stack = False
    if selection is not None:
        indices, stack = [i for i in selection[0] if i < len(infos)], bool(selection[1])
        infos = [infos[i] for i in indices] or infos[:1]
    elif len(infos) > 1:
        if choose is None:
            from ._chooser import choose_images as choose
        picked = choose(stem, [_preview_scene(i) for i in infos])
        if picked is None:
            return NO_LAYERS
        indices, stack = picked
        infos = [infos[i] for i in indices]
    if not infos:
        return NO_LAYERS
    first = infos[0]
    if isinstance(layout, dict):
        layout = Layout.parse(layout["order"], layout["sizes"])
    if layout is None and (ask_always or not first.known):
        ask = ask or ask_layout
        layout = ask(first)
        if layout is None:  # cancelled
            return NO_LAYERS
    scenes = []
    for info in infos:
        use = layout if layout is not None and layout.total == info.total else info.layout
        scenes.append(_scene(path, info, use))
    out = scene_layers(scenes, stem, str(path), stack=stack)
    stacked = len(out) == 1 and len(scenes) > 1
    for k, (_, kwargs, _) in enumerate(out):
        ranges = (scenes[0] if stacked else scenes[k]).ranges if k < len(scenes) else []
        limits = kwargs.get("contrast_limits")
        if ranges and isinstance(limits, list) and len(limits) == len(ranges) and \
                all(isinstance(v, list) for v in limits):
            kwargs["contrast_limits"] = [list(r) if r[1] > r[0] else v for r, v in zip(ranges, limits)]
        elif ranges and len(ranges) == 1 and ranges[0][1] > ranges[0][0]:
            kwargs["contrast_limits"] = list(ranges[0])
        metas = kwargs.get("metadata")
        for meta in metas if isinstance(metas, list) else [metas]:
            if isinstance(meta, dict):
                meta["ibif_tiff_layout"] = (layout or first.layout).to_dict()
    return out


def _preview_scene(info):
    """A Scene for the image chooser (no pixels read)."""
    import dask.array as da

    lay = info.layout
    axes = "".join(a for a in LETTERS if a in lay.order)
    shape = tuple(lay.sizes[a] for a in axes)
    return Scene(info.index, info.name, da.zeros(shape, dtype=info.dtype, chunks=-1), axes, [], (None,) * 3, None)


def _scene(path, info, layout):
    """The series as a Scene with axes in layer order (T P C Z Y X S)."""
    if layout.total != info.total:
        raise ValueError(f"the sizes give {layout.total:,} values but the image has {info.total:,} "
                         f"({layout.describe()})")
    data = _arrange(_planes(path, info), info, layout)
    same = layout.to_dict() == info.layout.to_dict()
    n_c = layout.sizes.get("C", 1)
    names = info.channel_names if len(info.channel_names) == n_c else []
    scene = Scene(
        index=info.index, name=info.name, data=data,
        axes="".join(a for a in LETTERS if a in layout.order),
        channel_names=names, zyx_um=info.zyx_um, time_interval_s=info.time_interval_s,
        emissions=info.emissions if len(info.emissions) == n_c else [],
        colormaps=info.colormaps if len(info.colormaps) == n_c else [],
    )
    scene.ranges = info.ranges if same and len(info.ranges) == n_c else []
    return scene


def _planes(path, info):
    """The series as (planes, values per plane): numpy for small files, else a dask array
    reading one plane at a time."""
    import tifffile

    n = info.n_planes
    nbytes = info.total * np.dtype(info.dtype).itemsize
    if nbytes <= LAZY_BYTES:
        with tifffile.TiffFile(str(path)) as tif:
            return np.asarray(tif.series[info.index].asarray()).reshape(n, info.page_elems)
    import dask.array as da

    reader = _PlaneReader(str(path), info.index)

    def load(block_id=None):
        return reader.read(block_id[0]).reshape(1, info.page_elems)

    return da.map_blocks(load, dtype=info.dtype, chunks=((1,) * n, (info.page_elems,)))


class _PlaneReader:
    """Reads single planes of one series from an open file (thread-safe)."""

    def __init__(self, path, series):
        self.path, self.series = path, series
        self._tif, self._lock = None, threading.Lock()

    def read(self, i):
        import tifffile

        with self._lock:
            if self._tif is None:
                self._tif = tifffile.TiffFile(self.path)
            return np.asarray(self._tif.series[self.series].asarray(key=int(i)))


def _arrange(planes, info, layout):
    """Reshape (planes, values per plane) into the layout and put the axes in layer order."""
    order = layout.order
    slow_first = order[::-1]
    shape = [layout.sizes[a] for a in slow_first]
    # The fastest axes that fill exactly one plane: then each plane stays one chunk.
    k, size = 0, 1
    while k < len(order) and size < info.page_elems:
        size *= layout.sizes[order[k]]
        k += 1
    aligned = size == info.page_elems
    if not isinstance(planes, np.ndarray) and not aligned:
        planes = np.asarray(planes)  # planes cut across: read it all
    if isinstance(planes, np.ndarray):
        data = planes.reshape(shape)
    else:
        n_slow = len(order) - k
        data = planes.reshape(tuple(shape[:n_slow]) + (info.page_elems,)) if n_slow else \
            planes.reshape((info.page_elems,))
        data = data.reshape(shape)
    target = [a for a in LETTERS if a in order]
    return data.transpose([slow_first.index(a) for a in target])


# ---- what the file says ---------------------------------------------------------------------


def inspect(path):
    """A SeriesInfo per image in the file (reads metadata only)."""
    import tifffile

    out = []
    with tifffile.TiffFile(str(path)) as tif:
        ome = _ome(tif)
        ij = tif.imagej_metadata if tif.is_imagej else None
        for index, series in enumerate(tif.series):
            if series.kind and series.kind.lower() in ("thumbnail", "label", "macro"):
                continue
            out.append(_info(tif, index, series, ij, ome))
    return out


def _info(tif, index, series, ij, ome):
    shape, axes = tuple(int(s) for s in series.shape), str(series.axes).upper()
    key = series.keyframe
    page_elems = int(math.prod(key.shape)) or 1
    rgb = "S" in axes and axes.endswith("S") and dict(zip(axes, shape))["S"] in (3, 4) \
        and int(getattr(key, "photometric", 0)) == 2
    kind = (series.kind or "").lower()
    source = {"imagej": "ImageJ hyperstack", "ome": "OME-TIFF", "shaped": "tifffile"}.get(kind, kind or "TIFF")
    layout, known = _layout_from_axes(shape, axes, rgb, key)
    info = SeriesInfo(index=index, name=str(series.name or f"Image {index + 1}"), shape=shape, axes=axes,
                      dtype=series.dtype, page_elems=page_elems, layout=layout, known=known,
                      source=source, rgb=rgb)
    _scale(info, series, key, ij, ome)
    if ij is not None:
        _imagej_channels(info, ij)
    if ome is not None:
        _ome_channels(info, ome, index)
    return info


def _layout_from_axes(shape, axes, rgb, key):
    """(Layout, known) from tifffile's axes, e.g. 'ZCYX' -> 'XYCZ'. Axes tifffile could not
    name (I, Q: just "images") make the layout a guess: all planes as Z."""
    roles = []
    known = True
    for a, n in zip(axes, shape):
        if a in "TZCYX":
            role = a
        elif a == "S":
            role = "S" if rgb else "C"
        elif a == "E":
            role = "C"  # spectral (lambda) planes: one layer each
        elif a in "APRMV":
            role = "P"  # angles / positions / tiles / views
        else:  # I, Q, H, L, ...: tifffile does not know what they are
            role = "?"
        roles.append(role)
    named = [r for r, n in zip(roles, shape) if r != "?" or n > 1]
    dup = any(named.count(r) > 1 for r in set(named) if r != "?")
    if "?" in [r for r, n in zip(roles, shape) if n > 1] or dup or "Y" not in roles or "X" not in roles:
        known = False
        page = str(key.axes).upper()
        page_sizes = dict(zip(page, key.shape))
        fast = "".join(("S" if rgb else "C") if a == "S" else a for a in page[::-1] if a in "YXS")
        sizes = {("S" if rgb else "C") if a == "S" else a: int(page_sizes[a]) for a in page if a in "YXS"}
        n = int(math.prod(shape) // max(1, math.prod(key.shape)))
        order = fast + ("Z" if n > 1 else "")
        if n > 1:
            sizes["Z"] = n
        return Layout(order, sizes), known
    order, sizes = "", {}
    for r, n in zip(roles[::-1], shape[::-1]):
        if r == "?":  # length 1
            continue
        if n == 1 and r not in "YX":
            continue
        order += r
        sizes[r] = int(n)
    return Layout(order, sizes), known


def _scale(info, series, key, ij, ome):
    """Voxel size (µm) and time step, from tifffile's coordinates, else ImageJ / OME /
    resolution tags."""
    z = y = x = None
    scales = getattr(series, "coord_scales", None)
    units = getattr(series, "coord_units", None) or {}
    try:
        scales = dict(scales or {})
    except Exception:
        scales = {}
    for axis in "ZYX":
        if axis in scales and scales[axis]:
            factor = _UM.get(str(units.get(axis, "micrometer")).lower())
            if factor:
                value = abs(float(scales[axis])) * factor
                z, y, x = (value, y, x) if axis == "Z" else (z, value, x) if axis == "Y" else (z, y, value)
    if "T" in scales and scales["T"]:
        factor = _SECONDS.get(str(units.get("T", "s")).lower(), 1.0)
        info.time_interval_s = abs(float(scales["T"])) * factor
    if ij is not None:
        unit = _UM.get(str(ij.get("unit", "")).lower())
        if unit and (x is None or y is None):
            rx, ry = _resolution(key)
            if rx and ry:
                x, y = unit / rx, unit / ry
        if unit and z is None and ij.get("spacing"):
            z = float(ij["spacing"]) * unit
        if info.time_interval_s is None and ij.get("finterval") and int(ij.get("frames", 1) or 1) > 1:
            info.time_interval_s = float(ij["finterval"])
    if ome is not None and (x is None or y is None or z is None):
        pixels = _ome_pixels(ome, info.index)
        if pixels is not None:
            def size(axis):
                value = pixels.get(f"PhysicalSize{axis}")
                return float(value) * _UM.get(pixels.get(f"PhysicalSize{axis}Unit", "µm").lower(), 1.0) \
                    if value else None
            x, y, z = x or size("X"), y or size("Y"), z or size("Z")
            if info.time_interval_s is None and pixels.get("TimeIncrement"):
                info.time_interval_s = float(pixels["TimeIncrement"]) * _SECONDS.get(
                    pixels.get("TimeIncrementUnit", "s").lower(), 1.0)
    if (x is None or y is None) and ij is None and ome is None:
        rx, ry = _resolution(key)
        unit = {2: 25400.0, 3: 1e4}.get(int(getattr(key, "resolutionunit", 1) or 1))
        if rx and ry and unit and not (unit == 25400.0 and rx in (72.0, 96.0, 300.0)):  # dpi defaults
            x, y = unit / rx, unit / ry
    info.zyx_um = (z, y, x)


def _resolution(key):
    try:
        rx, ry = key.resolution
        return (float(rx) or None), (float(ry) or None)
    except Exception:
        return None, None


def _imagej_channels(info, ij):
    """Channel names, colours and display ranges from ImageJ metadata (and the Olympus /
    Bio-Formats "Info" text it often carries)."""
    n = info.layout.sizes.get("C", 1)
    text = str(ij.get("Info") or "")
    names = _numbered(text, r"\[Channel (\d+) Parameters\] DyeName = (.+)")
    excitation = _numbered(text, r"\[Channel (\d+) Parameters\] ExcitationWavelength = ([\d.]+)")
    emission = _numbered(text, r"\[Channel (\d+) Parameters\] EmissionWavelength = ([\d.]+)")
    if not names:
        names = _numbered(text, r"(?m)^\s*Channel ?(\d+)\s*[:_ ]?Name\s*=\s*(.+)$")
    if len(names) == n:
        info.channel_names = [s.strip() for s in names]
    if len(excitation) == n or len(emission) == n:
        info.emissions = [_emission(_float(excitation, i), _float(emission, i)) for i in range(n)]
    luts = ij.get("LUTs") or []
    if len(luts) == n:
        info.colormaps = [_lut_name(lut) for lut in luts]
    ranges = ij.get("Ranges") or ()
    if len(ranges) == 2 * n:
        info.ranges = [[float(ranges[2 * i]), float(ranges[2 * i + 1])] for i in range(n)]


def _numbered(text, pattern):
    found = {}
    for m in re.finditer(pattern, text):
        found.setdefault(int(m.group(1)), m.group(2).strip())
    return [found[k] for k in sorted(found)] if found and sorted(found) == list(range(min(found), min(found) + len(found))) else []


def _float(values, i):
    try:
        return float(values[i])
    except (IndexError, ValueError, TypeError):
        return None


def _emission(excitation, emission):
    """A wavelength for choosing a channel's colour. Microscopes often store the filter
    set, not the dye's emission, so the excitation laser (plus a typical Stokes shift)
    is the more reliable guide."""
    if excitation:
        return excitation + 30
    return emission


def _lut_name(lut):
    """'red', 'green', ... for a pure-colour ImageJ LUT; None for grey or other LUTs."""
    try:
        end = tuple(int(v > 127) for v in np.asarray(lut)[:3, -1])
    except Exception:
        return None
    return {(1, 0, 0): "red", (0, 1, 0): "green", (0, 0, 1): "blue", (0, 1, 1): "cyan",
            (1, 0, 1): "magenta", (1, 1, 0): "yellow"}.get(end)


def _ome(tif):
    if not tif.is_ome:
        return None
    try:
        return ET.fromstring(tif.ome_metadata)
    except Exception:
        return None


def _ome_image(root, index):
    images = [el for el in root if el.tag.endswith("}Image") or el.tag == "Image"]
    return images[index] if index < len(images) else None


def _ome_pixels(root, index):
    image = _ome_image(root, index)
    if image is None:
        return None
    for el in image:
        if el.tag.endswith("Pixels"):
            return el.attrib
    return None


def _ome_channels(info, root, index):
    image = _ome_image(root, index)
    if image is None:
        return
    pixels = next((el for el in image if el.tag.endswith("Pixels")), None)
    if pixels is None:
        return
    channels = [el for el in pixels if el.tag.endswith("Channel")]
    n = info.layout.sizes.get("C", 1)
    if len(channels) != n:
        return
    names = [c.attrib.get("Name") or c.attrib.get("Fluor") for c in channels]
    if all(names):
        info.channel_names = names
    info.emissions = [_emission(_float([c.attrib.get("ExcitationWavelength")], 0),
                                _float([c.attrib.get("EmissionWavelength")], 0)) for c in channels]
    colours = [_ome_color(c.attrib.get("Color")) for c in channels]
    if any(colours):
        info.colormaps = colours


def _ome_color(value):
    if value is None:
        return None
    try:
        v = int(value) & 0xFFFFFFFF
    except ValueError:
        return None
    rgb = ((v >> 24) & 255, (v >> 16) & 255, (v >> 8) & 255)
    return _lut_name(np.array([[0, rgb[0]], [0, rgb[1]], [0, rgb[2]]]))


# ---- asking -----------------------------------------------------------------------------------


def ask_layout(info):
    """The dimension-order dialog; without a running Qt application (scripts) the file's
    own layout (or the guess) is used."""
    from qtpy.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        return info.layout
    from ._tiff_dialog import DimensionOrderDialog

    dialog = DimensionOrderDialog(info, parent=app.activeWindow())
    return dialog.chosen() if dialog.exec() else None


def fill_sizes(order, sizes, total):
    """Sizes for `order`, with at most one unknown (0) size worked out from `total`."""
    sizes = {a: int(sizes.get(a, 0) or 0) for a in order}
    unknown = [a for a in order if not sizes[a]]
    if len(unknown) == 1:
        known = math.prod(sizes[a] for a in order if sizes[a])
        if known and total % known == 0:
            sizes[unknown[0]] = total // known
    return sizes


__all__ = ["Layout", "SeriesInfo", "read_tiff", "inspect", "napari_get_reader",
           "napari_get_reader_ask", "clean_order", "fill_sizes", "PRESETS"]
