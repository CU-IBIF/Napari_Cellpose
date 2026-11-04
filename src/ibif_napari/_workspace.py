"""Save and reopen a whole napari session ("workspace"), and save layers as files.

A workspace is a folder (name.ibif) holding:

    workspace.json        layers in order, their display settings, the view, sources
    layers/NN name.tif    labels and images made in napari (compressed TIFF)
    measurements/*.csv    the Measurements panel's tables

Images opened from microscope files (ND2 / CZI / LIF) are not copied: the workspace
stores where the file is (absolute, and relative to the workspace) and reopens it. With
`include_images=True` copies are written too, so the workspace can be shared with
someone who does not have the original files. Moving the workspace together with the
data folder keeps working (the relative path is tried when the absolute one fails).
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

FORMAT = "ibif-napari workspace"
VERSION = 1
SUFFIX = ".ibif"
LARGE = 2 * 1024**3  # copies bigger than this are written uncompressed, plane by plane


# ---- small helpers ---------------------------------------------------------------------


def _jsonable(value):
    """`value` with everything JSON cannot hold dropped (numpy scalars converted)."""
    if isinstance(value, dict):
        out = {}
        for k, v in value.items():
            v = _jsonable(v)
            if v is not _DROP:
                out[str(k)] = v
        return out
    if isinstance(value, (list, tuple)):
        items = [_jsonable(v) for v in value]
        return [v for v in items if v is not _DROP]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray) and value.size <= 1000:
        return value.tolist()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    return _DROP


_DROP = object()


def _safe(name):
    return "".join(c if c.isalnum() or c in " -_." else "_" for c in name).strip() or "layer"


def _array(data):
    return data[0] if isinstance(data, list) else data  # multiscale: full resolution


def write_tiff(path, data, compress=True):
    """Write an array (numpy or lazy dask) as TIFF; large ones plane by plane."""
    import tifffile

    data = _array(data)
    nbytes = int(np.prod(data.shape)) * np.dtype(data.dtype).itemsize
    if compress and nbytes <= LARGE:
        tifffile.imwrite(path, np.asarray(data), compression="zlib", bigtiff=nbytes > 3.5e9)
        return path
    mm = tifffile.memmap(path, shape=data.shape, dtype=data.dtype, bigtiff=True)
    lead = data.shape[:-2] if data.ndim > 2 else ()
    for index in np.ndindex(*lead):
        mm[index] = np.asarray(data[index])
    if not lead:
        mm[...] = np.asarray(data)
    mm.flush()
    del mm
    return path


def read_tiff(path):
    """A TIFF written by `write_tiff`: memory-mapped (lazy) when uncompressed."""
    import tifffile

    try:
        return tifffile.memmap(path)
    except ValueError:  # compressed
        return tifffile.imread(path)


# ---- saving -----------------------------------------------------------------------------


def _layer_entry(layer, folder, n, include_images, notes):
    from napari.layers import Image, Labels, Points, Shapes

    kind = layer._type_string
    entry = {
        "name": layer.name, "type": kind, "visible": bool(layer.visible),
        "opacity": float(layer.opacity), "blending": str(layer.blending),
        "scale": [float(s) for s in layer.scale], "translate": [float(t) for t in layer.translate],
        "metadata": _jsonable(dict(layer.metadata or {})),
    }
    if isinstance(layer, Image):
        entry.update(colormap=layer.colormap.name, contrast_limits=[float(c) for c in layer.contrast_limits],
                     gamma=float(layer.gamma), rgb=bool(layer.rgb))
        source = (layer.metadata or {}).get("ibif_source")
        if source and os.path.isabs(str(source)) and os.path.exists(source):
            entry["source"] = str(source)
            entry["source_rel"] = os.path.relpath(source, folder)
        if include_images or "source" not in entry:
            entry["file"] = f"layers/{n:02d} {_safe(layer.name)}.tif"
    elif isinstance(layer, Labels):
        entry["file"] = f"layers/{n:02d} {_safe(layer.name)}.tif"
    elif isinstance(layer, Shapes):
        entry.update(
            data=[np.asarray(d).tolist() for d in layer.data], shape_type=list(layer.shape_type),
            edge_width=[float(w) for w in layer.edge_width],
            edge_color=np.asarray(layer.edge_color).tolist(), face_color=np.asarray(layer.face_color).tolist(),
        )
    elif isinstance(layer, Points):
        entry.update(data=np.asarray(layer.data).tolist(), size=np.asarray(layer.size).tolist(),
                     face_color=np.asarray(layer.face_color).tolist())
    else:
        notes.append(f"{layer.name}: {kind} layers are not saved")
        return None
    return entry


def save_workspace(viewer, folder, include_images=False, progress=None):
    """Save the viewer's layers, view and measurements in `folder` (created; '.ibif' is
    added to the name if missing). Returns (folder, notes)."""
    from . import _measure

    folder = Path(folder)
    if folder.suffix != SUFFIX:
        folder = folder.with_name(folder.name + SUFFIX)
    (folder / "layers").mkdir(parents=True, exist_ok=True)
    notes, entries = [], []
    layers = list(viewer.layers)
    for n, layer in enumerate(layers):
        if progress:
            progress(n, len(layers))
        entry = _layer_entry(layer, folder, n, include_images, notes)
        if entry is None:
            continue
        if "file" in entry:
            write_tiff(folder / entry["file"], layer.data, compress=True)
        entries.append(entry)
    tables = _measure.tables_of(viewer)
    if tables:
        (folder / "measurements").mkdir(exist_ok=True)
    table_files = {}
    for name, table in tables.items():
        rel = f"measurements/{_safe(name)}.csv"
        table.to_csv(folder / rel, index=False)
        table_files[name] = rel
    camera = getattr(getattr(viewer, "scene", None), "camera", None) or viewer.camera
    state = {
        "format": FORMAT, "version": VERSION,
        "layers": entries,
        "dims": {"ndisplay": viewer.dims.ndisplay, "order": list(viewer.dims.order),
                 "point": [float(p) for p in viewer.dims.point],
                 "axis_labels": [str(a) for a in viewer.dims.axis_labels]},
        "camera": {"center": [float(c) for c in camera.center], "zoom": float(camera.zoom),
                   "angles": [float(a) for a in camera.angles]},
        "measurements": table_files,
    }
    tmp = folder / "workspace.json.tmp"
    tmp.write_text(json.dumps(state, indent=1))
    os.replace(tmp, folder / "workspace.json")
    if progress:
        progress(len(layers), len(layers))
    return folder, notes


def save_layers(layers, folder):
    """Save layers as files: images and labels as TIFF, shapes and points as CSV.
    Returns the paths written."""
    import pandas as pd
    from napari.layers import Image, Labels, Points, Shapes

    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    paths = []
    for layer in layers:
        base = folder / _safe(layer.name)
        if isinstance(layer, (Image, Labels)):
            paths.append(write_tiff(base.with_suffix(".tif"), layer.data))
        elif isinstance(layer, Points):
            data = np.asarray(layer.data)
            cols = [f"axis-{i}" for i in range(data.shape[1])] if data.size else []
            pd.DataFrame(data, columns=cols).to_csv(base.with_suffix(".csv"), index=False)
            paths.append(base.with_suffix(".csv"))
        elif isinstance(layer, Shapes):
            rows = []
            for i, (shape, kind) in enumerate(zip(layer.data, layer.shape_type)):
                for j, vertex in enumerate(np.asarray(shape)):
                    rows.append({"index": i, "shape-type": kind, "vertex-index": j,
                                 **{f"axis-{k}": v for k, v in enumerate(vertex)}})
            pd.DataFrame(rows).to_csv(base.with_suffix(".csv"), index=False)
            paths.append(base.with_suffix(".csv"))
    return paths


# ---- opening ------------------------------------------------------------------------------


def _find_source(entry, folder):
    candidates = [entry.get("source"), entry.get("source_rel") and str(folder / entry["source_rel"])]
    if entry.get("source"):
        candidates.append(str(folder / Path(entry["source"]).name))  # next to the workspace
        candidates.append(str(folder.parent / Path(entry["source"]).name))
    for c in candidates:
        if c and os.path.exists(c):
            return os.path.normpath(c)
    return None


def _open_source(path, selection=None):
    """{layer name: (data, kwargs, type)} for every layer a file opens as, one per channel.
    `selection` = (scene indices, stacked) reopens the same images of a LIF project."""
    from napari.layers.utils.stack_utils import split_channels

    from ._reader import napari_get_reader

    suffix = Path(path).suffix.lower()
    if suffix == ".lif":
        from ._lif import read_lif

        def choose(title, scenes):
            if selection is not None:
                return [i for i in selection[0] if i < len(scenes)], selection[1]
            return list(range(len(scenes))), False

        layer_data = read_lif(path, choose=choose)
    elif napari_get_reader(path) is not None:
        layer_data = napari_get_reader(path)(path)
    else:
        from npe2 import io_utils

        layer_data = io_utils.read([path], plugin_name=None)
    out = {}
    for item in layer_data:
        if len(item) < 2 or item[0] is None:
            continue
        data, kwargs, kind = item[0], dict(item[1]), item[2] if len(item) > 2 else "image"
        if kind == "image" and kwargs.get("channel_axis") is not None:
            axis = kwargs.pop("channel_axis")
            for d, k, t in split_channels(data, axis, **kwargs):
                out[k["name"]] = (d, k, t)
        else:
            out[kwargs.get("name", Path(path).stem)] = (data, kwargs, kind)
    return out


def load_workspace(viewer, path, replace=True):
    """Open a workspace folder (or its workspace.json) into `viewer`. Returns notes on
    anything that could not be restored."""
    from napari.layers import Layer

    from . import _measure

    path = Path(path)
    folder = path.parent if path.name == "workspace.json" else path
    state = json.loads((folder / "workspace.json").read_text())
    if state.get("format") != FORMAT:
        raise ValueError(f"{folder} is not an IBIF napari workspace")
    notes, sources = [], {}
    if replace:
        viewer.layers.clear()
    for entry in state["layers"]:
        kind, name = entry["type"], entry["name"]
        kwargs = {k: entry[k] for k in ("name", "visible", "opacity", "blending", "scale", "translate", "metadata")}
        data = None
        if kind == "image" and "source" in entry:
            src = _find_source(entry, folder)
            if src is not None:
                meta = entry.get("metadata", {})
                selection = (tuple(meta["ibif_scenes"]), bool(meta.get("ibif_stacked"))) \
                    if meta.get("ibif_scenes") is not None else None
                key = (src, selection)
                if key not in sources:
                    try:
                        sources[key] = _open_source(src, selection)
                    except Exception as e:  # unreadable: fall back to a copy below
                        sources[key] = {}
                        notes.append(f"could not open {src}: {e}")
                found = sources[key].get(name)
                if found is not None:
                    data = found[0]
                    kwargs["metadata"] = {**found[1].get("metadata", {}), **kwargs["metadata"]}
                    kwargs["metadata"]["ibif_source"] = src
        if data is None and "file" in entry and (folder / entry["file"]).exists():
            data = read_tiff(folder / entry["file"])
        if kind in ("image", "labels") and data is None:
            where = entry.get("source") or entry.get("file")
            notes.append(f"{name}: data not found ({where})")
            continue
        if kind == "image":
            kwargs.update(colormap=entry.get("colormap", "gray"), contrast_limits=entry.get("contrast_limits"),
                          gamma=entry.get("gamma", 1.0), rgb=entry.get("rgb", False))
        elif kind == "shapes":
            data = [np.asarray(d) for d in entry["data"]]
            kwargs.update(shape_type=entry["shape_type"], edge_width=entry["edge_width"],
                          edge_color=np.asarray(entry["edge_color"]), face_color=np.asarray(entry["face_color"]))
        elif kind == "points":
            data = np.asarray(entry["data"])
            kwargs.update(size=np.asarray(entry["size"]), face_color=np.asarray(entry["face_color"]))
        try:
            viewer.add_layer(Layer.create(data, kwargs, kind))
        except Exception as e:
            notes.append(f"{name}: could not be restored ({e})")
    _restore_view(viewer, state, notes)
    tables = {}
    if state.get("measurements"):
        import pandas as pd

        for name, rel in state["measurements"].items():
            if (folder / rel).exists():
                tables[name] = pd.read_csv(folder / rel)
    if tables:
        _measure.remember(viewer, tables, notify=True)
    return notes


def _restore_view(viewer, state, notes):
    dims = state.get("dims", {})
    try:
        if len(dims.get("point", [])) == viewer.dims.ndim:
            if len(dims.get("order", [])) == viewer.dims.ndim:
                viewer.dims.order = tuple(dims["order"])
            if dims.get("axis_labels") and len(dims["axis_labels"]) == viewer.dims.ndim:
                viewer.dims.axis_labels = tuple(dims["axis_labels"])
            viewer.dims.ndisplay = dims.get("ndisplay", 2)
            camera = getattr(getattr(viewer, "scene", None), "camera", None) or viewer.camera
            cam = state.get("camera", {})
            if cam.get("center"):
                camera.center = tuple(cam["center"])
            if cam.get("zoom"):
                camera.zoom = cam["zoom"]
            if cam.get("angles"):
                camera.angles = tuple(cam["angles"])
            viewer.dims.point = tuple(dims["point"])  # after the camera (it resets sliders)
    except Exception as e:
        notes.append(f"the view could not be restored ({e})")


def napari_get_reader(path):
    if isinstance(path, list):
        path = path[0] if len(path) == 1 else ""
    path = str(path).rstrip("/\\")
    if path.endswith(SUFFIX) and (Path(path) / "workspace.json").exists():
        return read_workspace
    return None


def read_workspace(path):
    """napari reader for .ibif folders (drag and drop, File > Open): restores the
    workspace into the current viewer, next to what is already open."""
    import napari

    if isinstance(path, list):
        path = path[0]
    viewer = napari.current_viewer()
    if viewer is None:
        return [(None,)]
    notes = load_workspace(viewer, path, replace=False)
    if notes:
        from napari.utils.notifications import show_warning

        show_warning("Workspace opened with problems:\n" + "\n".join(notes))
    return [(None,)]
