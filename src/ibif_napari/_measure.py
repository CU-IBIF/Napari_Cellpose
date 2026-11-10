"""Per-object measurements: size, position and intensity statistics per channel.

`measure_block` measures one labels volume; `measure_layer` measures a napari labels
layer over all its time points / positions, against image layers on the same grid;
`combine` joins the tables of objects that share label numbers (nucleus 17, cell 17 and
cytoplasm 17) into one row per object.
"""
from __future__ import annotations

import weakref

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

STATS = ("mean", "median", "min", "max", "std", "sum")
_FUNCS = {
    "mean": ndi.mean, "median": ndi.median, "min": ndi.minimum, "max": ndi.maximum,
    "std": ndi.standard_deviation, "sum": ndi.sum,
}

# Latest tables per viewer, so a saved workspace includes them; panels that show them
# listen for tables restored from a workspace.
_tables = weakref.WeakKeyDictionary()
_listeners = weakref.WeakKeyDictionary()


def remember(viewer, tables, notify=False):
    _tables[viewer] = dict(tables)
    if notify:
        for callback in list(_listeners.get(viewer, [])):
            callback(dict(tables))


def listen(viewer, callback):
    _listeners.setdefault(viewer, []).append(callback)


def tables_of(viewer):
    return dict(_tables.get(viewer, {}))


def measure_block(labels, channels=None, spacing=None, origin=None, units="µm", pieces=False):
    """One row per object of a 2D or 3D labels array.

    channels: {name: array of the same shape}; spacing: voxel size per axis;
    origin: world coordinate of voxel 0 (for centroids); pieces: add a 'pieces' column,
    the number of separate parts of each object (e.g. the vesicles of a cell, which share
    its number). Returns a DataFrame with 'label', size, centroid and '<channel> <stat>'
    columns.
    """
    labels = np.asarray(labels)
    nd = labels.ndim
    spacing = np.asarray(spacing if spacing is not None else (1.0,) * nd, float)
    origin = np.asarray(origin if origin is not None else (0.0,) * nd, float)
    ids = np.unique(labels)
    ids = ids[ids != 0]
    size_name = f"volume ({units}³)" if nd == 3 else f"area ({units}²)"
    if ids.size == 0:
        return pd.DataFrame(columns=["label", size_name])
    voxels = ndi.sum(np.ones(labels.shape, np.uint8), labels, ids).astype(np.int64)
    size = voxels * float(np.prod(spacing))
    data = {"label": ids.astype(np.int64), "voxels": voxels, size_name: size}
    if pieces:
        from ._secondary import count_pieces

        counts = count_pieces(labels)
        data["pieces"] = np.array([counts.get(int(i), 0) for i in ids], np.int64)
    # Diameter of the sphere (3D) or disc (2D) with the same volume / area.
    data[f"equivalent diameter ({units})"] = (
        np.cbrt(6 * size / np.pi) if nd == 3 else np.sqrt(4 * size / np.pi)
    )
    centroids = np.asarray(ndi.center_of_mass(np.ones(labels.shape, np.uint8), labels, ids), float)
    centroids = centroids.reshape(len(ids), nd) * spacing + origin
    for k, axis in enumerate("ZYX"[-nd:] if nd <= 3 else range(nd)):
        data[f"centroid {axis} ({units})"] = centroids[:, k]
    for name, image in (channels or {}).items():
        image = np.asarray(image)
        if image.shape != labels.shape:
            raise ValueError(f"{name} ({image.shape}) and the objects ({labels.shape}) differ in size")
        image = image.astype(np.float64, copy=False)
        for stat in STATS:
            data[f"{name} {stat}"] = np.asarray(_FUNCS[stat](image, labels, ids), float)
    return pd.DataFrame(data)


def measure_layer(viewer, labels, images=(), progress=None):
    """Measure a napari labels layer over all its time points / positions.

    `images` are image layers on the same pixel grid (matched by world coordinates, so
    cropped Cellpose region results work). Adds 'T' and 'P' columns when the layer has
    those axes, and a 'pieces' column for layers of structures (vesicles, nucleoli...). `progress(done, total)` is called per block.
    """
    from ._util import aligned_image, spatial_ndim

    n_sp = spatial_ndim(labels)
    lead = labels.ndim - n_sp
    axes = (labels.metadata or {}).get("ibif_axes")
    lead_names = list(axes[:lead]) if axes and len(axes) == labels.ndim else [f"axis{i}" for i in range(lead)]
    spacing = [float(s) for s in labels.scale[-n_sp:]]
    units = "µm" if (labels.metadata or {}).get("ibif_voxel_um") else "px"
    data = np.asarray(labels.data)
    rows = []
    positions = list(np.ndindex(*data.shape[:lead]))
    for n, pos in enumerate(positions):
        if progress:
            progress(n, len(positions))
        channels = {img.name: aligned_image(viewer, img, labels, pos) for img in images}
        origin = [float(t) for t in labels.translate[-n_sp:]]
        table = measure_block(data[pos], channels, spacing, origin, units,
                              pieces=bool((labels.metadata or {}).get("ibif_pieces")))
        for name, i in zip(lead_names, pos):
            table.insert(0, name, i)
        rows.append(table)
    if progress:
        progress(len(positions), len(positions))
    out = pd.concat(rows, ignore_index=True) if rows else measure_block(np.zeros((1, 1), np.uint8))
    return out


def keys_of(table):
    """The columns that identify an object: time / position axes and the label."""
    return [c for c in table.columns if c in ("T", "P") or c.startswith("axis")] + ["label"]


def combine(tables):
    """Join tables of layers whose objects share label numbers: one row per object, each
    layer's columns prefixed with its name (e.g. 'nuclei volume', 'cells GFP mean')."""
    combined = None
    for name, table in tables.items():
        keys = keys_of(table)
        renamed = table.rename(columns={c: f"{name}: {c}" for c in table.columns if c not in keys})
        combined = renamed if combined is None else combined.merge(renamed, on=keys, how="outer")
    if combined is None:
        return pd.DataFrame()
    return combined.sort_values(keys_of(combined)).reset_index(drop=True)


def save_table(table, path):
    """CSV, or an Excel workbook for .xlsx."""
    path = str(path)
    if path.lower().endswith(".xlsx"):
        table.to_excel(path, index=False)
    else:
        table.to_csv(path, index=False)
    return path
