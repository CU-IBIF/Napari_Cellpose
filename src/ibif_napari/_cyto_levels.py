"""Building spatial levels and moving between them (Cytometry workbench, H4).

The workbench keeps one analysis per level (`_cytometry.Levels`); the
geometry of the hierarchy - which objects of the level below each object of a
level contains - is kept beside it (a sparse membership per level, and for
windows the seed of each) and handed to vtea-core's `hierarchy` engine
whenever a level is built or information crosses levels. Positions are the
analysis's own (`Cytometry.positions`: world units, time points and
positions kept far apart), so no window or region ever spans two of them.

- `build_windows` / `build_regions`: a new level from the current one.
- `to_root`: a value per object of a level, as a value per segmented object,
  so any level can be painted on the image: each cell takes its own window's
  value (windows) or its region's (regions), down through every level.
- `related`: a selection at one level as a selection at another (members
  below, containers above).
- `bring`: columns of one level onto another (down: from the window centred
  on each object, or averaged over every window containing it; up: averaged
  over the members).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse

KEY_COLUMNS = ("T", "P")


def _hierarchy(levels, names=None):
    """A vtea-core Hierarchy holding the levels (all, or `names` and their
    ancestors), with objects identified by row (id = row + 1) and positions in
    the analysis's world units."""
    from vtea_core.hierarchy import Hierarchy, ObjectLevel

    root = levels.root
    points = root.positions()
    table = pd.DataFrame({"object_id": np.arange(1, len(points) + 1)})
    for axis in range(points.shape[1]):
        table[f"centroid-{axis}"] = points[:, axis]
    h = Hierarchy(table, name=_root_name(levels), contacts=levels._contacts)
    wanted = None if names is None else _with_ancestors(levels, names)
    for _depth, name in levels.level_tree()[1:]:
        if wanted is not None and name not in wanted:
            continue
        info = levels._info[name]
        geometry = info.get("geometry")
        if geometry is None:
            raise ValueError(f"level {name!r} has no record of its members (built before levels kept them)")
        parent_points = h.centers(info["parent"]) if info["parent"] in h else None
        membership, seeds = geometry["membership"], geometry.get("seeds")
        if info["kind"] == "windows":
            centres = parent_points[seeds]
        else:
            m = sparse.csr_matrix(membership, dtype=float)
            centres = (m @ parent_points) / np.maximum(np.asarray(m.sum(axis=1)).ravel(), 1)[:, None]
        level_table = pd.DataFrame({"object_id": np.arange(1, membership.shape[0] + 1)})
        for axis in range(centres.shape[1]):
            level_table[f"centroid-{axis}"] = centres[:, axis]
        h.levels[name] = ObjectLevel(
            name=name, kind=info["kind"], table=level_table, parent=info["parent"],
            depth=levels.level_depth(name), membership=membership, seeds=seeds,
        )
    return h


def _root_name(levels):
    from ._cytometry import ROOT_LEVEL

    return ROOT_LEVEL


def _with_ancestors(levels, names):
    out = set()
    for name in names:
        while name is not None:
            out.add(name)
            name = levels._info[name]["parent"]
    return out


def contacts(levels):
    """Touching pairs of segmented objects as row ids (row + 1), read from the
    objects layer's masks once per table; None when the layer is not open."""
    from vtea_core.spatial import touching_pairs

    root = levels.root
    try:
        blocks = root.label_blocks()
    except ValueError:
        return None
    pairs = []
    table = root.table
    for rows, image, _scale in blocks:
        labels = table["label"].to_numpy(np.int64)[rows]
        lookup = np.full(int(max(labels.max(), image.max())) + 1, -1, np.int64)
        lookup[labels] = rows
        found = touching_pairs(image)
        if len(found):
            mapped = lookup[found]
            mapped = mapped[(mapped >= 0).all(axis=1)]
            pairs.append(mapped + 1)
    return np.vstack(pairs) if pairs else np.zeros((0, 2), np.int64)


def _centroid_table(state, rows=None, membership=None):
    """The centroid and time point / position columns of new level objects, in the
    form Measurements tables have them: a seed's (`rows`), or the mean of the members'
    (`membership`)."""
    cols = state.centroid_columns()
    table = state.table
    out = {}
    for axis, col in cols.items():
        unit = col.rsplit("(", 1)[-1].rstrip(")")
        values = table[col].to_numpy(float)
        if rows is not None:
            out[f"centroid {axis} ({unit})"] = values[rows]
        else:
            m = sparse.csr_matrix(membership, dtype=float)
            out[f"centroid {axis} ({unit})"] = (m @ values) / np.maximum(np.asarray(m.sum(axis=1)).ravel(), 1)
    for key in KEY_COLUMNS:
        if key in table.columns:
            values = table[key].to_numpy()
            if rows is not None:
                out[key] = values[rows]
            else:
                first = sparse.csr_matrix(membership).indices[sparse.csr_matrix(membership).indptr[:-1]]
                out[key] = values[first]
    return out


def windows_level(levels, parent, settings, types=None):
    """(table, geometry, provenance) of a windows level on `parent` from neighbourhood
    `settings` (method, radius, k, classes, seeds, members as population keys).
    `types`: the parent's per-object neighbourhood type (the window centred on an
    object takes its type)."""
    from vtea_core.hierarchy import build_windows

    from ._cytometry import NO_CLASS

    state = levels.level(parent)
    h = _hierarchy(levels, [parent])
    base = h[parent]
    classes = state.classes(settings["classes"])
    base.table["class"] = np.array([c if c is not None else NO_CLASS for c in classes], dtype=object)
    seeds = _population(state, settings.get("seeds"))
    members = _population(state, settings.get("members"))
    method = settings.get("method", "radius")
    radius, k = float(settings.get("radius") or 1.0), int(settings.get("k") or 10)
    level = build_windows(base, h.positions(parent), name="new", method=method, radius=radius, k=k,
                          classes="class", seeds=seeds, members=members, unit=state.unit())
    rows = level.seeds
    table = pd.DataFrame({"object_id": np.arange(1, len(rows) + 1)})
    table = table.assign(**_centroid_table(state, rows=rows))
    table["n objects"] = level.table["n_objects"].to_numpy(float)
    table[f"mean distance ({state.unit()})"] = level.table["mean_distance"].to_numpy(float)
    for col in level.table.columns:
        if str(col).endswith("_ClassFraction"):
            value = str(col)[len("Class_"):-len("_ClassFraction")]
            if value != NO_CLASS:
                table[f"fraction {value}"] = level.table[col].to_numpy(float) / 100.0
    if types is not None:
        table["neighbourhood type"] = np.asarray(types, np.int64)[rows]
    seed_ids = state.table["label"].to_numpy()[rows] if "label" in state.table.columns else rows + 1
    table["seed"] = seed_ids
    geometry = {"membership": level.membership, "seeds": rows}
    provenance = {"builder": "windows", "method": method, "radius": radius, "k": k,
                  "classes": settings.get("class_label", settings["classes"]),
                  "seeds": settings.get("seeds"), "members": settings.get("members"),
                  "analysis": settings.get("analysis")}
    return table, geometry, provenance


def regions_level(levels, parent, types, *, values=None, touching=True, d=0.0, tau=0.5, min_size=1,
                  sigma=None, source=""):
    """(table, geometry, provenance) of a regions level on `parent`: contiguous objects
    with the same `types` value (per parent row; 0 = none) merged. Touching: masks at
    the segmentation, seeds or members above it; `d` > 0 also joins centres that
    close; `tau` the member-proximity threshold (see vtea-core `hierarchy`)."""
    from vtea_core.hierarchy import build_regions

    state = levels.level(parent)
    if touching:
        levels.touching_pairs()
    h = _hierarchy(levels, [parent])
    base = h[parent]
    kinds = pd.Series(np.asarray(types, dtype=np.int64)).where(lambda s: s > 0)
    base.table["type"] = kinds.to_numpy()
    member_positions = h.positions(base.parent) if base.membership is not None else None
    level = build_regions(
        base, h.positions(parent), name="new", types="type", values=values,
        adjacency=h.adjacency(parent) if touching else None, d=d, member_positions=member_positions,
        sigma=sigma, tau=tau, min_size=min_size, unit=state.unit(),
    )
    table = pd.DataFrame({"object_id": np.arange(1, len(level) + 1)})
    table = table.assign(**_centroid_table(state, membership=level.membership))
    table["n objects"] = level.table["n_objects"].to_numpy(float)
    table["neighbourhood type"] = level.table["type"].to_numpy(np.int64)
    table[f"radius of gyration ({state.unit()})"] = level.table["radius_of_gyration"].to_numpy(float)
    table["elongation"] = level.table["elongation"].replace(np.inf, np.nan).to_numpy(float)
    geometry = {"membership": level.membership, "seeds": None}
    provenance = {"builder": "regions", "types": source, "touching": bool(touching), "d": float(d),
                  "tau": float(tau), "sigma": level.provenance.get("sigma"), "min_size": int(min_size)}
    return table, geometry, provenance


def _population(state, key):
    """A boolean row mask for a population key ([kind, key]); None = every object."""
    if not key or key[0] == "all":
        return None
    return np.asarray(state.mask(key[0], key[1]), bool)


def own_rows(levels, name):
    """For each segmented object, the row of the level-`name` object that is its own:
    the window centred on it (or on its own window at the level in between), or the
    region containing it; -1 for none."""
    from ._cytometry import ROOT_LEVEL

    chain = []
    while name != ROOT_LEVEL:
        chain.insert(0, name)
        name = levels._info[name]["parent"]
    mapping = np.arange(len(levels.root.table))
    for level_name in chain:
        geometry = levels._info[level_name]["geometry"]
        if geometry is None:
            raise ValueError(f"level {level_name!r} has no record of its members, so it cannot be shown "
                             "on the image")
        n_parent = geometry["membership"].shape[1]
        here = np.full(n_parent, -1, np.int64)
        if levels._info[level_name]["kind"] == "windows":
            here[geometry["seeds"]] = np.arange(len(geometry["seeds"]))
        else:
            coo = sparse.coo_matrix(geometry["membership"])
            here[coo.col] = coo.row
        mapping = np.where(mapping >= 0, here[np.maximum(mapping, 0)], -1)
    return mapping


def to_root(levels, name, values):
    """`values` (one per object of level `name`, 0 = none) as one per segmented object,
    through `own_rows`."""
    rows = own_rows(levels, name)
    values = np.asarray(values)
    return np.where(rows >= 0, values[np.maximum(rows, 0)], 0)


def related(levels, source, mask, target):
    """The objects of level `target` related to the selected (`mask`) objects of level
    `source`: the members they contain when `target` is below, the objects containing
    them when above, and through the levels' common ancestor otherwise."""
    h = _hierarchy(levels, [source, target])
    mask = np.asarray(mask, bool)
    if source == target:
        return mask.copy()
    if target in h.path(source):
        return h.members_of(source, mask, target)
    if source in h.path(target):
        return h.containers_of(source, mask, target)
    common = next(n for n in reversed(h.path(source)) if n in h.path(target))
    down = h.members_of(source, mask, common)
    return h.containers_of(common, down, target)


def bring(levels, source, columns, target, how="mean"):
    """Columns of level `source` as columns of level `target` (one row per target
    object). Down (target below source): `how="own"` the value of the target object's
    own source object, `"mean"` the mean over every source object containing it (the
    most common for whole numbers that are types or classes). Up: the mean over the
    target object's members."""
    h = _hierarchy(levels, [source, target])
    table = levels.level(source).frame()
    values = pd.DataFrame({c: np.asarray(table[c]) for c in columns})
    if target in h.path(source) and target != source:  # down
        if how == "own":
            rows = _own_between(levels, h, source, target)
            return pd.DataFrame({c: np.where(rows >= 0, values[c].to_numpy(float)[np.maximum(rows, 0)], np.nan)
                                 for c in columns})
        categorical = [c for c in columns if _is_type_column(c)]
        h[source].table = h[source].table.assign(**{c: values[c].to_numpy() for c in columns})
        out = h.reflect(source, target, columns, how="member", categorical=categorical)
        return out[columns]
    if source in h.path(target) and target != source:  # up
        h[source].table = h[source].table.assign(**{c: values[c].to_numpy(float) for c in columns})
        out = h.aggregate(target, source, columns, "mean")
        return out.rename(columns={f"mean_{c}": c for c in columns})[columns]
    raise ValueError(f"{source!r} and {target!r} are not one above the other")


def _own_between(levels, h, source, target):
    """For each object of `target` (below `source`), the row of its own `source`
    object, following seeds (windows) and containment (regions) up the chain."""
    chain = h.path(source)
    steps = chain[chain.index(target) + 1:]
    mapping = np.arange(len(h[target]))
    for level_name in steps:
        level = h[level_name]
        here = np.full(level.membership.shape[1], -1, np.int64)
        if level.kind == "windows":
            here[level.seeds] = np.arange(len(level.seeds))
        else:
            coo = sparse.coo_matrix(level.membership)
            here[coo.col] = coo.row
        mapping = np.where(mapping >= 0, here[np.maximum(mapping, 0)], -1)
    return mapping


def _is_type_column(name):
    return str(name).endswith((" neighbourhood", " cluster", " class", "neighbourhood type"))


def save_geometry(geometry, folder, sub):
    """Write a level's membership (and seeds) under `folder/sub`; returns the paths."""
    (folder / sub).mkdir(parents=True, exist_ok=True)
    sparse.save_npz(folder / sub / "membership.npz", sparse.csr_matrix(geometry["membership"]))
    out = {"membership": f"{sub}/membership.npz", "seeds": None}
    if geometry.get("seeds") is not None:
        np.save(folder / sub / "seeds.npy", np.asarray(geometry["seeds"]))
        out["seeds"] = f"{sub}/seeds.npy"
    return out


def load_geometry(folder, paths):
    if not paths or not paths.get("membership") or not (folder / paths["membership"]).exists():
        return None
    seeds = np.load(folder / paths["seeds"]) if paths.get("seeds") else None
    return {"membership": sparse.load_npz(folder / paths["membership"]).tocsr(), "seeds": seeds}
