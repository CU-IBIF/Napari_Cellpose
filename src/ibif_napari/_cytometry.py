"""Cytometry state of a viewer: the object table, gates and the phenotype tree.

The analysis engine is vtea-core (VTEA's headless library): plot gates and their
hierarchy are `vtea_core.gates.GateSet`, cell types are a
`vtea_core.classes.PhenotypeTree` whose rules are class expressions over the table
(`vtea_core.classes.expression`). This module holds one analysis per napari viewer and
adds what IBIF napari needs around it:

- the **object table**: a Measurements table (one row per object and time point /
  position), plus derived axis columns such as ``asinh(CD3 mean/150)``;
- **gates** drawn on the plot over two columns; a gate drawn while the plot shows a gate
  is that gate's subgate, and one drawn while it shows a phenotype is limited to that
  phenotype (its *scope*);
- **image gates**: polygons drawn on the image; an object is in one when its centroid is
  (Y/X, so a 3D object is in a region drawn on any of its planes);
- each gate becomes a column of the table (``gate: <name>``), so a phenotype rule can
  use it: ```gate: CD3+` AND NOT `gate: CD4+```;
- saving all of it with the workspace.
"""
from __future__ import annotations

import re
import weakref

import numpy as np
import pandas as pd

GATE_PREFIX = "gate: "
IMAGE_GATE_PREFIX = "image gate: "
PALETTE = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2",
           "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
           "#c5b0d5", "#c49c94", "#f7b6d2", "#dbdb8d", "#9edae5", "#393b79"]
TRANSFORMS = ("linear", "log10", "asinh")
ASINH_COFACTOR = 150.0  # typical for imaging intensities (counts); flow uses 5
KEYS = ("T", "P", "label")
_TRANSFORMED = re.compile(r"^(log10|asinh)\((.*?)(?:/([0-9.]+))?\)$")

_states = weakref.WeakKeyDictionary()


def state_of(viewer):
    """The viewer's cytometry analysis (created when first asked for)."""
    if viewer not in _states:
        _states[viewer] = Cytometry(viewer)
    return _states[viewer]


def axis_column(column, transform="linear", cofactor=ASINH_COFACTOR):
    """The name of a (possibly transformed) axis column."""
    if transform == "linear":
        return column
    if transform == "log10":
        return f"log10({column})"
    if transform == "asinh":
        return f"asinh({column}/{cofactor:g})"
    raise ValueError(transform)


def split_axis(name):
    """(column, transform, cofactor) of an axis column name."""
    m = _TRANSFORMED.match(name)
    if not m:
        return name, "linear", ASINH_COFACTOR
    return m.group(2), m.group(1), float(m.group(3) or ASINH_COFACTOR)


def quote(column):
    """A column as it is written in a phenotype rule."""
    return column if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_.]*", column) else f"`{column}`"


class Cytometry:
    """One viewer's cytometry analysis."""

    def __init__(self, viewer):
        from vtea_core.classes import PhenotypeTree
        from vtea_core.gates import GateSet

        self.viewer = viewer
        self.source = None  # name of the Measurements table
        self.layer_name = None  # the labels layer its objects are in
        self.table = None  # pandas DataFrame, index 0..n-1
        self.gates = GateSet()
        self.image_gates = {}  # name -> {"polygons": [[(y, x), ...]], "color"}
        # What each gate was drawn within, besides its parent gate: ("phenotype", name) or
        # ("image", name). Keyed by ("gate", id) / ("image", name).
        self.scope = {}
        self.order = []  # gates of both kinds in creation order (a gate may use earlier ones)
        self.tree = PhenotypeTree()
        # Clustering / mapping runs: name -> settings; their output columns (cluster ids,
        # map coordinates) live in `derived`, one row per table row.
        self.analyses = {}
        self.derived = None
        self._listeners = []
        self._cache = None

    # ---- events --------------------------------------------------------------------------

    def listen(self, callback):
        self._listeners.append(callback)

    def changed(self):
        self._cache = None
        for callback in list(self._listeners):
            callback()

    # ---- the object table ------------------------------------------------------------------

    def set_table(self, source, table, layer_name):
        """Use a Measurements table. Gates and phenotypes are kept (their rules still
        apply if the columns exist)."""
        same = self.table is not None and len(self.table) == len(table) and self.source == source
        self.source = source
        self.layer_name = layer_name
        self.table = table.reset_index(drop=True).copy()
        if not same:  # clustering results belong to the objects they were run on
            self.analyses, self.derived = {}, None
        self.changed()

    def features(self):
        """The measurements of the segmented objects that can be plotted: size (volume /
        area, equivalent diameter), position (centroids) and intensity statistics per
        channel, from the Measurements table. Not the ids (T, P, label), the raw voxel
        count (the volume in µm³ / px³ is there) or gate columns."""
        if self.table is None:
            return []
        out = [c for c in self.table.columns
               if not _is_key(c) and not _is_voxels(c) and pd.api.types.is_numeric_dtype(self.table[c])
               and not pd.api.types.is_bool_dtype(self.table[c])]
        if self.derived is not None:
            out += list(self.derived.columns)
        return out

    def measured_features(self):
        """Numeric measurement columns that describe the objects (what clustering is run
        on): size and intensities, not where the objects are."""
        derived = set(self.derived.columns) if self.derived is not None else set()
        return [c for c in self.features() if c not in derived and " centroid " not in f" {c}"]

    def prefix(self):
        """The column prefix of the objects layer's own measurements in a joined table
        ("cells: " in "All objects", where every layer's columns are prefixed), else ""."""
        if self.table is None or not self.layer_name:
            return ""
        prefix = f"{self.layer_name}: "
        return prefix if any(str(c).startswith(prefix) for c in self.table.columns) else ""

    def centroid_columns(self):
        """{axis: column} of the objects' centroids (world units), preferring the objects
        layer's own in a joined table."""
        return _centroid_columns(self.table, self.prefix()) if self.table is not None else {}

    def describe(self):
        """What is being analysed, in words: how many objects, which segmented layer they
        are in, and which measurements there are."""
        if self.table is None:
            return "No objects chosen yet."
        feats = self.measured_features()
        size = [f for f in feats if any(w in f for w in ("volume", "area", "diameter", "pieces"))]
        channels = sorted({f.rsplit(" ", 1)[0].split(": ")[-1] for f in feats
                           if f.rsplit(" ", 1)[-1] in _STATS})
        found, total = self.objects_found()
        where = (f"segmented in <b>{self.layer_name}</b>" if self.layer_name else "no objects layer chosen")
        text = (f"<b>{len(self.table):,} objects</b> {where}, measured in {self.source}: "
                f"{len(size)} size and {len(feats) - len(size)} intensity measurements"
                + (f" ({', '.join(channels)})" if channels else "") + ".")
        if total and found < total:
            text += (f"<br><font color='#d62728'>Only {found:,} of the {total:,} labels in the table "
                     f"are in {self.layer_name}: choose the layer that was measured.</font>")
        return text

    def objects_found(self):
        """(labels of the table found in the objects layer, labels in the table)."""
        layer = self.labels_layer()
        if layer is None or self.table is None or "label" not in self.table.columns:
            return 0, 0
        wanted = np.unique(self.table["label"].dropna().to_numpy(np.int64))
        present = np.unique(np.asarray(layer.data))
        return int(np.isin(wanted, present).sum()), len(wanted)

    def column(self, name):
        """A table column, or a transformed one computed on the fly."""
        if name in self.table.columns:
            return self.table[name].to_numpy(float)
        if self.derived is not None and name in self.derived.columns:
            return self.derived[name].to_numpy(float)
        column, transform, cofactor = split_axis(name)
        values = self.column(column) if column not in self.table.columns else self.table[column].to_numpy(float)
        if transform == "log10":
            with np.errstate(divide="ignore", invalid="ignore"):
                return np.log10(np.where(values > 0, values, np.nan))
        return np.arcsinh(values / cofactor)

    def frame(self):
        """The table with every gate as a boolean column (cached until a change)."""
        if self.table is None:
            return None
        if self._cache is not None:
            return self._cache
        frame = self.table.copy()
        if self.derived is not None:
            for c in self.derived.columns:
                frame[c] = self.derived[c].to_numpy()
        for gate in self.gates:  # the axis columns gates are drawn on
            for axis in (gate.x_axis, gate.y_axis):
                if axis not in frame.columns:
                    frame[axis] = self.column(axis)
        # In creation order: a gate drawn within a phenotype or an image gate needs the
        # columns of the gates made before it.
        for kind, key in self.order:
            if kind == "gate":
                gate = self.gates.get(key)
                mask = self.gates.mask(key, frame)
                column = GATE_PREFIX + gate.name
            else:
                mask = self._image_gate_mask(self.image_gates[key], frame)
                column = IMAGE_GATE_PREFIX + key
            mask = mask & self._scope_mask((kind, key), frame)
            frame[column] = mask
        self._cache = frame
        return frame

    def _scope_mask(self, item, frame):
        scope = self.scope.get(item)
        if not scope:
            return np.ones(len(frame), bool)
        kind, key = scope
        if kind == "phenotype" and key in self.tree:
            return self.tree.members(key, frame, exclusive=True)
        if kind == "image" and IMAGE_GATE_PREFIX + key in frame.columns:
            return frame[IMAGE_GATE_PREFIX + key].to_numpy(bool)
        if kind == "gate" and key in self.gates and GATE_PREFIX + self.gates.get(key).name in frame.columns:
            return frame[GATE_PREFIX + self.gates.get(key).name].to_numpy(bool)
        return np.ones(len(frame), bool)

    # ---- populations ---------------------------------------------------------------------

    def populations(self):
        """(kind, key, label) of everything the plot can show."""
        out = [("all", None, "All objects")]
        for gate in self.gates:
            depth = 0
            parent = gate.parent_id
            while parent:
                depth += 1
                parent = self.gates.get(parent).parent_id
            out.append(("gate", gate.id, "  " * depth + f"Gate: {gate.name}"))
        for name in self.image_gates:
            out.append(("image", name, f"Image gate: {name}"))
        for node in self.tree:
            out.append(("phenotype", node.name, "  " * (self.tree.depth(node.name) - 1)
                        + f"Phenotype: {node.name}"))
        return out

    def mask(self, kind, key):
        frame = self.frame()
        if frame is None:
            return np.zeros(0, bool)
        if kind == "all":
            return np.ones(len(frame), bool)
        if kind == "gate":
            return frame[GATE_PREFIX + self.gates.get(key).name].to_numpy(bool)
        if kind == "image":
            return frame[IMAGE_GATE_PREFIX + key].to_numpy(bool)
        if kind == "phenotype":
            return self.tree.members(key, frame, exclusive=True)
        raise ValueError(kind)

    # ---- gates -----------------------------------------------------------------------------

    def gate_names(self):
        return {g.name for g in self.gates} | set(self.image_gates)

    def unique_name(self, base):
        names = self.gate_names() | {p.name for p in self.tree}
        if base not in names:
            return base
        k = 2
        while f"{base} {k}" in names:
            k += 1
        return f"{base} {k}"

    def add_gate(self, name, x_axis, y_axis, vertices, population=("all", None), color=None):
        """A plot gate over two (axis) columns, drawn while `population` was shown."""
        from vtea_core.gates import Gate

        if name in self.gate_names():
            raise ValueError(f"there is already a gate called {name!r}")
        kind, key = population
        parent = key if kind == "gate" else None
        gate = Gate(name=name, x_axis=x_axis, y_axis=y_axis, vertices=np.asarray(vertices, float),
                    parent_id=parent, color=color or self._next_color())
        self.gates.add(gate)
        self.order.append(("gate", gate.id))
        if kind in ("phenotype", "image"):
            self.scope[("gate", gate.id)] = (kind, key)
        self.changed()
        return gate

    def _next_color(self):
        return PALETTE[(len(self.order) + len(self.tree)) % len(PALETTE)]

    def add_image_gate(self, name, polygons, population=("all", None), color=None):
        """Objects whose centroid (Y, X, world units) is inside any of `polygons`."""
        if name in self.gate_names():
            raise ValueError(f"there is already a gate called {name!r}")
        polygons = [np.asarray(p, float)[:, -2:].tolist() for p in polygons if len(p) >= 3]
        if not polygons:
            raise ValueError("draw at least one polygon or rectangle")
        self.image_gates[name] = {"polygons": polygons, "color": color or self._next_color()}
        self.order.append(("image", name))
        kind, key = population
        if kind != "all":
            self.scope[("image", name)] = (kind, key)
        self.changed()
        return name

    def remove_gate(self, kind, key):
        """Remove a gate (and its subgates). Phenotype rules that used it are kept and
        reported by `broken_phenotypes`."""
        for item, scope in list(self.scope.items()):  # gates drawn within this one go too
            if scope == (kind, key) and item in self.order:
                self.remove_gate(*item)
        if kind == "gate":
            for child in self.gates.children(key):
                self.remove_gate("gate", child.id)
            self.gates.remove(key)
        else:
            self.image_gates.pop(key)
        self.scope.pop((kind, key), None)
        self.order.remove((kind, key))
        self.changed()

    def rename_gate(self, kind, key, new):
        if new in self.gate_names():
            raise ValueError(f"there is already a gate called {new!r}")
        old = self.gates.get(key).name if kind == "gate" else key
        if kind == "gate":
            self.gates.get(key).name = new
        else:
            self.image_gates[new] = self.image_gates.pop(key)
            self.order[self.order.index(("image", key))] = ("image", new)
            for item, scope in list(self.scope.items()):
                if scope == ("image", key):
                    self.scope[item] = ("image", new)
            if ("image", key) in self.scope:
                self.scope[("image", new)] = self.scope.pop(("image", key))
        prefix = GATE_PREFIX if kind == "gate" else IMAGE_GATE_PREFIX
        old_col, new_col = quote(prefix + old), quote(prefix + new)
        for node in self.tree:  # keep phenotype rules pointing at it
            node.rule = node.rule.replace(old_col, new_col)
        self.changed()

    def gate_column(self, kind, key):
        if kind == "gate":
            return GATE_PREFIX + self.gates.get(key).name
        return IMAGE_GATE_PREFIX + key

    def _image_gate_mask(self, gate, frame):
        from vtea_core.gates import polygon_gate

        cols = _centroid_columns(frame, self.prefix())
        ys = frame[cols["Y"]].to_numpy(float) if "Y" in cols else None
        xs = frame[cols["X"]].to_numpy(float) if "X" in cols else None
        if ys is None or xs is None:
            return np.zeros(len(frame), bool)
        mask = np.zeros(len(frame), bool)
        for poly in gate["polygons"]:
            mask |= polygon_gate(ys, xs, np.asarray(poly, float))
        return mask

    # ---- phenotypes ------------------------------------------------------------------------

    def add_phenotype(self, name, rule="", parent=None, color=None):
        from vtea_core.classes import Phenotype

        if name in self.gate_names():
            raise ValueError(f"{name!r} is already a gate name")
        node = self.tree.add(Phenotype(name, rule, parent, color or self._next_color()))
        try:
            self._check_rule(name)
        except ValueError:
            self.tree.remove(name)
            raise
        self.changed()
        return node

    def phenotype_from(self, kind, key, name, parent=None, color=None):
        """A phenotype whose rule is membership of a gate."""
        column = self.gate_column(kind, key)
        if color is None:
            color = self.gates.get(key).color if kind == "gate" else self.image_gates[key]["color"]
        return self.add_phenotype(name, quote(column), parent, color)

    def edit_phenotype(self, name, rule=None, color=None, new_name=None):
        node = self.tree.get(name)
        old_rule = node.rule
        if rule is not None:
            node.rule = rule
            try:
                self._check_rule(name)
            except ValueError:
                node.rule = old_rule
                raise
        if color is not None:
            node.color = color
        if new_name and new_name != name:
            if new_name in self.gate_names():
                raise ValueError(f"{new_name!r} is already a gate name")
            self.tree.rename(name, new_name)
            for item, scope in list(self.scope.items()):
                if scope == ("phenotype", name):
                    self.scope[item] = ("phenotype", new_name)
        self.changed()

    def remove_phenotype(self, name):
        removed = self.tree.remove(name)
        for item, scope in list(self.scope.items()):
            if scope[0] == "phenotype" and scope[1] in removed:
                del self.scope[item]  # the gate stays, now over its parent gate / all objects
        self.changed()
        return removed

    def _check_rule(self, name):
        from vtea_core.classes import ExpressionError

        from vtea_core.classes import parse

        rule = self.tree.get(name).rule
        if not rule.strip():
            return
        try:
            parse(rule)
        except ExpressionError as e:
            raise ValueError(f"the rule cannot be read: {e}") from e
        self._cache = None
        frame = self.frame() if self.table is not None else None
        if frame is not None:
            missing = self.tree.missing_columns(frame).get(name)
            if missing:
                raise ValueError("the rule uses unknown columns: " + ", ".join(sorted(missing)))

    def broken_phenotypes(self):
        frame = self.frame()
        return {} if frame is None else self.tree.missing_columns(frame)

    def assignment(self):
        """Per object: phenotype, path, and phenotype at each level (None without a table)."""
        frame = self.frame()
        if frame is None:
            return None
        if self.broken_phenotypes():
            raise ValueError("fix the phenotypes whose rules use missing columns first")
        return self.tree.assign(frame)

    def counts(self):
        frame = self.frame()
        if frame is None or self.broken_phenotypes():
            return None
        return self.tree.counts(frame)

    def typed_table(self):
        """The object table with every gate column and the phenotype columns."""
        frame = self.frame()
        out = frame.copy()
        assignment = self.assignment()
        for column in assignment.columns:
            out[column] = assignment[column].to_numpy()
        return out

    # ---- showing on the image ---------------------------------------------------------------

    def labels_layer(self):
        from napari.layers import Labels

        if self.layer_name and self.layer_name in self.viewer.layers:
            layer = self.viewer.layers[self.layer_name]
            if isinstance(layer, Labels):
                return layer
        return None

    def paint(self, values, name, colors):
        """A labels layer over the objects layer, with each object's value from
        `values` (one int per table row, 0 = hidden) coloured by `colors` {value: hex}."""
        from napari.utils.colormaps import DirectLabelColormap

        layer = self.labels_layer()
        if layer is None:
            raise ValueError(f"the objects layer {self.layer_name!r} is not open")
        data = np.asarray(layer.data)
        out = np.zeros(data.shape, np.uint16 if max(values, default=0) < 65535 else np.uint32)
        lead = data.ndim - _spatial_ndim(layer)
        table = self.table
        keys = [k for k in table.columns if _is_key(k) and k != "label"][:lead]
        positions = list(np.ndindex(*data.shape[:lead])) if lead else [()]
        values = np.asarray(values)
        for pos in positions:
            rows = np.ones(len(table), bool)
            for k, p in zip(keys, pos):
                rows &= table[k].to_numpy() == p
            labels = table.loc[rows, "label"].to_numpy(np.int64)
            if not len(labels):
                continue
            block = data[pos]
            # Each object's value goes on its own segmented pixels (label -> value).
            lut = np.zeros(int(max(block.max(), labels.max())) + 1, out.dtype)
            lut[labels] = values[rows]
            out[pos] = lut[block]
        cmap = DirectLabelColormap(color_dict={None: (0, 0, 0, 0), 0: (0, 0, 0, 0),
                                               **{int(v): c for v, c in colors.items()}})
        if name in self.viewer.layers:
            target = self.viewer.layers[name]
            target.data = out
            target.colormap = cmap
        else:
            target = self.viewer.add_labels(out, name=name, scale=layer.scale, translate=layer.translate,
                                            opacity=0.7, metadata={"ibif_cytometry": True})
            target.colormap = cmap
        return target

    def show_phenotypes(self, level=None):
        """Colour every object by its phenotype (deepest, or at a level 1, 2, ...)."""
        assignment = self.assignment()
        column = "phenotype" if level is None else f"phenotype level {level}"
        names = [p.name for p in self.tree]
        code = {n: i + 1 for i, n in enumerate(names)}
        values = [code.get(v, 0) for v in assignment[column]]
        colors = {code[n]: self.tree.get(n).color or PALETTE[i % len(PALETTE)] for i, n in enumerate(names)}
        suffix = "phenotypes" if level is None else f"phenotypes level {level}"
        return self.paint(values, f"{self.layer_name} {suffix}", colors)

    def show_population(self, kind, key, color=None):
        mask = self.mask(kind, key)
        if color is None:
            color = (self.gates.get(key).color if kind == "gate" else
                     self.image_gates[key]["color"] if kind == "image" else
                     self.tree.get(key).color if kind == "phenotype" else "#ffff00")
        values = np.where(mask, 1, 0)
        return self.paint(values, f"{self.layer_name} selection", {1: color})

    # ---- clustering and maps -----------------------------------------------------------------

    def unique_analysis_name(self, base="Clustering"):
        k = 1
        while f"{base} {k}" in self.analyses:
            k += 1
        return f"{base} {k}"

    def matrix(self, population, features, asinh=False, cofactor=ASINH_COFACTOR):
        """(row indices, n x features matrix) of the objects in `population` that have
        every feature (rows with NaN are left out)."""
        mask = self.mask(*population)
        values = np.column_stack([self.column(f) for f in features]) if features else np.zeros((len(mask), 0))
        if asinh:
            values = np.arcsinh(values / cofactor)
        ok = mask & np.all(np.isfinite(values), axis=1)
        rows = np.flatnonzero(ok)
        return rows, values[rows]

    def add_analysis(self, name, settings, rows, labels=None, embedding=None):
        """Store a clustering / map run on table rows `rows`: cluster ids (1..k, by size;
        0 = not clustered) in "<name> cluster", map coordinates in "<name> <map> 1/2"."""
        if name in self.analyses:
            raise ValueError(f"there is already an analysis called {name!r}")
        if name in self.gate_names() or name in self.tree:
            raise ValueError(f"{name!r} is already a gate or phenotype name")
        n = len(self.table)
        derived = self.derived if self.derived is not None else pd.DataFrame(index=self.table.index)
        columns = {}
        if labels is not None:
            ids = _by_size(np.asarray(labels))
            values = np.zeros(n, dtype=np.int64)
            values[rows] = ids
            columns["cluster"] = f"{name} cluster"
            derived[columns["cluster"]] = values
        if embedding is not None:
            emb = np.asarray(embedding, float)
            label = settings.get("map", "map")
            columns["map"] = []
            for k in range(emb.shape[1]):
                col = f"{name} {label} {k + 1}"
                values = np.full(n, np.nan)
                values[rows] = emb[:, k]
                derived[col] = values
                columns["map"].append(col)
        self.derived = derived
        self.analyses[name] = {**settings, "columns": columns, "n_objects": int(len(rows))}
        self.changed()
        return self.analyses[name]

    def remove_analysis(self, name):
        info = self.analyses.pop(name)
        cols = [info["columns"].get("cluster")] + list(info["columns"].get("map", []))
        self.derived = self.derived.drop(columns=[c for c in cols if c and c in self.derived.columns])
        if self.derived.shape[1] == 0:
            self.derived = None
        self.changed()

    def clusters(self, name):
        """Cluster id per table row (0 = not clustered)."""
        return self.derived[self.analyses[name]["columns"]["cluster"]].to_numpy(np.int64)

    def profiles(self, name, features=None):
        """vtea-core cluster profiles (mean, z, counts) over the analysis's features."""
        from vtea_core.clustering import cluster_profiles, describe_clusters

        info = self.analyses[name]
        features = features or info["features"]
        labels = self.clusters(name)
        frame = pd.DataFrame({f: self.column(f) for f in features})
        if info.get("asinh"):
            frame = np.arcsinh(frame / info.get("cofactor", ASINH_COFACTOR))
        out = cluster_profiles(frame, np.where(labels > 0, labels, -1), features)
        out["description"] = describe_clusters(out["z"])
        return out

    def label_clusters(self, name, names, parent=None):
        """Phenotypes from clusters: `names` maps cluster id -> phenotype name; clusters
        given the same name become one phenotype. Returns the phenotypes made."""
        column = quote(self.analyses[name]["columns"]["cluster"])
        groups = {}
        for cluster, pheno in names.items():
            if pheno and pheno.strip():
                groups.setdefault(pheno.strip(), []).append(int(cluster))
        clash = [p for p in groups if p in self.tree or p in self.gate_names()]
        if clash:
            raise ValueError("these names are already used: " + ", ".join(clash))
        made = []
        for pheno, ids in groups.items():
            rule = f"{column} == {ids[0]}" if len(ids) == 1 else f"{column} in [{', '.join(map(str, sorted(ids)))}]"
            made.append(self.add_phenotype(pheno, rule, parent))
        return made

    def show_clusters(self, name):
        labels = self.clusters(name)
        colors = {int(c): PALETTE[(int(c) - 1) % len(PALETTE)] for c in np.unique(labels) if c > 0}
        return self.paint(labels, f"{self.layer_name} {name}", colors)

    # ---- saving ----------------------------------------------------------------------------

    def to_dict(self):
        from vtea_core.gates.io import gate_set_to_dict

        return {"version": 1, "source": self.source, "layer": self.layer_name,
                "gates": gate_set_to_dict(self.gates), "image_gates": self.image_gates,
                "order": [list(item) for item in self.order],
                "scope": [[list(item), list(scope)] for item, scope in self.scope.items()],
                "phenotypes": self.tree.to_dict(), "analyses": self.analyses}

    def save_derived(self, folder):
        """Write the clustering / map columns next to the workspace; returns the path."""
        if self.derived is None:
            return None
        keys = [k for k in KEYS if k in self.table.columns]
        out = self.table[keys].join(self.derived)
        (folder / "cytometry").mkdir(exist_ok=True)
        out.to_csv(folder / "cytometry" / "derived.csv", index=False)
        return "cytometry/derived.csv"

    def load_derived(self, path):
        keys = [k for k in KEYS if k in self.table.columns]
        saved = pd.read_csv(path)
        merged = self.table[keys].merge(saved, on=keys, how="left")
        self.derived = merged.drop(columns=keys)
        for info in self.analyses.values():
            col = info.get("columns", {}).get("cluster")
            if col in self.derived.columns:
                self.derived[col] = self.derived[col].fillna(0).astype(np.int64)

    def load_dict(self, data, tables):
        from vtea_core.classes import PhenotypeTree
        from vtea_core.gates.io import gate_set_from_dict

        notes = []
        self.gates = gate_set_from_dict(data["gates"])
        self.image_gates = dict(data.get("image_gates") or {})
        self.order = [tuple(item) for item in data.get("order") or []]
        self.scope = {tuple(item): tuple(scope) for item, scope in data.get("scope") or []}
        self.tree = PhenotypeTree.from_dict(data["phenotypes"])
        self.source, self.layer_name = data.get("source"), data.get("layer")
        self.analyses, self.derived = dict(data.get("analyses") or {}), None
        if self.source in tables:
            self.table = tables[self.source].reset_index(drop=True).copy()
        else:
            self.table = None
            if self.source:
                notes.append(f"cytometry: the measurement table {self.source!r} was not found")
        self.changed()
        return notes


def _spatial_ndim(layer):
    from ._util import spatial_ndim

    return spatial_ndim(layer)


_STATS = {"mean", "median", "min", "max", "std", "sum"}


def _is_key(column):
    """An id column: time point, position (or other leading axis) or label."""
    return column in KEYS or str(column).startswith("axis")


def _is_voxels(column):
    return column == "voxels" or str(column).endswith(": voxels")


def _centroid_columns(frame, prefix=""):
    """{axis: column} of centroids, e.g. {"Y": "cells: centroid Y (µm)", ...}. In a table
    joining several layers ("All objects"), `prefix` picks that layer's; otherwise the
    first layer that has them."""
    names = [str(c) for c in frame.columns]
    for want in ([prefix] if prefix else []) + [""]:
        for unit in ("µm", "px"):
            found = {}
            for axis in "ZYX":
                ends = f"centroid {axis} ({unit})"
                match = [c for c in names if c.startswith(want) and (c == ends or c.endswith(": " + ends))]
                if match:
                    found[axis] = match[0]
            if "Y" in found and "X" in found:
                return found
    return {}


# ---- workspace ---------------------------------------------------------------------------------


def _save(viewer, folder):
    state = _states.get(viewer)
    if state is None or (state.table is None and not state.order and not len(state.tree)):
        return None
    data = state.to_dict()
    if state.table is not None and state.derived is not None:
        data["derived"] = state.save_derived(folder)
    return data


def _load(viewer, folder, data):
    from . import _measure

    state = state_of(viewer)
    notes = state.load_dict(data, _measure.tables_of(viewer))
    if data.get("derived") and state.table is not None:
        path = folder / data["derived"]
        if path.exists():
            state.load_derived(path)
            state.changed()
        else:
            notes.append(f"cytometry: clustering results not found ({data['derived']})")
    return notes


def _by_size(labels):
    """Cluster ids renumbered 1..k from the largest cluster down."""
    ids, counts = np.unique(labels, return_counts=True)
    order = ids[np.argsort(-counts, kind="stable")]
    lut = {old: new for new, old in enumerate(order, start=1)}
    return np.array([lut[v] for v in labels], dtype=np.int64)


EMBEDDINGS = {"UMAP": "umap", "t-SNE": "tsne", "PCA": "pca", "None": None}
CLUSTERINGS = {"Leiden": "leiden", "Louvain": "louvain", "k-means": "kmeans",
               "Gaussian mixture": "gmm", "k-means, best k (BIC)": "autok", "None": None}
SCALINGS = ("zscore", "robust", "minmax", "none")


def available(method):
    """None if a map / clustering method can run here, else what to install."""
    from importlib.util import find_spec

    if method == "umap" and find_spec("umap") is None:
        return "pip install umap-learn"
    if method in ("leiden", "louvain") and find_spec("igraph") is None and find_spec("networkx") is None:
        return "pip install python-igraph leidenalg"
    return None


def compute_analysis(matrix, embedding=None, clustering=None, scaling="zscore", seed=0,
                     n_neighbors=15, resolution=1.0, k=8, k_max=12, perplexity=30.0):
    """Run a map and/or a clustering on a matrix of objects x features (vtea-core).
    Returns (cluster ids or None, embedding (n, 2) or None)."""
    from vtea_core import clustering as cl
    from vtea_core.reduction import reduction as rd

    n = len(matrix)
    if n < 3:
        raise ValueError("too few objects to analyse")
    labels = emb = None
    nn = int(min(n_neighbors, n - 1))
    if clustering == "leiden":
        labels = cl.leiden(matrix, n_neighbors=nn, resolution=resolution, random_state=seed, normalize=scaling)
    elif clustering == "louvain":
        labels = cl.louvain(matrix, n_neighbors=nn, resolution=resolution, random_state=seed, normalize=scaling)
    elif clustering == "kmeans":
        labels = cl.kmeans(matrix, int(min(k, n)), random_state=seed, normalize=scaling)
    elif clustering == "gmm":
        labels = cl.gaussian_mixture(matrix, int(min(k, n)), random_state=seed, normalize=scaling)
    elif clustering == "autok":
        labels, _ = cl.auto_k_kmeans(matrix, k_min=2, k_max=int(max(2, min(k_max, n - 1))),
                                     random_state=seed, normalize=scaling)
    if embedding == "umap":
        emb = rd.umap(matrix, 2, n_neighbors=nn, random_state=seed, normalize=scaling)
    elif embedding == "tsne":
        emb = rd.tsne(matrix, 2, perplexity=float(min(perplexity, max(2.0, (n - 1) / 3))),
                      random_state=seed, normalize=scaling)
    elif embedding == "pca":
        emb = rd.pca(matrix, 2 if matrix.shape[1] >= 2 else 1, normalize=scaling)
    return labels, emb


def _register():
    from . import _workspace

    _workspace.register_extension("cytometry", _save, _load)


_register()
