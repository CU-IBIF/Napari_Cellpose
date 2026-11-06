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
        self.source = source
        self.layer_name = layer_name
        self.table = table.reset_index(drop=True).copy()
        self.changed()

    def features(self):
        """Numeric columns that can be plotted (not ids or gate columns)."""
        if self.table is None:
            return []
        skip = set(KEYS) | {"voxels"}
        return [c for c in self.table.columns
                if c not in skip and pd.api.types.is_numeric_dtype(self.table[c])
                and not pd.api.types.is_bool_dtype(self.table[c])]

    def column(self, name):
        """A table column, or a transformed one computed on the fly."""
        if name in self.table.columns:
            return self.table[name].to_numpy(float)
        column, transform, cofactor = split_axis(name)
        values = self.table[column].to_numpy(float)
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

        ys, xs = _centroid_yx(frame)
        if ys is None:
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
        keys = [k for k in ("T", "P") if k in table.columns][:lead]
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

    # ---- saving ----------------------------------------------------------------------------

    def to_dict(self):
        from vtea_core.gates.io import gate_set_to_dict

        return {"version": 1, "source": self.source, "layer": self.layer_name,
                "gates": gate_set_to_dict(self.gates), "image_gates": self.image_gates,
                "order": [list(item) for item in self.order],
                "scope": [[list(item), list(scope)] for item, scope in self.scope.items()],
                "phenotypes": self.tree.to_dict()}

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


def _centroid_yx(frame):
    for unit in ("µm", "px"):
        y, x = f"centroid Y ({unit})", f"centroid X ({unit})"
        if y in frame.columns and x in frame.columns:
            return frame[y].to_numpy(float), frame[x].to_numpy(float)
    return None, None


# ---- workspace ---------------------------------------------------------------------------------


def _save(viewer, folder):
    state = _states.get(viewer)
    if state is None or (state.table is None and not state.order and not len(state.tree)):
        return None
    return state.to_dict()


def _load(viewer, folder, data):
    from . import _measure

    return state_of(viewer).load_dict(data, _measure.tables_of(viewer))


def _register():
    from . import _workspace

    _workspace.register_extension("cytometry", _save, _load)


_register()
