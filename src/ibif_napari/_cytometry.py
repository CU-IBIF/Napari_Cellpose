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
- **image gates**: regions painted on the image (or polygons, from scripts and older
  workspaces); an object is in one when its centroid is on a painted pixel. A region
  painted in 2D applies to every Z plane; one painted with the 3D ball brush keeps its
  depth;
- each gate becomes a column of the table (``gate: <name>``), so a phenotype rule can
  use it: ```gate: CD3+` AND NOT `gate: CD4+```;
- saving all of it with the workspace.
"""
from __future__ import annotations

import re
import weakref

import numpy as np
import pandas as pd

# The vtea-core this version needs (as in pyproject.toml; a test keeps them equal). pip
# does not replace an installed vtea-core with the same version number, so an upgrade of
# IBIF napari can leave an older one: `vtea_core_problem` notices, and gives the command.
VTEA_CORE = "vtea-core[graph,umap,opentsne] @ git+https://github.com/winfrees/vtea-python@772e5cf410c33f4277cd191b08c79477253b6fd3#subdirectory=packages/vtea-core"
VTEA_CORE_FEATURES = {"vtea_core.clustering": ["nearest_neighbors"],
                      "vtea_core.classes": ["column_references", "rename_columns"],
                      "vtea_core.spatial": ["surface_distance", "neighbor_enrichment"],
                      "vtea_core.classification": ["ANNClassifier", "load_model", "predictions"]}

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
        self.last_rewrite = {}  # names in the last rule added / edited that were matched to columns
        # The workbench plot's set-up (axes, scales, population, colour, point style),
        # saved with the workspace; view_version changes when a workspace restores it.
        self.view = {}
        self.view_version = 0
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
        self.dropped_analyses = []
        if not same and self.analyses:  # clustering results belong to the objects they were run on
            self.dropped_analyses = list(self.analyses)
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
        on): size and intensities, and the spatial / neighbourhood features computed
        from them; not where the objects are, cluster ids or map coordinates."""
        derived = set(self.derived.columns) if self.derived is not None else set()
        spatial = [c for info in self.analyses.values() for c in info.get("columns", {}).get("features", [])
                   if c in derived]
        return [c for c in self.features() if c not in derived and " centroid " not in f" {c}"] + spatial

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
        """A table column, or a transformed one computed on the fly. KeyError if there
        is no such column (e.g. the map of a clustering result that was deleted)."""
        if name in self.table.columns:
            return self.table[name].to_numpy(float)
        if self.derived is not None and name in self.derived.columns:
            return self.derived[name].to_numpy(float)
        column, transform, cofactor = split_axis(name)
        if transform == "linear" or column == name:
            raise KeyError(f"there is no column {name!r}")
        values = self.column(column)
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
        self._missing = {}  # (kind, key) -> columns a gate is drawn on that are gone
        for gate in self.gates:  # the axis columns gates are drawn on
            for axis in (gate.x_axis, gate.y_axis):
                if axis not in frame.columns:
                    try:
                        frame[axis] = self.column(axis)
                    except KeyError:
                        # e.g. a gate on the map of a clustering result that was deleted or
                        # cleared: it selects nothing until the column is back.
                        frame[axis] = np.full(len(frame), np.nan)
                        self._missing.setdefault(("gate", gate.id), []).append(axis)
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

    def missing_gate_columns(self):
        """{(kind, key): [column, ...]} for gates drawn on columns that are not in the
        table any more (they select no objects until the columns are back)."""
        if self.frame() is None:
            return {}
        return {item: cols for item, cols in getattr(self, "_missing", {}).items()
                if item[0] == "gate" and item[1] in self.gates}

    def gates_using(self, columns):
        """Names of the plot gates drawn on any of `columns`."""
        columns = set(columns)
        return [g.name for g in self.gates
                if {split_axis(g.x_axis)[0], split_axis(g.y_axis)[0], g.x_axis, g.y_axis} & columns]

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

    def add_image_gate(self, name, polygons=None, population=("all", None), color=None, *, mask=None,
                       scale=None, translate=None):
        """An image gate: objects whose centroid (world units) is inside any of
        `polygons` (Y, X), or on a painted pixel of `mask` (a Y X or Z Y X boolean
        array with the `scale` and `translate` of the layer it was painted on; a Y X
        mask applies to every Z plane)."""
        if name in self.gate_names():
            raise ValueError(f"there is already a gate called {name!r}")
        if mask is not None:
            gate = _painted_gate(mask, scale, translate)
            if gate is None:
                raise ValueError("paint over the objects to gate first")
        else:
            polygons = [np.asarray(p, float)[:, -2:].tolist() for p in polygons or [] if len(p) >= 3]
            if not polygons:
                raise ValueError("draw at least one polygon or rectangle")
            gate = {"polygons": polygons}
        gate["color"] = color or self._next_color()
        self.image_gates[name] = gate
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
        for node in self.tree:  # keep phenotype rules pointing at it
            node.rule = _rename_in_rule(node.rule, prefix + old, prefix + new)
        self.changed()

    def set_gate_color(self, kind, key, color):
        """Change the colour of a plot gate or image gate (its outline on the plot, its
        objects on the image, the outline of image-gated objects on the plot)."""
        if kind == "gate":
            self.gates.get(key).color = color
        else:
            self.image_gates[key]["color"] = color
        self.changed()

    def image_gated(self):
        """[(name, mask over the table rows, colour)] of every image gate."""
        if self.frame() is None:
            return []
        return [(name, self.mask("image", name), gate["color"]) for name, gate in self.image_gates.items()]

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
        if "mask" in gate:
            zs = frame[cols["Z"]].to_numpy(float) if "Z" in cols else None
            return _in_painted(gate, zs, ys, xs)
        mask = np.zeros(len(frame), bool)
        for poly in gate["polygons"]:
            mask |= polygon_gate(ys, xs, np.asarray(poly, float))
        return mask

    # ---- phenotypes ------------------------------------------------------------------------

    def add_phenotype(self, name, rule="", parent=None, color=None):
        """A phenotype from a rule over the object table (see `resolve_rule` for the
        names it understands); the rule is stored with the table's exact names."""
        from vtea_core.classes import Phenotype

        if name in self.gate_names():
            raise ValueError(f"{name!r} is already a gate name")
        rule, self.last_rewrite = self.resolve_rule(rule)
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
            rule, self.last_rewrite = self.resolve_rule(rule)
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

    # ---- reading rules -------------------------------------------------------------------

    def rule_columns(self):
        """{group: [column, ...]} that a rule can use: gates, image gates, clusters
        and maps, and measurements."""
        frame = self.frame()
        if frame is None:
            return {}
        cols = [str(c) for c in frame.columns]
        derived = set(self.derived.columns) if self.derived is not None else set()
        return {
            "Gates": [c for c in cols if c.startswith(GATE_PREFIX)],
            "Image gates": [c for c in cols if c.startswith(IMAGE_GATE_PREFIX)],
            "Clusters, maps and spatial": [c for c in cols if c in derived],
            "Measurements": [c for c in self.measured_features()],
        }

    def match_column(self, name):
        """The table column `name` stands for: (column, None), or (None, why not).

        Exact names first; then, ignoring case: a gate or image gate by its own name
        (`CD3+` for `gate: CD3+`), the short label of the heat map (`Cell Alexa647
        mean`), and finally the columns whose name contains every word of `name`
        (`Alexa647 mean`). Ambiguous names are not guessed: the candidates are given."""
        frame = self.frame()
        cols = [str(c) for c in frame.columns]
        if name in cols:
            return name, None
        low = name.strip().lower()
        by_lower = {c.lower(): c for c in cols}
        for candidate in (low, (GATE_PREFIX + low), (IMAGE_GATE_PREFIX + low)):
            if candidate in by_lower:
                return by_lower[candidate], None
        from ._cyto_labels import short_names

        named = [c for group in self.rule_columns().values() for c in group]
        shorts = {}
        for full, short in short_names(named).items():
            shorts.setdefault(short.lower(), []).append(full)
        if len(shorts.get(low, [])) == 1:
            return shorts[low][0], None
        words = [w for w in re.split(r"[^0-9a-z]+", low) if w and w not in _GENERIC_WORDS]
        hits = [c for c in named if words and all(w in _squash(c) for w in words)]
        if len(hits) == 1:
            return hits[0], None
        if hits:
            hits.sort(key=lambda c: (not c.endswith(" mean"), len(c)))
            listed = ", ".join(quote(c) for c in hits[:6]) + (" …" if len(hits) > 6 else "")
            return None, f"{name!r} could be any of {listed}: write the one you mean"
        import difflib

        close = difflib.get_close_matches(name, cols, n=3, cutoff=0.5)
        hint = f"; did you mean {', '.join(quote(c) for c in close)}?" if close else \
            " (use Insert a column… to pick one)"
        return None, f"there is no column {name!r}{hint}"

    def resolve_rule(self, rule):
        """(rule with every column written as the table's exact name, {as written:
        column} for the names that were matched). Raises ValueError when a name
        matches no column, or several."""
        rule = rule or ""
        if not rule.strip() or self.table is None:
            return rule, {}
        try:
            from vtea_core.classes import ExpressionError, column_references
        except ImportError:  # an older vtea-core: rules are used as written
            return rule, {}
        try:
            refs = column_references(rule)
        except ExpressionError as e:
            raise ValueError(f"the rule cannot be read: {e}") from e
        mapping, problems, quoted = {}, [], False
        for written, start, _ in refs:
            quoted |= rule[start] in "'\""  # stored with backticks, as every version reads
            if written in mapping:
                continue
            column, why = self.match_column(written)
            if column is None:
                problems.append(why)
            elif column != written:
                mapping[written] = column
        if problems:
            raise ValueError("; ".join(dict.fromkeys(problems)))
        if not mapping and not quoted:
            return rule, {}
        every = {name: mapping.get(name, name) for name, _, _ in refs}
        return _rename_in_rule(rule, every), mapping

    def rule_count(self, rule, parent=None):
        """(objects the rule selects, objects of the parent phenotype it selects, the
        rule as it will be stored): what the phenotype dialog shows while typing."""
        from vtea_core.classes import evaluate

        resolved, _ = self.resolve_rule(rule)
        frame = self.frame()
        if not resolved.strip():
            mask = np.ones(len(frame), bool)
        else:
            try:
                mask = evaluate(resolved, frame)
            except Exception as e:
                raise ValueError(str(e)) from e
        within = mask
        if parent and parent in self.tree and not self.broken_phenotypes():
            within = mask & self.tree.members(parent, frame, exclusive=True)
        return int(mask.sum()), int(within.sum()), resolved

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

    def population_color(self, kind, key):
        return (self.gates.get(key).color if kind == "gate" else
                self.image_gates[key]["color"] if kind == "image" else
                (self.tree.get(key).color or "#ffff00") if kind == "phenotype" else "#ffff00")

    def show_populations(self, items, overlap_color="#ffffff"):
        """Several gates (or phenotypes) on the image at once, each in its colour, in the
        "<objects> selection" layer; objects in more than one get `overlap_color`.
        Returns (layer, {item: objects}, objects in more than one)."""
        items = [tuple(i) for i in items]
        if len(items) == 1:
            kind, key = items[0]
            return self.show_population(kind, key), {items[0]: int(self.mask(kind, key).sum())}, 0
        masks = [self.mask(*i) for i in items]
        values = np.zeros(len(self.table), np.int64)
        for code, mask in enumerate(masks, start=1):
            values[mask] = code
        several = np.sum(masks, axis=0) > 1
        overlap = len(items) + 1
        values[several] = overlap
        colors = {code: self.population_color(*i) for code, i in enumerate(items, start=1)}
        colors[overlap] = overlap_color
        layer = self.paint(values, f"{self.layer_name} selection", colors)
        return layer, {i: int(m.sum()) for i, m in zip(items, masks)}, int(several.sum())

    def show_population(self, kind, key, color=None):
        mask = self.mask(kind, key)
        if color is None:
            color = self.population_color(kind, key)
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

    def analysis_kind(self, name):
        """"clustering", "spatial" or "neighbourhoods"."""
        return self.analyses[name].get("kind", "clustering")

    def analyses_of(self, kind):
        return [n for n in self.analyses if self.analysis_kind(n) == kind]

    def _check_analysis_name(self, name):
        if name in self.analyses:
            raise ValueError(f"there is already an analysis called {name!r}")
        if name in self.gate_names() or name in self.tree:
            raise ValueError(f"{name!r} is already a gate or phenotype name")

    def add_analysis(self, name, settings, rows, labels=None, embedding=None):
        """Store a clustering / map run on table rows `rows`: cluster ids (1..k, by size;
        0 = not clustered) in "<name> cluster", map coordinates in "<name> <map> 1/2"."""
        self._check_analysis_name(name)
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
        """Delete a clustering / map result. Returns the gates that were drawn on its
        map (they select nothing until a result of the same name is made again)."""
        info = self.analyses.pop(name)
        cols = _analysis_columns(info)
        orphaned = self.gates_using([c for c in cols if c])
        self.derived = self.derived.drop(columns=[c for c in cols if c and c in self.derived.columns])
        if self.derived.shape[1] == 0:
            self.derived = None
        self.changed()
        return orphaned

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

    # ---- spatial features and neighbourhoods -------------------------------------------------

    def class_choices(self):
        """[(label, key)] of what objects can be classed by for spatial analysis: the
        phenotypes (deepest, or at a level) and every clustering's clusters."""
        out = []
        if len(self.tree):
            out.append(("Phenotypes (deepest)", "phenotype"))
            depth = max(self.tree.depth(n.name) for n in self.tree)
            out += [(f"Phenotypes, level {k}", f"phenotype level {k}") for k in range(1, depth + 1)] \
                if depth > 1 else []
        for name in self.analyses_of("clustering"):
            if self.analyses[name]["columns"].get("cluster"):
                out.append((f"{name} clusters", self.analyses[name]["columns"]["cluster"]))
        for name in self.analyses_of("classification"):
            out.append((f"{name} classes", self.analyses[name]["columns"]["class"]))
        return out

    def classes(self, key):
        """Each object's class (None = none) for a `class_choices` key: phenotype names,
        or "cluster <i>"."""
        if key.startswith("phenotype"):
            values = self.assignment()[key].to_numpy(dtype=object)
            return np.array([None if v is None or v != v or v == "unlabelled" else str(v) for v in values],
                            dtype=object)
        ids = self.column(key).astype(np.int64)
        names = self.class_names(key)
        if names:
            return np.array([names.get(int(i)) for i in ids], dtype=object)
        return np.array([f"cluster {i}" if i > 0 else None for i in ids], dtype=object)

    def class_names(self, column):
        """{code: class name} when `column` is a classification's class column, else {}."""
        for info in self.analyses.values():
            if info.get("kind") == "classification" and info["columns"].get("class") == column:
                return {k + 1: name for k, name in enumerate(info["classes"])}
        return {}

    # ---- supervised classification -----------------------------------------------------------

    def add_classification(self, name, settings, rows, result, classes, colors=None):
        """Store a model's predictions for table rows `rows` (a vtea-core `predictions`
        frame): "<name> class" (1..k in the model's class order, 0 = not classified),
        "<name> confidence" and "<name> p <class>"."""
        self._check_analysis_name(name)
        n = len(self.table)
        rows = np.asarray(rows, dtype=np.int64)
        code = {c: k + 1 for k, c in enumerate(classes)}
        values = np.zeros(n, np.int64)
        values[rows] = [code.get(c, 0) for c in result["class"]]
        confidence = np.full(n, np.nan)
        confidence[rows] = result["confidence"].to_numpy(float)
        class_col, conf_col = f"{name} class", f"{name} confidence"
        columns = {class_col: values, conf_col: confidence}
        probabilities = []
        for c in classes:
            col = f"{name} p {c}"
            p = np.full(n, np.nan)
            p[rows] = result[f"p {c}"].to_numpy(float)
            columns[col] = p
            probabilities.append(col)
        colors = {c: (colors or {}).get(c) or PALETTE[k % len(PALETTE)] for k, c in enumerate(classes)}
        self._store(name, {**settings, "kind": "classification", "classes": list(classes), "colors": colors},
                    {"class": class_col, "confidence": conf_col, "features": [conf_col],
                     "probabilities": probabilities}, columns, len(rows))
        return self.analyses[name]

    def classification_counts(self, name):
        """[(class, objects, mean confidence)] of a classification."""
        info = self.analyses[name]
        codes = self.column(info["columns"]["class"]).astype(np.int64)
        conf = self.column(info["columns"]["confidence"])
        out = []
        for k, c in enumerate(info["classes"], start=1):
            hit = codes == k
            out.append((c, int(hit.sum()), float(np.nanmean(conf[hit])) if hit.any() else float("nan")))
        return out

    def phenotypes_from_classification(self, name, names=None, parent=None, min_confidence=None):
        """Phenotypes from predicted classes: `names` maps class -> phenotype name (default:
        the class names; the same name merges classes; empty skips a class). With
        `min_confidence`, only objects predicted at least that sure."""
        info = self.analyses[name]
        column = quote(info["columns"]["class"])
        names = names or {c: c for c in info["classes"]}
        groups = {}
        for k, c in enumerate(info["classes"], start=1):
            pheno = (names.get(c) or "").strip()
            if pheno:
                groups.setdefault(pheno, []).append(k)
        clash = [p for p in groups if p in self.tree or p in self.gate_names()]
        if clash:
            raise ValueError("these names are already used: " + ", ".join(clash))
        sure = f" AND {quote(info['columns']['confidence'])} >= {min_confidence:g}" if min_confidence else ""
        made = []
        for pheno, ids in groups.items():
            rule = f"{column} == {ids[0]}" if len(ids) == 1 else f"{column} in [{', '.join(map(str, ids))}]"
            color = info.get("colors", {}).get(next(c for k, c in enumerate(info["classes"], 1) if k in ids))
            made.append(self.add_phenotype(pheno, rule + sure, parent, color))
        return made

    def show_classification(self, name):
        info = self.analyses[name]
        codes = self.column(info["columns"]["class"]).astype(np.int64)
        colors = {k: info["colors"][c] for k, c in enumerate(info["classes"], start=1)}
        return self.paint(codes, f"{self.layer_name} {name}", colors)

    def unit(self):
        cols = self.centroid_columns()
        return "µm" if any("(µm)" in c for c in cols.values()) else "px"

    def positions(self):
        """Object centroids (world units; Z, Y, X as measured), each time point /
        position moved far from the others so that no distance crosses between them."""
        cols = self.centroid_columns()
        if "Y" not in cols:
            raise ValueError("the table has no centroids (measure the objects again)")
        points = np.column_stack([self.table[cols[a]].to_numpy(float) for a in "ZYX" if a in cols])
        keys = [k for k in ("T", "P") if k in self.table.columns]
        if keys:
            block = self.table.groupby(keys, sort=False).ngroup().to_numpy()
            points = points.copy()
            points[:, 0] += block * SEPARATE
        return points

    def label_blocks(self):
        """[(table rows, label image, voxel size or None)] for each time point /
        position of the objects layer (what surface distances and contacts read)."""
        layer = self.labels_layer()
        if layer is None:
            raise ValueError(f"the objects layer {self.layer_name!r} is not open")
        data = layer.data
        n = min(_spatial_ndim(layer), 3)
        lead = data.ndim - n
        table = self.table
        keys = [k for k in table.columns if _is_key(k) and k != "label"][:lead]
        scale = tuple(float(v) for v in layer.scale[-n:])
        out = []
        for pos in (list(np.ndindex(*data.shape[:lead])) if lead else [()]):
            rows = np.ones(len(table), bool)
            for k, p in zip(keys, pos):
                rows &= table[k].to_numpy() == p
            if rows.any():
                out.append((np.flatnonzero(rows), np.asarray(data[pos]), scale if self.unit() == "µm" else None))
        return out

    def spatial_job(self, settings):
        """A function (safe to run in a background thread) that computes the spatial
        features `settings` asks for, returning a DataFrame with one row per object.

        settings: "classes" (a `class_choices` key), "targets" (class names, default all),
        "measures" (any of "nearest", "within", "surface", "contacts"), "radius"."""
        from vtea_core.data import Spacing
        from vtea_core.spatial import contacts, nearest_distance, neighbors_within, surface_distance

        classes = self.classes(settings["classes"])
        targets = settings.get("targets") or None
        measures = set(settings["measures"])
        if not measures:
            raise ValueError("choose at least one feature to measure")
        points = self.positions() if measures & {"nearest", "within"} else None
        blocks = self.label_blocks() if measures & {"surface", "contacts"} else []
        labels = self.table["label"].to_numpy(np.int64)
        n = len(self.table)
        radius = float(settings.get("radius") or 0)

        def job():
            parts = []
            if "nearest" in measures:
                parts.append(nearest_distance(points, classes, targets=targets))
            if "within" in measures:
                parts.append(neighbors_within(points, classes, radius, targets=targets))
            for measure, fn in (("surface", surface_distance), ("contacts", contacts)):
                if measure not in measures:
                    continue
                pieces = []
                for rows, image, scale in blocks:
                    kwargs = {"targets": targets or sorted({c for c in classes if c is not None})}
                    if measure == "surface":
                        kwargs["spacing"] = Spacing(scale) if scale else None
                    part = fn(image, labels[rows], classes[rows], **kwargs)
                    part.index = rows
                    pieces.append(part)
                parts.append(pd.concat(pieces).reindex(range(n)) if pieces else pd.DataFrame(index=range(n)))
            return pd.concat([p.reset_index(drop=True) for p in parts], axis=1)

        return job

    def add_spatial(self, name, settings, frame):
        """Store spatial features (a `spatial_job` result) as "<name> <feature>" columns;
        distances get their unit."""
        self._check_analysis_name(name)
        unit = self.unit()
        columns = {}
        for col in frame.columns:
            label = f"{name} {col}" + (f" ({unit})" if "distance" in col else "")
            columns[label] = frame[col].to_numpy(float)
        self._store(name, {**settings, "kind": "spatial", "unit": unit}, {"features": list(columns)}, columns,
                    int(len(frame)))
        return self.analyses[name]

    def neighbourhood_job(self, settings):
        """A function (safe in a background thread) that builds each object's
        neighbourhood (itself and the objects within `radius`, or its `k` nearest),
        measures its class composition, sorts the neighbourhoods into `n_types` types
        (k-means on the composition; vtea-core `neighborhoods`) and computes the
        enrichment statistics (vtea-core `spatial.enrichment`). Returns a dict.

        Neighbourhoods overlap: one is drawn around every object, so an object is a
        member of its own and of every neighbour's. Each object gets both views: the
        type of its own neighbourhood ("what surrounds it"), and, over every
        neighbourhood it is a member of, how many there are and their most common
        type ("what it is part of")."""
        from vtea_core.neighborhoods import (
            NEIGHBORHOOD_TYPE,
            build_neighborhoods,
            classify_neighborhoods,
            composition_columns,
            neighborhood_features,
            reflect_neighborhoods,
        )
        from vtea_core.spatial import neighbor_enrichment, type_enrichment

        classes = self.classes(settings["classes"])
        points = self.positions()
        method = settings.get("method", "radius")
        radius, k = float(settings.get("radius") or 0), int(settings.get("k") or 10)
        n_types, seed = int(settings.get("n_types") or 6), int(settings.get("seed") or 0)
        perms = int(settings.get("permutations", 200))

        def job():
            named = np.array([c if c is not None else NO_CLASS for c in classes], dtype=object)
            data = pd.DataFrame({"object_id": np.arange(1, len(points) + 1),
                                 **{f"centroid-{i}": points[:, i] for i in range(points.shape[1])},
                                 "class": named})
            hoods = build_neighborhoods(data, method="nearest" if method == "nearest" else "radius",
                                        radius=radius if radius > 0 else 1.0, k=k)
            table = neighborhood_features(hoods, data, class_column="class")
            fractions = composition_columns(table)
            typed = classify_neighborhoods(table, method="kmeans", n_clusters=n_types, features=",".join(fractions),
                                           random_state=seed)
            order = typed["neighborhood_id"].to_numpy(np.int64) - 1  # centred: one per object
            types = np.zeros(len(points), np.int64)
            types[order] = _by_size(typed[NEIGHBORHOOD_TYPE].to_numpy())
            # The membership view: every neighbourhood an object falls in, its own included.
            typed[NEIGHBORHOOD_TYPE] = types[order]
            member = reflect_neighborhoods(hoods, typed[["neighborhood_id", NEIGHBORHOOD_TYPE]], data,
                                           relation="member")
            memberships = member["n_neighborhoods"].to_numpy(np.int64)
            member_types = np.maximum(member[NEIGHBORHOOD_TYPE].to_numpy(np.int64), 0)
            composition = {}
            for col in fractions:
                value = col[len("Class_"):-len("_ClassFraction")]
                values = np.full(len(points), np.nan)
                values[order] = typed[col].to_numpy(float) / 100.0
                composition[value] = values
            profile = pd.DataFrame(composition).groupby(types).mean()
            sizes = pd.Series(types).value_counts().sort_index()
            graph = {"k": k} if method == "nearest" else {"radius": radius}
            pairs = neighbor_enrichment(points, classes, n_permutations=perms, random_state=seed, **graph)
            by_type = type_enrichment(classes, types)
            return {"types": types, "composition": composition, "profile": profile, "sizes": sizes,
                    "pairs": pairs, "by_type": by_type, "memberships": memberships,
                    "member_types": member_types}

        return job

    def add_neighbourhoods(self, name, settings, result):
        """Store a `neighbourhood_job` result: the type of each object's own
        neighbourhood in "<name> neighbourhood" (1 = the most common type), the
        composition of its own neighbourhood in "<name> fraction <class>", how many
        neighbourhoods it is a member of in "<name> memberships" and their most common
        type in "<name> member neighbourhood", and the tables the results show."""
        self._check_analysis_name(name)
        type_col = f"{name} neighbourhood"
        columns = {type_col: result["types"]}
        features = []
        member_col = None
        if "memberships" in result:  # results made before memberships were kept have none
            member_col = f"{name} member neighbourhood"
            columns[member_col] = result["member_types"]
            columns[f"{name} memberships"] = result["memberships"]
            features.append(f"{name} memberships")
        for value, values in result["composition"].items():
            col = f"{name} fraction {value}"
            columns[col] = values
            features.append(col)
        tables = {"profile": _split(result["profile"]), "sizes": _split(result["sizes"].to_frame("objects")),
                  "pairs": _split(result["pairs"]), "by_type": _split(result["by_type"])}
        self._store(name, {**settings, "kind": "neighbourhoods", "tables": tables},
                    {"type": type_col, "member_type": member_col, "features": features}, columns,
                    len(result["types"]))
        return self.analyses[name]

    def analysis_table(self, name, which):
        """A table saved with a neighbourhood result ("profile", "sizes", "pairs",
        "by_type") as a DataFrame."""
        data = self.analyses[name]["tables"][which]
        return pd.DataFrame(data["data"], index=data["index"], columns=data["columns"])

    def _store(self, name, info, columns_info, columns, n_objects):
        derived = self.derived if self.derived is not None else pd.DataFrame(index=self.table.index)
        for col, values in columns.items():
            derived[col] = np.asarray(values)
        self.derived = derived
        self.analyses[name] = {**info, "columns": columns_info, "n_objects": int(n_objects)}
        self.changed()

    def show_neighbourhoods(self, name):
        types = self.derived[self.analyses[name]["columns"]["type"]].to_numpy(np.int64)
        colors = {int(c): PALETTE[(int(c) - 1) % len(PALETTE)] for c in np.unique(types) if c > 0}
        return self.paint(types, f"{self.layer_name} {name}", colors)

    def show_clusters(self, name):
        labels = self.clusters(name)
        colors = {int(c): PALETTE[(int(c) - 1) % len(PALETTE)] for c in np.unique(labels) if c > 0}
        return self.paint(labels, f"{self.layer_name} {name}", colors)

    # ---- saving ----------------------------------------------------------------------------

    def to_dict(self):
        from vtea_core.gates.io import gate_set_to_dict

        return {"version": 1, "source": self.source, "layer": self.layer_name,
                "gates": gate_set_to_dict(self.gates),
                "image_gates": {name: _image_gate_to_json(g) for name, g in self.image_gates.items()},
                "order": [list(item) for item in self.order],
                "scope": [[list(item), list(scope)] for item, scope in self.scope.items()],
                "phenotypes": self.tree.to_dict(), "analyses": self.analyses, "view": dict(self.view)}

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
            for key in ("cluster", "type", "member_type", "class"):
                col = info.get("columns", {}).get(key)
                if col in self.derived.columns:
                    self.derived[col] = self.derived[col].fillna(0).astype(np.int64)

    def load_dict(self, data, tables, derived_path=None):
        """Restore a saved analysis. `derived_path`: the saved clustering / map columns,
        read before anything is told about the change (gates drawn on a map need them)."""
        from vtea_core.classes import PhenotypeTree
        from vtea_core.gates.io import gate_set_from_dict

        notes = []
        self.gates = gate_set_from_dict(data["gates"])
        self.image_gates = {name: _image_gate_from_json(g) for name, g in (data.get("image_gates") or {}).items()}
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
        self.view = dict(data.get("view") or {})
        self.view_version += 1
        if derived_path is not None and self.table is not None:
            if derived_path.exists():
                self.load_derived(derived_path)
            else:
                notes.append(f"cytometry: clustering results not found ({derived_path.name})")
        missing = self.missing_gate_columns() if self.table is not None else {}
        if missing:
            names = sorted({c for cols in missing.values() for c in cols})
            notes.append("cytometry: gates drawn on columns that could not be restored select nothing: "
                         + ", ".join(names))
        self.changed()
        return notes


def model_library():
    """The folder of saved classification models: $IBIF_NAPARI_MODELS, else
    ~/.ibif-napari/models (created when first used)."""
    import os
    from pathlib import Path

    path = Path(os.environ.get("IBIF_NAPARI_MODELS") or Path.home() / ".ibif-napari" / "models")
    path.mkdir(parents=True, exist_ok=True)
    return path


SEPARATE = 1e7  # how far apart time points / positions are put for distances (world units)
NO_CLASS = "(none)"


def _split(frame):
    """A DataFrame as JSON-safe lists (saved with the workspace)."""
    data = frame.to_dict("split")
    clean = lambda v: v.item() if hasattr(v, "item") else (None if isinstance(v, float) and v != v else v)  # noqa: E731
    return {"index": [clean(v) for v in data["index"]], "columns": [clean(v) for v in data["columns"]],
            "data": [[clean(v) for v in row] for row in data["data"]]}


def _analysis_columns(info):
    """Every derived column an analysis made."""
    cols = info.get("columns", {})
    return [c for c in [cols.get("cluster"), cols.get("type"), cols.get("member_type"), cols.get("class"),
                        *cols.get("map", []),
                        *cols.get("features", []), *cols.get("probabilities", [])] if c]


_GENERIC_WORDS = {"intensity", "int", "channel", "ch", "value", "of", "the", "in"}


def _squash(text):
    """Lower case, letters and digits only: 'cells: Alexa Fluor-647 mean' -> 'cellsalexafluor647mean'."""
    return re.sub(r"[^0-9a-z]+", "", text.lower())


def _rename_in_rule(rule, old_or_mapping, new=None):
    """The rule with column names replaced (vtea-core's parser finds them, so a value
    that looks like a name is not touched); a plain text replace on an older vtea-core."""
    mapping = old_or_mapping if isinstance(old_or_mapping, dict) else {old_or_mapping: new}
    try:
        from vtea_core.classes import rename_columns
    except ImportError:
        for old, new_name in mapping.items():
            rule = rule.replace(quote(old), quote(new_name))
        return rule
    try:
        return rename_columns(rule, mapping)
    except Exception:
        return rule


def _painted_gate(mask, scale, translate):
    """An image gate from a painted Y X or Z Y X mask, cropped to what was painted
    (None if nothing was): {"mask", "scale", "translate"} in world units."""
    mask = np.asarray(mask) != 0
    if mask.ndim not in (2, 3):
        raise ValueError("a painted image gate is a Y X or Z Y X mask")
    n = mask.ndim
    scale = np.asarray(scale if scale is not None else np.ones(n), float)[-n:]
    translate = np.asarray(translate if translate is not None else np.zeros(n), float)[-n:]
    if not mask.any():
        return None
    lo = [int(np.flatnonzero(mask.any(axis=tuple(a for a in range(n) if a != k))).min()) for k in range(n)]
    hi = [int(np.flatnonzero(mask.any(axis=tuple(a for a in range(n) if a != k))).max()) + 1 for k in range(n)]
    crop = mask[tuple(slice(a, b) for a, b in zip(lo, hi))].copy()
    return {"mask": crop, "scale": scale.tolist(), "translate": (translate + np.asarray(lo) * scale).tolist()}


def _in_painted(gate, zs, ys, xs):
    """Which centroids (world units; zs may be None) fall on a painted pixel. A Z Y X
    mask without Z centroids counts every plane."""
    mask = gate["mask"]
    scale, translate = np.asarray(gate["scale"], float), np.asarray(gate["translate"], float)
    if mask.ndim == 3 and zs is None:
        mask, scale, translate = mask.any(axis=0), scale[1:], translate[1:]
    coords = [zs, ys, xs][-mask.ndim:]
    inside = np.ones(len(ys), bool)
    index = []
    for k, world in enumerate(coords):
        with np.errstate(invalid="ignore"):
            i = np.round((np.asarray(world, float) - translate[k]) / scale[k])
        ok = np.isfinite(i) & (i >= 0) & (i < mask.shape[k])
        inside &= ok
        index.append(np.where(ok, i, 0).astype(np.int64))
    return inside & mask[tuple(index)]


def _image_gate_to_json(gate):
    """A painted mask goes into the workspace file as zlib-compressed bits."""
    if "mask" not in gate:
        return dict(gate)
    import base64
    import zlib

    mask = np.asarray(gate["mask"], bool)
    packed = base64.b64encode(zlib.compress(np.packbits(mask).tobytes(), 9)).decode("ascii")
    return {**{k: v for k, v in gate.items() if k != "mask"}, "mask": {"shape": list(mask.shape), "bits": packed}}


def _image_gate_from_json(gate):
    gate = dict(gate)
    if isinstance(gate.get("mask"), dict):
        import base64
        import zlib

        shape = tuple(gate["mask"]["shape"])
        bits = np.frombuffer(zlib.decompress(base64.b64decode(gate["mask"]["bits"])), np.uint8)
        gate["mask"] = np.unpackbits(bits, count=int(np.prod(shape))).astype(bool).reshape(shape)
    return gate


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
    derived = folder / data["derived"] if data.get("derived") else None
    return state.load_dict(data, _measure.tables_of(viewer), derived_path=derived)


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
                     n_neighbors=15, resolution=1.0, k=8, k_max=12, perplexity=30.0, reproducible=True):
    """Run a map and/or a clustering on a matrix of objects x features (vtea-core).
    Returns (cluster ids or None, embedding (n, 2) or None).

    Speed: the features are scaled once; the neighbour search is made once (on every
    core) and shared by Leiden / Louvain and UMAP; Leiden runs on igraph's C
    implementation. `reproducible=False` lets UMAP use every core (umap-learn runs on
    one when seeded), so its layout varies a little from run to run; the clusters do
    not depend on it. t-SNE uses openTSNE (multi-core) when it is installed. The faster
    routes need vtea-core with `nearest_neighbors`; an older one runs as before."""
    from vtea_core import clustering as cl
    from vtea_core.measurements.normalize import normalize_features
    from vtea_core.reduction import reduction as rd

    n = len(matrix)
    if n < 3:
        raise ValueError("too few objects to analyse")
    labels = emb = None
    nn = int(min(n_neighbors, n - 1))
    data = normalize_features(np.asarray(matrix, float), scaling)  # once, for every step
    fast = _fast_vtea()
    knn = None
    if fast and (clustering in ("leiden", "louvain") or embedding == "umap"):
        knn = cl.nearest_neighbors(data, n_neighbors=nn)
    shared = {"knn": knn} if knn is not None else {}
    if clustering == "leiden":
        labels = cl.leiden(data, n_neighbors=nn, resolution=resolution, random_state=seed, normalize="none",
                           **shared)
    elif clustering == "louvain":
        labels = cl.louvain(data, n_neighbors=nn, resolution=resolution, random_state=seed, normalize="none",
                            **shared)
    elif clustering == "kmeans":
        labels = cl.kmeans(data, int(min(k, n)), random_state=seed, normalize="none")
    elif clustering == "gmm":
        labels = cl.gaussian_mixture(data, int(min(k, n)), random_state=seed, normalize="none")
    elif clustering == "autok":
        labels, _ = cl.auto_k_kmeans(data, k_min=2, k_max=int(max(2, min(k_max, n - 1))),
                                     random_state=seed, normalize="none")
    if embedding == "umap":
        emb = rd.umap(data, 2, n_neighbors=nn, random_state=seed if reproducible else None, normalize="none",
                      **shared)
    elif embedding == "tsne":
        options = {"method": "opentsne"} if fast and _has("openTSNE") else {}
        emb = rd.tsne(data, 2, perplexity=float(min(perplexity, max(2.0, (n - 1) / 3))),
                      random_state=seed, normalize="none", **options)
    elif embedding == "pca":
        emb = rd.pca(data, 2 if data.shape[1] >= 2 else 1, normalize="none")
    return labels, emb


def vtea_core_problem():
    """None, or what is missing from the installed vtea-core and how to update it."""
    import importlib

    missing = []
    for module, names in VTEA_CORE_FEATURES.items():
        try:
            mod = importlib.import_module(module)
        except Exception as e:
            return f"vtea-core cannot be imported ({e}). Install it with: pip install \"{VTEA_CORE}\""
        missing += [f"{module}.{n}" for n in names if not hasattr(mod, n)]
    if not missing:
        return None
    return ("The installed vtea-core is older than this version of IBIF napari needs (it has no "
            + ", ".join(missing) + "), so clustering is slower and rules must use exact names. "
            "Update it (pip keeps the old one when only its commit changed):\n"
            f"pip install --force-reinstall --no-deps \"{VTEA_CORE}\"")


def _fast_vtea():
    """vtea-core with the shared neighbour search (and fast Leiden, UMAP / t-SNE options)."""
    from vtea_core import clustering as cl

    return hasattr(cl, "nearest_neighbors")


def _has(module):
    from importlib.util import find_spec

    return find_spec(module) is not None


_warmed = {"umap": False}


def import_analysis_modules(*methods):
    """Import what clustering and maps use, in the calling (main) thread.

    Two threads importing scikit-learn (or anything that imports it) at the same
    time can fail with "cannot import name ... from partially initialized module
    'sklearn.base' (most likely due to a circular import)": Python's import locks
    hand one thread a half-imported module when the package imports itself in a
    circle, as scikit-learn does. So background work (the UMAP warm-up, a clustering
    run) must only start once its modules are imported here. `methods` are the map /
    clustering names of a run ("umap", "tsne", "leiden", ...) whose optional
    libraries are imported too. Cheap after the first call."""
    import importlib

    names = ["sklearn", "sklearn.base", "vtea_core.measurements.normalize", "vtea_core.clustering",
             "vtea_core.reduction.reduction"]
    optional = {"umap": ["umap"], "tsne": ["openTSNE"], "leiden": ["igraph", "leidenalg"],
                "louvain": ["igraph", "networkx"]}
    extra = [name for method in methods for name in optional.get(method, []) if _has(name)]
    for name in names + extra:
        try:
            importlib.import_module(name)
        except Exception:  # an optional library that does not import: the run says so itself
            pass


def warm_up(background=True):
    """Compile UMAP's numba code once per session, on a tiny made-up table, so the
    first real map does not wait for it (about 10 s). The imports are made here, in
    the calling thread (see `import_analysis_modules`); only the compiling, which
    imports nothing more, runs in a background thread unless `background` is False.
    Does nothing if umap-learn is missing, already warmed, or IBIF_NAPARI_NO_WARMUP
    is set."""
    import os
    import threading

    if _warmed["umap"] or os.environ.get("IBIF_NAPARI_NO_WARMUP") or not _has("umap"):
        return None
    _warmed["umap"] = True
    import_analysis_modules("umap")

    def work():
        import warnings

        try:
            from vtea_core.reduction import reduction as rd

            rng = np.random.default_rng(0)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rd.umap(rng.normal(size=(300, 4)), 2, n_neighbors=10, random_state=0)
        except Exception:  # a failed warm-up only means the first run compiles instead
            pass

    if not background:
        work()
        return None
    thread = threading.Thread(target=work, name="ibif-umap-warm-up", daemon=True)
    thread.start()
    return thread


def _register():
    from . import _workspace

    _workspace.register_extension("cytometry", _save, _load)


_register()
