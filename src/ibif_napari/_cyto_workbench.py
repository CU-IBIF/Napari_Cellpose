"""The Cytometry workbench: a separate window for gating and phenotyping.

Plot any two measurements (linear, log10 or asinh), draw rectangle / polygon / lasso
gates on it or polygons on the image, gate within a gate or within a phenotype, and
turn gates into a hierarchy of phenotypes. Everything is linked to the napari viewer:
populations are shown on the image, clicking a point goes to that object, and clicking
an object in the viewer rings it on the plot. It is one window per viewer and shares
the viewer's cytometry state with the Cytometry section.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QAbstractItemView,
    QColorDialog,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QInputDialog,
    QLabel,
    QLineEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _cytometry as cy
from ._cyto_plot import ScatterPlot

IMAGE_GATE_LAYER = "Cytometry image gate"
COLOR_POPULATION, COLOR_PHENOTYPE = "Population", "Phenotype"


class Workbench(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent, Qt.Window)
        self.viewer = viewer
        self.state = cy.state_of(viewer)
        self.setWindowTitle("Cytometry workbench")
        self.resize(1150, 720)
        self._image_layer = None
        self._watched = None
        self._filling = False
        outer = QVBoxLayout(self)
        self.tabs = QTabWidget()
        outer.addWidget(self.tabs)
        gating = QWidget()
        g_layout = QVBoxLayout(gating)
        g_layout.setContentsMargins(4, 4, 4, 4)
        self.tabs.addTab(gating, "Gating")

        # ---- what to plot ----
        row = QHBoxLayout()
        row.addWidget(QLabel("Show"))
        self.population = QComboBox()
        self.population.setToolTip("The objects plotted. A gate drawn now is drawn within them.")
        row.addWidget(self.population, 2)
        row.addWidget(QLabel("X"))
        self.x_col, self.x_tf = QComboBox(), _transform_box()
        row.addWidget(self.x_col, 2)
        row.addWidget(self.x_tf)
        row.addWidget(QLabel("Y"))
        self.y_col, self.y_tf = QComboBox(), _transform_box()
        row.addWidget(self.y_col, 2)
        row.addWidget(self.y_tf)
        row.addWidget(QLabel("Colour"))
        self.color_by = QComboBox()
        row.addWidget(self.color_by, 2)
        g_layout.addLayout(row)

        split = QSplitter(Qt.Horizontal)
        g_layout.addWidget(split, 1)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.plot = ScatterPlot()
        left_layout.addWidget(self.plot, 1)
        tools = QHBoxLayout()
        self.rect_btn = QPushButton("Rectangle gate")
        self.poly_btn = QPushButton("Polygon gate")
        self.poly_btn.setToolTip("Click the corners; click the first corner (or press Enter) to close")
        self.lasso_btn = QPushButton("Lasso gate")
        self.image_btn = QPushButton("Gate on image…")
        self.image_btn.setToolTip("Draw regions on the image (Shapes layer); objects whose centre is "
                                  "inside are gated. Click again when done.")
        for b, mode in ((self.rect_btn, "rectangle"), (self.poly_btn, "polygon"), (self.lasso_btn, "lasso")):
            b.clicked.connect(lambda _=False, m=mode: self.start_gate(m))
            tools.addWidget(b)
        self.image_btn.clicked.connect(self.image_gate_clicked)
        tools.addWidget(self.image_btn)
        left_layout.addLayout(tools)
        self.status = QLabel()
        self.status.setWordWrap(True)
        left_layout.addWidget(self.status)
        split.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        gates_box = QGroupBox("Gates")
        gl = QVBoxLayout(gates_box)
        self.gate_tree = _tree(["Gate", "Objects", "% of parent"])
        self.gate_tree.itemSelectionChanged.connect(self._gate_selected)
        gl.addWidget(self.gate_tree)
        gb = QHBoxLayout()
        self.gate_pheno_btn = _button("Make phenotype…", self.phenotype_from_gate, gb)
        self.gate_rename_btn = _button("Rename…", self.rename_gate, gb)
        self.gate_delete_btn = _button("Delete", self.delete_gate, gb)
        gl.addLayout(gb)
        right_layout.addWidget(gates_box, 1)

        pheno_box = QGroupBox("Phenotypes")
        pl = QVBoxLayout(pheno_box)
        self.pheno_tree = _tree(["Phenotype", "Objects", "% of parent", "Rule"])
        self.pheno_tree.itemDoubleClicked.connect(lambda *_: self.edit_phenotype())
        pl.addWidget(self.pheno_tree)
        pb = QHBoxLayout()
        _button("Add…", self.add_phenotype, pb)
        _button("Edit…", self.edit_phenotype, pb)
        _button("Delete", self.delete_phenotype, pb)
        pl.addLayout(pb)
        pb2 = QHBoxLayout()
        self.level = QSpinBox()
        self.level.setRange(0, 9)
        self.level.setSpecialValueText("deepest")
        self.level.setToolTip("Hierarchy level to colour by (deepest = the most specific phenotype)")
        pb2.addWidget(QLabel("Level"))
        pb2.addWidget(self.level)
        _button("Show phenotypes on image", self.show_phenotypes, pb2)
        _button("Export table…", self.export_table, pb2)
        pl.addLayout(pb2)
        right_layout.addWidget(pheno_box, 1)
        split.addWidget(right)
        split.setSizes([700, 450])

        for w in (self.population, self.x_col, self.y_col, self.color_by):
            w.currentIndexChanged.connect(self.replot)
        for w in (self.x_tf, self.y_tf):
            w.currentIndexChanged.connect(self.replot)
        self.plot.on_pick(self.go_to_row)
        from ._cyto_cluster import ClusteringTab

        self.clustering = ClusteringTab(self)
        self.tabs.addTab(self.clustering, "Clustering")
        self.state.listen(self.refresh)
        self.refresh()

    # ---- refreshing ----------------------------------------------------------------------

    def refresh(self):
        """Rebuild the choices and lists from the state, keeping the current choices."""
        state = self.state
        current_pop = self.population.currentData()  # "kind\tkey" (Qt keeps strings, not tuples)
        features = state.features()
        _fill(self.population, [(label, _pop_key(kind, key)) for kind, key, label in state.populations()]
              if state.table is not None else [], current_pop)
        for box, default in ((self.x_col, 0), (self.y_col, 1)):
            keep = box.currentText()
            _fill(box, [(f, f) for f in features], keep or (features[min(default, len(features) - 1)]
                                                            if features else None))
        keep = self.color_by.currentText()
        _fill(self.color_by, [(COLOR_POPULATION, None), (COLOR_PHENOTYPE, None)] + [(f, f) for f in features],
              keep or COLOR_POPULATION, by_text=True)
        enabled = state.table is not None and bool(features)
        for b in (self.rect_btn, self.poly_btn, self.lasso_btn, self.image_btn):
            b.setEnabled(enabled)
        self._fill_gates()
        self._fill_phenotypes()
        self._watch_layer()
        self.replot()
        if hasattr(self, "clustering"):
            self.clustering.refresh()
        if state.table is None:
            self.status.setText("Choose the objects to analyse in the Cytometry section (Measure them first).")

    def _fill_gates(self):
        self._filling = True  # rebuilding the list is not the user choosing a gate
        try:
            self._fill_gate_items()
        finally:
            self._filling = False

    def _fill_gate_items(self):
        state, tree = self.state, self.gate_tree
        selected = self._selected_gate()
        tree.clear()
        if state.table is None:
            return
        frame = state.frame()
        n_all = len(frame)
        items = {}
        for kind, key in state.order:
            if kind == "gate":
                gate = state.gates.get(key)
                name, color = gate.name, gate.color
                parent_item = ("gate", gate.parent_id) if gate.parent_id else state.scope.get((kind, key))
            else:
                name, color = key, state.image_gates[key]["color"]
                parent_item = state.scope.get((kind, key))
            mask = state.mask(kind, key)
            parent_n = state.mask(*parent_item).sum() if parent_item else n_all
            label = name if kind == "gate" else f"{name} (image)"
            if parent_item and parent_item[0] == "phenotype":
                label += f"  [in {parent_item[1]}]"
            item = QTreeWidgetItem([label, f"{int(mask.sum()):,}", _pct(mask.sum(), parent_n)])
            item.setData(0, Qt.UserRole, (kind, key))
            item.setForeground(0, QColor(color))
            parent = items.get(tuple(parent_item)) if parent_item and parent_item[0] != "phenotype" else None
            (parent.addChild(item) if parent is not None else tree.addTopLevelItem(item))
            items[(kind, key)] = item
            if selected == (kind, key):
                item.setSelected(True)
        tree.expandAll()
        for c in range(3):
            tree.resizeColumnToContents(c)

    def _fill_phenotypes(self):
        state, tree = self.state, self.pheno_tree
        selected = self._selected_phenotype()
        tree.clear()
        counts = state.counts() if state.table is not None else None
        broken = state.broken_phenotypes() if state.table is not None else {}
        rows = counts.set_index("phenotype") if counts is not None else None
        items = {}
        for node in state.tree:
            if rows is not None:
                n, pct = f"{int(rows.loc[node.name, 'count']):,}", f"{rows.loc[node.name, '% of parent']:.1f}%"
                overlap = int(rows.loc[node.name, "overlap"])
                if overlap:
                    n += f" ({overlap} also in an earlier sibling)"
            else:
                n, pct = ("missing: " + ", ".join(sorted(broken[node.name]))) if node.name in broken else "", ""
            item = QTreeWidgetItem([node.name, n, pct, node.rule or "(rest of parent)"])
            item.setData(0, Qt.UserRole, node.name)
            item.setForeground(0, QColor(node.color or "#888888"))
            parent = items.get(node.parent)
            (parent.addChild(item) if parent is not None else tree.addTopLevelItem(item))
            items[node.name] = item
            if selected == node.name:
                item.setSelected(True)
        tree.expandAll()
        for c in range(4):
            tree.resizeColumnToContents(c)

    def current_population(self):
        """(kind, key) of the objects shown."""
        return _pop_parse(self.population.currentData())

    def set_population(self, kind, key=None):
        index = self.population.findData(_pop_key(kind, key))
        if index < 0:
            raise ValueError(f"no population {kind} {key!r}")
        self.population.setCurrentIndex(index)

    def plot_map(self, analysis):
        """Show an analysis's map on the Gating tab, coloured by its clusters."""
        info = self.state.analyses[analysis]
        cols = info["columns"]
        if cols.get("map"):
            x, y = (cols["map"] + cols["map"])[:2]
            for box, tf, col in ((self.x_col, self.x_tf, x), (self.y_col, self.y_tf, y)):
                box.setCurrentText(col)
                tf.setCurrentText("linear")
        if cols.get("cluster"):
            self.color_by.setCurrentText(cols["cluster"])
        pop = info.get("population") or ["all", None]
        try:
            self.set_population(pop[0], pop[1])
        except ValueError:
            self.set_population("all")
        self.tabs.setCurrentIndex(0)
        self.replot()

    def axes(self):
        return (cy.axis_column(self.x_col.currentText(), self.x_tf.currentText()),
                cy.axis_column(self.y_col.currentText(), self.y_tf.currentText()))

    def replot(self, *_):
        state = self.state
        if state.table is None or not self.x_col.currentText() or not self.y_col.currentText():
            self.plot.show([], [], [], xlabel="", ylabel="")
            return
        x_name, y_name = self.axes()
        x, y = state.column(x_name), state.column(y_name)
        kind, key = self.current_population()
        mask = state.mask(kind, key)
        index = np.flatnonzero(mask)
        background = (x[~mask], y[~mask]) if kind != "all" else None
        color_by = self.color_by.currentText()
        colors, legend, values = None, None, None
        if color_by == COLOR_PHENOTYPE and len(state.tree) and not state.broken_phenotypes():
            assign = state.assignment()["phenotype"].to_numpy()[index]
            palette = {n.name: n.color for n in state.tree}
            colors = np.array([palette.get(a, "#9e9e9e") for a in assign], dtype=object)
            legend = [(n, palette[n]) for n in palette if (assign == n).any()]
        elif color_by == COLOR_POPULATION:
            colors = _population_color(state, kind, key)
        elif color_by in state.features():
            values = state.column(color_by)[index]
            if _categorical(color_by, values):
                ids = values.astype(np.int64)
                colors = np.array([_category_color(i) for i in ids], dtype=object)
                legend = [(f"cluster {i}" if i > 0 else "not clustered", _category_color(i))
                          for i in np.unique(ids)][:20]
                values = None
        gates = []
        for gate in state.gates:  # gates drawn on these axes, directly within the shown population
            if (gate.x_axis, gate.y_axis) != (x_name, y_name):
                continue
            within = ("gate", gate.parent_id) if gate.parent_id else state.scope.get(("gate", gate.id), ("all", None))
            if tuple(within) == (kind, key) or (kind, key) == ("gate", gate.id):
                gates.append((gate.vertices, gate.color, gate.name))
        self.plot.show(x[index], y[index], index, colors=colors, background=background,
                       xlabel=x_name, ylabel=y_name, gates=gates, legend=legend, values=values,
                       cbar_label=color_by if values is not None else None)

    # ---- gating -----------------------------------------------------------------------------

    def start_gate(self, mode):
        if self.state.table is None:
            return
        self.status.setText({"rectangle": "Drag a rectangle on the plot.",
                             "polygon": "Click the corners; click the first one again to close.",
                             "lasso": "Drag around the objects."}[mode])
        self.plot.draw_gate(mode, self._gate_drawn)

    def _gate_drawn(self, vertices):
        name, ok = QInputDialog.getText(self, "New gate", "Name of the gate:",
                                        text=self.state.unique_name("Gate"))
        if not ok or not name.strip():
            self.status.setText("Gate discarded.")
            return
        self.add_gate(name.strip(), vertices)

    def add_gate(self, name, vertices):
        """A gate on the current axes within the shown population (also used by scripts)."""
        x_name, y_name = self.axes()
        population = self.current_population()
        try:
            gate = self.state.add_gate(name, x_name, y_name, vertices, population)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        n = int(self.state.mask("gate", gate.id).sum())
        self.status.setText(f"Gate {name}: {n:,} objects.")
        self._map_gate(("gate", gate.id))
        return gate

    def image_gate_clicked(self):
        """First click: a Shapes layer to draw on. Second click: make the image gate."""
        layer = self._image_layer
        if layer is None or layer not in self.viewer.layers:
            objects = self.state.labels_layer()
            if objects is None:
                self.status.setText(f"Open the objects layer {self.state.layer_name!r} first.")
                return
            layer = self.viewer.add_shapes(name=IMAGE_GATE_LAYER, ndim=2, scale=objects.scale[-2:],
                                           translate=objects.translate[-2:], edge_color="yellow",
                                           face_color=[1, 1, 0, 0.15], edge_width=2)
            layer.mode = "add_polygon"
            self.viewer.layers.selection.active = layer
            self._image_layer = layer
            self.image_btn.setText("Make image gate")
            self.status.setText("Draw regions on the image (polygons or rectangles), then click "
                                "Make image gate.")
            return
        polygons = [_to_world(layer, v) for v in layer.data]
        if not polygons:
            self.status.setText("Draw at least one region first.")
            return
        name, ok = QInputDialog.getText(self, "New image gate", "Name of the gate:",
                                        text=self.state.unique_name("Region"))
        if ok and name.strip():
            self.add_image_gate(name.strip(), polygons)
        self.viewer.layers.remove(layer)
        self._image_layer = None
        self.image_btn.setText("Gate on image…")

    def add_image_gate(self, name, polygons):
        population = self.current_population()
        try:
            self.state.add_image_gate(name, polygons, population)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        n = int(self.state.mask("image", name).sum())
        self.status.setText(f"Image gate {name}: {n:,} objects.")
        self._map_gate(("image", name))
        return name

    def _selected_gate(self):
        items = self.gate_tree.selectedItems()
        return tuple(items[0].data(0, Qt.UserRole)) if items else None

    def _gate_selected(self):
        sel = self._selected_gate()
        for b in (self.gate_pheno_btn, self.gate_rename_btn, self.gate_delete_btn):
            b.setEnabled(sel is not None)
        if sel is not None and not self._filling:
            self._map_gate(sel)

    def _map_gate(self, sel):
        """Colour the gate's objects on their segmented pixels in the image (the
        "<objects> selection" layer), keeping the status message."""
        text = self.status.text()
        try:
            layer = self.state.show_population(*sel)
        except (ValueError, KeyError) as e:
            self.status.setText(f"{text} Not shown on the image: {e}".strip())
            return None
        n = int(self.state.mask(*sel).sum())
        self.status.setText(f"{text} Shown on the image in {layer.name} ({n:,} objects).".strip()
                            if text else f"{n:,} objects shown on the image in {layer.name}.")
        return layer

    def show_gate(self):
        """Show the selected gate's objects on the image (also done when it is selected)."""
        sel = self._selected_gate()
        if sel:
            self.status.clear()
            self._map_gate(sel)

    def phenotype_from_gate(self, name=None, parent=None):
        sel = self._selected_gate()
        if not sel:
            return None
        if name is None:
            dialog = PhenotypeDialog(self.state, self, title="Phenotype from gate",
                                     name=self._gate_name(sel), parent_name=self._default_parent(sel),
                                     rule=cy.quote(self.state.gate_column(*sel)))
            if dialog.exec() != QDialog.Accepted:
                return None
            name, parent, rule, color = dialog.values()
            return self._add_phenotype(name, rule, parent, color)
        return self._add_phenotype(name, cy.quote(self.state.gate_column(*sel)), parent, None)

    def _gate_name(self, sel):
        return self.state.gates.get(sel[1]).name if sel[0] == "gate" else sel[1]

    def _default_parent(self, sel):
        scope = self.state.scope.get(sel)
        return scope[1] if scope and scope[0] == "phenotype" else None

    def rename_gate(self):
        sel = self._selected_gate()
        if not sel:
            return
        new, ok = QInputDialog.getText(self, "Rename gate", "New name:", text=self._gate_name(sel))
        if ok and new.strip():
            try:
                self.state.rename_gate(*sel, new.strip())
            except ValueError as e:
                self.status.setText(str(e))

    def delete_gate(self):
        sel = self._selected_gate()
        if sel:
            self.state.remove_gate(*sel)
            broken = self.state.broken_phenotypes()
            if broken:
                self.status.setText("These phenotypes used the deleted gate and need a new rule: "
                                    + ", ".join(broken))

    # ---- phenotypes --------------------------------------------------------------------------

    def _selected_phenotype(self):
        items = self.pheno_tree.selectedItems()
        return items[0].data(0, Qt.UserRole) if items else None

    def _add_phenotype(self, name, rule, parent, color):
        try:
            node = self.state.add_phenotype(name, rule, parent, color)
        except (ValueError, KeyError) as e:
            self.status.setText(str(e))
            return None
        self.status.setText(f"Phenotype {self.state.tree.path(name)} added.")
        return node

    def add_phenotype(self):
        dialog = PhenotypeDialog(self.state, self, title="New phenotype",
                                 parent_name=self._selected_phenotype())
        if dialog.exec() == QDialog.Accepted:
            self._add_phenotype(*dialog.values())

    def edit_phenotype(self):
        name = self._selected_phenotype()
        if not name:
            return
        node = self.state.tree.get(name)
        dialog = PhenotypeDialog(self.state, self, title="Edit phenotype", name=name, parent_name=node.parent,
                                 rule=node.rule, color=node.color, editing=True)
        if dialog.exec() == QDialog.Accepted:
            new_name, _, rule, color = dialog.values()
            try:
                self.state.edit_phenotype(name, rule=rule, color=color, new_name=new_name)
            except ValueError as e:
                self.status.setText(str(e))

    def delete_phenotype(self):
        name = self._selected_phenotype()
        if name:
            removed = self.state.remove_phenotype(name)
            self.status.setText("Removed " + ", ".join(removed) + ".")

    def show_phenotypes(self):
        level = self.level.value() or None
        self._paint(lambda: self.state.show_phenotypes(level))

    def export_table(self, path=None):
        if self.state.table is None:
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(self, "Export objects with phenotypes",
                                                  "cytometry objects.csv", "CSV (*.csv);;Excel (*.xlsx)")
            if not path:
                return None
        try:
            from . import _measure

            path = _measure.save_table(self.state.typed_table(), path)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        self.status.setText(f"Saved {path}")
        return path

    # ---- linking to the viewer -------------------------------------------------------------

    def _paint(self, make):
        try:
            layer = make()
        except ValueError as e:
            self.status.setText(str(e))
            return
        self.status.setText(f"Shown in layer {layer.name}.")

    def go_to_row(self, row):
        """Select the object of table row `row` in the viewer and centre on it."""
        state, table = self.state, self.state.table
        layer = state.labels_layer()
        if layer is None:
            return
        rec = table.iloc[row]
        layer.selected_label = int(rec["label"])
        dims = self.viewer.dims
        world = [float(rec[c]) for axis, c in state.centroid_columns().items()
                 if np.isfinite(rec[c])]
        if world:
            camera = getattr(getattr(self.viewer, "scene", None), "camera", None) or self.viewer.camera
            camera.center = tuple(world[-dims.ndisplay:])
            if dims.ndisplay == 2 and len(world) == 3 and dims.ndim >= 3:
                dims.set_point(dims.ndim - 3, world[0])
        axes = (layer.metadata or {}).get("ibif_axes") or ""
        offset = dims.ndim - layer.ndim
        for k, name in enumerate(axes):
            if name in ("T", "P") and name in table.columns:
                dims.set_point(offset + k, float(layer.translate[k] + rec[name] * layer.scale[k]))
        self.plot.highlight(row)
        self.status.setText(f"Object {int(rec['label'])}.")

    def _watch_layer(self):
        layer = self.state.labels_layer()
        if layer is self._watched:
            return
        if self._watched is not None:
            try:
                self._watched.events.selected_label.disconnect(self._label_selected)
            except Exception:
                pass
        self._watched = layer
        if layer is not None:
            layer.events.selected_label.connect(self._label_selected)

    def _label_selected(self, event=None):
        table, layer = self.state.table, self._watched
        if table is None or layer is None:
            return
        rows = table["label"].to_numpy() == layer.selected_label
        dims = self.viewer.dims
        axes = (layer.metadata or {}).get("ibif_axes") or ""
        offset = dims.ndim - layer.ndim
        for k, name in enumerate(axes):
            if name in ("T", "P") and name in table.columns:
                step = round((dims.point[offset + k] - layer.translate[k]) / layer.scale[k])
                rows &= table[name].to_numpy() == step
        hit = np.flatnonzero(rows)
        self.plot.highlight(int(hit[0]) if len(hit) else None)


class PhenotypeDialog(QDialog):
    """Name, parent, rule and colour of a phenotype."""

    def __init__(self, state, parent=None, title="Phenotype", name="", parent_name=None, rule="",
                 color="", editing=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.state = state
        form = QFormLayout(self)
        self.name = QLineEdit(name or state.unique_name("Phenotype"))
        form.addRow("Name", self.name)
        self.parent_box = QComboBox()
        self.parent_box.addItem("(top level)", None)
        for node in state.tree:
            if editing and (node.name == name or name in state.tree.ancestors(node.name)):
                continue
            self.parent_box.addItem("  " * (state.tree.depth(node.name) - 1) + node.name, node.name)
        index = self.parent_box.findData(parent_name)
        self.parent_box.setCurrentIndex(max(index, 0))
        self.parent_box.setEnabled(not editing)
        form.addRow("Within", self.parent_box)
        self.rule = QLineEdit(rule)
        self.rule.setPlaceholderText("empty = every object of the parent not in an earlier phenotype")
        self.rule.setToolTip("A rule over the object table, e.g. `gate: CD3+` AND NOT `gate: CD4+`, "
                             "`CD8 mean` > 200, or cluster in [3, 7]. Names with spaces go in backticks.")
        form.addRow("Rule", self.rule)
        insert = QComboBox()
        insert.addItem("Insert a column…")
        frame = state.frame()
        if frame is not None:
            for c in frame.columns:
                insert.addItem(c)
        insert.activated.connect(lambda i: i and (self.rule.insert(cy.quote(insert.itemText(i)) + " "),
                                                  insert.setCurrentIndex(0)))
        form.addRow("", insert)
        self._color = color or state._next_color()
        self.color_btn = QPushButton()
        self.color_btn.clicked.connect(self._pick_color)
        self._show_color()
        form.addRow("Colour", self.color_btn)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)

    def _pick_color(self):
        c = QColorDialog.getColor(QColor(self._color), self)
        if c.isValid():
            self._color = c.name()
            self._show_color()

    def _show_color(self):
        self.color_btn.setText(self._color)
        self.color_btn.setStyleSheet(f"background-color: {self._color};")

    def values(self):
        return self.name.text().strip(), self.parent_box.currentData(), self.rule.text().strip(), self._color


# ---- small helpers ----------------------------------------------------------------------------


def _pop_key(kind, key):
    return f"{kind}\t{key if key is not None else ''}"


def _pop_parse(value):
    if not value:
        return ("all", None)
    kind, _, key = value.partition("\t")
    return (kind, key or None)


def _categorical(name, values):
    """Cluster ids are coloured as categories, not on a colour scale."""
    return name.endswith(" cluster") and len(values) > 0


def _category_color(i):
    return "#bdbdbd" if i <= 0 else cy.PALETTE[(int(i) - 1) % len(cy.PALETTE)]


def _transform_box():
    box = QComboBox()
    box.addItems(list(cy.TRANSFORMS))
    box.setToolTip("Axis scale: linear, log10, or asinh(x/150) (like log, but keeps zero and negatives)")
    return box


def _tree(headers):
    tree = QTreeWidget()
    tree.setHeaderLabels(headers)
    tree.setSelectionMode(QAbstractItemView.SingleSelection)
    tree.setRootIsDecorated(True)
    return tree


def _button(text, slot, layout):
    b = QPushButton(text)
    b.clicked.connect(lambda _=False: slot())
    layout.addWidget(b)
    return b


def _fill(box, items, keep, by_text=False):
    box.blockSignals(True)
    box.clear()
    for label, data in items:
        box.addItem(label, data)
    index = box.findText(keep) if by_text or isinstance(keep, str) else box.findData(keep)
    box.setCurrentIndex(max(index, 0) if box.count() else -1)
    box.blockSignals(False)


def _pct(n, total):
    return f"{100.0 * n / total:.1f}%" if total else ""


def _population_color(state, kind, key):
    if kind == "gate":
        return state.gates.get(key).color
    if kind == "image":
        return state.image_gates[key]["color"]
    if kind == "phenotype":
        return state.tree.get(key).color or "#1f77b4"
    return "#1f77b4"


def _to_world(layer, vertices):
    """Shapes-layer vertices (data coordinates) in world coordinates (Y, X)."""
    v = np.asarray(vertices, float)[:, -2:]
    return v * np.asarray(layer.scale[-2:], float) + np.asarray(layer.translate[-2:], float)


_workbenches = {}


def open_workbench(viewer, near=None):
    """The viewer's workbench window, created on first use and raised. `near` is a
    widget in the napari window (the workbench stays above that window and closes with
    it; napari only gives plugins a proxy of its window object, which Qt cannot use)."""
    bench = _workbenches.get(id(viewer))
    if bench is None or bench.viewer is not viewer:
        window = near.window() if near is not None else None
        bench = Workbench(viewer, window)
        _workbenches[id(viewer)] = bench
    bench.show()
    bench.raise_()
    bench.activateWindow()
    return bench


__all__ = ["Workbench", "PhenotypeDialog", "open_workbench"]
