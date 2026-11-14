"""The Cytometry workbench: a separate window for gating and phenotyping.

Plot any two measurements (linear, log10 or asinh), draw rectangle / polygon / lasso
gates on it or paint regions on the image (with the same brush, and 3D ball brush, as
the segmentation region), gate within a gate or within a phenotype, and turn gates into
a hierarchy of phenotypes. Everything is linked to the napari viewer: selecting gates or
phenotypes shows their objects on the image, objects in an image gate are ringed on the
plot, clicking a point goes to that object, and clicking an object in the viewer rings
it on the plot. It is one window per viewer and shares
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
    QTreeWidgetItemIterator,
    QVBoxLayout,
    QWidget,
)

from . import _cytometry as cy
from ._cyto_plot import ASPECT, ScatterPlot

IMAGE_GATE_LAYER = "Cytometry image gate"
COLOR_POPULATION, COLOR_PHENOTYPE = "Population", "Phenotype"


class Workbench(QWidget):
    def __init__(self, viewer, parent=None):
        super().__init__(parent, Qt.Window)
        self.viewer = viewer
        self.state = cy.state_of(viewer)
        self.setWindowTitle("Cytometry workbench")
        self._image_layer = None
        self._watched = None
        self._filling = False
        outer = QVBoxLayout(self)
        # ---- which level of the spatial hierarchy is analysed ----
        level_row = QHBoxLayout()
        level_row.addWidget(QLabel("Level"))
        self.level_box = QComboBox()
        self.level_box.setToolTip(
            "Which objects every tab works on: the segmented objects, or a spatial level built "
            "on them (neighbourhood windows or regions), at any depth")
        self.level_box.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.level_box.currentIndexChanged.connect(self._level_chosen)
        level_row.addWidget(self.level_box)
        self.level_path = QLabel()
        self.level_path.setToolTip("Where this level sits: what it is built on, down to the segmentation")
        level_row.addWidget(self.level_path, 1)
        outer.addLayout(level_row)
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
        self._top_row = row

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
        self.image_btn.setToolTip("Paint over the objects on the image (the segmentation brush; in the 3D "
                                  "view it paints balls). Objects whose centre is painted are gated. "
                                  "Click again (Make image gate) when done.")
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
        self._left = left

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        gates_box = QGroupBox("Gates")
        gl = QVBoxLayout(gates_box)
        self.gate_tree = _tree(["Gate", "Objects", "% of parent"])
        self.gate_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.gate_tree.setToolTip("Select gates to show them on the image; Ctrl- or Shift-click "
                                  "to show several at once")
        self.gate_tree.itemSelectionChanged.connect(self._gate_selected)
        gl.addWidget(self.gate_tree)
        gb = QHBoxLayout()
        self.gate_pheno_btn = _button("Make phenotype…", self.phenotype_from_gate, gb)
        self.gate_rename_btn = _button("Rename…", self.rename_gate, gb)
        self.gate_color_btn = _button("Colour…", self.recolor_gate, gb)
        self.gate_color_btn.setToolTip("Colour of the gate: its outline on the plot and its objects on the "
                                       "image; for an image gate also the ring around its objects on the plot")
        self.gate_delete_btn = _button("Delete", self.delete_gate, gb)
        gl.addLayout(gb)
        right_layout.addWidget(gates_box, 1)

        pheno_box = QGroupBox("Phenotypes")
        pl = QVBoxLayout(pheno_box)
        self.pheno_tree = _tree(["Phenotype", "Objects", "% of parent", "Rule"])
        self.pheno_tree.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.pheno_tree.setToolTip("Select phenotypes to show their objects on the image; Ctrl- or "
                                   "Shift-click to show several at once. Double-click to edit one.")
        self.pheno_tree.itemDoubleClicked.connect(lambda *_: self.edit_phenotype())
        self.pheno_tree.itemSelectionChanged.connect(self._phenotype_selected)
        pl.addWidget(self.pheno_tree)
        pb = QHBoxLayout()
        _button("Add…", self.add_phenotype, pb)
        self.pheno_edit_btn = _button("Edit…", self.edit_phenotype, pb)
        self.pheno_delete_btn = _button("Delete", self.delete_phenotype, pb)
        pl.addLayout(pb)
        pb2 = QHBoxLayout()
        self.level = QSpinBox()
        self.level.setRange(0, 9)
        self.level.setSpecialValueText("deepest")
        self.level.setToolTip("Hierarchy level to colour by (deepest = the most specific phenotype)")
        pb2.addWidget(QLabel("Level"))
        pb2.addWidget(self.level)
        _button("Show on image", self.show_phenotypes, pb2).setToolTip(
            "Colour every object on the image by its phenotype at this level")
        _button("Export…", self.export_table, pb2).setToolTip(
            "Export every object with its gates, phenotype and path (CSV or Excel)")
        pl.addLayout(pb2)
        right_layout.addWidget(pheno_box, 1)
        split.addWidget(right)
        split.setStretchFactor(0, 2)  # the side panels stay about half the plot's width
        split.setStretchFactor(1, 1)
        split.setChildrenCollapsible(False)
        self.split, self.side = split, right

        for w in (self.population, self.x_col, self.y_col, self.color_by):
            w.currentIndexChanged.connect(self.replot)
            # Long feature names must not widen the window: the box shows about 18
            # characters, its list the whole name, and its tooltip the current one.
            w.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            w.setMinimumContentsLength(14)
            w.currentTextChanged.connect(lambda text, box=w: box.setToolTip(text))
        for w in (self.x_tf, self.y_tf):
            w.currentIndexChanged.connect(self.replot)
        self.plot.on_pick(self.go_to_row)
        self.plot.on_style = self._style_changed
        self.plot.size_source = lambda name: self.state.column(name)
        self.plot.size_choices = lambda: self.state.features()
        viewer.layers.events.removed.connect(self._layer_removed)
        from ._cyto_cluster import ClusteringTab

        from ._cyto_classify import ClassificationTab
        from ._cyto_spatial import NeighbourhoodTab, SpatialTab

        self.clustering = ClusteringTab(self)
        self.tabs.addTab(self.clustering, "Clustering")
        self.spatial = SpatialTab(self)
        self.tabs.addTab(self.spatial, "Spatial")
        self.neighbourhoods = NeighbourhoodTab(self)
        self.tabs.addTab(self.neighbourhoods, "Neighbourhoods")
        self.classification = ClassificationTab(self)
        self.tabs.addTab(self.classification, "Classification")
        self.state.listen(self.refresh)
        self.refresh()
        self.fit_to_screen()

    # ---- levels -------------------------------------------------------------------------

    def _fill_levels(self):
        levels = self.state
        box = self.level_box
        box.blockSignals(True)
        box.clear()
        for depth, name in levels.level_tree():
            info = levels.level_info(name)
            box.addItem(f"{'   ' * depth}{levels.level_label(name)}  ({info['objects']:,})", name)
        box.setCurrentIndex(max(box.findData(levels.current_level), 0))
        box.setEnabled(box.count() > 1)
        box.blockSignals(False)
        path = []
        name = levels.current_level
        while name is not None:
            path.insert(0, name)
            name = levels.level_info(name)["parent"]
        self.level_path.setText(" › ".join(path))
        depth = levels.level_info(levels.current_level)["depth"]
        self.setWindowTitle("Cytometry workbench" + (f": spatial level {depth}, {levels.current_level}"
                                                     if depth else ""))

    def _level_chosen(self, index):
        name = self.level_box.itemData(index)
        if name and name != self.state.current_level:
            self.set_level(name)

    def set_level(self, name):
        """Work on level `name` (every tab follows); also used by scripts."""
        self.state.set_level(name)

    def fit_to_screen(self, available=None):
        """Size the window for a 4:3 plot with side panels about half its width: as large
        as fits on the screen (92% x 88% of its available area), but never so small that
        the controls are squeezed. `available` = (width, height) in pixels, default the
        screen of the napari window. Returns (plot width, side width)."""
        if available is None:
            available = _screen_size(self.parentWidget())
        max_w, max_h = int(available[0] * 0.92), int(available[1] * 0.88)
        self.layout().activate()
        side_min = max(self.side.minimumSizeHint().width(), 260)
        controls_w = self.tabs.minimumSizeHint().width()  # the Show / X / Y / Colour row
        # Height of everything in the plot's column that is not the plot: the row above
        # it, the plot toolbar, the gate buttons and status below, the tab bar, margins.
        # (The side tables shrink and scroll, so they do not count.)
        line = self.status.fontMetrics().lineSpacing()
        chrome_h = (self._top_row.sizeHint().height() + self.plot.toolbar.sizeHint().height()
                    + self.rect_btn.sizeHint().height() + 2 * line
                    + self.tabs.tabBar().sizeHint().height() + 48)
        chrome_w = 36
        plot_w = min((max_w - chrome_w) / 1.5, (max_h - chrome_h) * ASPECT)
        plot_w = int(max(plot_w, 480))
        side_w = max(plot_w // 2, side_min)
        min_w = min(max(side_min + 480 + chrome_w, controls_w), max_w)
        min_h = min(int(480 / ASPECT) + chrome_h, max_h)
        self.setMinimumSize(min_w, min_h)
        width = max(min(plot_w + side_w + chrome_w, max_w), min_w)
        height = max(min(int(plot_w / ASPECT) + chrome_h, max_h), min_h)
        self.resize(width, height)
        self.split.setSizes([plot_w, side_w])
        return plot_w, side_w

    # ---- refreshing ----------------------------------------------------------------------

    def refresh(self):
        """Rebuild the choices and lists from the state, keeping the current choices."""
        state = self.state
        self._fill_levels()
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
        view_key = (state.current_level, state.view_version)
        if getattr(self, "_view_version", None) != view_key:
            self._view_version = view_key
            self.apply_view(state.view)  # e.g. restored from a workspace, or another level's
        self.replot()
        for tab in ("clustering", "spatial", "neighbourhoods", "classification"):
            if hasattr(self, tab):
                getattr(self, tab).refresh()
        if state.table is None:
            self.status.setText("Choose the objects to analyse in the Cytometry section (Measure them first).")
        else:
            missing = state.missing_gate_columns()
            if missing:
                names = ", ".join(self._gate_name(i) for i in missing)
                cols = sorted({c for v in missing.values() for c in v})
                self.status.setText(
                    f"Gates {names} are drawn on {', '.join(cols)}, which is not in the table any more: "
                    "they select nothing. Run the clustering again under the same name to bring them "
                    "back, or delete them.")

    def _fill_gates(self):
        self._filling = True  # rebuilding the list is not the user choosing a gate
        try:
            self._fill_gate_items()
        finally:
            self._filling = False

    def _fill_gate_items(self):
        state, tree = self.state, self.gate_tree
        selected = set(self.selected_gates())
        tree.clear()
        if state.table is None:
            return
        frame = state.frame()
        n_all = len(frame)
        missing = state.missing_gate_columns()
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
            if (kind, key) in missing:
                gone = ", ".join(missing[(kind, key)])
                item.setText(1, "missing column")
                item.setText(2, "")
                item.setForeground(1, QColor("#d62728"))
                for c in range(3):
                    item.setToolTip(c, f"Drawn on {gone}, which is not in the table any more (a "
                                       "clustering result deleted, or cleared by measuring again). "
                                       "It selects nothing until a result of the same name is made "
                                       "again; or delete the gate.")
            parent = items.get(tuple(parent_item)) if parent_item and parent_item[0] != "phenotype" else None
            (parent.addChild(item) if parent is not None else tree.addTopLevelItem(item))
            items[(kind, key)] = item
            if (kind, key) in selected:
                item.setSelected(True)
        tree.expandAll()
        for c in range(3):
            tree.resizeColumnToContents(c)

    def _fill_phenotypes(self):
        self._filling = True  # rebuilding the list is not the user choosing phenotypes
        try:
            self._fill_phenotype_items()
        finally:
            self._filling = False
        self._phenotype_buttons()

    def _fill_phenotype_items(self):
        state, tree = self.state, self.pheno_tree
        selected = set(self.selected_phenotypes())
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
            if node.name in selected:
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

    def view(self):
        """The plot's set-up: axes, scales, population, colour and point style."""
        plot = self.plot
        return {"x": self.x_col.currentText(), "x_scale": self.x_tf.currentText(),
                "y": self.y_col.currentText(), "y_scale": self.y_tf.currentText(),
                "population": self.population.currentData(), "colour": self.color_by.currentText(),
                "point_size": plot.point_size, "marker": plot.marker, "alpha": plot.alpha,
                "size_by": plot.size_by, "size_range": list(plot.size_range),
                "ring_image_gates": plot.outline_image_gates}

    def _style_changed(self):
        if self.state.table is not None and self.x_col.currentText():
            self.state.view = self.view()

    def apply_view(self, view):
        """Set the plot up as `view` says (what is not available any more is skipped)."""
        if not view:
            return
        boxes = [self.population, self.x_col, self.y_col, self.color_by, self.x_tf, self.y_tf]
        for box in boxes:
            box.blockSignals(True)
        try:
            for box, key in ((self.x_col, "x"), (self.y_col, "y"), (self.color_by, "colour"),
                             (self.x_tf, "x_scale"), (self.y_tf, "y_scale")):
                if view.get(key) and box.findText(view[key]) >= 0:
                    box.setCurrentText(view[key])
            index = self.population.findData(view.get("population"))
            if index >= 0:
                self.population.setCurrentIndex(index)
        finally:
            for box in boxes:
                box.blockSignals(False)
        style = {k: view[k] for k in ("point_size", "marker", "alpha", "size_by", "size_range") if k in view}
        if "ring_image_gates" in view:
            style["outline_image_gates"] = bool(view["ring_image_gates"])
        if style.get("size_by") is not None and style["size_by"] not in self.state.features():
            style["size_by"] = None  # a measurement that is not in the table any more
        if style:
            self.plot.set_point_style(**style)  # ticks the menu (and redraws)

    def replot(self, *_):
        state = self.state
        if state.table is not None and self.x_col.currentText():
            state.view = self.view()  # saved with the workspace
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
                names = state.class_names(color_by)
                if names:  # a classification: its class names and colours
                    info = next(i for i in state.analyses.values() if i["columns"].get("class") == color_by)
                    palette = {k: info["colors"][c] for k, c in names.items()}
                    colors = np.array([palette.get(int(i), "#bdbdbd") for i in ids], dtype=object)
                    legend = [(names.get(int(i), "not classified"), palette.get(int(i), "#bdbdbd"))
                              for i in np.unique(ids)][:20]
                else:
                    what = "type" if color_by.endswith(" neighbourhood") else "cluster"
                    legend = [(f"{what} {i}" if i > 0 else f"no {what}", _category_color(i))
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
                       cbar_label=color_by if values is not None else None,
                       outlines=state.image_gated())

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
        """First click: a labels layer to paint the region on, over the objects (the
        segmentation region's brush). Second click: make the image gate from it."""
        layer = self._image_layer
        if layer is None or layer not in self.viewer.layers:
            self.start_image_gate()
            return
        if not np.asarray(layer.data).any():
            self.status.setText(f"Paint over the objects in the {layer.name!r} layer first (or delete "
                                "that layer to cancel).")
            self.viewer.layers.selection.active = layer
            layer.mode = "paint"
            return
        name, ok = QInputDialog.getText(self, "New image gate", "Name of the gate:",
                                        text=self.state.unique_name("Region"))
        if not ok or not name.strip():
            self.status.setText("Not made: keep painting and click Make image gate again, or delete the "
                                f"{layer.name!r} layer to cancel.")
            return
        if self.make_image_gate(name.strip()) is not None:
            self._end_image_gate()

    def start_image_gate(self):
        """Add the layer to paint an image gate on (also used by tests and scripts)."""
        from ._cytometry import _spatial_ndim

        objects = self.state.labels_layer()
        if objects is None:
            self.status.setText(f"Open the objects layer {self.state.layer_name!r} first.")
            return None
        n = min(_spatial_ndim(objects), 3)
        shape = tuple(objects.data.shape[-n:])
        layer = self.viewer.add_labels(np.zeros(shape, np.uint8), name=IMAGE_GATE_LAYER,
                                       scale=objects.scale[-n:], translate=objects.translate[-n:],
                                       opacity=0.4, metadata={"ibif_region": True, "ibif_cytometry": True})
        layer.brush_size = max(10, min(shape[-2:]) // 15)
        layer.selected_label = 1
        layer.mode = "paint"
        try:  # the 3D view paints balls on the image, as for the segmentation region
            from ._draw3d import brush_for

            brush_for(self.viewer)
        except Exception:
            pass
        self.viewer.layers.selection.active = layer
        self._image_layer = layer
        self.image_btn.setText("Make image gate")
        depth = (" A region painted in 2D counts on every Z plane; painted with the ball brush in the 3D "
                 "view it keeps its depth." if n == 3 else "")
        self.status.setText(f"Paint over the objects to gate in the {layer.name!r} layer (brush size and "
                            f"eraser in the layer controls), then click Make image gate.{depth} Delete the "
                            "layer to cancel.")
        return layer

    def make_image_gate(self, name):
        """The image gate `name` from what is painted (also used by tests and scripts)."""
        layer = self._image_layer
        if layer is None or layer not in self.viewer.layers:
            return None
        mask = np.asarray(layer.data) != 0
        scale, translate = np.asarray(layer.scale, float), np.asarray(layer.translate, float)
        if mask.ndim == 3 and not (layer.metadata or {}).get("ibif_region_3d"):
            # Painted plane by plane: what was painted on any plane counts on all of them.
            mask, scale, translate = mask.any(axis=0), scale[-2:], translate[-2:]
        return self.add_image_gate(name, mask=mask, scale=scale, translate=translate)

    def _end_image_gate(self):
        layer, self._image_layer = self._image_layer, None
        self.image_btn.setText("Gate on image…")
        if layer is not None and layer in self.viewer.layers:
            self.viewer.layers.remove(layer)

    def _layer_removed(self, event=None):
        if self._image_layer is not None and self._image_layer not in self.viewer.layers:
            self._image_layer = None
            self.image_btn.setText("Gate on image…")
            self.status.setText("Image gate cancelled.")

    def add_image_gate(self, name, polygons=None, mask=None, scale=None, translate=None):
        population = self.current_population()
        try:
            self.state.add_image_gate(name, polygons, population, mask=mask, scale=scale, translate=translate)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        n = int(self.state.mask("image", name).sum())
        self.status.setText(f"Image gate {name}: {n:,} objects.")
        self._map_gate(("image", name))
        return name

    def _selected_gate(self):
        """The selected gate when exactly one is selected (for Make phenotype, Rename)."""
        gates = self.selected_gates()
        return gates[0] if len(gates) == 1 else None

    def selected_gates(self):
        """Every selected gate, in list order."""
        chosen = {id(i) for i in self.gate_tree.selectedItems()}
        out = []
        it = QTreeWidgetItemIterator(self.gate_tree)
        while it.value():
            if id(it.value()) in chosen:
                out.append(tuple(it.value().data(0, Qt.UserRole)))
            it += 1
        return out

    def select_gates(self, items):
        """Select these gates (kind, key) in the list (also used by tests and scripts)."""
        wanted = {tuple(i) for i in items}
        self.gate_tree.blockSignals(True)
        it = QTreeWidgetItemIterator(self.gate_tree)
        while it.value():
            it.value().setSelected(tuple(it.value().data(0, Qt.UserRole)) in wanted)
            it += 1
        self.gate_tree.blockSignals(False)
        self._gate_selected()

    def _gate_selected(self):
        gates = self.selected_gates()
        self.gate_pheno_btn.setEnabled(len(gates) == 1)
        self.gate_rename_btn.setEnabled(len(gates) == 1)
        self.gate_color_btn.setEnabled(len(gates) == 1)
        self.gate_delete_btn.setEnabled(bool(gates))
        if gates and not self._filling:
            self._clear_selection(self.pheno_tree)  # the image shows one list's choice at a time
            self._map_gate(gates)

    def _clear_selection(self, tree):
        if tree.selectedItems():
            tree.blockSignals(True)
            tree.clearSelection()
            tree.blockSignals(False)
            if tree is self.pheno_tree:
                self._phenotype_buttons()
            else:
                self.gate_pheno_btn.setEnabled(False)
                self.gate_rename_btn.setEnabled(False)
                self.gate_color_btn.setEnabled(False)
                self.gate_delete_btn.setEnabled(False)

    def _map_gate(self, sel):
        """Colour the objects of one gate (kind, key) or several [(kind, key), ...] on
        their segmented pixels in the image (the "<objects> selection" layer), each gate
        in its colour; objects in more than one selected gate are white."""
        items = [sel] if sel and not isinstance(sel[0], (tuple, list)) else list(sel)
        text = self.status.text() if len(items) == 1 else ""
        try:
            layer, counts, several = self.state.show_populations(items)
        except (ValueError, KeyError) as e:
            self.status.setText(f"{text} Not shown on the image: {e}".strip())
            return None
        what = "phenotypes" if all(i[0] == "phenotype" for i in items) else "gates"
        if len(items) == 1:
            n = next(iter(counts.values()))
            self.status.setText(f"{text} Shown on the image in {layer.name} ({n:,} objects).".strip()
                                if text else f"{self._gate_name(items[0])}: {n:,} objects shown on the "
                                             f"image in {layer.name}.")
        else:
            parts = ", ".join(f"{self._gate_name(i)} {n:,}" for i, n in counts.items())
            extra = f"; {several:,} in more than one (white)" if several else ""
            self.status.setText(f"{len(items)} {what} shown on the image in {layer.name}: {parts}{extra}.")
        return layer

    def show_gate(self):
        """Show the selected gates' objects on the image (also done when they are selected)."""
        gates = self.selected_gates()
        if gates:
            self.status.clear()
            self._map_gate(gates)

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

    def recolor_gate(self, color=None):
        """Change the colour of the selected gate (asks for it when `color` is None)."""
        sel = self._selected_gate()
        if not sel:
            return None
        if color is None:
            current = self.state.population_color(*sel)
            picked = QColorDialog.getColor(QColor(current), self, f"Colour of {self._gate_name(sel)}")
            if not picked.isValid():
                return None
            color = picked.name()
        self.state.set_gate_color(*sel, color)
        self._map_gate(sel)
        return color

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
        for sel in self.selected_gates():
            if sel[0] == "gate" and sel[1] not in self.state.gates or \
                    sel[0] == "image" and sel[1] not in self.state.image_gates:
                continue  # removed with its parent gate already
            self.state.remove_gate(*sel)
        broken = self.state.broken_phenotypes()
        if broken:
            self.status.setText("These phenotypes used the deleted gates and need a new rule: "
                                + ", ".join(broken))

    # ---- phenotypes --------------------------------------------------------------------------

    def _selected_phenotype(self):
        """The selected phenotype when exactly one is selected (for Edit, and as the
        parent of a new one)."""
        names = self.selected_phenotypes()
        return names[0] if len(names) == 1 else None

    def selected_phenotypes(self):
        """Every selected phenotype, in tree order."""
        chosen = {id(i) for i in self.pheno_tree.selectedItems()}
        out = []
        it = QTreeWidgetItemIterator(self.pheno_tree)
        while it.value():
            if id(it.value()) in chosen:
                out.append(it.value().data(0, Qt.UserRole))
            it += 1
        return out

    def select_phenotypes(self, names):
        """Select these phenotypes in the list, which shows them on the image (also used
        by tests and scripts)."""
        wanted = set(names)
        self.pheno_tree.blockSignals(True)
        it = QTreeWidgetItemIterator(self.pheno_tree)
        while it.value():
            it.value().setSelected(it.value().data(0, Qt.UserRole) in wanted)
            it += 1
        self.pheno_tree.blockSignals(False)
        self._phenotype_selected()

    def _phenotype_buttons(self):
        names = self.selected_phenotypes()
        self.pheno_edit_btn.setEnabled(len(names) == 1)
        self.pheno_delete_btn.setEnabled(bool(names))

    def _phenotype_selected(self):
        """Selecting phenotypes shows their objects on the image, as selecting gates
        does: each in its colour, objects in more than one in white."""
        self._phenotype_buttons()
        names = self.selected_phenotypes()
        if not names or self._filling:
            return
        self._clear_selection(self.gate_tree)
        broken = self.state.broken_phenotypes()
        if broken:
            self.status.setText("Not shown on the image: fix the phenotypes whose rules use missing "
                                "columns first (" + ", ".join(sorted(broken)) + ").")
            return
        self.status.clear()
        self._map_gate([("phenotype", n) for n in names])

    def _add_phenotype(self, name, rule, parent, color):
        try:
            node = self.state.add_phenotype(name, rule, parent, color)
        except (ValueError, KeyError) as e:
            self.status.setText(str(e))
            return None
        rewrite = getattr(self.state, "last_rewrite", {})
        read = ("; read " + ", ".join(f"{k!r} as {cy.quote(v)}" for k, v in rewrite.items())) if rewrite else ""
        self.status.setText(f"Phenotype {self.state.tree.path(name)} added{read}.")
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
        removed = []
        for name in self.selected_phenotypes():
            if name in self.state.tree:  # not already removed with its parent
                removed += self.state.remove_phenotype(name)
        if removed:
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


RULE_HELP = (
    "<b>Columns</b>: gates <code>`gate: CD3+`</code>, image gates <code>`image gate: tumour`</code>, "
    "clusters <code>`Clustering 1 cluster`</code>, measurements <code>`cells: … CD8 mean`</code>, "
    "spatial <code>`Spatial 1 distance to Tumour (µm)`</code>, "
    "neighbourhoods <code>`Neighbourhoods 1 neighbourhood`</code>. "
    "Names with spaces or symbols go in <code>`backticks`</code> or quotes. A gate can be given by "
    "its name, a measurement by its heat-map label (<i>Cell CD8 mean</i>).<br>"
    "<b>Conditions</b>: <code>== != &lt; &lt;= &gt; &gt;=</code>, <code>in [3, 7]</code>, "
    "<code>50 &lt;= x &lt;= 150</code>; a gate alone means \"in the gate\".<br>"
    "<b>Combine</b>: <code>AND OR NOT XOR</code> (or <code>&amp; | ~ ^</code>), with ( ). "
    "Example: <code>`Clustering 1 cluster` == 2 &amp; `cells: CD8 mean` &lt; 100</code>.<br>"
    "Phenotypes at one level are exclusive: an object goes to the first that matches. "
    "An empty rule takes the rest of the parent."
)


class PhenotypeDialog(QDialog):
    """Name, parent, rule and colour of a phenotype, with a live check of the rule:
    how many objects it selects, or what is wrong with it."""

    def __init__(self, state, parent=None, title="Phenotype", name="", parent_name=None, rule="",
                 color="", editing=False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(620)
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
        form.addRow("Rule", self.rule)

        self.insert = QComboBox()
        self.insert.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.insert.setMinimumContentsLength(22)
        self._fill_insert()
        self.insert.activated.connect(self._insert_column)
        form.addRow("", self.insert)
        tools = QHBoxLayout()
        for text in ("AND", "OR", "NOT", "==", "<", ">", "in [ ]", "( )"):
            b = QPushButton(text)
            b.setToolTip({"in [ ]": "in a list of values, e.g. cluster in [3, 7]",
                          "( )": "brackets (around the selected text)"}.get(text, f"insert {text}"))
            b.clicked.connect(lambda _=False, t=text: self._insert_operator(t))
            tools.addWidget(b)
        form.addRow("", tools)
        help_label = QLabel(RULE_HELP)
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: gray;")
        form.addRow("", help_label)
        self.check = QLabel()
        self.check.setWordWrap(True)
        self.check.setTextInteractionFlags(Qt.TextSelectableByMouse)
        form.addRow("Check", self.check)

        self._color = color or state._next_color()
        self.color_btn = QPushButton()
        self.color_btn.clicked.connect(self._pick_color)
        self._show_color()
        form.addRow("Colour", self.color_btn)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        form.addRow(self.buttons)
        self.rule.textChanged.connect(self.check_rule)
        self.parent_box.currentIndexChanged.connect(self.check_rule)
        self.check_rule()

    def _fill_insert(self):
        """The columns a rule can use, grouped, each by its short label (the full name
        is in the tooltip and is what gets inserted)."""
        from ._cyto_labels import short_names

        box = self.insert
        box.addItem("Insert a column…", None)
        groups = self.state.rule_columns() if self.state.table is not None else {}
        short = short_names([c for cols in groups.values() for c in cols]) if groups else {}
        for group, cols in groups.items():
            if not cols:
                continue
            box.addItem(f"— {group} —", None)
            box.model().item(box.count() - 1).setEnabled(False)
            for c in cols:
                label = c if group != "Measurements" else short.get(c, c)
                box.addItem("   " + label, c)
                box.setItemData(box.count() - 1, c, Qt.ToolTipRole)

    def _insert_column(self, index):
        column = self.insert.itemData(index)
        if column:
            self._insert_text(cy.quote(column))
        self.insert.setCurrentIndex(0)

    def _insert_operator(self, text):
        if text == "in [ ]":
            self._insert_text("in [")
            self.rule.insert("]")
            self.rule.cursorBackward(False)
        elif text == "( )":
            selected = self.rule.selectedText()
            self.rule.insert(f"({selected})" if selected else "()")
            if not selected:
                self.rule.cursorBackward(False)
        else:
            self._insert_text(text)

    def _insert_text(self, text):
        """Insert at the cursor with a space on either side where needed."""
        line, pos = self.rule.text(), self.rule.cursorPosition()
        before = "" if pos == 0 or line[pos - 1] in " ([" else " "
        after = "" if pos < len(line) and line[pos] == " " else " "
        self.rule.insert(before + text + after)
        self.rule.setFocus()

    def check_rule(self, *_):
        """Show how many objects the rule selects (and within the chosen parent), how
        it will be stored when names were matched, or what is wrong; OK only when it
        can be used."""
        ok_button = self.buttons.button(QDialogButtonBox.Ok)
        text = self.rule.text().strip()
        if self.state.table is None:
            self.check.setText("")
            ok_button.setEnabled(True)
            return
        parent = self.parent_box.currentData()
        try:
            total, within, stored = self.state.rule_count(text, parent)
        except ValueError as e:
            self.check.setText(f"<span style='color:#d62728'>✗ {_html(str(e))}</span>")
            ok_button.setEnabled(False)
            return
        n_all = len(self.state.table)
        if not text:
            message = "✓ the rest of " + (f"<b>{_html(parent)}</b>" if parent else "the objects")
        else:
            message = f"✓ <b>{total:,}</b> of {n_all:,} objects match"
            if parent:
                message += f"; <b>{within:,}</b> of them are in {_html(parent)}"
        if stored != text:
            message += f"<br>stored as: <code>{_html(stored)}</code>"
        self.check.setText(f"<span style='color:#2ca02c'>{message}</span>")
        ok_button.setEnabled(True)

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


def _html(text):
    import html

    return html.escape(str(text))


def _pop_key(kind, key):
    return f"{kind}\t{key if key is not None else ''}"


def _pop_parse(value):
    if not value:
        return ("all", None)
    kind, _, key = value.partition("\t")
    return (kind, key or None)


def _categorical(name, values):
    """Cluster ids are coloured as categories, not on a colour scale."""
    return name.endswith((" cluster", " neighbourhood", " class")) and len(values) > 0


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


def _screen_size(widget=None):
    """(width, height) of the available area of the screen `widget` is on."""
    from qtpy.QtWidgets import QApplication

    handle = widget.windowHandle() if widget is not None else None
    screen = (handle.screen() if handle is not None else None) or QApplication.primaryScreen()
    if screen is None:
        return 1400, 900
    geometry = screen.availableGeometry()
    return geometry.width(), geometry.height()


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
