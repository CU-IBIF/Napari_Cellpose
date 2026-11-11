"""Cytometry section: choose the objects, see the phenotypes, open the workbench.

The gating and phenotyping themselves happen in the Cytometry workbench (a separate
window, so the plot can be large next to the image). This section picks which measured
objects are analysed, summarises the phenotype hierarchy, and shows it on the image.
"""
from __future__ import annotations

from qtpy.QtGui import QColor
from qtpy.QtWidgets import (
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _cytometry as cy
from . import _measure
from ._measure_widget import COMBINED

COMING = ("Coming next in this section: spatial features (distance to cell types, by "
          "centroid or surface) and neighbourhood analysis (cellular neighbourhoods, enrichment).")


class CytometryWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.state = cy.state_of(napari_viewer)
        layout = QVBoxLayout(self)
        intro = QLabel("Gate objects on a plot or on the image, cluster them to find types, "
                       "and build a hierarchy of phenotypes (e.g. Immune > T cell > CD8 T cell). "
                       "Measure the objects first (Measurements), then choose them here.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        self.table_box = QComboBox()
        self.table_box.setToolTip("A Measurements table: one row per segmented object (and time point / "
                                  "position), with its size and intensity measurements")
        form.addRow("Objects", self.table_box)
        self.layer_box = QComboBox()
        self.layer_box.setToolTip("The segmentation (labels layer) those objects are in: gates and "
                                  "phenotypes are shown on its pixels")
        form.addRow("Segmented in", self.layer_box)
        layout.addLayout(form)
        self.use_btn = QPushButton("Use these objects")
        self.use_btn.clicked.connect(self.use)
        layout.addWidget(self.use_btn)
        self.summary = QLabel()
        self.summary.setWordWrap(True)
        layout.addWidget(self.summary)
        self.open_btn = QPushButton("Open Cytometry workbench")
        self.open_btn.setToolTip("Plots, gates and phenotypes in a separate window")
        self.open_btn.clicked.connect(self.open_workbench)
        layout.addWidget(self.open_btn)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabels(["Phenotype", "Objects", "% of parent"])
        layout.addWidget(self.tree, 1)
        row = QHBoxLayout()
        row.addWidget(QLabel("Level"))
        self.level = QSpinBox()
        self.level.setRange(0, 9)
        self.level.setSpecialValueText("deepest")
        row.addWidget(self.level)
        self.show_btn = QPushButton("Show phenotypes on image")
        self.show_btn.clicked.connect(self.show_phenotypes)
        row.addWidget(self.show_btn, 1)
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        coming = QLabel(COMING)
        coming.setWordWrap(True)
        coming.setStyleSheet("color: gray;")
        layout.addWidget(coming)

        for ev in (self.viewer.layers.events.inserted, self.viewer.layers.events.removed):
            ev.connect(self._fill_layers)
        self._used = None  # the Measurements table in use (to notice it being measured again)
        _measure.watch(self.viewer, lambda tables: self._tables_changed())
        self.table_box.currentTextChanged.connect(self._guess_layer)
        self.state.listen(self.refresh)
        self._fill_tables()
        self._fill_layers()
        self.refresh()

    # ---- choosing the objects -------------------------------------------------------------

    def _tables(self):
        return _measure.tables_of(self.viewer)

    def _fill_tables(self):
        keep = self.table_box.currentText() or self.state.source
        self.table_box.blockSignals(True)
        self.table_box.clear()
        names = list(self._tables())
        self.table_box.addItems(names)
        if keep in names:
            self.table_box.setCurrentText(keep)
        elif COMBINED in names:  # one row per object with every layer's measurements
            self.table_box.setCurrentText(COMBINED)
        self.table_box.blockSignals(False)
        self.use_btn.setEnabled(bool(names))
        self._guess_layer()

    def _tables_changed(self):
        """New measurements: offer them, and use them if no objects are chosen yet, or if
        the objects in use were measured again (gates and phenotypes are kept)."""
        self._fill_tables()
        tables = self._tables()
        if self.state.table is None and self.table_box.currentText() in tables:
            self.use()
        elif self.state.source in tables and tables[self.state.source] is not self._used:
            self.use(self.state.source, self.state.layer_name)

    def _fill_layers(self, event=None):
        from napari.layers import Labels

        keep = self.layer_box.currentText() or self.state.layer_name
        self.layer_box.clear()
        names = [l.name for l in self.viewer.layers
                 if isinstance(l, Labels) and not (l.metadata or {}).get("ibif_cytometry")
                 and not (l.metadata or {}).get("ibif_region")]
        self.layer_box.addItems(names)
        if keep in names:
            self.layer_box.setCurrentText(keep)

    def _guess_layer(self, *_):
        """The joined table ("All objects") is on the first layer measured; others by name."""
        name = self.table_box.currentText()
        names = [self.layer_box.itemText(i) for i in range(self.layer_box.count())]
        if name in names:
            self.layer_box.setCurrentText(name)
        elif name and names:
            tables = [t for t in self._tables() if t in names]
            if tables:
                self.layer_box.setCurrentText(tables[0])

    def use(self, source=None, layer=None):
        source = source or self.table_box.currentText()
        layer = layer or self.layer_box.currentText()
        table = self._tables().get(source)
        if table is None:
            self.status.setText("Measure the objects first (Measurements section).")
            return
        self.state.set_table(source, table, layer or None)
        self._used = table
        text = f"Analysing {len(table):,} objects from {source}."
        dropped = getattr(self.state, "dropped_analyses", [])
        if dropped:
            text += (f" The clustering results ({', '.join(dropped)}) were for other objects and were "
                     "cleared; gates drawn on their maps select nothing until they are run again.")
        self.status.setText(text)

    # ---- summary ---------------------------------------------------------------------------

    def refresh(self):
        state = self.state
        self.tree.clear()
        if state.table is None:
            self.summary.setText(state.describe())
            self.show_btn.setEnabled(False)
            return
        self.summary.setText(state.describe()
                             + f"<br>{len(state.order)} gates, {len(state.tree)} phenotypes.")
        counts = state.counts()
        broken = state.broken_phenotypes()
        rows = counts.set_index("phenotype") if counts is not None else None
        items = {}
        for node in state.tree:
            if rows is not None:
                cols = [node.name, f"{int(rows.loc[node.name, 'count']):,}",
                        f"{rows.loc[node.name, '% of parent']:.1f}%"]
            else:
                cols = [node.name, "rule needs: " + ", ".join(sorted(broken.get(node.name, ()))), ""]
            item = QTreeWidgetItem(cols)
            item.setForeground(0, QColor(node.color or "#888888"))
            parent = items.get(node.parent)
            (parent.addChild(item) if parent is not None else self.tree.addTopLevelItem(item))
            items[node.name] = item
        self.tree.expandAll()
        for c in range(3):
            self.tree.resizeColumnToContents(c)
        self.show_btn.setEnabled(bool(len(state.tree)) and not broken)

    def show_phenotypes(self):
        try:
            layer = self.state.show_phenotypes(self.level.value() or None)
        except ValueError as e:
            self.status.setText(str(e))
            return
        self.status.setText(f"Shown in layer {layer.name}.")

    def open_workbench(self):
        from ._cyto_workbench import open_workbench

        return open_workbench(self.viewer, near=self)


__all__ = ["CytometryWidget"]
