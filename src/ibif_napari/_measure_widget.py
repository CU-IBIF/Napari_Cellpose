"""Measurements panel: intensity statistics per nucleus, cell and cytoplasm, as a table."""
from __future__ import annotations

from pathlib import Path

import numpy as np
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from qtpy.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from qtpy.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from . import _measure

COMBINED = "All objects (one row per label)"


class DataFrameModel(QAbstractTableModel):
    def __init__(self, table=None):
        super().__init__()
        self.table = table

    def set_table(self, table):
        self.beginResetModel()
        self.table = table
        self.endResetModel()

    def rowCount(self, parent=QModelIndex()):
        return 0 if self.table is None else len(self.table)

    def columnCount(self, parent=QModelIndex()):
        return 0 if self.table is None else len(self.table.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or self.table is None:
            return None
        value = self.table.iat[index.row(), index.column()]
        if role == Qt.DisplayRole:
            if isinstance(value, (float, np.floating)):
                return "" if np.isnan(value) else f"{value:.4g}"
            return str(value)
        if role == Qt.UserRole:  # for sorting by number, not text
            return float(value) if isinstance(value, (int, float, np.number)) else str(value)
        return None

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole or self.table is None:
            return None
        return str(self.table.columns[section]) if orientation == Qt.Horizontal else str(section + 1)


class MeasureWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self.tables = {}
        self._worker = None
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Tick the objects (e.g. nuclei, cells, cytoplasm) and the channels to measure. Objects "
            "with the same number are shown on one row; click a row to go to the object."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        lists = QHBoxLayout()
        self.objects = self._checklist(lists, "Objects")
        self.channels = self._checklist(lists, "Channels")
        layout.addLayout(lists)
        self.measure_btn = QPushButton("Measure")
        self.measure_btn.clicked.connect(self.measure)
        layout.addWidget(self.measure_btn)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)

        row = QHBoxLayout()
        row.addWidget(QLabel("Show"))
        self.view = QComboBox()
        self.view.currentTextChanged.connect(self._show)
        row.addWidget(self.view, 1)
        layout.addLayout(row)
        self.model = DataFrameModel()
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(self.model)
        self.proxy.setSortRole(Qt.UserRole)
        self.table = QTableView()
        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.clicked.connect(self._go_to)
        layout.addWidget(self.table, 1)

        buttons = QHBoxLayout()
        self.save_btn = QPushButton("Save table…")
        self.save_btn.clicked.connect(self.save_table)
        self.save_all_btn = QPushButton("Save all tables…")
        self.save_all_btn.clicked.connect(self.save_all)
        buttons.addWidget(self.save_btn)
        buttons.addWidget(self.save_all_btn)
        layout.addLayout(buttons)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        for ev in (self.viewer.layers.events.inserted, self.viewer.layers.events.removed,
                   self.viewer.layers.events.reordered):
            ev.connect(self._refresh)
        self._refresh()
        self._enable_saving()

    def _checklist(self, parent_layout, title):
        box = QVBoxLayout()
        box.addWidget(QLabel(title))
        lst = QListWidget()
        lst.setMaximumHeight(130)
        box.addWidget(lst)
        parent_layout.addLayout(box)
        return lst

    # ---- layer lists -------------------------------------------------------------------

    def _refresh(self, event=None):
        for lst, kind in ((self.objects, Labels), (self.channels, Image)):
            checked = set(self._checked(lst))
            lst.clear()
            for layer in self.viewer.layers:
                if isinstance(layer, kind) and not getattr(layer, "rgb", False):
                    if kind is Labels and (layer.metadata or {}).get("ibif_region"):
                        continue
                    item = QListWidgetItem(layer.name)
                    item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                    item.setCheckState(Qt.Checked if layer.name in checked else Qt.Unchecked)
                    lst.addItem(item)

    @staticmethod
    def _checked(lst):
        return [lst.item(i).text() for i in range(lst.count()) if lst.item(i).checkState() == Qt.Checked]

    def check(self, objects=(), channels=()):
        """Tick layers by name (also used by tests and scripts)."""
        for lst, names in ((self.objects, objects), (self.channels, channels)):
            for i in range(lst.count()):
                if lst.item(i).text() in names:
                    lst.item(i).setCheckState(Qt.Checked)

    # ---- measuring ---------------------------------------------------------------------

    def measure(self):
        if self._worker is not None:
            return
        layers = [self.viewer.layers[n] for n in self._checked(self.objects) if n in self.viewer.layers]
        images = [self.viewer.layers[n] for n in self._checked(self.channels) if n in self.viewer.layers]
        if not layers:
            self.status.setText("Tick at least one objects layer.")
            return
        viewer = self.viewer
        try:  # report a pixel-size or extent mismatch before starting
            from ._util import aligned_image, spatial_ndim

            for layer in layers:
                for img in images:
                    aligned_image(viewer, img, layer, (0,) * (layer.ndim - spatial_ndim(layer)))
        except ValueError as e:
            self.status.setText(str(e))
            return

        @thread_worker
        def work():
            tables = {}
            for k, layer in enumerate(layers):
                yield (k, len(layers))
                tables[layer.name] = _measure.measure_layer(viewer, layer, images)
            return tables

        worker = work()
        worker.yielded.connect(self._on_progress)
        worker.returned.connect(self._measured)
        worker.errored.connect(lambda e: self.status.setText(f"Measuring failed: {e}"))
        worker.finished.connect(self._finished)
        self._worker = worker
        self.measure_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Measuring…")
        worker.start()

    def _on_progress(self, value):
        done, total = value
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _finished(self):
        self._worker = None
        self.measure_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _measured(self, tables):
        self.tables = dict(tables)
        if len(tables) > 1:
            self.tables[COMBINED] = _measure.combine(tables)
        _measure.remember(self.viewer, self.tables)
        self.view.blockSignals(True)
        self.view.clear()
        self.view.addItems(list(self.tables))
        self.view.setCurrentIndex(len(self.tables) - 1 if len(tables) > 1 else 0)
        self.view.blockSignals(False)
        self._show(self.view.currentText())
        counts = ", ".join(f"{n}: {len(t)}" for n, t in tables.items())
        self.status.setText(f"Measured {counts} objects.")
        self._enable_saving()

    def set_tables(self, tables):
        """Show tables measured earlier (e.g. from a reopened workspace)."""
        if tables:
            self._measured({k: v for k, v in tables.items() if k != COMBINED})

    def _show(self, name):
        self.model.set_table(self.tables.get(name))
        self.proxy.sort(-1)  # natural order (time point, label) until a header is clicked
        self.table.horizontalHeader().setSortIndicator(-1, Qt.AscendingOrder)
        self.table.resizeColumnsToContents()

    def proxy_row(self, **values):
        """The table row (as shown) whose columns have these values, e.g. T=1, label=2."""
        table = self.model.table
        for r in range(self.proxy.rowCount()):
            row = table.iloc[self.proxy.mapToSource(self.proxy.index(r, 0)).row()]
            if all(row[k] == v for k, v in values.items()):
                return self.proxy.index(r, 0)
        return None

    def _enable_saving(self):
        self.save_btn.setEnabled(bool(self.tables))
        self.save_all_btn.setEnabled(bool(self.tables))

    # ---- going to an object ----------------------------------------------------------------

    def _go_to(self, index):
        table = self.model.table
        if table is None:
            return
        row = table.iloc[self.proxy.mapToSource(index).row()]
        label = int(row["label"])
        centroid = {}
        for axis in "ZYX":
            cols = [c for c in table.columns if c.endswith(f"centroid {axis} (µm)") or c.endswith(f"centroid {axis} (px)")]
            values = [row[c] for c in cols if not np.isnan(row[c])]
            if values:
                centroid[axis] = float(values[0])
        dims = self.viewer.dims
        layers = [self.viewer.layers[n] for n in self.tables if n in self.viewer.layers]
        for layer in layers:
            layer.selected_label = label
        if layers:
            layer = layers[0]
            world = [centroid[a] for a in "ZYX" if a in centroid]
            in_2d = dims.ndisplay == 2 and "Z" in centroid and dims.ndim >= 3
            # Centre the camera first: in recent napari moving the camera also resets the
            # sliders, so the time point / position and Z are set after it.
            camera = getattr(getattr(self.viewer, "scene", None), "camera", None) or self.viewer.camera
            camera.center = tuple((world[1:] if in_2d else world)[-dims.ndisplay:])
            offset = dims.ndim - layer.ndim
            axes = (layer.metadata or {}).get("ibif_axes") or ""
            for k, name in enumerate(axes):
                if name in ("T", "P") and name in row.index:
                    dims.set_point(offset + k, float(layer.translate[k] + row[name] * layer.scale[k]))
            if in_2d:
                dims.set_point(dims.ndim - 3, centroid["Z"])
        self.status.setText(f"Object {label}.")

    # ---- saving ------------------------------------------------------------------------------

    def save_table(self, path=None):
        name = self.view.currentText()
        if name not in self.tables:
            return None
        if not path:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save measurements", f"{name.split(' (')[0]} measurements.csv",
                "CSV (*.csv);;Excel (*.xlsx)",
            )
            if not path:
                return None
        path = _measure.save_table(self.tables[name], path)
        self.status.setText(f"Saved {path}")
        return path

    def save_all(self, folder=None):
        if not self.tables:
            return []
        if not folder:
            folder = QFileDialog.getExistingDirectory(self, "Folder for the measurement tables")
            if not folder:
                return []
        paths = []
        for name, table in self.tables.items():
            stem = "all objects" if name == COMBINED else name
            safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in stem).strip()
            paths.append(_measure.save_table(table, Path(folder) / f"{safe} measurements.csv"))
        self.status.setText(f"Saved {len(paths)} tables in {folder}")
        return paths
