"""Processing section: an ordered pipeline of corrections (registration, and later
illumination correction and deconvolution) run on the image before segmentation.

The library lists the routines; add one to the pipeline, order the pipeline (drag, or
Up / Down), and select a step to see its settings below. Steps that are coming soon are
listed with their planned settings but cannot be added yet. Running the pipeline makes
new layers, '<channel> [processed]'; the originals are not changed.
"""
from __future__ import annotations

import threading
from pathlib import Path

from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from qtpy.QtCore import QObject, Qt, Signal
from qtpy.QtGui import QBrush, QColor, QFont
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import _processing
from ._processing import session
from ._processing.base import Cancelled

THIS_COMPUTER = "This computer"
COMING_SOON_TAG = "coming soon"


class _Progress(QObject):
    changed = Signal(float, str)  # emitted from the worker thread, shown in the GUI thread


class ProcessingWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._stop = None
        self._editors = {}  # parameter name -> editor widget of the step shown
        layout = QVBoxLayout(self)
        intro = QLabel("Correct the image before segmenting it: add steps from the library to the "
                       "pipeline, in the order they should run. New layers are made; the originals "
                       "are kept.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        form = self._form = QFormLayout()
        layout.addLayout(form)
        self.channels = QListWidget()
        self.channels.setMaximumHeight(110)
        self.channels.setToolTip("The channels of one image to process together (same size)")
        self.channels.itemChanged.connect(self._update_reference)
        form.addRow("Channels", self.channels)
        self.reference = QComboBox()
        self.reference.setToolTip("Channel that steps such as registration measure on; what they find is "
                                  "applied to every channel")
        form.addRow("Reference channel", self.reference)
        self.scope = QComboBox()
        self.scope.addItems([session.SCOPE_ALL, session.SCOPE_POSITION])
        form.addRow("Process", self.scope)

        lists = QHBoxLayout()
        layout.addLayout(lists)
        lib_box = QVBoxLayout()
        lib_box.addWidget(_bold("Library"))
        self.library = QListWidget()
        self.library.setToolTip("Routines that can be added; those coming soon are greyed")
        self.library.itemSelectionChanged.connect(self._library_selected)
        self.library.itemDoubleClicked.connect(lambda *_: self.add_selected())
        lib_box.addWidget(self.library)
        self.add_btn = QPushButton("Add to pipeline →")
        self.add_btn.clicked.connect(self.add_selected)
        lib_box.addWidget(self.add_btn)
        lists.addLayout(lib_box)

        pipe_box = QVBoxLayout()
        pipe_box.addWidget(_bold("Pipeline (runs top to bottom)"))
        self.pipeline_list = QListWidget()
        self.pipeline_list.setDragDropMode(QAbstractItemView.InternalMove)
        self.pipeline_list.setToolTip("Drag to reorder; untick a step to skip it")
        self.pipeline_list.itemSelectionChanged.connect(self._pipeline_selected)
        self.pipeline_list.itemChanged.connect(self._enabled_changed)
        self.pipeline_list.model().rowsMoved.connect(self._rows_moved)
        pipe_box.addWidget(self.pipeline_list)
        row = QHBoxLayout()
        self.up_btn, self.down_btn, self.remove_btn = QPushButton("Up"), QPushButton("Down"), QPushButton("Remove")
        self.up_btn.clicked.connect(lambda: self.move_selected(-1))
        self.down_btn.clicked.connect(lambda: self.move_selected(1))
        self.remove_btn.clicked.connect(self.remove_selected)
        for b in (self.up_btn, self.down_btn, self.remove_btn):
            row.addWidget(b)
        pipe_box.addLayout(row)
        row = QHBoxLayout()
        self.import_btn, self.export_btn = QPushButton("Import…"), QPushButton("Export…")
        self.import_btn.setToolTip("Open a pipeline saved as JSON")
        self.export_btn.setToolTip("Save this pipeline as JSON, to reuse it or run it elsewhere")
        self.import_btn.clicked.connect(self._import_dialog)
        self.export_btn.clicked.connect(self._export_dialog)
        row.addWidget(self.import_btn)
        row.addWidget(self.export_btn)
        pipe_box.addLayout(row)
        lists.addLayout(pipe_box)

        self.settings_box = QGroupBox("Step settings")
        box = QVBoxLayout(self.settings_box)
        self.step_title = _bold("")
        box.addWidget(self.step_title)
        self.step_description = QLabel()
        self.step_description.setWordWrap(True)
        box.addWidget(self.step_description)
        self.settings_form = QFormLayout()
        box.addLayout(self.settings_form)
        self.step_note = QLabel()
        self.step_note.setWordWrap(True)
        box.addWidget(self.step_note)
        layout.addWidget(self.settings_box)

        run_form = QFormLayout()
        layout.addLayout(run_form)
        self.run_on = QComboBox()
        self.run_on.addItem(THIS_COMPUTER)
        self.run_on.setToolTip("Where the pipeline runs")
        run_form.addRow("Run on", self.run_on)
        self.run_btn = QPushButton("Run pipeline")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self.results_box = QGroupBox("Registration results")
        results = QFormLayout(self.results_box)
        self.registered = QComboBox()
        self.registered.setToolTip("A layer made by a pipeline with Stack registration")
        results.addRow("Registered layer", self.registered)
        self.export_csv_btn = QPushButton("Export transforms (CSV)…")
        self.export_csv_btn.setToolTip("The shift, rotation and correlation of every frame, as a table")
        self.export_csv_btn.clicked.connect(self._export_transforms_dialog)
        results.addRow("", self.export_csv_btn)
        row = QHBoxLayout()
        self.apply_target = QComboBox()
        self.apply_target.setToolTip("Another layer of the same size (e.g. labels segmented before "
                                     "registering) to move the same way")
        self.apply_btn = QPushButton("Apply")
        self.apply_btn.clicked.connect(lambda: self.apply_registration())
        row.addWidget(self.apply_target, 1)
        row.addWidget(self.apply_btn)
        results.addRow("Apply to", row)
        layout.addWidget(self.results_box)
        layout.addStretch(1)

        self._fill_library()
        self._show_pipeline()
        self._show_step(None)
        for ev in (self.viewer.layers.events.inserted, self.viewer.layers.events.removed,
                   self.viewer.layers.events.reordered):
            ev.connect(self._refresh_layers)
        self._refresh_layers()
        session.listen(self.viewer, self._show_pipeline)  # a pipeline from a reopened workspace

    # ---- the pipeline ------------------------------------------------------------------

    @property
    def pipeline(self):
        return session.pipeline_of(self.viewer)

    def _fill_library(self):
        self.library.clear()
        for spec in _processing.library():
            text = spec.title if spec.available else f"{spec.title} — {COMING_SOON_TAG}"
            item = QListWidgetItem(text)
            item.setData(Qt.UserRole, spec.key)
            item.setToolTip(spec.description)
            if not spec.available:
                item.setForeground(QBrush(QColor(128, 128, 128)))
                font = item.font()
                font.setItalic(True)
                item.setFont(font)
            self.library.addItem(item)

    def showEvent(self, event):  # steps registered since (e.g. by a plugin) appear
        super().showEvent(event)
        if self.library.count() != len(_processing.STEPS):
            self._fill_library()

    def _show_pipeline(self):
        selected = self.selected_index()
        self.pipeline_list.blockSignals(True)
        self.pipeline_list.clear()
        for step in self.pipeline:
            item = QListWidgetItem(step.spec.title)
            item.setData(Qt.UserRole, step)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable | Qt.ItemIsDragEnabled)
            item.setFlags(item.flags() & ~Qt.ItemIsDropEnabled)
            item.setCheckState(Qt.Checked if step.enabled else Qt.Unchecked)
            self.pipeline_list.addItem(item)
        self.pipeline_list.blockSignals(False)
        if selected is not None and selected < self.pipeline_list.count():
            self.pipeline_list.setCurrentRow(selected)
        self._update_buttons()
        self._update_reference()

    def selected_index(self):
        rows = [i.row() for i in self.pipeline_list.selectedIndexes()]
        return rows[0] if rows else None

    def _selected_key(self):
        items = self.library.selectedItems()
        return items[0].data(Qt.UserRole) if items else None

    def add_selected(self):
        key = self._selected_key()
        if key is None:
            return None
        return self.add_step(key)

    def add_step(self, key):
        """Add a step from the library (also used by tests and scripts)."""
        spec = _processing.get(key)
        if not spec.available:
            self.status.setText(f"{spec.title} is coming soon; it cannot be added yet.")
            return None
        step = self.pipeline.add(key)
        self._show_pipeline()
        self.pipeline_list.setCurrentRow(len(self.pipeline) - 1)
        self.status.setText(f"Added {spec.title}.")
        return step

    def remove_selected(self):
        i = self.selected_index()
        if i is None:
            return
        self.pipeline.remove(i)
        self._show_pipeline()
        if len(self.pipeline):
            self.pipeline_list.setCurrentRow(min(i, len(self.pipeline) - 1))
        else:
            self._show_step(None)

    def move_selected(self, delta):
        i = self.selected_index()
        if i is None:
            return
        self.pipeline.move(i, i + delta)
        self._show_pipeline()
        self.pipeline_list.setCurrentRow(max(0, min(len(self.pipeline) - 1, i + delta)))

    def _rows_moved(self, *_):
        """Dragged in the list: take the list's order."""
        steps = [self.pipeline_list.item(i).data(Qt.UserRole) for i in range(self.pipeline_list.count())]
        self.pipeline.steps = steps
        self._update_buttons()

    def _enabled_changed(self, item):
        step = item.data(Qt.UserRole)
        if step is not None:
            step.enabled = item.checkState() == Qt.Checked
        self._update_buttons()

    def _update_buttons(self):
        i = self.selected_index()
        n = len(self.pipeline)
        self.up_btn.setEnabled(i is not None and i > 0)
        self.down_btn.setEnabled(i is not None and i < n - 1)
        self.remove_btn.setEnabled(i is not None)
        self.export_btn.setEnabled(n > 0)
        key = self._selected_key()
        self.add_btn.setEnabled(key is not None and _processing.get(key).available)
        if self._worker is None:
            self.run_btn.setEnabled(any(s.enabled for s in self.pipeline))

    # ---- step settings ------------------------------------------------------------------

    def _library_selected(self):
        key = self._selected_key()
        if key is not None:
            self.pipeline_list.blockSignals(True)
            self.pipeline_list.clearSelection()
            self.pipeline_list.blockSignals(False)
            self._show_step(_processing.get(key))
        self._update_buttons()

    def _pipeline_selected(self):
        i = self.selected_index()
        if i is not None:
            self.library.blockSignals(True)
            self.library.clearSelection()
            self.library.blockSignals(False)
            self._show_step(self.pipeline.steps[i].spec, i)
        self._update_buttons()

    def _show_step(self, spec, index=None):
        """Show a step's settings: editable for a pipeline step (`index`), read-only for a
        library entry, and "Feature coming soon" for a step not built yet."""
        while self.settings_form.rowCount():
            self.settings_form.removeRow(0)
        self._editors = {}
        if spec is None:
            self.step_title.setText("")
            self.step_description.setText("Select a step in the library or the pipeline to see its settings.")
            self.step_note.setText("")
            return
        self.step_title.setText(spec.title)
        self.step_description.setText(spec.description)
        values = self.pipeline.steps[index].params if index is not None else spec.defaults()
        for p in spec.params:
            editor = _editor(p, values[p.name])
            editor.setEnabled(index is not None)
            if index is not None:
                _connect(editor, p, lambda value, name=p.name, i=index: self._set_param(i, name, value))
            self.settings_form.addRow(p.label + (f" ({p.unit})" if p.unit and p.kind == "int" else ""), editor)
            self._editors[p.name] = editor
        if not spec.available:
            self.step_note.setText("<b>Feature coming soon.</b> These are its planned settings; it cannot "
                                   "be added to the pipeline yet.")
        elif index is None:
            self.step_note.setText("Add it to the pipeline to change its settings.")
        else:
            self.step_note.setText("")

    def _set_param(self, index, name, value):
        try:
            self.pipeline.set_params(index, **{name: value})
        except ValueError as e:
            self.status.setText(str(e))

    def editor(self, name):
        """The editor of a setting of the step shown (for tests and scripts)."""
        return self._editors[name]

    # ---- channels ------------------------------------------------------------------------

    def _refresh_layers(self, event=None):
        from ._processing.registration import records_of

        registered = [layer.name for layer in self.viewer.layers
                      if isinstance(layer, Image) and records_of(layer.metadata)[0]]
        targets = [layer.name for layer in self.viewer.layers if isinstance(layer, (Image, Labels))]
        for combo, names in ((self.registered, registered), (self.apply_target, targets)):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        self.results_box.setVisible(bool(registered))
        checked = set(self.checked_channels(names=True))
        self.channels.blockSignals(True)
        self.channels.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and not layer.rgb:
                item = QListWidgetItem(layer.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if layer.name in checked else Qt.Unchecked)
                self.channels.addItem(item)
        self.channels.blockSignals(False)
        self._update_reference()

    def checked_channels(self, names=False):
        out = [self.channels.item(i).text() for i in range(self.channels.count())
               if self.channels.item(i).checkState() == Qt.Checked]
        return out if names else [self.viewer.layers[n] for n in out if n in self.viewer.layers]

    def check_channels(self, names):
        """Tick channels by name (also used by tests and scripts)."""
        for i in range(self.channels.count()):
            item = self.channels.item(i)
            item.setCheckState(Qt.Checked if item.text() in names else Qt.Unchecked)

    def _update_reference(self, *_):
        current = self.reference.currentText()
        names = self.checked_channels(names=True)
        self.reference.blockSignals(True)
        self.reference.clear()
        self.reference.addItems(names)
        if current in names:
            self.reference.setCurrentText(current)
        self.reference.blockSignals(False)
        needs = any(s.spec.needs_reference for s in self.pipeline)
        self.reference.setVisible(needs)
        label = self._form.labelForField(self.reference)
        if label is not None:
            label.setVisible(needs)

    # ---- import / export -----------------------------------------------------------------

    def export_pipeline(self, path):
        path = Path(path)
        path.write_text(self.pipeline.to_json())
        self.status.setText(f"Pipeline saved to {path.name}.")
        return path

    def import_pipeline(self, path):
        try:
            pipe = _processing.Pipeline.from_json(Path(path).read_text())
        except (OSError, ValueError) as e:
            self.status.setText(f"Could not import {Path(path).name}: {e}")
            return None
        session.set_pipeline(self.viewer, pipe, notify=False)
        self._show_pipeline()
        self._show_step(None)
        self.status.setText(f"Imported {len(pipe)} step(s) from {Path(path).name}.")
        return pipe

    def _export_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save pipeline", "pipeline.json", "Pipeline (*.json)")
        if path:
            self.export_pipeline(path)

    def _import_dialog(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open pipeline", "", "Pipeline (*.json)")
        if path:
            self.import_pipeline(path)

    # ---- registration results ------------------------------------------------------------

    def _registration(self, name=None):
        from ._processing.registration import records_of

        name = name or self.registered.currentText()
        if not name or name not in self.viewer.layers:
            raise ValueError("choose a registered layer")
        layer = self.viewer.layers[name]
        records, along = records_of(layer.metadata)
        if not records:
            raise ValueError(f"{name} was not made by a registration")
        return layer, records, along

    def export_transforms(self, path, name=None):
        """Write the registration of a layer (one row per frame) as CSV; returns the path."""
        import pandas as pd

        layer, records, _ = self._registration(name)
        rows = []
        for r in records:
            row = {k: v for k, v in r.items() if k != "matrix"}
            for i, line in enumerate(r["matrix"]):
                for j, v in enumerate(line):
                    row[f"m{i}{j}"] = v  # output -> input, pixels
            rows.append(row)
        pd.DataFrame(rows).to_csv(path, index=False)
        self.status.setText(f"Transforms of {layer.name} saved to {Path(path).name}.")
        return Path(path)

    def apply_registration(self, target=None, name=None):
        """Move another layer (same size) frame by frame as the registered layer was moved;
        adds '<target> [registered]' and returns it, or None (with the reason shown)."""
        from ._processing.registration import apply_records
        from ._util import layer_axes

        try:
            layer, records, along = self._registration(name)
            target = target or self.apply_target.currentText()
            if target not in self.viewer.layers:
                raise ValueError("choose a layer to apply the registration to")
            other = self.viewer.layers[target]
            if tuple(other.data.shape) != tuple(layer.data.shape):
                raise ValueError(f"{other.name} {tuple(other.data.shape)} is not the size of "
                                 f"{layer.name} {tuple(layer.data.shape)}")
            is_labels = isinstance(other, Labels)
            moved = apply_records(other.data, layer_axes(layer), records, along, labels=is_labels)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        meta = {k: v for k, v in dict(other.metadata or {}).items() if k not in session.SOURCE_KEYS}
        meta["ibif_registered_like"] = layer.name
        kwargs = dict(name=f"{other.name} [registered]", scale=other.scale, translate=other.translate,
                      metadata=meta)
        added = self.viewer.add_labels(moved, **kwargs) if is_labels else self.viewer.add_image(moved, **kwargs)
        self.status.setText(f"Made {added.name}, moved as {layer.name} was (rigid part of the registration).")
        return added

    def _export_transforms_dialog(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save transforms", "transforms.csv", "CSV (*.csv)")
        if path:
            try:
                self.export_transforms(path)
            except ValueError as e:
                self.status.setText(str(e))

    # ---- running -------------------------------------------------------------------------

    def run(self):
        if self._worker is not None:  # Cancel
            self._stop.set()
            self.status.setText("Cancelling…")
            return
        layers = self.checked_channels()
        if not layers:
            self.status.setText("Tick the channel(s) to process.")
            return
        if not any(s.enabled for s in self.pipeline):
            self.status.setText("Add a step to the pipeline first.")
            return
        try:
            stack, index = session.stack_from_layers(
                self.viewer, layers, self.reference.currentText() or None, self.scope.currentText())
        except ValueError as e:
            self.status.setText(str(e))
            return
        pipeline = self.pipeline.copy()
        stop = threading.Event()
        signals = _Progress()
        signals.changed.connect(self._on_progress)

        def progress(fraction, message):
            if stop.is_set():
                raise Cancelled()
            signals.changed.emit(float(fraction), message)

        @thread_worker
        def work():
            try:
                return pipeline.run(stack, progress)
            except Cancelled:
                return None

        worker = work()
        worker.returned.connect(lambda result: self._done(result, layers, index))
        worker.errored.connect(lambda e: self.status.setText(f"Processing failed: {e}"))
        worker.finished.connect(self._finished)
        self._worker, self._stop, self._signals = worker, stop, signals
        self.run_btn.setText("Cancel")
        self.progress.setValue(0)
        self.progress.setVisible(True)
        self.status.setText("Processing…")
        worker.start()

    def _on_progress(self, fraction, message):
        if self._stop is not None and self._stop.is_set():
            return
        self.progress.setValue(int(1000 * fraction))
        if message:
            self.status.setText(message)

    def _done(self, result, layers, index):
        if result is None or (self._stop is not None and self._stop.is_set()):
            self.status.setText("Cancelled; no layers were added.")
            return
        added = session.add_result_layers(self.viewer, layers, result, index)
        summaries = [f"{s['title']}: {s['summary']}" if s["summary"] else s["title"] for s in result.steps]
        self.status.setText(f"Made {', '.join(a.name for a in added)}. " + "; ".join(summaries))

    def _finished(self):
        self._worker = self._stop = None
        self.run_btn.setText("Run pipeline")
        self.progress.setVisible(False)
        self._update_buttons()


def _bold(text):
    label = QLabel(text)
    font = QFont(label.font())
    font.setBold(True)
    label.setFont(font)
    return label


def _editor(p, value):
    if p.kind == "bool":
        w = QCheckBox()
        w.setChecked(bool(value))
    elif p.kind == "choice":
        w = QComboBox()
        w.addItems([str(o) for o in p.options])
        w.setCurrentText(str(value))
    elif p.kind == "int":
        w = QSpinBox()
        w.setRange(int(p.minimum if p.minimum is not None else -10**9),
                   int(p.maximum if p.maximum is not None else 10**9))
        w.setSingleStep(int(p.step or 1))
        w.setValue(int(value))
    else:
        w = QDoubleSpinBox()
        w.setDecimals(p.decimals)
        w.setRange(p.minimum if p.minimum is not None else -1e9, p.maximum if p.maximum is not None else 1e9)
        w.setSingleStep(p.step or 0.1)
        w.setValue(float(value))
        if p.unit:
            w.setSuffix(f" {p.unit}")
    w.setToolTip(p.tip)
    return w


def _connect(w, p, setter):
    if p.kind == "bool":
        w.toggled.connect(setter)
    elif p.kind == "choice":
        w.currentTextChanged.connect(setter)
    else:
        w.valueChanged.connect(setter)


__all__ = ["ProcessingWidget"]
