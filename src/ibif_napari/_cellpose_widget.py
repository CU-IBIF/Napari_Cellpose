"""Cellpose panel: 3D (or 2D) segmentation of the current or every time point/position."""
from __future__ import annotations

import os
import time

import numpy as np
from napari.layers import Image
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import _cellpose
from ._util import current_volume, layer_axes

MODE_3D = "3D (true 3D flows)"
MODE_STITCH = "2D per plane, stitched"
SCOPE_CURRENT = "Current time point / position"
SCOPE_ALL = "Every time point / position"
NO_NUCLEI = "(none)"


class CellposeWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._preload_worker = None

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.image = QComboBox()
        self.nuclei = QComboBox()
        form.addRow("Image", self.image)
        form.addRow("Nuclear channel", self.nuclei)

        model_row = QHBoxLayout()
        self.model = QComboBox()
        self.model.setEditable(True)
        self.model.addItems(_cellpose.builtin_models())
        browse = QPushButton("…")
        browse.setToolTip("Use a custom (trained) Cellpose model file")
        browse.clicked.connect(self._browse_model)
        model_row.addWidget(self.model, 1)
        model_row.addWidget(browse)
        form.addRow("Model", model_row)

        self.mode = QComboBox()
        self.mode.addItems([MODE_3D, MODE_STITCH])
        form.addRow("Mode", self.mode)

        self.diameter = _float_spin(0, 1000, 0, 1, "Cell diameter in pixels (0 = model default / auto)")
        self.diameter.setSpecialValueText("auto")
        form.addRow("Diameter (px)", self.diameter)

        self.anisotropy = _float_spin(0.1, 100, 1, 0.1, "Z step / XY pixel size; set from the image")
        form.addRow("Anisotropy", self.anisotropy)

        self.stitch = _float_spin(0, 1, 0.3, 0.05, "Overlap (IoU) needed to link masks between planes")
        form.addRow("Stitch threshold", self.stitch)

        self.flow = _float_spin(0, 3, 0.4, 0.1, "Higher = more (and less accurate) masks")
        form.addRow("Flow threshold", self.flow)

        self.cellprob = _float_spin(-6, 6, 0, 0.5, "Lower = larger / more masks")
        form.addRow("Cell probability", self.cellprob)

        self.min_size = QSpinBox()
        self.min_size.setRange(0, 1_000_000)
        self.min_size.setValue(15)
        form.addRow("Min size (px)", self.min_size)

        self.scope = QComboBox()
        self.scope.addItems([SCOPE_CURRENT, SCOPE_ALL])
        form.addRow("Run on", self.scope)

        self.gpu = QCheckBox("Use GPU")
        form.addRow("", self.gpu)

        self.run_btn = QPushButton("Run Cellpose")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.mode.currentTextChanged.connect(self._update_enabled)
        self.image.currentTextChanged.connect(self._on_image_changed)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.reordered.connect(self._refresh_layers)
        self._refresh_layers()
        self._update_enabled()

        if _cellpose.cellpose_major() is None:
            self.status.setText("Cellpose is not installed: pip install cellpose")
            self.run_btn.setEnabled(False)
            return
        self.gpu.setChecked(_cellpose.gpu_available())
        preload = os.environ.get(_cellpose.PRELOAD_ENV, "")
        if preload != "0":
            if preload:
                self.model.setEditText(preload)
            self.preload_model()

    # ---- model ------------------------------------------------------------------------

    def preload_model(self, name=None):
        """Load (and if needed download) the model in the background so the first run is quick."""
        name = name or self.model.currentText()
        if not name:
            return
        gpu = self.gpu.isChecked()
        self.status.setText(f"Loading Cellpose model '{name}'…")

        @thread_worker
        def load():
            yield from _cellpose.load_model(name, gpu)

        worker = load()
        worker.yielded.connect(self._show_preload_status)
        worker.returned.connect(
            lambda _: self.status.setText(f"Cellpose model '{name}' ready ({'GPU' if gpu else 'CPU'}).")
        )
        worker.errored.connect(lambda e: self.status.setText(f"Could not load model '{name}': {e}"))
        self._preload_worker = worker
        worker.start()

    def _show_preload_status(self, message):
        if self._worker is None:  # a run reports its own progress
            self.status.setText(message)

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cellpose model")
        if path:
            self.model.setEditText(path)

    # ---- layer choices ----------------------------------------------------------------

    def _image_layers(self):
        return [l for l in self.viewer.layers if isinstance(l, Image) and not l.rgb]

    def _refresh_layers(self, event=None):
        names = [l.name for l in self._image_layers()]
        for combo, extra in ((self.image, []), (self.nuclei, [NO_NUCLEI])):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(extra + names)
            if current in extra + names:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        self._on_image_changed()

    def _on_image_changed(self, *_):
        layer = self._layer(self.image.currentText())
        if layer is None:
            return
        axes = layer_axes(layer)
        if "Z" in axes and "Y" in axes:
            z = layer.scale[axes.index("Z")]
            y = layer.scale[axes.index("Y")]
            if y > 0:
                self.anisotropy.setValue(z / y)
        self._update_enabled()

    def _layer(self, name):
        return self.viewer.layers[name] if name and name in self.viewer.layers else None

    def _update_enabled(self, *_):
        layer = self._layer(self.image.currentText())
        is_3d = layer is not None and "Z" in layer_axes(layer)
        self.mode.setEnabled(is_3d)
        self.anisotropy.setEnabled(is_3d and self.mode.currentText() == MODE_3D)
        self.stitch.setEnabled(is_3d and self.mode.currentText() == MODE_STITCH)

    # ---- running ----------------------------------------------------------------------

    def _params(self):
        return dict(
            model=self.model.currentText(),
            gpu=self.gpu.isChecked(),
            do_3D=self.mode.currentText() == MODE_3D,
            anisotropy=self.anisotropy.value(),
            diameter=self.diameter.value() or None,
            flow_threshold=self.flow.value(),
            cellprob_threshold=self.cellprob.value(),
            stitch_threshold=self.stitch.value(),
            min_size=self.min_size.value(),
        )

    def run(self):
        if self._worker is not None:
            self._worker.quit()
            return
        layer = self._layer(self.image.currentText())
        if layer is None:
            self.status.setText("Choose an image layer first.")
            return
        nuc = self._layer(self.nuclei.currentText())
        if nuc is not None and nuc.data.shape != layer.data.shape:
            self.status.setText("The nuclear channel must have the same shape as the image.")
            return
        params = self._params()
        params["model"] = params["model"] or _cellpose.default_model()
        axes = layer_axes(layer)
        keep = "ZYX" if "Z" in axes else "YX"
        if self.scope.currentText() == SCOPE_CURRENT:
            worker = self._run_current(layer, nuc, keep, params)
        else:
            worker = self._run_all(layer, nuc, axes, keep, params)
        worker.yielded.connect(self._on_progress)
        worker.errored.connect(lambda e: self.status.setText(f"Cellpose failed: {e}"))
        worker.aborted.connect(lambda: self.status.setText("Cancelled."))
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self.run_btn.setText("Cancel")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Running Cellpose…")
        worker.start()

    def _run_current(self, layer, nuc, keep, params):
        vol, kept, scale, index = current_volume(self.viewer, layer, keep=keep)
        nuc_vol = np.asarray(nuc.data[index]) if nuc is not None else None
        translate = [t for t, i in zip(layer.translate, index) if isinstance(i, slice)]
        tag = _index_tag(layer_axes(layer), index)

        @thread_worker
        def work():
            yield from _cellpose.load_model(params["model"], params["gpu"])
            yield _segmenting_message(vol.shape, params)
            return _cellpose.segment(vol, nuc_vol, **params)

        worker = work()
        worker.returned.connect(self._add_labels_cb(layer, scale, translate, kept, tag))
        return worker

    def _run_all(self, layer, nuc, axes, keep, params):
        outer = [i for i, a in enumerate(axes) if a not in keep]
        outer_shape = [layer.data.shape[i] for i in outer]
        @thread_worker
        def work():
            yield from _cellpose.load_model(params["model"], params["gpu"])
            labels = np.zeros(layer.data.shape, dtype=np.uint32)
            positions = list(np.ndindex(*outer_shape))
            for n, pos in enumerate(positions):
                yield (n, len(positions))
                index = [slice(None)] * layer.ndim
                for i, p in zip(outer, pos):
                    index[i] = p
                index = tuple(index)
                vol = np.asarray(layer.data[index])
                nuc_vol = np.asarray(nuc.data[index]) if nuc is not None else None
                labels[index] = _cellpose.segment(vol, nuc_vol, **params)
            return labels

        worker = work()
        worker.returned.connect(
            self._add_labels_cb(layer, list(layer.scale), list(layer.translate), axes, "all")
        )
        return worker

    def _add_labels_cb(self, layer, scale, translate, axes, tag):
        def add(labels):
            meta = {**(layer.metadata or {}), "ibif_axes": axes}
            name = f"{layer.name} cellpose {tag}".strip()
            self.viewer.add_labels(labels, name=name, scale=scale, translate=translate, metadata=meta)
            self.status.setText(f"{name}: {int(labels.max())} objects (max label).")

        return add

    def _on_progress(self, value):
        if isinstance(value, str):
            self.status.setText(value)
            return
        done, total = value
        self.progress.setRange(0, total)
        self.progress.setValue(done)
        self.status.setText(f"Segmenting {done + 1} of {total}…")

    def _on_finished(self):
        self._worker = None
        self.run_btn.setText("Run Cellpose")
        self.progress.setVisible(False)


def _segmenting_message(shape, params):
    device = "GPU" if params["gpu"] else "CPU"
    mode = "3D" if params["do_3D"] and len(shape) == 3 else "2D"
    slow = " This can take several minutes on CPU." if device == "CPU" else ""
    started = time.strftime("%H:%M")
    return f"Segmenting {mode} volume {shape} on {device} (started {started}).{slow}"


def _index_tag(axes, index):
    parts = [f"{a}{i}" for a, i in zip(axes, index) if not isinstance(i, slice) and a in "TP"]
    return " ".join(parts)


def _float_spin(lo, hi, value, step, tip):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(2)
    s.setSingleStep(step)
    s.setValue(value)
    s.setToolTip(tip)
    return s
