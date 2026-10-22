"""Cellpose panel: 3D (or 2D) segmentation of the current or every time point/position."""
from __future__ import annotations

import os
import time

import numpy as np
from napari.layers import Image, Labels, Shapes
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

import logging

from . import _cellpose
from ._process import Cancelled, get_process
from ._region import keep_inside, painted_mask_yx, region_crop
from ._util import current_volume, layer_axes

MODE_3D = "3D (true 3D flows)"
MODE_STITCH = "2D per plane, stitched"
SCOPE_CURRENT = "Current time point / position"
SCOPE_ALL = "Every time point / position"
NO_NUCLEI = "(none)"
WHOLE_IMAGE = "(whole image)"


class CellposeWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._preload_worker = None
        self._cancelling = False
        self.process = get_process()  # Cellpose runs here, so Cancel can stop it at once

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

        region_row = QHBoxLayout()
        self.region = QComboBox()
        self.region.setToolTip(
            "Segment only this area: a Labels layer (painted) or a Shapes layer (rectangle)"
        )
        draw = QPushButton("Draw…")
        draw.setToolTip("Add a layer to paint the area to segment (applies to all Z planes)")
        draw.clicked.connect(self.add_region_layer)
        region_row.addWidget(self.region, 1)
        region_row.addWidget(draw)
        form.addRow("Region", region_row)

        z_row = QHBoxLayout()
        self.z_first = QSpinBox()
        self.z_last = QSpinBox()
        z_row.addWidget(self.z_first)
        z_row.addWidget(QLabel("to"))
        z_row.addWidget(self.z_last)
        form.addRow("Z planes", z_row)

        self.keep_inside = QCheckBox("Drop cells mostly outside the painted area")
        self.keep_inside.setChecked(True)
        form.addRow("", self.keep_inside)

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
        self.region.currentTextChanged.connect(self._update_enabled)
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
            yield from _cellpose.ensure_weights(name)
            try:
                yield from self.process.load(name, gpu)
            except Cancelled:
                return

        worker = load()
        worker.yielded.connect(self._show_preload_status)
        worker.returned.connect(
            lambda _: self.status.setText(f"Cellpose model '{name}' ready ({'GPU' if gpu else 'CPU'}).")
        )
        worker.errored.connect(lambda e: self._show_error(f"Could not load model '{name}'", e))
        self._preload_worker = worker
        worker.start()

    def _show_preload_status(self, message):
        if message and self._worker is None:  # a run reports its own progress
            self.status.setText(message)

    def _show_error(self, what, error):
        # Errors from the Cellpose process carry its traceback after a blank line.
        summary = str(error).split("\n\n")[0]
        logging.getLogger("ibif_napari").error("%s: %s", what, error)
        self.status.setText(f"{what}: {summary}")

    def _browse_model(self):
        path, _ = QFileDialog.getOpenFileName(self, "Cellpose model")
        if path:
            self.model.setEditText(path)

    # ---- layer choices ----------------------------------------------------------------

    def _image_layers(self):
        return [l for l in self.viewer.layers if isinstance(l, Image) and not l.rgb]

    def _region_layers(self):
        return [
            l
            for l in self.viewer.layers
            if isinstance(l, (Labels, Shapes)) and not (l.metadata or {}).get("ibif_cellpose")
        ]

    def _refresh_layers(self, event=None):
        names = [l.name for l in self._image_layers()]
        regions = [l.name for l in self._region_layers()]
        for combo, extra, names in (
            (self.image, [], names),
            (self.nuclei, [NO_NUCLEI], names),
            (self.region, [WHOLE_IMAGE], regions),
        ):
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
        nz = layer.data.shape[axes.index("Z")] if "Z" in axes else 1
        if self.z_last.maximum() != nz - 1 or self.z_last.value() > nz - 1:
            for spin in (self.z_first, self.z_last):
                spin.setRange(0, nz - 1)
            self.z_first.setValue(0)
            self.z_last.setValue(nz - 1)
        self._update_enabled()

    def _layer(self, name):
        return self.viewer.layers[name] if name and name in self.viewer.layers else None

    def _update_enabled(self, *_):
        layer = self._layer(self.image.currentText())
        is_3d = layer is not None and "Z" in layer_axes(layer)
        self.mode.setEnabled(is_3d)
        self.anisotropy.setEnabled(is_3d and self.mode.currentText() == MODE_3D)
        self.stitch.setEnabled(is_3d and self.mode.currentText() == MODE_STITCH)
        self.z_first.setEnabled(is_3d)
        self.z_last.setEnabled(is_3d)
        self.keep_inside.setEnabled(isinstance(self._layer(self.region.currentText()), Labels))

    def add_region_layer(self):
        """Add a 2D labels layer over the image to paint the area to segment."""
        layer = self._layer(self.image.currentText())
        if layer is None:
            self.status.setText("Choose an image layer first.")
            return
        axes = layer_axes(layer)
        iy, ix = axes.index("Y"), axes.index("X")
        shape = (layer.data.shape[iy], layer.data.shape[ix])
        region = self.viewer.add_labels(
            np.zeros(shape, dtype=np.uint8),
            name=f"{layer.name} region",
            scale=[layer.scale[iy], layer.scale[ix]],
            translate=[layer.translate[iy], layer.translate[ix]],
            opacity=0.4,
        )
        region.brush_size = max(10, min(shape) // 15)
        region.mode = "paint"
        self.viewer.layers.selection.active = region
        self.region.setCurrentText(region.name)
        self.status.setText(
            "Paint over the cells to segment (it applies to every plane in the Z range "
            "below), then Run. Only the bounding box of the painted area is segmented."
        )

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
            self._cancelling = True
            self._worker.quit()
            self.process.cancel()  # stops Cellpose at once; the model is reloaded after
            self.status.setText("Cancelling…")
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
        region = self._layer(self.region.currentText())
        z_range = (self.z_first.value(), self.z_last.value()) if "Z" in axes else None
        crop = region_crop(layer, axes, region, z_range)
        if crop is None:
            self.status.setText(
                "Nothing to segment: the region is empty or outside the image, "
                "or the Z range is empty."
            )
            return
        mask = None
        if isinstance(region, Labels) and self.keep_inside.isChecked():
            mask = painted_mask_yx(layer, region, crop)
        tag = " region" if region is not None else ""
        if self.scope.currentText() == SCOPE_CURRENT:
            worker = self._run_current(layer, nuc, keep, params, crop, mask, tag)
        else:
            worker = self._run_all(layer, nuc, axes, keep, params, crop, mask, tag)
        worker.yielded.connect(self._on_progress)
        worker.errored.connect(lambda e: self._show_error("Cellpose failed", e))
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self.run_btn.setText("Cancel")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Running Cellpose…")
        worker.start()

    def _run_current(self, layer, nuc, keep, params, crop, mask, tag):
        vol, kept, scale, index = current_volume(self.viewer, layer, keep=keep, crop=crop)
        nuc_vol = np.asarray(nuc.data[index]) if nuc is not None else None
        translate = _crop_translate(layer, index)
        tag = _index_tag(layer_axes(layer), index) + tag

        @thread_worker
        def work():
            try:
                yield from self._load(params)
                yield _segmenting_message(vol.shape, params)
                masks = yield from self.process.call("segment", vol, nuc_vol, **params)
            except Cancelled:
                return None
            return keep_inside(masks, mask)

        worker = work()
        worker.returned.connect(self._add_labels_cb(layer, scale, translate, kept, tag))
        return worker

    def _run_all(self, layer, nuc, axes, keep, params, crop, mask, tag):
        outer = [i for i, a in enumerate(axes) if a not in keep]
        outer_shape = [layer.data.shape[i] for i in outer]
        # Kept (spatial) axes are limited to the region; others are looped over.
        spatial = [slice(*crop[a]) if a in crop else slice(None) for a in axes]
        out_shape = [len(range(*s.indices(n))) for s, n in zip(spatial, layer.data.shape)]
        translate = _crop_translate(layer, spatial)

        @thread_worker
        def work():
            try:
                yield from self._load(params)
            except Cancelled:
                return None
            labels = np.zeros(out_shape, dtype=np.uint32)
            positions = list(np.ndindex(*outer_shape))
            for n, pos in enumerate(positions):
                yield (n, len(positions))
                index, out_index = list(spatial), [slice(None)] * layer.ndim
                for i, p in zip(outer, pos):
                    index[i] = out_index[i] = p
                index = tuple(index)
                vol = np.asarray(layer.data[index])
                nuc_vol = np.asarray(nuc.data[index]) if nuc is not None else None
                try:
                    masks = yield from self.process.call("segment", vol, nuc_vol, **params)
                except Cancelled:
                    return None
                labels[tuple(out_index)] = keep_inside(masks, mask)
            return labels

        worker = work()
        worker.returned.connect(
            self._add_labels_cb(layer, list(layer.scale), translate, axes, "all" + tag)
        )
        return worker

    def _load(self, params):
        yield from _cellpose.ensure_weights(params["model"])
        yield from self.process.load(params["model"], params["gpu"])

    def _add_labels_cb(self, layer, scale, translate, axes, tag):
        def add(labels):
            if labels is None or self._cancelling:
                return
            meta = {**(layer.metadata or {}), "ibif_axes": axes, "ibif_cellpose": True}
            name = f"{layer.name} cellpose {tag}".strip()
            self.viewer.add_labels(labels, name=name, scale=scale, translate=translate, metadata=meta)
            self.status.setText(f"{name}: {int(labels.max())} objects (max label).")

        return add

    def _on_progress(self, value):
        if value is None or self._cancelling:
            return
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
        if self._cancelling:
            self._cancelling = False
            self.status.setText("Cancelled. Reloading the model in the background…")
            self.preload_model()


def _segmenting_message(shape, params):
    device = "GPU" if params["gpu"] else "CPU"
    mode = "3D" if params["do_3D"] and len(shape) == 3 else "2D"
    slow = " This can take several minutes on CPU." if device == "CPU" else ""
    started = time.strftime("%H:%M")
    return f"Segmenting {mode} volume {shape} on {device} (started {started}).{slow}"


def _crop_translate(layer, index):
    """Translate of the sliced axes, so a cropped result sits where it was cut from."""
    return [
        t + (i.start or 0) * s
        for t, i, s in zip(layer.translate, index, layer.scale)
        if isinstance(i, slice)
    ]


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
