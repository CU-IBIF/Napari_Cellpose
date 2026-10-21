"""Dimensions panel: named sliders with physical values, channel toggles, 2D/3D, projections."""
from __future__ import annotations

import numpy as np
from napari.layers import Image
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from ._util import axis_labels_for, current_volume, layer_axes


class DimensionsWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._rows = {}  # world axis -> (slider, spinbox, readout)
        self._syncing = False
        self._watched = set()

        layout = QVBoxLayout(self)

        self._axes_box = QGroupBox("Position")
        self._axes_grid = QGridLayout(self._axes_box)
        layout.addWidget(self._axes_box)

        self._channels_box = QGroupBox("Channels")
        self._channels_layout = QVBoxLayout(self._channels_box)
        layout.addWidget(self._channels_box)

        view_box = QGroupBox("View")
        view = QVBoxLayout(view_box)
        row = QHBoxLayout()
        self.btn_2d = QPushButton("2D")
        self.btn_3d = QPushButton("3D")
        for b in (self.btn_2d, self.btn_3d):
            b.setCheckable(True)
            row.addWidget(b)
        self.btn_2d.clicked.connect(lambda: self._set_ndisplay(2))
        self.btn_3d.clicked.connect(lambda: self._set_ndisplay(3))
        view.addLayout(row)
        self.btn_mip = QPushButton("Max projection over Z (visible channels)")
        self.btn_mip.clicked.connect(self.max_projection)
        view.addWidget(self.btn_mip)
        self.btn_extract = QPushButton("Extract current time/position as new layers")
        self.btn_extract.clicked.connect(self.extract_current)
        view.addWidget(self.btn_extract)
        layout.addWidget(view_box)
        layout.addStretch(1)

        dims = self.viewer.dims
        dims.events.ndim.connect(self._rebuild)
        dims.events.order.connect(self._rebuild)
        dims.events.ndisplay.connect(self._rebuild)
        dims.events.range.connect(self._rebuild)
        dims.events.current_step.connect(self._sync_from_viewer)
        self.viewer.layers.events.inserted.connect(self._on_layers_changed)
        self.viewer.layers.events.removed.connect(self._on_layers_changed)
        self._on_layers_changed()

    # ---- position sliders -------------------------------------------------------------

    def _on_layers_changed(self, event=None):
        labels = axis_labels_for(self.viewer)
        if labels:
            self.viewer.dims.axis_labels = labels
        for layer in self.viewer.layers:
            if layer not in self._watched:
                layer.events.visible.connect(self._sync_channels)
                layer.events.name.connect(self._rebuild_channels)
                self._watched.add(layer)
        self._rebuild()
        self._rebuild_channels()

    def _rebuild(self, event=None):
        while self._axes_grid.count():
            w = self._axes_grid.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        self._rows.clear()

        dims = self.viewer.dims
        sliders = [a for a in dims.order[: dims.ndim - dims.ndisplay] if dims.nsteps[a] > 1]
        if not sliders:
            self._axes_grid.addWidget(QLabel("No time, position or Z steps."), 0, 0)
        for r, axis in enumerate(sliders):
            name = str(dims.axis_labels[axis])
            slider = QSlider(Qt.Horizontal)
            spin = QSpinBox()
            for w in (slider, spin):
                w.setRange(0, dims.nsteps[axis] - 1)
                w.setValue(dims.current_step[axis])
            slider.valueChanged.connect(lambda v, a=axis: self._set_step(a, v))
            spin.valueChanged.connect(lambda v, a=axis: self._set_step(a, v))
            readout = QLabel()
            readout.setMinimumWidth(80)
            self._axes_grid.addWidget(QLabel(name), r, 0)
            self._axes_grid.addWidget(slider, r, 1)
            self._axes_grid.addWidget(spin, r, 2)
            self._axes_grid.addWidget(readout, r, 3)
            self._rows[axis] = (slider, spin, readout)
        self.btn_2d.setChecked(dims.ndisplay == 2)
        self.btn_3d.setChecked(dims.ndisplay == 3)
        self._sync_from_viewer()

    def _set_step(self, axis, value):
        if not self._syncing:
            self.viewer.dims.set_current_step(axis, value)

    def _sync_from_viewer(self, event=None):
        dims = self.viewer.dims
        self._syncing = True
        try:
            for axis, (slider, spin, readout) in self._rows.items():
                if axis >= dims.ndim:
                    continue
                step = dims.current_step[axis]
                slider.setValue(step)
                spin.setValue(step)
                readout.setText(self._physical(axis, step))
        finally:
            self._syncing = False

    def _physical(self, axis, step):
        label = str(self.viewer.dims.axis_labels[axis])
        if label == "time":
            interval = self._metadata_value("ibif_time_interval_s")
            if interval:
                t = step * interval
                return f"{t / 60:.2f} min" if t >= 120 else f"{t:.1f} s"
        if label == "z":
            return f"{self.viewer.dims.point[axis]:.2f} µm"
        return f"{step + 1} / {self.viewer.dims.nsteps[axis]}"

    def _metadata_value(self, key):
        for layer in self.viewer.layers:
            if layer.metadata and layer.metadata.get(key):
                return layer.metadata[key]
        return None

    def _set_ndisplay(self, n):
        if n == 3 and self.viewer.dims.ndim < 3:
            n = 2
        self.viewer.dims.ndisplay = n
        self.btn_2d.setChecked(self.viewer.dims.ndisplay == 2)
        self.btn_3d.setChecked(self.viewer.dims.ndisplay == 3)

    # ---- channels ---------------------------------------------------------------------

    def _image_layers(self):
        return [l for l in self.viewer.layers if isinstance(l, Image)]

    def _rebuild_channels(self, event=None):
        while self._channels_layout.count():
            w = self._channels_layout.takeAt(0).widget()
            if w is not None:
                w.deleteLater()
        layers = self._image_layers()
        if not layers:
            self._channels_layout.addWidget(QLabel("Open an image (File > Open, or drag in)."))
        for layer in layers:
            box = QCheckBox(layer.name)
            box.setChecked(layer.visible)
            box.toggled.connect(lambda on, l=layer: setattr(l, "visible", on))
            box.setProperty("layer_name", layer.name)
            self._channels_layout.addWidget(box)

    def _sync_channels(self, event=None):
        for i in range(self._channels_layout.count()):
            box = self._channels_layout.itemAt(i).widget()
            if isinstance(box, QCheckBox):
                name = box.property("layer_name")
                if name in self.viewer.layers:
                    box.blockSignals(True)
                    box.setChecked(self.viewer.layers[name].visible)
                    box.blockSignals(False)

    # ---- derived layers ---------------------------------------------------------------

    def _visible_images(self):
        return [l for l in self._image_layers() if l.visible and not l.rgb]

    def max_projection(self):
        for layer in self._visible_images():
            if "Z" not in layer_axes(layer):
                continue
            vol, kept, scale, _ = current_volume(self.viewer, layer, keep="ZYX")
            mip = vol.max(axis=kept.index("Z"))
            self._add_like(layer, mip, scale[kept.index("Z") + 1 :], f"{layer.name} MIP", "YX")

    def extract_current(self):
        for layer in self._visible_images():
            vol, kept, scale, _ = current_volume(self.viewer, layer, keep="ZYX")
            self._add_like(layer, vol, scale, f"{layer.name} {self._position_tag()}", kept)

    def _position_tag(self):
        dims = self.viewer.dims
        parts = []
        for axis in dims.order[: dims.ndim - dims.ndisplay]:
            label = str(dims.axis_labels[axis])
            if label in ("time", "position") and dims.nsteps[axis] > 1:
                parts.append(f"{label[0].upper()}{dims.current_step[axis]}")
        return " ".join(parts) or "copy"

    def _add_like(self, layer, data, scale, name, axes):
        meta = {**(layer.metadata or {}), "ibif_axes": axes}
        self.viewer.add_image(
            np.asarray(data),
            name=name,
            scale=scale,
            colormap=layer.colormap,
            contrast_limits=layer.contrast_limits,
            blending=layer.blending,
            metadata=meta,
        )
