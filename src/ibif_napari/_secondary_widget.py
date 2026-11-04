"""Secondary objects panel: expand, ring, rim, or whole cells grown from nuclei."""
from __future__ import annotations

import numpy as np
from napari.layers import Image, Labels
from napari.qt.threading import thread_worker
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from . import _secondary
from ._util import current_data_index

EXPAND = "Expand (dilate)"
RING = "Ring around each object"
RIM = "Rim inside each object"
CELLS = "Whole cells from these seeds"
KINDS = {
    "Membranes (bright cell borders)": "membrane",
    "Cytoplasm (bright cell interior)": "cytoplasm",
    "Distance only (ignore intensity)": "distance",
}
THRESHOLDS = {"Otsu": "otsu", "Li": "li", "Triangle": "triangle", "Manual": "manual",
              "None (whole image)": "none"}


def spatial_ndim(layer):
    axes = (layer.metadata or {}).get("ibif_axes")
    if axes and len(axes) == layer.ndim:
        return sum(a in "ZYX" for a in axes)
    return 3 if layer.ndim >= 3 else 2


def aligned_image(viewer, image, labels, lead_index):
    """The pixels of `image` under one block of `labels` (its spatial volume at
    `lead_index` of its time/position axes). Image axes the labels do not have follow
    the viewer's current position. The two must share pixel size."""
    n_sp = spatial_ndim(labels)
    current = current_data_index(viewer, image)
    index = list(current)
    for k in range(1, image.ndim + 1):
        ia, la = image.ndim - k, labels.ndim - k
        if la < 0:
            continue
        ls, isc = float(labels.scale[la]), float(image.scale[ia])
        if abs(ls - isc) > 1e-6 * max(abs(ls), abs(isc), 1e-12):
            raise ValueError(f"{image.name} and {labels.name} have different pixel sizes")
        spatial = k <= n_sp
        start = float(labels.translate[la]) + (0 if spatial else lead_index[la] * ls)
        i0 = int(round((start - float(image.translate[ia])) / isc))
        if spatial:
            stop = i0 + labels.data.shape[la]
            if i0 < 0 or stop > image.data.shape[ia]:
                raise ValueError(f"{labels.name} reaches outside {image.name}")
            index[ia] = slice(i0, stop)
        else:
            if not 0 <= i0 < image.data.shape[ia]:
                raise ValueError(f"{labels.name} reaches outside {image.name}")
            index[ia] = i0
    data = image.data[0] if isinstance(image.data, list) else image.data
    return np.asarray(data[tuple(index)])


class SecondaryWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Make objects from a first segmentation (e.g. Cellpose nuclei). Each new object keeps "
            "the number of the object it came from."
        )
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        layout.addLayout(form)

        self.primary = QComboBox()
        form.addRow("Objects", self.primary)
        self.method = QComboBox()
        self.method.addItems([EXPAND, RING, RIM, CELLS])
        form.addRow("Make", self.method)

        self.distance = _spin(0, 1000, 2.0, "How far to grow each object")
        form.addRow("Distance", self.distance)
        self.gap = _spin(0, 1000, 0.0, "Space left between the object and its ring")
        form.addRow("Gap", self.gap)
        self.width = _spin(0, 1000, 2.0, "Thickness of the ring or rim")
        form.addRow("Width", self.width)

        self.channels = QListWidget()
        self.channels.setToolTip("Channel(s) that show the whole cells; several are averaged")
        self.channels.setMaximumHeight(110)
        form.addRow("Cell channels", self.channels)
        self.kind = QComboBox()
        self.kind.addItems(list(KINDS))
        form.addRow("Channel shows", self.kind)
        self.threshold = QComboBox()
        self.threshold.addItems(list(THRESHOLDS))
        self.threshold.setToolTip("What counts as cell (foreground) in the channel(s)")
        form.addRow("Foreground", self.threshold)
        self.factor = _spin(0.1, 10, 1.0, "Multiply the automatic threshold (lower = larger cells)", unit="", step=0.05)
        form.addRow("Threshold × ", self.factor)
        self.manual = _spin(0, 1, 0.2, "Manual threshold, on the channels scaled to 0-1", unit="", step=0.01)
        form.addRow("Manual threshold", self.manual)
        self.max_distance = _spin(0, 10000, 0.0, "Furthest a cell may reach from its seed (0 = no limit)")
        self.max_distance.setSpecialValueText("no limit")
        form.addRow("Max distance", self.max_distance)
        self.smooth = _spin(0, 100, 0.5, "Gaussian smoothing of the channels before thresholding")
        form.addRow("Smoothing", self.smooth)
        self.add_cytoplasm = QCheckBox("Also add cytoplasm (cell minus seed)")
        self.add_cytoplasm.setChecked(True)
        form.addRow("", self.add_cytoplasm)

        self.run_btn = QPushButton("Make objects")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self._rows = {w: form.labelForField(w) for w in (
            self.distance, self.gap, self.width, self.channels, self.kind, self.threshold,
            self.factor, self.manual, self.max_distance, self.smooth, self.add_cytoplasm)}
        self.method.currentTextChanged.connect(self._update)
        self.kind.currentTextChanged.connect(self._update)
        self.threshold.currentTextChanged.connect(self._update)
        self.primary.currentTextChanged.connect(self._update)
        for ev in (self.viewer.layers.events.inserted, self.viewer.layers.events.removed,
                   self.viewer.layers.events.reordered):
            ev.connect(self._refresh)
        self._refresh()

    # ---- choices --------------------------------------------------------------------------

    def _refresh(self, event=None):
        current = self.primary.currentText()
        names = [l.name for l in self.viewer.layers if isinstance(l, Labels)]
        self.primary.blockSignals(True)
        self.primary.clear()
        self.primary.addItems(names)
        if current in names:
            self.primary.setCurrentText(current)
        self.primary.blockSignals(False)
        checked = {self.channels.item(i).text() for i in range(self.channels.count())
                   if self.channels.item(i).checkState() == Qt.Checked}
        self.channels.clear()
        for layer in self.viewer.layers:
            if isinstance(layer, Image) and not layer.rgb:
                item = QListWidgetItem(layer.name)
                item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
                item.setCheckState(Qt.Checked if layer.name in checked else Qt.Unchecked)
                self.channels.addItem(item)
        self._update()

    def _units(self):
        layer = self._layer(self.primary.currentText())
        known = layer is not None and bool((layer.metadata or {}).get("ibif_voxel_um"))
        return "µm" if known else "px"

    def _update(self, *_):
        method = self.method.currentText()
        cells = method == CELLS
        shown = {
            self.distance: method == EXPAND,
            self.gap: method == RING,
            self.width: method in (RING, RIM),
            self.channels: cells, self.kind: cells, self.threshold: cells, self.smooth: cells,
            self.factor: cells and THRESHOLDS[self.threshold.currentText()] in ("otsu", "li", "triangle"),
            self.manual: cells and THRESHOLDS[self.threshold.currentText()] == "manual",
            self.max_distance: cells, self.add_cytoplasm: cells,
        }
        units = self._units()
        for w, visible in shown.items():
            w.setVisible(visible)
            label = self._rows.get(w)
            if label is not None:
                label.setVisible(visible)
        for w in (self.distance, self.gap, self.width, self.max_distance, self.smooth):
            w.setSuffix(f" {units}")

    def _layer(self, name):
        return self.viewer.layers[name] if name and name in self.viewer.layers else None

    def checked_channels(self):
        return [self.viewer.layers[self.channels.item(i).text()] for i in range(self.channels.count())
                if self.channels.item(i).checkState() == Qt.Checked
                and self.channels.item(i).text() in self.viewer.layers]

    # ---- running ----------------------------------------------------------------------------

    def run(self):
        if self._worker is not None:
            return
        labels = self._layer(self.primary.currentText())
        if labels is None:
            self.status.setText("Choose a labels layer (e.g. Cellpose nuclei) first.")
            return
        method = self.method.currentText()
        n_sp = spatial_ndim(labels)
        spacing = tuple(float(s) for s in labels.scale[-n_sp:])
        lead_shape = labels.data.shape[: labels.ndim - n_sp]
        channels = self.checked_channels() if method == CELLS else []
        if method == CELLS and not channels:
            self.status.setText("Tick the channel(s) that show the whole cells.")
            return
        settings = dict(
            distance=self.distance.value(), gap=self.gap.value(), width=self.width.value(),
            kind=KINDS[self.kind.currentText()], threshold=THRESHOLDS[self.threshold.currentText()],
            value=self.manual.value(), factor=self.factor.value(),
            max_distance=self.max_distance.value() or None, smooth=self.smooth.value(),
        )
        try:  # check alignment now, so a mismatch is reported before any work
            for c in channels:
                aligned_image(self.viewer, c, labels, (0,) * len(lead_shape))
        except ValueError as e:
            self.status.setText(str(e))
            return
        viewer, add_cyto = self.viewer, self.add_cytoplasm.isChecked() and method == CELLS

        @thread_worker
        def work():
            seeds_all = np.asarray(labels.data)
            out = np.zeros_like(seeds_all)
            cyto = np.zeros_like(seeds_all) if add_cyto else None
            positions = list(np.ndindex(*lead_shape))
            thresholds = []
            for n, pos in enumerate(positions):
                yield (n, len(positions))
                seeds = seeds_all[pos]
                if method == EXPAND:
                    out[pos] = _secondary.expand(seeds, settings["distance"], spacing)
                elif method == RING:
                    out[pos] = _secondary.ring(seeds, settings["width"], settings["gap"], spacing)
                elif method == RIM:
                    out[pos] = _secondary.ring(seeds, settings["width"], spacing=spacing, inside=True)
                else:
                    image = _secondary.combine_channels(
                        [aligned_image(viewer, c, labels, pos) for c in channels]
                    )
                    cells, used = _secondary.seeded_cells(
                        seeds, image, spacing, settings["kind"], settings["threshold"], settings["value"],
                        settings["factor"], settings["max_distance"], settings["smooth"],
                    )
                    out[pos] = cells
                    if used is not None:
                        thresholds.append(used)
                    if cyto is not None:
                        cyto[pos] = _secondary.cytoplasm(cells, seeds)
            return out, cyto, thresholds

        worker = work()
        worker.yielded.connect(self._on_progress)
        worker.returned.connect(lambda r: self._add(labels, method, settings, *r))
        worker.errored.connect(lambda e: self.status.setText(f"Failed: {e}"))
        worker.finished.connect(self._finished)
        self._worker = worker
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Working…")
        worker.start()

    def _on_progress(self, value):
        done, total = value
        if total > 1:
            self.progress.setRange(0, total)
            self.progress.setValue(done)

    def _finished(self):
        self._worker = None
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _add(self, labels, method, settings, out, cyto, thresholds):
        units = self._units()
        name = {
            EXPAND: f"{labels.name} expanded {settings['distance']:g} {units}",
            RING: f"{labels.name} ring {settings['gap']:g}+{settings['width']:g} {units}",
            RIM: f"{labels.name} rim {settings['width']:g} {units}",
            CELLS: f"{labels.name} cells",
        }[method]
        meta = {**(labels.metadata or {}), "ibif_cellpose": True, "ibif_secondary": method,
                "ibif_seeds": labels.name}
        common = dict(scale=labels.scale, translate=labels.translate, metadata=meta)
        self.viewer.add_labels(out, name=name, **common)
        if cyto is not None:
            self.viewer.add_labels(cyto, name=f"{labels.name} cytoplasm", **common)
        count = len(np.unique(out)) - (1 if (out == 0).any() else 0)
        text = f"{name}: {count} objects."
        if thresholds:
            text += f" Foreground threshold {np.mean(thresholds):.3f} (channels scaled to 0-1)."
        if method == CELLS:
            missing = len(np.unique(labels.data)) - len(np.unique(out))
            if missing > 0:
                text += f" {missing} seeds got no cell."
        self.status.setText(text)


def _spin(lo, hi, value, tip, unit="µm", step=0.5):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(2)
    s.setSingleStep(step)
    s.setValue(value)
    s.setToolTip(tip)
    if unit:
        s.setSuffix(f" {unit}")
    return s
