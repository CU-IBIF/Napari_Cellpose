"""Secondary objects panel: expand, ring, rim, whole cells grown from nuclei, or structures
(vesicles, nucleoli, filaments...) inside each object."""
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
from ._util import aligned_image, spatial_ndim  # noqa: F401  (aligned_image is re-exported)

EXPAND = "Expand (dilate)"
RING = "Ring around each object"
RIM = "Rim inside each object"
CELLS = "Whole cells from these seeds"
STRUCTURES = "Structures inside each object"
KINDS = {
    "Membranes (bright cell borders)": "membrane",
    "Cytoplasm (bright cell interior)": "cytoplasm",
    "Distance only (ignore intensity)": "distance",
}
STRUCTURE_KINDS = {
    "Spots (vesicles, puncta, nucleoli)": "spots",
    "Filaments (thin lines)": "filaments",
    "Regions (organelles, any brighter area)": "regions",
}
SIZE_TIPS = {
    "spots": "Diameter of the largest spot; the background is measured over this distance",
    "filaments": "Width of the filaments",
    "regions": "",
}
THRESHOLDS = {"Otsu": "otsu", "Li": "li", "Triangle": "triangle", "Manual": "manual",
              "None (whole image)": "none"}


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
        self.method.addItems([EXPAND, RING, RIM, CELLS, STRUCTURES])
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
        self.structure = QComboBox()
        self.structure.addItems(list(STRUCTURE_KINDS))
        self.structure.setToolTip("What to look for inside each object")
        form.addRow("Find", self.structure)
        self.size = _spin(0.05, 1000, 1.0, SIZE_TIPS["spots"], step=0.1)
        form.addRow("Size", self.size)
        self.dark = QCheckBox("Darker than their surroundings (e.g. nucleoli in DAPI)")
        form.addRow("", self.dark)
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
        self.per_object = QCheckBox("Threshold each object on its own")
        self.per_object.setToolTip(
            "Better when objects differ in brightness; but an object with no structures still gets some")
        form.addRow("", self.per_object)
        self.min_size = _spin(0, 1e6, 0.0, "Drop pieces smaller than this", step=0.1)
        form.addRow("Smallest piece", self.min_size)

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
            self.factor, self.manual, self.max_distance, self.smooth, self.add_cytoplasm,
            self.structure, self.size, self.dark, self.per_object, self.min_size)}
        self.method.currentTextChanged.connect(self._update)
        self.kind.currentTextChanged.connect(self._update)
        self.structure.currentTextChanged.connect(self._update)
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
        inner = method == STRUCTURES
        uses_channel = cells or inner
        threshold = THRESHOLDS[self.threshold.currentText()]
        structure = STRUCTURE_KINDS[self.structure.currentText()]
        shown = {
            self.distance: method == EXPAND,
            self.gap: method == RING,
            self.width: method in (RING, RIM),
            self.channels: uses_channel, self.kind: cells, self.threshold: uses_channel,
            self.smooth: uses_channel,
            self.factor: uses_channel and threshold in ("otsu", "li", "triangle"),
            self.manual: uses_channel and threshold == "manual",
            self.max_distance: cells, self.add_cytoplasm: cells,
            self.structure: inner, self.size: inner and structure != "regions", self.dark: inner,
            self.per_object: inner and threshold != "manual", self.min_size: inner,
        }
        self._rows[self.channels].setText("Channels" if inner else "Cell channels")
        self.channels.setToolTip("Channel(s) that show the structures; several are averaged" if inner
                                 else "Channel(s) that show the whole cells; several are averaged")
        self.size.setToolTip(SIZE_TIPS[structure])
        units = self._units()
        for w, visible in shown.items():
            w.setVisible(visible)
            label = self._rows.get(w)
            if label is not None:
                label.setVisible(visible)
        for w in (self.distance, self.gap, self.width, self.max_distance, self.smooth, self.size):
            w.setSuffix(f" {units}")
        layer = self._layer(self.primary.currentText())
        nd = spatial_ndim(layer) if layer is not None else 2
        self.min_size.setSuffix(f" {units}³" if nd == 3 else f" {units}²")

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
        channels = self.checked_channels() if method in (CELLS, STRUCTURES) else []
        if method == CELLS and not channels:
            self.status.setText("Tick the channel(s) that show the whole cells.")
            return
        if method == STRUCTURES and not channels:
            self.status.setText("Tick the channel(s) that show the structures.")
            return
        if method == STRUCTURES and THRESHOLDS[self.threshold.currentText()] == "none":
            self.status.setText("Choose a threshold (e.g. Otsu) to find the structures.")
            return
        settings = dict(
            distance=self.distance.value(), gap=self.gap.value(), width=self.width.value(),
            kind=KINDS[self.kind.currentText()], threshold=THRESHOLDS[self.threshold.currentText()],
            value=self.manual.value(), factor=self.factor.value(),
            max_distance=self.max_distance.value() or None, smooth=self.smooth.value(),
            structure=STRUCTURE_KINDS[self.structure.currentText()], size=self.size.value(),
            dark=self.dark.isChecked(), per_object=self.per_object.isChecked(),
            min_size=self.min_size.value(),
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
                elif method == STRUCTURES:
                    image = _secondary.combine_channels(
                        [aligned_image(viewer, c, labels, pos) for c in channels]
                    )
                    found, used = _secondary.structures(
                        seeds, image, spacing, settings["structure"], settings["size"],
                        settings["threshold"], settings["value"], settings["factor"],
                        settings["per_object"], settings["dark"], settings["smooth"], settings["min_size"],
                    )
                    out[pos] = found
                    thresholds.extend(used)
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
            STRUCTURES: f"{labels.name} {settings['structure']}",
        }[method]
        meta = {**(labels.metadata or {}), "ibif_cellpose": True, "ibif_secondary": method,
                "ibif_seeds": labels.name}
        meta.pop("ibif_pieces", None)
        if method == STRUCTURES:
            meta["ibif_pieces"] = True  # the measurements count the pieces of each object
        common = dict(scale=labels.scale, translate=labels.translate, metadata=meta)
        self.viewer.add_labels(out, name=name, **common)
        if cyto is not None:
            self.viewer.add_labels(cyto, name=f"{labels.name} cytoplasm", **common)
        count = len(np.unique(out)) - (1 if (out == 0).any() else 0)
        text = f"{name}: {count} objects."
        if method == STRUCTURES:
            parents = len(np.unique(labels.data)) - (1 if (labels.data == 0).any() else 0)
            text = (f"{name}: structures in {count} of {parents} objects; each object's structures "
                    f"share its number.")
        if thresholds:
            label = "Threshold" if method == STRUCTURES else "Foreground threshold"
            text += f" {label} {np.mean(thresholds):.3f} (channels scaled to 0-1)."
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
