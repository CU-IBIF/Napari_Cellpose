"""Secondary objects panel: expand, ring, rim, whole cells grown from nuclei, or structures
(vesicles, nucleoli, filaments...) inside each object."""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi
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
from ._process import Cancelled
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
INTENSITY_WATERSHED = "Intensity + watershed"
CELLPOSE = "Cellpose"
SPOT_METHODS = (INTENSITY_WATERSHED, CELLPOSE)
SPLITS = {
    "Watershed (distance)": "distance",
    "Watershed (intensity)": "intensity",
    "None (each connected piece)": "none",
}
THRESHOLDS = {"Otsu": "otsu", "Li": "li", "Triangle": "triangle", "Manual": "manual",
              "None (whole image)": "none"}


class SecondaryWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._backend = None  # Cellpose, while a Cellpose run is going on
        self._cancelling = False
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
        self.finder = QComboBox()
        self.finder.addItems(list(SPOT_METHODS))
        self.finder.setToolTip(
            "Intensity + watershed: background subtraction, a threshold and a watershed, here on the "
            "CPU. Cellpose: deep learning, where Segmentation → Cellpose runs (this computer, a "
            "server or a cluster), with its model")
        form.addRow("Method", self.finder)
        self.cellpose_info = QLabel()
        self.cellpose_info.setWordWrap(True)
        form.addRow("Cellpose", self.cellpose_info)
        self.cellprob = _spin(-6, 6, 0.0, "Cellpose cell probability threshold: lower finds more "
                              "(dimmer) spots, higher fewer", unit="", step=0.5)
        form.addRow("Probability threshold", self.cellprob)
        self.max_outside = _spin(0, 100, 50.0, "A spot with more than this share of its area outside "
                                 "its object is dropped; the rest is cut to the object", unit="%", step=5)
        form.addRow("Outside the object", self.max_outside)
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
        self.min_size = _spin(0, 1e6, 0.0, "Drop inclusions smaller than this", step=0.1)
        form.addRow("Smallest inclusion", self.min_size)
        self.split = QComboBox()
        self.split.addItems(list(SPLITS))
        self.split.setToolTip(
            "How touching inclusions are told apart. Distance: cut along the neck between round "
            "inclusions. Intensity: one inclusion per bright centre. None: touching inclusions "
            "count as one")
        form.addRow("Split touching", self.split)
        self.diameter = _spin(0, 1000, 0.0, "Typical inclusion diameter: seeds closer than half "
                              "of it are one inclusion (auto = the median size of the pieces)", step=0.1)
        self.diameter.setSpecialValueText("auto")
        form.addRow("Inclusion diameter", self.diameter)

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
            self.structure, self.size, self.dark, self.per_object, self.min_size, self.split,
            self.diameter, self.finder, self.cellpose_info, self.cellprob, self.max_outside)}
        self.method.currentTextChanged.connect(self._update)
        self.kind.currentTextChanged.connect(self._update)
        self.structure.currentTextChanged.connect(self._update)
        self.threshold.currentTextChanged.connect(self._update)
        self.split.currentTextChanged.connect(self._update)
        self.finder.currentTextChanged.connect(self._update)
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
        cellpose = self._uses_cellpose()
        classic = uses_channel and not cellpose  # thresholded (not Cellpose)
        self._update_cellpose_choice()
        shown = {
            self.distance: method == EXPAND,
            self.gap: method == RING,
            self.width: method in (RING, RIM),
            self.channels: uses_channel, self.kind: cells, self.threshold: classic,
            self.smooth: classic,
            self.factor: classic and threshold in ("otsu", "li", "triangle"),
            self.manual: classic and threshold == "manual",
            self.max_distance: cells, self.add_cytoplasm: cells,
            self.structure: inner, self.size: inner and structure != "regions", self.dark: inner and not cellpose,
            self.per_object: classic and inner and threshold != "manual", self.min_size: inner,
            self.split: classic and inner and structure != "filaments",
            self.diameter: classic and inner and structure != "filaments"
            and SPLITS[self.split.currentText()] != "none",
            self.finder: inner and structure == "spots",
            self.cellpose_info: cellpose, self.cellprob: cellpose, self.max_outside: cellpose,
        }
        self._rows[self.size].setText("Spot diameter" if cellpose else "Size")
        self._rows[self.channels].setText("Channels" if inner else "Cell channels")
        self.channels.setToolTip("Channel(s) that show the structures; several are averaged" if inner
                                 else "Channel(s) that show the whole cells; several are averaged")
        self.size.setToolTip("Typical spot diameter, given to Cellpose" if cellpose else SIZE_TIPS[structure])
        units = self._units()
        for w, visible in shown.items():
            w.setVisible(visible)
            label = self._rows.get(w)
            if label is not None:
                label.setVisible(visible)
        for w in (self.distance, self.gap, self.width, self.max_distance, self.smooth, self.size, self.diameter):
            w.setSuffix(f" {units}")
        layer = self._layer(self.primary.currentText())
        nd = spatial_ndim(layer) if layer is not None else 2
        self.min_size.setSuffix(f" {units}³" if nd == 3 else f" {units}²")

    def _uses_cellpose(self):
        return (self.method.currentText() == STRUCTURES
                and STRUCTURE_KINDS[self.structure.currentText()] == "spots"
                and self.finder.currentText() == CELLPOSE)

    def _update_cellpose_choice(self):
        """Grey out Cellpose when it cannot run, saying why; say where it runs otherwise."""
        from ._cellpose_widget import cellpose_backend

        backend, why = cellpose_backend(self.viewer)
        item = self.finder.model().item(SPOT_METHODS.index(CELLPOSE))
        item.setEnabled(backend is not None)
        item.setToolTip(why or f"Runs with {backend.describe()}, as set in Segmentation → Cellpose")
        self.cellpose_info.setText(
            why or f"{backend.describe()}, as set in Segmentation → Cellpose.")
        return backend, why

    def showEvent(self, event):  # Segmentation → Cellpose may have changed meanwhile
        super().showEvent(event)
        self._update()

    def _layer(self, name):
        return self.viewer.layers[name] if name and name in self.viewer.layers else None

    def checked_channels(self):
        return [self.viewer.layers[self.channels.item(i).text()] for i in range(self.channels.count())
                if self.channels.item(i).checkState() == Qt.Checked
                and self.channels.item(i).text() in self.viewer.layers]

    # ---- running ----------------------------------------------------------------------------

    def run(self):
        if self._worker is not None:
            if self._backend is not None:  # Cellpose: stop it at once
                self._cancelling = True
                self._worker.quit()
                self._backend.process.cancel()
                self.status.setText("Cancelling…")
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
            min_size=self.min_size.value(), split=SPLITS[self.split.currentText()],
            diameter=self.diameter.value(), cellprob=self.cellprob.value(),
            max_outside=self.max_outside.value() / 100.0,
        )
        backend = None
        if method == STRUCTURES and self._uses_cellpose():
            backend, why = self._update_cellpose_choice()
            if backend is None:
                self.status.setText(why)
                return
            settings["cellpose"] = backend.describe()
            cp_params = _secondary.cellpose_spot_params(spacing, settings["size"], backend.model,
                                                        backend.gpu, settings["cellprob"])
        try:  # check alignment now, so a mismatch is reported before any work
            for c in channels:
                aligned_image(self.viewer, c, labels, (0,) * len(lead_shape))
        except ValueError as e:
            self.status.setText(str(e))
            return
        viewer, add_cyto = self.viewer, self.add_cytoplasm.isChecked() and method == CELLS

        @thread_worker
        def work():
            if backend is not None:
                try:
                    yield from backend.load()
                except Cancelled:
                    return None
            seeds_all = np.asarray(labels.data)
            out = np.zeros_like(seeds_all)
            cyto = np.zeros_like(seeds_all) if add_cyto else None
            inclusions = np.zeros(seeds_all.shape, np.uint32) if method == STRUCTURES else None
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
                elif method == STRUCTURES and backend is not None:
                    image = _secondary.combine_channels(
                        [aligned_image(viewer, c, labels, pos) for c in channels]
                    )
                    masks = np.zeros(seeds.shape, np.uint32)
                    box = ndi.find_objects((seeds > 0).astype(np.uint8))
                    if box and box[0] is not None:
                        try:
                            masks[box[0]] = yield from backend.process.call(
                                "segment", np.ascontiguousarray(image[box[0]]), None, **cp_params)
                        except Cancelled:
                            return None
                    found, one_by_one = _secondary.assign_to_parents(
                        seeds, masks, spacing, settings["max_outside"], settings["min_size"])
                    out[pos] = found
                    inclusions[pos] = one_by_one
                elif method == STRUCTURES:
                    image = _secondary.combine_channels(
                        [aligned_image(viewer, c, labels, pos) for c in channels]
                    )
                    found, used, one_by_one = _secondary.structures(
                        seeds, image, spacing, settings["structure"], settings["size"],
                        settings["threshold"], settings["value"], settings["factor"],
                        settings["per_object"], settings["dark"], settings["smooth"], settings["min_size"],
                        split=settings["split"], diameter=settings["diameter"], return_inclusions=True,
                    )
                    out[pos] = found
                    inclusions[pos] = one_by_one
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
            return out, cyto, thresholds, inclusions

        worker = work()
        worker.yielded.connect(self._on_progress)
        worker.returned.connect(lambda r: None if r is None or self._cancelling
                                else self._add(labels, method, settings, *r))
        worker.errored.connect(lambda e: self.status.setText(f"Failed: {e}"))
        worker.finished.connect(self._finished)
        self._worker = worker
        self._backend = backend
        self._cancelling = False
        if backend is None:
            self.run_btn.setEnabled(False)
        else:
            self.run_btn.setText("Cancel")
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Working…")
        worker.start()

    def _on_progress(self, value):
        if value is None or self._cancelling:
            return
        if isinstance(value, str):  # Cellpose: loading the model, waiting for the server...
            self.status.setText(value)
            return
        done, total = value
        if total > 1:
            self.progress.setRange(0, total)
            self.progress.setValue(done)

    def _finished(self):
        if self._cancelling:
            self.status.setText("Cancelled.")
        self._worker = None
        self._backend = None
        self._cancelling = False
        self.run_btn.setText("Make objects")
        self.run_btn.setEnabled(True)
        self.progress.setVisible(False)

    def _add(self, labels, method, settings, out, cyto, thresholds, inclusions=None):
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
        for key in ("ibif_pieces", "ibif_inclusions", "ibif_inclusion_layer", "ibif_inclusion_ids"):
            meta.pop(key, None)
        common = dict(scale=labels.scale, translate=labels.translate)
        if method == STRUCTURES and inclusions is not None:
            # The same inclusions numbered one by one, each knowing its object (ibif_seeds).
            single = self.viewer.add_labels(inclusions, name=f"{name} inclusions", **common,
                                            metadata={**meta, "ibif_inclusion_ids": True})
            meta["ibif_inclusion_layer"] = single.name
        if method == STRUCTURES:
            meta["ibif_inclusions"] = True  # the measurements count the inclusions of each object
            if settings.get("cellpose"):
                meta["ibif_found_by"] = f"Cellpose, {settings['cellpose']}"
        common["metadata"] = meta
        self.viewer.add_labels(out, name=name, **common)
        if cyto is not None:
            self.viewer.add_labels(cyto, name=f"{labels.name} cytoplasm", **common)
        count = len(np.unique(out)) - (1 if (out == 0).any() else 0)
        text = f"{name}: {count} objects."
        if method == STRUCTURES:
            parents = len(np.unique(labels.data)) - (1 if (labels.data == 0).any() else 0)
            n_incl = int(sum(len(np.unique(p)) - 1 for p in _per_position(inclusions, labels))) \
                if inclusions is not None else 0
            text = (f"{name}: {n_incl} inclusions in {count} of {parents} objects. In {name!r} each "
                    f"object's inclusions share its number; in {name + ' inclusions'!r} each has its own.")
        if settings.get("cellpose") and method == STRUCTURES:
            text += f" Found by Cellpose ({settings['cellpose']})."
        if thresholds:
            label = "Threshold" if method == STRUCTURES else "Foreground threshold"
            text += f" {label} {np.mean(thresholds):.3f} (channels scaled to 0-1)."
        if method == CELLS:
            missing = len(np.unique(labels.data)) - len(np.unique(out))
            if missing > 0:
                text += f" {missing} seeds got no cell."
        self.status.setText(text)


def _per_position(data, labels):
    """The 2D / 3D blocks of `data` (laid out like `labels`), one per time point / position."""
    lead = labels.ndim - spatial_ndim(labels)
    return [data[pos] for pos in np.ndindex(*data.shape[:lead])]


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
