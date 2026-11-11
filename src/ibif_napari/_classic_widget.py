"""Segmentation, Threshold + watershed tab: classical segmentation on the CPU.

Local contrast (CLAHE), smoothing, an automatic / local / manual threshold, then touching
objects split by a watershed (or connected components). No GPU, model or server needed:
for testing, for bright separated objects, or when Cellpose cannot run. See `_classic`.
"""
from __future__ import annotations

import numpy as np
from napari.layers import Image, Labels, Shapes
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import _classic
from ._region import add_region_layer, keep_inside, painted_mask, region_crop
from ._util import current_volume, layer_axes

MODE_3D = "3D"
MODE_PLANES = "2D per plane, stitched"
SCOPE_CURRENT = "Current time point / position"
SCOPE_ALL = "Every time point / position"
WHOLE_IMAGE = "(whole image)"
THRESHOLDS = {"Otsu": "otsu", "Li": "li", "Triangle": "triangle", "Yen": "yen", "Mean": "mean",
              "Local (adaptive)": "local", "Manual": "manual"}
SPLITS = {"Distance (round objects)": "distance", "Intensity (bright centres)": "intensity",
          "None (connected components)": "none"}
PREVIEW = "foreground"


class ClassicWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._cancelling = False
        self._region_syncs = []
        layout = QVBoxLayout(self)
        intro = QLabel("Classical segmentation on this computer's CPU: no GPU, model or server "
                       "needed. Best for bright, fairly separated objects such as nuclei.")
        intro.setWordWrap(True)
        layout.addWidget(intro)
        form = QFormLayout()
        layout.addLayout(form)

        self.image = QComboBox()
        form.addRow("Image", self.image)
        self.mode = QComboBox()
        self.mode.addItems([MODE_3D, MODE_PLANES])
        self.mode.setToolTip("Z stacks: segment in 3D, or each plane in 2D and link the planes "
                             "(for few planes or a large Z step)")
        form.addRow("Mode", self.mode)

        clahe_row = QHBoxLayout()
        self.clahe = QCheckBox("Local contrast (CLAHE)")
        self.clahe.setToolTip("Contrast-limited adaptive histogram equalisation: evens out bright "
                              "and dim regions so one threshold fits the whole image")
        self.clahe_clip = _float_spin(0.001, 1, 0.05, 0.005, "Clip limit: higher = stronger contrast "
                                      "(and more noise)", decimals=3)
        clahe_row.addWidget(self.clahe)
        clahe_row.addWidget(QLabel("clip"))
        clahe_row.addWidget(self.clahe_clip)
        form.addRow("", clahe_row)
        self.smooth = _float_spin(0, 20, 1.0, 0.5, "Gaussian blur, sigma in XY pixels (Z follows the "
                                  "voxel size); removes noise that breaks objects up. 0 = none")
        form.addRow("Smoothing (px)", self.smooth)

        self.threshold = QComboBox()
        self.threshold.addItems(list(THRESHOLDS))
        self.threshold.setToolTip(
            "Otsu, Li, Triangle, Yen, Mean: one automatic threshold for the whole image.\n"
            "Local (adaptive): each pixel compared with its neighbourhood (uneven background).\n"
            "Manual: a value in the image's own units (Preview shows the automatic one).")
        form.addRow("Threshold", self.threshold)
        self.factor = _float_spin(0.1, 5, 1.0, 0.05, "The automatic threshold is multiplied by this "
                                  "(lower = larger objects, more included)")
        form.addRow("Threshold ×", self.factor)
        self.value = _float_spin(-1e9, 1e9, 0, 10, "Pixels brighter than this are objects", decimals=1)
        form.addRow("Value", self.value)
        local_row = QHBoxLayout()
        self.block = QSpinBox()
        self.block.setRange(3, 2001)
        self.block.setSingleStep(2)
        self.block.setValue(51)
        self.block.setToolTip("Neighbourhood size in pixels (about 2-3 object diameters)")
        self.offset = _float_spin(-1, 1, -0.05, 0.01, "Subtracted from the local mean (0-1 scale): "
                                  "-0.05 = a pixel must be 0.05 brighter than its neighbourhood; "
                                  "more negative = stricter", decimals=3)
        local_row.addWidget(QLabel("block"))
        local_row.addWidget(self.block)
        local_row.addWidget(QLabel("offset"))
        local_row.addWidget(self.offset)
        form.addRow("Local", local_row)
        self.fill_holes = QCheckBox("Fill holes")
        self.fill_holes.setChecked(True)
        form.addRow("", self.fill_holes)

        self.split = QComboBox()
        self.split.addItems(list(SPLITS))
        self.split.setToolTip(
            "Distance: seeds at object centres, cuts touching round objects along their neck.\n"
            "Intensity: seeds at the brightest points (objects brighter in the middle).\n"
            "None: every connected group of pixels is one object.")
        form.addRow("Split touching", self.split)
        self.diameter = _float_spin(0, 1000, 0, 1, "Typical object diameter in pixels: seeds closer "
                                    "than half of it are merged (0 = from the objects found)")
        self.diameter.setSpecialValueText("auto")
        form.addRow("Diameter (px)", self.diameter)
        self.min_size = QSpinBox()
        self.min_size.setRange(0, 1_000_000)
        self.min_size.setValue(15)
        form.addRow("Min size (px)", self.min_size)
        self.clear_border = QCheckBox("Drop objects touching the image border")
        form.addRow("", self.clear_border)

        self.scope = QComboBox()
        self.scope.addItems([SCOPE_CURRENT, SCOPE_ALL])
        form.addRow("Run on", self.scope)
        region_row = QHBoxLayout()
        self.region = QComboBox()
        self.region.setToolTip("Segment only this area: a Labels layer (painted) or a Shapes layer (rectangle)")
        draw = QPushButton("Draw…")
        draw.setToolTip("Add a layer to paint the area to segment")
        draw.clicked.connect(self.add_region_layer)
        region_row.addWidget(self.region, 1)
        region_row.addWidget(draw)
        form.addRow("Region", region_row)
        z_row = QHBoxLayout()
        self.z_first, self.z_last = QSpinBox(), QSpinBox()
        z_row.addWidget(self.z_first)
        z_row.addWidget(QLabel("to"))
        z_row.addWidget(self.z_last)
        form.addRow("Z planes", z_row)
        self.keep_inside = QCheckBox("Drop objects mostly outside the painted area")
        self.keep_inside.setChecked(True)
        form.addRow("", self.keep_inside)

        buttons = QHBoxLayout()
        self.preview_btn = QPushButton("Preview threshold")
        self.preview_btn.setToolTip("Show the pixels above the threshold (current time point / "
                                    "position) in a layer, to tune the settings before running")
        self.preview_btn.clicked.connect(self.preview)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(self.run)
        buttons.addWidget(self.preview_btn)
        buttons.addWidget(self.run_btn)
        layout.addLayout(buttons)
        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        layout.addStretch(1)

        self.threshold.currentTextChanged.connect(self._update_enabled)
        self.split.currentTextChanged.connect(self._update_enabled)
        self.clahe.toggled.connect(self._update_enabled)
        self.region.currentTextChanged.connect(self._update_enabled)
        self.image.currentTextChanged.connect(self._on_image_changed)
        self.z_first.valueChanged.connect(self._sync_region_ranges)
        self.z_last.valueChanged.connect(self._sync_region_ranges)
        for ev in (self.viewer.layers.events.inserted, self.viewer.layers.events.removed,
                   self.viewer.layers.events.reordered):
            ev.connect(self._refresh_layers)
        self._refresh_layers()

    # ---- settings ----------------------------------------------------------------------------

    def settings(self):
        """The current settings as a `_classic.Settings`."""
        return _classic.Settings(
            clahe=self.clahe.isChecked(), clahe_clip=self.clahe_clip.value(),
            smooth=self.smooth.value(), threshold=THRESHOLDS[self.threshold.currentText()],
            factor=self.factor.value(), value=self.value.value(), block=self.block.value(),
            offset=self.offset.value(), fill_holes=self.fill_holes.isChecked(),
            split=SPLITS[self.split.currentText()], diameter=self.diameter.value(),
            min_size=self.min_size.value(), clear_border=self.clear_border.isChecked())

    def set_threshold(self, method):
        """Choose the threshold by its short name (otsu, li, ..., local, manual)."""
        self.threshold.setCurrentText(next(k for k, v in THRESHOLDS.items() if v == method))

    def set_split(self, method):
        self.split.setCurrentText(next(k for k, v in SPLITS.items() if v == method))

    def _update_enabled(self, *_):
        method = THRESHOLDS[self.threshold.currentText()]
        self.factor.setEnabled(method not in ("local", "manual"))
        self.value.setEnabled(method == "manual")
        self.block.setEnabled(method == "local")
        self.offset.setEnabled(method == "local")
        self.clahe_clip.setEnabled(self.clahe.isChecked())
        self.diameter.setEnabled(SPLITS[self.split.currentText()] != "none")
        layer = self._layer(self.image.currentText())
        is_3d = layer is not None and "Z" in layer_axes(layer)
        for w in (self.mode, self.z_first, self.z_last):
            w.setEnabled(is_3d)
        self.keep_inside.setEnabled(isinstance(self._layer(self.region.currentText()), Labels))

    # ---- layers ------------------------------------------------------------------------------

    def _layer(self, name):
        return self.viewer.layers[name] if name and name in self.viewer.layers else None

    def _refresh_layers(self, event=None):
        images = [l.name for l in self.viewer.layers if isinstance(l, Image) and not l.rgb]
        regions = [l.name for l in self.viewer.layers if isinstance(l, (Labels, Shapes))
                   and not {"ibif_cellpose", "ibif_filaments", "ibif_preview"} & set(l.metadata or {})]
        for combo, extra, names in ((self.image, [], images), (self.region, [WHOLE_IMAGE], regions)):
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
        if layer is not None:
            axes = layer_axes(layer)
            nz = layer.data.shape[axes.index("Z")] if "Z" in axes else 1
            if self.z_last.maximum() != nz - 1 or self.z_last.value() > nz - 1:
                for spin in (self.z_first, self.z_last):
                    spin.setRange(0, nz - 1)
                self.z_first.setValue(0)
                self.z_last.setValue(nz - 1)
        self._update_enabled()

    def add_region_layer(self):
        layer = self._layer(self.image.currentText())
        if layer is None:
            self.status.setText("Choose an image layer first.")
            return
        region, sync = add_region_layer(self.viewer, layer, layer_axes(layer),
                                        (self.z_first.value(), self.z_last.value()), self._grow_z_range)
        if sync is not None:
            self._region_syncs.append(sync)
        self.viewer.layers.selection.active = region
        self.region.setCurrentText(region.name)
        self.status.setText("Paint over the objects to segment, then Run. Only the bounding box of "
                            "the painted area is segmented.")

    def _sync_region_ranges(self, *_):
        self._region_syncs = [s for s in self._region_syncs if s.layer in self.viewer.layers]
        for sync in self._region_syncs:
            sync.set_range(self.z_first.value(), self.z_last.value())

    def _grow_z_range(self, first, last):
        self.z_first.setValue(min(first, self.z_first.value()))
        self.z_last.setValue(max(last, self.z_last.value()))

    # ---- running -----------------------------------------------------------------------------

    def _prepare(self):
        layer = self._layer(self.image.currentText())
        if layer is None:
            self.status.setText("Choose an image layer first.")
            return None
        axes = layer_axes(layer)
        keep = "ZYX" if "Z" in axes else "YX"
        region = self._layer(self.region.currentText())
        z_range = (self.z_first.value(), self.z_last.value()) if "Z" in axes else None
        crop = region_crop(layer, axes, region, z_range)
        if crop is None:
            self.status.setText("Nothing to segment: the region is empty or outside the image, "
                                "or the Z range is empty.")
            return None
        mask = painted_mask(layer, region, crop) if isinstance(region, Labels) and \
            self.keep_inside.isChecked() else None
        planes = "Z" in axes and self.mode.currentText() == MODE_PLANES
        return dict(layer=layer, axes=axes, keep=keep, crop=crop, mask=mask, planes=planes,
                    settings=self.settings(), tag=" region" if region is not None else "")

    def preview(self):
        """The foreground (pixels above the threshold) of the current time point /
        position, in a labels layer; reports the threshold used."""
        req = self._prepare()
        if req is None:
            return None
        layer = req["layer"]
        vol, kept, scale, index = current_volume(self.viewer, layer, keep=req["keep"], crop=req["crop"])
        rel = np.asarray(scale, float) / float(scale[-1] or 1.0)
        if req["planes"]:
            rel[0] = np.inf
        try:
            fg, info, _ = _classic.foreground(vol, req["settings"], rel, planes=req["planes"])
        except ValueError as e:
            self.status.setText(str(e))
            return None
        name = f"{layer.name} {PREVIEW}"
        translate = _crop_translate(layer, index)
        data = fg.astype(np.uint8)
        if name in self.viewer.layers:
            target = self.viewer.layers[name]
            target.data, target.scale, target.translate = data, scale, translate
        else:
            target = self.viewer.add_labels(data, name=name, scale=scale, translate=translate, opacity=0.5,
                                            metadata={"ibif_axes": kept, "ibif_preview": True})
        threshold = info.get("threshold")
        if isinstance(threshold, float) and not req["settings"].clahe:
            self.value.setValue(threshold)  # Manual starts from the automatic value
        shown = f"{threshold:.4g}" if isinstance(threshold, float) else threshold
        self.status.setText(f"Threshold {shown}: {info['foreground']:.1%} of the pixels are foreground "
                            f"(shown in {name}).")
        return target

    def run(self):
        if self._worker is not None:
            self._cancelling = True
            self._worker.quit()
            self.status.setText("Cancelling… (after the current time point / position)")
            return
        req = self._prepare()
        if req is None:
            return
        worker = self._run_current(req) if self.scope.currentText() == SCOPE_CURRENT else self._run_all(req)
        worker.yielded.connect(self._on_progress)
        worker.errored.connect(lambda e: self.status.setText(f"Segmentation failed: {e}"))
        worker.finished.connect(self._on_finished)
        self._worker = worker
        self.run_btn.setText("Cancel")
        self.preview_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.progress.setRange(0, 0)
        self.status.setText("Segmenting…")
        worker.start()
        return worker

    def _segment(self, vol, scale, req):
        if req["planes"]:
            labels, info = _classic.segment_planes(vol, scale, req["settings"])
        else:
            labels, info = _classic.segment(vol, scale, req["settings"])
        return keep_inside(labels, req["mask"]), info

    def _run_current(self, req):
        layer = req["layer"]
        vol, kept, scale, index = current_volume(self.viewer, layer, keep=req["keep"], crop=req["crop"])
        translate = _crop_translate(layer, index)
        tag = _index_tag(layer_axes(layer), index) + req["tag"]

        @thread_worker
        def work():
            yield f"Segmenting {vol.shape} ({'2D per plane' if req['planes'] else f'{vol.ndim}D'})…"
            return self._segment(vol, scale, req)

        worker = work()
        worker.returned.connect(self._add_labels_cb(layer, scale, translate, kept, tag, req))
        return worker

    def _run_all(self, req):
        layer, axes, crop = req["layer"], req["axes"], req["crop"]
        outer = [i for i, a in enumerate(axes) if a not in req["keep"]]
        outer_shape = [layer.data.shape[i] for i in outer]
        spatial = [slice(*crop[a]) if a in crop else slice(None) for a in axes]
        out_shape = [len(range(*s.indices(n))) for s, n in zip(spatial, layer.data.shape)]
        scale = [s for a, s in zip(axes, layer.scale) if a in req["keep"]]

        @thread_worker
        def work():
            labels = np.zeros(out_shape, dtype=np.uint32)
            positions = list(np.ndindex(*outer_shape))
            info = {}
            for n, pos in enumerate(positions):
                yield (n, len(positions))
                index, out_index = list(spatial), [slice(None)] * layer.ndim
                for i, p in zip(outer, pos):
                    index[i] = out_index[i] = p
                masks, info = self._segment(np.asarray(layer.data[tuple(index)]), scale, req)
                labels[tuple(out_index)] = masks
            return labels, info

        worker = work()
        worker.returned.connect(self._add_labels_cb(layer, list(layer.scale), _crop_translate(layer, spatial),
                                                    axes, "all" + req["tag"], req))
        return worker

    def _add_labels_cb(self, layer, scale, translate, axes, tag, req):
        def add(result):
            if result is None or self._cancelling:
                return
            labels, info = result
            settings = {k: v for k, v in vars(req["settings"]).items()}
            meta = {**(layer.metadata or {}), "ibif_axes": axes, "ibif_cellpose": True,
                    "ibif_segmentation": "threshold + watershed", "ibif_classic": settings}
            name = f"{layer.name} threshold {tag}".strip()
            self.viewer.add_labels(labels, name=name, scale=scale, translate=translate, metadata=meta)
            threshold = info.get("threshold")
            shown = f"threshold {threshold:.4g}, " if isinstance(threshold, float) else ""
            self.status.setText(f"{name}: {shown}{int(labels.max())} objects (max label).")

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
        self.run_btn.setText("Run")
        self.preview_btn.setEnabled(True)
        self.progress.setVisible(False)
        if self._cancelling:
            self._cancelling = False
            self.status.setText("Cancelled.")


def _crop_translate(layer, index):
    return [t + (i.start or 0) * s for t, i, s in zip(layer.translate, index, layer.scale)
            if isinstance(i, slice)]


def _index_tag(axes, index):
    return " ".join(f"{a}{i}" for a, i in zip(axes, index) if not isinstance(i, slice) and a in "TP")


def _float_spin(lo, hi, value, step, tip, decimals=2):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setDecimals(decimals)
    s.setSingleStep(step)
    s.setValue(value)
    s.setToolTip(tip)
    return s


__all__ = ["ClassicWidget"]
