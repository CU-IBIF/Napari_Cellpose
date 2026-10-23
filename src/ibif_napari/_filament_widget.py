"""Filament Tracer panel: automatic and click-to-trace tracing of neurons, microtubules
and other filaments, with Imaris-style statistics and CSV/SWC export."""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from napari.layers import Image, Labels, Shapes
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QComboBox,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _filaments
from ._cellpose_widget import WHOLE_IMAGE, _crop_translate, _float_spin, _index_tag
from ._region import add_region_layer, region_crop
from ._util import current_data_index, current_volume, layer_axes

CLICK_SIZE, POINT_SIZE = 5, 6  # pixels
POINT_COLORS = {"root": "yellow", "branch": "magenta", "terminal": "cyan"}
TABLE_COLUMNS = [
    ("filament", "Filament"),
    ("length_um", "Length (µm)"),
    ("segments", "Segments"),
    ("branch_points", "Branch pts"),
    ("terminal_points", "Terminal pts"),
    ("mean_diameter_um", "Mean diam. (µm)"),
    ("max_branch_depth", "Max depth"),
]


@dataclass
class _Session:
    """The volume being traced (one time point/position, cropped to the region) and its
    layers. Coordinates inside are pixels of `volume`."""

    key: tuple
    image: Image
    volume: np.ndarray
    spacing: tuple
    translate: list
    tag: str
    mask: np.ndarray | None = None
    enhanced: tuple | None = None  # ((d_min, d_max), array) of the last automatic run
    skeleton: Labels | None = None
    points: object = None
    clicks: object = None
    result: object = None


class FilamentWidget(QWidget):
    def __init__(self, napari_viewer):
        super().__init__()
        self.viewer = napari_viewer
        self._worker = None
        self._cancelling = False
        self._session = None
        self._region_syncs = []
        self._pending = []  # clicked points waiting to be traced
        self._last = None  # end of the current click-traced filament
        self._seen = 0  # clicks already handled
        self._undo = []  # (voxels added, previous chain end) per click trace
        self._dirty = False  # skeleton changed since the last measurement

        layout = QVBoxLayout(self)
        form = QFormLayout()
        layout.addLayout(form)

        self.image = QComboBox()
        form.addRow("Image", self.image)

        d_row = QHBoxLayout()
        self.d_min = _float_spin(0.01, 100, 0.3, 0.1, "Thinnest filament to find (µm across)")
        self.d_max = _float_spin(0.01, 100, 1.5, 0.1, "Thickest filament to find (µm across)")
        d_row.addWidget(self.d_min)
        d_row.addWidget(QLabel("to"))
        d_row.addWidget(self.d_max)
        form.addRow("Diameter (µm)", d_row)

        self.threshold = _float_spin(0, 1, 0, 0.02, "Filament response needed (0..1); auto = Otsu")
        self.threshold.setSpecialValueText("auto")
        form.addRow("Threshold", self.threshold)

        self.prune = _float_spin(0, 1000, 1.0, 0.5, "Trim side branches shorter than this (µm)")
        form.addRow("Prune spurs < (µm)", self.prune)

        self.min_length = _float_spin(0, 10000, 3.0, 1, "Drop filaments shorter than this (µm)")
        form.addRow("Min length (µm)", self.min_length)

        self.sholl_step = _float_spin(0, 1000, 1.0, 0.5, "Radius step of the Sholl analysis (µm)")
        form.addRow("Sholl step (µm)", self.sholl_step)

        region_row = QHBoxLayout()
        self.region = QComboBox()
        self.region.setToolTip("Trace only this area: a Labels layer (painted) or a Shapes rectangle")
        draw = QPushButton("Draw…")
        draw.setToolTip("Add a layer to paint the area to trace (applies to all Z planes)")
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

        self.run_btn = QPushButton("Trace automatically")
        self.run_btn.clicked.connect(self.run)
        layout.addWidget(self.run_btn)

        click_row = QHBoxLayout()
        self.click_btn = QPushButton("Click to trace")
        self.click_btn.setCheckable(True)
        self.click_btn.setToolTip(
            "Click along a filament on the image (2D view): each click traces the "
            "brightest path from the previous one"
        )
        self.click_btn.toggled.connect(self._toggle_click_mode)
        self.new_btn = QPushButton("New filament")
        self.new_btn.setToolTip("Start the next click somewhere new instead of continuing")
        self.new_btn.clicked.connect(self.new_filament)
        self.undo_btn = QPushButton("Undo last")
        self.undo_btn.clicked.connect(self.undo_last)
        for b in (self.click_btn, self.new_btn, self.undo_btn):
            click_row.addWidget(b)
        layout.addLayout(click_row)

        out_row = QHBoxLayout()
        self.measure_btn = QPushButton("Measure")
        self.measure_btn.setToolTip("Recompute the statistics after editing the filaments layer")
        self.measure_btn.clicked.connect(self.measure)
        self.export_btn = QPushButton("Export…")
        self.export_btn.setToolTip("Save the statistics as CSV files and the tracing as SWC")
        self.export_btn.clicked.connect(self.export)
        out_row.addWidget(self.measure_btn)
        out_row.addWidget(self.export_btn)
        layout.addLayout(out_row)

        self.progress = QProgressBar()
        self.progress.setVisible(False)
        layout.addWidget(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)
        self.table = QTableWidget(0, len(TABLE_COLUMNS))
        self.table.setHorizontalHeaderLabels([h for _, h in TABLE_COLUMNS])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        layout.addWidget(self.table, 1)

        self.image.currentTextChanged.connect(self._on_image_changed)
        self.z_first.valueChanged.connect(self._sync_region_ranges)
        self.z_last.valueChanged.connect(self._sync_region_ranges)
        self.viewer.layers.events.inserted.connect(self._refresh_layers)
        self.viewer.layers.events.removed.connect(self._refresh_layers)
        self.viewer.layers.events.reordered.connect(self._refresh_layers)
        self._refresh_layers()
        self._update_buttons()

    # ---- layer choices ----------------------------------------------------------------

    def _layer(self, name):
        return self.viewer.layers[name] if name and name in self.viewer.layers else None

    def _refresh_layers(self, event=None):
        images = [l.name for l in self.viewer.layers if isinstance(l, Image) and not l.rgb]
        regions = [
            l.name
            for l in self.viewer.layers
            if isinstance(l, (Labels, Shapes))
            and not {"ibif_filaments", "ibif_cellpose"} & set(l.metadata or {})
        ]
        for combo, names in ((self.image, images), (self.region, [WHOLE_IMAGE, *regions])):
            current = combo.currentText()
            combo.blockSignals(True)
            combo.clear()
            combo.addItems(names)
            if current in names:
                combo.setCurrentText(current)
            combo.blockSignals(False)
        self._on_image_changed()

    def _on_image_changed(self, *_):
        layer = self._layer(self.image.currentText())
        if layer is None:
            return
        axes = layer_axes(layer)
        nz = layer.data.shape[axes.index("Z")] if "Z" in axes else 1
        if self.z_last.maximum() != nz - 1 or self.z_last.value() > nz - 1:
            for spin in (self.z_first, self.z_last):
                spin.setRange(0, nz - 1)
            self.z_first.setValue(0)
            self.z_last.setValue(nz - 1)
        self.z_first.setEnabled("Z" in axes)
        self.z_last.setEnabled("Z" in axes)

    def add_region_layer(self):
        layer = self._layer(self.image.currentText())
        if layer is None:
            self.status.setText("Choose an image layer first.")
            return
        region, sync = add_region_layer(
            self.viewer, layer, layer_axes(layer), (self.z_first.value(), self.z_last.value()),
            self._grow_z_range,
        )
        if sync is not None:
            self._region_syncs.append(sync)
        self.viewer.layers.selection.active = region
        self.region.setCurrentText(region.name)
        self.status.setText("Paint over the filaments to trace, then trace. Only the bounding box "
                            "of the painted area is used.")

    def _sync_region_ranges(self, *_):
        self._region_syncs = [s for s in self._region_syncs if s.layer in self.viewer.layers]
        for sync in self._region_syncs:
            sync.set_range(self.z_first.value(), self.z_last.value())

    def _grow_z_range(self, first, last):
        self.z_first.setValue(min(first, self.z_first.value()))
        self.z_last.setValue(max(last, self.z_last.value()))

    # ---- the volume being traced ------------------------------------------------------

    def _current_session(self):
        """The session for the chosen image, time point/position and region (a new one if
        any of them changed), or None with a message in the status line."""
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
            self.status.setText("Nothing to trace: the region is empty or outside the image, "
                                "or the Z range is empty.")
            return None
        at = [i for a, i in zip(axes, current_data_index(self.viewer, layer)) if a not in keep]
        key = (id(layer), tuple(at), tuple(sorted(crop.items())))
        s = self._session
        if s is not None and s.key == key and s.image in self.viewer.layers:
            return s
        vol, kept, scale, index = current_volume(self.viewer, layer, keep=keep, crop=crop)
        tag = " ".join(t for t in (_index_tag(axes, index), "region" if region is not None else "") if t)
        self._session = _Session(
            key=key, image=layer, volume=vol, spacing=tuple(float(v) for v in scale),
            translate=_crop_translate(layer, index), tag=tag,
        )
        self._reset_chain()
        self._undo.clear()
        return self._session

    def _layer_kwargs(self, s, name):
        meta = {"ibif_filaments": True, "ibif_axes": "ZYX"[-s.volume.ndim:]}
        return dict(name=f"{s.image.name} {name} {s.tag}".strip(), scale=s.spacing,
                    translate=s.translate, metadata=meta)

    def _ensure_skeleton(self, s):
        if s.skeleton is None or s.skeleton not in self.viewer.layers:
            s.skeleton = self.viewer.add_labels(
                np.zeros(s.volume.shape, np.uint32), **self._layer_kwargs(s, "filaments")
            )
        return s.skeleton

    # ---- automatic tracing ------------------------------------------------------------

    def _params(self):
        return dict(
            d_min=self.d_min.value(),
            d_max=self.d_max.value(),
            threshold=self.threshold.value() or None,
            prune=self.prune.value(),
            min_length=self.min_length.value(),
            sholl_step=self.sholl_step.value(),
        )

    def run(self):
        if self._worker is not None:
            if self.run_btn.text() == "Cancel":
                self._cancelling = True
                self._worker.quit()
                self.status.setText("Cancelling…")
            else:
                self.status.setText("Busy tracing a click; try again in a moment.")
            return
        s = self._current_session()
        if s is None:
            return
        p = self._params()
        scales = (p["d_min"], p["d_max"])
        cached = s.enhanced[1] if s.enhanced and s.enhanced[0] == scales else None
        vol, sp = s.volume, s.spacing

        @thread_worker
        def work():
            enh = cached
            if enh is None:
                yield f"Finding filaments in {vol.shape} …"
                enh = yield from _filaments.iter_enhance(vol, sp, *scales)
            yield "Thresholding and skeletonizing…"
            mask, thr = _filaments.segment(enh, p["threshold"], sp, p["d_max"])
            yield None
            skel = _filaments.skeleton_from_mask(mask, sp, p["prune"], p["min_length"])
            yield "Measuring…"
            result = _filaments.measure(skel, sp, vol, mask, s.translate, p["sholl_step"])
            return enh, mask, thr, result

        def done(out):
            if out is None or self._cancelling:
                return
            enh, mask, thr, result = out
            s.enhanced, s.mask = (scales, enh), mask
            self._undo.clear()
            self._reset_chain()
            self._show_result(s, result)
            how = "auto" if p["threshold"] is None else "set"
            self.status.setText(f"{self._summary(result)} Threshold {thr:.2f} ({how}): change it "
                                "and trace again to find more or fewer filaments; the filtering "
                                "is reused.")

        self._start(work(), done, "Tracing failed", cancel=True)

    # ---- click to trace ---------------------------------------------------------------

    def _toggle_click_mode(self, on):
        if not on:
            if self._session is not None and self._session.clicks in self.viewer.layers:
                self._session.clicks.mode = "pan_zoom"
            return
        s = self._current_session()
        if s is None:
            self.click_btn.setChecked(False)
            return
        self._ensure_skeleton(s)
        if s.clicks is None or s.clicks not in self.viewer.layers:
            s.clicks = self.viewer.add_points(
                np.zeros((0, s.volume.ndim)), size=CLICK_SIZE, face_color="orange", **self._layer_kwargs(s, "trace clicks")
            )
            s.clicks.events.data.connect(self._on_clicks_changed)
            self._seen = 0
        self.viewer.layers.selection.active = s.clicks
        s.clicks.mode = "add"
        self.status.setText(
            "Click along a filament (in the 2D view, on the plane where it is in focus). "
            "Each click traces from the previous one; clicks near a traced filament join it. "
            "New filament starts a separate one."
        )

    def _on_clicks_changed(self, event=None):
        s = self._session
        if s is None or s.clicks is None:
            return
        data = np.asarray(s.clicks.data)
        if len(data) < self._seen:  # points deleted
            self._seen = len(data)
            return
        self._pending.extend(list(data[self._seen:]))
        self._seen = len(data)
        self._next_click()

    def _next_click(self):
        s = self._session
        if self._worker is not None or s is None:
            return
        if not self._pending:
            if self._dirty:
                self.measure()
            return
        point = self._pending.pop(0)
        skeleton = np.asarray(s.skeleton.data) if s.skeleton is not None else None
        target = _filaments.snap(point, s.volume, skeleton, s.spacing,
                                 radius_um=max(self.d_max.value(), 3 * min(s.spacing[-2:])))
        if self._last is None:
            self._last = target
            self._next_click()
            return
        start, p = self._last, self._params()
        vol, sp = s.volume, s.spacing

        @thread_worker
        def work():
            return _filaments.trace_path(vol, sp, start, target, p["d_min"], p["d_max"], p["threshold"])

        def done(out):
            path, lo, mask_box = out
            skel = self._ensure_skeleton(s)
            data = skel.data
            idx = tuple(path.T)
            new = data[idx] == 0
            added = tuple(i[new] for i in idx)
            data[added] = max(int(data.max()), 0) + 1 if data.size else 1
            skel.refresh()
            if s.mask is None:
                s.mask = np.zeros(vol.shape, bool)
            box = tuple(slice(a, a + n) for a, n in zip(lo, mask_box.shape))
            s.mask[box] |= mask_box
            self._undo.append((added, start))
            self._last = path[-1]
            self._dirty = True
            self.status.setText(f"Traced {len(path)} voxels.")

        self._start(work(), done, "Tracing the path failed")

    def new_filament(self):
        self._reset_chain()
        s = self._session
        if s is not None and s.clicks is not None and s.clicks in self.viewer.layers:
            s.clicks.data = np.zeros((0, s.volume.ndim))
            self._seen = 0
        self.status.setText("The next click starts a new filament.")

    def _reset_chain(self):
        self._last = None
        self._pending.clear()

    def undo_last(self):
        s = self._session
        if not self._undo or s is None or s.skeleton is None:
            self.status.setText("Nothing to undo.")
            return
        added, start = self._undo.pop()
        s.skeleton.data[added] = 0
        s.skeleton.refresh()
        self._last = start
        self._dirty = True
        self.measure()

    # ---- measurements and export ------------------------------------------------------

    def measure(self):
        s = self._session
        if s is None or s.skeleton is None or s.skeleton not in self.viewer.layers:
            self.status.setText("Trace some filaments first.")
            return
        if self._worker is not None:
            self._dirty = True  # measured when the current step finishes
            return
        skel = np.asarray(s.skeleton.data) != 0
        mask, vol, sp, step = s.mask, s.volume, s.spacing, self.sholl_step.value()

        @thread_worker
        def work():
            return _filaments.measure(skel, sp, vol, mask, s.translate, step)

        def done(result):
            self._show_result(s, result)
            self.status.setText(self._summary(result))

        self._dirty = False
        self.status.setText("Measuring…")
        self._start(work(), done, "Measuring failed")

    def _summary(self, result):
        f = result.filaments
        if not len(f):
            return "No filaments found."
        return (f"{len(f)} filament(s), total length {f['length_um'].sum():.1f} µm, "
                f"{int(f['branch_points'].sum())} branch points.")

    def _show_result(self, s, result):
        s.result = result
        skel = self._ensure_skeleton(s)
        skel.data = result.labels
        try:
            skel.features = result.filaments.assign(index=result.filaments["filament"])
        except Exception:  # older napari: statistics stay in the table
            pass
        nd = s.volume.ndim
        pts = result.points
        coords = pts[[f"_{a}_px" for a in "zyx"[-nd:]]].to_numpy() if len(pts) else np.zeros((0, nd))
        colors = [POINT_COLORS.get(t, "white") for t in pts["type"]] if len(pts) else "white"
        features = pts[["filament", "type"]] if len(pts) else None
        if s.points is None or s.points not in self.viewer.layers:
            s.points = self.viewer.add_points(
                coords, size=POINT_SIZE, face_color=colors, features=features,
                **self._layer_kwargs(s, "filament points"),
            )
        else:
            s.points.data = coords
            if features is not None:
                s.points.features = features
                s.points.face_color = colors
        if s.clicks is not None and s.clicks in self.viewer.layers:
            self.viewer.layers.selection.active = s.clicks  # keep clicking
        self._fill_table(result.filaments)

    def _fill_table(self, filaments):
        self.table.setRowCount(len(filaments))
        for r, (_, row) in enumerate(filaments.iterrows()):
            for c, (col, _) in enumerate(TABLE_COLUMNS):
                v = row.get(col)
                text = "" if v is None or (isinstance(v, float) and np.isnan(v)) else (
                    f"{v:.2f}" if isinstance(v, float) else str(int(v)))
                self.table.setItem(r, c, QTableWidgetItem(text))
        self.table.resizeColumnsToContents()

    def export(self, base=None):
        s = self._session
        if s is None or s.result is None:
            self.status.setText("Trace and measure some filaments first.")
            return None
        if not base:
            path, _ = QFileDialog.getSaveFileName(
                self, "Export filament statistics", f"{s.image.name}.csv", "CSV files (*.csv)"
            )
            if not path:
                return None
            base = path[:-4] if path.lower().endswith(".csv") else path
        written = _filaments.export(s.result, base)
        self.status.setText("Saved " + ", ".join(p.rsplit("/", 1)[-1] for p in written))
        return written

    # ---- workers ----------------------------------------------------------------------

    def _start(self, worker, done, what, cancel=False):
        worker.returned.connect(done)
        worker.errored.connect(lambda e: self._show_error(what, e))
        if hasattr(worker, "yielded"):
            worker.yielded.connect(self._on_progress)
        worker.finished.connect(self._on_finished)
        self._worker = worker
        if cancel:
            self.run_btn.setText("Cancel")
            self.progress.setVisible(True)
            self.progress.setRange(0, 0)
        self._update_buttons()
        worker.start()

    def _on_progress(self, value):
        if value is None or self._cancelling:
            return
        if isinstance(value, str):
            self.status.setText(value)
            return
        done, total = value
        self.progress.setRange(0, total)
        self.progress.setValue(done)

    def _show_error(self, what, error):
        logging.getLogger("ibif_napari").error("%s: %s", what, error)
        self.status.setText(f"{what}: {error}")

    def _on_finished(self):
        self._worker = None
        self.run_btn.setText("Trace automatically")
        self.progress.setVisible(False)
        if self._cancelling:
            self._cancelling = False
            self.status.setText("Cancelled.")
        self._update_buttons()
        self._next_click()

    def _update_buttons(self):
        busy = self._worker is not None
        for b in (self.measure_btn, self.export_btn, self.undo_btn):
            b.setEnabled(not busy)
        self.run_btn.setEnabled(not busy or self.run_btn.text() == "Cancel")
