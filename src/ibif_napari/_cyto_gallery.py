"""The Cytometry gallery: thumbnails of the objects in a gate or phenotype.

A window of its own (Gates > Gallery…, or right-click a gate or phenotype), after
VTEA's Gallery View (Java `vtea.exploration.gallery`, ported to napari in
vtea-python's `vtea_napari.widgets.gallery`), written again here for IBIF napari's
layers and levels. A dot on the plot is a number; the gallery shows the cells behind
it, which is how a gate is checked and how training examples are chosen.

- **Objects**: any population (all, a gate, an image gate, a phenotype). With
  *Follow the lists* on, selecting a gate or phenotype in the workbench shows it.
  Up to *Show* objects, the first ones or a random sample.
- **Channels**: up to three image layers, each in its own colour, composited; every
  thumbnail is scaled by the layer's contrast limits in the viewer, so the grid is
  comparable. 3D objects are a maximum projection of a few planes around their
  centre, not of the whole stack.
- **Outline**: the object's mask on top, in a colour (the population's by default)
  with an optional fill.
- Hover a thumbnail to see it enlarged; click it to go to the object in the viewer.
- **Training**: select thumbnails (Ctrl / Shift-click) and *Label as* a class of the
  training set (Classification > Train a model…), or label every object shown.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import QPoint, QSize, Qt
from qtpy.QtGui import QColor, QCursor, QIcon, QImage, QPixmap
from qtpy.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QColorDialog,
    QComboBox,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QListView,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from . import _cytometry as cy

LUTS = {"Grey": (1, 1, 1), "Red": (1, 0, 0), "Green": (0, 1, 0), "Blue": (0, 0.45, 1),
        "Magenta": (1, 0, 1), "Cyan": (0, 1, 1), "Yellow": (1, 1, 0)}
DEFAULT_LUTS = ("Grey", "Green", "Magenta")
SIZES = (48, 64, 96, 128)
NONE = "(none)"


# ---- pictures of objects ------------------------------------------------------------------


def _object_window(state, layer, row, radius, z_radius):
    """(leading index, spatial slices, (y0, x0) offset in the window, n spatial axes)
    of a window of `radius` pixels around table row `row`'s centre in `layer`, or None."""
    table = state.table
    cols = state.centroid_columns()
    if layer is None or "Y" not in cols or "X" not in cols:
        return None
    rec = table.iloc[row]
    data = layer.data
    n = min(cy._spatial_ndim(layer), 3)
    lead = data.ndim - n
    keys = [k for k in table.columns if cy._is_key(k) and k != "label"][:lead]
    pos = tuple(int(rec[k]) for k in keys) + (0,) * (lead - len(keys))
    scale, translate = np.asarray(layer.scale[-n:], float), np.asarray(layer.translate[-n:], float)
    axes = "ZYX"[-n:]
    centre = []
    for i, a in enumerate(axes):
        world = float(rec[cols[a]]) if a in cols else translate[i]
        centre.append(int(round((world - translate[i]) / scale[i])))
    slices = []
    for i, a in enumerate(axes):
        extent = data.shape[lead + i]
        r = z_radius if a == "Z" else radius
        start = centre[i] - r
        slices.append(slice(max(start, 0), min(centre[i] + r + (1 if a == "Z" else 0), extent)))
    offset = (max(radius - centre[-2], 0), max(radius - centre[-1], 0))
    return pos, tuple(slices), offset, n


def _plane(layer, window):
    """The window of `layer` as one 2D plane (a max projection over Z), as floats."""
    pos, slices, _, n = window
    try:
        block = np.asarray(layer.data[pos + slices], dtype=float)
    except (IndexError, ValueError, TypeError):
        return None
    return block.max(axis=0) if n == 3 and block.ndim == 3 else block


def _place(plane, offset, side):
    """`plane` put into a (side, side) canvas at `offset` (where the window ran off the
    image, the canvas stays empty)."""
    out = np.zeros((side, side), dtype=float)
    if plane is None:
        return out
    y0, x0 = offset
    h, w = min(plane.shape[0], side - y0), min(plane.shape[1], side - x0)
    out[y0:y0 + h, x0:x0 + w] = plane[:h, :w]
    return out


def object_picture(state, row, channels, *, size=64, radius=None, z_radius=2, outline=None,
                   fill=0.0):
    """An RGB (size, size, 3) uint8 picture of table row `row`'s object.

    `channels`: [(image layer, colour name or RGB)], composited additively, each
    scaled by its layer's contrast limits. `radius`: pixels around the centre (of the
    first channel's layer; default size // 2, i.e. no rescaling). `outline`: a colour
    for the object's mask edge (segmented objects only), `fill` the opacity of the mask
    inside."""
    radius = int(radius or size // 2)
    side = 2 * radius
    rgb = np.zeros((side, side, 3), dtype=float)
    first = None
    for layer, lut in channels:
        if layer is None:
            continue
        window = _object_window(state, layer, row, radius, z_radius)
        if window is None:
            continue
        first = first or (layer, window)
        plane = _place(_plane(layer, window), window[2], side)
        lo, hi = layer.contrast_limits if getattr(layer, "contrast_limits", None) else (plane.min(), plane.max())
        value = np.clip((plane - lo) / ((hi - lo) or 1), 0, 1)
        colour = np.asarray(LUTS.get(lut, lut) if isinstance(lut, str) else lut, dtype=float)
        rgb += value[..., None] * colour
    rgb = np.clip(rgb, 0, 1)
    labels = state.labels_layer()
    if outline and first is not None and "label" in state.table.columns and labels is not None:
        layer, window = first
        if labels.data.shape[-window[3]:] == layer.data.shape[-window[3]:]:
            plane = _plane(labels, window)
            if plane is not None:
                from scipy import ndimage as ndi

                mask = _place(plane == int(state.table.iloc[row]["label"]), window[2], side) > 0
                colour = np.asarray(_rgb(outline), dtype=float)
                if fill:
                    rgb[mask] = (1 - fill) * rgb[mask] + fill * colour
                rgb[mask & ~ndi.binary_erosion(mask)] = colour
    out = (rgb * 255).astype(np.uint8)
    if side != size:  # nearest neighbour, so the pixels stay honest
        index = (np.arange(size) * side / size).astype(int)
        out = out[index][:, index]
    return out


def _rgb(colour):
    """'#rrggbb' -> (r, g, b) in 0..1."""
    q = QColor(colour)
    return q.redF(), q.greenF(), q.blueF()


def pixmap(rgb):
    rgb = np.ascontiguousarray(rgb)
    h, w, _ = rgb.shape
    return QPixmap.fromImage(QImage(rgb.data, w, h, 3 * w, QImage.Format_RGB888).copy())


# ---- the window -----------------------------------------------------------------------------


class GalleryWindow(QWidget):
    def __init__(self, bench, parent=None):
        super().__init__(parent or bench, Qt.Window)
        self.setWindowTitle("Cytometry gallery")
        self.bench = bench
        self.state = bench.state
        self.viewer = bench.viewer
        self._rows = []
        self._outline_colour = None  # None: the population's colour
        layout = QVBoxLayout(self)

        form = QFormLayout()
        top = QHBoxLayout()
        self.population = QComboBox()
        self.population.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.population.setMinimumContentsLength(18)
        self.population.currentIndexChanged.connect(lambda _: self.fill())
        top.addWidget(self.population, 1)
        self.follow = QCheckBox("Follow the lists")
        self.follow.setChecked(True)
        self.follow.setToolTip("Show the gate or phenotype selected in the workbench's lists")
        top.addWidget(self.follow)
        form.addRow("Objects", top)
        self.channels = []
        row = QHBoxLayout()
        for k in range(3):
            layer = QComboBox()
            layer.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            layer.setMinimumContentsLength(8)
            lut = QComboBox()
            lut.addItems(list(LUTS))
            lut.setCurrentText(DEFAULT_LUTS[k])
            layer.currentIndexChanged.connect(lambda _: self.fill())
            lut.currentIndexChanged.connect(lambda _: self.fill())
            row.addWidget(layer, 2)
            row.addWidget(lut, 1)
            self.channels.append((layer, lut))
        form.addRow("Channels", row)
        row = QHBoxLayout()
        self.outline = QCheckBox("Outline")
        self.outline.setChecked(True)
        self.outline.toggled.connect(lambda _: self.fill())
        self.outline_btn = QPushButton()
        self.outline_btn.setToolTip("Outline colour (default: the gate's or phenotype's)")
        self.outline_btn.clicked.connect(lambda _=False: self.pick_outline())
        self.fill_slider = QSlider(Qt.Horizontal)
        self.fill_slider.setRange(0, 80)
        self.fill_slider.setToolTip("Fill the object's mask (opacity)")
        self.fill_slider.valueChanged.connect(lambda _: self.fill())
        row.addWidget(self.outline)
        row.addWidget(self.outline_btn)
        row.addWidget(QLabel("Fill"))
        row.addWidget(self.fill_slider, 1)
        form.addRow("Mask", row)
        row = QHBoxLayout()
        self.size = QComboBox()
        for s in SIZES:
            self.size.addItem(f"{s} px", s)
        self.size.setCurrentIndex(1)
        self.size.currentIndexChanged.connect(lambda _: self.fill())
        self.radius = QSpinBox()
        self.radius.setRange(4, 200)
        self.radius.setValue(20)
        self.radius.setSuffix(" px")
        self.radius.setToolTip("How far around each object's centre to show (image pixels)")
        self.radius.valueChanged.connect(lambda _: self.fill())
        self.limit = QSpinBox()
        self.limit.setRange(1, 2000)
        self.limit.setValue(120)
        self.limit.setToolTip("At most this many objects")
        self.limit.valueChanged.connect(lambda _: self.fill())
        self.sample = QCheckBox("Random")
        self.sample.setToolTip("A random sample (the same one until the objects change), not the first ones")
        self.sample.toggled.connect(lambda _: self.fill())
        for w in (QLabel("Size"), self.size, QLabel("Around"), self.radius, QLabel("Show"), self.limit,
                  self.sample):
            row.addWidget(w)
        row.addStretch(1)
        form.addRow("", row)
        layout.addLayout(form)

        self.grid = QListWidget()
        self.grid.setViewMode(QListView.IconMode)
        self.grid.setResizeMode(QListView.Adjust)
        self.grid.setMovement(QListView.Static)
        self.grid.setSpacing(3)
        self.grid.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.grid.setMouseTracking(True)
        self.grid.itemClicked.connect(self._clicked)
        self.grid.itemEntered.connect(self._hover)
        self.grid.viewport().installEventFilter(self)
        layout.addWidget(self.grid, 1)

        row = QHBoxLayout()
        self.count = QLabel()
        row.addWidget(self.count, 1)
        row.addWidget(QLabel("Label as"))
        self.train_class = QComboBox()
        self.train_class.setEditable(True)
        self.train_class.setToolTip("A class of the training set (type a new name to add one)")
        self.train_class.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.train_class.setMinimumContentsLength(10)
        row.addWidget(self.train_class)
        self.label_sel_btn = QPushButton("Selected")
        self.label_sel_btn.setToolTip("Add the selected thumbnails' objects to the training set as this class")
        self.label_sel_btn.clicked.connect(lambda _=False: self.label(selected=True))
        self.label_all_btn = QPushButton("All shown")
        self.label_all_btn.clicked.connect(lambda _=False: self.label(selected=False))
        row.addWidget(self.label_sel_btn)
        row.addWidget(self.label_all_btn)
        layout.addLayout(row)
        self.status = QLabel()
        self.status.setWordWrap(True)
        layout.addWidget(self.status)

        self._zoom = QLabel(None, Qt.ToolTip)  # one popup, reused, for the enlarged view
        self._zoom.hide()
        self.state.listen(self.refresh)
        self.state.listen_training(self._fill_classes)
        self.refresh()
        self.resize(760, 640)

    # ---- choices -----------------------------------------------------------------------------

    def refresh(self):
        from napari.layers import Image

        from ._cyto_workbench import _fill, _pop_key

        state = self.state
        keep = self.population.currentData()
        _fill(self.population, [(label.strip(), _pop_key(kind, key)) for kind, key, label in state.populations()]
              if state.table is not None else [], keep)
        names = [layer.name for layer in self.viewer.layers if isinstance(layer, Image)]
        for k, (box, _) in enumerate(self.channels):
            keep = box.currentText()
            box.blockSignals(True)
            box.clear()
            box.addItem(NONE)
            box.addItems(names)
            if keep and box.findText(keep) >= 0:
                box.setCurrentText(keep)
            elif k < len(names) and keep == "":
                box.setCurrentText(names[k])
            box.blockSignals(False)
        self._fill_classes()
        self._show_outline_colour()
        self.fill()

    def _fill_classes(self):
        keep = self.train_class.currentText()
        self.train_class.blockSignals(True)
        self.train_class.clear()
        self.train_class.addItems(list(self.state.training_classes))
        self.train_class.setCurrentText(keep)
        self.train_class.blockSignals(False)

    def show_population(self, kind, key=None):
        """Show a population (as the Objects box does; also used by the workbench)."""
        from ._cyto_workbench import _pop_key

        index = self.population.findData(_pop_key(kind, key))
        if index >= 0:
            self.population.setCurrentIndex(index)
        return index >= 0

    def follow_selection(self, items):
        """The workbench's lists changed: show the first selected gate or phenotype."""
        if self.follow.isChecked() and items and self.isVisible():
            self.show_population(*items[0])

    def current(self):
        from ._cyto_workbench import _pop_parse

        return _pop_parse(self.population.currentData())

    def outline_colour(self):
        if self._outline_colour:
            return self._outline_colour
        kind, key = self.current()
        try:
            return self.state.population_color(kind, key) if kind != "all" else "#ffff00"
        except (KeyError, AttributeError):
            return "#ffff00"

    def pick_outline(self, colour=None):
        if colour is None:
            picked = QColorDialog.getColor(QColor(self.outline_colour()), self, "Outline colour")
            if not picked.isValid():
                return
            colour = picked.name()
        self._outline_colour = colour
        self._show_outline_colour()
        self.fill()

    def _show_outline_colour(self):
        c = self.outline_colour()
        self.outline_btn.setStyleSheet(f"background-color: {c}; min-width: 24px;")

    # ---- the grid ------------------------------------------------------------------------------

    def rows(self):
        """The table rows shown: the population's objects, the first `limit` or a
        random sample of them (fixed for a population, so redrawing keeps them)."""
        state = self.state
        if state.table is None:
            return []
        try:
            mask = state.mask(*self.current())
        except (KeyError, ValueError):
            return []
        rows = np.flatnonzero(mask)
        n = self.limit.value()
        if len(rows) > n and self.sample.isChecked():
            rows = np.sort(np.random.default_rng(len(rows)).choice(rows, n, replace=False))
        return [int(r) for r in rows[:n]], len(np.flatnonzero(mask))

    def channel_layers(self):
        out = []
        for box, lut in self.channels:
            name = box.currentText()
            if name and name != NONE and name in self.viewer.layers:
                out.append((self.viewer.layers[name], lut.currentText()))
        return out

    def fill(self):
        self.grid.clear()
        self._show_outline_colour()
        result = self.rows()
        if not result:
            self.count.setText("No objects.")
            self._rows = []
            return
        rows, total = result
        self._rows = rows
        size = self.size.currentData() or 64
        self.grid.setIconSize(QSize(size, size))
        channels = self.channel_layers()
        outline = self.outline_colour() if self.outline.isChecked() else None
        fill = self.fill_slider.value() / 100
        for row in rows:
            picture = object_picture(self.state, row, channels, size=size, radius=self.radius.value(),
                                     outline=outline, fill=fill)
            item = QListWidgetItem()
            item.setIcon(QIcon(pixmap(picture)))
            item.setData(Qt.UserRole, row)
            klass = self.state.training.get(row)
            item.setToolTip(f"Object {self.state.object_id(row)}" + (f" — training: {klass}" if klass else ""))
            self.grid.addItem(item)
        shown = f"{len(rows):,} of {total:,} objects" if total > len(rows) else f"{total:,} objects"
        self.count.setText(shown + (" (a random sample)" if total > len(rows) and self.sample.isChecked() else ""))
        if not channels:
            self.status.setText("Choose an image layer under Channels to see the objects.")

    def _clicked(self, item):
        self.bench.go_to_row(int(item.data(Qt.UserRole)))

    def selected_rows(self):
        return [int(i.data(Qt.UserRole)) for i in self.grid.selectedItems()]

    def label(self, selected=True, cls=None):
        """Add the selected (or all shown) objects to the training set as `cls` (the
        Label as box). Returns how many were labelled."""
        cls = (cls or self.train_class.currentText()).strip()
        rows = self.selected_rows() if selected else list(self._rows)
        if not cls:
            self.status.setText("Type or choose a class under Label as.")
            return 0
        if not rows:
            self.status.setText("Select thumbnails first (Ctrl- or Shift-click for several)." if selected
                                else "No objects shown.")
            return 0
        n = self.state.label_objects(rows, cls)
        self.status.setText(f"{n:,} objects labelled {cls} in the training set ({len(rows) - n:,} already were).")
        return n

    # ---- the enlarged view --------------------------------------------------------------------

    def _hover(self, item):
        row = int(item.data(Qt.UserRole))
        size = 2 * (self.size.currentData() or 64)
        picture = object_picture(self.state, row, self.channel_layers(), size=size, radius=self.radius.value(),
                                 outline=self.outline_colour() if self.outline.isChecked() else None,
                                 fill=self.fill_slider.value() / 100)
        self._zoom.setPixmap(pixmap(picture))
        self._zoom.adjustSize()
        self._zoom.move(QCursor.pos() + QPoint(16, 16))
        self._zoom.show()

    def eventFilter(self, obj, event):
        from qtpy.QtCore import QEvent

        if event.type() == QEvent.Leave:
            self._zoom.hide()
        return super().eventFilter(obj, event)

    def hideEvent(self, event):
        self._zoom.hide()
        super().hideEvent(event)


def object_thumbnail(state, image, row, size=56, color=None):
    """A picture of one object in one image layer, at the image's own resolution,
    with its outline in `color` (the Training workbench's examples)."""
    return object_picture(state, row, [(image, "Grey")], size=size, radius=size // 2, outline=color)


def open_gallery(bench):
    """The workbench's gallery window, created on first use and raised."""
    window = getattr(bench, "_gallery_window", None)
    if window is None:
        window = GalleryWindow(bench)
        bench._gallery_window = window
    window.show()
    window.raise_()
    window.activateWindow()
    return window


__all__ = ["GalleryWindow", "open_gallery", "object_picture", "object_thumbnail", "LUTS"]
