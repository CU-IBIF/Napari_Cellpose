"""The Cytometry scatter plot: objects as points, gates as outlines, drawing new gates.

matplotlib in Qt: zoom and pan from its toolbar, gates drawn with its selectors
(rectangle, polygon, lasso) in data coordinates of the axes shown. The selectors blit:
while a gate is drawn only its outline is redrawn, not the (up to 200 000) points, so
the lasso follows the mouse. The plot keeps a 4:3 shape (`AspectBox`), and the
**Points** menu (toolbar, or right-click on the plot) sets the size (a preset, a size in
pt, or by a measurement), style and opacity of the points, and whether objects in an
image gate are ringed in the gate's colour.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QAction,
    QActionGroup,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QLabel,
    QLayout,
    QMenu,
    QRadioButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

MAX_POINTS = 200_000  # above this, a random subset is drawn (gates still use every object)
ASPECT = 4 / 3  # plot width / height
# Point sizes (matplotlib's s, in points²); None = by the number of points.
SIZES = {"Automatic": None, "Tiny": 0.5, "Small": 2.0, "Medium": 6.0, "Large": 14.0, "Huge": 30.0}
STYLES = {"Dot": "o", "Square": "s", "Diamond": "D", "Triangle": "^", "Plus": "+", "Cross": "x",
          "Pixel": ","}
LINE_MARKERS = {"+", "x"}  # drawn with lines, not filled
OPACITIES = {"100%": 1.0, "70%": 0.7, "40%": 0.4, "15%": 0.15}
SIZE_RANGE = (2.0, 8.0)  # default smallest / largest point diameter (pt) when sized by a measurement
CUSTOM_SIZE, SIZE_BY = "Custom size…", "Size by a measurement…"


def sizes_from(values, d_min, d_max):
    """Point areas (pt², matplotlib's s) for `values`: diameters from d_min to d_max pt
    over the 1st-99th percentile of the values (outside it: the end sizes; NaN: d_min)."""
    values = np.asarray(values, float)
    ok = np.isfinite(values)
    d = np.full(len(values), float(d_min))
    if ok.any():
        lo, hi = np.percentile(values[ok], [1, 99])
        t = np.clip((values[ok] - lo) / (hi - lo), 0, 1) if hi > lo else np.full(ok.sum(), 0.5)
        d[ok] = d_min + t * (d_max - d_min)
    return d ** 2


class AspectBox(QWidget):
    """Holds one widget at a fixed width:height (the largest such rectangle that fits,
    centred), so the plot stays 4:3 however the window is resized."""

    def __init__(self, child, aspect=ASPECT, parent=None):
        super().__init__(parent)
        self.child, self.aspect = child, aspect
        child.setParent(self)
        self.setMinimumSize(240, int(240 / aspect))

    def resizeEvent(self, event):
        w, h = self.width(), self.height()
        cw = min(w, int(h * self.aspect))
        ch = int(cw / self.aspect)
        self.child.setGeometry((w - cw) // 2, (h - ch) // 2, cw, ch)
        super().resizeEvent(event)


class ScatterPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        import matplotlib

        matplotlib.use("QtAgg", force=False)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(4, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        self.point_size = None  # None: by the number of points
        self.marker = "o"
        self.alpha = 1.0
        self.size_by = None  # a measurement the points are sized by
        self.size_range = SIZE_RANGE  # its smallest and largest diameter (pt)
        self.size_source = None  # name -> values per table row (set by the workbench)
        self.size_choices = None  # () -> names the points can be sized by
        self.outline_image_gates = True  # ring objects in an image gate
        self._last = None  # the last show() call, to redraw with a new point style
        self.on_style = None  # called after the point style changes
        self.points_menu = self._make_points_menu()
        button = QToolButton()
        button.setText("Points")
        button.setToolTip("Size, style and opacity of the points (also: right-click the plot)")
        button.setPopupMode(QToolButton.InstantPopup)
        button.setMenu(self.points_menu)
        self.toolbar.addSeparator()
        self.toolbar.addWidget(button)
        self.points_button = button
        self.canvas.setContextMenuPolicy(Qt.CustomContextMenu)
        self.canvas.customContextMenuRequested.connect(self._context_menu)
        self.frame = AspectBox(self.canvas)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.frame, 1)
        self._selector = None
        self._on_drawn = None
        self._on_pick = None
        self._index = np.zeros(0, int)  # row of the table for each drawn point
        self._xy = np.zeros((0, 2))
        self._highlight = None
        self.canvas.mpl_connect("button_press_event", self._clicked)

    # ---- data ----------------------------------------------------------------------------

    def show(self, x, y, index, colors=None, background=None, xlabel="", ylabel="", gates=(),
             legend=None, cbar_label=None, values=None, outlines=()):
        """Draw points (x, y) for table rows `index`; `background` (x, y) in light grey
        (objects outside the shown population); `gates` = [(vertices, color, name)];
        `legend` = [(label, color)] or `values` with a colour bar; `outlines` =
        [(name, mask over the table rows, color)]: those objects are ringed."""
        self._last = dict(x=x, y=y, index=index, colors=colors, background=background, xlabel=xlabel,
                          ylabel=ylabel, gates=gates, legend=legend, cbar_label=cbar_label, values=values,
                          outlines=outlines)
        self.cancel()
        self.figure.clf()
        ax = self.axes = self.figure.add_subplot(111)
        x, y, index = np.asarray(x, float), np.asarray(y, float), np.asarray(index, int)
        ok = np.isfinite(x) & np.isfinite(y)
        x, y, index = x[ok], y[ok], index[ok]
        if colors is not None and not isinstance(colors, str):
            colors = np.asarray(colors, dtype=object)[ok] if len(colors) == len(ok) else colors
        if values is not None:
            values = np.asarray(values, float)[ok]
        draw = np.arange(len(x))
        if len(x) > MAX_POINTS:
            draw = np.sort(np.random.default_rng(0).choice(len(x), MAX_POINTS, replace=False))
        size = self.point_size if self.point_size is not None else \
            6 if len(x) < 5000 else 2 if len(x) < 50000 else 0.5
        sized_by = self._size_values(index)
        point_s = sizes_from(sized_by[draw], *self.size_range) if sized_by is not None else size
        style = dict(marker=self.marker, alpha=self.alpha, rasterized=True,
                     linewidths=0.8 if self.marker in LINE_MARKERS else 0)
        if background is not None and len(background[0]):
            bx, by = np.asarray(background[0], float), np.asarray(background[1], float)
            back_s = self.size_range[0] ** 2 if sized_by is not None else size
            ax.scatter(bx, by, s=back_s, c="#d0d0d0", zorder=1, **style)
        if values is not None:
            sc = ax.scatter(x[draw], y[draw], s=point_s, c=values[draw], cmap="viridis", zorder=2, **style)
            self.figure.colorbar(sc, ax=ax, label=cbar_label or "")
        else:
            c = colors if colors is None or isinstance(colors, str) else list(np.asarray(colors)[draw])
            ax.scatter(x[draw], y[draw], s=point_s, c=c if c is not None else "#1f77b4", zorder=2, **style)
        ringed = []
        if self.outline_image_gates:
            for k, (name, members, color) in enumerate(outlines or ()):
                members = np.asarray(members, bool)
                if len(members) <= (index.max() if len(index) else -1):
                    continue
                hit = members[index[draw]]
                if not hit.any():
                    continue
                inner = np.sqrt(np.broadcast_to(point_s, hit.shape)[hit])
                # Several image gates: each ring a little wider, so all of them show.
                ring = (np.maximum(inner, 2.0) + 3.0 + 2.5 * k) ** 2
                ax.scatter(x[draw][hit], y[draw][hit], s=ring, facecolors="none", edgecolors=color,
                           linewidths=1.0, marker="o", zorder=2.5 + 0.01 * k, rasterized=True)
                ringed.append((name, color))
        for vertices, color, name in gates:
            v = np.asarray(vertices, float)
            closed = np.vstack([v, v[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.5, zorder=3)
            ax.annotate(name, v[np.argmax(v[:, 1])], color=color, fontsize=8, zorder=4,
                        xytext=(2, 2), textcoords="offset points")
        if legend or ringed:
            from matplotlib.lines import Line2D

            handles = [Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in legend or ()]
            handles += [Line2D([], [], marker="o", ls="", mfc="none", mec=c, label=f"image gate: {n}")
                        for n, c in ringed]
            ax.legend(handles=handles, fontsize=7, loc="best", markerscale=0.8, frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        title = f"{len(x):,} objects" + (f" (showing {len(draw):,})" if len(draw) < len(x) else "")
        if sized_by is not None:
            title += f"; size: {self.size_by}"
        ax.set_title(title, fontsize=9)
        self._index, self._xy = index, np.column_stack([x, y])
        self._highlight = None
        self.canvas.draw_idle()

    # ---- point style --------------------------------------------------------------------------

    def _make_points_menu(self):
        menu = QMenu("Points", self)
        self._actions = {}
        for title, options, attr in (("Size", SIZES, "point_size"), ("Style", STYLES, "marker"),
                                     ("Opacity", OPACITIES, "alpha")):
            sub = menu.addMenu(title)
            group = QActionGroup(sub)
            group.setExclusive(True)
            for label, value in options.items():
                action = QAction(label, sub)
                action.setCheckable(True)
                action.setChecked(value == getattr(self, attr))
                # A preset size also stops sizing the points by a measurement.
                extra = {"size_by": None} if attr == "point_size" else {}
                action.triggered.connect(
                    lambda _=False, a=attr, v=value, e=extra: self.set_point_style(**{a: v}, **e))
                group.addAction(action)
                sub.addAction(action)
                self._actions[(attr, label)] = action
            if attr == "point_size":
                sub.addSeparator()
                for label in (CUSTOM_SIZE, SIZE_BY):
                    action = QAction(label, sub)
                    action.setCheckable(True)
                    action.triggered.connect(lambda _=False, by=label == SIZE_BY: self.size_dialog(by))
                    group.addAction(action)
                    sub.addAction(action)
                    self._actions[("size", label)] = action
        menu.addSeparator()
        ring = QAction("Ring objects in image gates", menu)
        ring.setCheckable(True)
        ring.setChecked(self.outline_image_gates)
        ring.setToolTip("Objects in an image gate are ringed in the gate's colour (Colour… in the Gates list)")
        ring.toggled.connect(lambda on: self.set_point_style(outline_image_gates=on))
        menu.addAction(ring)
        self._actions[("outline", "ring")] = ring
        return menu

    def set_point_style(self, point_size=..., marker=..., alpha=..., size_by=..., size_range=...,
                        outline_image_gates=...):
        """Change the size (points², None = automatic), marker and/or opacity of the
        points, the measurement they are sized by (None = one size) and its diameter
        range in pt, or whether image-gated objects are ringed; and redraw."""
        for name, value in (("point_size", point_size), ("marker", marker), ("alpha", alpha),
                            ("size_by", size_by), ("size_range", size_range),
                            ("outline_image_gates", outline_image_gates)):
            if value is not ...:
                setattr(self, name, tuple(value) if name == "size_range" else value)
        for (attr, label), action in self._actions.items():
            action.blockSignals(True)
            if attr == "size":
                action.setChecked(self.size_by is not None if label == SIZE_BY else
                                  self.size_by is None and self.point_size not in SIZES.values())
            elif attr == "outline":
                action.setChecked(self.outline_image_gates)
            else:
                options = {"point_size": SIZES, "marker": STYLES, "alpha": OPACITIES}[attr]
                action.setChecked(options[label] == getattr(self, attr)
                                  and (attr != "point_size" or self.size_by is None))
            action.blockSignals(False)
        if self.on_style is not None:
            self.on_style()
        if self._last is not None:
            self.show(**self._last)

    def _size_values(self, index):
        """Values of the size measurement for table rows `index`, or None."""
        if self.size_by is None or self.size_source is None:
            return None
        try:
            values = np.asarray(self.size_source(self.size_by), float)
        except (KeyError, ValueError):
            return None
        if len(index) and index.max() >= len(values):
            return None
        return values[index]

    def size_dialog(self, by_measurement=None):
        """Open the point size dialog; apply what is chosen. Returns the dialog."""
        dialog = PointSizeDialog(self, by_measurement)
        if dialog.exec() == QDialog.Accepted:
            dialog.apply()
        else:
            self.set_point_style()  # re-tick the menu as it was
        return dialog

    def _context_menu(self, pos):
        if self._selector is not None or self.toolbar.mode:
            return  # drawing a gate, or zooming / panning
        self.points_menu.popup(self.canvas.mapToGlobal(pos))

    def highlight(self, row):
        """Ring the point of table row `row` (None to clear)."""
        if self._highlight is not None:
            self._highlight.remove()
            self._highlight = None
        if row is not None:
            hit = np.flatnonzero(self._index == row)
            if len(hit):
                x, y = self._xy[hit[0]]
                (self._highlight,) = self.axes.plot([x], [y], "o", ms=12, mfc="none", mec="red", mew=2, zorder=5)
        self.canvas.draw_idle()

    # ---- drawing gates ------------------------------------------------------------------------

    def draw_gate(self, mode, on_drawn):
        """Let the user draw a 'rectangle', 'polygon' or 'lasso'; calls on_drawn(vertices).

        The selectors blit: the points are drawn once, kept as a picture, and only the
        gate's outline is redrawn as the mouse moves. (Without blitting every mouse move
        redrew all the points, which made the lasso lag far behind the mouse.)"""
        from matplotlib.widgets import LassoSelector, PolygonSelector, RectangleSelector

        self.cancel()
        self._on_drawn = on_drawn
        props = dict(color="red", linewidth=1.5)
        if mode == "rectangle":
            self._selector = RectangleSelector(self.axes, self._rect_done, useblit=True, interactive=False,
                                               props=dict(edgecolor="red", fill=False, linewidth=1.5))
        elif mode == "polygon":
            self._selector = PolygonSelector(self.axes, self._poly_done, useblit=True, props=props)
        elif mode == "lasso":
            self._selector = LassoSelector(self.axes, self._poly_done, useblit=True, props=props)
        else:
            raise ValueError(mode)
        self.canvas.draw()  # the selector keeps this picture of the points (its draw_event)

    def drawing(self):
        return self._selector is not None

    def cancel(self):
        if self._selector is not None:
            self._selector.set_active(False)
            try:
                self._selector.disconnect_events()
            except Exception:
                pass
            self._selector = None
            self.canvas.draw_idle()

    def _rect_done(self, press, release):
        x0, x1 = sorted([press.xdata, release.xdata])
        y0, y1 = sorted([press.ydata, release.ydata])
        self._finish([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])

    def _poly_done(self, vertices):
        if len(vertices) >= 3:
            self._finish([list(v) for v in vertices])

    def _finish(self, vertices):
        callback = self._on_drawn
        self.cancel()
        if callback is not None:
            callback(np.asarray(vertices, float))

    # ---- clicking a point ---------------------------------------------------------------------

    def on_pick(self, callback):
        """callback(table row) when a point is clicked (not while drawing a gate)."""
        self._on_pick = callback

    def _clicked(self, event):
        if self._selector is not None or self._on_pick is None or event.inaxes is not self.axes:
            return
        if self.toolbar.mode or event.xdata is None or not len(self._xy):
            return
        # Nearest point in screen space, within 6 pixels.
        screen = self.axes.transData.transform(self._xy)
        d = np.hypot(screen[:, 0] - event.x, screen[:, 1] - event.y)
        k = int(np.argmin(d))
        if d[k] <= 6:
            self._on_pick(int(self._index[k]))


class PointSizeDialog(QDialog):
    """Point size: one size for every point (diameter in pt), or sized by a
    measurement from a smallest to a largest diameter. Measurement names are shown
    short (the full name is in the tooltip), so the dialog stays narrow."""

    def __init__(self, plot, by_measurement=None):
        super().__init__(plot)
        self.plot = plot
        self.setWindowTitle("Point size")
        grid = QGridLayout(self)
        grid.setSizeConstraint(QLayout.SetFixedSize)  # as large as its contents, no larger
        self.fixed = QRadioButton("Same size for every point")
        self.by = QRadioButton("Size by a measurement")
        group = QButtonGroup(self)
        group.addButton(self.fixed)
        group.addButton(self.by)
        current = plot.point_size if plot.point_size is not None else 6.0
        self.diameter = _spin(np.sqrt(current), "Diameter of the points")
        self.feature = QComboBox()
        self.feature.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.feature.setMinimumContentsLength(20)
        self.feature.currentIndexChanged.connect(
            lambda _: self.feature.setToolTip(self.feature.currentData() or ""))
        choices = list(plot.size_choices()) if plot.size_choices is not None else []
        if choices:
            from ._cyto_labels import short_names

            short = short_names(choices, max_len=28)
            for name in choices:
                self.feature.addItem(short.get(name, name), name)
                self.feature.setItemData(self.feature.count() - 1, name, Qt.ToolTipRole)
        if plot.size_by is not None and self.feature.findData(plot.size_by) >= 0:
            self.feature.setCurrentIndex(self.feature.findData(plot.size_by))
        self.smallest = _spin(plot.size_range[0], "Diameter of the points with the lowest values")
        self.largest = _spin(plot.size_range[1], "Diameter of the points with the highest values")
        grid.addWidget(self.fixed, 0, 0, 1, 4)
        grid.addWidget(QLabel("Diameter"), 1, 1)
        grid.addWidget(self.diameter, 1, 2)
        grid.addWidget(QLabel("pt"), 1, 3)
        grid.addWidget(self.by, 2, 0, 1, 4)
        grid.addWidget(self.feature, 3, 1, 1, 3)
        grid.addWidget(QLabel("From"), 4, 1)
        grid.addWidget(self.smallest, 4, 2)
        grid.addWidget(QLabel("pt"), 4, 3)
        grid.addWidget(QLabel("to"), 5, 1)
        grid.addWidget(self.largest, 5, 2)
        grid.addWidget(QLabel("pt"), 5, 3)
        note = QLabel("Lowest 1% and highest 1% of the values get the end sizes.")
        note.setStyleSheet("color: gray;")
        note.setWordWrap(True)
        grid.addWidget(note, 6, 0, 1, 4)
        grid.setColumnMinimumWidth(0, 16)
        self.buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.buttons.accepted.connect(self.accept)
        self.buttons.rejected.connect(self.reject)
        grid.addWidget(self.buttons, 7, 0, 1, 4)
        self.by.setEnabled(bool(choices))
        use_by = (by_measurement if by_measurement is not None else plot.size_by is not None) and bool(choices)
        (self.by if use_by else self.fixed).setChecked(True)
        for w in (self.fixed, self.by):
            w.toggled.connect(self._enable)
        self._enable()
        note.setMaximumWidth(max(self.feature.sizeHint().width() + 40, 240))

    def _enable(self, *_):
        by = self.by.isChecked()
        self.diameter.setEnabled(not by)
        for w in (self.feature, self.smallest, self.largest):
            w.setEnabled(by)

    def values(self):
        """{size_by, point_size or size_range} as set_point_style takes them."""
        if self.by.isChecked() and self.feature.currentData():
            lo, hi = sorted((self.smallest.value(), self.largest.value()))
            return {"size_by": self.feature.currentData(), "size_range": (lo, hi)}
        return {"size_by": None, "point_size": round(self.diameter.value() ** 2, 4)}

    def apply(self):
        self.plot.set_point_style(**self.values())


def _spin(value, tip):
    box = QDoubleSpinBox()
    box.setRange(0.2, 40.0)
    box.setDecimals(1)
    box.setSingleStep(0.5)
    box.setValue(float(value))
    box.setToolTip(tip)
    return box
