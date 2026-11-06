"""The Cytometry scatter plot: objects as points, gates as outlines, drawing new gates.

matplotlib in Qt: zoom and pan from its toolbar, gates drawn with its selectors
(rectangle, polygon, lasso) in data coordinates of the axes shown.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtWidgets import QVBoxLayout, QWidget

MAX_POINTS = 200_000  # above this, a random subset is drawn (gates still use every object)


class ScatterPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        import matplotlib

        matplotlib.use("QtAgg", force=False)
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg, NavigationToolbar2QT
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.axes = self.figure.add_subplot(111)
        self.toolbar = NavigationToolbar2QT(self.canvas, self)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.toolbar)
        layout.addWidget(self.canvas, 1)
        self._selector = None
        self._on_drawn = None
        self._on_pick = None
        self._index = np.zeros(0, int)  # row of the table for each drawn point
        self._xy = np.zeros((0, 2))
        self._highlight = None
        self.canvas.mpl_connect("button_press_event", self._clicked)

    # ---- data ----------------------------------------------------------------------------

    def show(self, x, y, index, colors=None, background=None, xlabel="", ylabel="", gates=(),
             legend=None, cbar_label=None, values=None):
        """Draw points (x, y) for table rows `index`; `background` (x, y) in light grey
        (objects outside the shown population); `gates` = [(vertices, color, name)];
        `legend` = [(label, color)] or `values` with a colour bar."""
        ax = self.axes
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
        size = 6 if len(x) < 5000 else 2 if len(x) < 50000 else 0.5
        if background is not None and len(background[0]):
            bx, by = np.asarray(background[0], float), np.asarray(background[1], float)
            ax.scatter(bx, by, s=size, c="#d0d0d0", linewidths=0, rasterized=True, zorder=1)
        if values is not None:
            sc = ax.scatter(x[draw], y[draw], s=size, c=values[draw], cmap="viridis", linewidths=0,
                            rasterized=True, zorder=2)
            self.figure.colorbar(sc, ax=ax, label=cbar_label or "")
        else:
            c = colors if colors is None or isinstance(colors, str) else list(np.asarray(colors)[draw])
            ax.scatter(x[draw], y[draw], s=size, c=c if c is not None else "#1f77b4", linewidths=0,
                       rasterized=True, zorder=2)
        for vertices, color, name in gates:
            v = np.asarray(vertices, float)
            closed = np.vstack([v, v[:1]])
            ax.plot(closed[:, 0], closed[:, 1], color=color, lw=1.5, zorder=3)
            ax.annotate(name, v[np.argmax(v[:, 1])], color=color, fontsize=8, zorder=4,
                        xytext=(2, 2), textcoords="offset points")
        if legend:
            from matplotlib.lines import Line2D

            handles = [Line2D([], [], marker="o", ls="", color=c, label=l) for l, c in legend]
            ax.legend(handles=handles, fontsize=7, loc="best", markerscale=0.8, frameon=False)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{len(x):,} objects" + (f" (showing {len(draw):,})" if len(draw) < len(x) else ""),
                     fontsize=9)
        self._index, self._xy = index, np.column_stack([x, y])
        self._highlight = None
        self.canvas.draw_idle()

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
        """Let the user draw a 'rectangle', 'polygon' or 'lasso'; calls on_drawn(vertices)."""
        from matplotlib.widgets import LassoSelector, PolygonSelector, RectangleSelector

        self.cancel()
        self._on_drawn = on_drawn
        props = dict(color="red", linewidth=1.5)
        if mode == "rectangle":
            self._selector = RectangleSelector(self.axes, self._rect_done, useblit=False, interactive=False,
                                               props=dict(edgecolor="red", fill=False, linewidth=1.5))
        elif mode == "polygon":
            self._selector = PolygonSelector(self.axes, self._poly_done, useblit=False, props=props)
        elif mode == "lasso":
            self._selector = LassoSelector(self.axes, self._poly_done, useblit=False, props=props)
        else:
            raise ValueError(mode)

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
