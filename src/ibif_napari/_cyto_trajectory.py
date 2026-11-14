"""The Trajectory tab of the Cytometry workbench (phase 6): lineages and pseudotime.

After the objects have types (phenotypes, clusters, a classification, or neighbourhood
types), a trajectory asks how the types connect: which states lie between a healthy
and an injured type, where lineages branch, and what changes along each branch. It is
the analysis of Asghari et al. (Sci Adv 2025, figures 5 and 6) for one dataset:

- **Objects** (all, a gate, or a phenotype, e.g. only proximal tubule cells) and what
  they are classed by (**Classes**). Classed by neighbourhood types, the trajectory
  follows how one kind of neighbourhood turns into another.
- **Features** and a **Space**: PCA of the scaled features, or the map of a clustering
  (e.g. its UMAP), where the lineages are found.
- **Start** (a class chosen from biology, e.g. healthy PT or THY1+PROM1+ cells) and,
  optionally, **ends** that must be ends of lineages.
- Lineages and curves come from vtea-core `trajectory.slingshot` (Slingshot's cluster
  tree and simultaneous principal curves); each object gets a pseudotime per lineage.

The results show the lineages on the map, pseudotime on the map, a feature along
pseudotime per lineage (its "pseudotime spectrum"), all features along a lineage as a
heat map, which features change with pseudotime (Spearman rho, BH q < 0.05), and how
the classes, or a second classing (**Also follow**, e.g. neighbourhood types along a
cell-type trajectory), change along each lineage. "<name> pseudotime", "<name> lineage"
and "<name> pseudotime t<j>" become columns of the object table, to plot, gate, write
rules with and export.
"""
from __future__ import annotations

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from . import _cytometry as cy
from ._cyto_cluster import INTENSITY_STATS, Heatmap
from ._cyto_spatial import _AnalysisTab

VIEWS = ("Lineages on the map (by class)", "Pseudotime on the map", "A feature along pseudotime",
         "Features along a lineage (heat map)", "Which features change (Spearman ρ)",
         "Classes along pseudotime", "Also followed: classes along pseudotime")
# Euclidean first: Slingshot's covariance-scaled distance can join types out of order when
# they are cut from a continuum (each type is long along the trajectory and its spread
# shortens the distance along it); compare both.
DISTANCES = {"Euclidean (between class means)": "euclidean", "Slingshot (covariance-scaled)": "slingshot"}
MAX_POINTS = 30000  # drawn on the map (a random sample beyond that)


class TrajectoryTab(_AnalysisTab):
    """Phase 6: trajectories of cell types and neighbourhood types."""

    kind = "trajectory"
    base_name = "Trajectory"
    include_neighbourhoods = True

    def settings_rows(self):
        self.classes.setToolTip("What each object is: its phenotype, cluster or class (a trajectory of cell "
                                "types), or its neighbourhood type (a trajectory of neighbourhoods)")
        self.population = QComboBox()
        self.population.setToolTip("Find lineages among these objects: all, a gate, or a phenotype "
                                   "(e.g. only proximal tubule cells)")
        self.population.currentIndexChanged.connect(self._fill_targets)
        self.form.insertRow(0, "Objects", self.population)
        self.features = QListWidget()
        self.features.setMinimumHeight(110)
        self.features.setToolTip("The measurements lineages are found on and followed along. Classed by "
                                 "neighbourhood types, the neighbourhoods' composition is ticked")
        self.form.addRow("Features", self.features)
        quick = QHBoxLayout()
        for text, how in (("Intensities", "intensity"), ("Composition", "composition"), ("All", "all"),
                          ("None", "none")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, h=how: self.check_features(h))
            quick.addWidget(b)
        self.form.addRow("", quick)
        self.asinh = QCheckBox("asinh(x/150) first (for intensities)")
        self.asinh.setChecked(True)
        self.form.addRow("", self.asinh)
        self.space = QComboBox()
        self.space.setToolTip("Where lineages are found: PCA of the scaled features, or the map (UMAP, "
                              "t-SNE) of a clustering run")
        self.space.currentIndexChanged.connect(self._enable)
        self.form.addRow("Space", self.space)
        self.components = QSpinBox()
        self.components.setRange(2, 50)
        self.components.setValue(10)
        self.components.setToolTip("PCA: number of components the lineages are found in")
        self.form.addRow("Components", self.components)
        self.start = QComboBox()
        self.start.setToolTip("The class lineages start from (pseudotime 0): choose it from biology, e.g. "
                              "the healthy type, or a candidate progenitor / transitional state")
        self.form.addRow("Start", self.start)
        self.ends = QListWidget()
        self.ends.setMinimumHeight(70)
        self.ends.setMaximumHeight(120)
        self.ends.setToolTip("Optional: classes that must be ends of lineages (e.g. known end states). "
                             "None ticked: every leaf of the tree is an end")
        self.form.addRow("Ends", self.ends)
        self.distance = QComboBox()
        for text, value in DISTANCES.items():
            self.distance.addItem(text, value)
        self.distance.setToolTip("Distance between classes for the tree: between their means (default), "
                                 "or Slingshot's, scaled by the classes' spread. Slingshot's suits round "
                                 "clusters; with types cut from a continuum it can join them out of order. "
                                 "Compare both")
        self.form.addRow("Distance", self.distance)
        self.track = QComboBox()
        self.track.setToolTip("Also follow how these classes change along pseudotime, e.g. neighbourhood "
                              "types along a trajectory of cell types, or cell types along one of "
                              "neighbourhoods")
        self.form.addRow("Also follow", self.track)
        self.bins = QSpinBox()
        self.bins.setRange(4, 100)
        self.bins.setValue(20)
        self.bins.setToolTip("Pseudotime bins for the feature trends (half as many for class fractions)")
        self.form.addRow("Bins", self.bins)
        self._classes_key = None

    def result_rows(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Show"))
        self.view = QComboBox()
        self.view.addItems(list(VIEWS))
        self.view.currentIndexChanged.connect(lambda _: self.show_analysis(self.analysis.currentText()))
        row.addWidget(self.view, 1)
        self.feature = QComboBox()
        self.feature.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.feature.setMinimumContentsLength(14)
        self.feature.currentIndexChanged.connect(lambda _: self.show_analysis(self.analysis.currentText()))
        row.addWidget(self.feature, 1)
        self.lineage = QComboBox()
        self.lineage.currentIndexChanged.connect(lambda _: self.show_analysis(self.analysis.currentText()))
        row.addWidget(self.lineage)
        self.results.addLayout(row)
        self.stack = QStackedWidget()
        self.plot = TrajectoryPlot()
        self.heatmap = Heatmap()
        self.stack.addWidget(self.plot)
        self.stack.addWidget(self.heatmap)
        self.results.addWidget(self.stack, 1)
        row = QHBoxLayout()
        self.plot_btn = QPushButton("Colour the plot by pseudotime")
        self.plot_btn.setToolTip("Show the trajectory's map on the Gating tab, coloured by pseudotime")
        self.plot_btn.clicked.connect(lambda _=False: self.colour_plot())
        self.image_btn = QPushButton("Show pseudotime on image")
        self.image_btn.clicked.connect(lambda _=False: self.show_on_image())
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(lambda _=False: self.delete())
        for b in (self.plot_btn, self.image_btn, self.delete_btn):
            row.addWidget(b)
        self.results.addLayout(row)

    # ---- refreshing ---------------------------------------------------------------------

    def refresh(self):
        from ._cyto_workbench import _pop_key

        state = self.state
        keep = self.population.currentData()
        self.population.blockSignals(True)
        self.population.clear()
        if state.table is not None:
            for kind, key, label in state.populations():
                self.population.addItem(label.strip(), _pop_key(kind, key))
        index = self.population.findData(keep)
        self.population.setCurrentIndex(max(index, 0) if self.population.count() else -1)
        self.population.blockSignals(False)
        checked = set(self.checked_features())
        first = self.features.count() == 0
        self.features.clear()
        for f in state.measured_features() if state.table is not None else []:
            item = QListWidgetItem(f)
            item.setToolTip(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            on = f in checked or (first and f.endswith(INTENSITY_STATS[:1]))
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.features.addItem(item)
        keep = self.space.currentData()
        self.space.blockSignals(True)
        self.space.clear()
        self.space.addItem("PCA of the features", "pca")
        for name in state.analyses_of("clustering") if state.table is not None else []:
            if len(state.analyses[name]["columns"].get("map") or []) >= 2:
                self.space.addItem(f"{name} map ({state.analyses[name].get('map') or 'map'})", name)
        index = self.space.findData(keep)
        self.space.setCurrentIndex(max(index, 0))
        self.space.blockSignals(False)
        keep = self.track.currentData()
        self.track.clear()
        self.track.addItem("(nothing)", None)
        if state.table is not None and not state.broken_phenotypes():
            for label, key in state.class_choices(neighbourhoods=True):
                self.track.addItem(label, key)
        index = self.track.findData(keep)
        self.track.setCurrentIndex(max(index, 0))
        self._enable()
        super().refresh()

    def _enable(self, *_):
        self.components.setEnabled(self.space.currentData() in (None, "pca"))

    def class_values(self):
        """The classes of the chosen objects (what Start and Ends offer)."""
        from ._cyto_workbench import _pop_parse

        values = super().class_values()
        key = self.classes.currentData()
        if key is None or not self.population.count() or self.state.table is None:
            return values
        try:
            mask = self.state.mask(*_pop_parse(self.population.currentData()))
        except (KeyError, ValueError):
            return values
        present = {c for c in self.state.classes(key)[mask] if c is not None}
        return [v for v in values if v in present]

    def _fill_targets(self, *_):
        """Start and end choices for the classes; tick the neighbourhood composition when
        classed by neighbourhood types, the intensities when classed by cell types."""
        values = self.class_values()
        keep = self.start.currentText()
        self.start.clear()
        self.start.addItems(values)
        if keep in values:
            self.start.setCurrentText(keep)
        ticked = {self.ends.item(i).text() for i in range(self.ends.count())
                  if self.ends.item(i).checkState() == Qt.Checked}
        self.ends.clear()
        for value in values:
            item = QListWidgetItem(value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if value in ticked else Qt.Unchecked)
            self.ends.addItem(item)
        key = self.classes.currentData()
        if key != self._classes_key and self.features.count():
            was_hood = self._classes_key is not None and self.state.neighbourhood_of(self._classes_key)
            if self.state.neighbourhood_of(key) if key else False:
                self.check_features("composition")
            elif was_hood:
                self.check_features("intensity")
        self._classes_key = key

    def checked_features(self):
        return [self.features.item(i).text() for i in range(self.features.count())
                if self.features.item(i).checkState() == Qt.Checked]

    def checked_ends(self):
        return [self.ends.item(i).text() for i in range(self.ends.count())
                if self.ends.item(i).checkState() == Qt.Checked]

    def composition_features(self):
        """The fraction columns of the neighbourhood analysis being classed by (or of every
        neighbourhood analysis when classed by something else)."""
        key = self.classes.currentData()
        hood = self.state.neighbourhood_of(key) if key else None
        names = [hood] if hood else self.state.analyses_of("neighbourhoods")
        return {c for n in names for c in self.state.analyses[n]["columns"].get("features", [])}

    def check_features(self, how):
        wanted = self.composition_features() if how == "composition" else set()
        for i in range(self.features.count()):
            item = self.features.item(i)
            text = item.text()
            on = {"all": True, "none": False, "intensity": text.endswith(INTENSITY_STATS[:1]),
                  "composition": text in wanted}[how]
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)

    # ---- running ------------------------------------------------------------------------

    def settings(self):
        from ._cyto_workbench import _pop_parse

        out = super().settings()
        if not self.start.currentText():
            raise ValueError("Choose the class the lineages start from.")
        out.update(population=list(_pop_parse(self.population.currentData())) if self.population.count()
                   else ["all", None], features=self.checked_features(), asinh=self.asinh.isChecked(),
                   cofactor=cy.ASINH_COFACTOR, scaling="zscore", space=self.space.currentData() or "pca",
                   components=self.components.value(), start=self.start.currentText(),
                   ends=[e for e in self.checked_ends() if e != self.start.currentText()],
                   distance=self.distance.currentData(), track=self.track.currentData(),
                   track_label=self.track.currentText() if self.track.currentData() else None,
                   n_bins=self.bins.value())
        if not out["features"]:
            raise ValueError("Tick at least one feature.")
        return out

    def make_job(self, settings):
        return self.state.trajectory_job(settings)

    def store(self, name, settings, result):
        self.state.add_trajectory(name, settings, result)

    # ---- results ------------------------------------------------------------------------

    def show_analysis(self, name):
        state = self.state
        has = bool(name) and name in state.analyses
        for b in (self.plot_btn, self.image_btn, self.delete_btn):
            b.setEnabled(has)
        view = self.view.currentIndex()
        self.feature.setVisible(has and view == 2)
        self.lineage.setVisible(has and view == 3)
        if not has:
            self.info.setText("Find lineages through the classes from a start class, and pseudotime along "
                              "them (Slingshot). Classed by neighbourhood types, the lineages follow how "
                              "neighbourhoods change.")
            self.stack.setCurrentWidget(self.plot)
            self.plot.clear()
            return
        info = state.analyses[name]
        st = info["structure"]
        lines = [f"<b>{lin}</b>: {' → '.join(map(str, path))} ({n:,} objects)"
                 for lin, path, n in zip(st["names"], st["lineages"], st["objects"])]
        space = "PCA" if info.get("space", "pca") == "pca" else f"the {info['space']} map"
        dist = "Slingshot" if info.get("distance") == "slingshot" else "Euclidean"
        self.info.setText(f"<b>{name}</b>: {info['n_objects']:,} objects by {info.get('class_label', '')}, "
                          f"in {space} ({dist} tree), from <b>{st['start']}</b>"
                          + (f" (seconds: {info['seconds']:.1f})" if "seconds" in info else "") + ".<br>"
                          + "<br>".join(lines))
        self._fill_choices(info)
        if view in (0, 1):
            self._show_map(name, info, by_class=view == 0)
        elif view == 2:
            self._show_feature(name)
        elif view == 3:
            self._show_heatmap(name)
        elif view == 4:
            self._show_association(name)
        else:
            self._show_composition(name, info, tracked=view == 6)

    def _fill_choices(self, info):
        features = list(info.get("features", []))
        for box, items in ((self.feature, features), (self.lineage, info["structure"]["names"])):
            keep = box.currentText()
            if [box.itemText(i) for i in range(box.count())] != items:
                box.blockSignals(True)
                box.clear()
                box.addItems(items)
                if keep in items:
                    box.setCurrentText(keep)
                box.blockSignals(False)

    def _show_map(self, name, info, by_class):
        state = self.state
        st = info["structure"]
        x, y = (state.column(c) for c in st["display"])
        t = state.column(info["columns"]["pseudotime"])
        on = np.flatnonzero(np.isfinite(t) & np.isfinite(x) & np.isfinite(y))
        if len(on) > MAX_POINTS:
            on = np.sort(np.random.default_rng(0).choice(on, MAX_POINTS, replace=False))
        colors = values = legend = None
        if by_class:
            try:
                classes = state.classes(info["classes"])[on]
                palette = state.class_colors(info["classes"])
            except (KeyError, ValueError):
                classes, palette = np.array([None] * len(on), dtype=object), {}
            colors = np.array([palette.get(c, "#bdbdbd") for c in classes], dtype=object)
            legend = [(c, palette.get(c, "#bdbdbd")) for c in st["clusters"]][:20]
        else:
            values = t[on]
        self.stack.setCurrentWidget(self.plot)
        self.plot.draw_map(x[on], y[on], colors=colors, values=values, legend=legend, centers=st["centers"],
                           clusters=st["clusters"], edges=st["edges"], curves=st["curves"], names=st["names"],
                           xlabel=st["display"][0], ylabel=st["display"][1])

    def _show_feature(self, name):
        trends = self.state.analysis_table(name, "trends")
        feature = self.feature.currentText()
        part = trends[trends["feature"] == feature]
        series = [(lin, g["pseudotime"].to_numpy(float), g["mean"].to_numpy(float), g["sem"].to_numpy(float))
                  for lin, g in part.groupby("lineage", sort=False)]
        self.stack.setCurrentWidget(self.plot)
        self.plot.draw_trends(series, ylabel=f"{feature} (z-score)")

    def _show_heatmap(self, name):
        trends = self.state.analysis_table(name, "trends")
        part = trends[trends["lineage"] == self.lineage.currentText()]
        self.stack.setCurrentWidget(self.heatmap)
        if part.empty:
            self.heatmap.show(None)
            return
        grid = part.pivot(index="bin", columns="feature", values="mean")
        grid = grid[list(dict.fromkeys(part["feature"]))]
        self.heatmap.show(grid, row_label="pseudotime bin (early → late)", value_label="mean (z-score)",
                          color_rows=False)

    def _show_association(self, name):
        table = self.state.analysis_table(name, "association")
        self.stack.setCurrentWidget(self.heatmap)
        if table.empty:
            self.heatmap.show(None)
            return
        table = table.assign(rho=np.where(table["q"] < 0.05, table["rho"], np.nan))
        grid = table.pivot(index="lineage", columns="feature", values="rho")
        grid = grid.loc[list(dict.fromkeys(table["lineage"])), list(dict.fromkeys(table["feature"]))]
        counts = table.groupby("lineage")["objects"].max().to_frame("objects")
        self.heatmap.show(grid, counts, row_label="lineage (objects)",
                          value_label="Spearman ρ with pseudotime (q < 0.05)", limits=(-1, 1), color_rows=False,
                          annotate=True)

    def _show_composition(self, name, info, tracked):
        which = "tracked" if tracked else "composition"
        self.stack.setCurrentWidget(self.plot)
        if which not in info["tables"]:
            self.plot.clear("Nothing was followed: choose classes under Also follow and run again.")
            return
        table = self.state.analysis_table(name, which)
        key = info.get("track") if tracked else info.get("classes")
        try:
            palette = self.state.class_colors(key) if key else {}
        except (KeyError, ValueError):
            palette = {}
        label = (info.get("track_label") if tracked else info.get("class_label")) or "class"
        self.plot.draw_composition(table, palette, title=label)

    # ---- actions ------------------------------------------------------------------------

    def colour_plot(self):
        name = self.analysis.currentText()
        if name not in self.state.analyses:
            return
        info = self.state.analyses[name]
        space = info.get("space", "pca")
        self.bench.plot_map(name if space == "pca" else space)
        pop = info.get("population") or ["all", None]
        try:
            self.bench.set_population(pop[0], pop[1])
        except ValueError:
            pass
        self.bench.color_by.setCurrentText(info["columns"]["pseudotime"])
        self.bench.tabs.setCurrentIndex(0)
        self.bench.replot()

    def show_on_image(self):
        name = self.analysis.currentText()
        if name not in self.state.analyses:
            return None
        try:
            layer = self.state.show_trajectory(name)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        self.status.setText(f"Shown in layer {layer.name} (pseudotime: dark = early, yellow = late).")
        return layer


def _lineage_color(j):
    return cy.PALETTE[j % len(cy.PALETTE)]


class TrajectoryPlot(QWidget):
    """The map with the class tree and lineage curves, feature trends along pseudotime,
    and class fractions along pseudotime (matplotlib)."""

    def __init__(self):
        super().__init__()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 4), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self.canvas)

    def clear(self, text=""):
        self.figure.clf()
        if text:
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, text, ha="center", va="center", wrap=True, fontsize=9, color="gray")
            ax.set_axis_off()
        self.canvas.draw_idle()

    def draw_map(self, x, y, colors=None, values=None, legend=None, centers=(), clusters=(), edges=(),
                 curves=(), names=(), xlabel="", ylabel=""):
        import matplotlib.patheffects as pe

        self.figure.clf()
        ax = self.figure.add_subplot(111)
        size = 4 if len(x) < 5000 else 1.5
        if values is not None:
            sc = ax.scatter(x, y, c=values, s=size, cmap="viridis", linewidths=0, rasterized=True)
            self.figure.colorbar(sc, ax=ax, label="pseudotime", shrink=0.8)
        else:
            ax.scatter(x, y, c=list(colors) if colors is not None and len(colors) else None, s=size,
                       linewidths=0, rasterized=True)
        centre = {c: p for c, p in zip(clusters, centers)}
        for a, b in edges:  # the class tree
            if a in centre and b in centre:
                ax.plot([centre[a][0], centre[b][0]], [centre[a][1], centre[b][1]], color="#555555", lw=0.8,
                        ls="--", zorder=3)
        for c, p in centre.items():
            ax.plot(*p[:2], "o", color="white", mec="#333333", ms=5, zorder=4)
        outline = [pe.Stroke(linewidth=4, foreground="white"), pe.Normal()]
        for j, (curve, lin) in enumerate(zip(curves, names)):
            curve = np.asarray(curve, float)
            ax.plot(curve[:, 0], curve[:, 1], color=_lineage_color(j), lw=2.2, zorder=5, path_effects=outline)
            ax.annotate("", xy=curve[-1, :2], xytext=curve[max(len(curve) - 4, 0), :2], zorder=6,
                        arrowprops={"arrowstyle": "-|>", "color": _lineage_color(j), "lw": 2})
            ax.text(*curve[-1, :2], f" {lin}", fontsize=9, fontweight="bold", color=_lineage_color(j), zorder=6,
                    path_effects=[pe.withStroke(linewidth=3, foreground="white")])
        if legend:
            from matplotlib.lines import Line2D

            handles = [Line2D([], [], marker="o", ls="", color=col, label=str(lab), ms=5) for lab, col in legend]
            ax.legend(handles=handles, fontsize=7, loc="best", framealpha=0.8, markerscale=1)
        ax.set_xlabel(xlabel, fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        self.canvas.draw_idle()

    def draw_trends(self, series, ylabel=""):
        """`series`: [(lineage, pseudotime, mean, sem)], one line (and band) per lineage."""
        self.figure.clf()
        ax = self.figure.add_subplot(111)
        for j, (lin, t, mean, sem) in enumerate(series):
            ok = np.isfinite(mean)
            col = _lineage_color(j)
            ax.plot(t[ok], mean[ok], color=col, lw=2, label=lin)
            band = ok & np.isfinite(sem)
            ax.fill_between(t[band], (mean - sem)[band], (mean + sem)[band], color=col, alpha=0.2, lw=0)
        ax.axhline(0, color="#999999", lw=0.6)
        ax.set_xlabel("pseudotime", fontsize=8)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.tick_params(labelsize=7)
        if series:
            ax.legend(fontsize=7, title="lineage", title_fontsize=7)
        self.canvas.draw_idle()

    def draw_composition(self, table, palette, title="class"):
        """Stacked fractions of each class along pseudotime, one panel per lineage."""
        self.figure.clf()
        lineages = list(dict.fromkeys(table["lineage"]))
        if not lineages:
            self.canvas.draw_idle()
            return
        classes = list(dict.fromkeys(table["class"]))
        axes = self.figure.subplots(1, len(lineages), sharey=True, squeeze=False)[0]
        for k, (ax, lin) in enumerate(zip(axes, lineages)):
            part = table[table["lineage"] == lin]
            grid = part.pivot(index="pseudotime", columns="class", values="fraction").reindex(columns=classes)
            grid = grid.dropna(how="all").fillna(0.0)  # bins with no objects are left out
            colors = [palette.get(c, cy.PALETTE[i % len(cy.PALETTE)]) for i, c in enumerate(classes)]
            ax.stackplot(grid.index.to_numpy(float), grid.to_numpy(float).T, colors=colors,
                         labels=[str(c) for c in classes])
            ax.set_title(lin, fontsize=9, color=_lineage_color(k))
            ax.set_xlabel("pseudotime", fontsize=8)
            ax.set_ylim(0, 1)
            ax.tick_params(labelsize=7)
        axes[0].set_ylabel(f"fraction ({title})", fontsize=8)
        handles, labels = axes[0].get_legend_handles_labels()
        self.figure.legend(handles[::-1], labels[::-1], fontsize=7, loc="outside right upper")
        self.canvas.draw_idle()


__all__ = ["TrajectoryTab", "TrajectoryPlot", "VIEWS"]
