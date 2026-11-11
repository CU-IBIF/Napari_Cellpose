"""The Clustering tab of the Cytometry workbench: find object types without gating.

Pick the objects (all, a gate, or a phenotype, to work coarse-to-fine), the features,
how to scale them, a map (UMAP, t-SNE or PCA) and a clustering (Leiden, Louvain,
k-means, Gaussian mixture, or k-means choosing k by BIC). The run's cluster ids and map
coordinates become columns of the object table, so they can be plotted and gated on the
Gating tab and used in phenotype rules. A heat map of the clusters' profiles (z-score of
each feature) and their top markers is what clusters are named from: type a phenotype
name per cluster (the same name merges clusters) and make them phenotypes.
"""
from __future__ import annotations

import time

import numpy as np
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from . import _cytometry as cy

INTENSITY_STATS = (" mean", " median")


class ClusteringTab(QWidget):
    def __init__(self, bench):
        super().__init__()
        self.bench = bench
        self.state = bench.state
        self._worker = None
        split = QSplitter(Qt.Horizontal)
        QVBoxLayout(self).addWidget(split)

        # ---- settings ----
        left = QWidget()
        form = QFormLayout(left)
        self.population = QComboBox()
        self.population.setToolTip("Cluster these objects: all, a gate, or a phenotype (to split it further)")
        form.addRow("Objects", self.population)
        self.features = QListWidget()
        self.features.setMinimumHeight(140)
        form.addRow("Features", self.features)
        quick = QHBoxLayout()
        for text, how in (("Intensities", "intensity"), ("All", "all"), ("None", "none")):
            b = QPushButton(text)
            b.clicked.connect(lambda _=False, h=how: self.check_features(h))
            quick.addWidget(b)
        form.addRow("", quick)
        self.asinh = QCheckBox("asinh(x/150) first (for intensities)")
        self.asinh.setChecked(True)
        form.addRow("", self.asinh)
        self.scaling = QComboBox()
        self.scaling.addItems(list(cy.SCALINGS))
        self.scaling.setToolTip("Put features on one scale: z-score, robust (median/IQR), min-max, or none")
        form.addRow("Scaling", self.scaling)
        self.map = QComboBox()
        _add_methods(self.map, cy.EMBEDDINGS)
        form.addRow("Map", self.map)
        self.method = QComboBox()
        _add_methods(self.method, cy.CLUSTERINGS)
        form.addRow("Clustering", self.method)
        self.resolution = _dspin(0.05, 10, 1.0, 0.1, "Leiden / Louvain: higher = more, smaller clusters")
        form.addRow("Resolution", self.resolution)
        self.k = QSpinBox()
        self.k.setRange(2, 100)
        self.k.setValue(8)
        self.k.setToolTip("k-means / Gaussian mixture: number of clusters; best k: the most to try")
        form.addRow("Clusters (k)", self.k)
        self.neighbors = QSpinBox()
        self.neighbors.setRange(2, 200)
        self.neighbors.setValue(15)
        self.neighbors.setToolTip("UMAP and Leiden/Louvain: neighbours per object (more = broader structure)")
        form.addRow("Neighbours", self.neighbors)
        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        self.seed.setToolTip("Random seed: the same seed gives the same result")
        form.addRow("Seed", self.seed)
        self.reproducible = QCheckBox("Reproducible map (slower)")
        self.reproducible.setToolTip(
            "UMAP: the same seed gives exactly the same map, but it then runs on one core. "
            "Off, it uses every core (several times faster) and the layout varies a little "
            "between runs. Clusters are not affected.")
        form.addRow("", self.reproducible)
        self.name = QLineEdit()
        form.addRow("Name", self.name)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(lambda _=False: self.run())
        form.addRow(self.run_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        form.addRow(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        form.addRow(self.status)
        split.addWidget(left)

        # ---- results ----
        right = QWidget()
        rl = QVBoxLayout(right)
        row = QHBoxLayout()
        row.addWidget(QLabel("Results"))
        self.analysis = QComboBox()
        self.analysis.currentTextChanged.connect(self.show_analysis)
        row.addWidget(self.analysis, 1)
        rl.addLayout(row)
        self.info = QLabel()
        self.info.setWordWrap(True)
        rl.addWidget(self.info)
        self.heatmap = Heatmap()
        rl.addWidget(self.heatmap, 2)
        box = QGroupBox("Name the clusters (the same name merges clusters)")
        bl = QVBoxLayout(box)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Cluster", "Objects", "Markers", "Phenotype"])
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.Stretch)
        bl.addWidget(self.table)
        prow = QHBoxLayout()
        prow.addWidget(QLabel("Within"))
        self.parent_box = QComboBox()
        prow.addWidget(self.parent_box, 1)
        self.make_btn = QPushButton("Make phenotypes")
        self.make_btn.clicked.connect(lambda _=False: self.make_phenotypes())
        prow.addWidget(self.make_btn)
        bl.addLayout(prow)
        rl.addWidget(box, 1)
        brow = QHBoxLayout()
        self.plot_btn = QPushButton("Plot the map")
        self.plot_btn.setToolTip("Show the map on the Gating tab, coloured by cluster (gate on it there)")
        self.plot_btn.clicked.connect(lambda _=False: self.bench.plot_map(self.analysis.currentText()))
        self.image_btn = QPushButton("Show clusters on image")
        self.image_btn.clicked.connect(lambda _=False: self.show_on_image())
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(lambda _=False: self.delete())
        for b in (self.plot_btn, self.image_btn, self.delete_btn):
            brow.addWidget(b)
        rl.addLayout(brow)
        split.addWidget(right)
        split.setSizes([360, 760])

        self.method.currentIndexChanged.connect(self._update_params)
        self.map.currentIndexChanged.connect(self._update_params)
        self._update_params()
        cy.warm_up()  # compile UMAP in the background while the user picks settings

    # ---- refreshing ---------------------------------------------------------------------

    def refresh(self):
        state = self.state
        from ._cyto_workbench import _fill, _pop_key

        keep_pop = self.population.currentData()
        _fill(self.population, [(label.strip(), _pop_key(kind, key)) for kind, key, label in state.populations()]
              if state.table is not None else [], keep_pop)
        checked = self.checked_features()
        first = self.features.count() == 0
        self.features.clear()
        for f in state.measured_features() if state.table is not None else []:
            item = QListWidgetItem(f)
            item.setToolTip(f)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            on = f in checked or (first and f.endswith(INTENSITY_STATS[:1]))
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)
            self.features.addItem(item)
        keep_parent = self.parent_box.currentData()
        self.parent_box.clear()
        self.parent_box.addItem("(top level)", None)
        for node in state.tree:
            self.parent_box.addItem("  " * (state.tree.depth(node.name) - 1) + node.name, node.name)
        index = self.parent_box.findData(keep_parent)
        self.parent_box.setCurrentIndex(max(index, 0))
        current = self.analysis.currentText()
        self.analysis.blockSignals(True)
        self.analysis.clear()
        self.analysis.addItems(list(state.analyses))
        if current in state.analyses:
            self.analysis.setCurrentText(current)
        elif state.analyses:
            self.analysis.setCurrentIndex(len(state.analyses) - 1)
        self.analysis.blockSignals(False)
        if not self.name.text() or self.name.text() in state.analyses:
            self.name.setText(state.unique_analysis_name())
        self.run_btn.setEnabled(state.table is not None and self._worker is None)
        self.show_analysis(self.analysis.currentText())

    def checked_features(self):
        return [self.features.item(i).text() for i in range(self.features.count())
                if self.features.item(i).checkState() == Qt.Checked]

    def check_features(self, how, names=()):
        for i in range(self.features.count()):
            item = self.features.item(i)
            text = item.text()
            on = {"all": True, "none": False, "intensity": text.endswith(INTENSITY_STATS[:1]),
                  "names": text in names}[how]
            item.setCheckState(Qt.Checked if on else Qt.Unchecked)

    def _update_params(self, *_):
        method = self.method.currentData()
        self.resolution.setEnabled(method in ("leiden", "louvain"))
        self.k.setEnabled(method in ("kmeans", "gmm", "autok"))
        self.neighbors.setEnabled(method in ("leiden", "louvain") or self.map.currentData() == "umap")
        self.reproducible.setEnabled(self.map.currentData() == "umap")

    # ---- running ------------------------------------------------------------------------

    def settings(self):
        from ._cyto_workbench import _pop_parse

        return {
            "population": list(_pop_parse(self.population.currentData())),
            "features": self.checked_features(), "asinh": self.asinh.isChecked(),
            "cofactor": cy.ASINH_COFACTOR, "scaling": self.scaling.currentText(),
            "map": self.map.currentText() if self.map.currentData() else None,
            "embedding": self.map.currentData(), "clustering": self.method.currentData(),
            "method": self.method.currentText(), "resolution": self.resolution.value(),
            "k": self.k.value(), "n_neighbors": self.neighbors.value(), "seed": self.seed.value(),
            "reproducible": self.reproducible.isChecked(),
        }

    def run(self, wait=False):
        """Run the chosen map and clustering (in the background unless `wait`)."""
        state = self.state
        if state.table is None or self._worker is not None:
            return None
        settings = self.settings()
        name = self.name.text().strip() or state.unique_analysis_name()
        if name in state.analyses:
            self.status.setText(f"There is already a result called {name!r}.")
            return None
        if not settings["features"]:
            self.status.setText("Tick at least one feature.")
            return None
        if not settings["embedding"] and not settings["clustering"]:
            self.status.setText("Choose a map, a clustering, or both.")
            return None
        for method in (settings["embedding"], settings["clustering"]):
            need = cy.available(method)
            if need:
                self.status.setText(f"{method} is not installed here: {need}")
                return None
        rows, matrix = state.matrix(tuple(settings["population"]), settings["features"], settings["asinh"])
        if len(rows) < 3:
            self.status.setText("Too few objects (with every feature measured) to analyse.")
            return None
        kwargs = dict(embedding=settings["embedding"], clustering=settings["clustering"],
                      scaling=settings["scaling"], seed=settings["seed"], n_neighbors=settings["n_neighbors"],
                      resolution=settings["resolution"], k=settings["k"], k_max=settings["k"],
                      reproducible=settings["reproducible"])
        started = time.perf_counter()

        def done(result):
            labels, emb = result
            settings["seconds"] = round(time.perf_counter() - started, 2)
            try:
                state.add_analysis(name, settings, rows, labels, emb)
            except ValueError as e:
                self.status.setText(str(e))
                return
            k = len(np.unique(labels)) if labels is not None else 0
            self.status.setText(f"{name}: {len(rows):,} objects" + (f", {k} clusters" if k else "")
                                + f" ({settings['seconds']:.1f} s).")
            self.analysis.setCurrentText(name)

        if wait:
            done(cy.compute_analysis(matrix, **kwargs))
            return state.analyses.get(name)
        from napari.qt.threading import thread_worker

        # Import here, not in the worker thread: two threads importing scikit-learn at
        # once can fail (see cy.import_analysis_modules).
        cy.import_analysis_modules(settings["embedding"], settings["clustering"])

        @thread_worker(ignore_errors=True)
        def work():
            return cy.compute_analysis(matrix, **kwargs)

        worker = work()
        worker.returned.connect(done)
        worker.errored.connect(lambda e: self.status.setText(f"Failed: {e}"))
        worker.finished.connect(self._finished)
        self._worker = worker
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status.setText(f"Running on {len(rows):,} objects…")
        worker.start()
        return worker

    def _finished(self):
        self._worker = None
        self.progress.setVisible(False)
        self.run_btn.setEnabled(self.state.table is not None)

    # ---- results ------------------------------------------------------------------------

    def show_analysis(self, name):
        state = self.state
        self.table.setRowCount(0)
        has = bool(name) and name in state.analyses
        for b in (self.plot_btn, self.image_btn, self.delete_btn, self.make_btn):
            b.setEnabled(has)
        if not has:
            self.info.setText("Run a clustering to see its clusters here.")
            self.heatmap.show(None)
            return
        info = state.analyses[name]
        pop = info.get("population") or ["all", None]
        where = "all objects" if pop[0] == "all" else f"{pop[0]} {pop[1]}"
        self.info.setText(
            f"<b>{name}</b>: {info['n_objects']:,} objects ({where}); {len(info['features'])} features"
            f"{', asinh' if info.get('asinh') else ''}, {info.get('scaling')}; "
            f"map {info.get('map') or 'none'}; {info.get('method') if info.get('clustering') else 'no clustering'}.")
        has_clusters = bool(info["columns"].get("cluster"))
        self.image_btn.setEnabled(has_clusters)
        self.make_btn.setEnabled(has_clusters)
        if not has_clusters:
            self.heatmap.show(None)
            return
        profiles = state.profiles(name)
        self.heatmap.show(profiles["z"], profiles["counts"])
        counts, desc = profiles["counts"], profiles["description"]
        short_desc = _short_markers(profiles["z"])
        self.table.setRowCount(len(counts))
        for r, cluster in enumerate(counts.index):
            cells = [str(cluster), f"{int(counts.loc[cluster, 'objects']):,} ({100 * counts.loc[cluster, 'fraction']:.1f}%)",
                     short_desc.get(cluster, desc[cluster]), ""]
            for c, text in enumerate(cells):
                item = QTableWidgetItem(text)
                if c == 2:
                    item.setToolTip(desc[cluster])  # the full feature names
                if c < 3:
                    item.setFlags(item.flags() & ~Qt.ItemIsEditable)
                if c == 0:
                    item.setForeground(_qcolor(cy.PALETTE[(int(cluster) - 1) % len(cy.PALETTE)]))
                self.table.setItem(r, c, item)
        self.table.resizeColumnToContents(0)
        self.table.resizeColumnToContents(1)
        if pop[0] == "phenotype":
            index = self.parent_box.findData(pop[1])
            if index >= 0:
                self.parent_box.setCurrentIndex(index)

    def set_names(self, names):
        """Type phenotype names for clusters ({cluster id: name}); also used by scripts."""
        for r in range(self.table.rowCount()):
            cluster = int(self.table.item(r, 0).text())
            if cluster in names:
                self.table.item(r, 3).setText(names[cluster])

    def make_phenotypes(self):
        name = self.analysis.currentText()
        names = {int(self.table.item(r, 0).text()): self.table.item(r, 3).text()
                 for r in range(self.table.rowCount())}
        if not any(n.strip() for n in names.values()):
            self.status.setText("Type a phenotype name for at least one cluster.")
            return []
        try:
            made = self.state.label_clusters(name, names, self.parent_box.currentData())
        except (ValueError, KeyError) as e:
            self.status.setText(str(e))
            return []
        self.status.setText("Made " + ", ".join(self.state.tree.path(p.name) for p in made) + ".")
        return made

    def show_on_image(self):
        try:
            layer = self.state.show_clusters(self.analysis.currentText())
        except ValueError as e:
            self.status.setText(str(e))
            return
        self.status.setText(f"Shown in layer {layer.name}.")

    def delete(self):
        name = self.analysis.currentText()
        if name in self.state.analyses:
            orphaned = self.state.remove_analysis(name) or []
            broken = self.state.broken_phenotypes()
            text = f"Deleted {name}."
            if orphaned:
                text += (f" Gates drawn on its map ({', '.join(orphaned)}) select nothing now; run a "
                         f"clustering called {name!r} again to bring them back, or delete them.")
            if broken:
                text += " These phenotypes used it and need a new rule: " + ", ".join(broken)
            self.status.setText(text)


class Heatmap(QWidget):
    """Clusters x features, z-score of each cluster's mean, with cluster sizes. Feature
    names are shortened to fit (`_cyto_labels.short_names`); hovering a label shows the
    full name."""

    def __init__(self):
        super().__init__()
        from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
        from matplotlib.figure import Figure

        self.figure = Figure(figsize=(5, 3), layout="constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        QVBoxLayout(self).addWidget(self.canvas)
        self.layout().setContentsMargins(0, 0, 0, 0)
        self.labels = {}  # short label -> full name, for the tooltips
        self._ticks = []
        self.canvas.mpl_connect("motion_notify_event", self._hover)

    def show(self, z, counts=None):
        from ._cyto_labels import short_names

        self.figure.clf()
        self._ticks = []
        if z is None or z.empty:
            self.labels = {}
            self.canvas.draw_idle()
            return
        ax = self.figure.add_subplot(111)
        im = ax.imshow(z.to_numpy(float), aspect="auto", cmap="RdBu_r", vmin=-2, vmax=2,
                       interpolation="nearest")
        names = [str(c) for c in z.columns]
        short = short_names(names)
        self.labels = {short[n]: n for n in names}
        n = len(names)
        size = 7 if n <= 12 else 6 if n <= 24 else 5
        ax.set_xticks(range(n), [short[c] for c in names], rotation=45 if n <= 12 else 90,
                      ha="right" if n <= 12 else "center", fontsize=size)
        self._ticks = list(zip(ax.get_xticklabels(), names))
        labels = [f"{c} ({int(counts.loc[c, 'objects'])})" if counts is not None else str(c) for c in z.index]
        ax.set_yticks(range(z.shape[0]), labels, fontsize=7)
        for tick, c in zip(ax.get_yticklabels(), z.index):
            tick.set_color(cy.PALETTE[(int(c) - 1) % len(cy.PALETTE)])
        ax.set_ylabel("cluster (objects)", fontsize=8)
        self.figure.colorbar(im, ax=ax, label="z-score of mean", shrink=0.8)
        self.canvas.draw_idle()

    def full_name(self, x, y):
        """The full feature name of the label under canvas point (x, y), or None."""
        renderer = self.canvas.get_renderer()
        for tick, name in self._ticks:
            if tick.get_window_extent(renderer).contains(x, y):
                return name
        return None

    def _hover(self, event):
        name = self.full_name(event.x, event.y) if event.x is not None else None
        self.canvas.setToolTip(name or "")


def _short_markers(z):
    """{cluster: markers text} with the features' short labels (vtea-core's
    describe_clusters on the renamed profile)."""
    from vtea_core.clustering import describe_clusters

    from ._cyto_labels import short_names

    try:
        return describe_clusters(z.rename(columns=short_names([str(c) for c in z.columns])))
    except Exception:  # fall back to the full names
        return {}


def _add_methods(box, methods):
    for label, method in methods.items():
        need = cy.available(method)
        box.addItem(label + (" (not installed)" if need else ""), method)
        if need:
            box.model().item(box.count() - 1).setEnabled(False)
            box.setItemData(box.count() - 1, need, Qt.ToolTipRole)


def _dspin(lo, hi, value, step, tip):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(value)
    s.setToolTip(tip)
    return s


def _qcolor(hex_color):
    from qtpy.QtGui import QColor

    return QColor(hex_color)
