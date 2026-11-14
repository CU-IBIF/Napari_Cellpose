"""The Spatial and Neighbourhoods tabs of the Cytometry workbench (phases 3 and 4).

Once objects have identities (phenotypes, or clusters), these tabs describe each object
by its surroundings. Both run vtea-core (`vtea_core.spatial`, `vtea_core.neighborhoods`)
in the background, and their results become columns of the object table, like a
clustering's: they can be plotted, coloured by, gated, used in phenotype rules and
clustered on, and they are saved with the workspace.

- **Spatial**: per object and per class, the distance to the nearest other object of
  that class (centroids, or exactly between the masks, in µm and in 3D), how many
  objects of each class are within a radius (and what fraction of the neighbours they
  are), and how many touch it.
- **Neighbourhoods**: each object's neighbourhood (itself and the objects within a
  radius, or its k nearest), its class composition, neighbourhood types found by
  k-means on the composition (the CODEX / Schürch-Nolan approach), each type's
  composition, which cell types are enriched in which type, and which classes are
  neighbours of which more than chance (permutation z-scores).
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
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QSplitter,
    QVBoxLayout,
    QWidget,
)

from ._cyto_cluster import Heatmap

MEASURES = {  # check box text -> spatial measure
    "Distance to the nearest (centres)": "nearest",
    "Distance to the nearest (mask to mask)": "surface",
    "Objects within the radius (and fraction)": "within",
    "Contacts (touching masks)": "contacts",
}
VIEWS = ("Composition of each type", "Cell types in each type (log2 enrichment)",
         "Neighbours of each class (z-score)")


class _AnalysisTab(QWidget):
    """Settings on the left (what to class objects by, which classes, a name, Run),
    a result on the right; runs in the background."""

    kind = ""
    base_name = ""

    def __init__(self, bench):
        super().__init__()
        self.bench = bench
        self.state = bench.state
        self._worker = None
        split = QSplitter(Qt.Horizontal)
        QVBoxLayout(self).addWidget(split)
        left = QWidget()
        self.form = QFormLayout(left)
        self.classes = QComboBox()
        self.classes.setToolTip("What each object is: its phenotype (deepest, or at a level), or its "
                                "cluster in a clustering")
        self.classes.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
        self.classes.setMinimumContentsLength(16)
        self.classes.currentIndexChanged.connect(self._fill_targets)
        self.form.addRow("Classes", self.classes)
        self.settings_rows()
        self.name = QLineEdit()
        self.form.addRow("Name", self.name)
        self.run_btn = QPushButton("Run")
        self.run_btn.clicked.connect(lambda _=False: self.run())
        self.form.addRow(self.run_btn)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.setVisible(False)
        self.form.addRow(self.progress)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.form.addRow(self.status)
        split.addWidget(left)
        right = QWidget()
        self.results = QVBoxLayout(right)
        row = QHBoxLayout()
        row.addWidget(QLabel("Results"))
        self.analysis = QComboBox()
        self.analysis.currentTextChanged.connect(self.show_analysis)
        row.addWidget(self.analysis, 1)
        self.results.addLayout(row)
        self.info = QLabel()
        self.info.setWordWrap(True)
        self.info.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.results.addWidget(self.info)
        self.result_rows()
        split.addWidget(right)
        split.setSizes([360, 760])

    def settings_rows(self):
        pass

    def result_rows(self):
        pass

    # ---- refreshing ---------------------------------------------------------------------

    def refresh(self):
        state = self.state
        keep = self.classes.currentData()
        self.classes.blockSignals(True)
        self.classes.clear()
        if state.table is not None and not state.broken_phenotypes():
            for label, key in state.class_choices():
                self.classes.addItem(label, key)
        index = self.classes.findData(keep)
        self.classes.setCurrentIndex(max(index, 0) if self.classes.count() else -1)
        self.classes.blockSignals(False)
        self._fill_targets()
        current = self.analysis.currentText()
        mine = state.analyses_of(self.kind)
        self.analysis.blockSignals(True)
        self.analysis.clear()
        self.analysis.addItems(mine)
        if current in mine:
            self.analysis.setCurrentText(current)
        elif mine:
            self.analysis.setCurrentIndex(len(mine) - 1)
        self.analysis.blockSignals(False)
        if not self.name.text() or self.name.text() in state.analyses:
            self.name.setText(state.unique_analysis_name(self.base_name))
        ready = state.table is not None and self.classes.count() > 0
        self.run_btn.setEnabled(ready and self._worker is None)
        if state.table is not None and not self.classes.count():
            self.status.setText("Make phenotypes (Gating tab) or run a clustering first: these features "
                                "are measured per class.")
        self.show_analysis(self.analysis.currentText())

    def _fill_targets(self, *_):
        pass

    def class_values(self):
        key = self.classes.currentData()
        if key is None:
            return []
        values = {c for c in self.state.classes(key) if c is not None}
        return sorted(values, key=lambda v: (v.startswith("cluster "), int(v[8:]) if v.startswith("cluster ")
                                             and v[8:].isdigit() else 0, v))

    # ---- running ------------------------------------------------------------------------

    def settings(self):
        return {"classes": self.classes.currentData(), "class_label": self.classes.currentText()}

    def make_job(self, settings):
        raise NotImplementedError

    def store(self, name, settings, result):
        raise NotImplementedError

    def run(self, wait=False):
        """Run (in the background unless `wait`). Returns the stored result's info when
        waiting, else the worker."""
        state = self.state
        if state.table is None or self._worker is not None or self.classes.currentData() is None:
            return None
        name = self.name.text().strip() or state.unique_analysis_name(self.base_name)
        try:
            state._check_analysis_name(name)
            settings = self.settings()
            job = self.make_job(settings)
        except (ValueError, KeyError) as e:
            self.status.setText(str(e))
            return None
        started = time.perf_counter()

        def done(result):
            settings["seconds"] = round(time.perf_counter() - started, 2)
            try:
                self.store(name, settings, result)
            except ValueError as e:
                self.status.setText(str(e))
                return
            self.status.setText(f"{name}: done ({settings['seconds']:.1f} s).")
            self.analysis.setCurrentText(name)

        if wait:
            done(job())
            return state.analyses.get(name)
        from napari.qt.threading import thread_worker

        @thread_worker(ignore_errors=True)
        def work():
            return job()

        worker = work()
        worker.returned.connect(done)
        worker.errored.connect(lambda e: self.status.setText(f"Failed: {e}"))
        worker.finished.connect(self._finished)
        self._worker = worker
        self.run_btn.setEnabled(False)
        self.progress.setVisible(True)
        self.status.setText(f"Running on {len(state.table):,} objects…")
        worker.start()
        return worker

    def _finished(self):
        self._worker = None
        self.progress.setVisible(False)
        self.run_btn.setEnabled(self.state.table is not None and self.classes.count() > 0)

    def delete(self):
        name = self.analysis.currentText()
        if name in self.state.analyses:
            orphaned = self.state.remove_analysis(name)
            broken = self.state.broken_phenotypes()
            note = (f" Gates on its columns select nothing now: {', '.join(orphaned)}." if orphaned else "") + \
                   (f" Phenotypes that used it need a new rule: {', '.join(broken)}." if broken else "")
            self.status.setText(f"Deleted {name}.{note}")

    def show_analysis(self, name):
        pass


class SpatialTab(_AnalysisTab):
    """Phase 3: spatial features of each object, by class."""

    kind = "spatial"
    base_name = "Spatial"

    def settings_rows(self):
        self.targets = QListWidget()
        self.targets.setMinimumHeight(90)
        self.targets.setToolTip("Measure towards objects of these classes")
        self.form.addRow("Towards", self.targets)
        self.measures = {}
        for text, measure in MEASURES.items():
            box = QCheckBox(text)
            box.setChecked(measure in ("nearest", "within"))
            self.measures[measure] = box
            self.form.addRow("" if self.measures and len(self.measures) > 1 else "Measure", box)
        self.measures["surface"].setToolTip(
            "Exact distance between the masks, voxel to voxel, in µm (3D included); slower than centres. "
            "Touching objects are one voxel apart.")
        self.measures["contacts"].setToolTip("How many objects of each class touch the object's mask")
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.1, 10000)
        self.radius.setValue(30.0)
        self.radius.setToolTip("For 'Objects within the radius' (in the table's unit, µm when the voxel "
                               "size is known)")
        self.form.addRow("Radius", self.radius)

    def result_rows(self):
        self.columns = QListWidget()
        self.columns.setToolTip("The features made: plot them on the Gating tab (X / Y / Colour), gate on "
                                "them, use them in phenotype rules or cluster on them")
        self.results.addWidget(self.columns, 1)
        row = QHBoxLayout()
        self.plot_btn = QPushButton("Colour the plot by it")
        self.plot_btn.setToolTip("Colour the Gating tab's points by the selected feature")
        self.plot_btn.clicked.connect(lambda _=False: self.colour_plot())
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(lambda _=False: self.delete())
        row.addWidget(self.plot_btn)
        row.addWidget(self.delete_btn)
        self.results.addLayout(row)

    def _fill_targets(self, *_):
        checked = {self.targets.item(i).text() for i in range(self.targets.count())
                   if self.targets.item(i).checkState() == Qt.Checked}
        first = not self.targets.count()
        self.targets.clear()
        for value in self.class_values():
            item = QListWidgetItem(value)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if first or value in checked or not checked else Qt.Unchecked)
            self.targets.addItem(item)

    def checked_targets(self):
        return [self.targets.item(i).text() for i in range(self.targets.count())
                if self.targets.item(i).checkState() == Qt.Checked]

    def settings(self):
        out = super().settings()
        out.update(targets=self.checked_targets(), radius=self.radius.value(),
                   measures=[m for m, box in self.measures.items() if box.isChecked()])
        if not out["targets"]:
            raise ValueError("Tick at least one class to measure towards.")
        return out

    def make_job(self, settings):
        return self.state.spatial_job(settings)

    def store(self, name, settings, result):
        self.state.add_spatial(name, settings, result)

    def show_analysis(self, name):
        state = self.state
        self.columns.clear()
        has = bool(name) and name in state.analyses
        self.plot_btn.setEnabled(has)
        self.delete_btn.setEnabled(has)
        if not has:
            self.info.setText("Measure distances, neighbours and contacts by class here. They become "
                              "columns of the object table, to plot, gate, cluster and write rules with.")
            return
        info = state.analyses[name]
        measured = ", ".join(t for t, m in MEASURES.items() if m in info.get("measures", [])).lower()
        self.info.setText(f"<b>{name}</b>: {info['n_objects']:,} objects by {info.get('class_label', '')}, "
                          f"towards {', '.join(info.get('targets', []))}: {measured}"
                          + (f"; radius {info['radius']:g} {info.get('unit', '')}" if "within" in
                             info.get("measures", []) else "")
                          + (f" ({info['seconds']:.1f} s)" if "seconds" in info else "") + ".")
        for col in info["columns"]["features"]:
            values = state.column(col)
            ok = np.isfinite(values)
            item = QListWidgetItem(f"{col}    median {np.median(values[ok]):.3g}" if ok.any() else
                                   f"{col}    (none)")
            item.setData(Qt.UserRole, col)
            self.columns.addItem(item)
        if self.columns.count():
            self.columns.setCurrentRow(0)

    def colour_plot(self):
        item = self.columns.currentItem()
        if item is None:
            return
        bench = self.bench
        bench.color_by.setCurrentText(item.data(Qt.UserRole))
        bench.tabs.setCurrentIndex(0)


class NeighbourhoodTab(_AnalysisTab):
    """Phase 4: neighbourhood types and enrichment."""

    kind = "neighbourhoods"
    base_name = "Neighbourhoods"

    def settings_rows(self):
        self.method = QComboBox()
        self.method.addItem("Objects within a radius", "radius")
        self.method.addItem("k nearest objects", "nearest")
        self.method.currentIndexChanged.connect(self._enable)
        self.form.addRow("Neighbourhood", self.method)
        self.radius = QDoubleSpinBox()
        self.radius.setRange(0.1, 10000)
        self.radius.setValue(50.0)
        self.radius.setToolTip("Each object's neighbourhood: itself and the objects whose centre is "
                               "within this distance (µm when the voxel size is known)")
        self.form.addRow("Radius", self.radius)
        self.k = QSpinBox()
        self.k.setRange(1, 200)
        self.k.setValue(10)
        self.k.setToolTip("Each object's neighbourhood: itself and its k nearest objects")
        self.form.addRow("k", self.k)
        self.n_types = QSpinBox()
        self.n_types.setRange(2, 30)
        self.n_types.setValue(6)
        self.n_types.setToolTip("How many neighbourhood types to find (k-means on the composition)")
        self.form.addRow("Types", self.n_types)
        self.permutations = QSpinBox()
        self.permutations.setRange(0, 5000)
        self.permutations.setValue(200)
        self.permutations.setToolTip("Permutations of the classes for the neighbour enrichment z-scores "
                                     "(0: skip)")
        self.form.addRow("Permutations", self.permutations)
        self.seed = QSpinBox()
        self.seed.setRange(0, 1_000_000)
        self.form.addRow("Seed", self.seed)
        self._enable()

    def _enable(self, *_):
        nearest = self.method.currentData() == "nearest"
        self.radius.setEnabled(not nearest)
        self.k.setEnabled(nearest)

    def result_rows(self):
        row = QHBoxLayout()
        row.addWidget(QLabel("Show"))
        self.view = QComboBox()
        self.view.addItems(list(VIEWS))
        self.view.currentIndexChanged.connect(lambda _: self.show_analysis(self.analysis.currentText()))
        row.addWidget(self.view, 1)
        self.results.addLayout(row)
        self.heatmap = Heatmap()
        self.results.addWidget(self.heatmap, 1)
        row = QHBoxLayout()
        self.plot_btn = QPushButton("Colour the plot by type")
        self.plot_btn.clicked.connect(lambda _=False: self.colour_plot())
        self.image_btn = QPushButton("Show types on image")
        self.image_btn.clicked.connect(lambda _=False: self.show_on_image())
        self.delete_btn = QPushButton("Delete")
        self.delete_btn.clicked.connect(lambda _=False: self.delete())
        for b in (self.plot_btn, self.image_btn, self.delete_btn):
            row.addWidget(b)
        self.results.addLayout(row)

    def settings(self):
        out = super().settings()
        out.update(method=self.method.currentData(), radius=self.radius.value(), k=self.k.value(),
                   n_types=self.n_types.value(), permutations=self.permutations.value(),
                   seed=self.seed.value())
        return out

    def make_job(self, settings):
        return self.state.neighbourhood_job(settings)

    def store(self, name, settings, result):
        self.state.add_neighbourhoods(name, settings, result)

    def show_analysis(self, name):
        state = self.state
        has = bool(name) and name in state.analyses
        for b in (self.plot_btn, self.image_btn, self.delete_btn):
            b.setEnabled(has)
        if not has:
            self.info.setText("Find neighbourhood types: groups of objects with the same mix of classes "
                              "around them.")
            self.heatmap.show(None)
            return
        info = state.analyses[name]
        how = (f"its {info['k']} nearest" if info.get("method") == "nearest"
               else f"within {info['radius']:g} {state.unit()}")
        sizes = state.analysis_table(name, "sizes")["objects"]
        self.info.setText(f"<b>{name}</b>: {info['n_objects']:,} objects by {info.get('class_label', '')}; "
                          f"each object and the objects {how}; {len(sizes)} types "
                          f"(column <i>{info['columns']['type']}</i>)"
                          + (f" ({info['seconds']:.1f} s)" if "seconds" in info else "") + "."
                          + self._overlap_text(name, info))
        view = self.view.currentIndex()
        counts = sizes.to_frame("objects")
        if view == 0:
            profile = state.analysis_table(name, "profile")
            profile.index = [int(i) for i in profile.index]
            self.heatmap.show(profile, counts.set_axis([int(i) for i in counts.index]),
                              row_label="neighbourhood type (objects)", value_label="fraction of the "
                              "neighbourhood", limits=(0, 1), cmap="viridis", annotate=True)
        elif view == 1:
            table = state.analysis_table(name, "by_type")
            if table.empty:
                self.heatmap.show(None)
                return
            grid = table.pivot(index="type", columns="class", values="log2 enrichment")
            grid.index = [int(i) for i in grid.index]
            self.heatmap.show(grid, counts.set_axis([int(i) for i in counts.index]),
                              row_label="neighbourhood type (objects)",
                              value_label="log2(observed / expected)", limits=(-3, 3), annotate=True)
        else:
            table = state.analysis_table(name, "pairs")
            if table.empty or table["z"].isna().all():
                self.heatmap.show(None)
                self.info.setText(self.info.text() + " No permutations were run.")
                return
            grid = table.pivot(index="class", columns="neighbor", values="z")
            self.heatmap.show(grid, row_label="class (centre)", value_label="z-score (neighbours vs. chance)",
                              limits=(-5, 5), color_rows=False, annotate=True)

    def _overlap_text(self, name, info):
        """Neighbourhoods overlap: how many each object is a member of."""
        col = f"{name} memberships"
        if not info["columns"].get("member_type") or col not in self.state.features():
            return ""
        n = self.state.column(col)
        return (f" Neighbourhoods overlap: each object is a member of {int(n.min())}-{int(n.max())} "
                f"(median {np.median(n):g}); <i>{info['columns']['member_type']}</i> is the most "
                f"common type among them.")

    def colour_plot(self):
        name = self.analysis.currentText()
        if name in self.state.analyses:
            self.bench.color_by.setCurrentText(self.state.analyses[name]["columns"]["type"])
            self.bench.tabs.setCurrentIndex(0)

    def show_on_image(self):
        name = self.analysis.currentText()
        if name not in self.state.analyses:
            return None
        try:
            layer = self.state.show_neighbourhoods(name)
        except ValueError as e:
            self.status.setText(str(e))
            return None
        self.status.setText(f"Shown in layer {layer.name}.")
        return layer


__all__ = ["SpatialTab", "NeighbourhoodTab", "MEASURES", "VIEWS"]
