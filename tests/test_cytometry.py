import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("napari")
pytest.importorskip("vtea_core")

from napari.components import ViewerModel  # noqa: E402
from qtpy.QtCore import Qt  # noqa: E402

from ibif_napari import _cytometry as cy  # noqa: E402
from ibif_napari import _measure, _workspace  # noqa: E402


def scene():
    """A viewer with 2 time points of 3x4 square 'cells' and two channels: CD3 bright in
    the left half, CD8 bright in the top row (so: 6 T cells per frame, 2 of them CD8)."""
    v = ViewerModel()
    labels = np.zeros((2, 60, 80), np.uint16)
    cd3 = np.full(labels.shape, 10.0)
    cd8 = np.full(labels.shape, 10.0)
    k = 0
    for r in range(3):
        for c in range(4):
            k += 1
            sl = (slice(None), slice(r * 20 + 4, r * 20 + 16), slice(c * 20 + 4, c * 20 + 16))
            labels[sl] = k
            if c < 2:
                cd3[sl] = 500
                if r == 0:
                    cd8[sl] = 400
    meta = {"ibif_axes": "TYX", "ibif_voxel_um": {"Y": 0.5, "X": 0.5}}
    common = dict(scale=(1, 0.5, 0.5), metadata=meta)
    images = [v.add_image(cd3, name="CD3", **common), v.add_image(cd8, name="CD8", **common)]
    layer = v.add_labels(labels, name="cells", **common)
    table = _measure.measure_layer(v, layer, images)
    _measure.remember(v, {"cells": table})
    return v, table


def test_gates_subgates_and_phenotype_hierarchy():
    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    assert len(s.table) == 24 and "CD3 mean" in s.features()
    t = s.add_gate("T", cy.axis_column("CD3 mean", "asinh"), "CD8 mean",
                   [[1, 0], [10, 0], [10, 1e4], [1, 1e4]])
    assert s.mask("gate", t.id).sum() == 12
    cd8 = s.add_gate("CD8+", "CD3 mean", "CD8 mean", [[0, 100], [1e4, 100], [1e4, 1e4], [0, 1e4]],
                     population=("gate", t.id))
    assert cd8.parent_id == t.id and s.mask("gate", cd8.id).sum() == 4
    s.phenotype_from("gate", t.id, "T cell")
    s.phenotype_from("gate", cd8.id, "CD8 T", parent="T cell")
    s.add_phenotype("Other T", "", parent="T cell")
    a = s.assignment()
    assert a["phenotype"].value_counts().to_dict() == {"unlabelled": 12, "Other T": 8, "CD8 T": 4}
    assert set(a.loc[a["phenotype"] == "CD8 T", "phenotype path"]) == {"T cell > CD8 T"}
    counts = s.counts().set_index("phenotype")
    assert counts.loc["CD8 T", "% of parent"] == pytest.approx(100 / 3)
    # Gates drawn within a phenotype only see its objects.
    g = s.add_gate("bright CD8", "CD3 mean", "CD8 mean", [[0, 0], [1e4, 0], [1e4, 1e4], [0, 1e4]],
                   population=("phenotype", "CD8 T"))
    assert s.mask("gate", g.id).sum() == 4
    # Renaming a gate keeps the rules that use it.
    s.rename_gate("gate", t.id, "T cells")
    assert s.tree.get("T cell").rule == "`gate: T cells`" and s.assignment().equals(a)
    with pytest.raises(ValueError):
        s.add_phenotype("bad", "`no such column` > 1")
    assert "bad" not in s.tree


def test_image_gate_and_showing_on_the_image():
    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    # The left two columns of cells: x < 20 µm (pixels 0-40).
    s.add_image_gate("left", [[[0, 0], [0, 20], [30, 20], [30, 0]]])
    assert s.mask("image", "left").sum() == 12
    s.add_phenotype("Left", "`image gate: left`", color="#00ff00")
    layer = s.show_phenotypes()
    assert layer.name == "cells phenotypes" and layer.data.shape == (2, 60, 80)
    assert layer.data[1, 10, 10] == 1 and layer.data[1, 10, 70] == 0
    sel = s.show_population("image", "left")
    assert sel.name == "cells selection" and sel.data[0, 30, 30] == 1
    typed = s.typed_table()
    assert {"image gate: left", "phenotype", "phenotype level 1"} <= set(typed.columns)


def test_saved_with_the_workspace(tmp_path):
    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    t = s.add_gate("T", "CD3 mean", "CD8 mean", [[100, 0], [1e4, 0], [1e4, 1e4], [100, 1e4]])
    s.phenotype_from("gate", t.id, "T cell")
    s.add_image_gate("top", [[[0, 0], [0, 40], [10, 40], [10, 0]]], population=("phenotype", "T cell"))
    expected = s.typed_table()
    folder, notes = _workspace.save_workspace(v, tmp_path / "w")
    assert notes == [] and "cytometry" in json.loads((folder / "workspace.json").read_text())["extensions"]
    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    restored = cy.state_of(w)
    assert restored.source == "cells" and restored.layer_name == "cells"
    pd.testing.assert_frame_equal(restored.typed_table(), expected, check_dtype=False)
    assert restored.scope[("image", "top")] == ("phenotype", "T cell")


def test_section_and_workbench(qtbot, tmp_path):
    from ibif_napari._cyto_workbench import Workbench
    from ibif_napari._cytometry_widget import CytometryWidget

    v, table = scene()
    section = CytometryWidget(v)
    qtbot.addWidget(section)
    assert section.table_box.currentText() == "cells" and section.layer_box.currentText() == "cells"
    section.use()
    assert "24 objects" in section.summary.text()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    bench.x_col.setCurrentText("CD3 mean")
    bench.y_col.setCurrentText("CD8 mean")
    gate = bench.add_gate("T", [[100, 0], [1e4, 0], [1e4, 1e4], [100, 1e4]])
    assert gate is not None and bench.gate_tree.topLevelItemCount() == 1
    assert bench.gate_tree.topLevelItem(0).text(1) == "12"
    # A drawn gate is mapped on the segmented pixels straight away (no extra button).
    sel = v.layers["cells selection"].data
    assert (sel > 0).any() and ((sel > 0) <= (v.layers["cells"].data > 0)).all()
    assert not hasattr(bench, "gate_show_btn")
    bench.gate_tree.setCurrentItem(bench.gate_tree.topLevelItem(0))
    assert bench.phenotype_from_gate(name="T cell") is not None
    assert section.tree.topLevelItem(0).text(0) == "T cell"  # the section follows
    # Subgate: show the gate, draw within it.
    bench.set_population("gate", gate.id)
    sub = bench.add_gate("CD8+", [[0, 100], [1e4, 100], [1e4, 1e4], [0, 1e4]])
    assert sub.parent_id == gate.id
    assert bench.gate_tree.topLevelItem(0).child(0).text(1) == "4"
    bench.color_by.setCurrentText("Phenotype")
    bench.replot()
    # Clicking a point goes to the object; selecting one in the viewer rings it on the plot.
    bench.go_to_row(5)
    assert v.layers["cells"].selected_label == int(table.iloc[5]["label"])
    path = bench.export_table(str(tmp_path / "typed.csv"))
    out = pd.read_csv(path)
    assert (out["phenotype"] == "T cell").sum() == 12
    section.level.setValue(1)
    section.show_phenotypes()
    assert "cells phenotypes level 1" in v.layers


def cell_types(n=240, seed=0):
    """A table of n objects of 3 known types: CD8 T (CD3+CD8+), CD4 T (CD3+CD4+), other."""
    rng = np.random.default_rng(seed)
    kind = np.repeat([0, 1, 2], n // 3)
    table = pd.DataFrame({
        "label": np.arange(1, n + 1),
        "CD3 mean": np.where(kind < 2, 800, 50) * rng.lognormal(0, 0.15, n),
        "CD8 mean": np.where(kind == 0, 600, 40) * rng.lognormal(0, 0.15, n),
        "CD4 mean": np.where(kind == 1, 500, 30) * rng.lognormal(0, 0.15, n),
        "volume (µm³)": rng.normal(500, 50, n),
    })
    return table, kind


def test_clustering_finds_types_and_labels_them():
    from sklearn.metrics import adjusted_rand_score

    table, kind = cell_types()
    v = ViewerModel()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    feats = ["CD3 mean", "CD8 mean", "CD4 mean"]
    rows, m = s.matrix(("all", None), feats, asinh=True)
    labels, emb = cy.compute_analysis(m, "pca", "kmeans", k=3, seed=1)
    assert adjusted_rand_score(kind, labels) == pytest.approx(1.0)
    info = s.add_analysis("Clustering 1", {"population": ["all", None], "features": feats, "asinh": True,
                                           "map": "PCA"}, rows, labels, emb)
    assert info["columns"] == {"cluster": "Clustering 1 cluster", "map": ["Clustering 1 PCA 1", "Clustering 1 PCA 2"]}
    assert "Clustering 1 PCA 1" in s.features() and "Clustering 1 PCA 1" not in s.measured_features()
    clusters = s.clusters("Clustering 1")
    assert sorted(np.unique(clusters)) == [1, 2, 3]  # numbered by size
    profiles = s.profiles("Clustering 1")
    desc = profiles["description"]
    cd8 = next(c for c, d in desc.items() if d.startswith("CD8 mean+"))
    cd4 = next(c for c, d in desc.items() if d.startswith("CD4 mean+"))
    other = ({1, 2, 3} - {cd8, cd4}).pop()
    s.add_phenotype("T cell", "`CD3 mean` > 300")
    made = s.label_clusters("Clustering 1", {cd8: "CD8 T", cd4: "CD4 T"}, parent="T cell")
    assert [p.name for p in made] == ["CD8 T", "CD4 T"]
    a = s.assignment()
    assert (a["phenotype"].to_numpy()[kind == 0] == "CD8 T").all()
    assert (a["phenotype"].to_numpy()[kind == 2] == "unlabelled").all()
    # Clusters given the same name merge into one phenotype.
    made = s.label_clusters("Clustering 1", {other: "Rest", cd8: "Rest"})
    assert made[0].rule == f"`Clustering 1 cluster` in [{', '.join(map(str, sorted([other, cd8])))}]"
    with pytest.raises(ValueError):
        s.label_clusters("Clustering 1", {cd4: "CD4 T"})  # name taken


def test_reclustering_within_a_phenotype_and_saving(tmp_path):
    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    s.add_phenotype("T cell", "`CD3 mean` > 100")
    rows, m = s.matrix(("phenotype", "T cell"), ["CD8 mean", "CD3 mean"])
    assert len(rows) == 12
    labels, _ = cy.compute_analysis(m, None, "kmeans", k=2, seed=0)
    s.add_analysis("T split", {"population": ["phenotype", "T cell"], "features": ["CD8 mean", "CD3 mean"]},
                   rows, labels)
    clusters = s.clusters("T split")
    assert (clusters[rows] > 0).all() and (np.delete(clusters, rows) == 0).all()
    layer = s.show_clusters("T split")
    assert layer.name == "cells T split" and layer.data.max() == 2
    expected = s.typed_table()
    folder, notes = _workspace.save_workspace(v, tmp_path / "w")
    assert notes == [] and (folder / "cytometry" / "derived.csv").exists()
    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    restored = cy.state_of(w)
    assert list(restored.analyses) == ["T split"]
    pd.testing.assert_frame_equal(restored.typed_table(), expected, check_dtype=False)
    restored.remove_analysis("T split")
    assert restored.derived is None


def test_clustering_tab(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    table, kind = cell_types()
    v = ViewerModel()
    cy.state_of(v).set_table("cells", table, "cells")
    bench = Workbench(v)
    qtbot.addWidget(bench)
    tab = bench.clustering
    assert tab.checked_features() == ["CD3 mean", "CD8 mean", "CD4 mean"]  # intensities by default
    tab.map.setCurrentIndex(tab.map.findData("pca"))
    tab.method.setCurrentIndex(tab.method.findData("kmeans"))
    tab.k.setValue(3)
    assert not tab.reproducible.isEnabled()  # a UMAP option
    info = tab.run(wait=True)
    assert info is not None and tab.analysis.currentText() == "Clustering 1"
    assert "s)." in tab.status.text() and info["seconds"] >= 0 and info["reproducible"] is False
    assert tab.table.rowCount() == 3 and "+" in tab.table.item(0, 2).text()
    assert tab.name.text() == "Clustering 2"  # ready for the next run
    profiles = cy.state_of(v).profiles("Clustering 1")
    cd8 = next(c for c, d in profiles["description"].items() if d.startswith("CD8 mean+"))
    tab.set_names({cd8: "CD8 T"})
    made = tab.make_phenotypes()
    assert [p.name for p in made] == ["CD8 T"] and bench.pheno_tree.topLevelItemCount() == 1
    tab.bench.plot_map("Clustering 1")
    assert bench.x_col.currentText() == "Clustering 1 PCA 1" and bench.color_by.currentText() == "Clustering 1 cluster"
    assert bench.tabs.currentIndex() == 0
    # Re-cluster within the phenotype: the Objects box lists it.
    tab.population.setCurrentIndex(tab.population.findText("Phenotype: CD8 T"))
    tab.method.setCurrentIndex(tab.method.findData("kmeans"))
    tab.map.setCurrentIndex(tab.map.findData(None))
    tab.k.setValue(2)
    assert tab.run(wait=True)["n_objects"] == 80


def test_fast_routes_share_one_neighbour_search(monkeypatch):
    """Leiden and UMAP together search the neighbours once; Leiden gives the same
    clusters as the slower reference; a non-reproducible map runs (on every core);
    an older vtea-core without the fast routes still works."""
    pytest.importorskip("igraph")
    pytest.importorskip("umap")
    from sklearn.metrics import adjusted_rand_score
    from vtea_core import clustering as cl

    if not cy._fast_vtea():
        pytest.skip("vtea-core without nearest_neighbors")
    table, kind = cell_types()
    m = np.arcsinh(table[["CD3 mean", "CD8 mean", "CD4 mean"]].to_numpy() / cy.ASINH_COFACTOR)
    calls = []
    real = cl.nearest_neighbors
    monkeypatch.setattr(cl, "nearest_neighbors", lambda *a, **k: calls.append(1) or real(*a, **k))
    labels, emb = cy.compute_analysis(m, "umap", "leiden", seed=0)
    assert len(calls) == 1 and emb.shape == (len(m), 2)
    # Every cluster is one cell type (igraph's Leiden may split a large uniform type in
    # two at resolution 1: that partition has the higher modularity); at a lower
    # resolution the three types come out exactly.
    from sklearn.metrics import homogeneity_score

    assert homogeneity_score(kind, labels) == pytest.approx(1.0)
    low, _ = cy.compute_analysis(m, None, "leiden", seed=0, resolution=0.5)
    assert adjusted_rand_score(kind, low) == pytest.approx(1.0)
    again = cy.compute_analysis(m, "umap", None, seed=0)[1]
    np.testing.assert_allclose(emb, again)  # reproducible: same seed, same map
    labels2, fast = cy.compute_analysis(m, "umap", "leiden", seed=0, reproducible=False)
    assert fast.shape == emb.shape and np.isfinite(fast).all()
    np.testing.assert_array_equal(labels, labels2)  # clusters do not depend on the map
    monkeypatch.setattr(cy, "_fast_vtea", lambda: False)
    before = len(calls)
    old, _ = cy.compute_analysis(m, None, "leiden", seed=0, resolution=0.5)
    assert adjusted_rand_score(kind, old) == pytest.approx(1.0) and len(calls) == before


def test_umap_warm_up(monkeypatch):
    pytest.importorskip("umap")
    monkeypatch.delenv("IBIF_NAPARI_NO_WARMUP", raising=False)
    monkeypatch.setitem(cy._warmed, "umap", False)
    thread = cy.warm_up()
    assert thread is not None and cy.warm_up() is None  # once per session
    thread.join(120)
    assert not thread.is_alive()
    monkeypatch.setenv("IBIF_NAPARI_NO_WARMUP", "1")
    monkeypatch.setitem(cy._warmed, "umap", False)
    assert cy.warm_up() is None and not cy._warmed["umap"]


def test_measuring_makes_the_objects_available_in_cytometry(qtbot):
    """Both sections exist before anything is measured (as in the IBIF panel); measuring
    must offer the objects to Cytometry and choose them, and measuring again must update
    them without losing the gates."""
    from ibif_napari._cytometry_widget import CytometryWidget
    from ibif_napari._measure_widget import COMBINED, MeasureWidget

    v = ViewerModel()
    labels = np.zeros((40, 60), np.uint16)
    cd3 = np.full(labels.shape, 10.0)
    for k, x in enumerate((5, 20, 35, 50), start=1):
        labels[10:20, x - 4:x + 4] = k
        labels[25:35, x - 4:x + 4] = k + 4
        cd3[(labels == k)] = 500  # the top row is CD3+
    v.add_image(cd3, name="CD3")
    v.add_image(cd3 / 2, name="CD8")
    v.add_labels(labels, name="nuclei")
    v.add_labels(labels, name="cells")

    measure, cyto = MeasureWidget(v), CytometryWidget(v)
    qtbot.addWidget(measure)
    qtbot.addWidget(cyto)
    assert cyto.table_box.count() == 0 and not cyto.use_btn.isEnabled()

    measure.check(objects=["cells"], channels=["CD3", "CD8"])
    measure.measure()
    qtbot.waitUntil(lambda: measure._worker is None, timeout=30000)
    assert cyto.table_box.currentText() == "cells" and cyto.use_btn.isEnabled()
    assert cyto.layer_box.currentText() == "cells"
    state = cy.state_of(v)
    assert state.table is not None and len(state.table) == 8  # chosen without a click
    assert "8 objects" in cyto.summary.text()

    gate = state.add_gate("CD3+", "CD3 mean", "CD8 mean", [[100, 0], [1e4, 0], [1e4, 1e4], [100, 1e4]])
    assert state.mask("gate", gate.id).sum() == 4

    # Measuring more objects: the joined table is offered, the objects in use are updated
    # and the gate still applies.
    measure.check(objects=["nuclei", "cells"], channels=["CD3", "CD8"])
    measure.measure()
    qtbot.waitUntil(lambda: measure._worker is None, timeout=30000)
    names = [cyto.table_box.itemText(i) for i in range(cyto.table_box.count())]
    assert names == ["nuclei", "cells", COMBINED]
    assert state.source == "cells" and len(state.table) == 8
    assert state.mask("gate", gate.id).sum() == 4


def test_reopening_a_workspace_with_the_sections_open_keeps_the_analysis(qtbot, tmp_path):
    """Restored tables reach the open sections first; the saved gates, phenotypes and
    clustering must still be the ones in use afterwards."""
    from ibif_napari._cytometry_widget import CytometryWidget
    from ibif_napari._measure_widget import MeasureWidget

    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    s.add_phenotype("T cell", "`CD3 mean` > 100")
    rows, m = s.matrix(("phenotype", "T cell"), ["CD8 mean", "CD3 mean"])
    labels, _ = cy.compute_analysis(m, None, "kmeans", k=2, seed=0)
    s.add_analysis("T split", {"population": ["phenotype", "T cell"], "features": ["CD8 mean", "CD3 mean"]},
                   rows, labels)
    expected = s.typed_table()
    folder, _ = _workspace.save_workspace(v, tmp_path / "w")

    w = ViewerModel()
    measure, cyto = MeasureWidget(w), CytometryWidget(w)
    qtbot.addWidget(measure)
    qtbot.addWidget(cyto)
    assert _workspace.load_workspace(w, folder) == []
    restored = cy.state_of(w)
    assert restored.source == "cells" and list(restored.analyses) == ["T split"]
    pd.testing.assert_frame_equal(restored.typed_table(), expected, check_dtype=False)
    assert cyto.table_box.currentText() == "cells" and cyto.use_btn.isEnabled()


def test_joined_table_of_several_layers():
    """Nuclei and cells measured together: Cytometry uses the joined "All objects" table,
    whose columns are prefixed with the layer. Features are the measurements of the
    segmented objects, and positions come from the chosen layer's own centroids."""
    from ibif_napari._measure_widget import COMBINED

    v, cells = scene()
    nuclei = np.where(v.layers["cells"].data > 0, 0, 0).astype(np.uint16)
    data = v.layers["cells"].data
    for k in range(1, 13):  # a smaller nucleus inside each cell, same number
        ys, xs = np.nonzero(data[0] == k)
        nuclei[:, ys.min() + 3:ys.max() - 2, xs.min() + 3:xs.max() - 2] = k
    layer = v.add_labels(nuclei, name="nuclei", scale=(1, 0.5, 0.5),
                         metadata=v.layers["cells"].metadata)
    tables = {"cells": cells, "nuclei": _measure.measure_layer(v, layer, [v.layers["CD3"]])}
    tables[COMBINED] = _measure.combine(tables)
    s = cy.state_of(v)
    s.set_table(COMBINED, tables[COMBINED], "cells")
    feats = s.features()
    assert "cells: area (µm²)" in feats and "nuclei: CD3 mean" in feats
    assert not any(f.endswith("voxels") or f in ("T", "label") for f in feats)
    assert not any("centroid" in f for f in s.measured_features())
    assert s.centroid_columns() == {"Y": "cells: centroid Y (µm)", "X": "cells: centroid X (µm)"}
    assert "24 objects" in s.describe() and "cells" in s.describe()
    s.add_image_gate("left", [[[0, 0], [0, 20], [30, 20], [30, 0]]])
    assert s.mask("image", "left").sum() == 12
    shown = s.show_population("image", "left").data
    assert (shown > 0).sum() == ((data > 0) & (np.arange(80) < 40)[None, None, :]).sum()
    # A layer that does not hold the table's objects is reported.
    s.set_table(COMBINED, tables[COMBINED], "cells")
    empty = v.add_labels(np.zeros_like(data), name="other", scale=(1, 0.5, 0.5))
    s.layer_name = empty.name
    assert "Only 0 of the 12 labels" in s.describe()


def test_several_gates_on_the_image_at_once(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    left = s.add_image_gate("left", [[[0, 0], [0, 20], [30, 20], [30, 0]]], color="#00ff00")  # 12 cells
    top = s.add_gate("top", "CD3 mean", "CD8 mean", [[0, 100], [1e4, 100], [1e4, 1e4], [0, 1e4]],
                     color="#ff0000")  # the CD8-bright row: 4 cells, all also in "left"
    bench = Workbench(v)
    qtbot.addWidget(bench)
    bench.select_gates([("image", left), ("gate", top.id)])
    assert bench.selected_gates() == [("image", left), ("gate", top.id)]
    assert not bench.gate_pheno_btn.isEnabled() and bench.gate_delete_btn.isEnabled()
    layer = v.layers["cells selection"]
    codes = np.unique(layer.data)
    assert set(codes) == {0, 1, 3}  # left only, and both (top is inside left)
    assert tuple(layer.colormap.map(1)[:3]) == pytest.approx((0, 1, 0))
    assert tuple(layer.colormap.map(3)[:3]) == pytest.approx((1, 1, 1))  # in both: white
    assert "2 gates shown" in bench.status.text() and "4 in more than one" in bench.status.text()
    # One gate alone: its own colour, and the phenotype / rename buttons come back.
    bench.select_gates([("gate", top.id)])
    assert bench.gate_pheno_btn.isEnabled() and set(np.unique(v.layers["cells selection"].data)) == {0, 1}
    # Deleting several at once.
    bench.select_gates([("image", left), ("gate", top.id)])
    bench.delete_gate()
    assert not s.order


def test_short_feature_labels():
    from ibif_napari._cyto_labels import short_names

    stem = "Kidney_Cortex_Human_1-1"
    names = [f"cells: {stem} DAPI mean", f"nuclei: {stem} DAPI median", f"cells: {stem} Alexa Fluor 647 std",
             f"nuclei: {stem} Oregon Green 488 BAPTA-1 mean", "cells: volume (µm³)",
             "nuclei: equivalent diameter (µm)", "cells: centroid X (µm)"]
    short = short_names(names)
    assert short[names[0]] == "Cell DAPI mean" and short[names[1]] == "Nuc DAPI med"
    assert short[names[2]] == "Cell Alexa 647 sd"  # first and last words of the dye
    assert short[names[3]] == "Nuc Oregon… mean"
    assert short[names[4]] == "Cell vol" and short[names[5]] == "Nuc diam" and short[names[6]] == "Cell X"
    assert all(len(v) <= 18 for v in short.values()) and len(set(short.values())) == len(names)
    # One layer only: no layer prefix; nothing shared: names kept.
    assert short_names(["CD3 mean", "CD8 median", "volume (µm³)"]) == \
        {"CD3 mean": "CD3 mean", "CD8 median": "CD8 med", "volume (µm³)": "vol"}
    long = "a custom measurement with a very long name"
    assert short_names([long])[long].endswith("…") and len(short_names([long])[long]) <= 18
    # Labels that would collide fall back to (cut) full names.
    clash = short_names(["x: A B C mean", "x: A D C mean"], max_len=8)
    assert len(set(clash.values())) == 2


def test_heatmap_fits_long_names_and_shows_them_on_hover(qtbot):
    from ibif_napari._cyto_cluster import Heatmap

    z = pd.DataFrame(np.random.default_rng(0).normal(size=(3, 6)), index=[1, 2, 3],
                     columns=[f"cells: Some_Long_File_Name_2026 Channel{k} mean" for k in range(6)])
    counts = pd.DataFrame({"objects": [10, 20, 30]}, index=[1, 2, 3])
    h = Heatmap()
    qtbot.addWidget(h)
    h.resize(600, 400)
    h.show(z, counts)
    h.canvas.draw()
    labels = [t.get_text() for t in h.figure.axes[0].get_xticklabels()]
    assert labels[0] == "Channel0 mean"
    tick, full = h._ticks[2]
    box = tick.get_window_extent(h.canvas.get_renderer())
    assert h.full_name((box.x0 + box.x1) / 2, (box.y0 + box.y1) / 2) == full == z.columns[2]
    assert h.full_name(-50, -50) is None


def test_points_menu_and_window_shape(qtbot):
    from ibif_napari._cyto_plot import ASPECT, STYLES
    from ibif_napari._cyto_workbench import Workbench

    v, table = scene()
    cy.state_of(v).set_table("cells", table, "cells")
    bench = Workbench(v)
    qtbot.addWidget(bench)
    plot = bench.plot
    titles = [a.text() for a in plot.points_menu.actions() if not a.isSeparator()]
    assert titles == ["Size", "Style", "Opacity", "Ring objects in image gates"]
    plot.set_point_style(point_size=30.0, marker="+", alpha=0.4)
    sc = plot.axes.collections[-1]
    assert sc.get_sizes()[0] == 30.0 and sc.get_alpha() == 0.4 and sc.get_linewidths()[0] > 0  # + is lines
    assert plot._actions[("marker", "Plus")].isChecked() and not plot._actions[("marker", "Dot")].isChecked()
    plot._actions[("marker", "Square")].trigger()
    assert plot.marker == STYLES["Square"] and plot.axes.collections[-1].get_linewidths()[0] == 0
    bench.x_col.setCurrentText("CD8 mean")  # replotting keeps the chosen style
    assert plot.axes.collections[-1].get_sizes()[0] == 30.0
    for avail in ((1920, 1080), (1366, 768), (2560, 1440)):
        plot_w, side_w = bench.fit_to_screen(avail)
        assert bench.width() <= avail[0] and bench.height() <= avail[1]
        assert bench.width() >= bench.minimumWidth() and bench.height() >= bench.minimumHeight()
        assert side_w >= plot_w // 2 and (side_w == plot_w // 2 or side_w == max(
            bench.side.minimumSizeHint().width(), 260))
    bench.show()
    bench.fit_to_screen((1920, 1080))
    qtbot.waitExposed(bench)
    canvas = plot.canvas
    assert canvas.width() / canvas.height() == pytest.approx(ASPECT, abs=0.02)


def _rules_state():
    pytest.importorskip("vtea_core.classes.expression")
    from vtea_core import classes

    if not hasattr(classes, "column_references"):
        pytest.skip("vtea-core without column_references")
    table, kind = cell_types()
    table = table.rename(columns={"CD3 mean": "cells: s CD3 mean", "CD8 mean": "cells: s CD8 mean",
                                  "CD4 mean": "cells: s CD4 mean"})
    table["nuclei: s CD8 mean"] = table["cells: s CD8 mean"] * 0.9
    s = cy.state_of(ViewerModel())
    s.set_table("All objects", table, "cells")
    feats = ["cells: s CD3 mean", "cells: s CD8 mean", "cells: s CD4 mean"]
    rows, m = s.matrix(("all", None), feats, asinh=True)
    labels, _ = cy.compute_analysis(m, None, "kmeans", k=3, seed=1)
    s.add_analysis("Clustering 1", {"population": ["all", None], "features": feats,
                                    "clustering": "kmeans", "method": "k-means"}, rows, labels)
    s.add_gate("CD3+", "cells: s CD3 mean", "cells: s CD8 mean", [[300, 0], [1e5, 0], [1e5, 1e5], [300, 1e5]])
    return s, kind


def test_rules_combine_clusters_measurements_and_gates():
    s, kind = _rules_state()
    cd8 = int(pd.Series(s.clusters("Clustering 1")).groupby(kind).agg(lambda x: x.mode()[0])[0])
    # The rule from the request, written with the table's names: exactly the CD8 T cells.
    s.add_phenotype("CD8 T", f"`Clustering 1 cluster` == {cd8} & `cells: s CD4 mean` < 100")
    assert s.mask("phenotype", "CD8 T").sum() == (kind == 0).sum()
    # Quotes, the heat-map label, a gate by its name, case: all read, stored exactly.
    rule, read = s.resolve_rule(f"'Clustering 1 cluster' == {cd8} & 'Cell CD4 mean' < 100 & 'cd3+'")
    assert rule == f"`Clustering 1 cluster` == {cd8} & `cells: s CD4 mean` < 100 & `gate: CD3+`"
    assert read == {"Cell CD4 mean": "cells: s CD4 mean", "cd3+": "gate: CD3+"}
    # Words of the name, when only one column has them all.
    assert s.resolve_rule("'CD4 mean' > 1")[0] == "`cells: s CD4 mean` > 1"
    # Ambiguous: not guessed, the candidates are given.
    with pytest.raises(ValueError, match="could be any of `cells: s CD8 mean`, `nuclei: s CD8 mean`"):
        s.resolve_rule("'intensity-CD8' < 100")
    with pytest.raises(ValueError, match="no column 'CD99'"):
        s.resolve_rule("'CD99' > 1")
    with pytest.raises(ValueError, match="backticks or quotes"):
        s.resolve_rule("Clustering 1 cluster == 2")
    # A text value after a comparison is left as it is.
    assert s.resolve_rule("'gate: CD3+' == 'x'")[0] == "`gate: CD3+` == 'x'"
    # Counting for the dialog: overall and within a parent.
    # (Top-level phenotypes are exclusive: "CD8 T", made first, would keep its cells from "T".)
    s.remove_phenotype("CD8 T")
    s.add_phenotype("T", "`gate: CD3+`")
    total, within, _ = s.rule_count("'Cell CD8 mean' > 300", parent="T")
    assert total == (kind == 0).sum() and within == (kind == 0).sum()
    # Renaming a gate rewrites the rules through the parser, not a text replace.
    s.add_phenotype("Not T", "NOT `gate: CD3+` AND 'gate: CD3+' != 'gate: CD3+'")
    s.rename_gate("gate", next(iter(s.gates)).id, "T gate")
    assert s.tree.get("T").rule == "`gate: T gate`"
    assert s.tree.get("Not T").rule == "NOT `gate: T gate` AND `gate: T gate` != 'gate: CD3+'"


def test_phenotype_dialog_checks_the_rule_while_typing(qtbot):
    from ibif_napari._cyto_workbench import PhenotypeDialog

    s, kind = _rules_state()
    d = PhenotypeDialog(s, title="New phenotype")
    qtbot.addWidget(d)
    ok = d.buttons.button(d.buttons.Ok)
    d.rule.setText("'intensity-CD8' < 100")
    assert "✗" in d.check.text() and "could be any of" in d.check.text() and not ok.isEnabled()
    d.rule.setText("'Cell CD8 mean' > 300")
    assert f"<b>{(kind == 0).sum()}</b> of 240 objects" in d.check.text() and ok.isEnabled()
    assert "stored as" in d.check.text()
    d.rule.setText("")
    assert "the rest of the objects" in d.check.text() and ok.isEnabled()
    # The column list is grouped, and inserts the full name.
    items = [d.insert.itemText(i).strip() for i in range(d.insert.count())]
    assert "— Gates —" in items and "— Clusters, maps and spatial —" in items and "Cell CD8 mean" in items
    d.insert.activated.emit(items.index("gate: CD3+"))
    d._insert_operator("AND")
    assert d.rule.text().strip() == "`gate: CD3+` AND"


def test_a_gate_on_a_deleted_map_does_not_break_the_workbench(qtbot):
    """A gate drawn on a clustering's map, after the clustering result is deleted (or
    cleared by measuring other objects): the workbench used to recurse forever
    looking for the map's column."""
    from ibif_napari._cyto_workbench import Workbench

    table, kind = cell_types()
    v = ViewerModel()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    feats = ["CD3 mean", "CD8 mean"]

    def cluster():
        rows, m = s.matrix(("all", None), feats, asinh=True)
        labels, emb = cy.compute_analysis(m, "pca", "kmeans", k=3, seed=0)
        s.add_analysis("Clustering 1", {"population": ["all", None], "features": feats, "map": "UMAP",
                                        "clustering": "kmeans"}, rows, labels, emb)

    cluster()
    gate = s.add_gate("on map", "Clustering 1 UMAP 1", cy.axis_column("Clustering 1 UMAP 2", "asinh"),
                      [[-99, -99], [99, -99], [99, 99], [-99, 99]])
    assert s.mask("gate", gate.id).sum() == 240
    with pytest.raises(KeyError):
        s.column("no such column")
    with pytest.raises(KeyError):
        s.column("log10(no such column)")
    assert s.remove_analysis("Clustering 1") == ["on map"]
    assert s.mask("gate", gate.id).sum() == 0
    assert set(s.missing_gate_columns()[("gate", gate.id)]) == {"Clustering 1 UMAP 1",
                                                               "asinh(Clustering 1 UMAP 2/150)"}
    bench = Workbench(v)  # used to raise RecursionError
    qtbot.addWidget(bench)
    item = bench.gate_tree.topLevelItem(0)
    assert item.text(1) == "missing column" and "Clustering 1 UMAP 1" in item.toolTip(0)
    assert "select nothing" in bench.status.text()
    # Running a clustering of the same name brings the gate back.
    cluster()
    assert s.mask("gate", gate.id).sum() == 240 and not s.missing_gate_columns()
    assert bench.gate_tree.topLevelItem(0).text(1) == "240"
    # Measuring other objects clears the results, and says so.
    s.set_table("cells", table.iloc[:200], "cells")
    assert s.dropped_analyses == ["Clustering 1"] and s.mask("gate", gate.id).sum() == 0


def test_reopening_a_workspace_restores_the_map_gates_and_the_plot(qtbot, tmp_path):
    """The reported bug: a workspace with a gate drawn on a UMAP map. Reopening it
    told the Cytometry panel about the gates before the map's columns were read back,
    the panel's refresh recursed forever, and the map columns were never restored."""
    from ibif_napari._cyto_workbench import Workbench
    from ibif_napari._cytometry_widget import CytometryWidget
    from ibif_napari._measure_widget import MeasureWidget

    v, table = scene()
    measure, cyto = MeasureWidget(v), CytometryWidget(v)
    qtbot.addWidget(measure)
    qtbot.addWidget(cyto)
    measure._measured({"cells": table})
    s = cyto.state
    feats = ["CD3 mean", "CD8 mean"]
    rows, m = s.matrix(("all", None), feats, asinh=True)
    labels, emb = cy.compute_analysis(m, "pca", "kmeans", k=2, seed=0)
    s.add_analysis("Clustering 1", {"population": ["all", None], "features": feats, "map": "UMAP",
                                    "clustering": "kmeans"}, rows, labels, emb)
    gate = s.add_gate("map gate", "Clustering 1 UMAP 1", "Clustering 1 UMAP 2",
                      [[-99, -99], [99, -99], [99, 99], [-99, 99]])
    s.add_phenotype("In map gate", "`gate: map gate` & `Clustering 1 cluster` == 1")
    expected = int(s.mask("phenotype", "In map gate").sum())
    bench = Workbench(v)
    qtbot.addWidget(bench)
    bench.x_col.setCurrentText("CD8 mean")
    bench.y_tf.setCurrentText("asinh")
    bench.plot.set_point_style(point_size=14.0, marker="D", alpha=0.4)
    folder, _ = _workspace.save_workspace(v, tmp_path / "w")

    w = ViewerModel()
    measure2, cyto2 = MeasureWidget(w), CytometryWidget(w)  # the panels are open, as in the app
    qtbot.addWidget(measure2)
    qtbot.addWidget(cyto2)
    assert _workspace.load_workspace(w, folder) == []
    s2 = cy.state_of(w)
    assert "Clustering 1 UMAP 1" in s2.derived.columns and not s2.missing_gate_columns()
    assert s2.mask("gate", next(iter(s2.gates)).id).sum() == s.mask("gate", gate.id).sum()
    assert s2.mask("phenotype", "In map gate").sum() == expected
    bench2 = Workbench(w)  # used to raise RecursionError
    qtbot.addWidget(bench2)
    assert bench2.x_col.currentText() == "CD8 mean" and bench2.y_tf.currentText() == "asinh"
    assert (bench2.plot.point_size, bench2.plot.marker, bench2.plot.alpha) == (14.0, "D", 0.4)


def test_the_warm_up_thread_starts_only_after_its_imports(tmp_path):
    """Opening the workbench started the UMAP warm-up thread, which imported
    umap / scikit-learn while the Clustering tab imported scikit-learn in the main
    thread: on Windows this failed with "cannot import name 'clone' from partially
    initialized module 'sklearn.base'". Now every import is made in the calling
    thread before the thread starts. Checked in a fresh interpreter, where nothing is
    imported yet."""
    pytest.importorskip("umap")
    import subprocess
    import sys

    script = tmp_path / "check.py"
    script.write_text(
        "import os, sys, threading\n"
        "os.environ.pop('IBIF_NAPARI_NO_WARMUP', None)\n"
        "seen = {}\n"
        "real = threading.Thread.start\n"
        "def start(self):\n"
        "    if self.name == 'ibif-umap-warm-up':\n"
        "        seen['modules'] = set(sys.modules)\n"
        "    return real(self)\n"
        "threading.Thread.start = start\n"
        "from ibif_napari import _cytometry as cy\n"
        "assert 'sklearn' not in sys.modules, 'imported too early to test'\n"
        "thread = cy.warm_up()\n"
        "need = {'sklearn', 'sklearn.base', 'umap', 'vtea_core.clustering', 'vtea_core.reduction.reduction'}\n"
        "print(sorted(need - seen['modules']))\n"
        "thread.join(120)\n")
    env = {k: v for k, v in __import__("os").environ.items() if k != "IBIF_NAPARI_NO_WARMUP"}
    out = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, env=env, timeout=300)
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().splitlines()[-1] == "[]"


def test_the_vtea_core_pin_is_one_and_an_old_install_is_reported(qtbot, monkeypatch):
    """pip keeps an installed vtea-core with the same version number when only the
    pinned commit changes, so the panel says when the installed one is too old."""
    import re
    from pathlib import Path

    from ibif_napari._cytometry_widget import CytometryWidget

    root = Path(__file__).resolve().parents[1]
    for path in ("pyproject.toml", ".github/workflows/ci.yml", ".github/workflows/vv.yml"):
        pins = re.findall(r"vtea-core\[[^\]]*\] @ git\+[^\s\"]+", (root / path).read_text())
        assert pins and set(pins) == {cy.VTEA_CORE}, path
    assert cy.vtea_core_problem() is None
    from vtea_core import classes

    monkeypatch.delattr(classes, "column_references")
    problem = cy.vtea_core_problem()
    assert "vtea_core.classes.column_references" in problem and cy.VTEA_CORE in problem
    assert "--force-reinstall --no-deps" in problem
    v, _ = scene()
    w = CytometryWidget(v)
    qtbot.addWidget(w)
    assert not w.vtea_warning.isHidden() and "older than" in w.vtea_warning.text()


def test_painted_image_gate_rings_its_objects_and_is_saved(qtbot, tmp_path):
    from ibif_napari._cyto_workbench import IMAGE_GATE_LAYER, Workbench

    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    bench = Workbench(v)
    qtbot.addWidget(bench)
    bench.x_col.setCurrentText("CD3 mean")
    bench.y_col.setCurrentText("CD8 mean")
    layer = bench.start_image_gate()
    # A labels layer over the objects' Y X grid, ready to paint with the segmentation brush.
    assert layer.name == IMAGE_GATE_LAYER and layer.data.shape == (60, 80) and layer.mode == "paint"
    assert tuple(layer.scale) == (0.5, 0.5) and bench.image_btn.text() == "Make image gate"
    bench.image_gate_clicked()  # nothing painted yet: no gate, and it says so
    assert not s.image_gates and "Paint over" in bench.status.text()
    layer.data[:, :40] = 1  # the left two columns of cells
    assert bench.make_image_gate("left") == "left"
    bench._end_image_gate()
    assert IMAGE_GATE_LAYER not in v.layers and bench.image_btn.text() == "Gate on image…"
    assert s.mask("image", "left").sum() == 12
    # Its objects are ringed on the plot in the gate's colour; Colour… changes it.
    color = s.image_gates["left"]["color"]

    def rings():
        return [c for c in bench.plot.axes.collections if len(c.get_facecolors()) == 0]

    assert len(rings()) == 1 and len(rings()[0].get_offsets()) == 12
    assert tuple(rings()[0].get_edgecolors()[0][:3]) == pytest.approx(
        tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)))
    bench.select_gates([("image", "left")])
    assert bench.recolor_gate("#ff00ff") == "#ff00ff" and s.image_gates["left"]["color"] == "#ff00ff"
    assert tuple(rings()[0].get_edgecolors()[0][:3]) == pytest.approx((1, 0, 1))
    assert tuple(v.layers["cells selection"].colormap.map(1)[:3]) == pytest.approx((1, 0, 1))
    bench.plot.set_point_style(outline_image_gates=False)
    assert not rings() and bench.state.view["ring_image_gates"] is False
    bench.plot.set_point_style(outline_image_gates=True)
    # Saved with the workspace (the painted mask, compressed) and restored.
    folder, notes = _workspace.save_workspace(v, tmp_path / "w")
    assert notes == []
    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    restored = cy.state_of(w)
    assert restored.mask("image", "left").sum() == 12
    np.testing.assert_array_equal(restored.image_gates["left"]["mask"], s.image_gates["left"]["mask"])
    # Deleting the paint layer cancels.
    layer = bench.start_image_gate()
    v.layers.remove(layer)
    assert bench._image_layer is None and bench.image_btn.text() == "Gate on image…"


def test_painted_gate_in_3d():
    # A Z Y X mask counts objects by their Z too; without Z centroids every plane counts.
    mask = np.zeros((4, 10, 10), bool)
    mask[1, 2:5, 2:5] = True
    gate = cy._painted_gate(mask, (2.0, 0.5, 0.5), (0.0, 0.0, 0.0))
    assert gate["mask"].shape == (1, 3, 3) and gate["translate"] == [2.0, 1.0, 1.0]
    zs, ys, xs = np.array([2.0, 4.0, 2.0]), np.array([1.5, 1.5, 4.0]), np.array([1.5, 1.5, 1.5])
    assert cy._in_painted(gate, zs, ys, xs).tolist() == [True, False, False]
    assert cy._in_painted(gate, None, ys, xs).tolist() == [True, True, False]
    assert cy._image_gate_from_json(cy._image_gate_to_json(gate))["mask"].tolist() == gate["mask"].tolist()
    assert cy._painted_gate(np.zeros((5, 5)), None, None) is None


def test_selecting_phenotypes_shows_them_on_the_image(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    s.add_image_gate("left", [[[0, 0], [0, 20], [30, 20], [30, 0]]])  # 12 cells
    top = s.add_gate("top", "CD3 mean", "CD8 mean", [[0, 100], [1e4, 100], [1e4, 1e4], [0, 1e4]])  # 4
    s.add_phenotype("Left", "`image gate: left`", color="#00ff00")
    s.add_phenotype("Top", "`gate: top`", color="#ff0000")  # exclusive: Left took them all
    s.add_phenotype("Bright", "`gate: top`", parent="Left", color="#0000ff")
    bench = Workbench(v)
    qtbot.addWidget(bench)
    # One phenotype: its objects in its colour, as clicking a gate does.
    bench.select_phenotypes(["Left"])
    layer = v.layers["cells selection"]
    assert set(np.unique(layer.data)) == {0, 1} and (layer.data > 0).sum() == (v.layers["cells"].data[
        :, :, :40] > 0).sum()
    assert tuple(layer.colormap.map(1)[:3]) == pytest.approx((0, 1, 0))
    assert "Left: 12 objects" in bench.status.text()
    # Several: each in its colour, overlaps (Bright is within Left) in white.
    bench.select_phenotypes(["Left", "Bright"])
    assert bench.selected_phenotypes() == ["Left", "Bright"]
    assert not bench.pheno_edit_btn.isEnabled() and bench.pheno_delete_btn.isEnabled()
    assert set(np.unique(layer.data)) == {0, 1, 3}
    assert "2 phenotypes shown" in bench.status.text() and "4 in more than one" in bench.status.text()
    # Choosing a gate clears the phenotype choice (the image shows one list at a time), and back.
    bench.select_gates([("gate", top.id)])
    assert bench.selected_phenotypes() == []
    bench.select_phenotypes(["Bright"])
    assert bench.selected_gates() == [] and bench.pheno_edit_btn.isEnabled()
    # Refreshing the lists (any change) keeps the selection.
    s.changed()
    assert bench.selected_phenotypes() == ["Bright"]
    bench.select_phenotypes(["Left", "Top"])
    bench.delete_phenotype()
    assert [p.name for p in s.tree] == []


def test_point_size_custom_and_by_a_measurement(qtbot):
    from ibif_napari._cyto_plot import CUSTOM_SIZE, SIZE_BY, PointSizeDialog, sizes_from
    from ibif_napari._cyto_workbench import Workbench

    v, table = scene()
    table = table.rename(columns={"CD3 mean": "cells: Some_Very_Long_Acquisition_File_Name_2026 CD3 mean"})
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    bench = Workbench(v)
    qtbot.addWidget(bench)
    plot = bench.plot
    # A size in pt (diameter): stored as matplotlib's area.
    dialog = PointSizeDialog(plot, by_measurement=False)
    qtbot.addWidget(dialog)
    dialog.diameter.setValue(4.0)
    dialog.apply()
    assert plot.point_size == 16.0 and plot.size_by is None
    assert plot._actions[("size", CUSTOM_SIZE)].isChecked()
    assert plot.axes.collections[0].get_sizes()[0] == 16.0
    # By a measurement: sizes follow its values, from the smallest to the largest diameter.
    dialog = PointSizeDialog(plot, by_measurement=True)
    qtbot.addWidget(dialog)
    long = "cells: Some_Very_Long_Acquisition_File_Name_2026 CD3 mean"
    dialog.feature.setCurrentIndex(dialog.feature.findData(long))
    assert dialog.feature.currentText() != long and dialog.feature.toolTip() == long  # short label
    dialog.smallest.setValue(2.0)
    dialog.largest.setValue(10.0)
    dialog.show()
    assert dialog.width() < 520  # long names do not widen it
    dialog.apply()
    assert plot.size_by == long and plot._actions[("size", SIZE_BY)].isChecked()
    sizes = plot.axes.collections[0].get_sizes()
    assert sizes.min() == pytest.approx(4.0) and sizes.max() == pytest.approx(100.0)
    assert "size: " in plot.axes.get_title()
    assert bench.state.view["size_by"] == long and bench.state.view["size_range"] == [2.0, 10.0]
    # Replotting keeps it; a preset size goes back to one size for every point.
    bench.x_col.setCurrentText("CD8 mean")
    assert len(set(plot.axes.collections[0].get_sizes())) == 2  # CD3 is bright or dim
    plot._actions[("point_size", "Large")].trigger()
    assert plot.size_by is None and plot.axes.collections[0].get_sizes()[0] == 14.0
    assert sizes_from([np.nan, 1.0, 1.0], 2, 4).tolist() == [4.0, 9.0, 9.0]


def test_gates_are_drawn_with_blitting(qtbot):
    """The lasso redraws only its outline as the mouse moves, not every point."""
    from matplotlib.backend_bases import MouseButton, MouseEvent
    from qtpy.QtWidgets import QWidget

    from ibif_napari._cyto_plot import ScatterPlot

    plot = ScatterPlot()
    qtbot.addWidget(plot)
    plot.resize(640, 520)
    QWidget.show(plot)  # ScatterPlot.show draws points
    qtbot.waitExposed(plot)
    rng = np.random.default_rng(0)
    n = 50_000
    plot.show(rng.normal(size=n), rng.normal(size=n), np.arange(n))
    drawn = []
    plot.draw_gate("lasso", drawn.append)
    assert plot._selector.useblit
    full = []
    plot.canvas.mpl_connect("draw_event", lambda e: full.append(e))
    ax = plot.axes

    def event(name, x, y):
        px, py = ax.transData.transform((x, y))
        e = MouseEvent(name, plot.canvas, px, py, button=MouseButton.LEFT)
        plot.canvas.callbacks.process(name, e)

    event("button_press_event", -1, -1)
    for t in np.linspace(0, 2 * np.pi, 40):
        event("motion_notify_event", np.cos(t), np.sin(t))
    event("button_release_event", 1, 0)
    plot.canvas.flush_events()
    assert len(drawn) == 1 and len(drawn[0]) > 20
    assert len(full) <= 2  # not one full redraw per mouse move


def _typed_scene():
    """scene() with two phenotypes: T (the left two columns of cells) and Other."""
    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    s.add_phenotype("T", "`CD3 mean` > 100", color="#ff0000")
    s.add_phenotype("Other", "", color="#0000ff")
    return v, s


def test_spatial_tab_measures_distances_neighbours_and_contacts(qtbot, tmp_path):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _typed_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    tab = bench.spatial
    assert tab.classes.currentData() == "phenotype" and tab.checked_targets() == ["Other", "T"]
    for box in tab.measures.values():
        box.setChecked(True)
    tab.radius.setValue(15.0)
    info = tab.run(wait=True)
    assert info is not None and info["kind"] == "spatial" and s.analyses_of("spatial") == ["Spatial 1"]
    cols = info["columns"]["features"]
    assert "Spatial 1 distance to T (µm)" in cols and "Spatial 1 surface distance to Other (µm)" in cols
    t = s.mask("phenotype", "T")
    col = s.table["cells: centroid X (µm)"] if "cells: centroid X (µm)" in s.table else \
        s.table[s.centroid_columns()["X"]]
    x = col.to_numpy()
    d_t = s.column("Spatial 1 distance to T (µm)")
    # Cells are 20 px (10 µm) apart: every T cell has another T cell 10 µm away; the
    # Other column next to the T cells is 10 µm from one, the far column 20 µm.
    assert (d_t[t] == 10).all() and set(d_t[~t]) == {10.0, 20.0}
    assert d_t[x == x.max()].min() == 20
    # Masks 12 px wide with 8 px gaps: 9 voxel steps of 0.5 µm between nearest voxels.
    assert (s.column("Spatial 1 surface distance to T (µm)")[t] == 4.5).all()
    assert (s.column("Spatial 1 contacts") == 0).all()
    # Within 15 µm: side and diagonal neighbours, never across the two time points.
    n_all = s.column("Spatial 1 n T within 15") + s.column("Spatial 1 n Other within 15")
    assert n_all.max() == 8 and n_all.min() == 3
    # The features are like any other: listed for clustering, in rules, colour the plot.
    assert "Spatial 1 distance to T (µm)" in s.measured_features()
    assert s.rule_count("`Spatial 1 distance to T (µm)` > 15")[0] == 6  # the far column, 2 time points
    tab.columns.setCurrentRow(0)
    tab.colour_plot()
    assert bench.color_by.currentText() == tab.columns.item(0).data(Qt.UserRole)
    assert bench.tabs.currentIndex() == 0
    assert bench.clustering.analysis.count() == 0  # the Clustering tab lists only its own results
    folder, notes = _workspace.save_workspace(v, tmp_path / "w")
    assert notes == []
    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    restored = cy.state_of(w)
    np.testing.assert_allclose(restored.column("Spatial 1 distance to T (µm)"), d_t)
    tab.delete()
    assert not s.analyses and s.derived is None


def test_neighbourhood_tab_types_and_enrichment(qtbot):
    from ibif_napari._cyto_spatial import VIEWS
    from ibif_napari._cyto_workbench import Workbench

    v, s = _typed_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    tab = bench.neighbourhoods
    tab.radius.setValue(15.0)
    tab.n_types.setValue(2)
    tab.permutations.setValue(100)
    info = tab.run(wait=True)
    assert info is not None and info["kind"] == "neighbourhoods"
    types = s.column("Neighbourhoods 1 neighbourhood")
    assert set(types) == {1, 2}
    # The T side and the Other side of the image are the two types.
    t = s.mask("phenotype", "T")
    left_type = np.bincount(types[t].astype(int)).argmax()
    assert (types[t] == left_type).mean() > 0.6 and (types[~t] != left_type).mean() > 0.6
    frac = s.column("Neighbourhoods 1 fraction T")
    assert frac[t].mean() > frac[~t].mean()
    # Neighbourhoods overlap: one is drawn around every cell, so each cell is a member
    # of its own and of every neighbour's within the radius (3 to 8 neighbours here),
    # and the most common type among them is kept as well as its own.
    memberships = s.column("Neighbourhoods 1 memberships")
    assert memberships.min() == 4 and memberships.max() == 9
    assert info["columns"]["member_type"] == "Neighbourhoods 1 member neighbourhood"
    member = s.column("Neighbourhoods 1 member neighbourhood")
    assert set(member) <= {1, 2} and (member == types).mean() > 0.6
    assert "Neighbourhoods 1 memberships" in s.measured_features()
    assert "overlap" in tab.info.text()
    profile = s.analysis_table("Neighbourhoods 1", "profile")
    assert set(profile.columns) == {"T", "Other"} and len(profile) == 2
    pairs = s.analysis_table("Neighbourhoods 1", "pairs").set_index(["class", "neighbor"])
    assert pairs.loc[("T", "T"), "z"] > 0 > pairs.loc[("T", "Other"), "z"]
    for k in range(len(VIEWS)):  # every view draws
        tab.view.setCurrentIndex(k)
        assert tab.heatmap.figure.axes
    tab.colour_plot()
    assert bench.color_by.currentText() == "Neighbourhoods 1 neighbourhood"
    assert any(text.get_text().startswith("type ") for text in bench.plot.axes.get_legend().get_texts())
    layer = tab.show_on_image()
    assert layer.name == "cells Neighbourhoods 1" and set(np.unique(layer.data)) == {0, 1, 2}
    assert s.rule_count("`Neighbourhoods 1 neighbourhood` == 1")[0] == int((types == 1).sum())


def test_spatial_tabs_need_classes(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, table = scene()
    cy.state_of(v).set_table("cells", table, "cells")
    bench = Workbench(v)
    qtbot.addWidget(bench)
    assert not bench.spatial.run_btn.isEnabled() and "Make phenotypes" in bench.spatial.status.text()
    assert bench.spatial.run(wait=True) is None


def _truth_state(n=600):
    """cell_types() with its true types as phenotypes (from a 'truth' column)."""
    table, kind = cell_types(n=n)
    table["truth"] = kind
    s = cy.state_of(ViewerModel())
    s.set_table("cells", table, "cells")
    for code, name, color in ((0, "CD8 T", "#d62728"), (1, "CD4 T", "#1f77b4"), (2, "Other", "#7f7f7f")):
        s.add_phenotype(name, f"truth == {code}", color=color)
    return s, np.array(["CD8 T", "CD4 T", "Other"], dtype=object)[kind]


def test_classification_tab_trains_classifies_and_makes_phenotypes(qtbot, tmp_path, monkeypatch):
    from ibif_napari._cyto_workbench import Workbench

    monkeypatch.setenv("IBIF_NAPARI_MODELS", str(tmp_path / "models"))
    s, truth = _truth_state()
    bench = Workbench(s.viewer)
    qtbot.addWidget(bench)
    tab = bench.classification
    assert tab.model.count() == 0 and not tab.run_btn.isEnabled() and "Train a model" in tab.card.text()
    # Train on the phenotypes in the Training workbench, save to the library.
    trainer = tab.train_dialog()
    qtbot.addWidget(trainer)
    trainer.source.setCurrentIndex(trainer.source.findData("classes\tphenotype"))
    assert trainer.add_from_source() == 600 and s.training_counts() == {"CD8 T": 200, "CD4 T": 200, "Other": 200}
    trainer.check_features("names", ["CD3 mean", "CD8 mean", "CD4 mean"])
    result = trainer.train(wait=True)
    assert result["metrics"]["balanced_accuracy"] > 0.95 and result["metrics"]["n"] == 120  # 20% held out
    assert result["model"].card.colors["CD8 T"] == "#d62728"
    trainer.notes.setText("synthetic T cells")
    folder = trainer.save("T cell types")
    assert (tmp_path / "models" / "T cell types" / "card.json").exists() and folder.endswith("T cell types")
    # Apply it from the library.
    tab.refresh()
    assert tab.model.currentText() == "T cell types" and tab.run_btn.isEnabled()
    assert "3 classes" in tab.card.text() and "synthetic T cells" in tab.card.text()
    info = tab.run()
    assert info["kind"] == "classification" and info["classes"] == ["CD4 T", "CD8 T", "Other"]
    predicted = np.array(info["classes"], dtype=object)[s.column("Classes 1 class").astype(int) - 1]
    assert (predicted == truth).mean() > 0.95
    assert np.nanmin(s.column("Classes 1 confidence")) > 0 and "Classes 1 p CD8 T" in s.features()
    counts = dict((c, n) for c, n, _ in s.classification_counts("Classes 1"))
    assert sum(counts.values()) == 600
    # Phenotypes from the classes: rename them; only confident objects.
    assert tab.table.rowCount() == 3 and tab.table.item(0, 3).text() == ""  # "CD4 T" exists already
    for r in range(3):
        tab.table.item(r, 3).setText(f"{tab.table.item(r, 0).text()} (ANN)")
    tab.min_conf.setValue(0.5)
    made = tab.make_phenotypes()
    assert [p.name for p in made] == ["CD4 T (ANN)", "CD8 T (ANN)", "Other (ANN)"]
    assert "Classes 1 confidence` >= 0.5" in s.tree.get("CD8 T (ANN)").rule
    # Colour the plot by class: the classes' names and colours in the legend.
    tab.colour_plot()
    legend = [t.get_text() for t in bench.plot.axes.get_legend().get_texts()]
    assert legend == ["CD4 T", "CD8 T", "Other"]
    # The classes can be what spatial analysis classes objects by.
    assert ("Classes 1 classes", "Classes 1 class") in s.class_choices()
    assert set(s.classes("Classes 1 class")) == {"CD4 T", "CD8 T", "Other"}
    # A model whose features the table lacks says so and cannot run.
    s2 = cy.state_of(ViewerModel())
    table, _ = cell_types(n=30)
    s2.set_table("other", table.drop(columns=["CD8 mean"]), "cells")
    bench2 = Workbench(s2.viewer)
    qtbot.addWidget(bench2)
    bench2.classification.refresh()
    assert "no column for: CD8 mean" in bench2.classification.card.text()
    assert not bench2.classification.run_btn.isEnabled()
    tab.delete()
    assert not s.analyses_of("classification")


def test_classification_is_saved_and_shown_on_the_image(qtbot, tmp_path, monkeypatch):
    from ibif_napari._cyto_workbench import Workbench

    monkeypatch.setenv("IBIF_NAPARI_MODELS", str(tmp_path / "models"))
    v, s = _typed_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    trainer = bench.classification.train_dialog()
    qtbot.addWidget(trainer)
    s.label_from_classes("phenotype")
    trainer.check_features("names", ["CD3 mean", "CD8 mean"])
    trainer.split_by.setCurrentIndex(1)  # by time point: one frame trains, the other checks
    trainer.fraction.setValue(0.5)
    metrics = trainer.train(wait=True)["metrics"]
    assert metrics["n"] == 12 and metrics["balanced_accuracy"] == 1.0
    trainer.save()
    info = bench.classification.run()
    assert info["n_objects"] == 24
    layer = bench.classification.show_on_image()
    assert layer.name == "cells Classes 1" and set(np.unique(layer.data)) == {0, 1, 2}
    folder, notes = _workspace.save_workspace(v, tmp_path / "w")
    assert notes == []
    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    restored = cy.state_of(w)
    assert restored.analysis_kind("Classes 1") == "classification"
    np.testing.assert_array_equal(restored.column("Classes 1 class"), s.column("Classes 1 class"))
    assert restored.class_names("Classes 1 class") == {1: "Other", 2: "T"}


# ---- levels of the spatial hierarchy -------------------------------------------------------


def _level_table(s, rng_seed=0):
    """A stand-in for a level built on the cells: one object per cell (a window around
    it), at the cell's centre, with a composition-like feature."""
    rng = np.random.default_rng(rng_seed)
    cols = s.root.centroid_columns()
    t = s.root.mask("phenotype", "T").astype(float)
    return pd.DataFrame({"object_id": np.arange(1, len(t) + 1),
                         "centroid Y (µm)": s.root.table[cols["Y"]].to_numpy(),
                         "centroid X (µm)": s.root.table[cols["X"]].to_numpy(),
                         "fraction T": np.clip(t + rng.normal(0, 0.1, len(t)), 0, 1),
                         "n objects": rng.integers(3, 9, len(t)).astype(float)})


def test_levels_each_have_their_own_analysis():
    v, s = _typed_scene()
    assert isinstance(s, cy.Levels) and s.current_level == cy.ROOT_LEVEL and s.root is s.current
    assert s.level_tree() == [(0, "Objects")] and s.level_label("Objects") == "Objects (cells): segmentation"
    w = s.add_level("Neighbourhoods 1", _level_table(s), provenance={"radius": 15})
    assert s.current_level == "Objects" and len(s.tree) == 2  # still the cells' phenotypes
    s.set_level("Neighbourhoods 1")
    assert s.current is w and len(s.table) == 24 and "fraction T" in s.features()
    assert len(s.tree) == 0 and s.layer_name is None
    s.add_phenotype("T-rich", "`fraction T` > 0.5")
    g = s.add_gate("crowded", "n objects", "fraction T", [[5, -1], [10, -1], [10, 2], [5, 2]])
    assert s.mask("phenotype", "T-rich").sum() >= 1 and s.mask("gate", g.id).any()
    s.set_level("Objects")
    assert len(s.tree) == 2 and "T-rich" not in [n.name for n in s.tree]
    with pytest.raises(KeyError):
        s.gates.get(g.id)  # the level's gate is not the cells'
    s.add_level("Windows 2", _level_table(s, 1), parent="Neighbourhoods 1")
    assert s.level_tree() == [(0, "Objects"), (1, "Neighbourhoods 1"), (2, "Windows 2")]
    assert s.level_info("Windows 2")["depth"] == 2 and s.level_info("Windows 2")["objects"] == 24
    assert s.level_label("Windows 2") == "Spatial level 2: Windows 2 (windows on Neighbourhoods 1)"
    with pytest.raises(ValueError, match="already"):
        s.add_level("Windows 2", _level_table(s))
    with pytest.raises(ValueError, match="root"):
        s.remove_level("Objects")
    s.set_level("Windows 2")
    assert s.remove_level("Neighbourhoods 1") == ["Neighbourhoods 1", "Windows 2"]
    assert s.current_level == "Objects" and s.level_names() == ["Objects"]


def test_one_list_of_listeners_hears_every_level():
    v, s = _typed_scene()
    heard = []
    cy.root_of(v).listen(lambda: heard.append(s.current_level))
    s.add_level("NH", _level_table(s), select=True)
    s.current.changed()
    assert heard[-2:] == ["NH", "NH"]


def test_workbench_level_selector(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _typed_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    assert bench.level_box.count() == 1 and not bench.level_box.isEnabled()
    bench.x_col.setCurrentText("CD3 mean")
    s.add_level("NH 1", _level_table(s))
    assert bench.level_box.count() == 2 and bench.level_box.isEnabled()
    bench.level_box.setCurrentIndex(1)  # the user picks the level
    assert s.current_level == "NH 1" and bench.level_path.text() == "Objects › NH 1"
    assert "spatial level 1" in bench.windowTitle()
    assert bench.x_col.findText("fraction T") >= 0 and bench.x_col.findText("CD3 mean") < 0
    # every tab works on this level's objects
    bench.x_col.setCurrentText("n objects")
    bench.y_col.setCurrentText("fraction T")
    gate = bench.add_gate("rich", [[0, 0.5], [20, 0.5], [20, 2], [0, 2]])
    assert gate is not None and s.mask("gate", gate.id).sum() >= 1
    bench.clustering.refresh()
    assert bench.clustering.features.count() >= 2
    bench.set_level("Objects")
    assert bench.x_col.currentText() == "CD3 mean"  # the cells' plot set-up comes back
    assert bench.level_box.currentData() == "Objects" and bench.windowTitle() == "Cytometry workbench"


def test_levels_are_saved_with_the_workspace(tmp_path):
    v, s = _typed_scene()
    s.add_level("NH 1", _level_table(s), provenance={"radius": 15.0})
    s.add_level("NH 2", _level_table(s, 1), parent="NH 1", kind="regions")
    s.set_level("NH 1")
    s.add_phenotype("T-rich", "`fraction T` > 0.5")
    rows = np.arange(len(s.table))
    s.add_analysis("Clusters", {"features": ["fraction T"]}, rows, labels=(s.column("fraction T") > 0.5) + 1)
    expected = s.typed_table()
    folder, notes = _workspace.save_workspace(v, tmp_path / "w")
    assert notes == []
    data = json.loads((folder / "workspace.json").read_text())["extensions"]["cytometry"]
    assert data["version"] == 2 and [e["name"] for e in data["levels"]] == ["NH 1", "NH 2"]
    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    r = cy.state_of(w)
    assert r.level_tree() == [(0, "Objects"), (1, "NH 1"), (2, "NH 2")] and r.current_level == "NH 1"
    assert r.level_info("NH 2")["kind"] == "regions" and r.level_info("NH 1")["provenance"] == {"radius": 15.0}
    pd.testing.assert_frame_equal(r.typed_table(), expected, check_dtype=False)
    assert "Clusters" in r.analyses and (r.clusters("Clusters") > 0).all()
    assert len(r.root.tree) == 2  # the cells' own analysis is back too


def _tiled_scene():
    """One frame of 2 x 8 touching square cells (10 x 10 px, 0.5 µm): the left half
    bright in CD3 (T), the right half not (Other)."""
    v = ViewerModel()
    labels = np.zeros((20, 80), np.uint16)
    cd3 = np.full(labels.shape, 10.0)
    k = 0
    for r in range(2):
        for c in range(8):
            k += 1
            labels[r * 10:(r + 1) * 10, c * 10:(c + 1) * 10] = k
            if c < 4:
                cd3[r * 10:(r + 1) * 10, c * 10:(c + 1) * 10] = 500
    meta = {"ibif_axes": "YX", "ibif_voxel_um": {"Y": 0.5, "X": 0.5}}
    common = dict(scale=(0.5, 0.5), metadata=meta)
    image = v.add_image(cd3, name="CD3", **common)
    layer = v.add_labels(labels, name="cells", **common)
    table = _measure.measure_layer(v, layer, [image])
    _measure.remember(v, {"cells": table})
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    s.add_phenotype("T", "`CD3 mean` > 100")
    s.add_phenotype("Other", "")
    return v, s


def test_make_levels_build_on_them_and_show_them(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _typed_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    tab = bench.neighbourhoods
    tab.radius.setValue(15.0)
    tab.n_types.setValue(2)
    tab.permutations.setValue(0)
    assert tab.run(wait=True) is not None
    cell_types = s.column("Neighbourhoods 1 neighbourhood")
    level = tab.make_level("NH 1")
    assert level is not None and s.current_level == "NH 1" and len(s.table) == 24
    assert {"fraction T", "fraction Other", "n objects", "neighbourhood type", "T"} <= set(s.table.columns)
    np.testing.assert_array_equal(s.table["neighbourhood type"], cell_types)  # each window, its seed's type
    assert bench.level_box.count() == 2 and "NH 1" in bench.level_path.text()
    # The level is analysed like the cells: its windows' types are classes to build on.
    assert ("Neighbourhoods types (from the level below)" not in dict(s.class_choices())
            and cy.LEVEL_TYPE in dict(s.class_choices()).values())
    tab.refresh()
    tab.classes.setCurrentIndex(tab.classes.findData(cy.LEVEL_TYPE))
    tab.radius.setValue(25.0)
    assert tab.run(wait=True) is not None
    on_image = tab.show_on_image()  # a level is painted through the cells
    assert on_image.data.shape == v.layers["cells"].data.shape
    assert set(np.unique(on_image.data)) <= {0, 1, 2} and on_image.data.any()
    level2 = tab.make_level("NH 2")
    assert s.level_tree() == [(0, "Objects"), (1, "NH 1"), (2, "NH 2")] and len(level2.table) == 24
    # The cells' surface measures need masks: a level says so.
    bench.spatial.refresh()
    with pytest.raises(ValueError, match="masks"):
        s.label_blocks()


def test_neighbourhoods_around_a_population(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _typed_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    tab = bench.neighbourhoods
    tab.radius.setValue(15.0)
    tab.n_types.setValue(2)
    tab.permutations.setValue(0)
    tab.seed_pop.setCurrentIndex(tab.seed_pop.findData("phenotype\tT"))
    info = tab.run(wait=True)
    assert info["seeds"] == ["phenotype", "T"]
    t = s.mask("phenotype", "T")
    types = s.column("Neighbourhoods 1 neighbourhood")
    assert (types[t] > 0).all() and (types[~t] == 0).all()
    assert (s.column("Neighbourhoods 1 memberships")[~t] >= 0).all()
    level = tab.make_level("Around T")
    assert len(level.table) == int(t.sum())


def test_regions_follow_touching_masks_and_selections_follow_levels(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _tiled_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    tab = bench.neighbourhoods
    tab.radius.setValue(6.0)  # a cell and its side neighbours (5 µm apart)
    tab.n_types.setValue(2)
    tab.permutations.setValue(0)
    tab.run(wait=True)
    regions = tab.make_regions("Units", values=None, touching=True, d=0.0, tau=0.5, min_size=1)
    assert regions is not None and s.current_level == "Units"
    sizes = sorted(s.table["n objects"])
    assert sum(sizes) == 16 and len(sizes) >= 2  # the T half and the Other half, at least
    # Regions are painted through their cells, each cell its region's value.
    s.add_phenotype("big", "`n objects` >= 4")
    layer = s.show_population("phenotype", "big")
    assert layer.data.shape == v.layers["cells"].data.shape and layer.data.any()
    # Follow the big regions down to their cells: a phenotype there.
    bench.set_population("phenotype", "big")
    name = bench.follow_selection("Objects")
    assert s.current_level == "Objects" and name.startswith("members of big (Units)")
    big_cells = s.mask("selection", name).sum()
    assert bench.current_population() == ("selection", name)
    s.phenotype_from("selection", name, "in big regions")  # a selection can become a phenotype
    assert big_cells == int(s.level("Units").table.loc[s.level("Units").table["n objects"] >= 4,
                                                        "n objects"].sum())
    # and back up: the regions containing the T cells
    bench.set_population("phenotype", "T")
    up = bench.follow_selection("Units")
    assert s.current_level == "Units" and s.mask("selection", up).sum() >= 1
    # Columns from another level: the size of each cell's region, onto the cells.
    bench.set_level("Objects")
    new = bench.bring_columns("Units", ["n objects"], "mean")
    assert new == ["n objects (Units)"] and np.nansum(s.column(new[0])) > 0
    up_cols = bench.state.bring_columns("Objects", ["CD3 mean"], "Units")
    assert up_cols == ["CD3 mean (Objects)"] and len(s.level("Units").column(up_cols[0])) == len(sizes)


def test_built_levels_are_saved_with_their_members(tmp_path):
    from ibif_napari import _cyto_levels

    v, s = _typed_scene()
    s.neighbourhood_job  # noqa: B018 - the job is run below through the state
    settings = {"classes": "phenotype", "class_label": "Phenotypes", "method": "radius", "radius": 15.0,
                "k": 10, "n_types": 2, "permutations": 0, "seed": 0}
    s.add_neighbourhoods("NB", settings, s.neighbourhood_job(settings)())
    s.make_windows_level("NB", "W")
    folder, notes = _workspace.save_workspace(v, tmp_path / "w")
    assert notes == []
    w = ViewerModel()
    _workspace.load_workspace(w, folder)
    r = cy.state_of(w)
    assert r.level_names() == ["Objects", "W"]
    geometry = r._info["W"]["geometry"]
    assert geometry is not None and geometry["membership"].shape == (24, 24)
    assert (_cyto_levels.own_rows(r, "W") == np.arange(24)).all()


def test_a_selection_from_another_level_is_a_population(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _tiled_scene()
    settings = {"classes": "phenotype", "class_label": "P", "method": "radius", "radius": 6.0, "k": 10,
                "n_types": 2, "permutations": 0, "seed": 0}
    s.add_neighbourhoods("NB", settings, s.neighbourhood_job(settings)())
    s.make_regions_level("NB", "Units")
    bench = Workbench(v)
    qtbot.addWidget(bench)
    name = bench.follow_selection("Objects")  # all the regions, down to their cells
    assert ("selection", name, f"Selection: {name}") in s.populations()
    g = bench.add_gate("bright", [[0, 0], [1e5, 0], [1e5, 1e5], [0, 1e5]])  # a gate within it
    assert s.scope[("gate", g.id)] == ("selection", name)
    s.set_gate_color("selection", name, "#123456")
    assert s.population_color("selection", name) == "#123456"
    with pytest.raises(ValueError, match="keeps"):
        s.rename_gate("selection", name, "x")
    bench._fill_gates()
    assert any(bench.gate_tree.topLevelItem(i).data(0, Qt.UserRole) == ("selection", name)
               for i in range(bench.gate_tree.topLevelItemCount()))
    s.remove_gate("selection", name)
    assert name not in s.selections() and ("gate", g.id) not in s.order


def repair_scene(n=150, seed=0):
    """A proximal tubule repair continuum (as in Asghari et al. 2025): healthy PT
    (LRP2 high) -> transitional (THY1+PROM1+) -> either THY1+ or PROM1+, as phenotypes
    from a 'truth' column. Laid out in space along the same Y: healthy on the left, the
    THY1 branch top right, the PROM1 branch bottom right, with immune cells (CD45)
    mixed into the PROM1 branch. The transitional cells straddle the branch point (as a
    clustering puts one cluster there), so the cluster tree can find it."""
    rng = np.random.default_rng(seed)
    parts = []
    # (name, code, progress from, to, branch: 0 trunk, 1 THY1, 2 PROM1)
    for name, code, a, b, br in (("Healthy PT", 0, 0.0, 0.3, 0), ("Transitional", 1, 0.3, 0.62, 0),
                                 ("THY1+ PT", 2, 0.62, 0.85, 1), ("PROM1+ PT", 3, 0.62, 0.85, 2)):
        p = rng.uniform(a, b, n)
        # Past the branch point (0.55) the transitional cells are on either branch.
        br = np.where(p > 0.55, rng.integers(1, 3, n), 0) if code == 1 else np.full(n, br)
        late = np.clip((p - 0.55) / 0.3, 0, 1)
        trunk = np.clip(p / 0.55, 0, 1)
        thy1 = 50 + 300 * trunk * (br != 2) + 500 * late * (br == 1) - 150 * late * (br == 2)
        prom1 = 50 + 300 * trunk * (br != 1) + 500 * late * (br == 2) - 150 * late * (br == 1)
        x = 200 * p + rng.normal(0, 3, n)
        y = np.where(br == 1, 100 * late, np.where(br == 2, -100 * late, 0)) + rng.normal(0, 3, n)
        parts.append(pd.DataFrame({
            "LRP2 mean": (800 - 700 * p) * rng.lognormal(0, 0.1, n),
            "THY1 mean": np.maximum(thy1, 10) * rng.lognormal(0, 0.1, n),
            "PROM1 mean": np.maximum(prom1, 10) * rng.lognormal(0, 0.1, n),
            "centroid Y (µm)": y, "centroid X (µm)": x, "truth": code, "progress": p, "branch": br}))
    immune = pd.DataFrame({"LRP2 mean": rng.lognormal(3, 0.1, n // 2), "THY1 mean": rng.lognormal(3, 0.1, n // 2),
                           "PROM1 mean": rng.lognormal(3, 0.1, n // 2),
                           "centroid Y (µm)": -100 * rng.uniform(0.5, 1, n // 2),
                           "centroid X (µm)": 200 * rng.uniform(0.75, 1, n // 2), "truth": 4,
                           "progress": np.nan, "branch": -1})
    table = pd.concat(parts + [immune], ignore_index=True)
    table.insert(0, "label", np.arange(1, len(table) + 1))
    s = cy.state_of(ViewerModel())
    s.set_table("cells", table, "cells")
    s.add_phenotype("PT", "truth < 4", color="#2ca02c")
    for code, name, color in ((0, "Healthy PT", "#1f77b4"), (1, "Transitional", "#ff7f0e"),
                              (2, "THY1+ PT", "#d62728"), (3, "PROM1+ PT", "#9467bd")):
        s.add_phenotype(name, f"truth == {code}", parent="PT", color=color)
    s.add_phenotype("Immune", "truth == 4", color="#7f7f7f")
    return s, table


def test_trajectory_tab_finds_the_repair_lineages(qtbot, tmp_path):
    from ibif_napari._cyto_trajectory import VIEWS
    from ibif_napari._cyto_workbench import Workbench

    s, table = repair_scene()
    bench = Workbench(s.viewer)
    qtbot.addWidget(bench)
    tab = bench.trajectory
    assert bench.tabs.tabText(bench.tabs.count() - 1) == "Trajectory"
    assert tab.checked_features() == ["LRP2 mean", "THY1 mean", "PROM1 mean"]
    tab.population.setCurrentIndex(tab.population.findData("phenotype\tPT"))
    assert tab.population.currentText() == "Phenotype: PT"
    tab.classes.setCurrentIndex(tab.classes.findData("phenotype"))
    starts = [tab.start.itemText(i) for i in range(tab.start.count())]
    assert "Healthy PT" in starts and "Immune" not in starts  # only the classes of the objects chosen
    tab.start.setCurrentText("Healthy PT")
    info = tab.run(wait=True)
    assert info is not None and info["kind"] == "trajectory", tab.status.text()
    st = info["structure"]
    assert sorted(map(tuple, st["lineages"])) == [("Healthy PT", "Transitional", "PROM1+ PT"),
                                                ("Healthy PT", "Transitional", "THY1+ PT")]
    assert info["n_objects"] == 4 * 150  # PT only: the immune cells are not in it
    t = s.column("Trajectory 1 pseudotime")
    assert np.isnan(t[table["truth"] == 4]).all()
    pt = table["truth"].to_numpy() < 4
    rho = pd.Series(t[pt]).corr(table["progress"][pt], method="spearman")
    assert rho > 0.9
    # Each branch's cells are on their own lineage.
    lineage = s.column("Trajectory 1 lineage").astype(int)
    thy1_lin = 1 + next(j for j, lin in enumerate(st["lineages"]) if lin[-1] == "THY1+ PT")
    assert (lineage[table["truth"] == 2] == thy1_lin).mean() > 0.95
    assert (lineage[table["truth"] == 3] != thy1_lin).mean() > 0.95
    # THY1 rises along the THY1 lineage, PROM1 along the other.
    # tradeSeq: THY1 and PROM1 change along their lineages and end differently.
    tests = s.analysis_table("Trajectory 1", "tests").set_index(["test", "scope", "feature"])
    thy1_name = f"t{thy1_lin}"
    prom1_name = "t2" if thy1_name == "t1" else "t1"
    assert set(tests.index.get_level_values("test")) == {"association", "start_vs_end", "diff_end", "pattern"}
    assert tests.loc[("association", thy1_name, "THY1 mean"), "q"] < 1e-6
    assert tests.loc[("association", prom1_name, "PROM1 mean"), "q"] < 1e-6
    for feature in ("THY1 mean", "PROM1 mean"):
        assert tests.loc[("diff_end", "global", feature), "q"] < 1e-6
        assert tests.loc[("pattern", "global", feature), "q"] < 1e-6
    curves = s.analysis_table("Trajectory 1", "smoothers")
    thy1 = curves[(curves["feature"] == "THY1 mean") & (curves["lineage"] == thy1_name)]["fit"].to_numpy()
    assert thy1[-1] > thy1[0] + 1  # the fitted THY1 curve rises along the THY1 lineage (z-scores)
    # The columns are like any other: in rules, colour the plot, the map.
    assert "Trajectory 1 pseudotime" in s.measured_features() and "Trajectory 1 PC 1" in s.features()
    assert s.rule_count("`Trajectory 1 pseudotime` >= 0")[0] == 600
    for k in range(len(VIEWS)):  # every view draws
        tab.view.setCurrentIndex(k)
        widget = tab.stack.currentWidget()
        assert widget.figure.axes, VIEWS[k]
    tab.colour_plot()
    assert bench.color_by.currentText() == "Trajectory 1 pseudotime"
    assert bench.x_col.currentText() == "Trajectory 1 PC 1"
    bench.color_by.setCurrentText("Trajectory 1 lineage")
    assert {text.get_text() for text in bench.plot.axes.get_legend().get_texts()} >= {"t1", "t2"}
    # Saved and restored with the workspace (no image needed for the tables).
    data = json.loads(json.dumps(s.to_dict()))
    assert data["analyses"]["Trajectory 1"]["structure"]["start"] == "Healthy PT"
    s.save_derived(tmp_path)
    s2 = cy.state_of(ViewerModel())
    s2.load_dict(data, {"cells": table}, tmp_path / "cytometry" / "derived.csv")
    np.testing.assert_allclose(s2.column("Trajectory 1 pseudotime"), t)
    assert s2.analysis_table("Trajectory 1", "trends").shape == s.analysis_table("Trajectory 1", "trends").shape
    tab.delete()
    assert not s.analyses_of("trajectory")


def test_trajectory_of_neighbourhood_types_and_following_cell_types(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    s, table = repair_scene()
    bench = Workbench(s.viewer)
    qtbot.addWidget(bench)
    hoods = bench.neighbourhoods
    hoods.radius.setValue(25.0)
    hoods.n_types.setValue(4)
    hoods.permutations.setValue(0)
    assert hoods.run(wait=True) is not None
    tab = bench.trajectory
    key = s.analyses["Neighbourhoods 1"]["columns"]["type"]
    index = tab.classes.findData(key)
    assert index >= 0 and tab.classes.itemText(index) == "Neighbourhoods 1 types"
    assert bench.spatial.classes.findData(key) < 0  # only the Trajectory tab offers them
    tab.classes.setCurrentIndex(index)
    # Classed by neighbourhood types: the composition is ticked, not the intensities.
    assert set(tab.checked_features()) == set(s.analyses["Neighbourhoods 1"]["columns"]["features"])
    # Start from the neighbourhood type of the healthy cells.
    types = s.column(key).astype(int)
    healthy = np.bincount(types[table["truth"] == 0]).argmax()
    tab.start.setCurrentText(f"type {healthy}")
    tab.track.setCurrentIndex(tab.track.findData("phenotype"))
    info = tab.run(wait=True)
    assert info is not None, tab.status.text()
    assert info["structure"]["start"] == f"type {healthy}" and len(info["structure"]["lineages"]) >= 1
    # Healthy PT cells are early in neighbourhood pseudotime, the branches late.
    t = s.column(f"{info['columns']['pseudotime']}")
    truth = table["truth"].to_numpy()
    assert np.nanmedian(t[truth == 0]) < np.nanmedian(t[truth >= 2])
    tracked = s.analysis_table("Trajectory 1", "tracked")
    assert {"Healthy PT", "Immune"} <= set(tracked["class"])
    first = tracked[(tracked["lineage"] == "t1") & (tracked["bin"] == 0)].set_index("class")["fraction"]
    assert first["Healthy PT"] > 0.5
    tab.view.setCurrentIndex(6)
    assert tab.plot.figure.axes
    # Switching back to cell types ticks the intensities again.
    tab.classes.setCurrentIndex(tab.classes.findData("phenotype"))
    assert tab.checked_features() == ["LRP2 mean", "THY1 mean", "PROM1 mean"]


def test_trajectory_needs_a_start_with_objects(qtbot):
    s, table = repair_scene(n=30)
    job_settings = {"population": ["all", None], "classes": "phenotype", "features": ["LRP2 mean"],
                    "start": "Nope"}
    with pytest.raises(ValueError, match="start"):
        s.trajectory_job(job_settings)
    with pytest.raises(ValueError, match="feature"):
        s.trajectory_job({**job_settings, "features": []})


def test_trajectory_pseudotime_on_the_image(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _typed_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    tab = bench.trajectory
    tab.check_features("all")
    tab.start.setCurrentText("T")
    info = tab.run(wait=True)
    assert info is not None, tab.status.text()
    assert info["structure"]["lineages"] == [["T", "Other"]]
    t = s.column("Trajectory 1 pseudotime")
    assert np.nanmedian(t[s.mask("phenotype", "T")]) < np.nanmedian(t[~s.mask("phenotype", "T")])
    layer = tab.show_on_image()
    assert layer is not None and layer.name == "cells Trajectory 1"
    assert set(np.unique(layer.data)) - {0} <= set(range(1, 11)) and layer.data.max() > 1


def test_trajectory_r_engine_is_used_when_chosen(qtbot, monkeypatch):
    """With the R engine, the trajectory and the tradeSeq tests come from
    vtea-core's r_engine (here stand-ins that run the Python ports and note the
    call); without R the choice is greyed out."""
    from vtea_core.trajectory import r_engine, slingshot

    from ibif_napari._cyto_workbench import Workbench

    calls = []

    def fake_slingshot_r(points, labels, start, **kw):
        calls.append("slingshot")
        return slingshot(points, labels, start, ends=kw.get("ends"), distance=kw.get("distance", "slingshot"))

    def fake_tradeseq_r(pt, w, values, **kw):
        calls.append("tradeSeq")
        n = values.shape[1]
        table = pd.DataFrame({"waldStat": np.ones(n), "df": 2, "pvalue": 0.5, "waldStat_1": 1.0, "df_1": 1,
                              "pvalue_1": 0.5}, index=values.columns)
        return {"association": table, "start_vs_end": table}

    monkeypatch.setattr(r_engine, "slingshot_r", fake_slingshot_r)
    monkeypatch.setattr(r_engine, "tradeseq_r", fake_tradeseq_r)
    s, table = repair_scene(n=60)
    bench = Workbench(s.viewer)
    qtbot.addWidget(bench)
    tab = bench.trajectory
    has_r = r_engine.r_available("slingshot", "tradeSeq")
    assert tab.engine.model().item(1).isEnabled() == has_r
    tab.population.setCurrentIndex(tab.population.findData("phenotype\tPT"))
    tab.start.setCurrentText("Healthy PT")
    tab.engine.setCurrentIndex(1)
    info = tab.run(wait=True)
    assert info is not None and info["engine"] == "r", tab.status.text()
    assert calls == ["slingshot", "tradeSeq"]
    tests = s.analysis_table("Trajectory 1", "tests")
    assert set(tests["scope"]) == {"global", "t1"} and "smoothers" not in info["tables"]
    tab.view.setCurrentIndex(2)  # the feature view without GAM curves: the binned means
    assert tab.plot.figure.axes


# ---- classification at each level (H5) -----------------------------------------------------


def _write_card(folder, family, inputs):
    """A model folder with only its card: enough for the library to list it."""
    folder.mkdir(parents=True)
    card = {"family": family, "classes": ["a", "b"], "inputs": inputs}
    (folder / "card.json").write_text(json.dumps(card))
    (folder / "config.json").write_text(json.dumps({"family": family, "classes": ["a", "b"]}))


def test_models_apply_to_the_level_they_were_trained_on(qtbot, tmp_path, monkeypatch):
    from vtea_core.classification import read_card

    from ibif_napari._cyto_classify import fits_level, model_needs
    from ibif_napari._cyto_workbench import Workbench

    monkeypatch.setenv("IBIF_NAPARI_MODELS", str(tmp_path / "models"))
    s, truth = _truth_state()
    bench = Workbench(s.viewer)
    qtbot.addWidget(bench)
    tab = bench.classification
    # A model trained on the cells records that it is for segmented objects.
    trainer = tab.train_dialog()
    qtbot.addWidget(trainer)
    s.label_from_classes("phenotype")
    trainer.check_features("names", ["CD3 mean", "CD8 mean", "CD4 mean"])
    trainer.train(wait=True)
    cells_model = trainer.save("cell types")
    card = read_card(cells_model)
    assert card.inputs["kind"] == "features" and card.training["level"]["kind"] == "objects"
    assert model_needs(card) == {"input_kind": "features", "level_kind": "objects", "depth": 0}
    # NephNet3D reads image crops around masks: segmented objects only.
    _write_card(tmp_path / "models" / "nuclei", "nephnet3d", {"kind": "crops", "channels": ["DAPI"]})
    # A card from before levels were recorded: trained on segmented objects.
    _write_card(tmp_path / "models" / "old", "ann", {"kind": "features", "features": ["CD3 mean"]})
    tab.refresh()
    names = [tab.model.itemText(i) for i in range(tab.model.count())]
    assert all(tab.model.model().item(i).isEnabled() for i in range(tab.model.count())), names
    # At a level of neighbourhood windows none of them applies, and each says why.
    rng = np.random.default_rng(0)
    level = pd.DataFrame({"object_id": np.arange(1, 201), "centroid Y (µm)": rng.uniform(0, 100, 200),
                          "centroid X (µm)": rng.uniform(0, 100, 200),
                          "fraction T": np.r_[rng.uniform(0.6, 1, 100), rng.uniform(0, 0.4, 100)],
                          "n objects": rng.integers(3, 12, 200).astype(float)})
    s.add_level("Windows", level, select=True)
    tab.refresh()
    enabled = {tab.model.itemText(i): tab.model.model().item(i).isEnabled() for i in range(tab.model.count())}
    assert enabled == {"cell types": False, "nuclei": False, "old": False}
    i = tab.model.findText("nuclei")
    assert "image crops" in tab.model.itemData(i, Qt.ToolTipRole)
    ok, why = fits_level(read_card(tmp_path / "models" / "cell types"), s)
    assert not ok and "trained on segmented objects" in why and "neighbourhood windows" in why
    tab.model.setCurrentIndex(tab.model.findText("cell types"))
    assert not tab.run_btn.isEnabled() and "Not for this level" in tab.card.text()
    assert tab.run() is None and "not for this level" in tab.status.text()
    # A model trained on the windows is for windows - and not for the cells.
    s.add_phenotype("T-rich", "`fraction T` > 0.5")
    s.add_phenotype("other windows", "")
    s.label_from_classes("phenotype")  # the windows' own training set
    trainer.check_features("names", ["fraction T", "n objects"])
    trainer.train(wait=True)
    windows_model = trainer.save("window types")
    card = read_card(windows_model)
    assert card.training["level"] == {"kind": "windows", "depth": 1, "name": "Windows", "built": {}}
    tab.refresh()
    assert tab.model.currentText() == "window types" and tab.run_btn.isEnabled()
    assert tab.run() is not None
    s.set_level("Objects")
    tab.refresh()
    enabled = {tab.model.itemText(i): tab.model.model().item(i).isEnabled() for i in range(tab.model.count())}
    assert enabled == {"cell types": True, "nuclei": True, "old": True, "window types": False}
    # Fine-tuning offers only models for this level.
    trainer.refresh()
    offered = [trainer.start_from.itemText(i) for i in range(trainer.start_from.count())]
    assert "fine-tune cell types" in offered and "fine-tune window types" not in offered


# ---- the hierarchy map (H4b) -----------------------------------------------------------------


def _two_levels(s):
    """Neighbourhoods of the tiled cells, their windows, and regions of those windows."""
    settings = {"classes": "phenotype", "class_label": "P", "method": "radius", "radius": 6.0, "k": 10,
                "n_types": 2, "permutations": 0, "seed": 0}
    s.add_neighbourhoods("NB", settings, s.neighbourhood_job(settings)())
    s.make_windows_level("NB", "W", select=True)
    level_settings = {**settings, "classes": cy.LEVEL_TYPE, "class_label": "types"}
    s.add_neighbourhoods("NB2", level_settings, s.neighbourhood_job(level_settings)())
    s.make_regions_level("NB2", "R", touching=True, select=False)
    s.set_level("Objects")


def test_hierarchy_map_shows_and_navigates_the_levels(qtbot, tmp_path):
    from ibif_napari._cyto_map import COL_LEVEL, COL_SELECTION, type_mix
    from ibif_napari._cyto_workbench import Workbench

    v, s = _tiled_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    hmap = bench.hierarchy_map
    assert not hmap.isVisibleTo(bench) and not bench.map_btn.isChecked()  # one level: no map
    _two_levels(s)
    assert hmap.isVisibleTo(bench) and bench.map_btn.isChecked()
    items = hmap.items
    assert list(items) == ["Objects", "W", "R"]
    assert items["W"].parent() is items["Objects"] and items["R"].parent() is items["W"]
    assert items["W"].text(COL_LEVEL).startswith("◆") and items["R"].text(COL_LEVEL).startswith("▣")
    assert items["W"].text(1) == "16" and "windows · r 6" in items["W"].child(0).text(0)
    assert type_mix(s.root) and abs(sum(f for f, _ in type_mix(s.root)) - 1) < 1e-9
    # The Selection column follows what is shown: all 16 cells -> every window, every region.
    assert items["Objects"].text(COL_SELECTION) == "16 shown"
    assert items["W"].text(COL_SELECTION) == "16" and items["R"].text(COL_SELECTION) == str(len(s.level("R").table))
    bench.set_population("phenotype", "T")
    assert hmap.items["Objects"].text(COL_SELECTION) == "8 shown"
    # Click a level to work on it; double-click a count to follow the shown objects there.
    hmap._clicked(hmap.items["W"], COL_LEVEL)
    assert s.current_level == "W" and bench.level_box.currentData() == "W"
    hmap._clicked(hmap.items["Objects"], COL_LEVEL)
    bench.set_population("phenotype", "T")
    hmap._double_clicked(hmap.items["R"], COL_SELECTION)
    assert s.current_level == "R" and bench.current_population()[0] == "selection"
    # Export, rename, delete.
    s.set_level("Objects")
    path = hmap.export("W", tmp_path / "w.csv")
    assert pd.read_csv(path).shape[0] == 16
    assert hmap.rename("W", "Windows") and s.level_tree() == [(0, "Objects"), (1, "Windows"), (2, "R")]
    assert not hmap.rename("Objects", "Cells")
    bench.map_btn.setChecked(False)  # hidden on request, and stays hidden
    s.current.changed()
    assert not hmap.isVisibleTo(bench)
    bench.map_btn.setChecked(True)
    assert hmap.delete("Windows", confirm=False) == ["Windows", "R"] and s.level_names() == ["Objects"]


def test_a_level_is_out_of_date_when_its_parent_changes_and_can_be_made_again(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _tiled_scene()
    _two_levels(s)
    assert s.staleness("W") == "" and s.staleness("R") == ""
    # The cells' classes change: the windows were built from the old ones.
    s.edit_phenotype("T", rule="`CD3 mean` > 1e9")
    assert "changed since it was built" in s.staleness("W")
    assert s.staleness("R") == "W is out of date"
    bench = Workbench(v)
    qtbot.addWidget(bench)
    assert "(out of date)" in bench.hierarchy_map.items["W"].text(0)
    # The types it was built from are stale too: run the neighbourhoods again, then rebuild.
    settings = dict(s.analyses["NB"])
    s.remove_analysis("NB")
    assert "was deleted" in s.staleness("W")
    with pytest.raises(ValueError, match="deleted"):
        s.rebuild_level("W")
    s.add_neighbourhoods("NB", settings, s.neighbourhood_job(settings)())
    state = bench.hierarchy_map.rebuild("W", confirm=False)
    assert state is not None and s.current_level == "W" and s.staleness("W") == ""
    assert s.level_names() == ["Objects", "W"]  # what was built on the old windows is gone
    assert "fraction T" not in s.table.columns and (s.table["fraction Other"] == 1).all()  # new classes


def test_levels_are_skipped_not_fatal_when_their_objects_are_not_restored(tmp_path):
    v, s = _typed_scene()
    s.add_level("NH 1", _level_table(s))
    folder, _ = _workspace.save_workspace(v, tmp_path / "w")
    w = ViewerModel()  # the measurement table is not there: nothing to build the levels on
    notes = cy._load(w, folder, json.loads((folder / "workspace.json").read_text())["extensions"]["cytometry"])
    assert any("spatial levels were not restored" in n and "NH 1" in n for n in notes)
    assert cy.state_of(w).level_names() == ["Objects"]


def test_workbench_tab_order(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _typed_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    assert [bench.tabs.tabText(i) for i in range(bench.tabs.count())] == \
        ["Gating", "Clustering", "Classification", "Spatial", "Neighbourhoods", "Trajectory"]


def test_training_set_labels_from_phenotypes_gates_clicks_and_tables(qtbot, tmp_path):
    from ibif_napari._cyto_workbench import Workbench

    v, s = _typed_scene()  # 24 objects: 12 T (the left half), 12 Other, on 2 time points
    seen = []
    s.listen_training(lambda: seen.append(1))
    assert s.label_from_classes("phenotype") == 24 and seen
    assert s.training_counts() == {"T": 12, "Other": 12}
    assert s.training_classes["T"] == "#ff0000"  # the phenotype's colour
    # A gate's objects as one class; an object relabelled; one unlabelled.
    top = s.add_gate("top", "CD3 mean", "CD8 mean", [[0, 100], [1e4, 100], [1e4, 1e4], [0, 1e4]])
    assert s.label_population("gate", top.id, "CD8 T") == 4 and s.training_counts()["CD8 T"] == 4
    s.label_objects([0], None)
    assert s.training_labels()[0] is None and sum(s.training_counts().values()) == 23
    # Clicking an object in the viewer labels it as the active class.
    bench = Workbench(v)
    qtbot.addWidget(bench)
    trainer = bench.classification.train_dialog()
    qtbot.addWidget(trainer)
    trainer.active.setCurrentText("Other")
    trainer.click.setChecked(True)
    layer = v.layers["cells"]
    v.dims.set_point(0, 0)
    layer.selected_label = 0  # napari only tells about a change
    layer.selected_label = int(s.table.iloc[0]["label"])  # the first object of time point 0
    assert s.table.iloc[0]["T"] == 0
    assert s.training_labels()[0] == "Other"
    trainer.click.setChecked(False)
    before = s.training_labels()[1]
    layer.selected_label = int(s.table.iloc[1]["label"])
    assert s.training_labels()[1] == before != "Other"  # not labelling any more
    # Going to an example from the gallery does not label it, even while labelling.
    trainer.click.setChecked(True)
    n_cd8 = s.training_counts()["CD8 T"]
    trainer.show_what.setCurrentIndex(trainer.show_what.findData("class\tCD8 T"))
    assert trainer.gallery.count() == n_cd8 > 0
    trainer._example_clicked(trainer.gallery.item(0))
    assert s.training_counts()["CD8 T"] == n_cd8
    trainer.click.setChecked(False)
    # Relabel and remove from the gallery.
    trainer.gallery.item(0).setSelected(True)
    trainer.relabel_selected(None)
    assert s.training_counts()["CD8 T"] == n_cd8 - 1 == trainer.gallery.count()
    # A table of labels.
    import pandas as pd

    t = s.table
    csv = tmp_path / "labels.csv"
    pd.DataFrame({"T": t["T"][:3], "label": t["label"][:3], "class": ["A", "A", "B"]}).to_csv(csv, index=False)
    assert trainer.import_labels(str(csv)) == 3 and s.training_labels()[:3].tolist() == ["A", "A", "B"]
    # Saved with the workspace, by object ids.
    expected = s.training_labels().tolist()
    folder, notes = _workspace.save_workspace(v, tmp_path / "w")
    assert notes == []
    w = ViewerModel()
    assert _workspace.load_workspace(w, folder) == []
    restored = cy.state_of(w)
    assert restored.training_labels().tolist() == expected
    assert list(restored.training_classes) == list(s.training_classes)
    # Other objects: the labels go.
    table, _ = cell_types(n=30)
    s.set_table("another", table, "cells")
    assert not s.training


def test_training_workbench_thumbnails_train_review_save_and_fine_tune(qtbot, tmp_path, monkeypatch):
    from ibif_napari._cyto_training import object_thumbnail
    from ibif_napari._cyto_workbench import Workbench

    monkeypatch.setenv("IBIF_NAPARI_MODELS", str(tmp_path / "models"))
    v, s = _typed_scene()
    bench = Workbench(v)
    qtbot.addWidget(bench)
    trainer = bench.classification.train_dialog()
    qtbot.addWidget(trainer)
    trainer.fit_to_screen((1366, 768))
    trainer.show()
    assert trainer.width() <= 1366 and trainer.height() <= 768
    s.label_from_classes("phenotype")
    # A thumbnail: the CD3 image around the object, its outline in the class colour.
    trainer.image.setCurrentText("CD3")
    rgb = object_thumbnail(s, v.layers["CD3"], 0, 56, "#ff0000")
    assert rgb.shape == (56, 56, 3) and (rgb == [255, 0, 0]).all(axis=2).any()
    assert trainer.gallery.count() == 24 and "24 objects" in trainer.gallery_note.text()
    # Train with live curves; the results page shows the scores.
    trainer.check_features("names", ["CD3 mean", "CD8 mean"])
    trainer.split_by.setCurrentIndex(1)
    trainer.fraction.setValue(0.5)
    result = trainer.train(wait=True)
    assert result["metrics"]["balanced_accuracy"] == 1.0 and len(trainer._history["epoch"]) >= 1
    assert trainer.model_tabs.currentIndex() == 1 and "precision" in trainer.metrics.text()
    assert trainer.curves_fig.axes and trainer.save_btn.isEnabled()
    assert trainer.show_what.findData("wrong") >= 0
    trainer.show_what.setCurrentIndex(trainer.show_what.findData("wrong"))
    assert trainer.gallery.count() == 0  # nothing wrong
    first = trainer.save("cells T")
    tab = bench.classification
    assert tab.model.currentData() == first  # offered on the Classification tab at once
    # Fine-tune it with a new class: the model gains it.
    top = s.add_gate("top", "CD3 mean", "CD8 mean", [[0, 100], [1e4, 100], [1e4, 1e4], [0, 1e4]])
    s.label_population("gate", top.id, "CD8 T")
    trainer.refresh()
    trainer.start_from.setCurrentIndex(trainer.start_from.findData(first))
    trainer.split_by.setCurrentIndex(0)
    trainer.fraction.setValue(0.25)
    result = trainer.train(wait=True)
    model = result["model"]
    assert model.classes == ["CD8 T", "Other", "T"] and model.card.training["new_classes"] == ["CD8 T"]
    assert "new classes: CD8 T" in trainer.status.text()
    second = trainer.save("cells T + CD8")
    assert second != first and tab.model.currentData() == second
    info = tab.run()
    assert info["classes"] == ["CD8 T", "Other", "T"]


def test_a_levels_training_set_is_kept_by_object_id():
    v, s = _typed_scene()
    s.add_level("W", _level_table(s), select=True)
    s.add_phenotype("T-rich", "`fraction T` > 0.5")
    s.add_phenotype("rest", "")
    assert s.label_from_classes("phenotype") == 24 and s.object_keys() == ["object_id"]
    saved = s.current._training_to_dict()
    assert saved["keys"] == ["object_id"] and len(saved["labels"]) == 24
    expected = s.training_labels().tolist()
    s.current._training_from_dict(saved)
    assert s.training_labels().tolist() == expected
    assert s.object_id(3) == int(s.table.iloc[3]["object_id"])
    s.set_level("Objects")
    assert not s.training  # each level has its own training set


def test_double_clicking_a_gate_plots_what_it_was_drawn_on(qtbot):
    from ibif_napari._cyto_workbench import Workbench

    v, table = scene()
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    t = s.add_gate("T", cy.axis_column("CD3 mean", "asinh"), "CD8 mean", [[1, 0], [10, 0], [10, 1e4], [1, 1e4]])
    sub = s.add_gate("CD8+", "CD3 mean", "CD8 mean", [[0, 100], [1e4, 100], [1e4, 1e4], [0, 1e4]],
                     population=("gate", t.id))
    s.add_image_gate("left", [[[0, 0], [0, 20], [30, 20], [30, 0]]])
    bench = Workbench(v)
    qtbot.addWidget(bench)
    bench.x_col.setCurrentText("volume (µm³)")
    bench.tabs.setCurrentIndex(1)
    items = {tuple(bench.gate_tree.topLevelItem(i).data(0, Qt.UserRole)): bench.gate_tree.topLevelItem(i)
             for i in range(bench.gate_tree.topLevelItemCount())}
    bench._gate_double_clicked(items[("gate", t.id)])
    assert (bench.x_col.currentText(), bench.x_tf.currentText()) == ("CD3 mean", "asinh")
    assert (bench.y_col.currentText(), bench.y_tf.currentText()) == ("CD8 mean", "linear")
    assert bench.current_population() == ("all", None) and bench.tabs.currentIndex() == 0
    assert [g[2] for g in bench.plot._last["gates"]] == ["T"]  # its outline is on the plot
    assert "drawn on asinh(CD3 mean/150) × CD8 mean, within all objects" in bench.status.text()
    # A subgate: drawn within its parent gate.
    bench.select_gates([("gate", sub.id)])
    assert bench.plot_gate_btn.isEnabled() and bench.plot_selected_gate()
    assert bench.x_tf.currentText() == "linear" and bench.current_population() == ("gate", t.id)
    assert [g[2] for g in bench.plot._last["gates"]] == ["CD8+"]
    # An image gate has no plot; a gate on a column that is gone says so.
    assert not bench.plot_gate(("image", "left")) and "drawn on the image" in bench.status.text()
    s.gates.get(sub.id).x_axis = "no such column"
    assert not bench.plot_gate(("gate", sub.id)) and "not in the table any more" in bench.status.text()


def test_gallery_shows_a_gates_objects_and_labels_training_examples(qtbot):
    from ibif_napari._cyto_gallery import object_picture
    from ibif_napari._cyto_workbench import Workbench

    v, table = scene()  # 2 time points of 12 square cells; CD3 bright in the left half
    s = cy.state_of(v)
    s.set_table("cells", table, "cells")
    t = s.add_gate("T", "CD3 mean", "CD8 mean", [[100, 0], [1e4, 0], [1e4, 1e4], [100, 1e4]], color="#ff0000")
    top = s.add_gate("top", "CD3 mean", "CD8 mean", [[0, 100], [1e4, 100], [1e4, 1e4], [0, 1e4]])
    bench = Workbench(v)
    qtbot.addWidget(bench)
    bench.select_gates([("gate", t.id)])
    gallery = bench.open_gallery()
    qtbot.addWidget(gallery)
    assert gallery.current() == ("gate", t.id) and gallery.grid.count() == 12
    assert [b.currentText() for b, _ in gallery.channels] == ["CD3", "CD8", "(none)"]
    assert gallery.count.text() == "12 objects"
    # A thumbnail: CD3 in grey, CD8 in green, the outline in the gate's colour.
    row = gallery._rows[0]
    rgb = object_picture(s, row, gallery.channel_layers(), size=64, radius=20, outline="#ff0000")
    assert rgb.shape == (64, 64, 3) and (rgb == [255, 0, 0]).all(axis=2).any()
    assert (rgb[32, 32] > 0).all()  # the bright cell is at the centre
    # Following the lists; a limit and a random sample.
    bench.select_gates([("gate", top.id)])
    assert gallery.current() == ("gate", top.id) and gallery.grid.count() == 4
    gallery.show_population("all")
    gallery.limit.setValue(10)
    assert gallery.grid.count() == 10 and gallery.count.text() == "10 of 24 objects"
    gallery.sample.setChecked(True)
    assert gallery.grid.count() == 10 and "random sample" in gallery.count.text()
    # Training examples: the selected thumbnails, or all shown.
    gallery.sample.setChecked(False)
    gallery.limit.setValue(120)
    gallery.show_population("gate", t.id)
    gallery.grid.item(0).setSelected(True)
    gallery.grid.item(1).setSelected(True)
    gallery.train_class.setCurrentText("T cell")
    assert gallery.label(selected=True) == 2 and s.training_counts() == {"T cell": 2}
    assert gallery.label(selected=False) == 10 and s.training_counts() == {"T cell": 12}
    assert "T cell" in [gallery.train_class.itemText(i) for i in range(gallery.train_class.count())]
    # Clicking a thumbnail goes to the object; hovering enlarges it.
    gallery._clicked(gallery.grid.item(3))
    assert v.layers["cells"].selected_label == int(s.table.iloc[gallery._rows[3]]["label"])
    gallery._hover(gallery.grid.item(3))
    assert gallery._zoom.pixmap().width() == 128
    # The outline colour can be chosen; no channel: a hint.
    gallery.pick_outline("#00ffff")
    assert gallery.outline_colour() == "#00ffff"
    for box, _ in gallery.channels:
        box.setCurrentText("(none)")
    assert "Choose an image layer" in gallery.status.text()
