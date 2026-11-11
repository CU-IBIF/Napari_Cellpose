import json

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("napari")
pytest.importorskip("vtea_core")

from napari.components import ViewerModel  # noqa: E402

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
    titles = [a.text() for a in plot.points_menu.actions()]
    assert titles == ["Size", "Style", "Opacity"]
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
