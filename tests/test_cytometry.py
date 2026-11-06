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
