import numpy as np
import pandas as pd
import pytest

pytest.importorskip("napari")

from test_secondary_widget import make_viewer  # noqa: E402

from ibif_napari._measure import tables_of  # noqa: E402
from ibif_napari._measure_widget import COMBINED, MeasureWidget  # noqa: E402
from ibif_napari._secondary import seeded_cells  # noqa: E402


def test_measurements_panel(qtbot, tmp_path):
    v = make_viewer()  # T=2 Z=3 membrane + DAPI channels; nuclei labels cropped to a region
    nuclei = v.layers["nuclei"]
    cells = np.stack([seeded_cells(nuclei.data[t], v.layers["s Membrane"].data[t, :, 5:35, 0:50],
                                   spacing=(2, .5, .5), factor=0.5, smooth=0)[0] for t in range(2)])
    v.add_labels(cells, name="cells", scale=nuclei.scale, translate=nuclei.translate, metadata=dict(nuclei.metadata))
    w = MeasureWidget(v)
    qtbot.addWidget(w)
    assert not w.save_btn.isEnabled()
    w.measure()
    assert "Tick at least one" in w.status.text()

    w.check(objects=["nuclei", "cells"], channels=["s Membrane"])
    w.measure()
    qtbot.waitUntil(lambda: w._worker is None, timeout=30000)
    assert w.status.text().startswith("Measured nuclei: 4, cells: 4")
    assert [w.view.itemText(i) for i in range(w.view.count())] == ["nuclei", "cells", COMBINED]
    combined = w.tables[COMBINED]
    assert list(combined[["T", "label"]].itertuples(index=False, name=None)) == [(0, 1), (0, 2), (1, 1), (1, 2)]
    assert "nuclei: volume (µm³)" in combined and "cells: s Membrane mean" in combined
    # Nucleus volume: 49 px per plane x 3 planes x 2 x 0.5 x 0.5 µm³.
    assert combined["nuclei: volume (µm³)"].iloc[0] == pytest.approx(49 * 3 * 0.5)
    # Membrane intensity inside nucleus 1 (cell interior, 0.3) and whole cell 1 (includes membrane).
    assert combined["nuclei: s Membrane mean"].iloc[0] == pytest.approx(0.3)
    assert combined["cells: s Membrane max"].iloc[0] == pytest.approx(1.0)
    assert w.model.rowCount() == 4 and tables_of(v)[COMBINED] is combined

    # Clicking a row goes to that object: time point 1, nucleus 2.
    assert w.proxy_row(T=1, label=2).row() == 3  # shown in natural order
    w._go_to(w.proxy_row(T=1, label=2))
    assert v.dims.current_step[0] == 1 and nuclei.selected_label == 2
    camera = getattr(getattr(v, "scene", None), "camera", None) or v.camera
    assert camera.center[-1] == pytest.approx(combined["nuclei: centroid X (µm)"].iloc[3])

    paths = w.save_all(tmp_path)
    assert sorted(p.split("/")[-1] for p in paths) == [
        "all objects measurements.csv", "cells measurements.csv", "nuclei measurements.csv"]
    assert pd.read_csv(tmp_path / "all objects measurements.csv").shape == combined.shape
    w.view.setCurrentText("nuclei")
    assert w.save_table(str(tmp_path / "n.xlsx")).endswith("n.xlsx")
