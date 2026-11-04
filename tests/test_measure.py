import numpy as np
import pandas as pd
import pytest

from ibif_napari._measure import combine, measure_block, save_table


def test_measure_block_2d():
    labels = np.zeros((10, 10), np.uint16)
    labels[1:3, 1:4] = 4  # 6 px
    labels[6:9, 6:9] = 7  # 9 px
    image = np.arange(100, dtype=np.float32).reshape(10, 10)
    t = measure_block(labels, {"GFP": image}, spacing=(0.5, 0.5), origin=(10.0, 20.0))
    assert list(t["label"]) == [4, 7] and list(t["voxels"]) == [6, 9]
    assert t["area (µm²)"].tolist() == pytest.approx([1.5, 2.25])
    obj = image[labels == 4]
    row = t.iloc[0]
    for stat, value in (("mean", obj.mean()), ("median", np.median(obj)), ("min", obj.min()),
                        ("max", obj.max()), ("std", obj.std()), ("sum", obj.sum())):
        assert row[f"GFP {stat}"] == pytest.approx(value)
    assert row["centroid Y (µm)"] == pytest.approx(10 + 1.5 * 0.5)
    assert row["centroid X (µm)"] == pytest.approx(20 + 2 * 0.5)
    assert row["equivalent diameter (µm)"] == pytest.approx(np.sqrt(4 * 1.5 / np.pi))


def test_measure_block_3d_and_empty():
    labels = np.zeros((4, 6, 6), np.uint8)
    labels[1:3, 2:4, 2:4] = 1  # 8 voxels of 2 x 0.5 x 0.5 µm
    t = measure_block(labels, spacing=(2, 0.5, 0.5))
    assert t["volume (µm³)"].tolist() == pytest.approx([8 * 0.5])
    assert t["centroid Z (µm)"].tolist() == pytest.approx([3.0])
    assert len(measure_block(np.zeros((4, 4), np.uint8))) == 0
    with pytest.raises(ValueError, match="differ in size"):
        measure_block(labels, {"x": np.zeros((2, 2))})


def test_combine_and_save(tmp_path):
    nuclei = pd.DataFrame({"T": [0, 0, 1], "label": [1, 2, 1], "area (µm²)": [5.0, 6.0, 7.0]})
    cells = pd.DataFrame({"T": [0, 1, 1], "label": [1, 1, 3], "area (µm²)": [20.0, 21.0, 9.0],
                          "GFP mean": [1.0, 2.0, 3.0]})
    c = combine({"nuclei": nuclei, "cells": cells})
    assert list(c.columns) == ["T", "label", "nuclei: area (µm²)", "cells: area (µm²)", "cells: GFP mean"]
    assert len(c) == 4  # (0,1) (0,2) (1,1) (1,3)
    row = c[(c["T"] == 0) & (c["label"] == 1)].iloc[0]
    assert row["nuclei: area (µm²)"] == 5 and row["cells: area (µm²)"] == 20
    assert np.isnan(c[(c["T"] == 0) & (c["label"] == 2)]["cells: area (µm²)"]).all()  # nucleus without cell
    csv = save_table(c, tmp_path / "m.csv")
    assert pd.read_csv(csv).shape == c.shape
    xlsx = save_table(c, tmp_path / "m.xlsx")
    assert pd.read_excel(xlsx).shape == c.shape
