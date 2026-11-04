"""Tests of the V&V suite itself: data generation, metrics, checks, report and CLI."""
import dataclasses
import json
from pathlib import Path

import numpy as np
import pytest

from ibif_napari.vv import metrics, synth
from ibif_napari.vv.requirements import REQUIREMENTS, markdown

ROOT = Path(__file__).resolve().parents[1]


def tiny(name="t", **kw):
    base = dict(shape=(64, 64), spacing=(0.5, 0.5), n_objects=4, seed=3)
    base.update(kw)
    return synth.SceneSpec(name, **base)


def test_scenes_are_reproducible_and_consistent():
    a, b = synth.generate(tiny()), synth.generate(tiny())
    assert np.array_equal(a.acquired["nuclei"], b.acquired["nuclei"])
    assert np.array_equal(a.nuclei, b.nuclei)
    assert not np.array_equal(a.acquired["nuclei"], synth.generate(tiny(seed=4)).acquired["nuclei"])
    # Every nucleus sits inside its own cell; numbers 1..n.
    assert set(np.unique(a.nuclei)) == set(range(5))
    assert np.all(a.cells[a.nuclei > 0] == a.nuclei[a.nuclei > 0])
    # The truth table matches the arrays.
    t = a.truth.set_index("label")
    assert t.loc[2, "nucleus voxels"] == np.sum(a.nuclei == 2)
    assert t.loc[2, "nucleus nuclei acquired mean"] == pytest.approx(a.acquired["nuclei"][a.nuclei == 2].mean())


def test_noise_level_follows_the_snr():
    lo, hi = synth.generate(tiny(snr=2)), synth.generate(tiny(snr=20))
    assert lo.noise_sigma / hi.noise_sigma == pytest.approx(10)
    # Mean nuclear signal / noise sigma = SNR; the image noise has that sigma.
    signal = hi.truth["nucleus intensity"].mean() * hi.spec.peak
    assert signal / hi.noise_sigma == pytest.approx(20)
    residual = hi.acquired["nuclei"].astype(float) - hi.ideal["nuclei"]
    assert residual.std() == pytest.approx(hi.noise_sigma, rel=0.05)


def test_3d_cells_time_points_and_files(tmp_path):
    spec = tiny(shape=(8, 40, 40), spacing=(1.0, 0.5, 0.5), n_objects=3, timepoints=2, intensity_drift=0.5,
                channels=("nuclei", "cytoplasm", "membrane"), nucleus_radius_um=(2.0, 2.5),
                psf_sigma_um=(0.8, 0.4, 0.4), origin_um=(1.0, 2.0, 3.0))
    scene = synth.generate(spec)
    assert scene.acquired["membrane"].shape == (2, 8, 40, 40)
    t = scene.truth
    assert len(t) == 6 and set(t["T"]) == {0, 1}
    m0 = t[t["T"] == 0]["nucleus nuclei ideal mean"].to_numpy() - spec.background
    m1 = t[t["T"] == 1]["nucleus nuclei ideal mean"].to_numpy() - spec.background
    assert m1 / m0 == pytest.approx(1.5)
    assert np.all(scene.ideal["cytoplasm"][0][scene.nuclei > 0] < scene.ideal["cytoplasm"][0][
        (scene.cells > 0) & (scene.nuclei == 0)].mean())
    back = synth.load(synth.save(scene, tmp_path / "s"))
    assert back.spec == spec
    assert np.array_equal(back.acquired["membrane"], scene.acquired["membrane"])
    assert back.truth.shape == scene.truth.shape


def test_metrics_on_hand_made_cases():
    truth = np.zeros((20, 20), int)
    truth[2:6, 2:6] = 1  # 16 px
    truth[10:14, 10:14] = 2
    truth[2:6, 12:18] = 3
    pred = np.zeros_like(truth)
    pred[2:6, 2:5] = 7  # IoU 12/16 = 0.75
    pred[10:14, 10:14] = 9  # perfect
    pred[15:19, 0:3] = 4  # false positive; object 3 missed
    m = metrics.match(truth, pred)
    assert (m["n_true"], m["n_pred"]) == (3, 3)
    k = m["thresholds"].index(0.75)
    assert m["tp"][0] == 2 and m["fp"][0] == 1 and m["fn"][0] == 1
    assert m["f1@0.5"] == pytest.approx(2 / 3) and m["ap@0.5"] == pytest.approx(0.5)
    assert m["tp"][k] == 2 and m["tp"][k + 1] == 1  # 0.75 still matches, 0.80 does not
    assert m["mean_iou"] == pytest.approx((0.75 + 1) / 2)
    assert m["pq"] == pytest.approx(m["mean_iou"] * 2 / 3)
    empty = metrics.match(np.zeros((4, 4), int), np.zeros((4, 4), int))
    assert empty["f1@0.5"] == 1.0 and empty["n_true"] == 0
    with pytest.raises(ValueError):
        metrics.match(np.zeros((4, 4)), np.zeros((5, 4)))


def test_surface_distance():
    a = np.zeros((30, 30), bool)
    a[5:25, 5:25] = True
    b = np.zeros_like(a)
    b[6:24, 6:24] = True
    mean, p95 = metrics.surface_distance(a, b, (0.5, 0.5))
    assert 0.5 <= mean <= 0.75 and p95 <= 0.75


def test_requirements_doc_is_up_to_date():
    ids = [r.id for r in REQUIREMENTS]
    assert len(ids) == len(set(ids))
    from ibif_napari.vv.checks import CHECKS

    assert set(CHECKS) == set(ids)  # every requirement has a check, and vice versa
    doc = (ROOT / "docs" / "vv" / "requirements.md").read_text(encoding="utf-8")
    assert doc == markdown(), "regenerate with: ibif-napari-vv requirements > docs/vv/requirements.md"


@pytest.fixture(scope="module")
def quick_truth(tmp_path_factory):
    folder = tmp_path_factory.mktemp("vv") / "truth"
    synth.generate_all(folder, quick=True)
    return folder


def test_suite_with_a_perfect_segmentation_passes_everything(quick_truth, tmp_path):
    pytest.importorskip("napari")
    from ibif_napari.vv.report import build
    from ibif_napari.vv.run import run

    summary = run(quick_truth, tmp_path / "res", "truth", log=lambda *a: None)
    statuses = {e["id"]: e["status"] for e in summary["results"]}
    allowed = {"PASS"}
    for e in summary["results"]:
        if e["status"] == "NOT RUN":  # e.g. pylibCZIrw not installed here
            assert e["id"] in ("VV-IO-01", "VV-IO-02"), e
            allowed.add("NOT RUN")
    assert set(statuses.values()) <= allowed, {k: v for k, v in statuses.items() if v != "PASS"}
    html, md = build(tmp_path / "res")
    text = html.read_text(encoding="utf-8")
    assert "Why the tool is working" in text and "data:image/png;base64," in text
    assert "not Cellpose" in text  # the oracle is flagged as a stand-in
    assert "| VV-SEG-01 |" in md.read_text(encoding="utf-8")
    saved = json.loads((tmp_path / "res" / "results.json").read_text())
    assert saved["backend"] == "truth" and saved["counts"]["PASS"] >= 16
    assert (tmp_path / "res" / "predictions" / "nuclei_2d_nuclei.tif").exists()


def test_a_poor_segmentation_fails_and_cellpose_absent_is_not_run(quick_truth, tmp_path):
    from ibif_napari.vv import backends
    from ibif_napari.vv.checks import Context, run_checks

    class Shrunk(backends.TruthBackend):
        name = "shrunk"

        def segment(self, image, **kw):
            from scipy import ndimage as ndi

            labels = super().segment(image, **kw)
            return np.where(ndi.binary_erosion(labels > 0, iterations=3), labels, 0)

    ctx = Context(quick_truth, tmp_path, Shrunk(), log=lambda *a: None)
    (res,) = run_checks(ctx, only={"VV-SEG-01"})
    assert res.status == "FAIL"
    assert [c["passed"] for c in res.criteria][1] is False  # the IoU criterion

    class Missing(backends.Backend):
        name = "cellpose"

        def available(self):
            return False, "no weights"

    ctx = Context(quick_truth, tmp_path, Missing(), log=lambda *a: None)
    results = run_checks(ctx, only={"VV-SEG-01", "VV-SEC-01"})
    assert [r.status for r in results] == ["NOT RUN", "PASS"]
    assert "no weights" in results[0].notes[0]


def test_cli(tmp_path, capsys):
    from ibif_napari.vv.cli import main

    assert main(["requirements"]) == 0
    assert "VV-SEG-02" in capsys.readouterr().out
    spec = dataclasses.replace(synth.standard_scenes(True)[0])
    synth.generate_all(tmp_path / "truth", specs=[spec])
    assert main(["run", str(tmp_path / "truth"), str(tmp_path / "res"), "--backend", "truth",
                 "--only", "VV-SEG-01", "VV-H-01"]) == 0
    assert (tmp_path / "res" / "report.html").exists()
    limits = tmp_path / "limits.json"
    limits.write_text(json.dumps({"seg2d_iou": 1.01}))  # impossible: must fail
    assert main(["run", str(tmp_path / "truth"), str(tmp_path / "res2"), "--backend", "truth",
                 "--only", "VV-SEG-01", "--limits", str(limits)]) == 1


def test_broken_cellpose_install_is_reported_not_raised(monkeypatch):
    # A PyTorch / torchvision mismatch raises RuntimeError on import, not ImportError.
    from ibif_napari import _cellpose
    from ibif_napari.vv.backends import CellposeBackend

    def broken():
        raise RuntimeError("operator torchvision::nms does not exist")

    monkeypatch.setattr(_cellpose, "default_model", broken)
    ok, why = CellposeBackend().available()
    assert not ok and "torchvision::nms" in why
    assert CellposeBackend().describe()["backend"] == "cellpose"
