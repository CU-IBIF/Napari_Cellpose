"""Processing section: the step library, pipelines (order, settings, JSON), stubs that
cannot be added, running on layers, and the panel."""
import json

import numpy as np
import pytest

from ibif_napari import _processing
from ibif_napari._processing import Cancelled, Param, Pipeline, Stack, StepSpec


@pytest.fixture
def test_steps():
    """Two steps for the tests: add a value to every channel, and scale them, each
    recording what it did per time point."""

    def add(stack, params, progress):
        out = {}
        for i, (name, a) in enumerate(stack.channels.items()):
            progress(i / len(stack.channels), name)
            out[name] = a + params["value"]
        return stack.with_channels(out), [{"T": 0, "added": params["value"]}], f"+{params['value']:g}"

    def scale(stack, params, progress):
        return stack.with_channels({n: a * params["factor"] for n, a in stack.channels.items()})

    specs = [
        StepSpec("test_add", "Add a value", "Test", "Adds a value.", run=add,
                 params=(Param("value", "Value", "float", 1.0, minimum=-10, maximum=10),)),
        StepSpec("test_scale", "Scale", "Test", "Multiplies.", run=scale, needs_reference=True,
                 params=(Param("factor", "Factor", "float", 2.0, minimum=0, maximum=10),
                         Param("mode", "Mode", "choice", "a", options=("a", "b")),
                         Param("n", "Count", "int", 3, minimum=1, maximum=9),
                         Param("on", "On", "bool", True))),
    ]
    for s in specs:
        _processing.register(s)
    yield specs
    for s in specs:
        _processing.unregister(s.key)


def stack(shape=(2, 3, 4), channels=("A", "B")):
    return Stack({c: np.full(shape, k + 1.0, np.float32) for k, c in enumerate(channels)}, "TYX",
                 {"Y": 0.5, "X": 0.5})


# ---- library, parameters and stubs ---------------------------------------------------------

def test_library_lists_the_stubs_as_coming_soon():
    stubs = [s for s in _processing.library() if not s.available]
    assert {s.key for s in stubs} >= {"illumination", "deconvolution"}
    for s in stubs:
        assert s.status == _processing.COMING_SOON and s.run is None and "coming soon" in s.description
        assert s.params  # planned settings are shown
    lib = _processing.library()
    assert [s.available for s in lib] == sorted([s.available for s in lib], reverse=True)  # available first


def test_stubs_cannot_be_added():
    with pytest.raises(ValueError, match="coming soon"):
        Pipeline().add("illumination")
    with pytest.raises(ValueError, match="step 1: .*Deconvolution.*coming soon"):
        Pipeline.from_json(json.dumps({"steps": [{"key": "deconvolution"}]}))


def test_parameters_are_checked():
    p = Param("x", "X", "float", 1.0, minimum=0, maximum=2)
    assert p.check(1) == 1.0 and isinstance(p.check(1), float)
    for bad in (3, -1, "1", True, None):
        with pytest.raises(ValueError):
            p.check(bad)
    assert Param("n", "N", "int", 1).check(2.0) == 2
    with pytest.raises(ValueError, match="whole number"):
        Param("n", "N", "int", 1).check(2.5)
    with pytest.raises(ValueError, match="one of"):
        Param("c", "C", "choice", "a", options=("a", "b")).check("c")
    with pytest.raises(ValueError):
        Param("b", "B", "bool", True).check(1)
    with pytest.raises(ValueError, match="not one of the options"):
        Param("c", "C", "choice", "z", options=("a",))
    with pytest.raises(ValueError, match="unknown parameter kind"):
        Param("q", "Q", "text", "")


# ---- pipelines -----------------------------------------------------------------------------

def test_pipeline_order_settings_and_json(test_steps):
    pipe = Pipeline()
    pipe.add("test_add", {"value": 3})
    pipe.add("test_scale")
    assert pipe.steps[1].params == {"factor": 2.0, "mode": "a", "n": 3, "on": True}  # defaults
    with pytest.raises(ValueError, match="unknown setting"):
        pipe.add("test_add", {"valeu": 1})
    with pytest.raises(ValueError, match="above"):
        pipe.set_params(0, value=11)
    pipe.move(1, 0)
    assert [s.key for s in pipe] == ["test_scale", "test_add"]
    pipe.move(0, 99)  # clipped
    assert [s.key for s in pipe] == ["test_add", "test_scale"]
    pipe.steps[1].enabled = False
    back = Pipeline.from_json(pipe.to_json())
    assert back.to_dict() == pipe.to_dict() and back.to_dict()["format"] == "ibif-napari-processing"
    with pytest.raises(ValueError, match="unknown processing step 'nope'"):
        Pipeline.from_dict({"steps": [{"key": "nope"}]})
    with pytest.raises(ValueError, match="invalid JSON"):
        Pipeline.from_json("{")
    with pytest.raises(ValueError, match="no list of steps"):
        Pipeline.from_dict({"step": []})


def test_pipeline_runs_the_enabled_steps_in_order(test_steps):
    pipe = Pipeline()
    pipe.add("test_add", {"value": 3})
    pipe.add("test_scale", {"factor": 10})
    s = stack()
    seen = []
    result = pipe.run(s, lambda f, m: seen.append((round(f, 3), m)))
    assert np.all(result.stack.channels["A"] == (1 + 3) * 10) and np.all(result.stack.channels["B"] == 20 + 30)
    assert np.all(s.channels["A"] == 1)  # the input is unchanged
    assert [r["key"] for r in result.steps] == ["test_add", "test_scale"]
    assert result.steps[0]["records"] == [{"T": 0, "added": 3.0}] and result.steps[0]["summary"] == "+3"
    assert seen[0] == (0.0, "Add a value (1/2)") and seen[-1] == (1.0, "done")
    assert all(0 <= f <= 1 for f, _ in seen) and [f for f, _ in seen] == sorted(f for f, _ in seen)
    pipe.steps[1].enabled = False
    assert np.all(pipe.run(stack()).stack.channels["A"] == 4)
    pipe.steps[0].enabled = False
    with pytest.raises(ValueError, match="no enabled steps"):
        pipe.run(stack())


def test_a_step_that_changes_the_image_size_is_refused(test_steps):
    _processing.register(StepSpec("test_crop", "Crop", "Test", "", run=lambda s, p, pr: s.with_channels(
        {n: a[:, :1] for n, a in s.channels.items()})))
    try:
        pipe = Pipeline()
        pipe.add("test_crop")
        with pytest.raises(ValueError, match="changed the image's size"):
            pipe.run(stack())
    finally:
        _processing.unregister("test_crop")


def test_cancel_from_the_progress_callback(test_steps):
    pipe = Pipeline()
    pipe.add("test_add")

    def stop(fraction, message):
        if fraction > 0:
            raise Cancelled()

    with pytest.raises(Cancelled):
        pipe.run(stack(), stop)


def test_stack_checks():
    with pytest.raises(ValueError, match="differ in size"):
        Stack({"A": np.zeros((2, 2)), "B": np.zeros((2, 3))}, "YX")
    with pytest.raises(ValueError, match="do not match"):
        Stack({"A": np.zeros((2, 2))}, "ZYX")
    with pytest.raises(ValueError, match="no channels"):
        Stack({}, "YX")
    with pytest.raises(ValueError, match="reference channel"):
        Stack({"A": np.zeros((2, 2))}, "YX", reference="B")
    assert Stack({"A": np.zeros((2, 2))}, "YX").reference == "A"


# ---- the panel -----------------------------------------------------------------------------

napari = pytest.importorskip("napari")


@pytest.fixture
def viewer():
    from napari.components import ViewerModel

    v = ViewerModel()
    meta = {"ibif_axes": "TYX", "ibif_voxel_um": {"Y": 0.5, "X": 0.5}, "ibif_source": "/data/cells.nd2"}
    v.add_image(np.ones((2, 3, 4), np.float32), name="s GFP", scale=(1, 0.5, 0.5), metadata=dict(meta),
                colormap="green")
    v.add_image(np.full((2, 3, 4), 2, np.float32), name="s DAPI", scale=(1, 0.5, 0.5), metadata=dict(meta))
    return v


@pytest.fixture
def panel(viewer, qtbot):
    from ibif_napari._processing import session
    from ibif_napari._processing_widget import ProcessingWidget

    session.set_pipeline(viewer, Pipeline(), notify=False)
    w = ProcessingWidget(viewer)
    qtbot.addWidget(w)
    return w


def _library_row(w, key):
    from qtpy.QtCore import Qt

    return next(i for i in range(w.library.count()) if w.library.item(i).data(Qt.UserRole) == key)


def test_panel_shows_stubs_as_coming_soon(panel):
    row = _library_row(panel, "deconvolution")
    item = panel.library.item(row)
    assert item.text().endswith("— coming soon") and item.font().italic()
    panel.library.setCurrentRow(row)
    assert panel.step_title.text() == "Deconvolution (Richardson–Lucy)"
    assert "Feature coming soon" in panel.step_note.text()
    assert not panel.add_btn.isEnabled()
    assert panel.editor("iterations").value() == 20 and not panel.editor("iterations").isEnabled()
    assert panel.add_selected() is None and len(panel.pipeline) == 0  # not addable
    assert "coming soon" in panel.status.text()
    assert not panel.run_btn.isEnabled()


def test_panel_builds_and_edits_a_pipeline(test_steps, panel, tmp_path):
    panel.library.setCurrentRow(_library_row(panel, "test_scale"))
    assert panel.add_btn.isEnabled() and "Add it to the pipeline" in panel.step_note.text()
    panel.add_selected()
    panel.add_step("test_add")
    assert [s.key for s in panel.pipeline] == ["test_scale", "test_add"]
    assert panel.pipeline_list.currentRow() == 1 and panel.step_title.text() == "Add a value"
    panel.editor("value").setValue(4)
    assert panel.pipeline.steps[1].params["value"] == 4.0
    panel.pipeline_list.setCurrentRow(0)  # every kind of editor
    panel.editor("factor").setValue(3)
    panel.editor("mode").setCurrentText("b")
    panel.editor("n").setValue(5)
    panel.editor("on").setChecked(False)
    assert panel.pipeline.steps[0].params == {"factor": 3.0, "mode": "b", "n": 5, "on": False}
    assert panel.reference.isVisibleTo(panel)  # the scale step needs a reference channel
    panel.move_selected(1)
    assert [s.key for s in panel.pipeline] == ["test_add", "test_scale"] and panel.pipeline_list.currentRow() == 1
    assert not panel.down_btn.isEnabled() and panel.up_btn.isEnabled()
    # Dragging in the list reorders the pipeline.
    panel.pipeline_list.model().moveRow(panel.pipeline_list.rootIndex(), 1, panel.pipeline_list.rootIndex(), 0)
    assert [s.key for s in panel.pipeline] == ["test_scale", "test_add"]
    from qtpy.QtCore import Qt

    panel.pipeline_list.item(0).setCheckState(Qt.Unchecked)
    assert not panel.pipeline.steps[0].enabled
    path = panel.export_pipeline(tmp_path / "p.json")
    panel.pipeline_list.setCurrentRow(0)
    panel.remove_selected()
    assert [s.key for s in panel.pipeline] == ["test_add"] and not panel.reference.isVisibleTo(panel)
    assert panel.import_pipeline(path) is not None
    assert [(s.key, s.enabled) for s in panel.pipeline] == [("test_scale", False), ("test_add", True)]
    assert panel.pipeline_list.item(0).checkState() == Qt.Unchecked
    (tmp_path / "stub.json").write_text(json.dumps({"steps": [{"key": "illumination"}]}))
    assert panel.import_pipeline(tmp_path / "stub.json") is None
    assert "coming soon" in panel.status.text() and len(panel.pipeline) == 2  # unchanged


def test_panel_runs_on_layers(test_steps, panel, viewer, qtbot):
    panel.run()
    assert "Tick the channel" in panel.status.text()
    panel.check_channels(["s GFP", "s DAPI"])
    panel.run()
    assert "Add a step" in panel.status.text()
    panel.add_step("test_add")
    panel.editor("value").setValue(5)
    panel.run()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=10000)
    gfp, dapi = viewer.layers["s GFP [processed]"], viewer.layers["s DAPI [processed]"]
    assert np.all(gfp.data == 6) and np.all(dapi.data == 7) and np.all(viewer.layers["s GFP"].data == 1)
    assert tuple(gfp.scale) == (1, 0.5, 0.5) and gfp.colormap.name == "green"
    meta = gfp.metadata
    assert "ibif_source" not in meta  # a workspace must save the processed data, not reread the file
    assert meta["ibif_processed_from"] == "s GFP" and meta["ibif_axes"] == "TYX"
    assert meta["ibif_processing"][0]["key"] == "test_add" and meta["ibif_processing"][0]["params"] == {"value": 5.0}
    assert meta["ibif_processing"][0]["records"] == [{"T": 0, "added": 5.0}]
    assert "Made s GFP [processed], s DAPI [processed]" in panel.status.text() and "+5" in panel.status.text()


def test_panel_current_position_only(test_steps, panel, qtbot):
    v = panel.viewer
    v.layers.clear()
    v.add_image(np.arange(2 * 3 * 4 * 5, dtype=np.float32).reshape(2, 3, 4, 5), name="img",
                metadata={"ibif_axes": "PTYX"}, scale=(1, 1, 0.5, 0.5))
    v.dims.set_current_step(0, 1)  # position 1
    panel.check_channels(["img"])
    panel.scope.setCurrentText("Current position (every time point)")
    panel.add_step("test_add")
    panel.run()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=10000)
    out = v.layers["img [processed]"]
    assert out.data.shape == (3, 4, 5) and out.metadata["ibif_axes"] == "TYX"
    assert np.array_equal(out.data, v.layers["img"].data[1] + 1)


def test_panel_cancel(panel, viewer, qtbot):
    import time

    def slow(stack, params, progress):
        for i in range(200):
            progress(i / 200, "")
            time.sleep(0.02)
        return stack

    _processing.register(StepSpec("test_slow", "Slow", "Test", "", run=slow))
    try:
        panel.check_channels(["s GFP"])
        panel.add_step("test_slow")
        n = len(viewer.layers)
        panel.run()
        assert panel.run_btn.text() == "Cancel"
        panel.run()
        qtbot.waitUntil(lambda: panel._worker is None, timeout=10000)
        assert len(viewer.layers) == n and "Cancelled" in panel.status.text()
        assert panel.run_btn.text() == "Run pipeline"
    finally:
        _processing.unregister("test_slow")


def test_pipeline_is_saved_with_the_workspace(test_steps, panel, viewer, tmp_path):
    from napari.components import ViewerModel

    from ibif_napari import _workspace
    from ibif_napari._processing import session

    panel.add_step("test_add")
    panel.editor("value").setValue(2)
    folder, notes = _workspace.save_workspace(viewer, tmp_path / "w")
    state = json.loads((folder / "workspace.json").read_text())
    assert state["extensions"]["processing"]["steps"][0] == {"key": "test_add", "params": {"value": 2.0},
                                                           "enabled": True}
    v2 = ViewerModel()
    shown = []
    session.listen(v2, lambda: shown.append(1))
    assert _workspace.load_workspace(v2, folder) == [] and shown == [1]
    assert session.pipeline_of(v2).to_dict() == panel.pipeline.to_dict()
    # A panel open on the viewer shows the restored pipeline.
    session.set_pipeline(viewer, Pipeline())
    assert panel.pipeline_list.count() == 0
    _workspace.load_workspace(viewer, folder)
    assert panel.pipeline_list.count() == 1


def test_panel_registration_results(panel, viewer, qtbot, tmp_path):
    import pandas as pd

    from ibif_napari.vv import synth

    sc = synth.motion_scene(seed=1, n=3)
    viewer.layers.clear()
    meta = {"ibif_axes": "TYX"}
    viewer.add_image(sc.frames, name="GFP", scale=(1, 0.5, 0.5), metadata=dict(meta))
    labels = np.zeros(sc.frames.shape, np.uint16)
    labels[:, 70:90, 70:90] = 3
    viewer.add_labels(labels, name="cells", scale=(1, 0.5, 0.5), metadata=dict(meta))
    assert not panel.results_box.isVisibleTo(panel)
    panel.check_channels(["GFP"])
    panel.add_step("registration")
    assert panel.reference.isVisibleTo(panel) and panel.reference.currentText() == "GFP"
    panel.run()
    qtbot.waitUntil(lambda: panel._worker is None, timeout=30000)
    out = viewer.layers["GFP [processed]"]
    assert "3 frames along T" in panel.status.text()
    assert panel.results_box.isVisibleTo(panel) and panel.registered.currentText() == "GFP [processed]"
    table = pd.read_csv(panel.export_transforms(tmp_path / "t.csv"))
    assert list(table["T"]) == [0, 1, 2] and {"shift Y (µm)", "m00", "m12", "correlation after"} <= set(table)
    assert "matrix" not in table
    moved = panel.apply_registration("cells")
    assert moved.name == "cells [registered]" and moved.data.dtype == np.uint16
    assert set(np.unique(moved.data)) == {0, 3} and moved.metadata["ibif_registered_like"] == "GFP [processed]"
    viewer.add_labels(np.zeros((3, 10, 10), np.uint16), name="small")
    assert panel.apply_registration("small") is None and "is not the size" in panel.status.text()
    assert out.metadata["ibif_processing"][0]["records"][1]["registered"]
