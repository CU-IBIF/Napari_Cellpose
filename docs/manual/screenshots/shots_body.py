"""Drive IBIF napari on synthetic data and save the user guide's screenshots.

Started by run_shots.py (see README.md in this folder).
"""
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
OUT = Path(sys.argv[1])
OUT.mkdir(parents=True, exist_ok=True)
ONLY = set(sys.argv[2:])

import numpy as np  # noqa: E402
import napari  # noqa: E402
from qtpy.QtCore import QCoreApplication, QEvent, Qt  # noqa: E402
from qtpy.QtWidgets import QApplication  # noqa: E402

import data  # noqa: E402
from ibif_napari import _process  # noqa: E402
from ibif_napari._reader import layer_data  # noqa: E402

# Segment with the stand-in in a child process, like the real Cellpose panel does.
os.environ["PYTHONPATH"] = str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", "")
_process._process = _process.CellposeProcess(backend="demo_seg")
os.environ["IBIF_NAPARI_PRELOAD"] = ""
from ibif_napari import _cellpose  # noqa: E402

_cellpose.ensure_weights = lambda name: iter(())  # weights can't be downloaded here


def pump(seconds=0.3):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        QApplication.processEvents()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
        time.sleep(0.02)


def wait(cond, timeout=300, what=""):
    end = time.monotonic() + timeout
    while not cond():
        if time.monotonic() > end:
            raise TimeoutError(what)
        pump(0.1)
    pump(0.5)


def snap(name, widget=None):
    pump(0.3)
    from napari._qt.dialogs.qt_notification import NapariQtNotification
    for toast in win.findChildren(NapariQtNotification) if 'win' in globals() else []:
        toast.close()
    pump(0.6)
    w = widget or viewer.window._qt_window
    img = w.grab()
    img.save(str(OUT / f"{name}.png"))
    print("saved", name, img.width(), img.height(), flush=True)


viewer = napari.Viewer(title="IBIF napari")
win = viewer.window._qt_window
win.resize(1560, 980)
dock, panel = viewer.window.add_plugin_dock_widget("ibif-napari", "IBIF napari")
dock.setMinimumWidth(470)
pump(1)

# ---- the sample acquisition, loaded the way the ND2 reader loads files -------------------
arr, names, emissions, scene = data.tissue()
sizes = {"T": arr.shape[0], "Z": arr.shape[1], "C": arr.shape[2], "Y": arr.shape[3], "X": arr.shape[4]}
for ld in layer_data(arr, sizes, names, emissions, data.SPACING, 60.0, "sample", "sample.nd2"):
    viewer._add_layer_from_data(*ld)
names = [f"sample {n}" for n in names]
viewer.dims.set_current_step(1, arr.shape[1] // 2)
viewer.reset_view()
viewer.camera.zoom *= 1.35
pump(2)
snap("01_overview")

# the image chooser for multi-image files (LIF projects)
from ibif_napari._chooser import ImageChooser  # noqa: E402

scenes = [SimpleNamespace(name=n, summary=s) for n, s in (
    ("Region 1 / Tile scan", "T 2 × Z 14 × C 4 × 220 × 300"),
    ("Region 2 / Tile scan", "T 2 × Z 14 × C 4 × 220 × 300"),
    ("Region 3 / Tile scan", "T 2 × Z 14 × C 4 × 220 × 300"),
    ("Neuron 40x / z-stack", "Z 16 × C 1 × 300 × 320"),
    ("Overview 10x", "C 4 × 1024 × 1024"))]
chooser = ImageChooser("sample.lif", scenes)
chooser.show()
snap("02_lif_chooser", chooser)
chooser.close()

# icon bar close-up
snap("03_panel", dock)

# ---- 3D view --------------------------------------------------------------------------
config = panel.pages["configuration"]
config.btn_3d.click()
viewer.camera.angles = (-35, 25, 60)
pump(1.5)
snap("04_view3d")
config.btn_2d.click()
viewer.reset_view()
viewer.camera.zoom *= 1.35
pump(1)

# ---- segmentation ------------------------------------------------------------------------
seg = panel.show_section("segmentation")
wait(lambda: "ready" in seg.status.text() or "Could not" in seg.status.text(), 180, "model ready")
print("status:", seg.status.text(), flush=True)
seg.image.setCurrentText(names[0])
seg.nuclei.setCurrentIndex(0)
seg.diameter.setValue(18)
from ibif_napari._cellpose_widget import SCOPE_ALL
seg.scope.setCurrentText(SCOPE_ALL)
for layer in viewer.layers:
    if layer.name in names[1:]:
        layer.visible = False
n_before = len(viewer.layers)
seg.run()
wait(lambda: len(viewer.layers) > n_before and seg.run_btn.text() == "Run Cellpose", 600, "segmentation")
nuclei = viewer.layers[-1]
nuclei.name = "nuclei"
nuclei.contour = 0
nuclei.opacity = 0.8
viewer.layers.selection.active = viewer.layers[names[0]]
snap("05_segmentation")

# region drawing
seg.add_region_layer()
region = viewer.layers[-1]
yy, xx = np.indices(region.data.shape[-2:])
shape = (((yy - 110) / 70.0) ** 2 + ((xx - 120) / 95.0) ** 2) <= 1
painted = np.asarray(region.data)
painted[..., shape] = 1
region.data = painted
region.refresh()
viewer.layers.selection.active = region
pump(1)
snap("06_region")
viewer.layers.remove(region)
pump(0.5)

# ---- secondary objects -------------------------------------------------------------------
sec = panel.show_section("secondary_objects")
from ibif_napari._secondary_widget import CELLS  # noqa: E402

sec.primary.setCurrentText("nuclei")
sec.method.setCurrentText(CELLS)
for i in range(sec.channels.count()):
    item = sec.channels.item(i)
    item.setCheckState(Qt.Checked if item.text() == names[1] else Qt.Unchecked)
sec.kind.setCurrentIndex(0)  # membranes
sec.max_distance.setValue(3)

viewer.layers[names[1]].visible = True
n_before = len(viewer.layers)
sec.run()
wait(lambda: sec._worker is None and len(viewer.layers) > n_before, 300, "secondary")
cells = viewer.layers["nuclei cells"] if "nuclei cells" in viewer.layers else viewer.layers[-2]
cells.contour = 2
nuclei.opacity = 0.35
viewer.layers["nuclei cytoplasm"].visible = False
viewer.layers[names[0]].visible = True
snap("07_secondary")

# ---- measurements ------------------------------------------------------------------------
meas = panel.show_section("measurements")
meas.check(objects=["nuclei", cells.name, "nuclei cytoplasm"], channels=names)
meas.measure()
wait(lambda: meas._worker is None and meas.tables, 300, "measure")
dock.setMinimumWidth(640)
pump(1)
snap("08_measurements")
dock.setMinimumWidth(470)

# ---- cytometry ---------------------------------------------------------------------------
cyto = panel.show_section("cytometry")
cyto._fill_tables()
cyto._fill_layers()
cyto.table_box.setCurrentText(cells.name)
cyto.layer_box.setCurrentText(cells.name)
pump(0.5)
cyto.use(cells.name, cells.name)
cyto.refresh()
print("cyto:", cyto.status.text(), flush=True)
pump(0.5)
bench = cyto.open_workbench()
bench.resize(1300, 820)
from ibif_napari import _cytometry as cy  # noqa: E402

state = cy.state_of(viewer)
table = state.table
print([c for c in table.columns][:40], flush=True)
X = next(c for c in table.columns if c.endswith("CD3 mean"))
Y = next(c for c in table.columns if c.endswith("CD8 mean"))
bench.x_col.setCurrentText(X)
bench.y_col.setCurrentText(Y)
bench.replot()
xs, ys = table[X].to_numpy(), table[Y].to_numpy()
cut3 = float(np.percentile(xs, 30) + np.percentile(xs, 70)) / 2
big = float(max(xs.max(), ys.max()) * 1.04)
t_gate = bench.add_gate("CD3+", [[cut3, 0], [big, 0], [big, big], [cut3, big]])
bench.set_population("gate", t_gate.id)
ys_t = ys[xs > cut3]
cut8 = float(np.percentile(ys_t, 25) + np.percentile(ys_t, 75)) / 2
cd8_gate = bench.add_gate("CD8+", [[0, cut8], [big, cut8], [big, big], [0, big]])
bench.set_population("all")
bench.gate_tree.setCurrentItem(bench.gate_tree.topLevelItem(0))
bench.phenotype_from_gate(name="T cell")
state.phenotype_from("gate", cd8_gate.id, "CD8 T", parent="T cell")
state.add_phenotype("Other T", "", parent="T cell")
state.add_phenotype("Non-T", "NOT `gate: CD3+`")
bench.refresh()
bench.color_by.setCurrentText("Phenotype")
bench.replot()
bench.tabs.setCurrentIndex(0)
pump(1)
snap("09_workbench_gating", bench)

layer = state.show_phenotypes(None)
cyto.refresh() if hasattr(cyto, "refresh") else None
for l in viewer.layers:
    if l.name in ("nuclei", cells.name, *names):
        l.visible = False
layer.opacity = 0.9
pump(1)
snap("10_phenotypes_image")

tab = bench.clustering
bench.tabs.setCurrentIndex(1)
tab.check_features("names", [X, Y])
tab.map.setCurrentIndex(tab.map.findData("umap"))
tab.method.setCurrentIndex(tab.method.findData("leiden"))
info = tab.run(wait=True)
pump(1)
snap("11_workbench_clustering", bench)
bench.close()

# ---- save workspace ------------------------------------------------------------------------
panel.show_section("save_workspace")
pump(0.5)
dock.grab(__import__("qtpy.QtCore", fromlist=["QRect"]).QRect(0, 0, dock.width(), 300)).save(str(OUT / "12_workspace.png"))
print("saved 12_workspace", flush=True)

# ---- filament tracing ----------------------------------------------------------------------
viewer.layers.clear()
img, sp = data.neuron()
viewer.add_image(img, name="neuron", scale=sp, colormap="gray",
                 metadata={"ibif_axes": "ZYX", "ibif_voxel_um": dict(zip("ZYX", sp))})
viewer.layers["neuron"].contrast_limits = (200, 1100)
viewer.dims.set_current_step(0, 6)
viewer.reset_view()
viewer.camera.zoom *= 1.3
fil = panel.show_section("filament_tracing")
fil.image.setCurrentText("neuron")
fil.d_min.setValue(0.3)
fil.d_max.setValue(1.0)
fil.run()
wait(lambda: fil._worker is None and not fil._pending, 600, "trace")
pump(1)
snap("13_filaments")
config = panel.pages["configuration"]
config.btn_3d.click()
viewer.camera.angles = (-25, 20, 70)
viewer.camera.zoom *= 1.2
pump(1.5)
snap("14_filaments_3d")
print("DONE", flush=True)
QApplication.quit()
