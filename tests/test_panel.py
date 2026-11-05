import numpy as np
import pytest

pytest.importorskip("napari")

from napari.components import ViewerModel  # noqa: E402

from ibif_napari import _cellpose, _workspace  # noqa: E402
from ibif_napari._panel import ICONS, SECTIONS, IbifPanel  # noqa: E402

ORDER = ["Configuration", "Segmentation", "Secondary objects", "Measurements", "Cytometry",
         "Filament tracing", "Save workspace"]


@pytest.fixture
def panel(qtbot, monkeypatch):
    monkeypatch.setenv(_cellpose.PRELOAD_ENV, "0")
    viewer = ViewerModel()
    w = IbifPanel(viewer)
    qtbot.addWidget(w)
    return w


def test_icon_buttons_in_order_with_tooltips(panel):
    assert [s.title for s in SECTIONS] == ORDER
    for s in SECTIONS:
        button = panel.buttons[s.key]
        assert (ICONS / f"{s.key}.png").exists() and not button.icon().isNull()
        assert button.text() == ""  # the name is in the tooltip, not on the button
        assert s.title in button.toolTip() and button.accessibleName() == s.title
    # Cytometry is there but inactive for now.
    assert not panel.buttons["cytometry"].isEnabled() and panel.pages["cytometry"] is None
    assert all(panel.buttons[s.key].isEnabled() for s in SECTIONS if s.key != "cytometry")


def test_opens_on_configuration_and_switches(panel):
    from ibif_napari._cellpose_widget import CellposeWidget
    from ibif_napari._dims_widget import DimensionsWidget
    from ibif_napari._workspace_widget import WorkspaceWidget

    assert panel.current_section() == "configuration" and panel.title.text() == "Configuration"
    assert isinstance(panel.pages["configuration"], DimensionsWidget)
    assert panel.buttons["configuration"].isChecked()
    panel.buttons["segmentation"].click()
    assert panel.current_section() == "segmentation" and panel.title.text() == "Segmentation"
    assert isinstance(panel.pages["segmentation"], CellposeWidget)
    assert not panel.buttons["configuration"].isChecked()
    assert isinstance(panel.show_section("save_workspace"), WorkspaceWidget)
    with pytest.raises(ValueError):
        panel.show_section("cytometry")


def test_loading_a_workspace_goes_back_to_configuration(panel, tmp_path):
    viewer = panel.viewer
    viewer.add_image(np.zeros((8, 8), np.uint16), name="img")
    folder, _ = _workspace.save_workspace(viewer, tmp_path / "w")
    panel.show_section("measurements")
    assert _workspace.load_workspace(viewer, folder) == []
    assert panel.current_section() == "configuration"
    # From the Save workspace section's own Open button too.
    panel.show_section("save_workspace")
    assert panel.pages["save_workspace"].open(str(folder), replace=True) == []
    assert panel.current_section() == "configuration"


def test_manifest_lists_the_panel_and_sections():
    from pathlib import Path

    import yaml

    manifest = yaml.safe_load((Path(_workspace.__file__).with_name("napari.yaml")).read_text())
    names = [w["display_name"] for w in manifest["contributions"]["widgets"]]
    assert names == ["IBIF napari", "Configuration", "Segmentation (Cellpose)", "Secondary objects",
                     "Measurements", "Filament tracing", "Save workspace"]
