"""The user guide: installed with the package, complete, and opened from napari's Help menu."""
import re
from pathlib import Path

import pytest

from ibif_napari import _guide

ROOT = Path(__file__).resolve().parents[1]


def test_guide_is_installed_with_its_images():
    text = _guide.GUIDE.read_text(encoding="utf-8")
    images = re.findall(r"\]\((images/[^)]+)\)", text)
    assert images and all((_guide.GUIDE.parent / i).is_file() for i in images)
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"manual/*.md", "manual/images/*.png"' in pyproject  # in the package data
    # Links to other docs are absolute, so they work inside napari too.
    assert not re.search(r"\]\(\.\./", text)


def test_contents_links_match_headings():
    text = _guide.GUIDE.read_text(encoding="utf-8")
    anchors = set(re.findall(r"\]\(#([^)]+)\)", text))
    headings = {_guide.slug(h) for h in re.findall(r"^#{1,6} (.+)$", text, flags=re.M)}
    assert anchors and anchors <= headings
    assert _guide.slug("7. Segmentation with Cellpose") == "7-segmentation-with-cellpose"
    assert "```mermaid" not in _guide.guide_text()


def test_guide_window_and_help_menu(qtbot):
    pytest.importorskip("napari")
    window = _guide.GuideWindow()
    qtbot.addWidget(window)
    window.resize(800, 600)
    window.show()
    qtbot.waitExposed(window)
    window.fit_images()
    assert window.go_to("7-segmentation-with-cellpose")
    assert window.browser.verticalScrollBar().value() > 0
    assert not window.go_to("no-such-heading")
    window.find_box.setText("anisotropy")
    assert window.find_next()

    from napari._app_model import get_app_model

    assert _guide.install_help_menu()
    app = get_app_model()
    assert _guide.ACTION_ID in app.commands
    items = [getattr(i, "command", None) for i in app.menus.get_menu("napari/help")]
    assert any(c is not None and c.id == _guide.ACTION_ID for c in items)
