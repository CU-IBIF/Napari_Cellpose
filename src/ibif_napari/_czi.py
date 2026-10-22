"""Zeiss CZI reader (via bioio-czi).

Scenes of equal size become the position axis; scenes of different sizes open as
separate layers.
"""
from __future__ import annotations

from ._bioio import import_reader, read_scene, scene_layers, stem_of


def read_czi(path, reader=None):
    """Read a CZI file. `reader` (a bioio-czi Reader) can be passed for testing."""
    if reader is None:
        reader = import_reader("bioio_czi", "bioio-czi", "Zeiss CZI")(str(path))
    scenes = [read_scene(reader, i) for i in range(len(reader.scenes))]
    reader.set_scene(0)
    emissions = _emissions(reader.metadata, len(scenes[0].channel_names))
    for s in scenes:
        s.emissions = emissions
    return scene_layers(scenes, stem_of(path), str(path), stack=True)


def _emissions(xml, n):
    out = []
    if xml is not None:
        for ch in xml.findall(".//Image/Dimensions/Channels/Channel")[:n]:
            text = ch.findtext("EmissionWavelength")
            try:
                out.append(float(text) if text else None)
            except ValueError:
                out.append(None)
    return out + [None] * (n - len(out))
