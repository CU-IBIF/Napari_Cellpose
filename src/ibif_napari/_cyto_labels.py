"""Short labels for measurement names, for plots where long names do not fit.

Measurement names look like ``cells: sample_file DAPI mean`` (a joined "All objects"
table: the objects' layer, the image layer measured, the statistic) or
``volume (µm³)``. `short_names` turns a set of them into labels such as
``Cell DAPI mean`` or ``Nuc vol``:

- the text every channel name shares (usually the file name) is dropped;
- the objects' layer is abbreviated (nuclei -> Nuc, cytoplasm -> Cyto, ...), and left out
  when every feature comes from the same layer;
- statistics and sizes are abbreviated (median -> med, std -> sd,
  volume (µm³) -> vol, equivalent diameter (µm) -> diam, centroid X (µm) -> X);
- a channel name that is still too long keeps its first and last words
  (``Alexa Fluor 647`` -> ``Alexa 647``), else its first word, else it is cut with "…".

Labels stay unique; the full name is meant to be shown as a tooltip.
"""
from __future__ import annotations

import re

STATS = {"mean": "mean", "median": "med", "min": "min", "max": "max", "std": "sd", "sum": "sum"}
LAYERS = {"nuclei": "Nuc", "nucleus": "Nuc", "cells": "Cell", "cell": "Cell", "cytoplasm": "Cyto",
          "membrane": "Mem", "membranes": "Mem", "ring": "Ring", "rim": "Rim", "spots": "Spot",
          "vesicles": "Ves", "nucleoli": "Ncl"}
SIZES = [(re.compile(r"^volume \((.+)\)$"), "vol"), (re.compile(r"^area \((.+)\)$"), "area"),
         (re.compile(r"^equivalent diameter \((.+)\)$"), "diam"),
         (re.compile(r"^centroid ([ZYX]) \((.+)\)$"), None), (re.compile(r"^pieces$"), "pcs"),
         (re.compile(r"^voxels$"), "vox")]
ELLIPSIS = "…"


def _parse(name):
    """(layer or None, channel or None, statistic or size label)."""
    layer, rest = None, name
    if ": " in name:
        layer, rest = name.split(": ", 1)
    for pattern, short in SIZES:
        m = pattern.match(rest)
        if m:
            return layer, None, short or m.group(1)
    head, _, stat = rest.rpartition(" ")
    if head and stat in STATS:
        return layer, head, STATS[stat]
    return layer, rest, ""


def _common_words(texts):
    """The words every text starts with (a shared file name), keeping at least one word."""
    split = [t.split() for t in texts]
    if len(split) < 2:
        return 0
    n = 0
    while all(len(w) > n + 1 for w in split) and len({tuple(w[: n + 1]) for w in split}) == 1:
        n += 1
    return n


def _layer_abbreviations(layers):
    out = {}
    for layer in layers:
        key = layer.lower()
        out[layer] = LAYERS.get(key) or (layer[:4].capitalize() if len(layer) > 5 else layer)
    # Different layers that abbreviate alike keep more of their names.
    taken = {}
    for layer, short in out.items():
        taken.setdefault(short, []).append(layer)
    for short, same in taken.items():
        if len(same) > 1:
            for layer in same:
                out[layer] = layer
    return out


def _elide(text, max_len):
    """Cut at the end with "…"."""
    if len(text) <= max_len:
        return text
    return text[: max(max_len - 1, 0)] + ELLIPSIS


def _shorten_channel(channel, room):
    """A channel (dye) name within `room` characters: whole; first and last words
    ("Alexa Fluor 647" -> "Alexa 647"); first word; or cut with "…"."""
    if len(channel) <= room:
        return channel
    words = channel.split()
    if len(words) > 1:
        both = f"{words[0]} {words[-1]}"
        if len(both) <= room:
            return both
        if len(words[0]) <= room - 1:
            return words[0] + ELLIPSIS
    return _elide(channel, room)


def short_names(names, max_len=18):
    """{full name: short label} for a list of measurement names (see the module doc)."""
    names = [str(n) for n in names]
    parsed = {n: _parse(n) for n in names}
    # The shared file name: only from real "<channel> <statistic>" measurements.
    channels = sorted({c for _, c, stat in parsed.values() if c and stat})
    drop = _common_words(channels)
    layers = sorted({layer for layer, _, _ in parsed.values() if layer})
    abbrev = _layer_abbreviations(layers) if len(layers) > 1 else {layer: "" for layer in layers}
    out = {}
    for name, (layer, channel, stat) in parsed.items():
        if channel and stat and drop:
            channel = " ".join(channel.split()[drop:])
        prefix = abbrev.get(layer, "") if layer else ""
        fixed = " ".join(p for p in (prefix, stat) if p)
        if channel:
            room = max_len - len(fixed) - (1 if fixed else 0)  # one more space joins the channel
            channel = _shorten_channel(channel, max(room, 4)) if stat else _elide(channel, max(room, 4))
        out[name] = " ".join(p for p in (prefix, channel, stat) if p) or name
    # Unique labels: where two collide, fall back to the full name, cut to fit.
    seen = {}
    for name, label in out.items():
        seen.setdefault(label, []).append(name)
    for label, same in seen.items():
        if len(same) > 1:
            for name in same:
                out[name] = _elide(name, max(max_len, 24))
    return out


__all__ = ["short_names", "ELLIPSIS"]
