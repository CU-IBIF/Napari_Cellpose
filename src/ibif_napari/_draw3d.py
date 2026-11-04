"""Draw labels in the 3D view on a Z plane.

napari's brush in 3D paints only where the view ray first meets an existing label, so an
empty labels layer (a new region, a new annotation) cannot be drawn on in 3D. With a
drawing plane on, the brush paints where the ray crosses a Z plane instead, and the
image layers show that plane (napari's "plane" depiction), so you see the slice you draw
on inside the 3D view. Moving the plane changes the drawing depth.
"""
from __future__ import annotations

import weakref

import numpy as np

_active = weakref.WeakSet()  # DrawingPlane objects that are switched on
_original = None


def _install_hook():
    """Route napari's labels brush through `_coordinate` (once per session)."""
    global _original
    if _original is not None:
        return True
    try:
        from napari.layers.labels import _labels_mouse_bindings as bindings
    except ImportError:
        return False
    if not hasattr(bindings, "mouse_event_to_labels_coordinate"):
        return False
    _original = bindings.mouse_event_to_labels_coordinate

    def _coordinate(layer, event):
        # napari's own 3D brush only re-paints existing labels: step aside for layers a
        # drawing plane handles (its `_drag` paints them).
        for plane in list(_active):
            if plane.owns(layer) and layer.mode in ("paint", "erase"):
                return None
        return _original(layer, event)

    bindings.mouse_event_to_labels_coordinate = _coordinate
    return True


class DrawingPlane:
    """A Z plane in world coordinates for one viewer: shown on image layers, drawn on
    by the labels brush in the 3D view."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.enabled = False
        self.z = None  # world coordinate along the Z axis
        self._saved = {}  # image layer -> (depiction, plane position, plane normal)

    # -- which axis is Z ------------------------------------------------------------------

    def z_axis(self):
        """World index of the Z axis: the first displayed axis in 3D (napari shows the
        last three dims, Z Y X), else the third-from-last axis."""
        dims = self.viewer.dims
        if dims.ndim < 3:
            return None
        return dims.order[-3]

    def z_range(self):
        """(start, stop, step) of the Z axis in world coordinates."""
        axis = self.z_axis()
        if axis is None:
            return None
        start, stop, step = self.viewer.dims.range[axis]
        return float(start), float(stop), float(step)

    # -- on / off -------------------------------------------------------------------------

    def enable(self, z=None):
        if not _install_hook():
            raise RuntimeError("this napari version does not allow drawing on a plane in 3D")
        rng = self.z_range()
        if rng is None:
            raise RuntimeError("drawing on a plane needs a Z axis")
        if z is None:
            axis = self.z_axis()
            z = self.viewer.dims.point[axis]
        self.enabled = True
        _active.add(self)
        self._hook_layers()
        self.viewer.layers.events.inserted.connect(self._hook_layers)
        self.set_z(z)

    def _hook_layers(self, event=None):
        from napari.layers import Labels

        for layer in self.viewer.layers:
            if isinstance(layer, Labels) and self._drag not in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.append(self._drag)
        if event is not None and self.enabled and self.z is not None:
            self._show()  # a newly added image shows the plane too

    def disable(self):
        self.enabled = False
        _active.discard(self)
        try:
            self.viewer.layers.events.inserted.disconnect(self._hook_layers)
        except (TypeError, ValueError):
            pass
        for layer in self.viewer.layers:
            if self._drag in getattr(layer, "mouse_drag_callbacks", []):
                layer.mouse_drag_callbacks.remove(self._drag)
        for layer, (depiction, position, normal) in list(self._saved.items()):
            try:
                layer.plane.position, layer.plane.normal = position, normal
                layer.depiction = depiction
            except Exception:
                pass
        self._saved.clear()

    def set_z(self, z):
        start, stop, step = self.z_range()
        self.z = float(np.clip(z, start, stop - step if stop - step >= start else start))
        if self.enabled:
            self._show()

    def owns(self, layer):
        return self.enabled and layer in self.viewer.layers and self.viewer.dims.ndisplay == 3

    # -- showing the plane on image layers --------------------------------------------------

    def _image_layers(self):
        from napari.layers import Image

        return [l for l in self.viewer.layers
                if isinstance(l, Image) and l.ndim >= 3 and not l.rgb and hasattr(l, "depiction")]

    def _show(self):
        axis = self.z_axis()
        world_ndim = self.viewer.dims.ndim
        for layer in self._image_layers():
            if layer not in self._saved:
                self._saved[layer] = (layer.depiction, tuple(layer.plane.position), tuple(layer.plane.normal))
            # The plane's position and normal are in the layer's displayed data dims (Z Y X).
            k = axis - (world_ndim - layer.ndim)  # this layer's index of the Z axis
            if k < 0:
                continue
            z_data = (self.z - layer.translate[k]) / layer.scale[k]
            shape = layer.data.shape[-3:] if not isinstance(layer.data, list) else layer.data[0].shape[-3:]
            layer.plane.normal = (1, 0, 0)
            layer.plane.position = (z_data, shape[1] / 2, shape[2] / 2)
            layer.depiction = "plane"

    # -- the brush ------------------------------------------------------------------------------

    def _drag(self, layer, event):
        """Paint or erase on the plane while the mouse is dragged (one undo step)."""
        if not self.owns(layer) or layer.mode not in ("paint", "erase") or layer.cursor == "circle_frozen":
            return
        if getattr(event, "button", 1) != 1:
            return
        if layer.mode == "erase":
            label = getattr(layer.colormap, "background_value", 0)
        else:
            label = layer.selected_label
        coords = self.labels_coordinate(layer, event)
        with layer.block_history():
            self._paint(layer, label, coords, coords)
            yield
            last = coords
            while event.type == "mouse_move":
                coords = self.labels_coordinate(layer, event)
                self._paint(layer, label, last if last is not None else coords, coords)
                last = coords
                yield

    @staticmethod
    def _paint(layer, label, start, end):
        if end is None:
            return
        from napari.layers.labels._labels_utils import interpolate_coordinates

        for c in interpolate_coordinates(start, end, layer.brush_size):
            layer.paint(c, label, refresh=False)
        layer.refresh()

    # -- where the brush paints ---------------------------------------------------------------

    def labels_coordinate(self, layer, event):
        """Data coordinate of `layer` where the mouse ray crosses the plane, or None."""
        position = np.asarray(event.position, float)
        direction = np.asarray(event.view_direction, float) if event.view_direction is not None else None
        axis = self.z_axis()
        if direction is None or axis is None:
            return None
        if direction.shape != position.shape:  # given for the displayed dims only
            full = np.zeros_like(position)
            full[list(event.dims_displayed)] = direction
            direction = full
        if abs(direction[axis]) < 1e-9:
            return None  # looking along the plane
        t = (self.z - position[axis]) / direction[axis]
        hit = position + t * direction
        coords = np.asarray(layer.world_to_data(hit), float)[-layer.ndim:]
        shape = layer.data.shape[-layer.ndim:]
        if np.any(coords < -0.5) or np.any(coords > np.asarray(shape) - 0.5):
            return None  # outside the layer
        return np.round(coords).astype(int)
