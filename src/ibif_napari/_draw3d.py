"""Draw labels in the 3D view.

napari's brush in 3D paints only where the view ray first meets an existing label, so an
empty labels layer (a new region, a new annotation) cannot be drawn on in 3D. The 3D
brush here replaces it for paint and erase, in one of two ways:

* On the image (the default): the brush paints a ball (sized by the brush size, in µm
  where the voxel size is known) where the ray meets the signal you see: the brightest
  point for a max projection, else the first point that reaches half the brightness
  along the ray. Rotate the view to paint around an object. Existing labels in front of
  the image are erased first; painting follows the image, so a stroke stays on the
  object instead of climbing onto what it has just painted.
* On a Z plane: the image layers show a single Z plane (napari's "plane" depiction) and
  the brush paints a disc where the ray crosses that plane. Moving the plane changes the
  depth.

Each viewer gets one brush (`brush_for`); the panels switch its mode.
"""
from __future__ import annotations

import weakref

import numpy as np

SURFACE = "surface"
PLANE = "plane"
OFF = "off"
BRUSH_MODES = {  # panel text -> mode
    "On the image (paint a ball)": SURFACE,
    "On a Z plane": PLANE,
    "napari default (existing labels only)": OFF,
}

_active = weakref.WeakSet()  # brushes that are switched on
_brushes = []  # one per viewer, see brush_for
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
        # 3D brush handles (its `_drag` paints them).
        for brush in list(_active):
            if brush.owns(layer) and layer.mode in ("paint", "erase"):
                return None
        return _original(layer, event)

    bindings.mouse_event_to_labels_coordinate = _coordinate
    return True


def brush_for(viewer, mode=SURFACE):
    """The viewer's 3D brush, created (and switched to `mode`) the first time."""
    for brush in _brushes:
        if brush.viewer is viewer:
            return brush
    brush = DrawingPlane(viewer)
    _brushes.append(brush)
    try:
        brush.set_mode(mode)
    except RuntimeError:
        pass  # e.g. an napari version without the hook: napari's own brush stays
    return brush


class DrawingPlane:
    """The 3D labels brush of one viewer: paints on the image surface or on a Z plane."""

    def __init__(self, viewer):
        self.viewer = viewer
        self.mode = OFF
        self.z = None  # world coordinate of the plane along the Z axis
        self._saved = {}  # image layer -> (depiction, plane position, plane normal)

    @property
    def enabled(self):
        return self.mode != OFF

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

    def set_mode(self, mode, z=None):
        if mode not in (SURFACE, PLANE, OFF):
            raise ValueError(mode)
        if mode == OFF:
            self._switch_off()
            return
        if not _install_hook():
            raise RuntimeError("this napari version does not allow drawing in 3D")
        if mode == PLANE:
            if self.z_range() is None:
                raise RuntimeError("drawing on a plane needs a Z axis")
            if z is None:
                z = self.z if self.mode == PLANE and self.z is not None else self.viewer.dims.point[self.z_axis()]
        if self.mode == PLANE and mode != PLANE:
            self._restore()
        if self.mode == OFF:
            _active.add(self)
            self._hook_layers()
            self.viewer.layers.events.inserted.connect(self._hook_layers)
        self.mode = mode
        if mode == PLANE:
            self.set_z(z)

    def enable(self, z=None):
        """Paint on a Z plane (at world `z`, default: the current Z)."""
        self.set_mode(PLANE, z)

    def enable_surface(self):
        """Paint balls on the image surface."""
        self.set_mode(SURFACE)

    def disable(self):
        self.set_mode(OFF)

    def _switch_off(self):
        self._restore()
        self.mode = OFF
        _active.discard(self)
        try:
            self.viewer.layers.events.inserted.disconnect(self._hook_layers)
        except (TypeError, ValueError):
            pass
        for layer in self.viewer.layers:
            if self._drag in getattr(layer, "mouse_drag_callbacks", []):
                layer.mouse_drag_callbacks.remove(self._drag)

    def _hook_layers(self, event=None):
        from napari.layers import Labels

        for layer in self.viewer.layers:
            if isinstance(layer, Labels) and self._drag not in layer.mouse_drag_callbacks:
                layer.mouse_drag_callbacks.append(self._drag)
        if event is not None and self.mode == PLANE and self.z is not None:
            self._show()  # a newly added image shows the plane too

    def set_z(self, z):
        start, stop, step = self.z_range()
        self.z = float(np.clip(z, start, stop - step if stop - step >= start else start))
        if self.mode == PLANE:
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

    def _restore(self):
        for layer, (depiction, position, normal) in list(self._saved.items()):
            try:
                layer.plane.position, layer.plane.normal = position, normal
                layer.depiction = depiction
            except Exception:
                pass
        self._saved.clear()

    # -- the brush ------------------------------------------------------------------------------

    def _drag(self, layer, event):
        """Paint or erase while the mouse is dragged (one undo step)."""
        if not self.owns(layer) or layer.mode not in ("paint", "erase") or layer.cursor == "circle_frozen":
            return
        if getattr(event, "button", 1) != 1:
            return
        erase = layer.mode == "erase"
        label = getattr(layer.colormap, "background_value", 0) if erase else layer.selected_label
        surface = self.mode == SURFACE
        if surface and (layer.metadata or {}).get("ibif_region") and not layer.metadata.get("ibif_region_3d"):
            # A Cellpose region painted in 3D keeps its shape instead of being copied
            # to every plane of the Z range.
            layer.metadata["ibif_region_3d"] = True
        where = self.surface_coordinate if surface else self.labels_coordinate
        saved_dims = layer.n_edit_dimensions
        try:
            if surface and layer.ndim >= 3:
                layer.n_edit_dimensions = 3  # a ball, sized in µm by napari
            coords = where(layer, event)
            with layer.block_history():
                self._paint(layer, label, coords, coords)
                yield
                last = coords
                while event.type == "mouse_move":
                    coords = where(layer, event)
                    self._paint(layer, label, last if last is not None else coords, coords)
                    last = coords
                    yield
        finally:
            if layer.n_edit_dimensions != saved_dims:
                layer.n_edit_dimensions = saved_dims

    @staticmethod
    def _paint(layer, label, start, end):
        if end is None:
            return
        from napari.layers.labels._labels_utils import interpolate_coordinates

        if start is None:
            start = end
        for c in interpolate_coordinates(start, end, layer.brush_size):
            layer.paint(c, label, refresh=False)
        layer.refresh()

    # -- where the brush paints ---------------------------------------------------------------

    @staticmethod
    def _ray(event):
        position = np.asarray(event.position, float)
        if event.view_direction is None:
            return position, None
        direction = np.asarray(event.view_direction, float)
        if direction.shape != position.shape:  # given for the displayed dims only
            full = np.zeros_like(position)
            full[list(event.dims_displayed)] = direction
            direction = full
        return position, direction

    @staticmethod
    def _to_labels(layer, world):
        """Rounded data coordinate of `layer` at a world point, or None if outside."""
        coords = np.asarray(layer.world_to_data(world), float)[-layer.ndim:]
        shape = layer.data.shape[-layer.ndim:]
        if np.any(coords < -0.5) or np.any(coords > np.asarray(shape) - 0.5):
            return None
        return np.round(coords).astype(int)

    def labels_coordinate(self, layer, event):
        """Data coordinate of `layer` where the mouse ray crosses the plane, or None."""
        position, direction = self._ray(event)
        axis = self.z_axis()
        if direction is None or axis is None or self.z is None:
            return None
        if abs(direction[axis]) < 1e-9:
            return None  # looking along the plane
        t = (self.z - position[axis]) / direction[axis]
        return self._to_labels(layer, position + t * direction)

    def surface_coordinate(self, layer, event):
        """Data coordinate of `layer` where the mouse ray meets the visible image, or None.

        Erasing takes the first label along the ray (what you see in front). Painting takes
        the image's signal, so a stroke stays on the object instead of climbing onto the
        label it has just painted; with no image signal it paints over another label in
        the way (like napari does)."""
        position, direction = self._ray(event)
        if direction is None or not np.any(direction):
            return None
        direction = direction / np.linalg.norm(direction)
        erase = layer.mode == "erase"
        hits = []
        if erase:
            c = labels_hit(layer, position, direction)
            if c is not None:
                return c
        for img in self._image_layers():
            if img.visible and img.opacity > 0:
                w = image_hit(img, position, direction)
                if w is not None:
                    hits.append(self._full_world(position, w))
        if not hits:
            if erase:
                return None
            return labels_hit(layer, position, direction, skip=layer.selected_label)
        nearest = min(hits, key=lambda w: float(np.dot(w - position, direction)))
        return self._to_labels(layer, nearest)

    @staticmethod
    def _full_world(position, world):
        """A layer's world point padded to all world dims (the sliders' position)."""
        full = position.copy()
        world = np.asarray(world, float)
        full[len(full) - len(world):] = world
        return full


def _ray_samples(layer, position, direction):
    """(start, end, s) of the ray through `layer`'s displayed volume, in its data
    coordinates, with sample fractions `s` about half a voxel apart, or None."""
    disp = [int(d) for d in layer._slice_input.displayed]
    if len(disp) != 3:
        return None
    try:
        start, end = layer.get_ray_intersections(
            position=position, view_direction=direction, dims_displayed=disp, world=True
        )
    except Exception:
        return None
    if start is None or end is None:
        return None
    start, end = np.asarray(start, float), np.asarray(end, float)
    n = int(np.ceil(np.linalg.norm(end[disp] - start[disp]) * 2)) + 2
    return start, end, np.linspace(0.0, 1.0, n), disp


def labels_hit(layer, position, direction, skip=None):
    """Data coordinate of the first label along the ray (ignoring label `skip`), or None."""
    ray = _ray_samples(layer, position, direction)
    if ray is None:
        return None
    start, end, s, _ = ray
    shape = np.asarray(layer.data.shape)
    idx = np.clip(np.round(start + (end - start) * s[:, None]).astype(int), 0, shape - 1)
    values = np.asarray(layer.data[tuple(idx.T)]) if not hasattr(layer.data, "vindex") \
        else np.asarray(layer.data.vindex[tuple(idx.T)])
    found = np.flatnonzero((values != 0) & (values != (skip if skip is not None else 0)))
    return idx[found[0]] if len(found) else None


def image_hit(img, position, direction):
    """World point (img's dims) where a view ray meets the signal of an image layer in
    3D, or None. Uses the volume napari shows (the current time point / channel)."""
    ray = _ray_samples(img, position, direction)
    if ray is None:
        return None
    start, end, s, disp = ray
    vol = np.asarray(img._slice.image.raw)
    if vol.ndim != 3:
        return None
    factors = np.ones(3)
    if img.multiscale:
        factors = np.asarray(img.downsample_factors[img.data_level], float)[disp]
    a, b = start[disp] / factors, end[disp] / factors
    idx = np.clip(np.round(a + (b - a) * s[:, None]).astype(int), 0, np.asarray(vol.shape) - 1)
    raw = vol[tuple(idx.T)].astype(float)
    lo, hi = (float(v) for v in img.contrast_limits)
    v = np.clip((raw - lo) / ((hi - lo) or 1.0), 0, 1)
    rendering = str(getattr(img.rendering, "value", img.rendering))
    if rendering == "minip":
        k = int(np.argmin(v))
        if v[k] > 0.95:
            return None
    elif rendering == "iso":
        above = np.flatnonzero(raw >= float(img.iso_threshold))
        if not len(above):
            return None
        k = int(above[0])
    else:
        if v.max() < 0.05:
            return None  # nothing visible along the ray
        if rendering == "mip":
            k = int(np.argmax(v))
        else:  # translucent, attenuated MIP, additive, average: the front of the signal
            k = int(np.flatnonzero(v >= 0.5 * v.max())[0])
    point = start + (end - start) * s[k]
    return np.asarray(img.data_to_world(point), float)
