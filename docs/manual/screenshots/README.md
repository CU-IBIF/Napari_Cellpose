# Screenshots for the user guide

These scripts make the pictures in `src/ibif_napari/manual/images/` by driving the real IBIF napari GUI on
synthetic data:

- `data.py`: a synthetic 4-channel 3D sample (DAPI, Membrane, CD3, CD8; two time points),
  made with the V&V scene generator, and a synthetic neuron for filament tracing.
- `demo_seg.py`: a classical distance-transform watershed that stands in for the Cellpose
  model, so the screenshots can be made without downloading the Cellpose weights. Remove
  the stand-in lines at the top of `shots_body.py` to use real Cellpose instead.
- `shots_body.py`: loads the data the way the ND2 reader does, clicks through every section
  and saves one PNG per step.
- `run_shots.py`: the entry point (guarded, because Cellpose runs in a spawned process).

Run them in an environment with IBIF napari installed (`pip install -e ".[dev,vv]"`). On a
machine without a display, use Xvfb:

```bash
cd docs/manual/screenshots
xvfb-run -a -s "-screen 0 1920x1200x24" python run_shots.py ../../../src/ibif_napari/manual/images
```

It takes a few minutes on a CPU. Check the pictures before committing them.
