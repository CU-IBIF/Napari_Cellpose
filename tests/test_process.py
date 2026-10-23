import time

import numpy as np
import pytest

from ibif_napari._process import Cancelled, CellposeProcess, wait


@pytest.fixture
def proc():
    p = CellposeProcess(backend="_fake_cellpose")
    yield p
    p.close()


def test_load_and_segment_in_child(proc):
    messages = list(m for m in proc.load("cpsam") if m)
    assert messages == ["Loading Cellpose model 'cpsam' (CPU)…"]
    assert list(m for m in proc.load("cpsam") if m) == []  # already loaded
    masks = wait(proc.call("segment", np.zeros((3, 8, 8)), model="x"), timeout=60)
    assert masks.shape == (3, 8, 8) and masks.max() == 2


def test_errors_are_reported(proc):
    with pytest.raises(RuntimeError, match="no such model"):
        wait(proc.load("missing"), timeout=60)


def test_cancel_stops_the_computation_at_once(proc):
    wait(proc.load("cpsam"), timeout=60)
    gen = proc.call("segment", np.zeros((8, 8)), model="slow")
    t = time.monotonic()
    while time.monotonic() - t < 1:  # let it start
        next(gen)
    child = proc._proc
    t = time.monotonic()
    proc.cancel()
    with pytest.raises(Cancelled):
        wait(gen, timeout=10)
    assert time.monotonic() - t < 5
    assert not child.is_alive()
    assert not proc.busy and proc.loaded == set()
    # A fresh process serves the next request.
    assert wait(proc.call("segment", np.zeros((4, 4)), model="x"), timeout=60).shape == (4, 4)


def test_abandoned_request_does_not_block_the_next(proc):
    gen = proc.call("segment", np.zeros((8, 8)), model="slow")
    next(gen)
    proc.cancel()
    del gen  # e.g. a napari worker that was aborted mid-request
    assert wait(proc.call("segment", np.zeros((4, 4)), model="x"), timeout=60).shape == (4, 4)


def test_crash_is_an_error_not_a_hang(proc):
    with pytest.raises(RuntimeError, match="stopped unexpectedly"):
        wait(proc.call("segment", np.zeros((4, 4)), model="crash"), timeout=60)
    assert wait(proc.call("segment", np.zeros((4, 4)), model="x"), timeout=60).shape == (4, 4)


def test_closing_a_request_midway_does_not_mix_up_replies(proc):
    wait(proc.load("cpsam"), timeout=60)
    gen = proc.call("segment", np.zeros((8, 8)), model="slow")
    next(gen)
    gen.close()  # abandoned without cancel()
    out = wait(proc.call("segment", np.zeros((5, 5)), model="x"), timeout=60)
    assert out.shape == (5, 5)
