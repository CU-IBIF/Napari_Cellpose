"""Run Cellpose in a separate process that keeps the model loaded.

A Python thread cannot be interrupted while Cellpose (PyTorch) is computing, so cancelling
a thread only takes effect when the segmentation finishes. A process can be stopped at
once, which frees the CPU/GPU immediately; a crash or out-of-memory in Cellpose also
leaves napari running.
"""
from __future__ import annotations

import importlib
import logging
import multiprocessing as mp
import threading
import time
import traceback

BACKEND = "ibif_napari._cellpose"  # module with get_model(name, gpu) and segment(...)


class Cancelled(Exception):
    pass


def _serve(conn, backend):
    """Child process: answer ("load", args) and ("segment", args) requests until EOF."""
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(asctime)s %(name)s: %(message)s", "%H:%M:%S"))
    logging.getLogger("cellpose").addHandler(handler)
    logging.getLogger("cellpose").setLevel(logging.INFO)
    module = importlib.import_module(backend)
    while True:
        try:
            op, args, kwargs = conn.recv()
        except (EOFError, OSError):
            return
        try:
            if op == "load":
                module.get_model(*args, **kwargs)
                result = None
            elif op == "segment":
                result = module.segment(*args, **kwargs)
            else:
                raise ValueError(f"unknown request {op!r}")
            conn.send(("ok", result))
        except Exception as e:
            conn.send(("error", f"{type(e).__name__}: {e}\n\n{traceback.format_exc()}"))


class CellposeProcess:
    """One child process, one request at a time. `call` is a generator: it yields None
    while waiting (so a napari worker can be cancelled between polls) and returns the
    result."""

    poll_interval = 0.1

    def __init__(self, backend=BACKEND):
        self.backend = backend
        self._proc = None
        self._conn = None
        # The request in progress. A token rather than a lock: a cancelled request's
        # generator may be abandoned without ever reaching its `finally`.
        self._mutex = threading.Lock()
        self._owner = None
        self._cancelled = False
        self.loaded = set()  # (model, gpu) loaded in the current child

    def _start(self):
        if self._proc is not None and self._proc.is_alive():
            return
        ctx = mp.get_context("spawn")  # fork is unsafe with Qt/PyTorch threads
        parent, child = ctx.Pipe()
        self._proc = ctx.Process(target=_serve, args=(child, self.backend), daemon=True)
        self._proc.start()
        child.close()
        self._conn = parent
        self.loaded = set()

    def call(self, op, *args, **kwargs):
        token = object()
        waited = False
        while True:
            with self._mutex:
                if self._owner is None:
                    self._owner = token
                    break
            # Another request (e.g. the model preload) is still running.
            yield None if waited else "Waiting for the model to finish loading…"
            waited = True
        try:
            self._cancelled = False
            self._start()
            proc, conn = self._proc, self._conn
            conn.send((op, args, kwargs))
            while True:
                try:
                    ready = conn.poll(self.poll_interval)
                except (EOFError, OSError):
                    ready = True
                if ready:
                    try:
                        status, value = conn.recv()
                        break
                    except (EOFError, OSError):
                        status, value = "died", None
                        break
                if not proc.is_alive() or self._conn is not conn:
                    status, value = "died", None
                    break
                yield None
            if status == "died":
                self._reset(conn)
                if self._cancelled:
                    raise Cancelled()
                raise RuntimeError(
                    f"the Cellpose process stopped unexpectedly (exit code {proc.exitcode}); "
                    "it may have run out of memory, try a smaller region"
                )
            if status == "error":
                raise RuntimeError(value)
            if op == "load":
                self.loaded.add((args[0], bool(args[1]) if len(args) > 1 else False))
            return value
        finally:
            with self._mutex:
                if self._owner is token:
                    self._owner = None

    def load(self, name, gpu=False):
        if (name, bool(gpu)) not in self.loaded:
            yield f"Loading Cellpose model '{name}' ({'GPU' if gpu else 'CPU'})…"
            yield from self.call("load", name, gpu)

    def cancel(self):
        """Stop the running request immediately (the model must be loaded again)."""
        self._cancelled = True
        proc = self._proc
        if proc is not None and proc.is_alive():
            proc.terminate()
            proc.join(2)
            if proc.is_alive():
                proc.kill()
                proc.join(2)
        self._reset()
        with self._mutex:
            self._owner = None

    def _reset(self, conn=None):
        """Forget the child (only if it is still `conn`'s child, when given)."""
        with self._mutex:
            if conn is not None and self._conn is not conn:
                return
            old, self._conn, self._proc = self._conn, None, None
            self.loaded = set()
        if old is not None:
            try:
                old.close()
            except OSError:
                pass

    def close(self):
        self.cancel()

    @property
    def busy(self):
        return self._owner is not None


_process = None


def get_process():
    global _process
    if _process is None:
        _process = CellposeProcess()
    return _process


def wait(gen, timeout=None):
    """Run a `call`/`load` generator to completion in the current thread (for scripts)."""
    start = time.monotonic()
    try:
        while True:
            next(gen)
            if timeout is not None and time.monotonic() - start > timeout:
                raise TimeoutError
    except StopIteration as stop:
        return stop.value
