"""Client for `ibif-napari-server`: runs Cellpose on a facility GPU workstation.

`RemoteProcess` has the same generator API as `_process.CellposeProcess` (`load`, `call`,
`cancel`), so the Cellpose panel drives a local process and a remote server the same
way. Saved servers live in ~/.ibif-napari/servers.json (readable only by the user).
"""
from __future__ import annotations

import io
import json
import os
import ssl
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

import numpy as np

from ._process import Cancelled

CONFIG = Path(os.environ.get("IBIF_NAPARI_HOME", Path.home() / ".ibif-napari")) / "servers.json"


class RemoteError(RuntimeError):
    pass


class RemoteProcess:
    """One job at a time on a remote server. Generators yield None while waiting and a
    string when there is news for the status line."""

    local = False
    poll_interval = 0.3

    def __init__(self, url, token, name=None, cafile=None, timeout=30):
        self.url = url.rstrip("/")
        self.token = token
        self.name = name or self.url
        self.timeout = timeout
        ctx = ssl.create_default_context(cafile=cafile) if cafile else None
        # A facility server is on the local network: never route through a web proxy.
        self._opener = urllib.request.build_opener(
            urllib.request.ProxyHandler({}), urllib.request.HTTPSHandler(context=ctx)
        )
        self._job = None
        self._cancelled = False
        self.info = None  # last /status reply
        self.loaded = set()

    # -- HTTP --------------------------------------------------------------------------

    def _request(self, method, path, body=None, content_type=None, timeout=None):
        req = urllib.request.Request(self.url + path, data=body, method=method)
        req.add_header("Authorization", f"Bearer {self.token}")
        if content_type:
            req.add_header("Content-Type", content_type)
        try:
            with self._opener.open(req, timeout=timeout or self.timeout) as r:
                data = r.read()
                if r.headers.get("Content-Type", "").startswith("application/json"):
                    return json.loads(data)
                return data
        except urllib.error.HTTPError as e:
            try:
                message = json.loads(e.read()).get("error", e.reason)
            except Exception:
                message = e.reason
            if e.code == 401:
                message = "the access token was not accepted; check it under Servers…"
            raise RemoteError(f"{self.name}: {message}") from None
        except (urllib.error.URLError, OSError) as e:
            reason = getattr(e, "reason", e)
            raise RemoteError(f"cannot reach {self.name} ({reason})") from None

    def status(self):
        self.info = self._request("GET", "/status")
        return self.info

    # -- process API -------------------------------------------------------------------

    def load(self, name, gpu=False):
        """Ask the server to load (and if needed download) the model."""
        if name in self.loaded:
            return
        yield f"Connecting to {self.name}…"
        info = self.status()
        if info.get("server") != "ibif-napari-server":
            raise RemoteError(f"{self.name} is not an ibif-napari-server")
        yield from self._run(json.dumps({"op": "load", "model": name}).encode(), "application/json", None)
        self.loaded.add(name)

    def call(self, op, image, nuclei=None, **params):
        if op != "segment":
            raise ValueError(f"unknown request {op!r}")
        params = {k: v for k, v in params.items() if k != "gpu"}  # the server decides
        buf = io.BytesIO()
        arrays = {"image": np.ascontiguousarray(image), "params": np.frombuffer(json.dumps(params).encode(), np.uint8)}
        if nuclei is not None:
            arrays["nuclei"] = np.ascontiguousarray(nuclei)
        np.savez_compressed(buf, **arrays)
        mb = buf.tell() / 1e6
        yield f"Sending {mb:.1f} MB to {self.name}…"
        job_id = yield from self._run(buf.getvalue(), "application/octet-stream", None, keep=True)
        try:
            data = self._request("GET", f"/jobs/{job_id}/result", timeout=max(self.timeout, 120))
        finally:
            self._job = None
        with np.load(io.BytesIO(data), allow_pickle=False) as npz:
            return npz["labels"]

    def _run(self, body, content_type, done_message, keep=False):
        self._cancelled = False
        reply = self._request("POST", "/jobs", body, content_type, timeout=max(self.timeout, 300))
        job_id = self._job = reply["id"]
        try:
            last = None
            while True:
                if self._cancelled:
                    raise Cancelled()
                st = self._request("GET", f"/jobs/{job_id}")
                message = st["message"]
                if st["state"] == "queued" and st.get("position"):
                    ahead = st["position"]
                    message = f"Waiting for {ahead} other job{'s' if ahead > 1 else ''} on {self.name}…"
                if message != last:
                    last = message
                    yield message
                if st["state"] == "done":
                    if done_message:
                        yield done_message
                    return job_id
                if st["state"] == "cancelled":
                    raise Cancelled()
                if st["state"] == "error":
                    raise RemoteError(f"{self.name}: {st.get('error') or message}")
                deadline = time.monotonic() + self.poll_interval
                while time.monotonic() < deadline:
                    if self._cancelled:
                        raise Cancelled()
                    yield None
                    time.sleep(0.05)
        finally:
            if not keep or self._cancelled:
                self._job = None

    def cancel(self):
        """Cancel the current job on the server (in the background, so the UI never waits)."""
        self._cancelled = True
        job_id, self._job = self._job, None
        if job_id:
            threading.Thread(target=self._delete, args=(job_id,), daemon=True).start()

    def _delete(self, job_id):
        try:
            self._request("DELETE", f"/jobs/{job_id}", timeout=10)
        except RemoteError:
            pass

    def close(self):
        self.cancel()

    @property
    def busy(self):
        return self._job is not None


# ---- saved servers ------------------------------------------------------------------


def load_servers(path=None):
    path = Path(path or CONFIG)
    try:
        return json.loads(path.read_text()).get("servers", [])
    except (OSError, ValueError):
        return []


def save_servers(servers, path=None):
    path = Path(path or CONFIG)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"servers": servers}, indent=2))
    try:
        os.chmod(tmp, 0o600)  # holds access tokens
    except OSError:
        pass
    os.replace(tmp, path)
