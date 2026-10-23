"""`ibif-napari-server`: run Cellpose on this computer's GPU for IBIF napari users.

Install it on a facility GPU workstation; napari users pick it under "Run on" in the
Cellpose panel. Jobs from all users go through one queue and run one at a time in a
Cellpose process that keeps the model loaded (so Cancel stops a run at once).

    ibif-napari-server add-user alice         # prints alice's access token (once)
    ibif-napari-server run --allow 10.0.0.0/8 # serve on port 8765

Protocol (HTTP, every request carries "Authorization: Bearer <token>"):
    GET    /status            server, Cellpose and GPU versions, queue length
    POST   /jobs              JSON {"op": "load", "model": ...} or an .npz upload with
                              "image", optional "nuclei" and "params" (JSON as uint8)
    GET    /jobs/<id>         {"state", "message", "position", "error"}
    GET    /jobs/<id>/result  .npz with "labels"
    DELETE /jobs/<id>         cancel (queued or running)
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import hmac
import io
import ipaddress
import json
import logging
import os
import secrets
import ssl
import sys
import threading
import time
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

from . import __version__
from ._process import BACKEND, Cancelled, CellposeProcess

log = logging.getLogger("ibif_napari.server")

DEFAULT_PORT = 8765
CONFIG_DIR = Path(os.environ.get("IBIF_NAPARI_SERVER_HOME", Path.home() / ".ibif-napari-server"))

# Cellpose settings a client may send, with their types; anything else is refused.
PARAMS = {
    "model": str,
    "do_3D": bool,
    "anisotropy": (int, float, type(None)),
    "diameter": (int, float, type(None)),
    "flow_threshold": (int, float),
    "cellprob_threshold": (int, float),
    "stitch_threshold": (int, float),
    "min_size": int,
}
FINISHED = {"done", "error", "cancelled"}


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status


# ---- users and tokens --------------------------------------------------------------


def _hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Users:
    """User names and SHA-256 hashes of their tokens, in a JSON file (tokens are not stored)."""

    def __init__(self, path):
        self.path = Path(path)
        self.hashes = {}
        if self.path.exists():
            self.hashes = json.loads(self.path.read_text()).get("users", {})

    def save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"users": self.hashes}, indent=2))
        try:
            os.chmod(tmp, 0o600)
        except OSError:
            pass
        os.replace(tmp, self.path)

    def add(self, name):
        token = secrets.token_urlsafe(32)
        self.hashes[name] = _hash(token)
        self.save()
        return token

    def remove(self, name):
        if self.hashes.pop(name, None) is None:
            raise KeyError(name)
        self.save()

    def check(self, token):
        """The user whose token this is, or None (constant-time comparison)."""
        if not token:
            return None
        h = _hash(token)
        found = None
        for name, stored in self.hashes.items():
            if hmac.compare_digest(h, stored):
                found = name
        return found


# ---- jobs ----------------------------------------------------------------------------


class Job:
    def __init__(self, user, op, model, image=None, nuclei=None, params=None):
        self.id = uuid.uuid4().hex
        self.user = user
        self.op = op
        self.model = model
        self.image = image
        self.nuclei = nuclei
        self.params = params or {}
        self.state = "queued"
        self.message = "Queued"
        self.error = None
        self.result = None  # encoded .npz bytes
        self.finished_at = None
        self.cancel_requested = False

    def finish(self, state, message, error=None):
        self.state, self.message, self.error = state, message, error
        self.image = self.nuclei = None
        self.finished_at = time.monotonic()


class CellposeServer:
    """The job queue and the Cellpose process; the HTTP layer is `Handler`."""

    def __init__(
        self,
        users,
        backend=BACKEND,
        gpu=None,
        models_dir=None,
        model_names=None,
        prepare=None,
        max_mb=2048,
        result_ttl_s=3600,
    ):
        from . import _cellpose

        self.users = users
        self.process = CellposeProcess(backend=backend)
        self.gpu = _cellpose.gpu_available() if gpu is None else gpu
        self.models_dir = Path(models_dir) if models_dir else None
        self._model_names = model_names or _cellpose.builtin_models
        self._prepare = prepare or _cellpose.ensure_weights
        self.max_bytes = int(max_mb * 1024 * 1024)
        self.result_ttl_s = result_ttl_s
        self.jobs = {}
        self.queue = collections.deque()
        self.current = None
        self.last_model = None
        self._cond = threading.Condition()
        self._stop = False
        self._worker = threading.Thread(target=self._work, name="cellpose-jobs", daemon=True)
        self._worker.start()

    # -- model names ------------------------------------------------------------------

    def models(self):
        names = list(self._model_names())
        if self.models_dir and self.models_dir.is_dir():
            names += sorted(p.name for p in self.models_dir.iterdir() if p.is_file())
        return names

    def resolve_model(self, name):
        """A built-in model name, or a file in --models-dir. Paths are refused: a model
        file is loaded with torch, which must only ever read files the admin put there."""
        if not isinstance(name, str) or not name:
            raise RequestError(HTTPStatus.BAD_REQUEST, "missing model name")
        if name in self._model_names():
            return name
        if self.models_dir and "/" not in name and "\\" not in name and name not in (".", ".."):
            path = self.models_dir / name
            if path.is_file():
                return str(path)
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            f"model {name!r} is not available on this server (available: {', '.join(self.models())})",
        )

    # -- queue ---------------------------------------------------------------------------

    def submit(self, job):
        with self._cond:
            self._expire()
            self.jobs[job.id] = job
            self.queue.append(job)
            self._cond.notify_all()
        log.info("%s queued %s job %s (model %s)", job.user, job.op, job.id[:8], job.model)
        return job

    def get(self, job_id, user):
        job = self.jobs.get(job_id)
        if job is None or job.user != user:
            raise RequestError(HTTPStatus.NOT_FOUND, "no such job")
        return job

    def position(self, job):
        with self._cond:
            if job in self.queue:
                return list(self.queue).index(job) + (1 if self.current else 0)
        return 0

    def cancel(self, job):
        with self._cond:
            if job in self.queue:
                self.queue.remove(job)
                job.finish("cancelled", "Cancelled")
                return
            running = job is self.current and job.state not in FINISHED
            if running:
                job.cancel_requested = True
        if running:
            self.process.cancel()  # stops Cellpose at once; the worker marks the job

    def _expire(self):
        now = time.monotonic()
        for job_id, job in list(self.jobs.items()):
            if job.finished_at is not None and now - job.finished_at > self.result_ttl_s:
                del self.jobs[job_id]

    def _work(self):
        while True:
            with self._cond:
                while not self.queue and not self._stop:
                    self._cond.wait()
                if self._stop:
                    return
                job = self.queue.popleft()
                self.current = job
            try:
                self._run(job)
            except Cancelled:
                job.finish("cancelled", "Cancelled")
                log.info("%s cancelled job %s", job.user, job.id[:8])
                self._warm_up()
            except Exception as e:  # reported to the client
                summary = str(e).split("\n\n")[0]
                job.finish("error", f"Failed: {summary}", error=str(e))
                log.warning("job %s failed: %s", job.id[:8], e)
            finally:
                with self._cond:
                    self.current = None

    def _drive(self, job, gen):
        """Run a process generator, passing its messages to the job; returns its result."""
        try:
            while True:
                if job.cancel_requested:
                    raise Cancelled()
                message = next(gen)
                if message:
                    job.message = message
        except StopIteration as stop:
            return stop.value

    def _run(self, job):
        job.state = "loading"
        self._drive(job, self._prepare(job.model))
        self._drive(job, self.process.load(job.model, self.gpu))
        self.last_model = job.model
        if job.op == "load":
            job.finish("done", f"Model ready on {self.device_name()}")
            return
        job.state = "running"
        shape = "×".join(map(str, job.image.shape))
        job.message = f"Segmenting {shape} on {self.device_name()}"
        params = {**job.params, "model": job.model, "gpu": self.gpu}
        start = time.monotonic()
        labels = self._drive(job, self.process.call("segment", job.image, job.nuclei, **params))
        buf = io.BytesIO()
        np.savez_compressed(buf, labels=np.asarray(labels))
        job.result = buf.getvalue()
        job.finish("done", f"Done in {time.monotonic() - start:.1f} s on {self.device_name()}")
        log.info("%s job %s done (%s)", job.user, job.id[:8], job.message)

    def _warm_up(self):
        """After a cancel the Cellpose process restarts: load the last model again so the
        next job does not wait for it. The load always runs to the end: abandoning a
        request mid-way would leave its reply in the pipe for the next one."""
        if self.last_model:
            try:
                for _ in self.process.load(self.last_model, self.gpu):
                    pass
            except Exception as e:
                log.warning("could not reload model %s: %s", self.last_model, e)

    def close(self):
        with self._cond:
            self._stop = True
            self._cond.notify_all()
        self.process.close()

    # -- status --------------------------------------------------------------------------

    def device_name(self):
        if not hasattr(self, "_device"):
            self._device = "CPU"
            if self.gpu:
                try:
                    import torch

                    if torch.cuda.is_available():
                        self._device = f"GPU ({torch.cuda.get_device_name(0)})"
                    elif torch.backends.mps.is_available():
                        self._device = "GPU (Apple)"
                except Exception:
                    self._device = "GPU"
        return self._device

    def status(self):
        from importlib.metadata import PackageNotFoundError, version

        def v(pkg):
            try:
                return version(pkg)
            except PackageNotFoundError:
                return None

        return {
            "server": "ibif-napari-server",
            "version": __version__,
            "cellpose": v("cellpose"),
            "torch": v("torch"),
            "device": self.device_name(),
            "models": self.models(),
            "queued": len(self.queue),
            "busy": self.current is not None,
        }


# ---- HTTP --------------------------------------------------------------------------


def _decode_upload(body, max_bytes):
    try:
        with np.load(io.BytesIO(body), allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
    except Exception as e:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"could not read the upload: {e}") from None
    if "image" not in arrays or "params" not in arrays:
        raise RequestError(HTTPStatus.BAD_REQUEST, "the upload needs 'image' and 'params'")
    try:
        params = json.loads(arrays["params"].tobytes().decode())
    except Exception:
        raise RequestError(HTTPStatus.BAD_REQUEST, "'params' is not JSON") from None
    image, nuclei = arrays["image"], arrays.get("nuclei")
    for name, a in (("image", image), ("nuclei", nuclei)):
        if a is None:
            continue
        if a.ndim not in (2, 3) or a.dtype.kind not in "uif":
            raise RequestError(HTTPStatus.BAD_REQUEST, f"{name} must be a 2D or 3D numeric array")
        if a.nbytes > max_bytes:
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, f"{name} is too large")
    if nuclei is not None and nuclei.shape != image.shape:
        raise RequestError(HTTPStatus.BAD_REQUEST, "nuclei and image differ in shape")
    return image, nuclei, _check_params(params)


def _check_params(params):
    if not isinstance(params, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "params must be an object")
    out = {}
    for key, value in params.items():
        if key == "gpu":
            continue  # the server decides
        kind = PARAMS.get(key)
        if kind is None:
            raise RequestError(HTTPStatus.BAD_REQUEST, f"unknown setting {key!r}")
        if not isinstance(value, kind) or (kind is int and isinstance(value, bool)):
            raise RequestError(HTTPStatus.BAD_REQUEST, f"bad value for {key!r}")
        out[key] = value
    return out


class Handler(BaseHTTPRequestHandler):
    server_version = f"ibif-napari-server/{__version__}"
    protocol_version = "HTTP/1.1"

    @property
    def app(self) -> CellposeServer:
        return self.server.app

    def log_message(self, fmt, *args):  # route http.server's access log to logging
        log.debug("%s %s", self.address_string(), fmt % args)

    def _send(self, status, body=b"", content_type="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def _handle(self, method):
        try:
            allowed = self.server.allowed
            if allowed and not any(ipaddress.ip_address(self.client_address[0]) in n for n in allowed):
                raise RequestError(HTTPStatus.FORBIDDEN, "this address is not allowed")
            auth = self.headers.get("Authorization", "")
            user = self.app.users.check(auth[7:] if auth.startswith("Bearer ") else "")
            if user is None:
                raise RequestError(HTTPStatus.UNAUTHORIZED, "missing or unknown access token")
            parts = [p for p in self.path.split("?")[0].split("/") if p]
            getattr(self, f"_{method}")(user, parts)
        except RequestError as e:
            self.close_connection = True  # an unread upload may still be on the wire
            self._send(e.status, {"error": str(e)})
            if e.status in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
                log.warning("%s: %s", self.client_address[0], e)
        except Exception as e:
            log.exception("request failed")
            self._send(HTTPStatus.INTERNAL_SERVER_ERROR, {"error": f"server error: {e}"})

    def do_GET(self):
        self._handle("get")

    def do_POST(self):
        self._handle("post")

    def do_DELETE(self):
        self._handle("delete")

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > self.app.max_bytes:
            self.close_connection = True
            raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "upload too large")
        return self.rfile.read(length)

    def _get(self, user, parts):
        if parts == ["status"]:
            return self._send(HTTPStatus.OK, {**self.app.status(), "user": user})
        if len(parts) == 2 and parts[0] == "jobs":
            job = self.app.get(parts[1], user)
            return self._send(
                HTTPStatus.OK,
                {
                    "state": job.state,
                    "message": job.message,
                    "position": self.app.position(job),
                    "error": job.error,
                },
            )
        if len(parts) == 3 and parts[0] == "jobs" and parts[2] == "result":
            job = self.app.get(parts[1], user)
            if job.state != "done" or job.result is None:
                raise RequestError(HTTPStatus.CONFLICT, f"job is {job.state}")
            data = job.result
            del self.app.jobs[job.id]  # fetched: free the memory
            return self._send(HTTPStatus.OK, data, "application/octet-stream")
        raise RequestError(HTTPStatus.NOT_FOUND, "not found")

    def _post(self, user, parts):
        if parts != ["jobs"]:
            raise RequestError(HTTPStatus.NOT_FOUND, "not found")
        body = self._read_body()
        if self.headers.get("Content-Type", "").startswith("application/json"):
            try:
                req = json.loads(body)
            except ValueError:
                raise RequestError(HTTPStatus.BAD_REQUEST, "body is not JSON") from None
            if req.get("op") != "load":
                raise RequestError(HTTPStatus.BAD_REQUEST, "unknown op")
            job = Job(user, "load", self.app.resolve_model(req.get("model")))
        else:
            image, nuclei, params = _decode_upload(body, self.app.max_bytes)
            model = self.app.resolve_model(params.pop("model", None))
            job = Job(user, "segment", model, image, nuclei, params)
        self.app.submit(job)
        self._send(HTTPStatus.ACCEPTED, {"id": job.id, "position": self.app.position(job)})

    def _delete(self, user, parts):
        if len(parts) == 2 and parts[0] == "jobs":
            self.app.cancel(self.app.get(parts[1], user))
            return self._send(HTTPStatus.OK, {"ok": True})
        raise RequestError(HTTPStatus.NOT_FOUND, "not found")


def make_http_server(app, host="0.0.0.0", port=DEFAULT_PORT, allow=(), certfile=None, keyfile=None):
    httpd = ThreadingHTTPServer((host, port), Handler)
    httpd.daemon_threads = True
    httpd.app = app
    httpd.allowed = [ipaddress.ip_network(a, strict=False) for a in allow]
    httpd.tls = bool(certfile)
    if certfile:
        ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ctx.load_cert_chain(certfile, keyfile)
        # Handshake in the request thread, so one slow client cannot stall the others.
        httpd.socket = ctx.wrap_socket(httpd.socket, server_side=True, do_handshake_on_connect=False)
    return httpd


# ---- command line --------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="ibif-napari-server", description=__doc__.splitlines()[0],
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-dir", default=str(CONFIG_DIR), help=f"users file location (default {CONFIG_DIR})")
    sub = parser.add_subparsers(dest="command", required=True)
    run = sub.add_parser("run", help="serve Cellpose to napari users")
    run.add_argument("--host", default="0.0.0.0", help="address to listen on (default all)")
    run.add_argument("--port", type=int, default=DEFAULT_PORT)
    run.add_argument("--allow", action="append", default=[], metavar="CIDR",
                     help="only accept these networks, e.g. --allow 10.0.0.0/8 (repeatable)")
    run.add_argument("--certfile", help="TLS certificate (PEM) to serve HTTPS")
    run.add_argument("--keyfile", help="TLS private key (PEM)")
    run.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    run.add_argument("--models-dir", help="folder of custom Cellpose models users may choose by file name")
    run.add_argument("--preload", default="", help="model to load at start (default: the newest built-in)")
    run.add_argument("--max-mb", type=float, default=2048, help="largest upload in MB (default 2048)")
    run.add_argument("--log-file", help="also write the log here (useful when run as a service)")
    run.add_argument("-v", "--verbose", action="store_true")
    add = sub.add_parser("add-user", help="add a user (or replace their token) and print the token")
    add.add_argument("name")
    rm = sub.add_parser("remove-user", help="remove a user")
    rm.add_argument("name")
    sub.add_parser("list-users", help="list users")
    args = parser.parse_args(argv)

    users = Users(Path(args.config_dir) / "users.json")
    if args.command == "add-user":
        token = users.add(args.name)
        print(f"Access token for {args.name} (shown once; give it to them privately):\n\n  {token}\n")
        print("If the server is running it picks up the change within a minute; or restart it.")
        return 0
    if args.command == "remove-user":
        try:
            users.remove(args.name)
        except KeyError:
            print(f"No user {args.name!r}.", file=sys.stderr)
            return 1
        print(f"Removed {args.name}.")
        return 0
    if args.command == "list-users":
        print("\n".join(sorted(users.hashes)) or "(no users)")
        return 0

    handlers = [logging.StreamHandler()]
    if args.log_file:
        handlers.append(logging.FileHandler(args.log_file, encoding="utf-8"))
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(name)s: %(message)s",
        handlers=handlers,
    )
    logging.getLogger("cellpose").setLevel(logging.INFO)
    if not users.hashes:
        print("No users yet: add one with  ibif-napari-server add-user <name>", file=sys.stderr)
        return 1
    if args.keyfile and not args.certfile:
        parser.error("--keyfile needs --certfile")
    gpu = {"auto": None, "gpu": True, "cpu": False}[args.device]
    app = CellposeServer(users, gpu=gpu, models_dir=args.models_dir, max_mb=args.max_mb)
    threading.Thread(target=_reload_users, args=(app,), daemon=True).start()
    httpd = make_http_server(app, args.host, args.port, args.allow, args.certfile, args.keyfile)
    scheme = "https" if args.certfile else "http"
    log.info("ibif-napari-server %s on %s://%s:%s, %s, %d users%s",
             __version__, scheme, args.host, args.port, app.device_name(), len(users.hashes),
             f", allowing {', '.join(args.allow)}" if args.allow else "")
    if not args.certfile:
        log.warning("serving plain HTTP: use only on a trusted network, or add --certfile/--keyfile")
    from . import _cellpose

    model = args.preload or _cellpose.default_model()
    if model:
        app.submit(Job("(server)", "load", app.resolve_model(model)))
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
        app.close()
    return 0


def _reload_users(app, interval=60):
    """Pick up add-user / remove-user changes without a restart."""
    path = app.users.path
    last = path.stat().st_mtime if path.exists() else None
    while True:
        time.sleep(interval)
        try:
            mtime = path.stat().st_mtime
            if mtime != last:
                app.users.hashes = Users(path).hashes
                last = mtime
                log.info("reloaded users (%d)", len(app.users.hashes))
        except OSError:
            pass


if __name__ == "__main__":
    sys.exit(main())
