import json
import threading
import time
import urllib.request

import numpy as np
import pytest

from ibif_napari._process import Cancelled, wait
from ibif_napari._remote import RemoteError, RemoteProcess, load_servers, save_servers
from ibif_napari._server import CellposeServer, Users, main, make_http_server


@pytest.fixture
def server(tmp_path):
    users = Users(tmp_path / "users.json")
    tokens = {"alice": users.add("alice"), "bob": users.add("bob")}
    app = CellposeServer(
        users,
        backend="_fake_cellpose",
        gpu=False,
        model_names=lambda: ["cpsam_v2", "slow", "crash"],
        prepare=lambda name: iter(()),
    )
    httpd = make_http_server(app, "127.0.0.1", 0)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    url = f"http://127.0.0.1:{httpd.server_address[1]}"
    yield app, url, tokens, httpd
    httpd.shutdown()
    httpd.server_close()
    app.close()


def client(url, token, **kw):
    return RemoteProcess(url, token, name="facility-gpu", **kw)


def test_segment_round_trip(server):
    app, url, tokens, _ = server
    c = client(url, tokens["alice"])
    messages = [m for m in c.load("cpsam_v2") if m]
    assert messages[0] == "Connecting to facility-gpu…"
    assert messages[-1] == "Model ready on CPU"
    assert c.info["device"] == "CPU" and c.info["user"] == "alice"
    img = np.random.rand(3, 16, 16).astype(np.float32)
    gen = c.call("segment", img, img, model="cpsam_v2", gpu=True, do_3D=True, anisotropy=2.0)
    labels = wait(gen, timeout=60)
    assert labels.shape == (3, 16, 16) and labels.max() == 2
    assert not c.busy
    assert app.jobs == {} or all(j.state != "done" or j.result is None for j in app.jobs.values())


def test_auth_and_ownership(server):
    _, url, tokens, _ = server
    with pytest.raises(RemoteError, match="access token was not accepted"):
        client(url, "wrong").status()
    alice, bob = client(url, tokens["alice"]), client(url, tokens["bob"])
    reply = alice._request("POST", "/jobs", json.dumps({"op": "load", "model": "cpsam_v2"}).encode(), "application/json")
    with pytest.raises(RemoteError, match="no such job"):
        bob._request("GET", f"/jobs/{reply['id']}")
    assert alice._request("GET", f"/jobs/{reply['id']}")["state"] in {"queued", "loading", "done"}


def test_bad_requests_are_refused(server):
    _, url, tokens, _ = server
    c = client(url, tokens["alice"])
    with pytest.raises(RemoteError, match="not available on this server"):
        wait(c.call("segment", np.zeros((8, 8)), model="../../etc/passwd"))
    with pytest.raises(RemoteError, match="unknown setting"):
        wait(c.call("segment", np.zeros((8, 8)), model="cpsam_v2", evil=1))
    with pytest.raises(RemoteError, match="bad value"):
        wait(c.call("segment", np.zeros((8, 8)), model="cpsam_v2", min_size="15"))
    with pytest.raises(RemoteError, match="2D or 3D"):
        wait(c.call("segment", np.zeros((2, 2, 2, 2)), model="cpsam_v2"))


def test_cancel_stops_the_running_job_at_once(server):
    app, url, tokens, _ = server
    c = client(url, tokens["alice"])
    gen = c.call("segment", np.zeros((8, 8)), model="slow")
    t = time.monotonic()
    while time.monotonic() - t < 30:
        msg = next(gen)
        if msg and msg.startswith("Segmenting"):
            break
    child = app.process._proc
    t = time.monotonic()
    c.cancel()
    with pytest.raises(Cancelled):
        wait(gen, timeout=10)
    deadline = time.monotonic() + 10
    while child.is_alive() and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not child.is_alive() and time.monotonic() - t < 5
    # The server keeps serving.
    assert wait(c.call("segment", np.zeros((4, 4)), model="cpsam_v2"), timeout=60).shape == (4, 4)


def test_queue_reports_position_to_the_second_user(server):
    _, url, tokens, _ = server
    alice, bob = client(url, tokens["alice"]), client(url, tokens["bob"])
    slow = alice.call("segment", np.zeros((8, 8)), model="slow")
    t = time.monotonic()
    while time.monotonic() - t < 30:
        msg = next(slow)
        if msg and msg.startswith("Segmenting"):
            break
    gen = bob.call("segment", np.zeros((4, 4)), model="cpsam_v2")
    messages = []
    while not any("Waiting for 1 other job" in m for m in messages):
        m = next(gen)
        if m:
            messages.append(m)
    alice.cancel()
    assert wait(gen, timeout=60).shape == (4, 4)


def test_crash_is_reported(server):
    _, url, tokens, _ = server
    c = client(url, tokens["alice"])
    with pytest.raises(RemoteError, match="stopped unexpectedly"):
        wait(c.call("segment", np.zeros((4, 4)), model="crash"), timeout=60)


def test_allowlist(tmp_path):
    users = Users(tmp_path / "users.json")
    token = users.add("alice")
    app = CellposeServer(users, backend="_fake_cellpose", gpu=False, model_names=lambda: ["m"], prepare=lambda n: iter(()))
    httpd = make_http_server(app, "127.0.0.1", 0, allow=["10.0.0.0/8"])
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        with pytest.raises(RemoteError, match="not allowed"):
            client(f"http://127.0.0.1:{httpd.server_address[1]}", token).status()
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def test_https_with_a_facility_certificate(tmp_path):
    import shutil
    import subprocess

    if not shutil.which("openssl"):
        pytest.skip("openssl not installed")
    cert, key = tmp_path / "cert.pem", tmp_path / "key.pem"
    subprocess.run(
        ["openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes", "-days", "1", "-subj", "/CN=localhost",
         "-addext", "subjectAltName=DNS:localhost,IP:127.0.0.1", "-keyout", str(key), "-out", str(cert)],
        check=True, capture_output=True,
    )
    users = Users(tmp_path / "users.json")
    token = users.add("alice")
    app = CellposeServer(users, backend="_fake_cellpose", gpu=False, model_names=lambda: ["m"], prepare=lambda n: iter(()))
    httpd = make_http_server(app, "127.0.0.1", 0, certfile=str(cert), keyfile=str(key))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    url = f"https://127.0.0.1:{httpd.server_address[1]}"
    try:
        with pytest.raises(RemoteError, match="cannot reach"):  # unknown certificate
            client(url, token).status()
        assert client(url, token, cafile=str(cert)).status()["user"] == "alice"
    finally:
        httpd.shutdown()
        httpd.server_close()
        app.close()


def test_user_admin_commands(tmp_path, capsys):
    cfg = ["--config-dir", str(tmp_path)]
    assert main(cfg + ["add-user", "carol"]) == 0
    token = capsys.readouterr().out.split("\n\n")[1].strip()
    users = Users(tmp_path / "users.json")
    assert users.check(token) == "carol" and users.check("nope") is None
    assert token not in (tmp_path / "users.json").read_text()  # only a hash is stored
    assert main(cfg + ["list-users"]) == 0 and "carol" in capsys.readouterr().out
    assert main(cfg + ["remove-user", "carol"]) == 0
    assert main(cfg + ["remove-user", "carol"]) == 1


def test_saved_servers(tmp_path):
    path = tmp_path / "servers.json"
    assert load_servers(path) == []
    save_servers([{"name": "facility-gpu", "url": "http://x:8765", "token": "t"}], path)
    assert load_servers(path)[0]["name"] == "facility-gpu"


def test_servers_dialog(server, qtbot, tmp_path):
    from ibif_napari._servers_dialog import ServersDialog

    _, url, tokens, _ = server
    path = tmp_path / "servers.json"
    dlg = ServersDialog(path=path)
    qtbot.addWidget(dlg)
    dlg.name.setText("Facility GPU")
    dlg.url.setText(url.removeprefix("http://"))  # scheme added automatically
    dlg.token.setText(tokens["alice"])
    assert dlg.test()
    assert "Connected as alice" in dlg.result.text() and "CPU" in dlg.result.text()
    dlg._save_current()
    assert load_servers(path) == [{"name": "Facility GPU", "url": url, "token": tokens["alice"]}]
    dlg.token.setText("wrong")
    assert not dlg.test() and "not accepted" in dlg.result.text()
    dlg._remove()
    assert load_servers(path) == []


# ---- Processing pipelines on the server ------------------------------------------------------

def _pipeline(**params):
    from ibif_napari._processing import Pipeline

    pipe = Pipeline()
    pipe.add("registration", params)
    return pipe


def _moving_stack(positions=0):
    from ibif_napari._processing import Stack
    from ibif_napari.vv import synth

    frames = synth.motion_scene(seed=4, n=3, size=96, max_shift=6).frames
    if positions:
        data = np.stack([np.roll(frames, p, axis=-1) for p in range(positions)])
        return Stack({"GFP": data, "DAPI": data * 2}, "PTYX", {"Y": 0.5, "X": 0.5}, "GFP")
    return Stack({"GFP": frames, "DAPI": (frames * 2).astype(np.float32)}, "TYX", {"Y": 0.5, "X": 0.5}, "GFP")


def test_pipeline_runs_on_the_server_as_it_does_here(server):
    app, url, tokens, _ = server
    stack = _moving_stack()
    pipe = _pipeline()
    local = pipe.run(stack)
    c = client(url, tokens["alice"])
    messages = []
    gen = c.process(stack, pipe)
    try:
        while True:
            m = next(gen)
            if m:
                messages.append(m)
    except StopIteration as stop:
        remote_stack, steps = stop.value
    assert any(m.startswith("Processing 3×96×96 (2 channel(s)): Stack registration") for m in messages)
    for name in ("GFP", "DAPI"):
        assert np.array_equal(remote_stack.channels[name], local.stack.channels[name])
    assert steps[0]["records"] == local.steps[0]["records"] and steps[0]["summary"] == local.steps[0]["summary"]
    info = c.status()
    assert "registration" in info["processing"]["steps"] and "illumination" not in info["processing"]["steps"]


def test_positions_are_sent_one_at_a_time(server):
    from ibif_napari._processing import Pipeline

    _, url, tokens, _ = server
    stack = _moving_stack(positions=2)
    pipe = _pipeline()
    local = pipe.run(stack)
    c = client(url, tokens["alice"])
    remote_stack, steps = wait(c.process(stack, pipe), timeout=120)
    assert np.array_equal(remote_stack.channels["DAPI"], local.stack.channels["DAPI"])
    key = lambda r: (r["P"], r["T"])  # noqa: E731
    assert sorted(steps[0]["records"], key=key) == sorted(local.steps[0]["records"], key=key)
    assert steps[0]["summary"].startswith("2 positions: ")
    assert isinstance(Pipeline, type)


def test_bad_pipelines_are_refused(server):
    from ibif_napari._processing import remote

    _, url, tokens, _ = server
    c = client(url, tokens["alice"])
    stack = _moving_stack()

    def post(pipeline_dict):
        body = remote.encode_request(stack, pipeline_dict)
        return c._request("POST", "/jobs", body, "application/octet-stream")

    with pytest.raises(RemoteError, match="unknown processing step 'os.system'"):
        post({"steps": [{"key": "os.system", "params": {"cmd": "rm -rf /"}}]})
    with pytest.raises(RemoteError, match="coming soon"):
        post({"steps": [{"key": "deconvolution"}]})
    with pytest.raises(RemoteError, match="unknown setting"):
        post({"steps": [{"key": "registration", "params": {"evil": 1}}]})
    with pytest.raises(RemoteError, match="is not one of"):
        post({"steps": [{"key": "registration", "params": {"model": "__import__"}}]})
    with pytest.raises(RemoteError, match="no enabled steps"):
        post({"steps": [{"key": "registration", "enabled": False}]})
    import io

    def raw(manifest, **arrays):
        buf = io.BytesIO()
        np.savez(buf, manifest=np.frombuffer(json.dumps(manifest).encode(), np.uint8), **arrays)
        return c._request("POST", "/jobs", buf.getvalue(), "application/octet-stream")

    good = {"op": "process", "channels": ["A"], "axes": "TYX", "spacing": {}, "reference": "A",
            "pipeline": _pipeline().to_dict()}
    frames = np.zeros((2, 8, 8), np.float32)
    with pytest.raises(RemoteError, match="bad axes"):
        raw({**good, "axes": "TQX"}, channel_0=frames)
    with pytest.raises(RemoteError, match="channel 'A' must be a numeric array with axes TYX"):
        raw(good, channel_0=frames[0])
    with pytest.raises(RemoteError, match="channel 'B' is missing"):
        raw({**good, "channels": ["A", "B"]}, channel_0=frames)
    with pytest.raises(RemoteError, match="bad voxel size"):
        raw({**good, "spacing": {"Y": -1}}, channel_0=frames)
    assert raw(good, channel_0=frames)["id"]


def test_old_server_and_too_large_are_caught_before_sending(server, monkeypatch):
    app, url, tokens, _ = server
    c = client(url, tokens["alice"])
    stack = _moving_stack()
    status = app.status
    monkeypatch.setattr(app, "status", lambda: {k: v for k, v in status().items() if k != "processing"})
    with pytest.raises(RemoteError, match="cannot run processing pipelines .* 0.37 or later"):
        wait(c.process(stack, _pipeline()))
    monkeypatch.setattr(app, "status", lambda: {**status(), "max_mb": 0.01})
    with pytest.raises(RemoteError, match="too large for facility-gpu: it accepts 0 MB"):
        wait(c.process(stack, _pipeline()))
    assert app.jobs == {}  # nothing was sent
    # The server refuses too, should a client not check.
    from ibif_napari._processing import remote

    app.max_bytes = 1000
    with pytest.raises(RemoteError, match="too large"):
        c._request("POST", "/jobs", remote.encode_request(stack, _pipeline()), "application/octet-stream")
    zeros = remote.sub_stack(stack, (slice(None),) * 3)
    zeros.channels = {n: np.zeros_like(a) for n, a in zeros.channels.items()}  # compresses well
    app.max_bytes = 20000  # the upload fits, the image (221 kB) does not
    with pytest.raises(RemoteError, match="larger than this server accepts"):
        c._request("POST", "/jobs", remote.encode_request(zeros, _pipeline()), "application/octet-stream")


def test_cancel_a_pipeline_on_the_server(server):
    from ibif_napari._processing import Stack

    app, url, tokens, _ = server
    c = client(url, tokens["alice"])
    rng = np.random.default_rng(0)
    big = Stack({"A": rng.random((12, 20, 160, 160)).astype(np.float32)}, "TZYX", {"Z": 1, "Y": .5, "X": .5})
    gen = c.process(big, _pipeline(model="Non-rigid (after rigid)"))
    deadline = time.monotonic() + 30
    while time.monotonic() < deadline:
        m = next(gen)
        if m and m.startswith("Processing"):
            break
    c.cancel()
    with pytest.raises(Cancelled):
        while True:
            next(gen)
    deadline = time.monotonic() + 20
    while time.monotonic() < deadline and any(j.state not in ("cancelled",) for j in app.jobs.values()):
        time.sleep(0.1)
    assert [j.state for j in app.jobs.values()] == ["cancelled"]


def test_processing_panel_runs_on_a_server(server, qtbot, monkeypatch):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    from ibif_napari import _cellpose_widget
    from ibif_napari._processing import session
    from ibif_napari._processing_widget import ProcessingWidget

    _, url, tokens, _ = server
    monkeypatch.setattr(_cellpose_widget, "remote_targets", lambda viewer: {
        "facility-gpu": (lambda: client(url, tokens["alice"]), None),
        "HPC (cluster)": (None, "Connect to HPC (cluster) in Segmentation → Cellpose first."),
    })
    v = ViewerModel()
    stack = _moving_stack()
    for name, data in stack.channels.items():
        v.add_image(data, name=name, scale=(1, 0.5, 0.5), metadata={"ibif_axes": "TYX"})
    session.set_pipeline(v, _pipeline(), notify=False)
    w = ProcessingWidget(v)
    qtbot.addWidget(w)
    assert [w.run_on.itemText(i) for i in range(w.run_on.count())] == ["This computer", "facility-gpu",
                                                                        "HPC (cluster)"]
    w.check_channels(["GFP", "DAPI"])
    w.reference.setCurrentText("GFP")
    w.run_on.setCurrentText("HPC (cluster)")
    w.run()
    assert "Connect to HPC (cluster)" in w.status.text() and w._worker is None
    w.run_on.setCurrentText("facility-gpu")
    w.run()
    qtbot.waitUntil(lambda: w._worker is None, timeout=60000)
    assert "on facility-gpu" in w.status.text(), w.status.text()
    local = _pipeline().run(stack)
    assert np.array_equal(v.layers["DAPI [processed]"].data, local.stack.channels["DAPI"])
    assert v.layers["GFP [processed]"].metadata["ibif_processing"][0]["records"] == local.steps[0]["records"]
