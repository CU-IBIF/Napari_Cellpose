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
