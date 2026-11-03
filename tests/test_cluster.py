import getpass
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

paramiko = pytest.importorskip("paramiko")

from ibif_napari._cluster import (  # noqa: E402
    ClusterError,
    ClusterProfile,
    ClusterSession,
    SSH,
    load_profiles,
    save_profiles,
    check_login,
)
from ibif_napari._process import Cancelled, wait  # noqa: E402

TESTS = Path(__file__).parent
FAKE_SLURM = TESTS / "fake_slurm"


class Prompts:
    """Answers prompts automatically and records them."""

    def __init__(self, answers=(), confirm=True):
        self.answers, self.confirm_answer = list(answers), confirm
        self.asked, self.confirmed = [], []

    def ask(self, title, instructions, prompts):
        self.asked.append((title, instructions, [p for p, _ in prompts]))
        return [self.answers.pop(0) for _ in prompts]

    def confirm(self, title, text):
        self.confirmed.append(title)
        return self.confirm_answer


def free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


# ---- profile and job script ------------------------------------------------------------


def test_job_script_and_profiles(tmp_path):
    p = ClusterProfile(name="Anvil", host="anvil.example.edu", user="x-seth", partition="gpu",
                       account="abc123-gpu", gpus="1", extra_sbatch="#SBATCH --qos=normal",
                       setup="module load anaconda", idle_minutes=20, server_args="--models-dir /shared/models")
    script = p.job_script("s1")
    for line in ("#SBATCH --partition=gpu", "#SBATCH --account=abc123-gpu", "#SBATCH --gpus=1",
                 "#SBATCH --time=04:00:00", "#SBATCH --qos=normal",
                 "#SBATCH --output=$HOME/.ibif-napari-cluster/job-%j.log", "module load anaconda",
                 "source ~/.ibif-napari-cluster/venv/bin/activate"):
        assert line in script
    last = script.strip().splitlines()[-1]
    assert last.startswith("exec ibif-napari-server run --host 0.0.0.0 --port 0")
    assert "--token-file $HOME/.ibif-napari-cluster/token-s1" in last
    assert "--idle-exit 20 --models-dir /shared/models" in last
    assert "--gpus" not in ClusterProfile(name="c", host="h", user="u", gpus="").job_script("s")
    assert p.shell("squeue -o '%T|%R'") == "bash -lc 'module load anaconda\nsqueue -o '\"'\"'%T|%R'\"'\"''"
    assert "--no-deps" in p.install_script() and "--download-models" in p.install_script()

    path = tmp_path / "clusters.json"
    save_profiles([p], path)
    assert load_profiles(path) == [p]


# ---- two-factor login against an in-process SSH server --------------------------------


class TwoFactorServer(paramiko.ServerInterface):
    """Key first (partial success), then a keyboard-interactive code, like Duo setups."""

    def __init__(self, key):
        self.key, self.key_ok = key, False

    def get_allowed_auths(self, username):
        return "keyboard-interactive" if self.key_ok else "publickey"

    def check_auth_publickey(self, username, key):
        if key == self.key:
            self.key_ok = True
            return paramiko.AUTH_PARTIALLY_SUCCESSFUL
        return paramiko.AUTH_FAILED

    def check_auth_interactive(self, username, submethods):
        if not self.key_ok:
            return paramiko.AUTH_FAILED
        q = paramiko.server.InteractiveQuery("Duo two-factor", "Enter a passcode or 1 for a push.")
        q.add_prompt("Passcode or option (1-1): ", echo=True)
        return q

    def check_auth_interactive_response(self, responses):
        return paramiko.AUTH_SUCCESSFUL if list(responses) == ["123456"] else paramiko.AUTH_FAILED


def two_factor_host(host_key, client_key):
    listener = socket.socket()
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)

    def serve():
        conn, _ = listener.accept()
        t = paramiko.Transport(conn)
        t.add_server_key(host_key)
        t.start_server(server=TwoFactorServer(client_key))
        t.accept(10)

    threading.Thread(target=serve, daemon=True).start()
    return listener.getsockname()[1]


def test_key_then_two_factor_login(tmp_path):
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / "host")], check=True)
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / "client")], check=True)
    host_key = paramiko.PKey.from_path(tmp_path / "host")
    client_key = paramiko.PKey.from_path(tmp_path / "client")
    known = tmp_path / "known_hosts"

    port = two_factor_host(host_key, client_key)
    profile = ClusterProfile(name="c", host="127.0.0.1", port=port, user="seth", key_file=str(tmp_path / "client"))
    prompts = Prompts(answers=["123456"])
    ssh = SSH(profile, prompts, known).connect()
    assert ssh.transport.is_authenticated()
    assert prompts.confirmed == ["First connection to 127.0.0.1"]  # host key shown once
    assert prompts.asked == [("Duo two-factor", "Enter a passcode or 1 for a push.", ["Passcode or option (1-1): "])]
    assert f"[127.0.0.1]:{port} ssh-ed25519" in known.read_text()
    ssh.close()

    # Same host on a new port: known_hosts is per host:port, so it asks again; refusing cancels.
    port = two_factor_host(host_key, client_key)
    profile.port = port
    with pytest.raises(Cancelled):
        SSH(profile, Prompts(confirm=False), known).connect()

    # A changed host key is refused outright.
    subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(tmp_path / "host2")], check=True)
    port = two_factor_host(paramiko.PKey.from_path(tmp_path / "host2"), client_key)
    known.write_text(known.read_text() + f"[127.0.0.1]:{port} ssh-ed25519 {host_key.get_base64()}\n")
    profile.port = port
    with pytest.raises(ClusterError, match="CHANGED"):
        SSH(profile, Prompts(answers=["123456"]), known).connect()


# ---- end to end through a real OpenSSH server and stand-in Slurm ----------------------


@pytest.fixture(scope="module")
def sshd(tmp_path_factory):
    if not shutil.which("sshd") and not Path("/usr/sbin/sshd").exists():
        pytest.skip("OpenSSH server not installed")
    if os.geteuid() != 0:
        pytest.skip("needs root to run a throwaway sshd")
    d = tmp_path_factory.mktemp("sshd")
    for name in ("host", "client"):
        subprocess.run(["ssh-keygen", "-q", "-t", "ed25519", "-N", "", "-f", str(d / name)], check=True)
    shutil.copy(d / "client.pub", d / "authorized_keys")
    port = free_port()
    (d / "sshd_config").write_text(
        f"Port {port}\nListenAddress 127.0.0.1\nHostKey {d / 'host'}\nAuthorizedKeysFile {d / 'authorized_keys'}\n"
        f"PermitRootLogin yes\nStrictModes no\nPasswordAuthentication no\nKbdInteractiveAuthentication no\n"
        f"UsePAM no\nPidFile {d / 'sshd.pid'}\nAllowTcpForwarding yes\nSubsystem sftp internal-sftp\n"
    )
    Path("/run/sshd").mkdir(parents=True, exist_ok=True)
    proc = subprocess.Popen(["/usr/sbin/sshd", "-D", "-f", str(d / "sshd_config"), "-E", str(d / "log")])
    for _ in range(50):
        try:
            socket.create_connection(("127.0.0.1", port), timeout=0.2).close()
            break
        except OSError:
            time.sleep(0.1)
    yield d, port
    proc.terminate()
    proc.wait(5)


@pytest.fixture
def profile(sshd, tmp_path):
    d, port = sshd
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "fake").write_text("weights")
    state = tmp_path / "slurm"
    setup = "\n".join([
        f"export PATH={FAKE_SLURM}:$PATH",
        f"export FAKE_SLURM_DIR={state}",
        f"export PYTHONPATH={TESTS}",
        "export IBIF_NAPARI_SERVER_BACKEND=_fake_cellpose",
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY",
    ])
    p = ClusterProfile(
        name="Test cluster", host="127.0.0.1", port=port, user=getpass.getuser(), key_file=str(d / "client"),
        setup=setup, workdir=str(tmp_path / "work"), activate=f"source {Path(sys.executable).parent}/activate",
        gpus="", server_args=f"--models-dir {tmp_path / 'models'} --preload fake --device cpu",
    )
    p.state = state
    return p


def run_messages(gen, messages, timeout=120):
    start = time.monotonic()
    try:
        while True:
            m = next(gen)
            if m:
                messages.append(m)
            if time.monotonic() - start > timeout:
                raise TimeoutError(messages)
    except StopIteration as stop:
        return stop.value


def test_session_end_to_end(profile, tmp_path):
    profile.setup += "\nexport FAKE_SLURM_PENDING=2"  # queued for 2 s first
    known = tmp_path / "known_hosts"
    session = ClusterSession(profile, Prompts(), known)
    session.poll_s = 0.5
    messages = []
    process = run_messages(session.start(), messages)
    job = session.job_id
    try:
        assert messages[0] == "Logging in to 127.0.0.1…"
        assert any(m.startswith("Waiting in the Test cluster queue (Priority)") for m in messages)
        assert any("starting the Cellpose server" in m for m in messages)
        assert messages[-1].startswith(f"Connected to Test cluster: CPU on localhost, job {job}")

        # The panel's usual calls go through the tunnel to the job's server.
        assert [m for m in process.load("fake") if m][-1].startswith("Model ready")
        labels = wait(process.call("segment", np.zeros((3, 8, 8), np.float32), model="fake"), timeout=60)
        assert labels.shape == (3, 8, 8) and labels.max() == 2
        token = next((profile.state.parent / "work").glob("token-*"))
        assert oct(token.stat().st_mode)[-3:] == "600"
    finally:
        session.stop()
    assert (profile.state / "cancelled").read_text().split() == [job]
    work = profile.state.parent / "work"
    assert not list(work.glob("token-*")) and not list(work.glob("job-*.sh"))  # cleaned up
    assert list(work.glob(f"job-{job}.log"))  # the log is kept
    pid = int((profile.state / f"{job}.pid").read_text())
    for _ in range(50):
        try:
            os.kill(pid, 0)
            time.sleep(0.1)
        except ProcessLookupError:
            break
    else:
        pytest.fail("job still running after stop")


def test_cancel_while_queued_removes_the_job(profile, tmp_path):
    profile.setup += "\nexport FAKE_SLURM_PENDING=60"
    session = ClusterSession(profile, Prompts(), tmp_path / "known_hosts")
    session.poll_s = 0.5
    gen = session.start()
    messages = []
    while not any(m.startswith("Waiting in the") for m in messages):
        m = next(gen)
        if m:
            messages.append(m)
    job = session.job_id
    session.stop()  # the panel's Cancel
    with pytest.raises(Cancelled):
        run_messages(gen, messages)
    assert (profile.state / "cancelled").read_text().split()[-1] == job


def test_job_that_fails_reports_its_log(profile, tmp_path):
    profile.activate = "echo 'ModuleNotFoundError: cellpose' >&2; exit 3"
    session = ClusterSession(profile, Prompts(), tmp_path / "known_hosts")
    session.poll_s = 0.5
    with pytest.raises(ClusterError, match="ModuleNotFoundError: cellpose"):
        run_messages(session.start(), [])


def test_test_login_lists_partitions(profile, tmp_path):
    info = check_login(profile, Prompts(), tmp_path / "known_hosts")
    assert "gpu gpu:a100:4" in info["partitions"]
    import ibif_napari

    assert info["server_version"] == ibif_napari.__version__


def test_server_exits_when_idle(tmp_path):
    token = tmp_path / "token"
    token.write_text("secret\n")
    ready = tmp_path / "ready.json"
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "fake").write_text("w")
    env = {**os.environ, "PYTHONPATH": str(TESTS), "IBIF_NAPARI_SERVER_BACKEND": "_fake_cellpose"}
    proc = subprocess.Popen(
        [sys.executable, "-m", "ibif_napari._server", "run", "--host", "127.0.0.1", "--port", "0",
         "--token-file", str(token), "--ready-file", str(ready), "--idle-exit", "0.05",
         "--models-dir", str(tmp_path / "models"), "--preload", "fake", "--device", "cpu"],
        env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
    )
    try:
        for _ in range(100):
            if ready.exists():
                break
            time.sleep(0.1)
        info = json.loads(ready.read_text())
        assert info["port"] > 0 and info["host"]
        assert proc.wait(timeout=30) == 0  # stopped by itself after ~3 s idle
        assert not ready.exists()
        assert b"idle for" in proc.stdout.read()
    finally:
        proc.kill()


# ---- the napari panel ------------------------------------------------------------------


@pytest.fixture
def panel(profile, sshd, tmp_path, monkeypatch, qtbot):
    pytest.importorskip("napari")
    from napari.components import ViewerModel

    from ibif_napari import _cellpose, _cluster, _remote
    from ibif_napari._cellpose_widget import CellposeWidget
    from ibif_napari._reader import layer_data

    d, port = sshd
    known = tmp_path / "known_hosts"
    known.write_text(f"[127.0.0.1]:{port} {(d / 'host.pub').read_text().split()[0]} "
                     f"{(d / 'host.pub').read_text().split()[1]}\n")
    monkeypatch.setattr(_cluster, "KNOWN_HOSTS", known)
    monkeypatch.setattr(_remote, "CONFIG", tmp_path / "servers.json")
    monkeypatch.setattr(_cluster, "CONFIG", tmp_path / "clusters.json")
    monkeypatch.setenv(_cellpose.PRELOAD_ENV, "0")
    monkeypatch.setattr(_cluster.ClusterSession, "poll_s", 0.5)
    save_profiles([profile], tmp_path / "clusters.json")

    viewer = ViewerModel()
    data = np.random.randint(0, 100, (1, 4, 8, 8), dtype=np.uint16)  # T Z Y X, one channel
    sizes = {"T": 1, "Z": 4, "Y": 8, "X": 8}
    (ld,) = layer_data(data[:, :, None].squeeze(2), sizes, [], [], (2.0, 0.5, 0.5), None, "s", "s")
    viewer._add_layer_from_data(*ld)
    w = CellposeWidget(viewer)
    qtbot.addWidget(w)
    yield w, viewer
    w._stop_sessions()


def test_panel_connects_segments_and_disconnects(panel, profile, qtbot):
    w, viewer = panel
    label = "Test cluster (cluster)"
    assert label in [w.run_on.itemText(i) for i in range(w.run_on.count())]
    w.run_on.setCurrentText(label)
    assert w.connect_btn.isVisibleTo(w) and w.connect_btn.text() == "Connect"
    assert not w.run_btn.isEnabled() and w.status.text().startswith("Click Connect")

    w.model.setEditText("fake")  # preloaded on connect
    w.toggle_cluster()
    assert not w.run_on.isEnabled() and w.connect_btn.text() == "Cancel"
    qtbot.waitUntil(lambda: w.status.text().startswith("Connected to Test cluster (localhost), job"), timeout=60000)
    assert w.run_btn.isEnabled() and w.connect_btn.text() == "Disconnect" and w.run_on.isEnabled()
    assert "ends on Disconnect, or after 15 min unused" in w.status.text()
    job = w._sessions[label].job_id

    w.image.setCurrentText("s")
    assert w.model.currentText() == "fake"
    w.run()
    qtbot.waitUntil(lambda: any("cellpose" in l.name for l in viewer.layers), timeout=60000)
    labels = [l for l in viewer.layers if "cellpose" in l.name][0]
    assert labels.data.shape == (4, 8, 8) and labels.data.max() == 2
    qtbot.waitUntil(lambda: w._worker is None, timeout=10000)

    w.toggle_cluster()  # Disconnect
    assert w.connect_btn.text() == "Connect" and not w.run_btn.isEnabled()
    qtbot.waitUntil(lambda: (profile.state / "cancelled").exists(), timeout=20000)
    assert (profile.state / "cancelled").read_text().split() == [job]


def test_panel_cancel_while_queued(panel, profile, qtbot):
    w, _ = panel
    profile.setup += "\nexport FAKE_SLURM_PENDING=60"
    save_profiles([profile], profile.state.parent / "clusters.json")
    w._refresh_servers()
    w.run_on.setCurrentText("Test cluster (cluster)")
    w.toggle_cluster()
    qtbot.waitUntil(lambda: w.status.text().startswith("Waiting in the Test cluster queue"), timeout=60000)
    w.toggle_cluster()  # Cancel
    qtbot.waitUntil(lambda: w._connect_worker is None, timeout=20000)
    assert w.status.text().startswith("Connection cancelled")
    assert w.connect_btn.text() == "Connect" and not w._sessions
    qtbot.waitUntil(lambda: (profile.state / "cancelled").exists(), timeout=20000)


def test_generated_scripts_are_valid_bash(tmp_path):
    p = ClusterProfile(name="c", host="h", user="u", setup="module load python", extra_sbatch="#SBATCH --qos=x")
    for name, text in (("job.sh", p.job_script("s")), ("install.sh", p.install_script())):
        (tmp_path / name).write_text(text)
        subprocess.run(["bash", "-n", str(tmp_path / name)], check=True)


def test_command_line(tmp_path, monkeypatch, capsys):
    from ibif_napari import _cluster

    monkeypatch.setattr(_cluster, "CONFIG", tmp_path / "clusters.json")
    assert _cluster.main(["add", "Anvil", "--host", "anvil.example.edu", "--user", "x-seth",
                          "--partition", "gpu", "--account", "abc123", "--idle-minutes", "30"]) == 0
    (p,) = load_profiles(tmp_path / "clusters.json")
    assert (p.host, p.partition, p.idle_minutes, p.port) == ("anvil.example.edu", "gpu", 30, 22)
    assert _cluster.main(["list"]) == 0 and "x-seth@anvil.example.edu" in capsys.readouterr().out
    assert _cluster.main(["script", "Anvil"]) == 0 and "--account=abc123" in capsys.readouterr().out
    assert _cluster.main(["script", "Nope"]) == 1


def test_clusters_dialog(profile, sshd, tmp_path, monkeypatch, qtbot):
    from ibif_napari import _cluster
    from ibif_napari._cluster_ui import ClustersDialog

    d, port = sshd
    known = tmp_path / "known_hosts"
    pub = (d / "host.pub").read_text().split()
    known.write_text(f"[127.0.0.1]:{port} {pub[0]} {pub[1]}\n")
    monkeypatch.setattr(_cluster, "KNOWN_HOSTS", known)
    path = tmp_path / "clusters.json"
    dlg = ClustersDialog(path=path, prompts=Prompts())
    qtbot.addWidget(dlg)
    dlg._set_fields(profile)
    assert dlg.profile() == ClusterProfile(**{f: getattr(profile, f) for f in profile.__dataclass_fields__})
    dlg._show_script()
    assert "exec ibif-napari-server run" in dlg.log.toPlainText()
    dlg.test()
    qtbot.waitUntil(lambda: not dlg._busy, timeout=60000)
    text = dlg.log.toPlainText()
    assert "Logged in. IBIF napari on the cluster:" in text and "gpu gpu:a100:4" in text
    assert [p.name for p in load_profiles(path)] == ["Test cluster"]  # saved before testing


def test_refused_tunnel_is_explained(profile, sshd, tmp_path):
    d, _ = sshd
    # A second sshd that forbids port forwarding, like some clusters.
    port = free_port()
    cfg = (d / "sshd_config").read_text().replace("AllowTcpForwarding yes", "AllowTcpForwarding no")
    cfg = "\n".join(l for l in cfg.splitlines() if not l.startswith(("Port ", "PidFile")))
    (tmp_path / "sshd_config").write_text(f"Port {port}\nPidFile {tmp_path / 'pid'}\n{cfg}\n")
    proc = subprocess.Popen(["/usr/sbin/sshd", "-D", "-f", str(tmp_path / "sshd_config")])
    try:
        time.sleep(0.5)
        profile.port = port
        session = ClusterSession(profile, Prompts(), tmp_path / "known_hosts")
        session.poll_s = 0.5
        with pytest.raises(ClusterError, match="refused the SSH tunnel"):
            run_messages(session.start(), [])
        assert (profile.state / "cancelled").exists()  # the job was ended
    finally:
        proc.terminate()
