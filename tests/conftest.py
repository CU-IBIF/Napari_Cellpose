"""Fixtures: a throwaway OpenSSH server (key login, SFTP, port forwarding) and a cluster
profile that uses it with stand-in Slurm commands and a stand-in Cellpose."""
import getpass
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import pytest

from _ssh_helpers import FAKE_SLURM, TESTS, free_port


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
        "export IBIF_NAPARI_BATCH_BACKEND=_fake_cellpose",
        "unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY",
    ])
    from ibif_napari._cluster import ClusterProfile

    p = ClusterProfile(
        name="Test cluster", host="127.0.0.1", port=port, user=getpass.getuser(), key_file=str(d / "client"),
        setup=setup, workdir=str(tmp_path / "work"), activate=f"source {Path(sys.executable).parent}/activate",
        gpus="", server_args=f"--models-dir {tmp_path / 'models'} --preload fake --device cpu",
    )
    p.state = state
    return p
