"""Run the Cellpose server as a GPU job on a Slurm cluster (e.g. an NSF ACCESS system).

One SSH login (key, then two-factor if the cluster asks) carries everything: submitting
the job, following it in the queue, and a tunnel from this computer to the job's compute
node. The job runs `ibif-napari-server` with a one-off token and stops by itself after a
period without use, so a forgotten session does not keep spending allocation credits.

Profiles live in ~/.ibif-napari/clusters.json. `ibif-napari-cluster` does the same from a
terminal (install, test, connect).
"""
from __future__ import annotations

import argparse
import dataclasses
import getpass
import json
import os
import posixpath
import secrets
import select
import shlex
import socket
import sys
import threading
import time
from pathlib import Path

from ._process import Cancelled
from ._remote import CONFIG as _SERVERS_CONFIG
from ._remote import RemoteProcess

CONFIG = _SERVERS_CONFIG.with_name("clusters.json")
KNOWN_HOSTS = _SERVERS_CONFIG.with_name("known_hosts")
PACKAGE = "ibif-napari @ git+https://github.com/CU-IBIF/Napari_Cellpose"


class ClusterError(RuntimeError):
    pass


@dataclasses.dataclass
class ClusterProfile:
    name: str
    host: str
    user: str
    port: int = 22
    key_file: str = ""  # empty: the SSH agent and ~/.ssh/id_* keys
    setup: str = ""  # shell lines run first, e.g. "module load python/3.11"
    workdir: str = "~/.ibif-napari-cluster"
    activate: str = ""  # empty: the environment that "Install" creates in workdir
    partition: str = ""
    account: str = ""
    gpus: str = "1"  # becomes "#SBATCH --gpus=<gpus>"; empty for none
    time: str = "04:00:00"
    cpus: int = 8
    mem: str = "32G"
    extra_sbatch: str = ""  # more "#SBATCH ..." lines, e.g. a QOS or constraint
    idle_minutes: int = 15
    server_args: str = ""  # extra ibif-napari-server options, e.g. --models-dir
    batch_time: str = "12:00:00"  # time limit of batch jobs (whole series, may run overnight)

    @classmethod
    def from_dict(cls, d):
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})

    def activate_line(self):
        return self.activate or f"source {self.workdir}/venv/bin/activate"

    def shell(self, command):
        """`command` with the profile's setup lines run first, for a non-interactive shell."""
        return "bash -lc " + shlex.quote(f"{self.setup}\n{command}" if self.setup.strip() else command)

    def _header(self, job_name, time_limit, log):
        lines = ["#!/bin/bash", f"#SBATCH --job-name={job_name}"]
        for opt, value in (("partition", self.partition), ("account", self.account), ("gpus", self.gpus),
                           ("time", time_limit), ("mem", self.mem)):
            if str(value).strip():
                lines.append(f"#SBATCH --{opt}={value}")
        if self.cpus:
            lines.append(f"#SBATCH --cpus-per-task={self.cpus}")
        lines.append(f"#SBATCH --output={log}")
        lines += [l.strip() for l in self.extra_sbatch.splitlines() if l.strip()]
        lines += ["", self.setup.strip(), self.activate_line(), ""]
        return lines

    def job_script(self, session):
        lines = self._header("ibif-napari", self.time, self._abs("job-%j.log"))
        lines.append(
            "exec ibif-napari-server run --host 0.0.0.0 --port 0"
            f" --token-file {self._abs(f'token-{session}')}"
            f" --ready-file {self._abs(f'ready-{session}.json')}"
            f" --idle-exit {self.idle_minutes} {self.server_args}".rstrip()
        )
        return "\n".join(lines) + "\n"

    def batch_script(self, folder):
        """Job script of a batch: segments every position in `folder` (absolute path)."""
        lines = self._header("ibif-napari-batch", self.batch_time, f"{folder}/job-%j.log")
        lines.append(f"exec python -m ibif_napari._batch run {shlex.quote(folder)}")
        return "\n".join(lines) + "\n"

    def install_script(self):
        venv = f"{self.workdir}/venv"
        return "\n".join([
            "set -e",
            f"mkdir -p {self.workdir} && chmod 700 {self.workdir}",
            f"if [ ! -x {venv}/bin/python ]; then python3 -m venv {venv}; fi" if not self.activate else "",
            self.activate_line(),
            "python --version",
            "python -m pip install --upgrade pip",
            # The server needs only numpy and Cellpose (with PyTorch), not napari or Qt.
            f"python -m pip install --upgrade --no-deps {shlex.quote(PACKAGE)}",
            "python -m pip install --upgrade numpy cellpose",
            # Compute nodes often have no internet: fetch the weights on the login node.
            "ibif-napari --download-models",
            'python -c "import torch; print(\'PyTorch\', torch.__version__, \'CUDA build:\', torch.version.cuda)"',
            "echo IBIF_INSTALL_OK",
        ])

    def _abs(self, name):
        # Paths in #SBATCH lines are not tilde-expanded: use $HOME there.
        base = self.workdir.replace("~", "$HOME", 1) if self.workdir.startswith("~") else self.workdir
        return f"{base}/{name}"


def load_profiles(path=None):
    path = Path(path or CONFIG)
    try:
        return [ClusterProfile.from_dict(d) for d in json.loads(path.read_text()).get("clusters", [])]
    except (OSError, ValueError, TypeError):
        return []


def save_profiles(profiles, path=None):
    path = Path(path or CONFIG)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"clusters": [dataclasses.asdict(p) for p in profiles]}, indent=2))
    os.replace(tmp, path)


# ---- prompts (terminal by default; the napari panel passes a dialog version) --------


class TerminalPrompts:
    def ask(self, title, instructions, prompts):
        """prompts: [(text, echo)] -> answers, or None to cancel."""
        if title or instructions:
            print("\n".join(x for x in (title, instructions) if x))
        return [input(t) if echo else getpass.getpass(t) for t, echo in prompts]

    def confirm(self, title, text):
        return input(f"{title}\n{text}\nContinue? [y/N] ").strip().lower() in ("y", "yes")


# ---- SSH ---------------------------------------------------------------------------


class SSH:
    """One authenticated SSH connection: commands, file writes and port forwarding."""

    def __init__(self, profile, prompts=None, known_hosts=None, timeout=20):
        import paramiko

        self.paramiko = paramiko
        self.profile = profile
        self.prompts = prompts or TerminalPrompts()
        self.known_hosts = Path(known_hosts or KNOWN_HOSTS)
        self.timeout = timeout
        self.transport = None

    def connect(self):
        p = self.paramiko
        try:
            sock = socket.create_connection((self.profile.host, self.profile.port), timeout=self.timeout)
        except OSError as e:
            raise ClusterError(f"cannot reach {self.profile.host}:{self.profile.port} ({e})") from None
        t = p.Transport(sock)
        t.start_client(timeout=self.timeout)
        self._check_host_key(t.get_remote_server_key())
        self.transport = t
        self._authenticate()
        t.set_keepalive(30)
        return self

    def _check_host_key(self, key):
        p, host = self.paramiko, self.profile.host
        name = host if self.profile.port == 22 else f"[{host}]:{self.profile.port}"
        known = p.HostKeys()
        for f in (Path.home() / ".ssh" / "known_hosts", self.known_hosts):
            try:
                known.load(str(f))
            except (OSError, p.SSHException):
                pass
        entry = known.lookup(name)
        if entry and key.get_name() in entry:
            if entry[key.get_name()] == key:
                return
            raise ClusterError(
                f"the host key of {host} has CHANGED. This may be an attack, or the cluster changed "
                f"its key: check with the cluster's support before removing it from {self.known_hosts}"
            )
        fingerprint = key.fingerprint if hasattr(key, "fingerprint") else key.get_base64()[:32]
        if not self.prompts.confirm(
            f"First connection to {host}",
            f"Its {key.get_name()} key fingerprint is {fingerprint}.\n"
            "Compare it with the one the cluster documents before continuing.",
        ):
            raise Cancelled()
        self.known_hosts.parent.mkdir(parents=True, exist_ok=True)
        with open(self.known_hosts, "a") as f:
            f.write(f"{name} {key.get_name()} {key.get_base64()}\n")

    def _keys(self):
        p = self.paramiko
        try:
            yield from p.Agent().get_keys()
        except Exception:
            pass
        files = [self.profile.key_file] if self.profile.key_file else [
            str(Path.home() / ".ssh" / n) for n in ("id_ed25519", "id_ecdsa", "id_rsa")
        ]
        for f in files:
            f = os.path.expanduser(f)
            if not os.path.exists(f):
                continue
            try:
                yield p.PKey.from_path(f)
            except p.PasswordRequiredException:
                answers = self.prompts.ask("SSH key", f"Passphrase for {f}", [("Passphrase: ", False)])
                if answers is None:
                    raise Cancelled() from None
                yield p.PKey.from_path(f, passphrase=answers[0])
            except Exception:
                continue

    def _authenticate(self):
        p, t, user = self.paramiko, self.transport, self.profile.user
        allowed = ["publickey", "keyboard-interactive", "password"]
        for key in self._keys():
            try:
                allowed = t.auth_publickey(user, key)
            except p.AuthenticationException:
                continue
            except p.BadAuthenticationType as e:
                allowed = e.allowed_types
                break
            if t.is_authenticated() or allowed:
                break  # done, or the key was accepted and a second factor follows
        if t.is_authenticated():
            return

        def handler(title, instructions, prompt_list):
            if not prompt_list:
                return []
            answers = self.prompts.ask(title or self.profile.host, instructions, prompt_list)
            if answers is None:
                raise Cancelled()
            return answers

        try:
            if "keyboard-interactive" in allowed:
                t.auth_interactive(user, handler)
            elif "password" in allowed:
                answers = self.prompts.ask(self.profile.host, f"Password for {user}", [("Password: ", False)])
                if answers is None:
                    raise Cancelled()
                t.auth_password(user, answers[0])
        except p.AuthenticationException as e:
            raise ClusterError(f"login to {self.profile.host} failed ({e})") from None
        if not t.is_authenticated():
            raise ClusterError(
                f"login to {self.profile.host} failed: no accepted key. Add your public key to the "
                "cluster (often through its web portal) or set the key file in the profile"
            )

    def run(self, command, timeout=120, stream=None):
        """Run a command; returns (exit code, stdout, stderr). `stream` gets output lines."""
        chan = self.transport.open_session(timeout=self.timeout)
        chan.exec_command(command)
        out, err, buf = [], [], ""
        deadline = time.monotonic() + timeout
        while True:
            if chan.recv_ready():
                data = chan.recv(65536).decode(errors="replace")
                out.append(data)
                if stream:
                    buf += data
                    *lines, buf = buf.split("\n")
                    for line in lines:
                        stream(line)
            elif chan.recv_stderr_ready():
                data = chan.recv_stderr(65536).decode(errors="replace")
                err.append(data)
                if stream:
                    for line in data.splitlines():
                        stream(line)
            elif chan.exit_status_ready():
                break
            elif time.monotonic() > deadline:
                chan.close()
                raise ClusterError(f"command timed out on {self.profile.host}: {command[:80]}")
            else:
                time.sleep(0.05)
        while chan.recv_ready():
            out.append(chan.recv(65536).decode(errors="replace"))
        while chan.recv_stderr_ready():
            err.append(chan.recv_stderr(65536).decode(errors="replace"))
        if stream and buf:
            stream(buf)
        return chan.recv_exit_status(), "".join(out), "".join(err)

    def check(self, command, what, timeout=120):
        code, out, err = self.run(command, timeout)
        if code != 0:
            detail = (err or out).strip().splitlines()[-3:]
            raise ClusterError(f"{what} failed on {self.profile.host}: {' / '.join(detail) or f'exit {code}'}")
        return out

    def write_file(self, path, text, mode=0o600):
        self.put_bytes(path, text.encode(), mode)

    def sftp(self):
        if getattr(self, "_sftp", None) is None:
            self._sftp = self.paramiko.SFTPClient.from_transport(self.transport)
        return self._sftp

    def put_bytes(self, path, data, mode=0o600):
        """Write a file on the cluster (written under a temporary name, then renamed)."""
        sftp = self.sftp()
        tmp = path + ".part"
        with sftp.open(tmp, "wb") as f:
            f.set_pipelined(True)
            f.write(data)
        sftp.chmod(tmp, mode)
        sftp.posix_rename(tmp, path)

    def get_bytes(self, path):
        with self.sftp().open(path, "rb") as f:
            f.prefetch()
            return f.read()

    def listdir(self, path):
        try:
            return self.sftp().listdir(path)
        except OSError:
            return []

    def close(self):
        if getattr(self, "_sftp", None) is not None:
            self._sftp.close()
            self._sftp = None
        if self.transport is not None:
            self.transport.close()
            self.transport = None


class Forwarder:
    """Listen on 127.0.0.1 and pass each connection through SSH to host:port."""

    def __init__(self, transport, host, port):
        self.transport, self.host, self.port = transport, host, port
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(16)
        self.local_port = self.sock.getsockname()[1]
        self.error = None  # why the cluster refused the last tunnel, if it did
        self._stop = False
        threading.Thread(target=self._accept, daemon=True).start()

    def _accept(self):
        while not self._stop:
            try:
                client, addr = self.sock.accept()
            except OSError:
                return
            try:
                chan = self.transport.open_channel("direct-tcpip", (self.host, self.port), addr, timeout=20)
            except Exception as e:
                self.error = str(e) or type(e).__name__
                client.close()
                continue
            threading.Thread(target=self._pipe, args=(client, chan), daemon=True).start()

    @staticmethod
    def _pipe(client, chan):
        try:
            while True:
                r, _, _ = select.select([client, chan], [], [], 1)
                if client in r:
                    data = client.recv(65536)
                    if not data:
                        break
                    chan.sendall(data)
                if chan in r:
                    data = chan.recv(65536)
                    if not data:
                        break
                    client.sendall(data)
        except OSError:
            pass
        finally:
            chan.close()
            client.close()

    def close(self):
        self._stop = True
        self.sock.close()


# ---- a session: one GPU job -----------------------------------------------------------


class ClusterSession:
    """Connect, submit the server job, wait for it, tunnel to it; `stop` ends the job."""

    poll_s = 5

    def __init__(self, profile, prompts=None, known_hosts=None):
        self.profile = profile
        self.ssh = SSH(profile, prompts, known_hosts)
        self.job_id = None
        self.forwarder = None
        self.process = None  # RemoteProcess once connected
        self._files = []  # this session's token and job script on the cluster
        self._stopped = False

    def start(self):
        """Generator: yields status messages, returns a RemoteProcess for the panel."""
        p = self.profile
        done = False
        try:
            yield f"Logging in to {p.host}…"
            self.ssh.connect()
            yield f"Submitting a GPU job on {p.name}…"
            session = secrets.token_hex(6)
            workdir = self.ssh.check(p.shell(f"mkdir -p {p.workdir} && chmod 700 {p.workdir} && cd {p.workdir} && pwd"),
                                     "creating the work folder").strip().splitlines()[-1]
            token = secrets.token_urlsafe(32)
            self._files = [posixpath.join(workdir, f"token-{session}"), posixpath.join(workdir, f"job-{session}.sh")]
            self.ssh.write_file(self._files[0], token + "\n")
            script = self._files[1]
            self.ssh.write_file(script, p.job_script(session), 0o700)
            out = self.ssh.check(p.shell(f"sbatch --parsable {shlex.quote(script)}"), "sbatch")
            self.job_id = out.strip().splitlines()[-1].split(";")[0]
            node = yield from self._wait_running()
            ready = yield from self._wait_ready(posixpath.join(workdir, f"ready-{session}.json"), node)
            host = node or ready["host"]
            self.forwarder = Forwarder(self.ssh.transport, host, ready["port"])
            self.process = RemoteProcess(f"http://127.0.0.1:{self.forwarder.local_port}", token,
                                         name=f"{p.name} ({host})", timeout=60)
            try:
                info = self.process.status()
            except Exception:
                if self.forwarder.error:
                    raise ClusterError(
                        f"{p.host} refused the SSH tunnel to {host}:{ready['port']} ({self.forwarder.error}). "
                        "Ask the cluster's support whether SSH port forwarding to compute nodes is allowed"
                    ) from None
                raise
            yield f"Connected to {p.name}: {info.get('device')} on {host}, job {self.job_id}"
            done = True
            return self.process
        finally:
            if not done:  # failed or cancelled part-way: do not leave a job queued
                self.stop()

    def _wait_running(self):
        start, last = time.monotonic(), None
        while True:
            if self._stopped:
                raise Cancelled()
            out = self.ssh.check(self.profile.shell(f"squeue -h -j {self.job_id} -o '%T|%R|%N'"), "squeue")
            line = out.strip().splitlines()[-1] if out.strip() else ""
            state, reason, node = (line.split("|") + ["", "", ""])[:3]
            if state == "RUNNING":
                return node.strip()
            if not state or state in ("FAILED", "CANCELLED", "TIMEOUT", "COMPLETED", "NODE_FAIL", "OUT_OF_MEMORY"):
                raise ClusterError(f"job {self.job_id} ended before starting the server. {self._log_tail()}")
            minutes = int((time.monotonic() - start) // 60)
            message = f"Waiting in the {self.profile.name} queue ({reason or state}), {minutes} min, job {self.job_id}…"
            if message != last:
                last = message
                yield message
            yield from self._sleep(self.poll_s)

    def _wait_ready(self, ready_file, node):
        yield f"Job {self.job_id} running on {node or 'its node'}: starting the Cellpose server…"
        while True:
            if self._stopped:
                raise Cancelled()
            code, out, _ = self.ssh.run(f"cat {shlex.quote(ready_file)} 2>/dev/null", timeout=30)
            if code == 0 and out.strip():
                return json.loads(out.strip().splitlines()[-1])  # ignore any shell start-up output
            code, out, _ = self.ssh.run(self.profile.shell(f"squeue -h -j {self.job_id} -o '%T'"), timeout=60)
            if not out.strip():
                raise ClusterError(f"job {self.job_id} stopped while starting. {self._log_tail()}")
            yield from self._sleep(2)

    def _sleep(self, seconds):
        end = time.monotonic() + seconds
        while time.monotonic() < end:
            if self._stopped:
                raise Cancelled()
            yield None
            time.sleep(0.1)

    def _log_tail(self):
        try:
            log = self.profile._abs(f"job-{self.job_id}.log")
            _, out, _ = self.ssh.run(self.profile.shell(f"tail -n 5 {log}"), timeout=30)
            return f"Last lines of its log:\n{out.strip()}" if out.strip() else ""
        except Exception:
            return ""

    def stop(self):
        """End the job (scancel) and close the tunnel and the SSH connection."""
        self._stopped = True
        if self.process is not None:
            self.process.cancel()
        if self.forwarder is not None:
            self.forwarder.close()
            self.forwarder = None
        if self.ssh.transport is not None:
            commands = [f"scancel {self.job_id}"] if self.job_id else []
            if self._files:  # the token is no longer needed; job logs are kept
                commands.append("rm -f " + " ".join(shlex.quote(f) for f in self._files))
            if commands:
                try:
                    self.ssh.run(self.profile.shell("; ".join(commands)), timeout=30)
                except Exception:
                    pass
        self.job_id = None
        self._files = []
        self.ssh.close()


def install(profile, prompts=None, stream=print, known_hosts=None):
    """Install or update the server on the cluster (on the login node). Returns True on success."""
    ssh = SSH(profile, prompts, known_hosts).connect()
    try:
        code, out, _ = ssh.run(profile.shell(profile.install_script()), timeout=3600, stream=stream)
        return code == 0 and "IBIF_INSTALL_OK" in out
    finally:
        ssh.close()


def check_login(profile, prompts=None, known_hosts=None):
    """Log in and report the Slurm partitions and the installed server version."""
    ssh = SSH(profile, prompts, known_hosts).connect()
    try:
        _, parts, _ = ssh.run(profile.shell("sinfo -h -o '%P %G' 2>/dev/null | sort -u | head -20"), timeout=60)
        _, ver, _ = ssh.run(profile.shell(f"{profile.activate_line()} 2>/dev/null && "
                                          "python -c 'import ibif_napari as m; print(m.__version__)'"), timeout=120)
        return {"partitions": parts.strip(), "server_version": ver.strip().splitlines()[-1] if ver.strip() else None}
    finally:
        ssh.close()


# ---- command line --------------------------------------------------------------------


def main(argv=None):
    parser = argparse.ArgumentParser(prog="ibif-napari-cluster", description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("list", help="list cluster profiles")
    add = sub.add_parser("add", help="add or replace a cluster profile")
    for f in dataclasses.fields(ClusterProfile):
        flag = "--" + f.name.replace("_", "-")
        if f.name == "name":
            add.add_argument("name")
        elif f.default is dataclasses.MISSING:
            add.add_argument(flag, required=True, type=f.type if f.type in (int, "int") else str)
        else:
            add.add_argument(flag, default=f.default, type=int if isinstance(f.default, int) else str)
    for name, help_ in (("test", "log in and list partitions"), ("install", "install or update the server on the cluster"),
                        ("connect", "start a GPU job and keep a tunnel open until Ctrl+C"), ("script", "print the job script")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("name")
    sub.add_parser("batches", help="list batch jobs and their status")
    for name, help_ in (("fetch", "download a batch's labels as a TIFF"), ("cancel", "cancel a batch job"),
                        ("resubmit", "submit a batch again (finished positions are skipped)")):
        sp = sub.add_parser(name, help=help_)
        sp.add_argument("batch", help="batch ID (see: batches)")
        if name == "fetch":
            sp.add_argument("--folder", help="where to save the TIFF (default: next to the image)")
    args = parser.parse_args(argv)
    if args.command in ("batches", "fetch", "cancel", "resubmit"):
        return _batch_command(args)

    profiles = load_profiles()
    if args.command == "list":
        for p in profiles:
            print(f"{p.name}: {p.user}@{p.host} partition={p.partition or '-'} account={p.account or '-'}")
        return 0
    if args.command == "add":
        values = {f.name: getattr(args, f.name) for f in dataclasses.fields(ClusterProfile)}
        profiles = [p for p in profiles if p.name != args.name] + [ClusterProfile(**values)]
        save_profiles(profiles)
        print(f"Saved {args.name} in {CONFIG}")
        return 0
    profile = next((p for p in profiles if p.name == args.name), None)
    if profile is None:
        print(f"No cluster profile {args.name!r} (see: ibif-napari-cluster list)", file=sys.stderr)
        return 1
    try:
        if args.command == "script":
            print(profile.job_script("<session>"))
        elif args.command == "test":
            info = check_login(profile)
            print(f"Logged in. Server installed: {info['server_version'] or 'no'}\nPartitions:\n{info['partitions']}")
        elif args.command == "install":
            return 0 if install(profile) else 1
        elif args.command == "connect":
            session = ClusterSession(profile)
            gen = session.start()
            try:
                while True:
                    m = next(gen)
                    if m:
                        print(m, flush=True)
            except StopIteration as stop:
                proc = stop.value
            print(f"\nServer: {proc.url}  token: {proc.token}\nCtrl+C to end the job.")
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("Ending the job…")
                session.stop()
    except ClusterError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    return 0


def _batch_command(args):
    from . import _batch

    records = _batch.load_records()
    profiles = {p.name: p for p in load_profiles()}
    clients = {}

    def client(rec):
        if rec.cluster not in profiles:
            raise ClusterError(f"no cluster profile named {rec.cluster!r}")
        if rec.cluster not in clients:
            clients[rec.cluster] = _batch.BatchClient(profiles[rec.cluster])
        return clients[rec.cluster]

    try:
        if args.command == "batches":
            if not records:
                print("No batch jobs.")
            for rec in records:
                try:
                    st = client(rec).status(rec)
                    state = f"{st['state']} {st['done']}/{st['total']}: {st['message']}"
                except (ClusterError, OSError) as e:
                    state = f"unknown ({e})"
                print(f"{rec.id}  {rec.cluster}  job {rec.job_id}  {rec.name}  {state}"
                      + (f"  fetched: {rec.fetched}" if rec.fetched else ""))
            return 0
        rec = next((r for r in records if r.id == args.batch), None)
        if rec is None:
            print(f"No batch {args.batch!r} (see: ibif-napari-cluster batches)", file=sys.stderr)
            return 1
        if args.command == "cancel":
            client(rec).cancel(rec)
            print(f"Cancelled job {rec.job_id}.")
        elif args.command == "resubmit":
            client(rec).resubmit(rec)
            print(f"Resubmitted as job {rec.job_id}.")
        else:
            gen = client(rec).fetch(rec)
            try:
                while True:
                    done, total = next(gen)
                    print(f"\rDownloading {done}/{total}", end="", flush=True)
            except StopIteration as stop:
                labels, missing = stop.value
            path = _batch.save_tiff(labels, rec, args.folder)
            rec.fetched = str(path)
            _batch.update_record(rec)
            print(f"\nSaved {path}" + (f" ({len(missing)} positions had no result yet)" if missing else ""))
        return 0
    except ClusterError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1
    finally:
        for c in clients.values():
            c.close()


if __name__ == "__main__":
    sys.exit(main())
