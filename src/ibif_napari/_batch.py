"""Batch mode: segment every time point / position as one cluster job.

napari does not need to stay open: the series (cropped to the region) is uploaded once,
a GPU job segments it, possibly overnight, and the labels are fetched later, from any
napari session, through the "Batch jobs" dialog.

On the cluster each batch is a folder in the profile's work folder:

    batch-<id>/job.json      settings, positions, Cellpose parameters
    batch-<id>/mask.npy      painted area (optional): objects mostly outside are dropped
    batch-<id>/input/NNNNN.npz   one image (+ nuclear channel) per position
    batch-<id>/out/NNNNN.npy     labels, written as each position finishes
    batch-<id>/progress.json     {"state", "done", "total", "message"}

The job resumes where it stopped: resubmitting a batch skips positions already done.
"""
from __future__ import annotations

import dataclasses
import datetime
import importlib
import io
import json
import os
import posixpath
import secrets
import sys
import traceback
from pathlib import Path

import numpy as np

from ._remote import CONFIG as _SERVERS_CONFIG

RECORDS = _SERVERS_CONFIG.with_name("batches.json")


# ---- runs on the cluster ------------------------------------------------------------


def _write_json(path, data):
    tmp = Path(str(path) + ".tmp")
    tmp.write_text(json.dumps(data))
    os.replace(tmp, path)


def run(folder):
    """Segment every position in `folder` that has no result yet. Returns an exit code."""
    folder = Path(folder)
    job = json.loads((folder / "job.json").read_text())
    total = job["count"]
    out = folder / "out"
    out.mkdir(exist_ok=True)
    progress = folder / "progress.json"

    def report(state, message):
        done = sum(1 for i in range(total) if (out / f"{i:05d}.npy").exists())
        _write_json(progress, {"state": state, "done": done, "total": total, "message": message,
                               "updated": datetime.datetime.now().isoformat(timespec="seconds")})

    # IBIF_NAPARI_BATCH_BACKEND swaps in a stand-in Cellpose module (tests only).
    backend = importlib.import_module(os.environ.get("IBIF_NAPARI_BATCH_BACKEND", "ibif_napari._cellpose"))
    from ._region import keep_inside

    gpu = bool(getattr(backend, "gpu_available", lambda: False)())
    params = {k: v for k, v in job["params"].items() if k != "gpu"}
    mask = np.load(folder / "mask.npy") if (folder / "mask.npy").exists() else None
    report("running", f"Starting on {'GPU' if gpu else 'CPU'}")
    try:
        for i in range(total):
            target = out / f"{i:05d}.npy"
            if target.exists():
                continue
            with np.load(folder / "input" / f"{i:05d}.npz", allow_pickle=False) as npz:
                image = npz["image"]
                nuclei = npz["nuclei"] if "nuclei" in npz.files else None
            labels = keep_inside(np.asarray(backend.segment(image, nuclei, gpu=gpu, **params)), mask)
            tmp = out / f"{i:05d}.part.npy"
            np.save(tmp, labels)
            os.replace(tmp, target)
            report("running", f"Segmented position {i + 1} of {total}")
    except Exception as e:
        report("failed", f"{type(e).__name__}: {e}")
        traceback.print_exc()
        return 1
    report("done", f"Segmented {total} positions")
    return 0


# ---- local records -----------------------------------------------------------------


@dataclasses.dataclass
class BatchRecord:
    id: str
    cluster: str  # profile name
    remote_dir: str
    job_id: str
    name: str  # labels layer name
    source: str  # image file, if known
    axes: str
    scale: list
    translate: list
    out_shape: list  # full labels shape (positions + cropped volume)
    outer: list  # indices of the position axes in out_shape
    count: int
    submitted: str
    fetched: str = ""  # TIFF path once fetched

    @classmethod
    def from_dict(cls, d):
        fields = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in d.items() if k in fields})


def load_records(path=None):
    try:
        return [BatchRecord.from_dict(d) for d in json.loads(Path(path or RECORDS).read_text())["batches"]]
    except (OSError, ValueError, KeyError, TypeError):
        return []


def save_records(records, path=None):
    path = Path(path or RECORDS)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"batches": [dataclasses.asdict(r) for r in records]}, indent=2))
    os.replace(tmp, path)


def update_record(record, path=None):
    records = [r for r in load_records(path) if r.id != record.id] + [record]
    save_records(sorted(records, key=lambda r: r.submitted), path)


def remove_record(record, path=None):
    save_records([r for r in load_records(path) if r.id != record.id], path)


# ---- local side: talking to the cluster ------------------------------------------------


class BatchClient:
    """One SSH login to a cluster, reused for submitting, checking and fetching batches."""

    def __init__(self, profile, prompts=None, known_hosts=None):
        from ._cluster import SSH

        self.profile = profile
        self.ssh = SSH(profile, prompts, known_hosts)

    def _connected(self):
        if self.ssh.transport is None or not self.ssh.transport.is_active():
            self.ssh.transport = None
            self.ssh.connect()
        return self.ssh

    def close(self):
        self.ssh.close()

    def submit(self, positions, params, mask, info, records_path=None):
        """Generator: uploads every position, submits the job; yields messages and
        (done, total) upload progress; returns the BatchRecord.

        `positions` yields (image, nuclei or None) in order; `info` gives the record's
        name, source, axes, scale, translate, out_shape, outer and count.
        """
        from ._cluster import ClusterError

        p = self.profile
        yield f"Logging in to {p.host}…"
        ssh = self._connected()
        workdir = ssh.check(p.shell(f"mkdir -p {p.workdir} && chmod 700 {p.workdir} && cd {p.workdir} && pwd"),
                            "creating the work folder").strip().splitlines()[-1]
        batch_id = datetime.datetime.now().strftime("%Y%m%d-%H%M%S-") + secrets.token_hex(2)
        folder = posixpath.join(workdir, f"batch-{batch_id}")
        ssh.check(f"mkdir -p {folder}/input {folder}/out", "creating the batch folder")
        done = False
        try:
            total = info["count"]
            for n, (image, nuclei) in enumerate(positions):
                yield (n, total)
                buf = io.BytesIO()
                arrays = {"image": np.ascontiguousarray(image)}
                if nuclei is not None:
                    arrays["nuclei"] = np.ascontiguousarray(nuclei)
                np.savez_compressed(buf, **arrays)
                ssh.put_bytes(f"{folder}/input/{n:05d}.npz", buf.getvalue())
            yield (total, total)
            if mask is not None:
                buf = io.BytesIO()
                np.save(buf, np.asarray(mask))
                ssh.put_bytes(f"{folder}/mask.npy", buf.getvalue())
            params = {k: v for k, v in params.items() if k != "gpu"}
            ssh.write_file(f"{folder}/job.json", json.dumps({"count": total, "params": params, "axes": info["axes"]}))
            ssh.write_file(f"{folder}/batch.sh", p.batch_script(folder), 0o700)
            yield f"Submitting the batch job on {p.name}…"
            out = ssh.check(p.shell(f"sbatch --parsable {folder}/batch.sh"), "sbatch")
            job_id = out.strip().splitlines()[-1].split(";")[0]
            if not job_id:
                raise ClusterError("sbatch did not return a job ID")
            record = BatchRecord(
                id=batch_id, cluster=p.name, remote_dir=folder, job_id=job_id,
                submitted=datetime.datetime.now().isoformat(timespec="seconds"),
                **{k: info[k] for k in ("name", "source", "axes", "scale", "translate", "out_shape", "outer", "count")},
            )
            update_record(record, records_path)
            done = True
            yield f"Batch job {job_id} submitted on {p.name}: {total} positions."
            return record
        finally:
            if not done:  # cancelled or failed during upload: remove the partial batch
                try:
                    ssh.run(f"rm -rf {folder}", timeout=60)
                except Exception:
                    pass

    def status(self, record):
        """{"state": queued/running/done/failed/stopped, "done", "total", "message"}."""
        p = self.profile
        ssh = self._connected()
        _, queue, _ = ssh.run(p.shell(f"squeue -h -j {record.job_id} -o '%T|%R'"), timeout=60)
        code, prog, _ = ssh.run(f"cat {record.remote_dir}/progress.json 2>/dev/null", timeout=30)
        progress = json.loads(prog.strip().splitlines()[-1]) if code == 0 and prog.strip() else {}
        done, total = progress.get("done", 0), progress.get("total", record.count)
        line = queue.strip().splitlines()[-1] if queue.strip() else ""
        state, reason = (line.split("|") + [""])[:2]
        if state == "PENDING":
            return {"state": "queued", "done": done, "total": total, "message": f"Waiting in the queue ({reason})"}
        if state in ("RUNNING", "COMPLETING", "CONFIGURING"):
            return {"state": "running", "done": done, "total": total,
                    "message": progress.get("message", "Starting")}
        # Not in the queue any more: finished, failed, cancelled or hit its time limit.
        if progress.get("state") == "done":
            return {"state": "done", "done": done, "total": total, "message": progress["message"]}
        if progress.get("state") == "failed":
            return {"state": "failed", "done": done, "total": total, "message": progress["message"]}
        tail = ""
        _, log, _ = ssh.run(f"tail -n 3 {record.remote_dir}/job-{record.job_id}.log 2>/dev/null", timeout=30)
        if log.strip():
            tail = " Log: " + " / ".join(log.strip().splitlines())
        return {"state": "stopped", "done": done, "total": total,
                "message": f"Job ended after {done} of {total} positions (cancelled or time limit).{tail}"}

    def cancel(self, record):
        self._connected().run(self.profile.shell(f"scancel {record.job_id}"), timeout=60)

    def resubmit(self, record, records_path=None):
        """Submit the batch again; positions already done are skipped."""
        from ._cluster import ClusterError

        out = self._connected().check(self.profile.shell(f"sbatch --parsable {record.remote_dir}/batch.sh"), "sbatch")
        record.job_id = out.strip().splitlines()[-1].split(";")[0]
        if not record.job_id:
            raise ClusterError("sbatch did not return a job ID")
        update_record(record, records_path)
        return record

    def remove(self, record, records_path=None):
        """Delete the batch folder on the cluster and forget the batch."""
        ssh = self._connected()
        if record.remote_dir and "/batch-" in record.remote_dir:  # never anything else
            ssh.run(self.profile.shell(f"scancel {record.job_id}; rm -rf {record.remote_dir}"), timeout=120)
        remove_record(record, records_path)

    def fetch(self, record):
        """Generator: downloads the labels; yields (done, total); returns (labels, missing
        positions). Positions without a result (job still running or stopped) stay 0."""
        ssh = self._connected()
        names = set(ssh.listdir(f"{record.remote_dir}/out"))
        labels = np.zeros(record.out_shape, dtype=np.uint32)
        outer_shape = [record.out_shape[i] for i in record.outer]
        missing = []
        for n, pos in enumerate(np.ndindex(*outer_shape)):
            yield (n, record.count)
            name = f"{n:05d}.npy"
            if name not in names:
                missing.append(n)
                continue
            index = [slice(None)] * len(record.out_shape)
            for axis, i in zip(record.outer, pos):
                index[axis] = i
            labels[tuple(index)] = np.load(io.BytesIO(ssh.get_bytes(f"{record.remote_dir}/out/{name}")),
                                           allow_pickle=False)
        yield (record.count, record.count)
        return labels, missing


def save_tiff(labels, record, folder=None):
    """Save the labels as a compressed TIFF next to the source image (or in `folder`, or in
    ~/ibif-napari-results if the source folder is not writable). Returns the path."""
    import tifffile

    candidates = [Path(folder)] if folder else []
    if record.source and Path(record.source).is_absolute() and Path(record.source).parent.is_dir():
        candidates.append(Path(record.source).parent)
    candidates.append(Path.home() / "ibif-napari-results")
    stem = "".join(c if c.isalnum() or c in "-_ " else "_" for c in record.name).strip()
    for d in candidates:
        try:
            d.mkdir(parents=True, exist_ok=True)
            path = d / f"{stem} batch {record.id}.tif"
            tifffile.imwrite(path, labels, compression="zlib", metadata={"axes": record.axes})
            return path
        except OSError:
            continue
    raise OSError("could not save the labels TIFF")


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if len(argv) == 2 and argv[0] == "run":
        return run(argv[1])
    print("usage: python -m ibif_napari._batch run <batch folder>", file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main())
