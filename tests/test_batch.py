import json
import time

import numpy as np
import pytest

pytest.importorskip("paramiko")

from _ssh_helpers import Prompts, run_messages  # noqa: E402

from ibif_napari import _batch  # noqa: E402
from ibif_napari._batch import BatchClient, load_records, run, save_tiff  # noqa: E402


def make_batch(folder, count, mask=None, model="fake"):
    (folder / "input").mkdir(parents=True)
    for i in range(count):
        np.savez_compressed(folder / "input" / f"{i:05d}.npz", image=np.full((3, 8, 8), i, np.float32))
    (folder / "job.json").write_text(json.dumps({"count": count, "params": {"model": model, "gpu": True}}))
    if mask is not None:
        np.save(folder / "mask.npy", mask)


def test_runner_segments_resumes_and_reports(tmp_path, monkeypatch):
    monkeypatch.setenv("IBIF_NAPARI_BATCH_BACKEND", "_fake_cellpose")
    folder = tmp_path / "batch-1"
    mask = np.zeros((8, 8), bool)
    mask[:4, :4] = True  # keeps object 1 (top-left corner), drops object 2
    make_batch(folder, 3, mask)
    (folder / "out").mkdir()
    np.save(folder / "out" / "00001.npy", np.full((3, 8, 8), 7, np.uint16))  # already done
    assert run(folder) == 0
    progress = json.loads((folder / "progress.json").read_text())
    assert progress["state"] == "done" and progress["done"] == 3 and progress["total"] == 3
    first = np.load(folder / "out" / "00000.npy")
    assert set(np.unique(first)) == {0, 1}
    assert (np.load(folder / "out" / "00001.npy") == 7).all()  # not redone
    assert not list((folder / "out").glob("*.part.npy"))


def test_runner_failure_is_recorded(tmp_path, monkeypatch):
    monkeypatch.setenv("IBIF_NAPARI_BATCH_BACKEND", "_fake_cellpose")
    folder = tmp_path / "batch-2"
    make_batch(folder, 2)
    (folder / "input" / "00001.npz").write_bytes(b"not an npz")
    assert run(folder) == 1
    progress = json.loads((folder / "progress.json").read_text())
    assert progress["state"] == "failed" and progress["done"] == 1
    assert progress["message"]


INFO = dict(name="s GFP cellpose batch", source="", axes="TZYX", scale=[1.0, 2.0, 0.5, 0.5],
            translate=[0.0, 2.0, 1.0, 1.5], out_shape=[3, 2, 4, 5], outer=[0], count=3)


def positions(count=3):
    for t in range(count):
        yield np.full((2, 4, 5), t, np.uint16), None


def wait_for(client, record, states, timeout=60):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        st = client.status(record)
        if st["state"] in states:
            return st
        time.sleep(0.3)
    raise TimeoutError(st)


def test_submit_fetch_end_to_end(profile, tmp_path):
    records = tmp_path / "batches.json"
    client = BatchClient(profile, Prompts(), tmp_path / "known_hosts")
    messages = []
    record = run_messages(client.submit(positions(), {"model": "fake", "do_3D": True}, None, INFO, records),
                          messages)
    try:
        assert messages[0].startswith("Logging in") and messages[-1].startswith(f"Batch job {record.job_id} submitted")
        assert (1, 3) in messages and (3, 3) in messages  # upload progress
        assert [r.id for r in load_records(records)] == [record.id]
        st = wait_for(client, record, {"done", "failed", "stopped"})
        assert st == {"state": "done", "done": 3, "total": 3, "message": "Segmented 3 positions"}

        labels, missing = run_messages(client.fetch(record), [])
        assert missing == [] and labels.shape == (3, 2, 4, 5) and labels.dtype == np.uint32
        assert labels[:, :, :2, :2].min() == 1 and labels[:, :, -2:, -2:].min() == 2

        path = save_tiff(labels, record, tmp_path / "results")
        import tifffile

        assert np.array_equal(tifffile.imread(path), labels)
        assert path.name == f"s GFP cellpose batch batch {record.id}.tif"
    finally:
        client.close()


def test_cancel_resubmit_and_remove(profile, tmp_path):
    records = tmp_path / "batches.json"
    profile.setup += "\nexport FAKE_SLURM_PENDING=60"  # stays queued
    client = BatchClient(profile, Prompts(), tmp_path / "known_hosts")
    try:
        record = run_messages(client.submit(positions(), {"model": "fake"}, None, INFO, records), [])
        assert client.status(record)["state"] == "queued"
        assert client.status(record)["message"] == "Waiting in the queue (Priority)"
        client.cancel(record)
        st = wait_for(client, record, {"stopped"})
        assert st["done"] == 0 and "cancelled or time limit" in st["message"]

        # Partial results can be fetched; missing positions are reported.
        labels, missing = run_messages(client.fetch(record), [])
        assert missing == [0, 1, 2] and labels.max() == 0

        old = record.job_id
        profile.setup = profile.setup.replace("FAKE_SLURM_PENDING=60", "FAKE_SLURM_PENDING=0")
        client.resubmit(record, records)
        assert record.job_id != old and load_records(records)[0].job_id == record.job_id
        assert wait_for(client, record, {"done"})["done"] == 3

        folder = record.remote_dir
        client.remove(record, records)
        assert load_records(records) == []
        _, out, _ = client.ssh.run(f"ls {folder} 2>&1")
        assert "No such file" in out
    finally:
        client.close()


def test_cancelled_upload_removes_the_partial_batch(profile, tmp_path):
    client = BatchClient(profile, Prompts(), tmp_path / "known_hosts")
    try:
        gen = client.submit(positions(), {"model": "fake"}, None, INFO, tmp_path / "batches.json")
        while next(gen) != (1, 3):
            pass
        gen.close()  # the panel's Cancel during upload
        workdir = profile.state.parent / "work"
        assert not list(workdir.glob("batch-*"))
        assert load_records(tmp_path / "batches.json") == []
    finally:
        client.close()


def test_batch_script(profile):
    script = profile.batch_script("/home/u/.ibif-napari-cluster/batch-x")
    assert "#SBATCH --job-name=ibif-napari-batch" in script
    assert "#SBATCH --time=12:00:00" in script
    assert "#SBATCH --output=/home/u/.ibif-napari-cluster/batch-x/job-%j.log" in script
    assert script.strip().endswith("exec python -m ibif_napari._batch run /home/u/.ibif-napari-cluster/batch-x")


def test_records_survive_restarts(tmp_path):
    path = tmp_path / "batches.json"
    rec = _batch.BatchRecord(id="a", cluster="c", remote_dir="/x/batch-a", job_id="1", submitted="2026-11-03T10:00:00",
                             **{k: INFO[k] for k in ("name", "source", "axes", "scale", "translate", "out_shape",
                                                     "outer", "count")})
    _batch.update_record(rec, path)
    rec.fetched = "/tmp/x.tif"
    _batch.update_record(rec, path)
    assert load_records(path) == [rec]
    _batch.remove_record(rec, path)
    assert load_records(path) == []


def test_tiff_goes_next_to_the_image_when_possible(tmp_path, monkeypatch):
    import tifffile

    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    rec = _batch.BatchRecord(id="b", cluster="c", remote_dir="/x/batch-b", job_id="1", submitted="t",
                             **{**{k: INFO[k] for k in ("axes", "scale", "translate", "out_shape", "outer", "count")},
                                "name": "cells/odd:name", "source": str(tmp_path / "data" / "movie.nd2")})
    (tmp_path / "data").mkdir()
    labels = np.ones((3, 2, 4, 5), np.uint32)
    path = save_tiff(labels, rec)
    assert path.parent == tmp_path / "data" and path.name == "cells_odd_name batch b.tif"
    assert tifffile.imread(path).shape == (3, 2, 4, 5)
    rec.source = "movie.nd2"  # not a full path: never the current folder
    assert save_tiff(labels, rec).parent == tmp_path / "home" / "ibif-napari-results"


def test_command_line_batches(profile, tmp_path, monkeypatch, capsys):
    from ibif_napari import _cluster

    monkeypatch.setattr(_batch, "RECORDS", tmp_path / "batches.json")
    monkeypatch.setattr(_cluster, "CONFIG", tmp_path / "clusters.json")
    monkeypatch.setattr(_cluster, "KNOWN_HOSTS", tmp_path / "known_hosts")
    monkeypatch.setattr(_cluster, "TerminalPrompts", Prompts)  # accept the host key
    _cluster.save_profiles([profile])
    client = BatchClient(profile, Prompts(), tmp_path / "known_hosts")
    record = run_messages(client.submit(positions(), {"model": "fake"}, None, INFO), [])
    wait_for(client, record, {"done"})
    client.close()

    assert _cluster.main(["batches"]) == 0
    out = capsys.readouterr().out
    assert record.id in out and "done 3/3" in out
    assert _cluster.main(["fetch", record.id, "--folder", str(tmp_path / "out")]) == 0
    assert "Saved" in capsys.readouterr().out
    assert list((tmp_path / "out").glob("*.tif"))
    assert _cluster.main(["fetch", "nope"]) == 1
