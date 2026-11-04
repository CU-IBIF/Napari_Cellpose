"""Run the checks and save everything a report needs (results.json, tables, figures)."""
from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

from . import backends
from .checks import Context, run_checks


def environment(backend):
    def v(dist):
        try:
            return version(dist)
        except PackageNotFoundError:
            return None

    commit = None
    try:
        here = Path(__file__).resolve().parent
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=here, capture_output=True,
                                text=True, timeout=5).stdout.strip() or None
    except Exception:
        pass
    return {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "ibif-napari": _own_version(), "git commit": commit,
        "python": sys.version.split()[0], "platform": platform.platform(), "machine": platform.machine(),
        "processor": platform.processor() or None,
        "numpy": v("numpy"), "scipy": v("scipy"), "scikit-image": v("scikit-image"), "napari": v("napari"),
        **{f"segmentation {k}": val for k, val in backend.describe().items()},
    }


def run(truth_dir, out_dir, backend="cellpose", only=None, limits=None, log=print, **backend_options):
    """Run the suite on the scenes in `truth_dir`; write results into `out_dir`.
    Returns the summary dict (also saved as results.json)."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    be = backends.make(backend, **backend_options) if isinstance(backend, str) else backend
    ctx = Context(truth_dir, out, be, limits, log)
    if not ctx.backend_ok:
        log(f"Segmentation back-end '{be.name}' is not available: {ctx.backend_why}")
    t0 = time.perf_counter()
    results = run_checks(ctx, only)
    figures = out / "figures"
    tables = out / "tables"
    figures.mkdir(exist_ok=True)
    tables.mkdir(exist_ok=True)
    entries = []
    for res in results:
        entry = res.to_json()
        entry["figure_files"] = []
        for k, (caption, png) in enumerate(res.figures):
            name = f"{res.req.id}_{k + 1}.png"
            (figures / name).write_bytes(png)
            entry["figure_files"].append({"caption": caption, "file": f"figures/{name}"})
        entry["table_files"] = []
        for name, table in res.tables.items():
            safe = "".join(c if c.isalnum() else "_" for c in name)
            path = tables / f"{res.req.id}_{safe}.csv"
            table.to_csv(path, index=False)
            entry["table_files"].append({"name": name, "file": f"tables/{path.name}"})
        entries.append(entry)
    manifest = Path(truth_dir) / "manifest.json"
    summary = {
        "environment": environment(be),
        "backend": be.name, "verifies_cellpose": be.verifies_cellpose,
        "backend_available": ctx.backend_ok, "backend_note": ctx.backend_why,
        "truth": str(Path(truth_dir).resolve()),
        "truth_manifest": json.loads(manifest.read_text()) if manifest.exists() else None,
        "scenes": {name: {"spec": ctx.scene(name).spec.to_json(), "noise_sigma": ctx.scene(name).noise_sigma}
                   for name in sorted(ctx._scenes)},
        "limits": ctx.limits, "timings_s": ctx.timings,
        "seconds": round(time.perf_counter() - t0, 1),
        "results": entries,
        "counts": {s: sum(1 for e in entries if e["status"] == s) for s in ("PASS", "FAIL", "NOT RUN", "ERROR")},
    }
    (out / "results.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary


def _own_version():
    from .. import __version__

    return __version__
