"""Command line: ibif-napari-vv generate | run | report | all | requirements."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None):
    p = argparse.ArgumentParser(
        prog="ibif-napari-vv",
        description="Verification & validation of IBIF napari on synthetic data with known truth.")
    sub = p.add_subparsers(dest="command", required=True)

    g = sub.add_parser("generate", help="make the synthetic test data (truth)")
    g.add_argument("truth", type=Path, help="folder to write the scenes into")
    g.add_argument("--quick", action="store_true", help="smaller scenes (minutes instead of tens of minutes on CPU)")

    def run_args(sp):
        sp.add_argument("--backend", default="cellpose", choices=["cellpose", "baseline", "truth"],
                        help="segmentation to test (default: cellpose, as the Cellpose 3D panel runs it)")
        sp.add_argument("--model", help="Cellpose model (default: the panel's default model)")
        sp.add_argument("--gpu", action="store_true", help="run Cellpose on the GPU")
        sp.add_argument("--diameter", type=float, help="object diameter in pixels (default: automatic)")
        sp.add_argument("--only", nargs="+", metavar="ID", help="run only these requirements, e.g. VV-SEG-01")
        sp.add_argument("--limits", type=Path, help="JSON file overriding acceptance limits")
        sp.add_argument("--strict", action="store_true",
                        help="exit with status 1 unless every requirement passed (NOT RUN counts as failure)")

    r = sub.add_parser("run", help="run the checks on generated data")
    r.add_argument("truth", type=Path)
    r.add_argument("results", type=Path)
    run_args(r)

    rp = sub.add_parser("report", help="write report.html and report.md from a results folder")
    rp.add_argument("results", type=Path)

    a = sub.add_parser("all", help="generate, run and report into one folder (truth/ and results/)")
    a.add_argument("folder", type=Path)
    a.add_argument("--quick", action="store_true")
    run_args(a)

    sub.add_parser("requirements", help="print the requirements as Markdown")

    args = p.parse_args(argv)
    if args.command == "requirements":
        from .requirements import markdown

        sys.stdout.write(markdown())
        return 0
    if args.command == "generate":
        return _generate(args.truth, args.quick)
    if args.command == "report":
        return _report(args.results)
    truth, results = (args.truth, args.results) if args.command == "run" else \
        (args.folder / "truth", args.folder / "results")
    if args.command == "all":
        _generate(truth, args.quick)
    status = _run(truth, results, args)
    _report(results)
    return status


def _generate(truth, quick):
    from . import synth

    print(f"Writing synthetic scenes to {truth} …")
    for folder in synth.generate_all(truth, quick=quick):
        print(f"  {folder.name}")
    return 0


def _run(truth, results, args):
    from .run import run

    limits = json.loads(args.limits.read_text()) if args.limits else None
    options = {}
    if args.backend == "cellpose":
        options = {"model": args.model, "gpu": args.gpu, "diameter": args.diameter}
    summary = run(truth, results, args.backend, only=args.only, limits=limits, **options)
    counts = summary["counts"]
    print("Result: " + ", ".join(f"{k} {v}" for k, v in counts.items() if v))
    bad = counts["FAIL"] + counts["ERROR"] + (counts["NOT RUN"] if args.strict else 0)
    return 1 if bad else 0


def _report(results):
    from .report import build

    html_path, md_path = build(results)
    print(f"Report: {html_path}\n        {md_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
