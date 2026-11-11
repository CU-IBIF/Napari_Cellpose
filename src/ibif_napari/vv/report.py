"""Turn a results folder (results.json, figures/, tables/) into report.html and report.md."""
from __future__ import annotations

import base64
import html
import json
from pathlib import Path

import pandas as pd

from .requirements import BY_ID, REQUIREMENTS

AREAS = ["Harness", "File reading", "Segmentation", "Secondary objects", "Measurements", "Saving"]
LIMITATIONS = [
    "The scenes are synthetic: ellipsoidal nuclei and cells with uniform staining, a Gaussian "
    "point spread function and Gaussian noise. Real tissue (texture, irregular shapes, dense "
    "clusters, uneven illumination) is harder; segmentation scores here are an upper bound to "
    "check against annotated real images.",
    "Nikon ND2 files cannot be written by open tools, so the ND2 reader is covered by its unit "
    "tests (tests/test_reader.py), not by this suite.",
    "Interactive steps (painting, the 3D brush, dialogs, remote servers and clusters) are covered "
    "by the unit tests in tests/; this suite calls the same functions the panels call.",
    "Segmentation results depend on the Cellpose version, model and device recorded under "
    "Environment; re-run the suite after changing any of them.",
]

CSS = """
:root { --bg:#ffffff; --fg:#1d232b; --muted:#5b6673; --line:#d9dee5; --card:#f6f8fa;
  --pass:#1a7f37; --pass-bg:#e6f4ea; --fail:#c62828; --fail-bg:#fdecea; --skip:#8a6d00;
  --skip-bg:#fff6d6; --accent:#2457a6; }
@media (prefers-color-scheme: dark) { :root:not([data-theme="light"]) { --bg:#14181d; --fg:#e4e8ee;
  --muted:#9aa5b1; --line:#2d343c; --card:#1b2027; --pass:#5cc97a; --pass-bg:#163222; --fail:#ff7b72;
  --fail-bg:#3a1b1b; --skip:#e3c35a; --skip-bg:#3a3215; --accent:#8ab4f8; } }
* { box-sizing:border-box; }
body { margin:0; background:var(--bg); color:var(--fg); font:15px/1.55 system-ui,-apple-system,"Segoe UI",
  Roboto,sans-serif; }
main { max-width:1080px; margin:0 auto; padding:28px 16px 60px; }
h1 { font-size:26px; margin:0 0 4px; } h2 { font-size:20px; margin:36px 0 10px; border-bottom:1px solid
  var(--line); padding-bottom:6px; } h3 { font-size:16px; margin:0 0 6px; }
.muted { color:var(--muted); } code { font-size:13px; }
.banner { padding:12px 14px; border-radius:8px; margin:16px 0; background:var(--skip-bg); color:var(--fg);
  border:1px solid var(--skip); }
.verdict { padding:14px 16px; border-radius:8px; margin:16px 0; border:1px solid var(--line);
  background:var(--card); }
.counts { display:flex; gap:10px; flex-wrap:wrap; margin:10px 0 0; }
.chip { padding:2px 10px; border-radius:999px; font-weight:600; font-size:13px; display:inline-block; }
.PASS { background:var(--pass-bg); color:var(--pass); } .FAIL, .ERROR { background:var(--fail-bg);
  color:var(--fail); } .NOT_RUN { background:var(--skip-bg); color:var(--skip); }
table { border-collapse:collapse; width:100%; font-size:14px; margin:8px 0; }
th, td { text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }
th { color:var(--muted); font-weight:600; }
.wrap { overflow-x:auto; }
.req { border:1px solid var(--line); border-radius:10px; padding:14px 16px; margin:14px 0;
  background:var(--card); }
.req p { margin:6px 0; }
figure { margin:10px 0; } figure img { max-width:100%; height:auto; border-radius:6px; background:#fff; }
figcaption { color:var(--muted); font-size:13px; }
.ok { color:var(--pass); font-weight:600; } .bad { color:var(--fail); font-weight:600; }
details summary { cursor:pointer; color:var(--accent); }
pre { white-space:pre-wrap; font-size:12px; }
"""


def _chip(status):
    return f'<span class="chip {status.replace(" ", "_")}">{html.escape(status)}</span>'


def _table_html(df, max_rows=40):
    if df is None or df.empty:
        return "<p class='muted'>(empty)</p>"
    shown = df.head(max_rows)
    fmt = shown.copy()
    for c in fmt.columns:
        if pd.api.types.is_float_dtype(fmt[c]):
            fmt[c] = fmt[c].map(lambda x: "" if pd.isna(x) else f"{x:.4g}")
    out = "<div class='wrap'>" + fmt.to_html(index=False, escape=True, border=0) + "</div>"
    if len(df) > max_rows:
        out += f"<p class='muted'>First {max_rows} of {len(df)} rows.</p>"
    return out


def _conclusion(summary):
    counts = summary["counts"]
    total = sum(counts.values())
    if counts["FAIL"] or counts["ERROR"]:
        bad = [e["id"] for e in summary["results"] if e["status"] in ("FAIL", "ERROR")]
        return (f"{counts['FAIL'] + counts['ERROR']} of {total} requirements did not pass ({', '.join(bad)}). "
                "See their sections below for which criteria failed and by how much.")
    ran = counts["PASS"]
    text = f"All {ran} requirements that were run passed"
    if counts["NOT RUN"]:
        text += f" ({counts['NOT RUN']} not run; see below)"
    return text + "."


def build(results_dir):
    """Write report.html and report.md into `results_dir`; returns their paths."""
    results_dir = Path(results_dir)
    summary = json.loads((results_dir / "results.json").read_text())
    (results_dir / "report.html").write_text(_html(summary, results_dir), encoding="utf-8")
    (results_dir / "report.md").write_text(_markdown(summary), encoding="utf-8")
    return results_dir / "report.html", results_dir / "report.md"


def _why(summary):
    """Per area: the first sentence of evidence of each requirement that passed."""
    out = []
    by_area = {}
    for e in summary["results"]:
        req = BY_ID[e["id"]]
        by_area.setdefault(req.area, []).append((req, e))
    for area in AREAS:
        items = by_area.get(area, [])
        if not items:
            continue
        passed = [(r, e) for r, e in items if e["status"] == "PASS"]
        other = [(r, e) for r, e in items if e["status"] != "PASS"]
        lines = [f"{r.id} ({r.title}): {e['evidence'][0] if e['evidence'] else 'passed.'}" for r, e in passed]
        lines += [f"{r.id} ({r.title}): {e['status']}" + (f" — {e['notes'][0]}" if e["notes"] else "")
                  for r, e in other]
        out.append((area, lines))
    return out


def _html(summary, results_dir):
    env = summary["environment"]
    counts = summary["counts"]
    h = ["<!doctype html><html lang='en'><head><meta charset='utf-8'>",
         "<meta name='viewport' content='width=device-width, initial-scale=1'>",
         "<title>IBIF napari V&amp;V report</title>", f"<style>{CSS}</style></head><body><main>",
         "<h1>IBIF napari: verification &amp; validation report</h1>",
         f"<p class='muted'>{html.escape(env.get('date') or '')} · ibif-napari {html.escape(str(env.get('ibif-napari')))}"
         f" · commit {html.escape(str(env.get('git commit')))} · segmentation: "
         f"{html.escape(summary['backend'])}</p>"]
    if not summary["verifies_cellpose"]:
        h.append("<div class='banner'><b>Segmentation back-end: "
                 f"{html.escape(summary['backend'])}.</b> The segmentation results in this report come from "
                 f"{_backend_words(summary['backend'])}, not Cellpose, so they do not verify Cellpose. Run "
                 "<code>ibif-napari-vv run --backend cellpose</code> for that.</div>")
    elif not summary["backend_available"]:
        h.append("<div class='banner'><b>Cellpose was not available</b>, so the segmentation requirements "
                 f"were not run: {html.escape(summary['backend_note'] or '')}</div>")
    h.append("<div class='verdict'><b>Verdict.</b> " + html.escape(_conclusion(summary)) + "<div class='counts'>"
             + "".join(f"{_chip(s)} {n}" for s, n in counts.items() if n) + "</div></div>")

    good = not (counts["FAIL"] or counts["ERROR"])
    h.append("<h2>Why the tool is working</h2>" if good else "<h2>What the checks show</h2>")
    h.append("<p>Each requirement below was checked on synthetic images whose true content is known exactly "
             "(see <a href='#data'>Test data</a>). Verification checks compare the tool with an independent "
             "calculation of the same quantity; validation checks compare it with the truth.</p>")
    for area, lines in _why(summary):
        h.append(f"<h3>{html.escape(area)}</h3><ul>" + "".join(f"<li>{html.escape(l)}</li>" for l in lines) + "</ul>")

    h.append("<h2>Requirements</h2><div class='wrap'><table><tr><th>ID</th><th>Requirement</th><th>Kind</th>"
             "<th>Result</th><th>Key numbers</th></tr>")
    for e in summary["results"]:
        req = BY_ID[e["id"]]
        key = "; ".join(f"{c['name']}: {c['shown']} ({c['want']})" for c in e["criteria"][:3])
        h.append(f"<tr><td><a href='#{e['id']}'>{e['id']}</a></td><td>{html.escape(req.title)}</td>"
                 f"<td>{req.kind}</td><td>{_chip(e['status'])}</td><td>{html.escape(key)}</td></tr>")
    h.append("</table></div>")

    h.append("<h2>Details</h2>")
    for e in summary["results"]:
        req = BY_ID[e["id"]]
        h.append(f"<section class='req' id='{e['id']}'><h3>{e['id']}: {html.escape(req.title)} "
                 f"{_chip(e['status'])}</h3>")
        h.append(f"<p><b>Requirement.</b> {html.escape(req.statement)}</p>")
        h.append(f"<p><b>Feature.</b> {html.escape(req.feature)} · <b>Kind.</b> {req.kind}</p>")
        h.append(f"<p><b>Method.</b> {html.escape(req.method)}</p>")
        if e["criteria"]:
            h.append("<table><tr><th>Criterion</th><th>Measured</th><th>Required</th><th></th></tr>")
            for c in e["criteria"]:
                mark = "<span class='ok'>pass</span>" if c["passed"] else "<span class='bad'>fail</span>"
                h.append(f"<tr><td>{html.escape(c['name'])}</td><td>{html.escape(c['shown'])}</td>"
                         f"<td>{html.escape(c['want'])}</td><td>{mark}</td></tr>")
            h.append("</table>")
        else:
            h.append("<p><b>Acceptance criteria.</b></p><ul>" + "".join(
                f"<li>{html.escape(c)}</li>" for c in req.criteria_text(summary["limits"])) + "</ul>")
        if e["evidence"]:
            h.append("<p><b>Evidence.</b> " + html.escape(" ".join(e["evidence"])) + "</p>")
        for f in e.get("figure_files", []):
            data = base64.b64encode((results_dir / f["file"]).read_bytes()).decode()
            h.append(f"<figure><img alt='{html.escape(f['caption'])}' src='data:image/png;base64,{data}'>"
                     f"<figcaption>{html.escape(f['caption'])}</figcaption></figure>")
        for t in e.get("table_files", []):
            df = pd.read_csv(results_dir / t["file"])
            h.append(f"<details><summary>Table: {html.escape(t['name'])} ({len(df)} rows, "
                     f"<code>{html.escape(t['file'])}</code>)</summary>{_table_html(df)}</details>")
        if e["notes"]:
            h.append("<details><summary>Notes</summary><pre>" + html.escape("\n".join(e["notes"])) + "</pre></details>")
        h.append(f"<p class='muted'>{e['seconds']:.1f} s</p></section>")

    h.append("<h2 id='data'>Test data</h2><p>Generated by <code>ibif-napari-vv generate</code> from fixed "
             "seeds (identical on every computer). Truth labels, noise-free images and the expected values are "
             "saved next to each image.</p><div class='wrap'><table><tr><th>Scene</th><th>Size (px)</th>"
             "<th>Voxel (µm)</th><th>Objects</th><th>Channels</th><th>SNR</th><th>PSF σ (µm)</th>"
             "<th>Notes</th></tr>")
    for name, s in summary["scenes"].items():
        sp = s["spec"]
        shape = "×".join(map(str, sp["shape"])) + (f" × {sp['timepoints']} T" if sp["timepoints"] > 1 else "")
        h.append(f"<tr><td>{html.escape(name)}</td><td>{shape}</td><td>{'×'.join(map(str, sp['spacing']))}</td>"
                 f"<td>{sp['n_objects']}</td><td>{', '.join(sp['channels'])}</td><td>{sp['snr']:g}</td>"
                 f"<td>{'×'.join(f'{v:g}' for v in sp['psf_sigma_um'])}</td><td>{html.escape(sp['notes'])}</td></tr>")
    h.append("</table></div>")

    h.append("<h2>Scope and limitations</h2><ul>" + "".join(f"<li>{html.escape(l)}</li>" for l in LIMITATIONS)
             + "</ul>")
    h.append("<h2>Environment</h2><table>" + "".join(
        f"<tr><th>{html.escape(k)}</th><td>{html.escape(str(v))}</td></tr>" for k, v in env.items()) + "</table>")
    if summary.get("timings_s"):
        h.append("<p class='muted'>Segmentation time: " + html.escape(", ".join(
            f"{k} {v:.1f} s" for k, v in summary["timings_s"].items())) + f". Whole run {summary['seconds']} s.</p>")
    h.append("<h2>Acceptance limits</h2><p class='muted'>From <code>vv/requirements.py</code> (LIMITS), "
             "possibly overridden for this run.</p><table>" + "".join(
                 f"<tr><th>{html.escape(k)}</th><td>{v}</td></tr>" for k, v in summary["limits"].items()) + "</table>")
    h.append("</main></body></html>")
    return "\n".join(h)


def _markdown(summary):
    env = summary["environment"]
    lines = ["# IBIF napari V&V report", "",
             f"{env.get('date')} · ibif-napari {env.get('ibif-napari')} · commit {env.get('git commit')} · "
             f"segmentation: {summary['backend']}", ""]
    if not summary["verifies_cellpose"]:
        lines += [f"> **Segmentation back-end: {summary['backend']}** — {_backend_words(summary['backend'])}, "
                  "not Cellpose; segmentation results do not verify Cellpose.", ""]
    elif not summary["backend_available"]:
        lines += [f"> **Cellpose was not available**: {summary['backend_note']}", ""]
    lines += [f"**Verdict:** {_conclusion(summary)}", "",
              "| ID | Requirement | Kind | Result | Key numbers |", "|---|---|---|---|---|"]
    for e in summary["results"]:
        req = BY_ID[e["id"]]
        key = "; ".join(f"{c['name']}: {c['shown']} ({c['want']})" for c in e["criteria"][:3])
        lines.append(f"| {e['id']} | {req.title} | {req.kind} | {e['status']} | {key.replace('|', '/')} |")
    good = not (summary["counts"]["FAIL"] or summary["counts"]["ERROR"])
    lines += ["", "## Why the tool is working" if good else "## What the checks show", ""]
    for area, items in _why(summary):
        lines += [f"**{area}**", ""] + [f"- {l}" for l in items] + [""]
    lines += ["## Scope and limitations", ""] + [f"- {l}" for l in LIMITATIONS]
    return "\n".join(lines) + "\n"


__all__ = ["build", "REQUIREMENTS"]


def _backend_words(backend):
    """What a non-Cellpose back-end is, for the report's banner."""
    return {"classic": "the Threshold + watershed tab", "baseline": "the Threshold + watershed tab",
            "truth": "an oracle that returns the truth (a self-check of the suite)"}.get(backend, "a stand-in")
