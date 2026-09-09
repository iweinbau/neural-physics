"""Publish trained models as a static site (GitHub Pages).

Two steps, both without a server:

    pdm run publish-model runs/cloth models/cloth     # snapshot: copy model.pt, pca.npz, summary, history  (commit these)
    pdm run pages-build                               # build:    models/* -> site/  (landing page + interactive viewers)
    pdm run pages-serve                               # preview:  http://localhost:8000

`build` re-evaluates every model of `configs/pages.yml` on its held-out episode (the dataset is regenerated
when missing, so nothing big has to be committed) and writes `site/index.html` plus one self-contained
viewer per model. `.github/workflows/pages.yml` runs exactly this on GitHub Actions and deploys `site/`.
"""
from __future__ import annotations

import argparse
import html
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from neural_physics.artifacts import ArtifactMismatch, divergence_warnings, load_run
from neural_physics.pipeline import load_dataset
from neural_physics.runtime.export import export_viewer
from neural_physics.runtime.rollout import evaluate_sequence
SNAPSHOT_FILES = ["model.pt", "pca.npz", "pca_external.npz", "summary.json"]


# ---------------------------------------------------------------------------- snapshot
def snapshot(run_dir, model_dir) -> Path:
    """Copy the publishable part of a finished training run into `model_dir` (small: ~0.5-2 MB)."""
    run_dir, model_dir = Path(run_dir), Path(model_dir)
    if not (run_dir / "model.pt").exists():
        raise SystemExit(f"{run_dir / 'model.pt'} not found -- is the training run finished?")
    model_dir.mkdir(parents=True, exist_ok=True)
    copied = []
    for name in SNAPSHOT_FILES:
        if (run_dir / name).exists():
            shutil.copy2(run_dir / name, model_dir / name)
            copied.append(name)
    if (run_dir / "checkpoints" / "history.json").exists():
        shutil.copy2(run_dir / "checkpoints" / "history.json", model_dir / "history.json")
        copied.append("history.json")
    (model_dir / "SNAPSHOT.txt").write_text(f"snapshot of {run_dir} taken {time.strftime('%Y-%m-%d %H:%M')}\n"
                                            f"files: {', '.join(copied)}\n")
    print(f"copied {', '.join(copied)} -> {model_dir}")
    return model_dir


# ---------------------------------------------------------------------------- build
def _ensure_dataset(config: Dict, entry: Dict) -> None:
    """Regenerate a synthetic dataset when it is missing (the .npz files are not committed)."""
    path = config["data"].get("dataset")
    if path is None or Path(path).exists():
        return
    gen = entry.get("generate")
    if gen == "cloth":
        from neural_physics.data.cloth_sim import ClothConfig, generate
        print(f"[{entry['name']}] dataset {path} missing -> generating synthetic cloth data (a few minutes) ...")
        generate(episodes=int(entry.get("episodes", 8)), frames_per_episode=int(entry.get("frames", 2500)),
                 cfg=ClothConfig(), seed=int(entry.get("seed", 0))).save(path)
    elif gen == "gravity":
        from neural_physics.data.point_mass import generate
        generate().save(path)
    else:
        raise SystemExit(f"[{entry['name']}] dataset {path} not found and no `generate` rule given")


def build_model_page(entry: Dict, site: Path, root: Path) -> Dict:
    name = entry["name"]
    with open(root / entry["config"]) as fh:
        config = yaml.safe_load(fh)
    for key in ("dataset", "data_file", "external_file", "faces_file"):
        if key in config["data"] and not Path(config["data"][key]).is_absolute():
            config["data"][key] = str(root / config["data"][key])
    model_dir = root / entry["model_dir"]
    if not (model_dir / "model.pt").exists():
        raise SystemExit(f"[{name}] {model_dir / 'model.pt'} not found -- run `pdm run publish-model runs/{name} {entry['model_dir']}` first")
    _ensure_dataset(config, entry)

    ds = load_dataset(config["data"])
    _, test_ds = ds.split_segments(float(config["data"].get("test_fraction", 0.2)), int(config["data"].get("split_seed", 0)))
    model, pca, ext_pca = load_run(model_dir, n_verts=ds.n_verts)
    for warning in divergence_warnings(model, pca, config,
                                       expect_n_components=int(config.get("pca", {}).get("n_components", pca.n_components))):
        print(f"[{name}] warning: {warning}  (re-run `pdm run publish-model runs/{name} {entry['model_dir']}`?)")
    rt = config.get("runtime", {})
    seg = test_ds.segments[0]
    result = evaluate_sequence(model, pca, test_ds.positions, test_ds.external if ds.n_external > 0 else None, ds.dt,
                               test_ds.faces, ext_pca, start=int(seg[0]),
                               n_frames=min(int(rt.get("eval_frames", 1500)), int(seg[1] - seg[0])), clip=bool(rt.get("clip", True)))
    s = result.summary
    viewer_opts = entry.get("viewer", {})
    out = site / name / "index.html"
    export_viewer(result, out, title=entry.get("title", rt.get("title", name)),
                  subtitle=f"{pca.n_components} PCA bases · {model.net.n_linear_layers}-layer network · held-out episode · {s['n_frames']} frames",
                  units=rt.get("units", ds.meta.get("units", "")), external_label=rt.get("external_label", ds.meta.get("external", "external state")),
                  notes=rt.get("notes", ""), max_frames=viewer_opts.get("max_frames"), frame_stride=int(viewer_opts.get("frame_stride", 1)),
                  include_full_ground_truth=viewer_opts.get("full_ground_truth"))
    history = json.load(open(model_dir / "history.json"))["history"] if (model_dir / "history.json").exists() else []
    print(f"[{name}] viewer -> {out}  ({out.stat().st_size / 2**20:.1f} MB)   network {s['mean_vertex_error']['network']:.4g} "
          f"vs alpha/beta {s['mean_vertex_error']['alpha_beta']:.4g} vs PCA floor {s['mean_vertex_error']['pca']:.4g} {rt.get('units', '')}")
    return {"name": name, "title": entry.get("title", name), "description": entry.get("description", ""), "summary": s,
            "units": rt.get("units", ds.meta.get("units", "")), "bases": pca.n_components, "layers": model.net.n_linear_layers,
            "linear_skip": bool(getattr(model, "linear_skip", False)), "params": sum(p.numel() for p in model.net.parameters()),
            "epochs": history[-1]["epoch"] if history else None, "train_time_s": history[-1]["time_s"] if history else None,
            "val_ratio": (history[-1]["val_network"] / history[-1]["val_baseline"]) if history and history[-1].get("val_baseline") else None,
            "n_train_frames": int(ds.n_frames - test_ds.n_frames), "n_verts": int(ds.n_verts), "href": f"{name}/index.html"}


def build(pages_cfg_path, out_dir, root: Optional[Path] = None) -> Path:
    """Build the site. Paths inside the configs are resolved against `root` (default: the current directory,
    i.e. the project root when run through `pdm run`)."""
    root = Path(root or Path.cwd()).resolve()
    with open(pages_cfg_path) as fh:
        pcfg = yaml.safe_load(fh)
    site = Path(out_dir)
    if site.exists():
        shutil.rmtree(site)
    site.mkdir(parents=True)
    cards = [build_model_page(entry, site, root) for entry in pcfg.get("models", [])]
    (site / "index.html").write_text(render_index(pcfg, cards), encoding="utf-8")
    (site / ".nojekyll").write_text("")            # serve files starting with '_' / no Jekyll processing
    print(f"site written to {site}  ({sum(f.stat().st_size for f in site.rglob('*') if f.is_file()) / 2**20:.1f} MB)")
    return site


# ---------------------------------------------------------------------------- landing page
_CSS = """
:root{color-scheme:light;--surface:#fcfcfb;--plane:#f9f9f7;--ink:#0b0b0b;--ink-2:#52514e;--muted:#898781;--grid:#e1e0d9;
--border:rgba(11,11,11,.10);--net:#2a78d6;--base:#eb6834;--pca:#1baf7a}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){color-scheme:dark;--surface:#1a1a19;--plane:#0d0d0d;--ink:#fff;
--ink-2:#c3c2b7;--muted:#898781;--grid:#2c2c2a;--border:rgba(255,255,255,.10);--net:#3987e5;--base:#d95926;--pca:#199e70}}
*{box-sizing:border-box}body{margin:0;font:15px/1.55 system-ui,-apple-system,"Segoe UI",sans-serif;color:var(--ink);background:var(--plane)}
main{max-width:1100px;margin:0 auto;padding:40px 24px 64px}h1{font-size:28px;margin:0 0 6px}h2{font-size:20px;margin:36px 0 10px}
.lead{color:var(--ink-2);max-width:760px}a{color:var(--net)}code{font-size:.92em;background:var(--surface);border:1px solid var(--border);border-radius:4px;padding:1px 5px}
.card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:20px 22px;margin:18px 0}
.card h3{margin:0 0 4px;font-size:18px}.card .desc{color:var(--ink-2);margin:0 0 14px}
.tiles{display:flex;gap:10px;flex-wrap:wrap;margin:10px 0 14px}.tile{background:var(--plane);border:1px solid var(--border);border-radius:8px;padding:8px 12px;min-width:150px}
.tile .l{color:var(--ink-2);font-size:12px}.tile .v{font-weight:600;font-size:20px}.tile .v small{font-weight:400;font-size:11px;color:var(--muted);margin-left:3px}
.k{display:inline-block;width:10px;height:3px;border-radius:2px;vertical-align:middle;margin-right:6px}
.meta{color:var(--muted);font-size:13px;margin-top:8px}
iframe{width:100%;height:640px;border:1px solid var(--border);border-radius:8px;background:var(--surface);margin-top:14px}
.btn{display:inline-block;padding:7px 14px;border-radius:6px;background:var(--net);color:#fff;text-decoration:none;font-weight:600}
table{border-collapse:collapse;font-size:14px}td,th{padding:4px 10px;border-bottom:1px solid var(--grid);text-align:left}
footer{color:var(--muted);font-size:13px;margin-top:40px;border-top:1px solid var(--border);padding-top:14px}
"""


def _fmt(v, d=4):
    return "–" if v is None else f"{v:.{d}f}"


def render_index(pcfg: Dict, cards: List[Dict]) -> str:
    e = html.escape
    title = pcfg.get("title", "Subspace Neural Physics — trained models")
    repo = pcfg.get("repo", "")
    parts = [f"<!DOCTYPE html><html lang='en'><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'>"
             f"<title>{e(title)}</title><style>{_CSS}</style></head><body><main>",
             f"<h1>{e(title)}</h1>",
             "<p class='lead'>Re-implementation of <a href='https://doi.org/10.1145/3309486.3340245'>Subspace Neural Physics: Fast Data-Driven "
             "Interactive Simulation</a> (Holden, Duong, Datta, Nowrouzezahrai, SCA 2019). The simulated object is compressed with PCA, "
             "a linear model <code>z̄<sub>t</sub> = α⊙z<sub>t−1</sub> + β⊙(z<sub>t−1</sub> − z<sub>t−2</sub>)</code> predicts the next subspace "
             "state and a small feed-forward network corrects it from <code>[z̄<sub>t</sub>, z<sub>t−1</sub>, w<sub>t</sub>]</code>, where "
             "<code>w<sub>t</sub></code> is the external state (wind, obstacles, …). The viewers below roll the trained model out over a "
             "held-out episode the network has never seen, next to the ground-truth simulation and the α/β-only model.</p>"]
    if pcfg.get("intro"):
        parts.append(f"<p class='lead'>{e(pcfg['intro'])}</p>")
    if repo:
        parts.append(f"<p><a href='{e(repo)}'>Source code on GitHub</a></p>")
    parts.append("<h2>Models</h2>")
    for c in cards:
        s, u = c["summary"], c["units"]
        m, ratio = s["mean_vertex_error"], s["mean_vertex_error"]["network"] / max(s["mean_vertex_error"]["alpha_beta"], 1e-12)
        trained = ""
        if c["epochs"]:
            trained = f"trained {c['epochs']} epochs" + (f" in {c['train_time_s'] / 60:.0f} min" if c["train_time_s"] else "")
            if c["val_ratio"]:
                trained += f" · held-out 32-frame error {c['val_ratio']:.2f}× α/β"
        parts.append(
            f"<section class='card' id='{e(c['name'])}'><h3>{e(c['title'])}</h3>"
            + (f"<p class='desc'>{e(c['description'])}</p>" if c["description"] else "")
            + "<div class='tiles'>"
            f"<div class='tile'><div class='l'><i class='k' style='background:var(--net)'></i>network · mean vertex error</div><div class='v'>{_fmt(m['network'])}<small>{e(u)}</small></div></div>"
            f"<div class='tile'><div class='l'><i class='k' style='background:var(--base)'></i>α/β only</div><div class='v'>{_fmt(m['alpha_beta'])}<small>{e(u)}</small></div></div>"
            f"<div class='tile'><div class='l'><i class='k' style='background:var(--pca)'></i>PCA floor ({c['bases']} bases)</div><div class='v'>{_fmt(m['pca'])}<small>{e(u)}</small></div></div>"
            f"<div class='tile'><div class='l'>network / α/β</div><div class='v'>{ratio:.2f}×</div></div>"
            f"<div class='tile'><div class='l'>roll-out</div><div class='v'>{s['n_frames']}<small>frames @ {1 / s['dt']:.0f} fps</small></div></div>"
            "</div>"
            f"<a class='btn' href='{e(c['href'])}'>Open interactive viewer →</a>"
            f"<div class='meta'>{c['n_verts']} vertices · {c['bases']} PCA bases · {c['layers']}-layer network"
            + (" + linear skip" if c["linear_skip"] else "") + f" · {c['params']:,} parameters · {c['n_train_frames']:,} training frames"
            + (f" · {trained}" if trained else "") + "</div>"
            f"<iframe src='{e(c['href'])}' loading='lazy' title='{e(c['title'])} viewer'></iframe></section>")
    parts.append("<h2>Reading the viewer</h2><table><tr><th>ground truth</th><td>the original simulation frames (or their PCA reconstruction when the full positions are not embedded)</td></tr>"
                 "<tr><th><i class='k' style='background:var(--net)'></i>network</th><td>the model rolled out from the first two ground-truth frames, coloured by per-vertex error</td></tr>"
                 "<tr><th><i class='k' style='background:var(--base)'></i>α/β only</th><td>the linear initial model alone (Eq. 1) — what the network has to improve on</td></tr>"
                 "<tr><th><i class='k' style='background:var(--pca)'></i>PCA floor</th><td>error of the PCA reconstruction of the ground truth — the best any model in this subspace can do</td></tr></table>")
    parts.append(f"<footer>Built {time.strftime('%Y-%m-%d %H:%M UTC', time.gmtime())} by <code>pdm run pages-build</code>. "
                 "Viewers are self-contained (three.js r128, MIT).</footer></main></body></html>")
    return "".join(parts)


# ---------------------------------------------------------------------------- cli
def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="build the static site from configs/pages.yml")
    b.add_argument("pages_config", nargs="?", default="configs/pages.yml")
    b.add_argument("--out", default="site")
    b.add_argument("--root", default=None, help="project root the config paths are relative to (default: cwd)")
    s = sub.add_parser("snapshot", help="copy a finished run's model into models/<name>/")
    s.add_argument("run_dir")
    s.add_argument("model_dir")
    args = p.parse_args(argv)
    try:
        if args.cmd == "snapshot":
            snapshot(args.run_dir, args.model_dir)
        else:
            build(args.pages_config, args.out, args.root)
    except ArtifactMismatch as exc:
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    sys.exit(main())
