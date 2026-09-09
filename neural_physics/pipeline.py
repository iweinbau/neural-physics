"""End-to-end pipeline of the paper (Fig. 2): data -> PCA -> alpha/beta + network training -> runtime.

    python -m neural_physics.pipeline configs/cloth.yml [--epochs N] [--max-time SECONDS] [--eval-only]

The YAML config has five blocks (all keys optional unless stated):

    output_dir: runs/cloth
    data:
      dataset: bin/cloth.npz            # SimulationDataset (.npz), or the legacy pair:
      data_file: bin/flag_simulation.npy   # (frames x 3c) or (frames, c, 3)
      external_file: bin/flag_wind.npy     # (frames x e)
      faces_file: bin/flag.obj             # optional, for rendering
      dt: 0.0166667                        # required with the legacy pair
      test_fraction: 0.2                   # held out by episode (segments) or as the tail of a single sequence
    pca:
      n_components: 64                     # paper: 64 / 128 / 256
      external_components: null            # null = use the raw external state, int = PCA-compress it
      max_frames: 10000                    # frames used to build the basis (subsampled)
    model:
      n_layers: 10                         # linear layers incl. input and output (paper: 10)
      hidden_multiplier: 1.5
      linear_skip: false                   # not in the paper: zero-initialised linear input->output path
    train:                                  # any field of neural_physics.train.trainer.TrainConfig
      epochs: 100
      lr: 0.0001
      noise_std: 0.01
      noise_mode: relative
    runtime:
      eval_frames: 1500                    # frames of the first held-out episode to roll out
      viewer: viewer.html                  # written into output_dir
      clip: true                           # Sec. 8.1 clipping
      title: Flag
      units: m
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import fields
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import yaml

from neural_physics.artifacts import (ArtifactMismatch, basis_fingerprint, check_compatible,
                                       divergence_warnings, load_run)
from neural_physics.core_math.pca import PCA
from neural_physics.data.dataset import SimulationDataset
from neural_physics.runtime.export import export_viewer
from neural_physics.runtime.rollout import evaluate_sequence
from neural_physics.train.subspace_neural_physics import SubspaceNeuralPhysics
from neural_physics.train.trainer import TrainConfig, linear_residual_reference, train


def load_dataset(data_cfg: Dict) -> SimulationDataset:
    if "dataset" in data_cfg:
        return SimulationDataset.load(data_cfg["dataset"])
    if "dt" not in data_cfg:
        raise ValueError("data.dt is required when loading the legacy data_file / external_file pair")
    return SimulationDataset.from_legacy_npy(data_cfg["data_file"], data_cfg["external_file"], float(data_cfg["dt"]),
                                             data_cfg.get("faces_file"))


def run(config: Dict, epochs: Optional[int] = None, max_time: Optional[float] = None, eval_only: bool = False,
        verbose: bool = True, refit_basis: bool = False) -> Dict:
    t0 = time.time()
    out_dir = Path(config.get("output_dir", "runs/default"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log = print if verbose else (lambda *a, **k: None)

    # ------------------------------------------------------------------ 1. data
    ds = load_dataset(config["data"])
    train_ds, test_ds = ds.split_segments(float(config["data"].get("test_fraction", 0.2)), int(config["data"].get("split_seed", 0)))
    log(f"[data] {ds.n_frames} frames, {ds.n_verts} vertices, {ds.n_external} external dims, {len(ds.segments)} segments, "
        f"dt {ds.dt:.5f}  ->  train {train_ds.n_frames} / test {test_ds.n_frames} frames")

    # ------------------------------------------------------------------ 2. PCA (Sec. 5.1) and 3. model
    pca_cfg, model_cfg = config.get("pca", {}), config.get("model", {})
    u = int(pca_cfg.get("n_components", 64))
    X_train, X_test = train_ds.X(), test_ds.X()
    W_train, W_test = train_ds.Y(), test_ds.Y()
    ckpt = out_dir / "checkpoints" / "best.pt"
    model = None

    if eval_only:
        # Here the artifacts on disk are the truth, not the config: load model and basis as a pair and verify
        # they belong together (a basis left over from an earlier run used to fail deep inside the roll-out).
        model_file = "checkpoints/best.pt" if ckpt.exists() else "model.pt"
        if refit_basis:
            # Rescue a run whose pca.npz is missing or left over from an earlier configuration. The basis is
            # fully determined by dataset + split + n_components, so it can be re-created -- but only for the
            # *current* dataset and split, which is why this is opt-in.
            probe = SubspaceNeuralPhysics.load(out_dir / model_file)
            PCA(probe.n_components).fit(X_train, max_frames=pca_cfg.get("max_frames", 10000)).save(out_dir / "pca.npz")
            if pca_cfg.get("external_components") and ds.n_external > 0:
                PCA(int(pca_cfg["external_components"])).fit(W_train).save(out_dir / "pca_external.npz")
            log(f"[pca]  re-fitted a {probe.n_components}-component basis for {out_dir / model_file} "
                f"(only valid if the dataset and split are unchanged)")
        model, pca, ext_pca = load_run(out_dir, model_file=model_file, n_verts=ds.n_verts)
        u = pca.n_components
        log(f"[model] loaded {out_dir / model_file}: {u} PCA components, basis {basis_fingerprint(pca)}")
        for warning in divergence_warnings(model, pca, config, expect_n_components=int(pca_cfg.get("n_components", u))):
            log(f"[warn]  {warning}")
    else:
        pca = PCA(u).fit(X_train, max_frames=pca_cfg.get("max_frames", 10000))
        ext_pca = (PCA(int(pca_cfg["external_components"])).fit(W_train, max_frames=pca_cfg.get("max_frames", 10000))
                   if pca_cfg.get("external_components") and ds.n_external > 0 else None)
        # Write the basis *before* training starts. The checkpoints written every epoch are only usable
        # together with it, so an interrupted run must never leave the previous run's basis behind.
        pca.save(out_dir / "pca.npz")
        if ext_pca is not None:
            ext_pca.save(out_dir / "pca_external.npz")

    rec_train, rec_test = pca.reconstruction_error(X_train), pca.reconstruction_error(X_test)
    log(f"[pca]  {u} bases explain {pca.explained_variance_ratio.sum() * 100:.2f}% of the variance; "
        f"reconstruction RMS vertex error train {rec_train.mean():.4g} / test {rec_test.mean():.4g}")
    Z_train, Z_test = pca.encode(X_train), pca.encode(X_test)
    if ext_pca is not None:
        W_train, W_test = ext_pca.encode(W_train), ext_pca.encode(W_test)

    if model is None:
        model = SubspaceNeuralPhysics(u, W_train.shape[0], n_layers=int(model_cfg.get("n_layers", 10)),
                                      hidden_multiplier=float(model_cfg.get("hidden_multiplier", 1.5)), dt=ds.dt,
                                      linear_skip=bool(model_cfg.get("linear_skip", False)))
        model.fit_linear(Z_train, W_train, train_ds.segments)
        model.meta = {"basis": basis_fingerprint(pca), "pca": dict(pca_cfg), "model": dict(model_cfg),
                      "data": dict(config.get("data", {})), "name": config.get("name", "")}
    check_compatible(model, pca, n_verts=ds.n_verts, n_external=W_train.shape[0])

    a, b = model.alphas.numpy(), model.betas.numpy()
    z_std, res_std = model.z_std.numpy(), model.out_scale.numpy()
    tcfg_dict = {k: v for k, v in config.get("train", {}).items() if k in {f.name for f in fields(TrainConfig)}}
    tcfg = TrainConfig(**tcfg_dict)
    sigma = tcfg.noise_std * z_std if tcfg.noise_mode == "relative" else np.full_like(z_std, tcfg.noise_std)
    log(f"[model] alpha in [{a.min():.4f}, {a.max():.4f}]  beta in [{b.min():.4f}, {b.max():.4f}]  "
        f"(paper Fig. 14: alpha 0.995-1, beta 0.75-1)")
    log(f"[model] PCA component std: first {z_std[:3].round(4)} ... last {z_std[-3:].round(4)}")
    log(f"[model] residual (z - z_bar) std: first {res_std[:3].round(5)} ... last {res_std[-3:].round(5)}")
    log(f"[model] start noise sigma: first {sigma[:3].round(5)} ... last {sigma[-3:].round(5)}  "
        f"-> sigma / residual std: median {np.median(sigma / res_std):.2f}, max {np.max(sigma / res_std):.2f}")
    log(f"[model] network: {model.net.n_linear_layers} linear layers, hidden width {round(model.hidden_multiplier * u)}, "
        f"{sum(p.numel() for p in model.net.parameters())} parameters" + (", linear skip" if model.linear_skip else ""))
    ref = linear_residual_reference(model, Z_train, W_train, train_ds.segments, Z_test, W_test, test_ds.segments,
                                    tcfg.eval_horizon, tcfg.eval_windows, bool(config.get("runtime", {}).get("clip", True)))
    log(f"[model] reference: a closed-form linear fit of the residual explains R^2 = {ref['r2']:.2f} of it and gives a "
        f"held-out {tcfg.eval_horizon}-frame error of {ref['linear']:.5f} vs alpha/beta {ref['baseline']:.5f} "
        f"(ratio {ref['linear'] / max(ref['baseline'], 1e-12):.2f}) -- the trained network should get below this")

    # ------------------------------------------------------------------ 4. training  (Sec. 5.4, Algorithm 1)
    history = []
    if not eval_only:
        if epochs is not None:
            tcfg.epochs = epochs
        if max_time is not None:
            tcfg.max_time_s = max_time
        tcfg.checkpoint_dir = str(out_dir / "checkpoints")
        tcfg.eval_clip = bool(config.get("runtime", {}).get("clip", True))
        if tcfg.log_dir is None and config.get("train", {}).get("tensorboard", False):
            tcfg.log_dir = str(out_dir / "tensorboard")
        tcfg.verbose = verbose
        history = train(model, Z_train, W_train, train_ds.segments, Z_test, W_test, test_ds.segments, tcfg)
        if ckpt.exists():
            model = SubspaceNeuralPhysics.load(ckpt)   # best held-out model
    model = model.cpu().eval()
    model.save(out_dir / "model.pt")                    # the basis was already written before training started

    # ------------------------------------------------------------------ 5. runtime  (Sec. 6 / 7)
    rt = config.get("runtime", {})
    seg = test_ds.segments[0]
    n_eval = int(rt.get("eval_frames", 1500))
    result = evaluate_sequence(model, pca, test_ds.positions, test_ds.external if ds.n_external > 0 else None, ds.dt,
                               test_ds.faces, ext_pca, start=int(seg[0]), n_frames=min(n_eval, int(seg[1] - seg[0])),
                               clip=bool(rt.get("clip", True)))
    s = result.summary
    log(f"[runtime] {s['n_frames']}-frame roll-out on held-out data: mean vertex error network {s['mean_vertex_error']['network']:.4g}, "
        f"alpha/beta only {s['mean_vertex_error']['alpha_beta']:.4g}, PCA floor {s['mean_vertex_error']['pca']:.4g}  "
        f"(ratio network/alpha-beta {s['mean_vertex_error']['network'] / max(s['mean_vertex_error']['alpha_beta'], 1e-12):.2f}); "
        f"finite: {s['finite']}")
    viewer = out_dir / rt.get("viewer", "viewer.html")
    export_viewer(result, viewer, title=rt.get("title", config.get("name", "Subspace Neural Physics")),
                  subtitle=f"{u} PCA bases · {model.net.n_linear_layers}-layer network · held-out episode · "
                           f"{s['n_frames']} frames", units=rt.get("units", ds.meta.get("units", "")),
                  external_label=rt.get("external_label", ds.meta.get("external", "external state")),
                  notes=rt.get("notes", ""), max_frames=rt.get("viewer_max_frames"))
    log(f"[runtime] viewer written to {viewer}")

    summary = {"config": config, "pca": {"n_components": u, "explained_variance_ratio": float(pca.explained_variance_ratio.sum()),
                                         "reconstruction_rms_train": float(rec_train.mean()), "reconstruction_rms_test": float(rec_test.mean())},
               "alpha_range": [float(a.min()), float(a.max())], "beta_range": [float(b.min()), float(b.max())],
               "history": history, "runtime": s, "viewer": str(viewer), "elapsed_s": time.time() - t0}
    with open(out_dir / "summary.json", "w") as fh:
        json.dump(summary, fh, indent=1, default=float)
    return summary


def main(argv=None):
    p = argparse.ArgumentParser(description="Subspace Neural Physics: train and evaluate from a YAML config")
    p.add_argument("config")
    p.add_argument("--epochs", type=int, default=None, help="override train.epochs")
    p.add_argument("--max-time", type=float, default=None, help="stop training after this many seconds")
    p.add_argument("--eval-only", action="store_true", help="skip training, load checkpoints/best.pt and export the viewer")
    p.add_argument("--refit-basis", action="store_true",
                   help="with --eval-only: re-create a missing or stale pca.npz for the checkpoint "
                        "(valid only if the dataset and split are unchanged)")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    torch.set_num_threads(max(1, torch.get_num_threads()))
    try:
        run(config, epochs=args.epochs, max_time=args.max_time, eval_only=args.eval_only, verbose=not args.quiet,
            refit_basis=args.refit_basis)
    except ArtifactMismatch as exc:                     # a config/artifact problem, not a crash: no traceback
        raise SystemExit(f"error: {exc}")


if __name__ == "__main__":
    main()
