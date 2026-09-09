"""Training procedure of the paper (Sec. 5.4, Algorithm 1) with mini-batches.

For every optimiser step: draw `batch_size` random windows of `window_size` frames, perturb the two
initial states with Gaussian noise, roll the model out over the window feeding its own predictions
back in, and minimise the mean absolute position + velocity error of the whole window (averaged over
the mini-batch). AmsGrad, lr 1e-4, learning-rate decay 0.999 per epoch, ~100 epochs.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

from neural_physics.train.subspace_neural_physics import SubspaceNeuralPhysics, loss_fn
from neural_physics.utils.data_preprocess import sample_window_batch, window_starts


@dataclass
class TrainConfig:
    epochs: int = 100
    steps_per_epoch: Optional[int] = None      # default: number of windows / batch_size
    batch_size: int = 16
    window_size: int = 32
    lr: float = 1e-4
    lr_decay: float = 0.999                    # multiplied into the lr once per epoch
    noise_std: float = 0.01
    noise_mode: str = "relative"               # "relative" (x component std) or "absolute" (paper: 0.01)
    grad_clip: Optional[float] = None
    eval_windows: int = 128                    # held-out windows for the per-epoch evaluation
    eval_horizon: int = 32
    eval_clip: bool = True                     # evaluate with the Sec. 8.1 clipping (as at runtime)
    max_time_s: Optional[float] = None         # stop early after this wall-clock budget
    checkpoint_dir: Optional[str] = None
    log_dir: Optional[str] = None              # tensorboard run directory (optional)
    device: str = "auto"
    seed: int = 0
    verbose: bool = True

    def resolve_device(self) -> torch.device:
        if self.device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(self.device)


def _to_device(x, device) -> torch.Tensor:
    return torch.as_tensor(np.asarray(x) if not isinstance(x, torch.Tensor) else x, dtype=torch.float32).to(device)


@torch.no_grad()
def evaluate_short_horizon(model: SubspaceNeuralPhysics, subspace_z: torch.Tensor, subspace_w: torch.Tensor,
                           segments=None, horizon: int = 32, n_windows: int = 128, clip: bool = True,
                           seed: int = 123) -> Dict[str, float]:
    """Mean absolute subspace error of `horizon`-frame roll-outs started from ground-truth frames (no noise),
    for the full model and for the alpha/beta-only initial model. Lower is better; the ratio tells whether
    the network improves on Eq. 1."""
    was_training = model.training
    model.eval()
    starts = window_starts(subspace_z.shape[1], horizon, segments)
    g = torch.Generator(device="cpu").manual_seed(seed)
    n_windows = min(n_windows, len(starts))
    pick = torch.as_tensor(starts)[torch.randperm(len(starts), generator=g)[:n_windows]].to(subspace_z.device)
    idx = pick[:, None] + torch.arange(horizon, device=subspace_z.device)[None]
    zw, ww = subspace_z.T[idx], subspace_w.T[idx]
    out = {}
    for name, use_net in (("network", True), ("baseline", False)):
        pred = model.rollout(zw[:, 0], zw[:, 1], ww, clip=clip, use_network=use_net)
        out[name] = float((pred[:, 2:] - zw[:, 2:]).abs().mean())
    if was_training:
        model.train()
    return out


def train(model: SubspaceNeuralPhysics, subspace_z, subspace_w, segments=None, val_z=None, val_w=None,
          val_segments=None, cfg: Optional[TrainConfig] = None) -> List[Dict]:
    """Train `model.net` with Algorithm 1. `model.fit_linear` must have been called (or alpha/beta set) before.
    @param subspace_z: (n_components x n_frames) training data in the subspace
    @param subspace_w: (n_external x n_frames) external state (may have 0 rows)
    @param segments: (k x 2) episode boundaries of the training data (windows never cross them)
    @param val_*: optional held-out data for the per-epoch evaluation
    @return history: one dict per epoch
    """
    cfg = cfg or TrainConfig()
    device = cfg.resolve_device()
    torch.manual_seed(cfg.seed)
    gen = torch.Generator(device=device).manual_seed(cfg.seed)

    model = model.to(device).train()
    Z, W = _to_device(subspace_z, device), _to_device(subspace_w, device)
    if W.ndim == 1:
        W = W[None]
    has_val = val_z is not None
    if has_val:
        Zv, Wv = _to_device(val_z, device), _to_device(val_w, device)
    starts = window_starts(Z.shape[1], cfg.window_size, segments)
    steps_per_epoch = cfg.steps_per_epoch or max(1, len(starts) // cfg.batch_size)
    dt = float(model.dt)

    sigma = cfg.noise_std * model.z_std if cfg.noise_mode == "relative" else torch.full_like(model.z_std, cfg.noise_std)

    optimizer = torch.optim.Adam(model.net.parameters(), lr=cfg.lr, amsgrad=True)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=cfg.lr_decay)
    writer = None
    if cfg.log_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            writer = SummaryWriter(cfg.log_dir)
        except Exception as exc:  # tensorboard is optional
            print(f"tensorboard logging disabled: {exc}")
    ckpt_dir = Path(cfg.checkpoint_dir) if cfg.checkpoint_dir else None
    if ckpt_dir:
        ckpt_dir.mkdir(parents=True, exist_ok=True)

    if cfg.verbose:
        print(f"training on {device}: {len(starts)} windows of {cfg.window_size}, batch {cfg.batch_size}, "
              f"{steps_per_epoch} steps/epoch, {cfg.epochs} epochs, noise {cfg.noise_std} ({cfg.noise_mode}), "
              f"network {model.net.n_linear_layers} linear layers")
        base = evaluate_short_horizon(model, Zv, Wv, val_segments, cfg.eval_horizon, cfg.eval_windows, cfg.eval_clip) if has_val else None
        if base:
            print(f"  before training: held-out {cfg.eval_horizon}-frame error  network {base['network']:.5f}   "
                  f"alpha/beta only {base['baseline']:.5f}")

    history: List[Dict] = []
    best_val, t_start, step = math.inf, time.time(), 0
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for _ in range(steps_per_epoch):
            zw, ww = sample_window_batch(Z, W, starts, cfg.batch_size, cfg.window_size, gen)
            r0 = torch.randn(zw.shape[0], zw.shape[2], generator=gen, device=device) * sigma
            r1 = torch.randn(zw.shape[0], zw.shape[2], generator=gen, device=device) * sigma
            z_star = model.rollout(zw[:, 0] + r0, zw[:, 1] + r1, ww)
            loss = loss_fn(z_star, zw, dt)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.net.parameters(), cfg.grad_clip)
            optimizer.step()
            losses.append(loss.item())
            step += 1
            if writer:
                writer.add_scalar("train/loss", loss.item(), step)
        scheduler.step()

        rec = {"epoch": epoch + 1, "step": step, "loss": float(np.mean(losses)), "lr": optimizer.param_groups[0]["lr"],
               "time_s": time.time() - t_start}
        if has_val:
            rec.update({f"val_{k}": v for k, v in evaluate_short_horizon(
                model, Zv, Wv, val_segments, cfg.eval_horizon, cfg.eval_windows, cfg.eval_clip).items()})
            if writer:
                writer.add_scalar("val/network", rec["val_network"], epoch + 1)
                writer.add_scalar("val/baseline", rec["val_baseline"], epoch + 1)
        history.append(rec)
        if cfg.verbose:
            msg = f"epoch {epoch + 1:4d}  loss {rec['loss']:.5f}  lr {rec['lr']:.2e}  {rec['time_s']:6.0f}s"
            if has_val:
                msg += f"  | held-out {cfg.eval_horizon}-frame error: network {rec['val_network']:.5f}  " \
                       f"alpha/beta {rec['val_baseline']:.5f}  ratio {rec['val_network'] / max(rec['val_baseline'], 1e-12):.2f}"
            print(msg, flush=True)
        if ckpt_dir:
            model.save(ckpt_dir / "last.pt")
            score = rec.get("val_network", rec["loss"])
            if score < best_val:
                best_val = score
                model.save(ckpt_dir / "best.pt")
            with open(ckpt_dir / "history.json", "w") as fh:
                json.dump({"config": asdict(cfg), "history": history}, fh, indent=1)
        if cfg.max_time_s is not None and time.time() - t_start > cfg.max_time_s:
            if cfg.verbose:
                print(f"stopping: wall-clock budget of {cfg.max_time_s:.0f}s reached")
            break
    if writer:
        writer.close()
    model.eval()
    return history


@torch.no_grad()
def linear_residual_reference(model: SubspaceNeuralPhysics, subspace_z, subspace_w, segments=None, val_z=None, val_w=None,
                              val_segments=None, horizon: int = 32, n_windows: int = 128, clip: bool = True) -> Dict[str, float]:
    """Reference point for training: replace Phi by the closed-form *linear* least-squares fit of the residual
    z_t - z_bar_t on [z_bar_t, z_{t-1}, w_t] and evaluate it with `evaluate_short_horizon`. A trained network
    should at least reach this ratio (in practice it does so quickly only with `linear_skip=True`).
    Returns {'linear': error, 'baseline': error, 'r2': mean R^2 of the fit on the training data}."""
    device = model.z_mean.device
    Z, W = _to_device(subspace_z, device), _to_device(subspace_w, device)
    segs = np.asarray(segments if segments is not None else [(0, Z.shape[1])]).reshape(-1, 2)
    feats, targets = [], []
    for s, e in segs:
        if e - s < 3:
            continue
        z, w = Z[:, s:e].T, W[:, s:e].T
        z_bar = model.initial_model(z[1:-1], z[:-2])
        parts = [(z_bar - model.z_mean) / model.z_std, (z[1:-1] - model.z_mean) / model.z_std]
        if model.n_external > 0:
            parts.append((w[2:] - model.w_mean) / model.w_std)
        feats.append(torch.cat(parts, 1))
        targets.append((z[2:] - z_bar) / model.out_scale)
    F = torch.cat(feats).double()
    F = torch.cat([F, torch.ones(F.shape[0], 1, dtype=F.dtype, device=device)], 1)
    T = torch.cat(targets).double()
    coef = torch.linalg.lstsq(F, T).solution                      # (in+1, u)
    r2 = 1 - ((T - F @ coef) ** 2).sum(0) / ((T - T.mean(0)) ** 2).sum(0).clamp_min(1e-12)

    ref = SubspaceNeuralPhysics(model.n_components, model.n_external, n_layers=2, hidden_multiplier=model.hidden_multiplier,
                                dt=float(model.dt), linear_skip=True).to(device)
    ref.load_state_dict({k: v for k, v in model.state_dict().items() if not k.startswith("net.")}, strict=False)
    for p in ref.net.parameters():
        p.zero_()
    ref.net.skip.weight.copy_(coef[:-1].T.float())
    ref.net.skip.bias.copy_(coef[-1].float())
    zv = _to_device(val_z, device) if val_z is not None else Z
    wv = _to_device(val_w, device) if val_w is not None else W
    out = evaluate_short_horizon(ref, zv, wv, val_segments if val_z is not None else segments, horizon, n_windows, clip)
    return {"linear": out["network"], "baseline": out["baseline"], "r2": float(r2.mean())}
