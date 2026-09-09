"""Runtime: auto-regressive simulation in the subspace (paper Sec. 6) and its evaluation (Sec. 7).

Given the first two subspace states and the external state for every frame, the model is rolled out
for the whole sequence (with the Sec. 8.1 clipping), decoded back to vertex positions with the PCA
basis and compared against the ground truth. The alpha/beta-only model is rolled out as well: it is the
reference the network has to beat, and the PCA reconstruction of the ground truth is the best any
model in this subspace can do.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional

import numpy as np
import torch

from neural_physics.artifacts import check_compatible
from neural_physics.core_math.pca import PCA
from neural_physics.train.subspace_neural_physics import SubspaceNeuralPhysics


@torch.no_grad()
def simulate(model: SubspaceNeuralPhysics, z0, z1, external, clip: bool = True, use_network: bool = True,
             n_steps: Optional[int] = None) -> np.ndarray:
    """Roll the model out over a whole sequence.
    @param z0, z1: (u,) first two subspace states
    @param external: (n_frames, v) external state per frame (frame i is used to predict z_i), or None
    @return (n_frames, u) subspace trajectory (frames 0 and 1 are z0, z1)
    """
    dev = model.z_mean.device
    model.eval()
    z0t = torch.as_tensor(np.asarray(z0), dtype=torch.float32, device=dev)[None]
    z1t = torch.as_tensor(np.asarray(z1), dtype=torch.float32, device=dev)[None]
    w = None if external is None else torch.as_tensor(np.asarray(external), dtype=torch.float32, device=dev)[None]
    n = n_steps if n_steps is not None else w.shape[1]
    return model.rollout(z0t, z1t, w, clip=clip, use_network=use_network, n_steps=n)[0].cpu().numpy()


@dataclass
class RolloutResult:
    """Everything needed to inspect a roll-out: subspace trajectories, decoded vertex positions and errors."""
    dt: float
    z_gt: np.ndarray                    # (n, u) ground-truth subspace trajectory
    z_pred: np.ndarray                  # (n, u) network roll-out
    z_base: np.ndarray                  # (n, u) alpha/beta-only roll-out
    x_gt: np.ndarray                    # (n, c, 3) ground-truth vertex positions
    x_pred: np.ndarray                  # (n, c, 3) decoded network roll-out
    x_base: np.ndarray                  # (n, c, 3) decoded alpha/beta-only roll-out
    x_recon: np.ndarray                 # (n, c, 3) PCA reconstruction of the ground truth
    external: Optional[np.ndarray]      # (n, v)
    faces: Optional[np.ndarray]
    pca_mean: np.ndarray                # (3c,)
    pca_basis: np.ndarray               # (u, 3c)
    summary: Dict = field(default_factory=dict)

    # per-frame mean vertex error (same units as the positions)
    def frame_error(self, which: str) -> np.ndarray:
        x = {"pred": self.x_pred, "base": self.x_base, "recon": self.x_recon}[which]
        return np.linalg.norm(x - self.x_gt, axis=2).mean(1)

    def vertex_error(self, which: str) -> np.ndarray:
        x = {"pred": self.x_pred, "base": self.x_base, "recon": self.x_recon}[which]
        return np.linalg.norm(x - self.x_gt, axis=2)      # (n, c)


def _decode(pca: PCA, z: np.ndarray) -> np.ndarray:
    """(n, u) -> (n, c, 3)"""
    x = pca.decode(z.T.astype(np.float64)).T      # (n, 3c)
    return x.reshape(x.shape[0], -1, 3).astype(np.float32)


def evaluate_sequence(model: SubspaceNeuralPhysics, pca: PCA, positions: np.ndarray, external: Optional[np.ndarray],
                      dt: float, faces: Optional[np.ndarray] = None, external_pca: Optional[PCA] = None,
                      start: int = 0, n_frames: Optional[int] = None, clip: bool = True) -> RolloutResult:
    """Roll the trained model out over a (held-out) sequence and decode everything to vertex positions.
    @param positions: (n, c, 3) ground-truth vertex positions of the sequence
    @param external: (n, e) raw external state; passed through `external_pca` if the model was trained on
                     PCA-compressed external states
    @param start, n_frames: sub-range of the sequence to simulate (the roll-out starts from frames start, start+1)
    """
    n_ext = None
    if external is not None and model.n_external > 0:
        n_ext = external_pca.n_components if external_pca is not None else np.asarray(external).shape[1]
    check_compatible(model, pca, n_verts=np.asarray(positions).shape[1], n_external=n_ext)
    n_total = positions.shape[0]
    end = n_total if n_frames is None else min(n_total, start + n_frames)
    pos = np.asarray(positions[start:end], dtype=np.float32)
    n = pos.shape[0]
    X = pos.reshape(n, -1).T                                   # (3c, n)
    z_gt = pca.encode(X).T.astype(np.float32)                  # (n, u)
    w = None
    if external is not None and model.n_external > 0:
        w_raw = np.asarray(external[start:end], dtype=np.float32)
        w = external_pca.encode(w_raw.T).T.astype(np.float32) if external_pca is not None else w_raw
    z_pred = simulate(model, z_gt[0], z_gt[1], w, clip=clip, use_network=True, n_steps=n)
    z_base = simulate(model, z_gt[0], z_gt[1], w, clip=clip, use_network=False, n_steps=n)

    res = RolloutResult(dt=dt, z_gt=z_gt, z_pred=z_pred, z_base=z_base, x_gt=pos, x_pred=_decode(pca, z_pred),
                        x_base=_decode(pca, z_base), x_recon=_decode(pca, z_gt),
                        external=None if external is None else np.asarray(external[start:end], dtype=np.float32),
                        faces=None if faces is None else np.asarray(faces), pca_mean=pca.mean[:, 0].astype(np.float32),
                        pca_basis=pca.U.astype(np.float32))
    e_pred, e_base, e_recon = res.frame_error("pred"), res.frame_error("base"), res.frame_error("recon")
    size = float(np.linalg.norm(pos.max((0, 1)) - pos.min((0, 1))))
    res.summary = {
        "n_frames": int(n), "n_verts": int(pos.shape[1]), "n_components": int(z_gt.shape[1]), "dt": float(dt),
        "object_size": size,
        "mean_vertex_error": {"network": float(e_pred.mean()), "alpha_beta": float(e_base.mean()), "pca": float(e_recon.mean())},
        "max_vertex_error": {"network": float(res.vertex_error("pred").max()), "alpha_beta": float(res.vertex_error("base").max()),
                             "pca": float(res.vertex_error("recon").max())},
        "mean_vertex_error_pct_of_size": {"network": float(e_pred.mean() / size * 100), "alpha_beta": float(e_base.mean() / size * 100),
                                          "pca": float(e_recon.mean() / size * 100)},
        "subspace_error": {"network": float(np.abs(z_pred - z_gt).mean()), "alpha_beta": float(np.abs(z_base - z_gt).mean())},
        "finite": bool(np.isfinite(z_pred).all()),
        "clipped": bool(clip),
    }
    return res
