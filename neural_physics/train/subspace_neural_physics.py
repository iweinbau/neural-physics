"""Subspace Neural Physics model (Holden et al., SCA 2019, Sec. 5.2 / 5.3).

    z_bar_t = alpha ⊙ z_{t-1} + beta ⊙ (z_{t-1} - z_{t-2})                   (Eq. 1, linear initial model)
    z_t     = z_bar_t + Phi([z_bar_t, z_{t-1}, w_t])                          (Eq. 3, network correction)

`SubSpaceNeuralNetwork` is Phi. `SubspaceNeuralPhysics` bundles everything the runtime needs in one
`nn.Module` whose state_dict can be saved and reloaded: alpha, beta, the per-component normalisation
of the network inputs / outputs, the clipping ranges of Sec. 8.1 and the network itself, plus a
batched, differentiable `rollout` that is used both for training (Algorithm 1) and at runtime.
"""
from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from torch import nn

from neural_physics.utils.data_preprocess import initial_model_params_segments

# default frame time (60 fps); pass `dt` explicitly wherever possible
delta_t = 1 / 60


class SubSpaceNeuralNetwork(nn.Module):
    """The correction network Phi: a plain feed-forward ReLU MLP.

    Paper Sec. 5.3: "a standard feed-forward neural network with 10 layers, each layer (except the output
    layer) using the ReLU activation ... Excluding the input and output layers we set the number of hidden
    units at each layer to 1.5x the PCA basis size."  Here `n_hidden_layers` counts the hidden->hidden
    layers only, so the total number of linear layers is `n_hidden_layers + 2`; the paper's network is
    `n_hidden_layers=8`.
    """

    def __init__(self, num_components_X: int = 256, num_components_Y: int = 4, n_hidden_layers: int = 8,
                 hidden_multiplier: float = 1.5, linear_skip: bool = False):
        super().__init__()
        self.input_size = num_components_X * 2 + num_components_Y  # (z_bar, z_star_prev, w) concatenated
        hidden_size = max(1, round(hidden_multiplier * num_components_X))

        self.encode = nn.Sequential(nn.Linear(self.input_size, hidden_size), nn.ReLU())
        self.feed_forward = nn.Sequential(
            *[nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.ReLU()) for _ in range(n_hidden_layers)]
        )
        self.decode = nn.Linear(hidden_size, num_components_X)
        # Optional (NOT in the paper): a zero-initialised linear input->output path in parallel with the MLP.
        # A large part of the residual is linear in [z_bar, z_{t-1}, w_t]; a plain deep ReLU MLP trained
        # through the 32-frame roll-out picks that part up slowly, a linear layer picks it up in minutes.
        self.skip = None
        if linear_skip:
            self.skip = nn.Linear(self.input_size, num_components_X)
            nn.init.zeros_(self.skip.weight)
            nn.init.zeros_(self.skip.bias)

    @property
    def n_linear_layers(self) -> int:
        """Depth of the MLP path (input, hidden and output layers; the optional skip is not counted)."""
        return sum(isinstance(m, nn.Linear) for m in (self.encode, *self.feed_forward, self.decode) for m in m.modules())

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        """
        @param inputs: (batch_size, 2 * n_components + n_external)
        @return: (batch_size, n_components) residual in subspace coordinates
        """
        assert inputs.shape[-1] == self.input_size, \
            f"input_size: {inputs.shape[-1]} does not match expected input_size: {self.input_size}"
        hidden = self.encode(inputs)
        hidden = self.feed_forward(hidden)
        out = self.decode(hidden)
        if self.skip is not None:
            out = out + self.skip(inputs)
        return out


class SubspaceNeuralPhysics(nn.Module):
    """Linear initial model + network correction, with normalisation and clipping baked in."""

    def __init__(self, n_components: int, n_external: int, n_layers: int = 10, hidden_multiplier: float = 1.5,
                 dt: float = delta_t, linear_skip: bool = False):
        super().__init__()
        self.n_components, self.n_external = int(n_components), int(n_external)
        self.n_layers, self.hidden_multiplier, self.linear_skip = int(n_layers), float(hidden_multiplier), bool(linear_skip)
        self.net = SubSpaceNeuralNetwork(n_components, n_external, n_hidden_layers=max(0, n_layers - 2),
                                         hidden_multiplier=hidden_multiplier, linear_skip=linear_skip)
        u, v = self.n_components, self.n_external
        # Eq. 1 parameters (alpha = 1, beta = 1 is plain constant-velocity extrapolation)
        self.register_buffer("alphas", torch.ones(u))
        self.register_buffer("betas", torch.ones(u))
        # per-component normalisation of the network inputs and scale of its output
        self.register_buffer("z_mean", torch.zeros(u))
        self.register_buffer("z_std", torch.ones(u))
        self.register_buffer("w_mean", torch.zeros(max(v, 1)))
        self.register_buffer("w_std", torch.ones(max(v, 1)))
        self.register_buffer("out_scale", torch.ones(u))
        # Sec. 8.1 clipping ranges (training-data min / max)
        self.register_buffer("z_min", torch.full((u,), -float("inf")))
        self.register_buffer("z_max", torch.full((u,), float("inf")))
        self.register_buffer("w_min", torch.full((max(v, 1),), -float("inf")))
        self.register_buffer("w_max", torch.full((max(v, 1),), float("inf")))
        self.register_buffer("dt", torch.tensor(float(dt)))
        # free-form provenance written into the checkpoint: which basis / dataset / config this was trained on
        # (see neural_physics.artifacts -- it is what lets a mismatched model + basis pair be detected)
        self.meta: dict = {}

    # ------------------------------------------------------------------ fitting the non-network parts
    @torch.no_grad()
    def fit_linear(self, subspace_z, subspace_w=None, segments=None) -> "SubspaceNeuralPhysics":
        """Fit alpha / beta (Eq. 2), the normalisation statistics and the clipping ranges from training data.
        @param subspace_z: (n_components x n_frames) numpy array or tensor
        @param subspace_w: (n_external x n_frames) or None
        @param segments: optional (k x 2) [start, end) episode boundaries
        """
        z = torch.as_tensor(np.asarray(subspace_z.detach().cpu() if isinstance(subspace_z, torch.Tensor) else subspace_z),
                            dtype=torch.float32)
        alphas, betas = initial_model_params_segments(z.numpy(), segments if segments is not None else [(0, z.shape[1])])
        self.alphas.copy_(alphas.to(self.alphas.device))
        self.betas.copy_(betas.to(self.betas.device))

        dev = self.z_mean.device
        self.z_mean.copy_(z.mean(1).to(dev))
        self.z_std.copy_(z.std(1).clamp_min(1e-8).to(dev))
        self.z_min.copy_(z.min(1).values.to(dev))
        self.z_max.copy_(z.max(1).values.to(dev))

        # residual the network has to explain (teacher forced, inside segments only)
        segs = np.asarray(segments if segments is not None else [(0, z.shape[1])]).reshape(-1, 2)
        res = []
        for s, e in segs:
            if e - s >= 3:
                seg = z[:, s:e].T.to(dev)                                  # (T, u)
                res.append(seg[2:] - self.initial_model(seg[1:-1], seg[:-2]))
        res = torch.cat(res, 0)
        self.out_scale.copy_(res.std(0).clamp_min(1e-8))

        if subspace_w is not None and self.n_external > 0:
            w = torch.as_tensor(np.asarray(subspace_w.detach().cpu() if isinstance(subspace_w, torch.Tensor) else subspace_w),
                                dtype=torch.float32)
            self.w_mean.copy_(w.mean(1).to(dev))
            self.w_std.copy_(w.std(1).clamp_min(1e-8).to(dev))
            self.w_min.copy_(w.min(1).values.to(dev))
            self.w_max.copy_(w.max(1).values.to(dev))
        return self

    # ------------------------------------------------------------------ model pieces
    def initial_model(self, z_prev: torch.Tensor, z_prev2: torch.Tensor) -> torch.Tensor:
        """Eq. 1 for single frames (u,) or batches (..., u)."""
        return self.alphas * z_prev + self.betas * (z_prev - z_prev2)

    def residual(self, z_bar: torch.Tensor, z_prev: torch.Tensor, w: Optional[torch.Tensor]) -> torch.Tensor:
        """Phi([z_bar, z_{t-1}, w_t]) with normalised inputs and rescaled output. Inputs (..., u) / (..., v)."""
        parts = [(z_bar - self.z_mean) / self.z_std, (z_prev - self.z_mean) / self.z_std]
        if self.n_external > 0:
            parts.append((w - self.w_mean) / self.w_std)
        return self.net(torch.cat(parts, dim=-1)) * self.out_scale

    def clip_z(self, z: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(z, self.z_max), self.z_min)

    def clip_w(self, w: torch.Tensor) -> torch.Tensor:
        return torch.maximum(torch.minimum(w, self.w_max), self.w_min)

    def step(self, z_prev: torch.Tensor, z_prev2: torch.Tensor, w: Optional[torch.Tensor] = None,
             clip: bool = False, use_network: bool = True) -> torch.Tensor:
        """One integration step z_{t-2}, z_{t-1}, w_t -> z_t (Eq. 3). `clip` applies Sec. 8.1 clipping of the
        network inputs and of the result; `use_network=False` gives the alpha/beta-only prediction."""
        z_bar = self.initial_model(z_prev, z_prev2)
        if not use_network:
            return self.clip_z(z_bar) if clip else z_bar
        if clip:
            z_next = self.clip_z(z_bar) + self.residual(self.clip_z(z_bar), self.clip_z(z_prev),
                                                        self.clip_w(w) if w is not None else None)
            return self.clip_z(z_next)
        return z_bar + self.residual(z_bar, z_prev, w)

    def rollout(self, z0: torch.Tensor, z1: torch.Tensor, w: Optional[torch.Tensor], clip: bool = False,
                use_network: bool = True, n_steps: Optional[int] = None) -> torch.Tensor:
        """Auto-regressive roll-out (the inner loop of Algorithm 1), differentiable through time.
        @param z0, z1: (batch, u) first two (possibly noisy) states
        @param w: (batch, s, v) external state for every frame of the window (w[:, i] is used to predict z_i),
                  or None when the model has no external inputs (then pass `n_steps`)
        @param n_steps: window length s (defaults to w.shape[1])
        @return (batch, s, u) predictions z*_0 ... z*_{s-1} (z*_0 = z0, z*_1 = z1)
        """
        s = n_steps if n_steps is not None else (w.shape[1] if w is not None else None)
        if s is None:
            raise ValueError("rollout needs either w or n_steps to know the window length")
        preds = [z0, z1]
        z_prev2, z_prev = z0, z1
        for i in range(2, s):
            z_next = self.step(z_prev, z_prev2, w[:, i] if self.n_external > 0 else None, clip=clip, use_network=use_network)
            preds.append(z_next)
            z_prev2, z_prev = z_prev, z_next
        return torch.stack(preds, dim=1)

    def forward(self, z0, z1, w, clip=False, n_steps=None):
        return self.rollout(z0, z1, w, clip=clip, n_steps=n_steps)

    # ------------------------------------------------------------------ io
    def config(self) -> dict:
        return {"n_components": self.n_components, "n_external": self.n_external, "n_layers": self.n_layers,
                "hidden_multiplier": self.hidden_multiplier, "dt": float(self.dt), "linear_skip": self.linear_skip}

    def save(self, path) -> None:
        torch.save({"config": self.config(), "state_dict": self.state_dict(), "meta": dict(self.meta)}, path)

    @classmethod
    def load(cls, path, map_location="cpu") -> "SubspaceNeuralPhysics":
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        model = cls(**ckpt["config"])
        model.load_state_dict(ckpt["state_dict"])
        model.meta = dict(ckpt.get("meta") or {})
        return model.eval()


# ---------------------------------------------------------------------------- losses (Algorithm 1)
def loss_position(z_star: torch.Tensor, z: torch.Tensor) -> torch.Tensor:
    """L_pos: mean absolute error between predicted and ground-truth subspace positions."""
    return torch.abs(z_star - z).mean()


def loss_velocity(z_star: torch.Tensor, z_star_prev: torch.Tensor, z: torch.Tensor, z_prev: torch.Tensor,
                  dt: float = delta_t) -> torch.Tensor:
    """L_vel: mean absolute error between predicted and ground-truth subspace velocities."""
    return torch.abs((z_star - z_star_prev) / dt - (z - z_prev) / dt).mean()


def loss_fn(z_star: torch.Tensor, z: torch.Tensor, dt: float = delta_t) -> torch.Tensor:
    """Total loss of Algorithm 1 over a window: L_pos on frames 2..s plus L_vel on the velocities 1->2 ... s-1->s.
    @param z_star: predictions (..., s, u) including the two (noisy) initial frames
    @param z: ground truth (..., s, u)
    """
    return loss_position(z_star[..., 2:, :], z[..., 2:, :]) + \
        loss_velocity(z_star[..., 2:, :], z_star[..., 1:-1, :], z[..., 2:, :], z[..., 1:-1, :], dt)
