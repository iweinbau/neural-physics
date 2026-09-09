"""Synthetic "flag on a pole" training data.

A rectangular cloth is pinned along one edge to a vertical pole and driven by gravity and a
time-varying wind. The wind velocity vector is the *external state* y_t of the paper (Table 1,
Flag: "wind speed and direction"). The simulator is a small position-based-dynamics (PBD) cloth
[Mueller et al. 2007] written in numpy: distance constraints for stretch, shear and bending,
Gauss-Seidel projection over vertex-disjoint constraint groups, quadratic aerodynamic drag on the
per-vertex normals.

It is *not* meant to match Maya nCloth; it is meant to give a deformable object with contact-free
but strongly non-linear, externally driven dynamics so that the whole pipeline (PCA -> alpha/beta
-> network -> runtime) can be exercised end-to-end before real data is available.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from .dataset import SimulationDataset


@dataclass
class ClothConfig:
    nx: int = 25                 # vertices along the flag (pole -> free edge)
    ny: int = 17                 # vertices top -> bottom
    spacing: float = 0.05        # m between neighbouring vertices  (1.2 m x 0.8 m flag)
    mass: float = 0.0015         # kg per vertex (~0.6 kg/m^2)
    fps: int = 60
    substeps: int = 3
    iterations: int = 6          # PBD constraint iterations per substep
    k_stretch: float = 1.0
    k_shear: float = 0.6
    k_bend: float = 0.15
    damping: float = 0.4         # 1/s velocity damping
    air_density: float = 1.2
    drag_coeff: float = 1.3      # normal (form) drag
    tangential_drag: float = 0.05
    gravity: float = 9.81
    # wind process (per episode a mean speed / direction is drawn, then an Ornstein-Uhlenbeck
    # process wanders around it; gusts add jumps)
    wind_speed_range: Tuple[float, float] = (1.5, 9.0)
    wind_dir_spread_deg: float = 50.0
    wind_speed_std: float = 1.5
    wind_dir_std_deg: float = 25.0
    wind_tau: float = 2.0        # s, OU relaxation time
    gust_rate: float = 0.15      # gusts per second
    gust_strength: float = 3.0   # m/s


class ClothSim:
    """PBD cloth pinned to a pole along its left edge (x = 0)."""

    def __init__(self, cfg: ClothConfig):
        self.cfg = cfg
        nx, ny, h = cfg.nx, cfg.ny, cfg.spacing
        cols, rows = np.meshgrid(np.arange(nx), np.arange(ny))        # (ny, nx)
        self.n = nx * ny
        self.rest = np.stack([cols.ravel() * h, (ny - 1 - rows.ravel()) * h + 1.5, np.zeros(self.n)], 1)
        self.inv_mass = np.full(self.n, 1.0 / cfg.mass)
        self.inv_mass[cols.ravel() == 0] = 0.0                          # pinned to the pole
        self.faces = self._make_faces(nx, ny)
        self.groups = self._make_constraint_groups(nx, ny, h)
        self.reset()

    @staticmethod
    def _make_faces(nx: int, ny: int) -> np.ndarray:
        idx = np.arange(nx * ny).reshape(ny, nx)
        a, b, c, d = idx[:-1, :-1].ravel(), idx[:-1, 1:].ravel(), idx[1:, 1:].ravel(), idx[1:, :-1].ravel()
        return np.concatenate([np.stack([a, b, c], 1), np.stack([a, c, d], 1)]).astype(np.int64)

    def _make_constraint_groups(self, nx, ny, h) -> List[Tuple[np.ndarray, np.ndarray, float, float]]:
        """Return a list of (i, j, rest_length, stiffness) groups; edges inside a group share no vertex,
        so each group can be projected in one vectorised Gauss-Seidel sweep."""
        cfg, idx, groups = self.cfg, np.arange(nx * ny).reshape(ny, nx), []

        def add(i, j, rest, k):
            i, j = np.asarray(i).ravel(), np.asarray(j).ravel()
            if i.size:
                # per-iteration stiffness so the result is (roughly) independent of the iteration count
                kk = 1.0 - (1.0 - k) ** (1.0 / cfg.iterations)
                wi, wj = self.inv_mass[i][:, None], self.inv_mass[j][:, None]
                groups.append((i, j, rest, kk * wi / (wi + wj + 1e-12), kk * wj / (wi + wj + 1e-12)))

        for par in (0, 1):  # structural, horizontal / vertical
            add(idx[:, par:-1:2], idx[:, par + 1::2], h, cfg.k_stretch)
            add(idx[par:-1:2, :], idx[par + 1::2, :], h, cfg.k_stretch)
        for pr in (0, 1):   # shear diagonals (4 parity groups each)
            for pc in (0, 1):
                add(idx[pr:-1:2, pc:-1:2], idx[pr + 1::2, pc + 1::2], h * math.sqrt(2), cfg.k_shear)
                add(idx[pr:-1:2, pc + 1::2], idx[pr + 1::2, pc:-1:2], h * math.sqrt(2), cfg.k_shear)
        for m in range(4):  # bending (skip-one) horizontal / vertical, groups by index mod 4
            add(idx[:, m:-2:4], idx[:, m + 2::4], 2 * h, cfg.k_bend)
            add(idx[m:-2:4, :], idx[m + 2::4, :], 2 * h, cfg.k_bend)
        return groups

    def reset(self):
        self.x = self.rest.copy()
        self.v = np.zeros_like(self.x)

    def vertex_normals(self, x: np.ndarray) -> np.ndarray:
        """Area-weighted vertex normals (magnitude ~ vertex area)."""
        f = self.faces
        fn = 0.5 * np.cross(x[f[:, 1]] - x[f[:, 0]], x[f[:, 2]] - x[f[:, 0]])
        n = np.zeros_like(x)
        for k in range(3):
            np.add.at(n, f[:, k], fn / 3.0)
        return n

    def wind_force(self, wind: np.ndarray) -> np.ndarray:
        cfg = self.cfg
        n_area = self.vertex_normals(self.x)
        area = np.linalg.norm(n_area, axis=1, keepdims=True) + 1e-12
        n_hat = n_area / area
        u_rel = wind[None, :] - self.v
        u_n = np.sum(u_rel * n_hat, axis=1, keepdims=True)
        f_normal = 0.5 * cfg.air_density * cfg.drag_coeff * area * np.abs(u_n) * u_n * n_hat
        f_tan = cfg.tangential_drag * area * (u_rel - u_n * n_hat)
        return f_normal + f_tan

    def step(self, wind: np.ndarray, dt: float):
        cfg = self.cfg
        h = dt / cfg.substeps
        free = self.inv_mass > 0
        for _ in range(cfg.substeps):
            acc = self.wind_force(wind) * self.inv_mass[:, None]
            acc[:, 1] -= cfg.gravity
            self.v[free] += acc[free] * h
            self.v *= max(0.0, 1.0 - cfg.damping * h)
            p = self.x + self.v * h
            p[~free] = self.rest[~free]
            for _ in range(cfg.iterations):
                for i, j, rest, si, sj in self.groups:
                    d = p[j] - p[i]
                    ln = np.sqrt(np.einsum("ij,ij->i", d, d))[:, None] + 1e-12
                    d *= (ln - rest) / ln
                    p[i] += si * d
                    p[j] -= sj * d
            self.v = (p - self.x) / h
            self.x = p


class WindProcess:
    """Ornstein-Uhlenbeck wind speed and direction with occasional gusts."""

    def __init__(self, cfg: ClothConfig, rng: np.random.Generator):
        self.cfg, self.rng = cfg, rng
        lo, hi = cfg.wind_speed_range
        self.mean_speed = rng.uniform(lo, hi)
        self.mean_dir = math.radians(rng.uniform(-cfg.wind_dir_spread_deg, cfg.wind_dir_spread_deg))
        self.speed, self.dir, self.elev = self.mean_speed, self.mean_dir, 0.0
        self.gust = np.zeros(3)

    def sample(self, dt: float) -> np.ndarray:
        cfg, rng = self.cfg, self.rng
        a = dt / cfg.wind_tau
        self.speed += a * (self.mean_speed - self.speed) + cfg.wind_speed_std * math.sqrt(2 * a) * rng.normal()
        self.dir += a * (self.mean_dir - self.dir) + math.radians(cfg.wind_dir_std_deg) * math.sqrt(2 * a) * rng.normal()
        self.elev += a * (0.0 - self.elev) + math.radians(8.0) * math.sqrt(2 * a) * rng.normal()
        self.speed = max(0.0, self.speed)
        if rng.random() < cfg.gust_rate * dt:
            self.gust += rng.normal(0, cfg.gust_strength, 3) * np.array([1.0, 0.3, 1.0])
        self.gust *= math.exp(-dt / 0.6)
        ce = math.cos(self.elev)
        return np.array([self.speed * math.cos(self.dir) * ce, self.speed * math.sin(self.elev), self.speed * math.sin(self.dir) * ce]) + self.gust


def generate(episodes: int = 8, frames_per_episode: int = 2500, cfg: Optional[ClothConfig] = None,
             seed: int = 0, verbose: bool = True) -> SimulationDataset:
    """Simulate `episodes` independent wind episodes and return them as one `SimulationDataset`
    (episode boundaries are recorded in `segments`)."""
    cfg = cfg or ClothConfig()
    rng = np.random.default_rng(seed)
    sim = ClothSim(cfg)
    dt = 1.0 / cfg.fps
    positions = np.zeros((episodes * frames_per_episode, sim.n, 3), dtype=np.float32)
    external = np.zeros((episodes * frames_per_episode, 3), dtype=np.float32)
    segments = []
    for ep in range(episodes):
        sim.reset()
        wind = WindProcess(cfg, rng)
        s = ep * frames_per_episode
        for f in range(frames_per_episode):
            w = wind.sample(dt)
            sim.step(w, dt)
            positions[s + f] = sim.x
            external[s + f] = w
        segments.append((s, s + frames_per_episode))
        if verbose:
            print(f"  episode {ep + 1}/{episodes} done (mean wind {wind.mean_speed:.1f} m/s)")
    meta: Dict = {"generator": "neural_physics.data.cloth_sim", "units": "m", "external": "wind velocity (m/s) x,y,z",
                  "pinned": "column 0 (x = 0) fixed to the pole", "config": asdict(cfg), "seed": seed}
    return SimulationDataset(positions, external, dt, sim.faces, np.array(segments), meta)
