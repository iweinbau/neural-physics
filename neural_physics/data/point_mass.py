"""Point mass under gravity (the original `bin/datagen` toy), as a `SimulationDataset`.

Forward-Euler integration of a single particle: x_t = x_{t-1} + v_{t-1} dt, v_t = v_{t-1} + g dt.
Note that this is solved *exactly* by the paper's linear initial model (alpha = beta = 1, constant
residual g dt^2), so it only smoke-tests the pipeline -- the network has nothing to learn.
"""
from __future__ import annotations

import argparse
from typing import Sequence

import numpy as np

from .dataset import SimulationDataset

MASS = 1.0  # kg


def forward_euler_step(m: float, dt: float, v_prev: np.ndarray, x_prev: np.ndarray, external_force: np.ndarray):
    """Forward Euler step for a mass subject to external_force."""
    v = v_prev + (external_force / m) * dt
    x = x_prev + v_prev * dt
    return x, v


def generate(number: int = 1000, fps: int = 60, y_init: float = 1500.0, gravity_acc: float = 9.81,
             gravity_dir: Sequence[float] = (0.0, -1.0, 0.0)) -> SimulationDataset:
    """Simulate `number` frames of a falling point mass; the external state is the (constant) gravity force."""
    dt = 1.0 / fps
    force = gravity_acc * MASS * np.asarray(gravity_dir, dtype=np.float64)
    x = np.zeros((number, 3))
    v = np.zeros((number, 3))
    x[0] = [0.0, y_init, 0.0]
    external = np.tile(force, (number, 1))
    for i in range(1, number):
        x[i], v[i] = forward_euler_step(MASS, dt, v[i - 1], x[i - 1], external[i])
    meta = {"generator": "neural_physics.data.point_mass", "units": "m", "external": "gravity force (N) x,y,z"}
    return SimulationDataset(x[:, None, :].astype(np.float32), external.astype(np.float32), dt, None, None, meta)


def main(argv=None):
    """CLI: `np-datagen-gravity -o bin/gravity.npz` (also writes the legacy data.npy / external.npy pair)."""
    p = argparse.ArgumentParser(description="Generate data for the position of a mass subject to gravity.")
    p.add_argument("-n", "--number", type=int, default=1000, help="Number of frames to generate.")
    p.add_argument("-f", "--fps", type=int, default=60, help="Frames per second.")
    p.add_argument("-y", "--y-init", type=float, default=1500, help="Initial height of the mass.")
    p.add_argument("-g", "--gravity-acc", type=float, default=9.81, help="Gravitational acceleration")
    p.add_argument("-gd", "--gravity-dir", nargs=3, default=(0.0, -1.0, 0.0), type=float, help="Gravity direction")
    p.add_argument("-o", "--output", default="bin/gravity.npz", help="Output SimulationDataset (.npz)")
    p.add_argument("--legacy", nargs=2, metavar=("DATA_NPY", "EXTERNAL_NPY"), default=None,
                   help="Additionally write the old two-file layout (frames x 3 positions, frames x 3 forces)")
    args = p.parse_args(argv)
    ds = generate(args.number, args.fps, args.y_init, args.gravity_acc, args.gravity_dir)
    ds.save(args.output)
    if args.legacy:
        np.save(args.legacy[0], ds.positions.reshape(ds.n_frames, -1).astype(np.float64))
        np.save(args.legacy[1], ds.external.astype(np.float64))
    print(f"saved {ds.n_frames} frames to {args.output}")


if __name__ == "__main__":
    main()
