"""Console entry points (registered in pyproject.toml, also reachable through `pdm run ...`).

    np-datagen-cloth    -o bin/cloth.npz [--episodes 8 --frames 2500 --nx 25 --ny 17]
    np-datagen-gravity  -o bin/gravity.npz
    np-train            configs/cloth.yml [--epochs N] [--max-time S] [--eval-only]
"""
from __future__ import annotations

import argparse
import time

from neural_physics.data.cloth_sim import ClothConfig, generate as generate_cloth
from neural_physics.data.point_mass import main as datagen_gravity  # noqa: F401  (re-exported entry point)
from neural_physics.pipeline import main as train  # noqa: F401  (re-exported entry point)


def datagen_cloth(argv=None):
    """Generate synthetic flag-on-a-pole cloth data (positions + wind) as a `SimulationDataset` .npz."""
    p = argparse.ArgumentParser(description=datagen_cloth.__doc__)
    p.add_argument("-o", "--output", default="bin/cloth.npz", help="output .npz (SimulationDataset)")
    p.add_argument("-e", "--episodes", type=int, default=8, help="independent wind episodes (kept as segments)")
    p.add_argument("-f", "--frames", type=int, default=2500, help="frames per episode")
    p.add_argument("--nx", type=int, default=25, help="vertices along the flag")
    p.add_argument("--ny", type=int, default=17, help="vertices top to bottom")
    p.add_argument("--fps", type=int, default=60)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    cfg = ClothConfig(nx=args.nx, ny=args.ny, fps=args.fps)
    print(f"simulating {args.episodes} x {args.frames} frames of a {cfg.nx}x{cfg.ny} cloth ...")
    t0 = time.time()
    ds = generate_cloth(episodes=args.episodes, frames_per_episode=args.frames, cfg=cfg, seed=args.seed)
    ds.save(args.output)
    print(f"saved {ds.n_frames} frames, {ds.n_verts} vertices, {len(ds.segments)} segments to {args.output} "
          f"({time.time() - t0:.0f}s)")
