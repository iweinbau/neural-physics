"""End-to-end example on the synthetic flag-on-a-pole cloth.

    pdm run train-cloth                  # generate data (once), train 100 epochs, export runs/cloth/viewer.html
    pdm run train-cloth --max-time 1800  # stop training after 30 minutes
    pdm run train-cloth --eval-only      # re-export the viewer from the best checkpoint

Open runs/cloth/viewer.html in a browser to inspect the roll-out.
"""
import argparse
from pathlib import Path

import yaml

from neural_physics.data.cloth_sim import ClothConfig, generate
from neural_physics.pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=ROOT / "configs" / "cloth.yml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-time", type=float, default=None, help="wall-clock training budget in seconds")
    p.add_argument("--eval-only", action="store_true")
    p.add_argument("--episodes", type=int, default=8, help="episodes to simulate if the dataset does not exist yet")
    p.add_argument("--frames", type=int, default=2500, help="frames per episode for data generation")
    args = p.parse_args()

    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    data_path = ROOT / config["data"]["dataset"]
    if not data_path.exists():
        print(f"generating {data_path} ({args.episodes} episodes x {args.frames} frames) ...")
        generate(episodes=args.episodes, frames_per_episode=args.frames, cfg=ClothConfig()).save(data_path)
    config["data"]["dataset"] = str(data_path)
    config["output_dir"] = str(ROOT / config.get("output_dir", "runs/cloth"))
    run(config, epochs=args.epochs, max_time=args.max_time, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
