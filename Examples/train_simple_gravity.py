"""Smoke test on a point mass under gravity (`neural_physics.data.point_mass`).

NOTE: this problem is solved exactly by the linear initial model (alpha = beta = 1, residual = g dt^2),
so the network has nothing to learn -- a run that "works" here says nothing about the pipeline.
Use Examples/train_cloth.py for a meaningful test.

    pdm run train-gravity            (= python Examples/train_simple_gravity.py)
"""
import argparse
from pathlib import Path

import yaml

from neural_physics.data.point_mass import generate
from neural_physics.pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=ROOT / "configs" / "gravity.yml")
    p.add_argument("--epochs", type=int, default=None)
    args = p.parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    data_path = ROOT / config["data"]["dataset"]
    if not data_path.exists():
        print(f"generating {data_path} ...")
        generate().save(data_path)
    config["data"]["dataset"] = str(data_path)
    config["output_dir"] = str(ROOT / config.get("output_dir", "runs/gravity"))
    run(config, epochs=args.epochs)


if __name__ == "__main__":
    main()
