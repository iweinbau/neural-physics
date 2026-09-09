"""Train on real flag data (see configs/flag.yml for the expected files and layout).

    pdm run train-flag [--epochs N] [--max-time SECONDS] [--eval-only]
"""
import argparse
from pathlib import Path

import yaml

from neural_physics.pipeline import run

ROOT = Path(__file__).resolve().parents[1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=ROOT / "configs" / "flag.yml")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--max-time", type=float, default=None)
    p.add_argument("--eval-only", action="store_true")
    args = p.parse_args()
    with open(args.config) as fh:
        config = yaml.safe_load(fh)
    for key in ("data_file", "external_file", "faces_file", "dataset"):
        if key in config["data"] and not Path(config["data"][key]).is_absolute():
            config["data"][key] = str(ROOT / config["data"][key])
    config["output_dir"] = str(ROOT / config.get("output_dir", "runs/flag"))
    missing = [config["data"][k] for k in ("data_file", "external_file", "dataset") if k in config["data"] and not Path(config["data"][k]).exists()]
    if missing:
        raise SystemExit("flag data not found: " + ", ".join(missing) + "\nexport your simulation to these files (see configs/flag.yml)")
    run(config, epochs=args.epochs, max_time=args.max_time, eval_only=args.eval_only)


if __name__ == "__main__":
    main()
