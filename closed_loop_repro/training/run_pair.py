from __future__ import annotations

import argparse

from closed_loop_repro.config import load_config
from closed_loop_repro.training.experiment import run_pair_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run one paired closed-loop/open-loop experiment.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.seed is not None:
        config["seed"] = args.seed
    result = run_pair_experiment(config, output_dir=args.output_dir)
    print(result["result_dir"])


if __name__ == "__main__":
    main()
