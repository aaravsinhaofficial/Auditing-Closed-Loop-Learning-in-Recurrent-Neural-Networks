from __future__ import annotations

import argparse
from pathlib import Path

from closed_loop_repro.config import load_config
from closed_loop_repro.io import write_table
from closed_loop_repro.training.experiment import run_pair_experiment


def run_seed_sweep(config: dict) -> list[dict]:
    seeds = config.get("seeds", [config.get("seed", 0)])
    if isinstance(seeds, dict):
        seeds = list(range(int(seeds.get("start", 0)), int(seeds.get("stop", 1))))
    output_dir = config.get("output_dir", "results/raw")
    rows = []
    for seed in seeds:
        run_cfg = dict(config)
        run_cfg["seed"] = int(seed)
        run_cfg.setdefault("experiment_name", "seed_sweep")
        result = run_pair_experiment(run_cfg, output_dir=output_dir)
        rows.append(result["metrics"])
    summary_path = Path(output_dir) / config.get("experiment_name", "seed_sweep") / "sweep_summary.csv"
    write_table(rows, summary_path)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paired closed/open experiments across seeds.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    rows = run_seed_sweep(load_config(args.config))
    print(f"completed {len(rows)} seeds")


if __name__ == "__main__":
    main()
