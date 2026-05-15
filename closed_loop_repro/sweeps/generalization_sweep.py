from __future__ import annotations

import argparse
from pathlib import Path

from closed_loop_repro.config import deep_update, load_config
from closed_loop_repro.io import write_table
from closed_loop_repro.progress import log
from closed_loop_repro.training.experiment import run_pair_experiment


def run_generalization_sweep(config: dict) -> list[dict]:
    base = config.get("base", {})
    variants = config.get("variants", [])
    seeds = config.get("seeds", [0])
    if isinstance(seeds, dict):
        seeds = list(range(int(seeds.get("start", 0)), int(seeds.get("stop", 1))))
    output_dir = config.get("output_dir", "results/raw")
    rows = []
    total = len(variants) * len(seeds)
    run_idx = 0
    log(f"generalization_sweep start experiment={config.get('experiment_name', 'generalization')} runs={total}")
    for variant in variants:
        name = variant.get("name", "variant")
        patch = variant.get("patch", {})
        for seed in seeds:
            run_idx += 1
            run_cfg = deep_update(base, patch)
            run_cfg["seed"] = int(seed)
            run_cfg["experiment_name"] = f"generalization_{name}"
            run_cfg["variant"] = name
            log(f"generalization_sweep run {run_idx}/{total} variant={name} seed={seed} started")
            result = run_pair_experiment(run_cfg, output_dir=output_dir)
            metrics = dict(result["metrics"])
            metrics["variant"] = name
            rows.append(metrics)
            log(
                f"generalization_sweep run {run_idx}/{total} variant={name} seed={seed} done "
                f"final_closed={metrics['final_closed_test_loss']:.6g} final_open={metrics['final_open_test_loss']:.6g}"
            )
    summary_path = Path(output_dir) / config.get("experiment_name", "generalization") / "generalization_summary.csv"
    write_table(rows, summary_path)
    log(f"generalization_sweep done wrote={summary_path}")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run architecture/task generalization sweep.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    rows = run_generalization_sweep(load_config(args.config))
    log(f"completed {len(rows)} generalization runs")


if __name__ == "__main__":
    main()
