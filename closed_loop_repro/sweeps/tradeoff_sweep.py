from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from closed_loop_repro.config import deep_update, load_config
from closed_loop_repro.io import write_table
from closed_loop_repro.progress import log
from closed_loop_repro.training.experiment import run_pair_experiment


def run_tradeoff_sweep(config: dict[str, Any]) -> list[dict[str, Any]]:
    base = config.get("base", {})
    seeds = _seeds(config.get("seeds", [0]))
    conditions = _conditions(config)
    output_dir = config.get("output_dir", "results/raw")
    rows = []
    total = len(conditions) * len(seeds)
    run_idx = 0
    log(f"tradeoff_sweep start experiment={config.get('experiment_name', 'tradeoff')} runs={total}")
    run_prefix = "smoke_tradeoff" if str(config.get("experiment_name", "")).startswith("smoke") else "tradeoff"
    for condition in conditions:
        name = condition.get("name", "condition")
        patch = condition.get("patch", {})
        for seed in seeds:
            run_idx += 1
            run_cfg = deep_update(base, patch)
            run_cfg["seed"] = int(seed)
            run_cfg["experiment_name"] = f"{run_prefix}_{name}"
            run_cfg["tradeoff_condition"] = name
            log(f"tradeoff_sweep run {run_idx}/{total} condition={name} seed={seed} started")
            result = run_pair_experiment(run_cfg, output_dir=output_dir)
            metrics = dict(result["metrics"])
            metrics["tradeoff_condition"] = name
            metrics.update(_summarize_tradeoff(result["records"], run_cfg))
            rows.append(metrics)
            log(
                f"tradeoff_sweep run {run_idx}/{total} condition={name} seed={seed} done "
                f"tradeoff_fraction={metrics.get('tradeoff_fraction', float('nan')):.4g} "
                f"short_loss={metrics.get('final_short_horizon_loss', float('nan')):.6g} "
                f"long_loss={metrics.get('final_long_horizon_loss', float('nan')):.6g}"
            )
    summary_path = Path(output_dir) / config.get("experiment_name", "tradeoff") / "tradeoff_summary.csv"
    write_table(rows, summary_path)
    log(f"tradeoff_sweep done wrote={summary_path}")
    return rows


def _seeds(config_seeds: Any) -> list[int]:
    if isinstance(config_seeds, dict):
        return list(range(int(config_seeds.get("start", 0)), int(config_seeds.get("stop", 1))))
    return [int(seed) for seed in config_seeds]


def _conditions(config: dict[str, Any]) -> list[dict[str, Any]]:
    if config.get("conditions"):
        return list(config["conditions"])
    penalties = config.get("control_penalties", [config.get("base", {}).get("training", {}).get("control_penalty", 0.005)])
    conditions = []
    for penalty in penalties:
        name = f"control_penalty_{float(penalty):g}".replace(".", "p")
        conditions.append({"name": name, "patch": {"training": {"control_penalty": float(penalty)}}})
    return conditions


def _summarize_tradeoff(records: list[dict[str, Any]], config: dict[str, Any]) -> dict[str, Any]:
    df = pd.DataFrame(records)
    horizons = _horizons(config)
    if len(horizons) < 2:
        return {"tradeoff_evaluable": False, "tradeoff_reason": "need_at_least_two_horizons"}
    short, long = min(horizons), max(horizons)
    short_col = f"closed_test_loss_T{short}"
    long_col = f"closed_test_loss_T{long}"
    if short_col not in df or long_col not in df:
        return {"tradeoff_evaluable": False, "tradeoff_reason": "missing_horizon_columns"}

    eval_df = df[["epoch", short_col, long_col, "closed_coupled_radius"]].dropna()
    if len(eval_df) < 4:
        return {"tradeoff_evaluable": False, "tradeoff_reason": "too_few_horizon_evaluations"}

    short_loss = eval_df[short_col].to_numpy(dtype=float)
    long_loss = eval_df[long_col].to_numpy(dtype=float)
    radius = eval_df["closed_coupled_radius"].to_numpy(dtype=float)
    finite = np.isfinite(short_loss) & np.isfinite(long_loss)
    if np.sum(finite) < 4:
        return {"tradeoff_evaluable": False, "tradeoff_reason": "nonfinite_horizon_losses"}
    short_loss = short_loss[finite]
    long_loss = long_loss[finite]
    radius = radius[finite]

    log_short = np.log(np.maximum(short_loss, 1e-12))
    log_long = np.log(np.maximum(long_loss, 1e-12))
    d_short = np.diff(log_short)
    d_long = np.diff(log_long)
    d_radius = np.diff(radius)

    min_improvement = float(config.get("analysis", {}).get("tradeoff_min_log_improvement", 0.005))
    min_worsening = float(config.get("analysis", {}).get("tradeoff_min_log_worsening", 0.005))
    myopic_improvement = d_short < -min_improvement
    long_worsening = d_long > min_worsening
    radius_worsening = d_radius > 0
    tradeoff_steps = myopic_improvement & (long_worsening | radius_worsening)

    improvement_count = int(np.sum(myopic_improvement))
    tradeoff_count = int(np.sum(tradeoff_steps))
    tradeoff_fraction = _safe_fraction(tradeoff_steps)
    conditional_tradeoff_fraction = float(tradeoff_count / improvement_count) if improvement_count else 0.0
    return {
        "tradeoff_evaluable": True,
        "short_horizon": int(short),
        "long_horizon": int(long),
        "horizon_evaluations": int(len(short_loss)),
        "final_short_horizon_loss": float(short_loss[-1]),
        "final_long_horizon_loss": float(long_loss[-1]),
        "short_horizon_loss_reduction": float(short_loss[0] - short_loss[-1]),
        "long_horizon_loss_reduction": float(long_loss[0] - long_loss[-1]),
        "peak_long_horizon_loss": float(np.max(long_loss)),
        "peak_long_to_initial_ratio": float(np.max(long_loss) / max(long_loss[0], 1e-12)),
        "tradeoff_step_count": tradeoff_count,
        "myopic_improvement_step_count": improvement_count,
        "tradeoff_fraction": tradeoff_fraction,
        "conditional_tradeoff_fraction": conditional_tradeoff_fraction,
        "short_long_delta_corr": _corr(-d_short, d_long),
        "short_radius_delta_corr": _corr(-d_short, d_radius),
        "claim_C4_tradeoff_quantified": bool(conditional_tradeoff_fraction >= 0.1 and tradeoff_count >= 3),
    }


def _horizons(config: dict[str, Any]) -> list[int]:
    horizons = config.get("training", {}).get("evaluation_horizons", [])
    return sorted({int(horizon) for horizon in horizons if int(horizon) > 0})


def _safe_fraction(values: np.ndarray) -> float:
    if values.size == 0:
        return float("nan")
    return float(np.mean(values))


def _corr(x: np.ndarray, y: np.ndarray) -> float:
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C4 short-vs-long horizon tradeoff sweep.")
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    rows = run_tradeoff_sweep(load_config(args.config))
    log(f"completed {len(rows)} tradeoff runs")


if __name__ == "__main__":
    main()
