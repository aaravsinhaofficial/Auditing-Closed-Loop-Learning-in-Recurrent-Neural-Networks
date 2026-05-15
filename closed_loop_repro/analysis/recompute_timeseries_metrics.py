from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml

from closed_loop_repro.analysis.gains import gain_distance
from closed_loop_repro.analysis.stages import detect_stages
from closed_loop_repro.analysis.statistics import bootstrap_ci
from closed_loop_repro.io import ensure_dir


def recompute_timeseries_metrics(results: str | Path = "results/raw", out: str | Path = "results/processed") -> dict[str, Path]:
    out = ensure_dir(out)
    rows = [_recompute_one(path) for path in _iter_timeseries(Path(results))]
    df = pd.DataFrame(row for row in rows if row is not None)
    metrics_path = out / "recomputed_timeseries_metrics.csv"
    stage_path = out / "stage_reanalysis.csv"
    claim_path = out / "recomputed_claim_matrix.csv"
    setting_path = out / "recomputed_setting_summary.csv"

    df.to_csv(metrics_path, index=False)
    _stage_columns(df).to_csv(stage_path, index=False)
    _claim_matrix(df).to_csv(claim_path, index=False)
    _setting_summary(df).to_csv(setting_path, index=False)
    return {
        "metrics_csv": metrics_path,
        "stage_csv": stage_path,
        "claim_csv": claim_path,
        "setting_csv": setting_path,
    }


def _iter_timeseries(root: Path) -> list[Path]:
    paths = []
    for path in sorted(root.glob("*/seed_*/timeseries.csv")):
        experiment = path.parts[-3]
        if experiment.startswith("smoke_") or experiment == "generalization_ring_gru":
            continue
        paths.append(path)
    return paths


def _recompute_one(path: Path) -> dict[str, Any] | None:
    df = pd.read_csv(path)
    if df.empty:
        return None
    config = _read_config(path.parent / "config.yaml")
    min_plateau = int(config.get("analysis", {}).get("min_plateau", 8))
    experiment = path.parts[-3]
    seed = int(path.parts[-2].split("_")[-1])
    kind, setting = _classify_experiment(experiment)

    closed_test = df["closed_test_loss"].to_numpy(dtype=float)
    open_test = df["open_test_loss"].to_numpy(dtype=float)
    closed_radius = df.get("closed_coupled_radius", pd.Series(np.nan, index=df.index)).to_numpy(dtype=float)
    stages = detect_stages(closed_test, closed_radius, min_plateau=min_plateau)

    final_closed = float(closed_test[-1])
    final_open = float(open_test[-1])
    finite_final_losses = bool(np.isfinite(final_closed) and np.isfinite(final_open))
    loss_gap = final_open - final_closed
    relative_gap = loss_gap / max(abs(final_closed), 1e-12) if finite_final_losses else float("nan")
    final_gain_distance = _final_gain_distance(df)
    open_peak = _safe_nanmax(open_test)
    initial_open = float(open_test[0]) if np.isfinite(open_test[0]) else 1e-12

    open_spike = bool(np.isfinite(open_peak) and open_peak > 2.0 * max(initial_open, final_open, 1e-12))
    recovered = bool(finite_final_losses and np.isfinite(closed_test[0]) and final_closed < closed_test[0])
    final_radius = float(closed_radius[-1])

    row = {
        "experiment": experiment,
        "kind": kind,
        "setting": setting,
        "seed": seed,
        "epochs": int(len(df)),
        "final_closed_test_loss": final_closed,
        "final_open_test_loss": final_open,
        "deployed_loss_gap": float(loss_gap),
        "deployed_loss_gap_relative_to_closed": float(relative_gap),
        "finite_final_losses": finite_final_losses,
        "peak_open_test_loss": open_peak,
        "peak_closed_test_loss": _safe_nanmax(closed_test),
        "trajectory_gain_distance": final_gain_distance,
        "open_loop_test_loss_spike": open_spike,
        "closed_recovered": recovered,
        "stage1_end": int(stages.stage1_end),
        "plateau_end": int(stages.plateau_end),
        "plateau_length": int(stages.as_dict()["plateau_length"]),
        "plateau_detected": bool(stages.plateau_detected),
        "plateau_exit_detected": bool(stages.plateau_exit_detected),
        "plateau_exit_reason": stages.plateau_exit_reason,
        "stability_crossing": np.nan if stages.stability_crossing is None else int(stages.stability_crossing),
        "stability_to_plateau_gap": np.nan
        if stages.stability_crossing is None
        else int(stages.plateau_end - stages.stability_crossing),
        "final_closed_coupled_radius": final_radius,
    }
    row["claim_C1_loss_divergence"] = bool(finite_final_losses and relative_gap > 0.05)
    row["claim_C1_gain_divergence"] = bool(np.isfinite(final_gain_distance) and final_gain_distance > 0.05)
    row["claim_C2_plateau_present"] = bool(finite_final_losses and stages.plateau_detected)
    row["claim_C2_three_stage"] = bool(finite_final_losses and stages.plateau_detected and stages.plateau_exit_detected)
    row["claim_C3_stability_transition"] = bool(np.isfinite(final_radius) and stages.stability_crossing is not None)
    row["claim_C4_proxy"] = bool(finite_final_losses and open_spike and recovered)
    return row


def _read_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _classify_experiment(experiment: str) -> tuple[str, str]:
    if experiment == "original_double_integrator_full":
        return "original", "original"
    if experiment.startswith("robustness_"):
        return "robustness", experiment.removeprefix("robustness_")
    if experiment.startswith("generalization_"):
        return "generalization", experiment.removeprefix("generalization_")
    return "other", experiment


def _final_gain_distance(df: pd.DataFrame) -> float:
    closed_gain = np.asarray([df.iloc[-1].get(f"closed_gain_{idx}", np.nan) for idx in range(8)], dtype=float)
    open_gain = np.asarray([df.iloc[-1].get(f"open_gain_{idx}", np.nan) for idx in range(8)], dtype=float)
    return gain_distance(np.nan_to_num(closed_gain), np.nan_to_num(open_gain))


def _safe_nanmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _stage_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "experiment",
        "kind",
        "setting",
        "seed",
        "stage1_end",
        "plateau_end",
        "plateau_length",
        "plateau_detected",
        "plateau_exit_detected",
        "plateau_exit_reason",
        "stability_crossing",
        "stability_to_plateau_gap",
    ]
    return df[columns].copy()


def _claim_matrix(df: pd.DataFrame) -> pd.DataFrame:
    rows = [
        _claim_row("C1", "Closed-loop/open-loop divergence", df, "claim_C1_loss_divergence", "deployed closed-loop loss gap >5%"),
        _claim_row(
            "C2",
            "Closed-loop stages",
            df,
            "claim_C2_three_stage",
            "plateau present plus non-fallback second stage boundary",
        ),
        _claim_row(
            "C2a",
            "Closed-loop slow-progress phase",
            df,
            "claim_C2_plateau_present",
            "early rapid improvement followed by slow-progress phase",
        ),
        _claim_row(
            "C3",
            "Coupled-system stability transition",
            df,
            "claim_C3_stability_transition",
            "finite coupled spectral radius crossing",
        ),
        _claim_row(
            "C4",
            "Short-term vs long-term tradeoff",
            df,
            "claim_C4_proxy",
            "proxy only: open-loop deployed spike plus closed-loop recovery",
        ),
    ]
    gen = df[df["kind"] == "generalization"]
    rows.append(
        _claim_row(
            "C5",
            "Generalization",
            gen,
            "claim_C1_loss_divergence",
            "generalization variants preserving deployed-loss divergence",
        )
    )
    return pd.DataFrame(rows)


def _claim_row(claim: str, description: str, df: pd.DataFrame, column: str, notes: str) -> dict[str, Any]:
    values = df[column].astype(float).to_numpy() if column in df else np.asarray([], dtype=float)
    mean, lo, hi = bootstrap_ci(values, n_boot=500)
    return {
        "claim": claim,
        "description": description,
        "n": int(len(values)),
        "support_fraction": mean,
        "ci_low": lo,
        "ci_high": hi,
        "notes": notes,
    }


def _setting_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (kind, setting), group in df.groupby(["kind", "setting"], dropna=True):
        rows.append(
            {
                "kind": kind,
                "setting": setting,
                "n": int(len(group)),
                "c1_loss_support": float(group["claim_C1_loss_divergence"].mean()),
                "c2_three_stage_support": float(group["claim_C2_three_stage"].mean()),
                "c2_plateau_support": float(group["claim_C2_plateau_present"].mean()),
                "c3_stability_support": float(group["claim_C3_stability_transition"].mean()),
                "c4_proxy_support": float(group["claim_C4_proxy"].mean()),
                "mean_final_closed_test_loss": float(pd.to_numeric(group["final_closed_test_loss"], errors="coerce").mean()),
                "mean_final_open_test_loss": float(pd.to_numeric(group["final_open_test_loss"], errors="coerce").mean()),
                "mean_deployed_loss_gap": float(pd.to_numeric(group["deployed_loss_gap"], errors="coerce").mean()),
                "stage_exit_reasons": ";".join(
                    f"{reason}:{count}" for reason, count in group["plateau_exit_reason"].value_counts().sort_index().items()
                ),
            }
        )
    return pd.DataFrame(rows).sort_values(["kind", "setting"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute audit metrics from saved per-seed time series.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--out", default="results/processed")
    args = parser.parse_args()
    paths = recompute_timeseries_metrics(args.results, args.out)
    print(paths)


if __name__ == "__main__":
    main()
