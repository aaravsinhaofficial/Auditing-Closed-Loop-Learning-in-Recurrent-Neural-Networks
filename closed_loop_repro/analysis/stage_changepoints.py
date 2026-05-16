from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from closed_loop_repro.config import load_config
from closed_loop_repro.io import ensure_dir


def analyze_stage_changepoints(
    results: str | Path = "results/raw",
    out: str | Path = "results/processed",
    experiment_glob: str = "*",
    min_segment: int = 20,
    stride: int = 2,
) -> dict[str, Path]:
    out = ensure_dir(out)
    rows = []
    for path in sorted(Path(results).glob(f"{experiment_glob}/seed_*/timeseries.csv")):
        experiment = path.parts[-3]
        if experiment.startswith("smoke_"):
            continue
        df = pd.read_csv(path)
        if "closed_test_loss" not in df:
            continue
        row = _analyze_one(df["closed_test_loss"].to_numpy(dtype=float), min_segment=min_segment, stride=stride)
        row["experiment"] = experiment
        row["seed"] = int(path.parts[-2].split("_")[-1])
        row["kind"], row["setting"] = _classify_experiment(experiment)
        rows.append(row)
    frame = pd.DataFrame(rows)
    summary = _summary(frame)
    detail_path = out / "stage_changepoint_details.csv"
    summary_path = out / "stage_changepoint_summary.csv"
    frame.to_csv(detail_path, index=False)
    summary.to_csv(summary_path, index=False)
    return {"details_csv": detail_path, "summary_csv": summary_path}


def _analyze_one(loss: np.ndarray, min_segment: int, stride: int) -> dict[str, Any]:
    finite = np.isfinite(loss) & (loss > 0)
    if np.sum(finite) < 3 * min_segment:
        return _invalid("too_few_finite_points")
    y_full = np.log(np.maximum(loss, 1e-12))
    idx = np.arange(len(y_full))[finite]
    y = y_full[finite]
    if stride > 1:
        idx = idx[::stride]
        y = y[::stride]
    if len(y) < 3 * min_segment:
        return _invalid("too_few_points_after_stride")

    x = idx.astype(float)
    segment_sse = _segment_sse_matrix(x, y)
    n = len(y)
    best = (float("inf"), None, None)
    for b1 in range(min_segment, n - 2 * min_segment):
        b2_candidates = np.arange(b1 + min_segment, n - min_segment)
        if b2_candidates.size == 0:
            continue
        values = segment_sse[0, b1] + segment_sse[b1, b2_candidates] + segment_sse[b2_candidates, n]
        local = int(np.argmin(values))
        value = float(values[local])
        if value < best[0]:
            best = (value, b1, int(b2_candidates[local]))
    if best[1] is None or best[2] is None:
        return _invalid("no_valid_split")

    b1, b2 = int(best[1]), int(best[2])
    one_segment_sse = float(segment_sse[0, n])
    two_segment_sse, two_b = _best_two_segment_sse(segment_sse, n, min_segment)
    slopes = [
        _segment_slope(x[:b1], y[:b1]),
        _segment_slope(x[b1:b2], y[b1:b2]),
        _segment_slope(x[b2:], y[b2:]),
    ]
    stage1_fast = slopes[0] < -0.02
    stage2_slow = abs(slopes[1]) < max(abs(slopes[0]) * 0.35, 0.004)
    stage3_reaccelerates = slopes[2] < slopes[1] - 0.002
    return {
        "valid": True,
        "boundary1": int(idx[b1]),
        "boundary2": int(idx[b2]),
        "one_segment_sse": one_segment_sse,
        "two_segment_sse": two_segment_sse,
        "three_segment_sse": float(best[0]),
        "best_two_segment_boundary": int(idx[two_b]) if two_b is not None else np.nan,
        "three_vs_one_sse_reduction": _relative_reduction(one_segment_sse, float(best[0])),
        "three_vs_two_sse_reduction": _relative_reduction(two_segment_sse, float(best[0])),
        "stage1_slope": float(slopes[0]),
        "stage2_slope": float(slopes[1]),
        "stage3_slope": float(slopes[2]),
        "stage1_fast": bool(stage1_fast),
        "stage2_slow": bool(stage2_slow),
        "stage3_reaccelerates": bool(stage3_reaccelerates),
        "claim_C2_changepoint_three_stage": bool(stage1_fast and stage2_slow and stage3_reaccelerates),
    }


def _invalid(reason: str) -> dict[str, Any]:
    return {
        "valid": False,
        "invalid_reason": reason,
        "claim_C2_changepoint_three_stage": False,
    }


def _segment_sse_matrix(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    n = len(y)
    matrix = np.full((n + 1, n + 1), np.inf, dtype=float)
    px = np.r_[0.0, np.cumsum(x)]
    py = np.r_[0.0, np.cumsum(y)]
    px2 = np.r_[0.0, np.cumsum(x * x)]
    pxy = np.r_[0.0, np.cumsum(x * y)]
    py2 = np.r_[0.0, np.cumsum(y * y)]
    for start in range(n):
        ends = np.arange(start + 2, n + 1)
        count = ends - start
        sx = px[ends] - px[start]
        sy = py[ends] - py[start]
        sx2 = px2[ends] - px2[start]
        sxy = pxy[ends] - pxy[start]
        sy2 = py2[ends] - py2[start]
        denom = count * sx2 - sx * sx
        slope = np.divide(count * sxy - sx * sy, denom, out=np.zeros_like(denom, dtype=float), where=np.abs(denom) > 1e-12)
        intercept = (sy - slope * sx) / count
        sse = sy2 + count * intercept**2 + slope**2 * sx2 + 2 * intercept * slope * sx - 2 * intercept * sy - 2 * slope * sxy
        matrix[start, ends] = np.maximum(sse, 0.0)
    return matrix


def _best_two_segment_sse(segment_sse: np.ndarray, n: int, min_segment: int) -> tuple[float, int | None]:
    candidates = np.arange(min_segment, n - min_segment)
    if candidates.size == 0:
        return float("nan"), None
    values = segment_sse[0, candidates] + segment_sse[candidates, n]
    idx = int(np.argmin(values))
    return float(values[idx]), int(candidates[idx])


def _segment_slope(x: np.ndarray, y: np.ndarray) -> float:
    if len(y) < 2:
        return float("nan")
    centered = x - np.mean(x)
    denom = float(np.sum(centered**2))
    if denom <= 1e-12:
        return 0.0
    return float(np.sum(centered * (y - np.mean(y))) / denom)


def _relative_reduction(baseline: float, fitted: float) -> float:
    if not np.isfinite(baseline) or abs(baseline) <= 1e-12:
        return float("nan")
    return float((baseline - fitted) / baseline)


def _classify_experiment(experiment: str) -> tuple[str, str]:
    if experiment == "original_double_integrator_full":
        return "original", "original"
    if experiment.startswith("robustness_"):
        return "robustness", experiment.removeprefix("robustness_")
    if experiment.startswith("generalization_"):
        return "generalization", experiment.removeprefix("generalization_")
    if experiment.startswith("tradeoff_"):
        return "tradeoff", experiment.removeprefix("tradeoff_")
    return "other", experiment


def _summary(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    rows = []
    for (kind, setting), group in frame.groupby(["kind", "setting"], dropna=True):
        valid = group[group["valid"] == True]  # noqa: E712
        rows.append(
            {
                "kind": kind,
                "setting": setting,
                "n": int(len(group)),
                "valid_n": int(len(valid)),
                "changepoint_three_stage_support": float(group["claim_C2_changepoint_three_stage"].fillna(False).mean()),
                "median_boundary1": float(pd.to_numeric(valid.get("boundary1", pd.Series(dtype=float)), errors="coerce").median()),
                "median_boundary2": float(pd.to_numeric(valid.get("boundary2", pd.Series(dtype=float)), errors="coerce").median()),
                "median_three_vs_two_sse_reduction": float(
                    pd.to_numeric(valid.get("three_vs_two_sse_reduction", pd.Series(dtype=float)), errors="coerce").median()
                ),
                "median_stage1_slope": float(pd.to_numeric(valid.get("stage1_slope", pd.Series(dtype=float)), errors="coerce").median()),
                "median_stage2_slope": float(pd.to_numeric(valid.get("stage2_slope", pd.Series(dtype=float)), errors="coerce").median()),
                "median_stage3_slope": float(pd.to_numeric(valid.get("stage3_slope", pd.Series(dtype=float)), errors="coerce").median()),
            }
        )
    return pd.DataFrame(rows).sort_values(["kind", "setting"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Run segmented-regression stage changepoint analysis.")
    parser.add_argument("--config", default=None)
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--out", default="results/processed")
    parser.add_argument("--experiment-glob", default="*")
    parser.add_argument("--min-segment", type=int, default=20)
    parser.add_argument("--stride", type=int, default=2)
    args = parser.parse_args()
    config = load_config(args.config) if args.config else {}
    paths = analyze_stage_changepoints(
        config.get("results", args.results),
        config.get("out", args.out),
        experiment_glob=config.get("experiment_glob", args.experiment_glob),
        min_segment=int(config.get("min_segment", args.min_segment)),
        stride=int(config.get("stride", args.stride)),
    )
    print(paths)


if __name__ == "__main__":
    main()
