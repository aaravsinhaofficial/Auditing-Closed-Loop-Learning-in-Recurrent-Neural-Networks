from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from closed_loop_repro.analysis.make_claim_tables import collect_metrics
from closed_loop_repro.analysis.stage_changepoints import (
    _best_two_segment_sse,
    _relative_reduction,
    _segment_slope,
    _segment_sse_matrix,
)
from closed_loop_repro.io import ensure_dir


def make_supplemental_audit_tables(
    results: str | Path = "results/raw",
    processed: str | Path = "results/processed",
) -> dict[str, Path]:
    results = Path(results)
    processed = ensure_dir(processed)
    metrics = collect_metrics(results)
    if metrics.empty:
        metrics = _read_csv(processed / "recomputed_timeseries_metrics.csv")
    if metrics is None or metrics.empty:
        raise FileNotFoundError("No run summaries or recomputed timeseries metrics found.")

    outputs = {
        "run_accounting_csv": processed / "run_accounting.csv",
        "original_vs_reproduction_csv": processed / "original_vs_reproduction.csv",
        "artifact_manifest_csv": processed / "artifact_manifest.csv",
        "stage_sensitivity_csv": processed / "stage_sensitivity.csv",
        "tradeoff_components_csv": processed / "tradeoff_component_summary.csv",
        "core_seed_summary_csv": processed / "core_seed_summary.csv",
    }
    _run_accounting(metrics).to_csv(outputs["run_accounting_csv"], index=False)
    _original_vs_reproduction(metrics, processed).to_csv(outputs["original_vs_reproduction_csv"], index=False)
    _artifact_manifest().to_csv(outputs["artifact_manifest_csv"], index=False)
    existing_stage_sensitivity = _read_csv(outputs["stage_sensitivity_csv"])
    if existing_stage_sensitivity is not None and not existing_stage_sensitivity.empty:
        existing_stage_sensitivity.to_csv(outputs["stage_sensitivity_csv"], index=False)
    else:
        _stage_sensitivity(results).to_csv(outputs["stage_sensitivity_csv"], index=False)
    _tradeoff_components(metrics).to_csv(outputs["tradeoff_components_csv"], index=False)
    _core_seed_summary(metrics).to_csv(outputs["core_seed_summary_csv"], index=False)
    return outputs


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _run_accounting(metrics: pd.DataFrame) -> pd.DataFrame:
    rows = [
        ("Core C1-C3 reproduction", "original", "configs/original/double_integrator_full.yaml", "Figure 2; C1-C3 original-setting tests"),
        ("Robustness sweep", "robustness", "configs/robustness/maximal.yaml", "Figure 4; perturbation summary"),
        ("Targeted A1 tradeoff", "tradeoff", "configs/tradeoff/control_penalty_horizon.yaml", "Figure 7; A1 component table"),
        ("A2 generalization", "generalization", "configs/generalization/maximal.yaml; configs/generalization/path_integration_hard.yaml", "Figure 6; variant summary"),
    ]
    output = []
    for component, kind, config, produces in rows:
        n = int((metrics["kind"] == kind).sum())
        output.append({"component": component, "kind": kind, "paired_runs": n, "config": config, "produces": produces})
    output.append(
        {
            "component": "Total completed paired runs",
            "kind": "all",
            "paired_runs": int(len(metrics)),
            "config": "all above",
            "produces": "All claim and audit tables",
        }
    )
    return pd.DataFrame(output)


def _original_vs_reproduction(metrics: pd.DataFrame, processed: Path) -> pd.DataFrame:
    original = metrics[metrics["kind"] == "original"]
    variants = _read_csv(processed / "variant_summary.csv")
    stage_sensitivity = _read_csv(processed / "stage_sensitivity.csv")
    loss_stage_support = _loss_stage_support(stage_sensitivity, len(original))
    tracking = variants[variants["variant"] == "tracking_task"].iloc[0] if variants is not None and "tracking_task" in set(variants["variant"]) else None
    ring = variants[variants["variant"].astype(str).str.contains("ring", regex=False)] if variants is not None else pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "original_claim": "Closed/open divergence",
                "original_evidence": "Main double-integrator loss/gain figures",
                "our_test": "50 paired seeds, same initialization, deployed loss trajectory and peak signature",
                "result": (
                    f"post-initial peak {int(original['open_loop_post_initial_peak'].sum())}/{len(original)}; "
                    f"final-gap criterion {int(original['claim_C1_loss_divergence'].sum())}/{len(original)}; "
                    f"mean final gap {original['deployed_loss_gap'].mean():.4f}"
                ),
                "decision": "Reproduced",
            },
            {
                "original_claim": "Closed-loop stage structure",
                "original_evidence": "Spectral stages and visual learning-stage plots",
                "our_test": "Spectral-stage detector plus downstream loss changepoint observer",
                "result": (
                    f"spectral three-stage {int(original['claim_C2_spectral_three_stage'].sum())}/{len(original)}; "
                    f"loss changepoint strict support {loss_stage_support}/{len(original)}"
                ),
                "decision": "Reproduced spectrally; loss-only boundaries are diagnostic-dependent",
            },
            {
                "original_claim": "Coupled-system stability",
                "original_evidence": "Coupled eigenvalue/spectral argument",
                "our_test": "Coupled vs RNN-only spectra over 50 paired seeds",
                "result": f"{int(original['claim_C3_stability_transition'].sum())}/{len(original)} seeds cross the coupled stability boundary",
                "decision": "Reproduced in linearizable setting",
            },
            {
                "original_claim": "Broader motor-control applicability",
                "original_evidence": "Tracking-task extension",
                "our_test": "Tracking plus ring/path-integration variants",
                "result": _generalization_result(tracking, ring),
                "decision": "Partial; boundary condition identified",
            },
        ]
    )


def _loss_stage_support(stage_sensitivity: pd.DataFrame | None, fallback_n: int) -> int:
    if stage_sensitivity is None or stage_sensitivity.empty:
        return 0
    main = stage_sensitivity[stage_sensitivity["detector_variant"].astype(str).str.lower().str.contains("main", regex=False)]
    if main.empty or "strict_three_stage_support" not in main:
        return 0
    value = pd.to_numeric(main.iloc[0]["strict_three_stage_support"], errors="coerce")
    if not np.isfinite(value):
        return 0
    return int(round(float(value)))


def _generalization_result(tracking: pd.Series | None, ring: pd.DataFrame) -> str:
    tracking_text = "tracking unavailable" if tracking is None else f"tracking {int(round(tracking['c1_deployed_loss_support'] * tracking['n']))}/{int(tracking['n'])}"
    if ring.empty:
        return tracking_text
    support = int(round((ring["c1_deployed_loss_support"] * ring["n"]).sum()))
    total = int(ring["n"].sum())
    return f"{tracking_text}; ring-family {support}/{total}"


def _artifact_manifest() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "component": "Smoke validation",
                "file_or_script": "scripts/run_smoke.sh",
                "produces": "tiny raw outputs, claim tables, figures",
                "runtime": "<10 min on CPU/GPU",
                "expected_output": "results/raw/smoke_*; results/processed/*.csv; results/figures/*.png",
            },
            {
                "component": "Full audit",
                "file_or_script": "scripts/run_full_audit.sh",
                "produces": "core, robustness, generalization, tables, figures",
                "runtime": "about 16-18 h on one L40 for completed settings",
                "expected_output": "results/raw/*_full and sweep summaries",
            },
            {
                "component": "Targeted C2/A1/A2",
                "file_or_script": "scripts/run_targeted_c2_a1_a2.sh",
                "produces": "tradeoff, hard path-integration, changepoint reanalysis",
                "runtime": "about 8-10 h on one L40",
                "expected_output": "tradeoff_summary.csv; stage_changepoint_*.csv; variant_summary.csv",
            },
            {
                "component": "Claim tables",
                "file_or_script": "python -m closed_loop_repro.analysis.make_claim_tables --results results/raw --out results/processed",
                "produces": "claim-level reproducibility matrix",
                "runtime": "<1 min",
                "expected_output": "results/processed/claim_reproducibility_matrix.{csv,md}",
            },
            {
                "component": "Supplemental audit tables",
                "file_or_script": "python -m closed_loop_repro.analysis.supplemental_audit_tables --results results/raw --processed results/processed",
                "produces": "run accounting, original-vs-reproduction, A1 split, C2 sensitivity",
                "runtime": "<1 min",
                "expected_output": "results/processed/{run_accounting,original_vs_reproduction,tradeoff_component_summary,stage_sensitivity}.csv",
            },
            {
                "component": "Figure build",
                "file_or_script": "python -m closed_loop_repro.plotting.make_all_figures --config configs/figures/tmlr.yaml",
                "produces": "paper figures",
                "runtime": "<1 min",
                "expected_output": "results/figures/figure_*.png",
            },
        ]
    )


def _stage_sensitivity(results: Path) -> pd.DataFrame:
    variants = [
        ("main segmented", 20, 2, -0.02, 0.35, 0.004, 0.002),
        ("loose segmented", 20, 2, -0.01, 0.50, 0.006, 0.001),
        ("strict segmented", 20, 2, -0.04, 0.20, 0.002, 0.004),
        ("short-segment binary style", 10, 2, -0.02, 0.35, 0.004, 0.002),
        ("very loose reacceleration", 20, 2, -0.005, 0.75, 0.010, 0.0005),
    ]
    losses = []
    for path in sorted((results / "original_double_integrator_full").glob("seed_*/timeseries.csv")):
        frame = pd.read_csv(path)
        losses.append(frame["closed_test_loss"].to_numpy(dtype=float))
    rows = []
    for name, min_segment, stride, fast, slow_scale, slow_abs, reaccel in variants:
        analyses = [
            _stage_variant(loss, min_segment, stride, fast, slow_scale, slow_abs, reaccel)
            for loss in losses
        ]
        valid = [row for row in analyses if row is not None]
        slow = sum(bool(row["stage1_fast"] and row["stage2_slow"]) for row in valid)
        strict = sum(bool(row["stage1_fast"] and row["stage2_slow"] and row["stage3_reaccelerates"]) for row in valid)
        rows.append(
            {
                "detector_variant": name,
                "n": len(analyses),
                "valid_n": len(valid),
                "slow_phase_support": slow,
                "strict_three_stage_support": strict,
                "median_boundary1": float(np.nanmedian([row["boundary1"] for row in valid])),
                "median_boundary2": float(np.nanmedian([row["boundary2"] for row in valid])),
                "median_stage1_slope": float(np.nanmedian([row["stage1_slope"] for row in valid])),
                "median_stage2_slope": float(np.nanmedian([row["stage2_slope"] for row in valid])),
                "median_stage3_slope": float(np.nanmedian([row["stage3_slope"] for row in valid])),
            }
        )
    return pd.DataFrame(rows)


def _stage_variant(
    loss: np.ndarray,
    min_segment: int,
    stride: int,
    fast_threshold: float,
    slow_scale: float,
    slow_abs: float,
    reaccel_margin: float,
) -> dict[str, Any] | None:
    finite = np.isfinite(loss) & (loss > 0)
    if np.sum(finite) < 3 * min_segment:
        return None
    y_full = np.log(np.maximum(loss, 1e-12))
    idx = np.arange(len(y_full))[finite][::stride]
    y = y_full[finite][::stride]
    if len(y) < 3 * min_segment:
        return None
    x = idx.astype(float)
    segment_sse = _segment_sse_matrix(x, y)
    n = len(y)
    best = (float("inf"), None, None)
    for b1 in range(min_segment, n - 2 * min_segment):
        candidates = np.arange(b1 + min_segment, n - min_segment)
        if candidates.size == 0:
            continue
        values = segment_sse[0, b1] + segment_sse[b1, candidates] + segment_sse[candidates, n]
        local = int(np.argmin(values))
        value = float(values[local])
        if value < best[0]:
            best = (value, b1, int(candidates[local]))
    if best[1] is None or best[2] is None:
        return None
    b1, b2 = int(best[1]), int(best[2])
    two_segment_sse, _ = _best_two_segment_sse(segment_sse, n, min_segment)
    slopes = [
        _segment_slope(x[:b1], y[:b1]),
        _segment_slope(x[b1:b2], y[b1:b2]),
        _segment_slope(x[b2:], y[b2:]),
    ]
    return {
        "boundary1": int(idx[b1]),
        "boundary2": int(idx[b2]),
        "three_vs_two_sse_reduction": _relative_reduction(two_segment_sse, float(best[0])),
        "stage1_slope": float(slopes[0]),
        "stage2_slope": float(slopes[1]),
        "stage3_slope": float(slopes[2]),
        "stage1_fast": bool(slopes[0] < fast_threshold),
        "stage2_slow": bool(abs(slopes[1]) < max(abs(slopes[0]) * slow_scale, slow_abs)),
        "stage3_reaccelerates": bool(slopes[2] < slopes[1] - reaccel_margin),
    }


def _tradeoff_components(metrics: pd.DataFrame) -> pd.DataFrame:
    tradeoff = metrics[metrics["kind"] == "tradeoff"].copy()
    components = [
        ("union_loss_or_radius", "claim_C4_tradeoff_quantified", "conditional_tradeoff_fraction", "tradeoff_step_count"),
        ("loss_worsening_any", "claim_A1_loss_tradeoff", "conditional_loss_tradeoff_fraction", "loss_tradeoff_step_count"),
        ("radius_increase_any", "claim_A1_radius_tradeoff", "conditional_radius_tradeoff_fraction", "radius_tradeoff_step_count"),
        ("both_loss_and_radius", "claim_A1_both_tradeoff", "conditional_both_tradeoff_fraction", "both_tradeoff_step_count"),
        ("loss_only_exclusive", None, "conditional_loss_only_tradeoff_fraction", "loss_only_tradeoff_step_count"),
        ("radius_only_exclusive", None, "conditional_radius_only_tradeoff_fraction", "radius_only_tradeoff_step_count"),
    ]
    rows = []
    for label, support_col, fraction_col, count_col in components:
        fractions = pd.to_numeric(tradeoff.get(fraction_col, pd.Series(dtype=float)), errors="coerce")
        counts = pd.to_numeric(tradeoff.get(count_col, pd.Series(dtype=float)), errors="coerce")
        if support_col is None:
            support = (fractions >= 0.1) & (counts >= 3)
        else:
            support = _bool_series(tradeoff.get(support_col, pd.Series(False, index=tradeoff.index)))
        rows.append(
            {
                "component": label,
                "n": int(len(tradeoff)),
                "support_count": int(support.fillna(False).sum()),
                "support_fraction": float(support.fillna(False).mean()) if len(support) else float("nan"),
                "mean_conditional_fraction": float(fractions.mean()),
                "median_conditional_fraction": float(fractions.median()),
                "mean_step_count": float(counts.mean()),
            }
        )
    return pd.DataFrame(rows)


def _core_seed_summary(metrics: pd.DataFrame) -> pd.DataFrame:
    original = metrics[metrics["kind"] == "original"].copy()
    rows = []
    for column in ["final_closed_test_loss", "final_open_test_loss", "deployed_loss_gap", "trajectory_gain_distance"]:
        values = pd.to_numeric(original[column], errors="coerce").dropna()
        rows.append(
            {
                "metric": column,
                "n": int(len(values)),
                "mean": float(values.mean()),
                "sd": float(values.std(ddof=1)),
                "iqr_low": float(values.quantile(0.25)),
                "iqr_high": float(values.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def _bool_series(values: object) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def main() -> None:
    parser = argparse.ArgumentParser(description="Build supplemental audit tables from processed/raw outputs.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--processed", default="results/processed")
    args = parser.parse_args()
    print(make_supplemental_audit_tables(args.results, args.processed))


if __name__ == "__main__":
    main()
