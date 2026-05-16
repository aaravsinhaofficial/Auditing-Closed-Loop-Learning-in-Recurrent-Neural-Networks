from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

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
    metrics = _read_csv(processed / "recomputed_timeseries_metrics.csv")
    if metrics is None:
        raise FileNotFoundError("Run recompute_timeseries_metrics before supplemental_audit_tables.")

    outputs = {
        "run_accounting_csv": processed / "run_accounting.csv",
        "original_vs_reproduction_csv": processed / "original_vs_reproduction.csv",
        "artifact_manifest_csv": processed / "artifact_manifest.csv",
        "stage_sensitivity_csv": processed / "stage_sensitivity.csv",
        "tradeoff_components_csv": processed / "tradeoff_component_summary.csv",
        "core_seed_summary_csv": processed / "core_seed_summary.csv",
        "open_loop_peak_csv": processed / "open_loop_peak_signature.csv",
        "stability_alignment_csv": processed / "stability_alignment_summary.csv",
        "loss_scale_csv": processed / "loss_scale_comparison.csv",
    }
    _run_accounting(metrics).to_csv(outputs["run_accounting_csv"], index=False)
    _original_vs_reproduction(metrics, processed).to_csv(outputs["original_vs_reproduction_csv"], index=False)
    _artifact_manifest().to_csv(outputs["artifact_manifest_csv"], index=False)
    _stage_sensitivity(results).to_csv(outputs["stage_sensitivity_csv"], index=False)
    _tradeoff_components(metrics).to_csv(outputs["tradeoff_components_csv"], index=False)
    _core_seed_summary(metrics).to_csv(outputs["core_seed_summary_csv"], index=False)
    _open_loop_peak_signature(results, metrics).to_csv(outputs["open_loop_peak_csv"], index=False)
    _stability_alignment(metrics, processed).to_csv(outputs["stability_alignment_csv"], index=False)
    _loss_scale_comparison(results, metrics).to_csv(outputs["loss_scale_csv"], index=False)
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
    tracking = variants[variants["variant"] == "tracking_task"].iloc[0] if variants is not None and "tracking_task" in set(variants["variant"]) else None
    ring = variants[variants["variant"].astype(str).str.contains("ring", regex=False)] if variants is not None else pd.DataFrame()
    return pd.DataFrame(
        [
            {
                "original_claim": "Closed/open divergence",
                "original_evidence": "Main double-integrator loss/gain figures",
                "our_test": "50 paired seeds, same initialization, deployed closed-loop loss; separate peak-signature check",
                "result": f"{int(original['claim_C1_loss_divergence'].sum())}/{len(original)} seeds support final-loss gap; post-initial open-loop peak is not recovered",
                "decision": "Partial claim-level reproduction",
            },
            {
                "original_claim": "Closed-loop stage structure",
                "original_evidence": "Spectral stages: negative-position policy, world-model/stability phase, policy refinement",
                "our_test": "Loss-only derivative detector plus segmented changepoint analysis",
                "result": f"loss slow phase {int(original['claim_C2_plateau_present'].sum())}/{len(original)}; loss-only strict three-stage {int(original['claim_C2_three_stage'].sum())}/{len(original)}",
                "decision": "Loss-only observer test; not a falsification of the spectral stage claim",
            },
            {
                "original_claim": "Coupled-system stability",
                "original_evidence": "Coupled eigenvalue/spectral argument",
                "our_test": "Coupled spectral radius crossing and alignment with loss changepoints",
                "result": f"{int(original['claim_C3_stability_transition'].sum())}/{len(original)} seeds cross the coupled stability boundary, but crossing is early relative to loss-only plateau exit",
                "decision": "Restricted support; alignment claim not reproduced by loss-only boundaries",
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
                "produces": "run accounting, original-vs-reproduction, A1 split, C2 sensitivity, peak/alignment/scale checks",
                "runtime": "<1 min",
                "expected_output": "results/processed/{run_accounting,original_vs_reproduction,tradeoff_component_summary,stage_sensitivity,open_loop_peak_signature,stability_alignment_summary,loss_scale_comparison}.csv",
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


def _open_loop_peak_signature(results: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    original_artifact = _artifact_loss_curve("open")
    if original_artifact is not None:
        rows.append(_peak_row("upstream_artifact_open_curve", original_artifact, n=1))
    closed_artifact = _artifact_loss_curve("closed")
    if closed_artifact is not None:
        rows.append(_peak_row("upstream_artifact_closed_curve", closed_artifact, n=1))

    independent = []
    for path in sorted((results / "original_double_integrator_full").glob("seed_*/timeseries.csv")):
        frame = pd.read_csv(path)
        independent.append(frame["open_test_loss"].to_numpy(dtype=float))
    if independent:
        peak_ratios_initial = []
        peak_ratios_final = []
        peak_epochs = []
        post_initial_support = 0
        for curve in independent:
            peak_epoch = int(np.nanargmax(curve))
            peak = float(np.nanmax(curve))
            initial = float(curve[0])
            final = float(curve[-1])
            peak_ratios_initial.append(peak / max(initial, 1e-12))
            peak_ratios_final.append(peak / max(final, 1e-12))
            peak_epochs.append(peak_epoch)
            post_initial_support += int(peak_epoch > 0 and peak > 1.5 * max(initial, 1e-12))
        rows.append(
            {
                "source": "independent_original_setting_open_curve",
                "n": len(independent),
                "post_initial_peak_support": post_initial_support,
                "post_initial_peak_fraction": post_initial_support / len(independent),
                "median_peak_epoch": float(np.nanmedian(peak_epochs)),
                "median_peak_ratio_to_initial": float(np.nanmedian(peak_ratios_initial)),
                "median_peak_ratio_to_final": float(np.nanmedian(peak_ratios_final)),
                "definition": "post-initial peak if argmax(epoch)>0 and max loss > 1.5 * initial loss",
            }
        )

    return pd.DataFrame(rows)


def _peak_row(source: str, curve: np.ndarray, n: int) -> dict[str, Any]:
    curve = np.asarray(curve, dtype=float)
    peak_epoch = int(np.nanargmax(curve))
    peak = float(np.nanmax(curve))
    initial = float(curve[0])
    final = float(curve[-1])
    support = int(peak_epoch > 0 and peak > 1.5 * max(initial, 1e-12))
    return {
        "source": source,
        "n": n,
        "post_initial_peak_support": support,
        "post_initial_peak_fraction": float(support),
        "median_peak_epoch": float(peak_epoch),
        "median_peak_ratio_to_initial": peak / max(initial, 1e-12),
        "median_peak_ratio_to_final": peak / max(final, 1e-12),
        "definition": "post-initial peak if argmax(epoch)>0 and max loss > 1.5 * initial loss",
    }


def _artifact_loss_curve(name: str) -> np.ndarray | None:
    path = Path("external/original_artifact/data") / f"non_linear_{name}.pkl"
    if not path.exists():
        return None
    with path.open("rb") as handle:
        obj = pickle.load(handle)
    if isinstance(obj, tuple) and len(obj) >= 2:
        return np.asarray(obj[1], dtype=float)
    return None


def _stability_alignment(metrics: pd.DataFrame, processed: Path) -> pd.DataFrame:
    original = metrics[metrics["kind"] == "original"].copy()
    details = _read_csv(processed / "stage_changepoint_details.csv")
    if details is not None:
        details = details[details["kind"] == "original"][["seed", "boundary1", "boundary2"]]
        original = original.merge(details, on="seed", how="left")
    rows = []
    for label, boundary in [
        ("loss_stage1_boundary", "stage1_end"),
        ("loss_plateau_fallback_end", "plateau_end"),
        ("changepoint_boundary1", "boundary1"),
        ("changepoint_boundary2", "boundary2"),
    ]:
        if boundary not in original:
            continue
        gap = pd.to_numeric(original[boundary], errors="coerce") - pd.to_numeric(original["stability_crossing"], errors="coerce")
        rows.append(
            {
                "comparison": label,
                "n": int(gap.notna().sum()),
                "median_boundary_epoch": float(pd.to_numeric(original[boundary], errors="coerce").median()),
                "median_stability_crossing_epoch": float(pd.to_numeric(original["stability_crossing"], errors="coerce").median()),
                "median_boundary_minus_stability": float(gap.median()),
                "iqr_low_gap": float(gap.quantile(0.25)),
                "iqr_high_gap": float(gap.quantile(0.75)),
            }
        )
    return pd.DataFrame(rows)


def _loss_scale_comparison(results: Path, metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name in ["closed", "open"]:
        curve = _artifact_loss_curve(name)
        if curve is not None:
            rows.append(
                {
                    "source": f"upstream_artifact_{name}",
                    "n": 1,
                    "initial_loss": float(curve[0]),
                    "final_loss": float(curve[-1]),
                    "min_loss": float(np.nanmin(curve)),
                    "max_loss": float(np.nanmax(curve)),
                    "note": "precomputed upstream artifact curve; original notebook metric/normalization",
                }
            )
    original = metrics[metrics["kind"] == "original"]
    if not original.empty:
        rows.extend(
            [
                {
                    "source": "independent_closed_mean",
                    "n": int(len(original)),
                    "initial_loss": _mean_initial_loss(results, "closed_test_loss"),
                    "final_loss": float(original["final_closed_test_loss"].mean()),
                    "min_loss": float(original["final_closed_test_loss"].min()),
                    "max_loss": float(original["peak_closed_test_loss"].max()),
                    "note": "independent implementation; mean deployed loss with repository evaluation convention",
                },
                {
                    "source": "independent_open_mean",
                    "n": int(len(original)),
                    "initial_loss": _mean_initial_loss(results, "open_test_loss"),
                    "final_loss": float(original["final_open_test_loss"].mean()),
                    "min_loss": float(original["final_open_test_loss"].min()),
                    "max_loss": float(original["peak_open_test_loss"].max()),
                    "note": "independent implementation; mean deployed loss with repository evaluation convention",
                },
            ]
        )
    return pd.DataFrame(rows)


def _mean_initial_loss(results: Path, column: str) -> float:
    values = []
    for path in sorted((results / "original_double_integrator_full").glob("seed_*/timeseries.csv")):
        frame = pd.read_csv(path)
        values.append(float(frame[column].iloc[0]))
    return float(np.mean(values)) if values else float("nan")


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
