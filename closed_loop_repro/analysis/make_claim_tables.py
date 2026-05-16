from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from closed_loop_repro.analysis.statistics import bootstrap_ci, fraction, pearson
from closed_loop_repro.io import ensure_dir, read_json


CLAIMS = {
    "C1": "Closed-loop/open-loop divergence",
    "C2": "Closed-loop stages",
    "C3": "Coupled-system stability transition",
    "A1": "Protocol robustness and short/long horizon tradeoff",
    "A2": "Broader generalization",
}


SUMMARY_FILENAMES = {"sweep_summary.csv", "robustness_summary.csv", "generalization_summary.csv", "tradeoff_summary.csv"}


def collect_metrics(root: str | Path) -> pd.DataFrame:
    return collect_runs(root)


def collect_runs(root: str | Path) -> pd.DataFrame:
    summary_paths = sorted(path for path in Path(root).rglob("*.csv") if path.name in SUMMARY_FILENAMES)
    if summary_paths:
        selected = _prefer_full_outputs(summary_paths)
        frames = []
        for path in selected:
            frame = pd.read_csv(path)
            frame["source_table"] = str(path)
            frames.append(frame)
        return _add_derived_columns(pd.concat(frames, ignore_index=True, sort=False))

    rows = []
    for path in Path(root).rglob("metrics.json"):
        if _is_smoke_path(path) and any(not _is_smoke_path(candidate) for candidate in Path(root).rglob("metrics.json")):
            continue
        row = read_json(path)
        row["metrics_path"] = str(path)
        rows.append(row)
    return _add_derived_columns(pd.DataFrame(rows))


def make_claim_tables(results: str | Path, out: str | Path) -> dict[str, Path]:
    out = ensure_dir(out)
    df = collect_metrics(results)
    if df.empty:
        empty = pd.DataFrame(columns=["claim", "description", "n", "support_fraction", "notes"])
        claim_path = out / "claim_reproducibility_matrix.csv"
        empty.to_csv(claim_path, index=False)
        (out / "claim_reproducibility_matrix.md").write_text(_to_markdown(empty), encoding="utf-8")
        return {"claim_csv": claim_path}

    rows = []
    rows.append(_claim_row("C1", df, "claim_C1_loss_divergence", "deployed closed-loop loss gap >5%; gain divergence reported in diagnostics"))
    rows.append(_claim_row("C2", df, "claim_C2_three_stage", "finite runs with algorithmic plateau and non-fallback exit detection"))
    rows.append(_claim_row("C3", df, "claim_C3_stability_transition", "finite coupled spectral radius crossing"))
    if "claim_C4_tradeoff_quantified" in df and df["claim_C4_tradeoff_quantified"].notna().any():
        c4_df = df[df["claim_C4_tradeoff_quantified"].notna()]
        rows.append(_claim_row("A1", c4_df, "claim_C4_tradeoff_quantified", "targeted short-vs-long horizon tradeoff sweep"))
    else:
        rows.append(_claim_row("A1", df, "claim_C4_tradeoff", "proxy only: open-loop deployed-loss spike plus closed-loop recovery"))
    if "variant" in df.columns:
        rows.append(_claim_row("A2", df[df["variant"].notna()], "claim_C1_loss_divergence", "generalization variants preserving deployed-loss divergence"))
    else:
        rows.append({"claim": "A2", "description": CLAIMS["A2"], "n": 0, "support_fraction": float("nan"), "notes": "no generalization runs found"})

    claim_df = pd.DataFrame(rows)
    claim_path = out / "claim_reproducibility_matrix.csv"
    claim_df.to_csv(claim_path, index=False)
    (out / "claim_reproducibility_matrix.md").write_text(_to_markdown(claim_df), encoding="utf-8")

    stats = {
        "n_runs": len(df),
        "n_finite_runs": int(_bool_series(df.get("finite_final_losses", pd.Series(False, index=df.index))).sum()),
        "closed_open_final_loss_corr": pearson(df.get("final_closed_test_loss", []), df.get("final_open_test_loss", [])),
        "c1_deployed_loss_support": fraction(df.get("claim_C1_loss_divergence", [])),
        "c1_gain_support": fraction(df.get("claim_C1_gain_divergence", [])),
        "mean_deployed_loss_gap": float(pd.to_numeric(df.get("deployed_loss_gap", pd.Series(dtype=float)), errors="coerce").mean()),
        "mean_stability_to_plateau_gap": float(pd.to_numeric(df.get("stability_to_plateau_gap", pd.Series(dtype=float)), errors="coerce").mean()),
    }
    pd.DataFrame([stats]).to_csv(out / "summary_statistics.csv", index=False)
    outputs = {"claim_csv": claim_path, "summary_csv": out / "summary_statistics.csv"}
    if "perturbation" in df.columns and df["perturbation"].notna().any():
        perturbation_path = out / "perturbation_summary.csv"
        _group_summary(df[df["perturbation"].notna()], "perturbation").to_csv(perturbation_path, index=False)
        outputs["perturbation_csv"] = perturbation_path
    if "variant" in df.columns and df["variant"].notna().any():
        variant_path = out / "variant_summary.csv"
        _group_summary(df[df["variant"].notna()], "variant").to_csv(variant_path, index=False)
        outputs["variant_csv"] = variant_path
    return outputs


def _claim_row(claim: str, df: pd.DataFrame, column: str, notes: str) -> dict:
    if df.empty or column not in df:
        return {"claim": claim, "description": CLAIMS[claim], "n": 0, "support_fraction": float("nan"), "notes": notes}
    values = _bool_series(df[column]).fillna(False)
    mean, lo, hi = bootstrap_ci(values.astype(float).to_numpy(), n_boot=500)
    return {
        "claim": claim,
        "description": CLAIMS[claim],
        "n": int(len(values)),
        "support_fraction": mean,
        "ci_low": lo,
        "ci_high": hi,
        "notes": notes,
    }


def _prefer_full_outputs(paths: list[Path]) -> list[Path]:
    full_paths = [path for path in paths if not _is_smoke_path(path)]
    return full_paths or paths


def _is_smoke_path(path: Path) -> bool:
    return any("smoke" in part.lower() for part in path.parts)


def _add_derived_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    closed = pd.to_numeric(df.get("final_closed_test_loss", pd.Series(np.nan, index=df.index)), errors="coerce")
    open_loop = pd.to_numeric(df.get("final_open_test_loss", pd.Series(np.nan, index=df.index)), errors="coerce")
    finite = np.isfinite(closed) & np.isfinite(open_loop)
    loss_gap = open_loop - closed
    relative_gap = loss_gap / np.maximum(np.abs(closed), 1e-12)
    df["finite_final_losses"] = finite
    df["deployed_loss_gap"] = loss_gap
    df["deployed_loss_gap_relative_to_closed"] = relative_gap
    df["claim_C1_loss_divergence"] = finite & (relative_gap > 0.05)

    gain_distance = pd.to_numeric(df.get("trajectory_gain_distance", pd.Series(np.nan, index=df.index)), errors="coerce")
    df["claim_C1_gain_divergence"] = np.isfinite(gain_distance) & (gain_distance > 0.05)
    if "claim_C1_divergence" not in df.columns:
        df["claim_C1_divergence"] = df["claim_C1_loss_divergence"] | df["claim_C1_gain_divergence"]

    if "claim_C2_stages" in df.columns:
        df["claim_C2_stages"] = _bool_series(df["claim_C2_stages"]) & finite
    if "claim_C2_three_stage" in df.columns:
        df["claim_C2_three_stage"] = _bool_series(df["claim_C2_three_stage"]) & finite
    elif "claim_C2_stages" in df.columns:
        exit_detected = _bool_series(df.get("plateau_exit_detected", pd.Series(True, index=df.index))).fillna(True)
        df["claim_C2_three_stage"] = df["claim_C2_stages"] & exit_detected
    if "claim_C3_stability_transition" in df.columns:
        radius = pd.to_numeric(df.get("final_closed_coupled_radius", pd.Series(np.nan, index=df.index)), errors="coerce")
        df["claim_C3_stability_transition"] = _bool_series(df["claim_C3_stability_transition"]) & np.isfinite(radius)

    open_spike = _bool_series(df.get("open_loop_test_loss_spike", pd.Series(False, index=df.index))).fillna(False)
    recovered = _bool_series(df.get("closed_recovered", pd.Series(False, index=df.index))).fillna(False)
    df["claim_C4_tradeoff"] = finite & open_spike & recovered
    if "claim_C4_tradeoff_quantified" in df.columns:
        nonmissing = df["claim_C4_tradeoff_quantified"].notna()
        quantified = pd.Series(np.nan, index=df.index, dtype=object)
        quantified.loc[nonmissing] = _bool_series(df.loc[nonmissing, "claim_C4_tradeoff_quantified"])
        df["claim_C4_tradeoff_quantified"] = quantified
    return df


def _bool_series(values: object) -> pd.Series:
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.dtype == bool:
        return series
    if pd.api.types.is_numeric_dtype(series):
        return series.fillna(0).astype(bool)
    normalized = series.astype(str).str.strip().str.lower()
    return normalized.map({"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}).fillna(False).astype(bool)


def _group_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    rows = []
    for name, group in df.groupby(group_col, dropna=True):
        finite = _bool_series(group.get("finite_final_losses", pd.Series(False, index=group.index)))
        rows.append(
            {
                group_col: name,
                "n": int(len(group)),
                "finite_runs": int(finite.sum()),
                "failure_fraction": float(1.0 - finite.mean()) if len(finite) else float("nan"),
                "mean_final_closed_test_loss": float(pd.to_numeric(group["final_closed_test_loss"], errors="coerce").mean()),
                "mean_final_open_test_loss": float(pd.to_numeric(group["final_open_test_loss"], errors="coerce").mean()),
                "mean_deployed_loss_gap": float(pd.to_numeric(group["deployed_loss_gap"], errors="coerce").mean()),
                "c1_deployed_loss_support": fraction(group["claim_C1_loss_divergence"]),
                "c1_gain_support": fraction(group["claim_C1_gain_divergence"]),
                "c2_stage_support": fraction(group.get("claim_C2_stages", [])),
                "c3_stability_support": fraction(group.get("claim_C3_stability_transition", [])),
                "open_loop_spike_fraction": fraction(group.get("open_loop_test_loss_spike", [])),
                "closed_recovery_fraction": fraction(group.get("closed_recovered", [])),
            }
        )
    return pd.DataFrame(rows).sort_values(group_col)


def _to_markdown(df: pd.DataFrame) -> str:
    columns = [str(column) for column in df.columns]
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for _, row in df.iterrows():
        values = [_format_markdown_value(row[column]) for column in df.columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines) + "\n"


def _format_markdown_value(value: object) -> str:
    if pd.isna(value):
        return ""
    text = str(value)
    return text.replace("|", "\\|").replace("\n", " ")


def main() -> None:
    parser = argparse.ArgumentParser(description="Create claim-level reproducibility tables.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--out", default="results/processed")
    args = parser.parse_args()
    paths = make_claim_tables(args.results, args.out)
    print(paths)


if __name__ == "__main__":
    main()
