from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from closed_loop_repro.analysis.statistics import bootstrap_ci, fraction, pearson
from closed_loop_repro.io import ensure_dir, read_json


CLAIMS = {
    "C1": "Closed-loop/open-loop divergence",
    "C2": "Closed-loop stages",
    "C3": "Coupled-system stability transition",
    "C4": "Short-term vs long-term tradeoff",
    "C5": "Generalization",
}


def collect_metrics(root: str | Path) -> pd.DataFrame:
    rows = []
    for path in Path(root).rglob("metrics.json"):
        row = read_json(path)
        row["metrics_path"] = str(path)
        rows.append(row)
    return pd.DataFrame(rows)


def make_claim_tables(results: str | Path, out: str | Path) -> dict[str, Path]:
    out = ensure_dir(out)
    df = collect_metrics(results)
    if df.empty:
        empty = pd.DataFrame(columns=["claim", "description", "n", "support_fraction", "notes"])
        claim_path = out / "claim_reproducibility_matrix.csv"
        empty.to_csv(claim_path, index=False)
        (out / "claim_reproducibility_matrix.md").write_text(empty.to_markdown(index=False), encoding="utf-8")
        return {"claim_csv": claim_path}

    rows = []
    rows.append(_claim_row("C1", df, "claim_C1_divergence", "fraction of runs with gain/performance divergence"))
    rows.append(_claim_row("C2", df, "claim_C2_stages", "fraction with algorithmic plateau/stage detection"))
    rows.append(_claim_row("C3", df, "claim_C3_stability_transition", "fraction with coupled spectral radius crossing"))
    rows.append(_numeric_row("C4", df, "stability_to_plateau_gap", "bootstrap CI for stability-to-plateau timing gap"))
    if "variant" in df.columns:
        rows.append(_claim_row("C5", df[df["variant"].notna()], "claim_C1_divergence", "generalization variants preserving C1 divergence"))
    else:
        rows.append({"claim": "C5", "description": CLAIMS["C5"], "n": 0, "support_fraction": float("nan"), "notes": "no generalization runs found"})

    claim_df = pd.DataFrame(rows)
    claim_path = out / "claim_reproducibility_matrix.csv"
    claim_df.to_csv(claim_path, index=False)
    (out / "claim_reproducibility_matrix.md").write_text(claim_df.to_markdown(index=False), encoding="utf-8")

    stats = {
        "n_runs": len(df),
        "closed_open_final_loss_corr": pearson(df.get("final_closed_test_loss", []), df.get("final_open_test_loss", [])),
    }
    pd.DataFrame([stats]).to_csv(out / "summary_statistics.csv", index=False)
    return {"claim_csv": claim_path, "summary_csv": out / "summary_statistics.csv"}


def _claim_row(claim: str, df: pd.DataFrame, column: str, notes: str) -> dict:
    if df.empty or column not in df:
        return {"claim": claim, "description": CLAIMS[claim], "n": 0, "support_fraction": float("nan"), "notes": notes}
    return {"claim": claim, "description": CLAIMS[claim], "n": int(df[column].notna().sum()), "support_fraction": fraction(df[column].fillna(False)), "notes": notes}


def _numeric_row(claim: str, df: pd.DataFrame, column: str, notes: str) -> dict:
    if column not in df:
        return {"claim": claim, "description": CLAIMS[claim], "n": 0, "support_fraction": float("nan"), "notes": notes}
    mean, lo, hi = bootstrap_ci(pd.to_numeric(df[column], errors="coerce").dropna().to_numpy(), n_boot=500)
    return {"claim": claim, "description": CLAIMS[claim], "n": int(df[column].notna().sum()), "support_fraction": mean, "ci_low": lo, "ci_high": hi, "notes": notes}


def main() -> None:
    parser = argparse.ArgumentParser(description="Create claim-level reproducibility tables.")
    parser.add_argument("--results", default="results/raw")
    parser.add_argument("--out", default="results/processed")
    args = parser.parse_args()
    paths = make_claim_tables(args.results, args.out)
    print(paths)


if __name__ == "__main__":
    main()
