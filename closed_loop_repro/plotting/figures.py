from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from closed_loop_repro.io import ensure_dir


def make_figures(results: str | Path = "results/raw", processed: str | Path = "results/processed", out: str | Path = "results/figures") -> list[Path]:
    out = ensure_dir(out)
    paths = []
    sns.set_theme(context="paper", style="ticks")
    timeseries = sorted(Path(results).rglob("timeseries.csv"))
    if timeseries:
        df = pd.read_csv(timeseries[0])
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.plot(df["epoch"], df["closed_test_loss"], label="Closed-loop", lw=2)
        ax.plot(df["epoch"], df["open_test_loss"], label="Open-loop", lw=2)
        ax.set(xlabel="Epoch", ylabel="Closed-loop test loss", yscale="log")
        ax.legend(frameon=False)
        sns.despine(fig)
        path = out / "figure_2_core_reproduction.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

        if "closed_coupled_radius" in df:
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.plot(df["epoch"], df["closed_coupled_radius"], label="Coupled system", lw=2)
            ax.plot(df["epoch"], df["closed_rnn_radius"], label="RNN only", lw=2)
            ax.axhline(1.0, color="k", ls="--", lw=1)
            ax.set(xlabel="Epoch", ylabel="Spectral radius")
            ax.legend(frameon=False)
            sns.despine(fig)
            path = out / "figure_4_stability_analysis.png"
            fig.savefig(path, dpi=200, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

    claim_path = Path(processed) / "claim_reproducibility_matrix.csv"
    if claim_path.exists():
        claim_df = pd.read_csv(claim_path)
        fig, ax = plt.subplots(figsize=(7, 3))
        plot_df = claim_df.copy()
        plot_df["support_fraction"] = pd.to_numeric(plot_df["support_fraction"], errors="coerce")
        ax.bar(plot_df["claim"], plot_df["support_fraction"].fillna(0.0), color=sns.color_palette("deep", len(plot_df)))
        ax.set(ylim=(0, 1), ylabel="Support fraction", xlabel="Claim")
        sns.despine(fig)
        path = out / "figure_5_claim_matrix.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)

    if not paths:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.text(0.5, 0.5, "No experiment outputs found yet", ha="center", va="center")
        ax.axis("off")
        path = out / "figure_1_project_overview.png"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        plt.close(fig)
        paths.append(path)
    return paths
