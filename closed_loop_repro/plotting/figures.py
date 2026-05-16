from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from closed_loop_repro.io import ensure_dir


def make_figures(results: str | Path = "results/raw", processed: str | Path = "results/processed", out: str | Path = "results/figures") -> list[Path]:
    results = Path(results)
    processed = Path(processed)
    out = ensure_dir(out)
    paths = []
    sns.set_theme(context="paper", style="ticks")

    original = _load_timeseries(results / "original_double_integrator_full")
    if original:
        paths.append(_figure_core_reproduction(original, processed, out))
        paths.append(_figure_stage_analysis(original, processed, out))
        paths.append(_figure_coupled_spectra(original, processed, out))

    setting_summary = _read_csv(processed / "recomputed_setting_summary.csv")
    if setting_summary is not None and not setting_summary.empty:
        paths.append(_figure_robustness_heatmap(setting_summary, out))
        paths.append(_figure_generalization(setting_summary, out))

    claim_path = processed / "claim_reproducibility_matrix.csv"
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


def _load_timeseries(experiment_dir: Path) -> list[pd.DataFrame]:
    frames = []
    for path in sorted(experiment_dir.glob("seed_*/timeseries.csv")):
        frame = pd.read_csv(path)
        frame["seed"] = int(path.parent.name.split("_")[-1])
        frames.append(frame)
    return frames


def _read_csv(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    return pd.read_csv(path)


def _stack(frames: list[pd.DataFrame], column: str) -> tuple[np.ndarray, np.ndarray]:
    common_len = min(len(frame) for frame in frames)
    epochs = frames[0]["epoch"].to_numpy()[:common_len]
    values = np.vstack([frame[column].to_numpy(dtype=float)[:common_len] for frame in frames])
    return epochs, values


def _plot_median_band(ax, epochs: np.ndarray, values: np.ndarray, label: str, color: tuple[float, float, float]) -> None:
    median = np.nanmedian(values, axis=0)
    lo = np.nanquantile(values, 0.25, axis=0)
    hi = np.nanquantile(values, 0.75, axis=0)
    ax.plot(epochs, median, label=label, color=color, lw=2.0)
    ax.fill_between(epochs, lo, hi, color=color, alpha=0.18, lw=0)


def _figure_core_reproduction(frames: list[pd.DataFrame], processed: Path, out: Path) -> Path:
    colors = sns.color_palette("deep")
    epochs, closed = _stack(frames, "closed_test_loss")
    _, open_loop = _stack(frames, "open_test_loss")
    final_df = _read_csv(processed / "recomputed_timeseries_metrics.csv")
    final_df = final_df[final_df["kind"] == "original"] if final_df is not None else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), gridspec_kw={"width_ratios": [1.45, 1.0]})
    _plot_median_band(axes[0], epochs, closed, "Closed-loop trained", colors[0])
    _plot_median_band(axes[0], epochs, open_loop, "Open-loop trained", colors[3])
    axes[0].set(xlabel="Epoch", ylabel="Deployed closed-loop loss", yscale="log")
    axes[0].legend(frameon=False, loc="upper right")

    if not final_df.empty:
        xs = np.zeros(len(final_df))
        axes[1].scatter(xs - 0.08, final_df["final_closed_test_loss"], s=18, alpha=0.65, color=colors[0], label="Closed")
        axes[1].scatter(xs + 0.08, final_df["final_open_test_loss"], s=18, alpha=0.65, color=colors[3], label="Open")
        for _, row in final_df.iterrows():
            axes[1].plot(
                [-0.08, 0.08],
                [row["final_closed_test_loss"], row["final_open_test_loss"]],
                color="0.75",
                lw=0.6,
                zorder=0,
            )
        axes[1].set_xticks([-0.08, 0.08], ["Closed", "Open"])
        axes[1].set(ylabel="Final deployed loss", yscale="log", xlim=(-0.35, 0.35), title="50 paired seeds")
    sns.despine(fig)
    fig.tight_layout()
    path = out / "figure_2_core_reproduction.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_stage_analysis(frames: list[pd.DataFrame], processed: Path, out: Path) -> Path:
    details = _read_csv(processed / "stage_changepoint_details.csv")
    seed = int(details.iloc[0]["seed"]) if details is not None and not details.empty else int(frames[0]["seed"].iloc[0])
    frame = next((item for item in frames if int(item["seed"].iloc[0]) == seed), frames[0])
    row = details[details["seed"] == seed].iloc[0] if details is not None and not details.empty else None
    x = frame["epoch"].to_numpy(dtype=float)
    y = np.log(np.maximum(frame["closed_test_loss"].to_numpy(dtype=float), 1e-12))
    b1 = int(row["boundary1"]) if row is not None and np.isfinite(row["boundary1"]) else int(0.2 * len(x))
    b2 = int(row["boundary2"]) if row is not None and np.isfinite(row["boundary2"]) else int(0.7 * len(x))

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), gridspec_kw={"width_ratios": [1.35, 1.0]})
    axes[0].plot(x, frame["closed_test_loss"], color=sns.color_palette("deep")[0], lw=1.8)
    axes[0].axvline(b1, color="0.2", ls="--", lw=1.0, label=r"$\tau_1$")
    axes[0].axvline(b2, color="0.45", ls=":", lw=1.2, label=r"$\tau_2$")
    axes[0].axvspan(b1, b2, color=sns.color_palette("deep")[2], alpha=0.12)
    axes[0].set(xlabel="Epoch", ylabel="Deployed loss", yscale="log", title=f"Seed {seed}")
    axes[0].legend(frameon=False)

    axes[1].plot(x, y, color="0.72", lw=1.0, label="log loss")
    for start, end, color, label in [
        (0, b1, sns.color_palette("deep")[0], "segment 1"),
        (b1, b2, sns.color_palette("deep")[2], "segment 2"),
        (b2, len(x) - 1, sns.color_palette("deep")[3], "segment 3"),
    ]:
        segment = (x >= start) & (x <= end)
        if np.sum(segment) >= 2:
            coeff = np.polyfit(x[segment], y[segment], 1)
            axes[1].plot(x[segment], np.polyval(coeff, x[segment]), color=color, lw=2.0, label=label)
    axes[1].set(xlabel="Epoch", ylabel="log deployed loss", title="Segmented fit")
    axes[1].legend(frameon=False, fontsize=7)
    sns.despine(fig)
    fig.tight_layout()
    path = out / "figure_3_stage_analysis.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_robustness_heatmap(setting_summary: pd.DataFrame, out: Path) -> Path:
    rob = setting_summary[setting_summary["kind"] == "robustness"].copy()
    order = [
        "original",
        "lr_low",
        "lr_high",
        "init_weak",
        "init_strong",
        "feedback_low",
        "feedback_high",
        "episode_short",
        "episode_long",
        "obs_noise",
        "high_control_penalty",
        "adam",
        "no_clip",
    ]
    rob["setting"] = pd.Categorical(rob["setting"], categories=order, ordered=True)
    rob = rob.sort_values("setting")
    columns = {
        "C1 loss": "c1_loss_support",
        "C2 slow": "c2_plateau_support",
        "C2 strict": "c2_three_stage_support",
        "C3 stability": "c3_stability_support",
        "C4 proxy": "c4_proxy_support",
    }
    heat = rob.set_index("setting")[[*columns.values()]].rename(columns={v: k for k, v in columns.items()})
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.heatmap(heat, ax=ax, vmin=0, vmax=1, cmap="viridis", annot=True, fmt=".2g", cbar_kws={"label": "Support fraction"})
    ax.set(xlabel="Claim diagnostic", ylabel="Perturbation")
    fig.tight_layout()
    path = out / "figure_4_robustness_heatmap.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_coupled_spectra(frames: list[pd.DataFrame], processed: Path, out: Path) -> Path:
    colors = sns.color_palette("deep")
    epochs, coupled = _stack(frames, "closed_coupled_radius")
    _, rnn = _stack(frames, "closed_rnn_radius")
    metrics = _read_csv(processed / "recomputed_timeseries_metrics.csv")
    crossings = metrics[metrics["kind"] == "original"]["stability_crossing"].dropna().to_numpy(dtype=float) if metrics is not None else []
    fig, ax = plt.subplots(figsize=(6.2, 3.4))
    _plot_median_band(ax, epochs, coupled, r"$\rho_{\mathrm{coup}}$", colors[0])
    _plot_median_band(ax, epochs, rnn, r"$\rho_{\mathrm{RNN}}$", colors[4])
    ax.axhline(1.0, color="0.15", ls="--", lw=1.0)
    if len(crossings):
        median_cross = float(np.nanmedian(crossings))
        ax.axvline(median_cross, color=colors[3], ls=":", lw=1.4)
        ax.text(median_cross + 18, 0.08, f"median crossing: {median_cross:.0f}", color=colors[3], fontsize=8)
    ax.set(xlabel="Epoch", ylabel="Spectral radius", ylim=(0, max(1.25, np.nanquantile(coupled, 0.98) * 1.05)))
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right")
    sns.despine(fig)
    fig.tight_layout()
    path = out / "figure_5_coupled_spectral_analysis.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_generalization(setting_summary: pd.DataFrame, out: Path) -> Path:
    gen = setting_summary[setting_summary["kind"] == "generalization"].copy()
    labels = {
        "tanh_rnn": "tanh",
        "gru": "GRU",
        "low_rank": "low-rank",
        "tracking_task": "tracking",
        "ring_path_integration": "ring",
    }
    order = ["tanh_rnn", "gru", "low_rank", "tracking_task", "ring_path_integration"]
    gen["setting"] = pd.Categorical(gen["setting"], categories=order, ordered=True)
    gen = gen.sort_values("setting")
    x = np.arange(len(gen))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    axes[0].bar(x, gen["c1_loss_support"], color=sns.color_palette("deep", len(gen)))
    axes[0].set_xticks(x, [labels.get(str(v), str(v)) for v in gen["setting"]], rotation=25, ha="right")
    axes[0].set(ylim=(0, 1.05), ylabel="C1 support fraction", xlabel="Variant")
    gap = np.maximum(gen["mean_deployed_loss_gap"].to_numpy(dtype=float), 1e-5)
    axes[1].scatter(x, gap, color=sns.color_palette("deep", len(gen)), s=48, zorder=3)
    axes[1].vlines(x, 1e-4, gap, color="0.78", lw=1.0, zorder=2)
    axes[1].set_xticks(x, [labels.get(str(v), str(v)) for v in gen["setting"]], rotation=25, ha="right")
    axes[1].set(ylabel="Mean deployed loss gap", xlabel="Variant", yscale="log", ylim=(1e-4, max(gap) * 2.0))
    sns.despine(fig)
    fig.tight_layout()
    path = out / "figure_6_generalization.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path
