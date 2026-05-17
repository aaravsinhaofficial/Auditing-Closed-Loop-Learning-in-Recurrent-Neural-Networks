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

    tradeoff = _load_tradeoff_summary(results, processed)
    if tradeoff is not None and not tradeoff.empty:
        paths.append(_figure_tradeoff(tradeoff, out))

    setting_summary = _read_csv(processed / "recomputed_setting_summary.csv")
    if setting_summary is not None and not setting_summary.empty:
        paths.append(_figure_robustness_heatmap(setting_summary, out))
        paths.append(_figure_generalization(setting_summary, processed, out))

    claim_path = processed / "claim_reproducibility_matrix.csv"
    if claim_path.exists():
        claim_df = _read_csv(claim_path)
    else:
        claim_df = None
    if claim_df is not None and not claim_df.empty:
        fig, ax = plt.subplots(figsize=(7, 3))
        plot_df = claim_df.copy()
        plot_df["support_fraction"] = pd.to_numeric(plot_df["support_fraction"], errors="coerce")
        ax.bar(plot_df["claim"], plot_df["support_fraction"].fillna(0.0), color=sns.color_palette("deep", len(plot_df)))
        ax.set(ylim=(0, 1), ylabel="Support fraction", xlabel="Claim/audit item")
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
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def _load_tradeoff_summary(results: Path, processed: Path) -> pd.DataFrame | None:
    recomputed = _read_csv(processed / "recomputed_timeseries_metrics.csv")
    if recomputed is not None and "kind" in recomputed:
        tradeoff = recomputed[recomputed["kind"] == "tradeoff"].copy()
        if not tradeoff.empty:
            if "tradeoff_condition" not in tradeoff:
                tradeoff["tradeoff_condition"] = tradeoff["setting"]
            return tradeoff
    candidates = sorted(results.glob("tradeoff_*/tradeoff_summary.csv"))
    candidates = [path for path in candidates if "smoke" not in str(path).lower()]
    if not candidates:
        return None
    frames = []
    for path in candidates:
        frame = pd.read_csv(path)
        frame["source_table"] = str(path)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True, sort=False)


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
    ax.fill_between(epochs, lo, hi, color=color, alpha=0.24, lw=0)


def _panel_label(ax, label: str) -> None:
    ax.text(-0.12, 1.06, label, transform=ax.transAxes, fontsize=10, fontweight="bold", va="top", ha="left")


def _figure_core_reproduction(frames: list[pd.DataFrame], processed: Path, out: Path) -> Path:
    colors = sns.color_palette("deep")
    epochs, closed = _stack(frames, "closed_test_loss")
    _, open_loop = _stack(frames, "open_test_loss")
    final_df = _read_csv(processed / "recomputed_timeseries_metrics.csv")
    final_df = final_df[final_df["kind"] == "original"] if final_df is not None else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.1), gridspec_kw={"width_ratios": [1.45, 1.0]})
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")
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
    if details is not None and not details.empty and {"kind", "setting"}.issubset(details.columns):
        details = details[(details["kind"] == "original") & (details["setting"] == "original")].copy()
    seed = int(details.iloc[0]["seed"]) if details is not None and not details.empty else int(frames[0]["seed"].iloc[0])
    frame = next((item for item in frames if int(item["seed"].iloc[0]) == seed), frames[0])
    row = details[details["seed"] == seed].iloc[0] if details is not None and not details.empty else None
    x = frame["epoch"].to_numpy(dtype=float)
    y = np.log(np.maximum(frame["closed_test_loss"].to_numpy(dtype=float), 1e-12))
    b1 = int(row["boundary1"]) if row is not None and np.isfinite(row["boundary1"]) else int(0.2 * len(x))
    b2 = int(row["boundary2"]) if row is not None and np.isfinite(row["boundary2"]) else int(0.7 * len(x))

    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.25), gridspec_kw={"width_ratios": [1.25, 1.0, 1.15]})
    for label, ax in zip(["(a)", "(b)", "(c)"], axes, strict=False):
        _panel_label(ax, label)
    stage_colors = sns.color_palette("deep", 4)
    axes[0].plot(x, frame["closed_test_loss"], color=sns.color_palette("deep")[0], lw=1.8)
    axes[0].axvspan(b1, b2, color=stage_colors[2], alpha=0.16, label="candidate slow phase")
    axes[0].axvline(b1, color="0.05", ls="--", lw=1.8, label=r"$\tau_1$")
    axes[0].axvline(b2, color=stage_colors[3], ls="-.", lw=1.8, label=r"$\tau_2$")
    axes[0].set(xlabel="Epoch", ylabel="Deployed loss", yscale="log", title=f"Seed {seed}")
    axes[0].legend(frameon=False, fontsize=7)

    axes[1].plot(x, y, color="0.72", lw=1.0, label="log loss")
    for start, end, color, label in [
        (0, b1, stage_colors[0], "segment 1"),
        (b1, b2, stage_colors[2], "segment 2"),
        (b2, len(x) - 1, stage_colors[3], "segment 3"),
    ]:
        segment = (x >= start) & (x <= end)
        if np.sum(segment) >= 2:
            coeff = np.polyfit(x[segment], y[segment], 1)
            axes[1].plot(x[segment], np.polyval(coeff, x[segment]), color=color, lw=2.0, label=label)
    axes[1].axvline(b1, color="0.05", ls="--", lw=1.2)
    axes[1].axvline(b2, color=stage_colors[3], ls="-.", lw=1.2)
    axes[1].set(xlabel="Epoch", ylabel="log deployed loss", title="Segmented fit")
    axes[1].legend(frameon=False, fontsize=7)

    _plot_stage_raster(axes[2], details, len(x), stage_colors)
    sns.despine(fig)
    fig.tight_layout()
    path = out / "figure_3_stage_analysis.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _plot_stage_raster(ax, details: pd.DataFrame | None, n_epochs: int, colors) -> None:
    if details is None or details.empty:
        ax.text(0.5, 0.5, "No aggregate stage details", ha="center", va="center", transform=ax.transAxes)
        ax.axis("off")
        return
    rows = details.sort_values("seed").reset_index(drop=True)
    palette = np.array([colors[0], colors[2], colors[3]], dtype=float)
    raster = np.zeros((len(rows), n_epochs, 3), dtype=float)
    y_positions = np.arange(len(rows))
    tau1 = []
    tau2 = []
    for row_idx, row in rows.iterrows():
        raw_b1 = float(row.get("boundary1", n_epochs // 5))
        raw_b2 = float(row.get("boundary2", int(0.7 * n_epochs)))
        if not np.isfinite(raw_b1):
            raw_b1 = n_epochs // 5
        if not np.isfinite(raw_b2):
            raw_b2 = int(0.7 * n_epochs)
        b1 = int(np.clip(raw_b1, 1, n_epochs - 2))
        b2 = int(np.clip(raw_b2, b1 + 1, n_epochs - 1))
        raster[row_idx, :b1] = palette[0]
        raster[row_idx, b1:b2] = palette[1]
        raster[row_idx, b2:] = palette[2]
        tau1.append(b1)
        tau2.append(b2)
    ax.imshow(raster, aspect="auto", interpolation="nearest", extent=[0, n_epochs, len(rows) - 0.5, -0.5])
    ax.scatter(tau1, y_positions, s=9, color="0.02", linewidths=0, label=r"$\tau_1$")
    ax.scatter(tau2, y_positions, s=12, facecolors="white", edgecolors="0.02", linewidths=0.6, label=r"$\tau_2$")
    slow = int((rows["stage1_fast"].fillna(False) & rows["stage2_slow"].fillna(False)).sum())
    strict = int(rows["claim_C2_changepoint_three_stage"].fillna(False).sum())
    ax.text(
        0.98,
        0.05,
        f"slow: {slow}/{len(rows)}\nstrict: {strict}/{len(rows)}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=7,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.85", "alpha": 0.92},
    )
    ax.set(xlabel="Epoch", ylabel="Seed", title="All original seeds")
    ax.legend(frameon=False, fontsize=7, loc="upper right")


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
        "A1 proxy": "c4_proxy_support",
    }
    heat = rob.set_index("setting")[[*columns.values()]].rename(columns={v: k for k, v in columns.items()})
    counts = rob.set_index("setting")["n"].reindex(heat.index).fillna(0).astype(int)
    labels = heat.copy().astype(object)
    for row_name in heat.index:
        n = int(counts.loc[row_name])
        for col_name in heat.columns:
            labels.loc[row_name, col_name] = f"{int(round(float(heat.loc[row_name, col_name]) * n))}/{n}"
    failure_rows = heat.index.astype(str) == "no_clip"
    mask = np.tile(failure_rows[:, None], (1, heat.shape[1]))
    fig, ax = plt.subplots(figsize=(7.4, 4.2))
    sns.heatmap(
        heat,
        ax=ax,
        vmin=0,
        vmax=1,
        cmap="viridis",
        annot=labels,
        fmt="",
        mask=mask,
        cbar_kws={"label": "Support fraction"},
    )
    for row_idx, failed in enumerate(failure_rows):
        if not failed:
            continue
        for col_idx in range(heat.shape[1]):
            ax.add_patch(plt.Rectangle((col_idx, row_idx), 1, 1, facecolor="0.82", edgecolor="0.35", hatch="///", lw=0.5))
            ax.text(col_idx + 0.5, row_idx + 0.5, "F", ha="center", va="center", color="0.1", fontsize=9, fontweight="bold")
    ax.set(xlabel="Claim/audit diagnostic", ylabel="Perturbation")
    fig.tight_layout()
    path = out / "figure_4_robustness_heatmap.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_tradeoff(tradeoff: pd.DataFrame, out: Path) -> Path:
    df = tradeoff.copy()
    df["condition_value"] = df["tradeoff_condition"].map(_parse_control_penalty)
    df = df.sort_values(["condition_value", "seed"])
    component_specs = [
        ("Union", "claim_C4_tradeoff_quantified", "conditional_tradeoff_fraction", "tradeoff_step_count"),
        ("Loss", "claim_A1_loss_tradeoff", "conditional_loss_tradeoff_fraction", "loss_tradeoff_step_count"),
        ("Radius", "claim_A1_radius_tradeoff", "conditional_radius_tradeoff_fraction", "radius_tradeoff_step_count"),
        ("Both", "claim_A1_both_tradeoff", "conditional_both_tradeoff_fraction", "both_tradeoff_step_count"),
    ]
    rows = []
    for label, support_col, fraction_col, count_col in component_specs:
        fractions = pd.to_numeric(df.get(fraction_col, pd.Series(dtype=float)), errors="coerce")
        counts = pd.to_numeric(df.get(count_col, pd.Series(dtype=float)), errors="coerce")
        if support_col in df:
            support = _bool_series(df[support_col])
        else:
            support = (fractions >= 0.1) & (counts >= 3)
        rows.append(
            {
                "label": label,
                "support": float(support.fillna(False).mean()),
                "support_count": int(support.fillna(False).sum()),
                "n": int(len(df)),
                "fractions": fractions.dropna().to_numpy(dtype=float),
            }
        )
    labels = [row["label"] for row in rows]
    x = np.arange(len(rows))
    colors = sns.color_palette("deep", len(rows))
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.1), gridspec_kw={"width_ratios": [0.95, 1.35]})
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")
    axes[0].bar(x, [row["support"] for row in rows], color=colors)
    for idx, row in enumerate(rows):
        n = int(row["n"])
        count = int(row["support_count"])
        axes[0].text(idx, min(1.09, float(row["support"]) + 0.035), f"{count}/{n}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, labels, rotation=20, ha="right")
    axes[0].set(ylim=(0, 1.14), xlabel="A1 component criterion", ylabel="Support fraction")

    for idx, row in enumerate(rows):
        values = row["fractions"]
        offsets = np.linspace(-0.13, 0.13, len(values)) if len(values) > 1 else np.array([0.0])
        axes[1].scatter(np.full(len(values), idx) + offsets, values, color=colors[idx], s=18, alpha=0.5, linewidths=0, zorder=2)
        if len(values):
            axes[1].scatter(idx, np.mean(values), color=colors[idx], s=60, marker="D", edgecolor="white", linewidth=0.8, zorder=4)
    axes[1].set_xticks(x, labels, rotation=20, ha="right")
    axes[1].set(
        xlabel="A1 component criterion",
        ylabel="Conditional tradeoff fraction",
        ylim=(0, max(0.42, float(np.nanmax(df["conditional_tradeoff_fraction"])) * 1.15)),
    )
    sns.despine(fig)
    fig.tight_layout()
    path = out / "figure_7_tradeoff_analysis.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _parse_control_penalty(value: object) -> float:
    text = str(value).removeprefix("control_penalty_").replace("p", ".")
    try:
        return float(text)
    except ValueError:
        return float("nan")


def _format_penalty(value: float) -> str:
    if not np.isfinite(value):
        return "unknown"
    return f"{value:g}"


def _fraction_bool(values: pd.Series) -> float:
    return float(_bool_series(values).mean())


def _bool_series(values: pd.Series) -> pd.Series:
    if values.dtype == bool:
        return values
    if pd.api.types.is_numeric_dtype(values):
        return values.fillna(0).astype(bool)
    normalized = values.astype(str).str.lower().str.strip()
    return normalized.isin({"true", "1", "yes"})


def _figure_coupled_spectra(frames: list[pd.DataFrame], processed: Path, out: Path) -> Path:
    colors = sns.color_palette("deep")
    epochs, coupled = _stack(frames, "closed_coupled_radius")
    _, rnn = _stack(frames, "closed_rnn_radius")
    metrics = _read_csv(processed / "recomputed_timeseries_metrics.csv")
    crossings = metrics[metrics["kind"] == "original"]["stability_crossing"].dropna().to_numpy(dtype=float) if metrics is not None else []
    fig, axes = plt.subplots(1, 2, figsize=(8.2, 3.25), gridspec_kw={"width_ratios": [1.45, 0.9]})
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")
    ax = axes[0]
    _plot_median_band(ax, epochs, coupled, r"$\rho_{\mathrm{coup}}$ median/IQR", colors[0])
    _plot_median_band(ax, epochs, rnn, r"$\rho_{\mathrm{RNN}}$ median/IQR", colors[4])
    ax.axhline(1.0, color="0.15", ls="--", lw=1.0, label=r"$\rho=1$ stability boundary")
    if len(crossings):
        median_cross = float(np.nanmedian(crossings))
        ax.axvline(median_cross, color=colors[3], ls=":", lw=1.4)
        ax.plot(crossings, np.full_like(crossings, 0.035, dtype=float), "|", color=colors[3], ms=7, alpha=0.65)
        ax.text(median_cross + 18, 0.08, f"median crossing: {median_cross:.0f}", color=colors[3], fontsize=8)
    ax.set(xlabel="Epoch", ylabel="Spectral radius", ylim=(0, max(1.25, np.nanquantile(coupled, 0.98) * 1.05)))
    ax.legend(frameon=True, facecolor="white", edgecolor="none", loc="upper right", fontsize=7)

    hist_ax = axes[1]
    if len(crossings):
        bins = np.arange(np.nanmin(crossings) - 0.5, np.nanmax(crossings) + 1.5, 2)
        hist_ax.hist(crossings, bins=bins, color=colors[3], alpha=0.75, edgecolor="white")
        hist_ax.axvline(np.nanmedian(crossings), color="0.1", ls="--", lw=1.0)
        hist_ax.set(xlabel="Crossing epoch", ylabel="Seeds", title="Stability crossings")
    else:
        hist_ax.text(0.5, 0.5, "No crossings", ha="center", va="center", transform=hist_ax.transAxes)
        hist_ax.set_axis_off()
    sns.despine(fig)
    fig.tight_layout()
    path = out / "figure_5_coupled_spectral_analysis.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path


def _figure_generalization(setting_summary: pd.DataFrame, processed: Path, out: Path) -> Path:
    gen = setting_summary[setting_summary["kind"] == "generalization"].copy()
    metrics = _read_csv(processed / "recomputed_timeseries_metrics.csv")
    if metrics is not None and not metrics.empty:
        metrics = metrics[metrics["kind"] == "generalization"].copy()
    labels = {
        "tanh_rnn": "tanh RNN",
        "gru": "GRU",
        "low_rank": "low-rank",
        "tracking_task": "tracking",
        "ring_path_integration": "ring",
        "ring_partial_obs_gru": "partial ring GRU",
        "ring_partial_obs_tanh": "partial ring tanh",
    }
    order = [
        "tanh_rnn",
        "gru",
        "low_rank",
        "tracking_task",
        "ring_path_integration",
        "ring_partial_obs_gru",
        "ring_partial_obs_tanh",
    ]
    gen["setting"] = pd.Categorical(gen["setting"], categories=order, ordered=True)
    gen = gen.sort_values("setting")
    x = np.arange(len(gen))
    fig, axes = plt.subplots(1, 2, figsize=(8.0, 3.1))
    _panel_label(axes[0], "(a)")
    _panel_label(axes[1], "(b)")
    colors = sns.color_palette("deep", len(gen))
    axes[0].bar(x, gen["c1_loss_support"], color=colors)
    for idx, row in gen.reset_index(drop=True).iterrows():
        n = int(row["n"])
        count = int(round(float(row["c1_loss_support"]) * n))
        axes[0].text(idx, min(1.09, float(row["c1_loss_support"]) + 0.035), f"{count}/{n}", ha="center", va="bottom", fontsize=8)
    axes[0].set_xticks(x, [labels.get(str(v), str(v)) for v in gen["setting"]], rotation=25, ha="right")
    axes[0].set(ylim=(0, 1.14), ylabel="C1 support fraction", xlabel="Variant")

    all_gaps = []
    for idx, row in gen.reset_index(drop=True).iterrows():
        setting = str(row["setting"])
        if metrics is not None and not metrics.empty:
            gaps = metrics[metrics["setting"] == setting]["deployed_loss_gap"].to_numpy(dtype=float)
        else:
            gaps = np.array([float(row["mean_deployed_loss_gap"])])
        gaps = gaps[np.isfinite(gaps)]
        if gaps.size == 0:
            continue
        all_gaps.extend(gaps.tolist())
        offsets = np.linspace(-0.13, 0.13, len(gaps)) if len(gaps) > 1 else np.array([0.0])
        axes[1].scatter(np.full(len(gaps), idx) + offsets, gaps, color=colors[idx], s=18, alpha=0.55, linewidths=0, zorder=2)
        axes[1].scatter(idx, np.mean(gaps), color=colors[idx], s=62, marker="D", edgecolor="white", linewidth=0.8, zorder=4)
    axes[1].axhline(0, color="0.2", lw=0.8)
    axes[1].set_xticks(x, [labels.get(str(v), str(v)) for v in gen["setting"]], rotation=25, ha="right")
    axes[1].set_yscale("symlog", linthresh=1e-3, linscale=0.7)
    if all_gaps:
        max_abs = max(abs(float(np.nanmin(all_gaps))), abs(float(np.nanmax(all_gaps))))
        min_gap = float(np.nanmin(all_gaps))
        axes[1].set_ylim(min(-0.002, min_gap * 1.3) if min_gap < 0 else 0, max_abs * 2.0)
    ring_rows = gen.reset_index(drop=True)
    if "ring_path_integration" in set(ring_rows["setting"].astype(str)):
        ring_idx = int(ring_rows.index[ring_rows["setting"].astype(str) == "ring_path_integration"][0])
        ring_gap = float(ring_rows.iloc[ring_idx]["mean_deployed_loss_gap"])
        axes[1].annotate(
            "near-zero gap",
            xy=(ring_idx, ring_gap),
            xytext=(ring_idx - 1.35, 0.006),
            arrowprops={"arrowstyle": "->", "lw": 0.8, "color": "0.25"},
            fontsize=8,
            ha="right",
        )
    axes[1].set(ylabel="Seed deployed-loss gap (symlog)", xlabel="Variant")
    sns.despine(fig)
    fig.tight_layout()
    path = out / "figure_6_generalization.png"
    fig.savefig(path, dpi=240, bbox_inches="tight")
    plt.close(fig)
    return path
