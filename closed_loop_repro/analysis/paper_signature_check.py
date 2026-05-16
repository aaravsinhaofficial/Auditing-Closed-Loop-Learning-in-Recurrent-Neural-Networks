from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from closed_loop_repro.analysis.spectral_stages import detect_spectral_stages
from closed_loop_repro.io import ensure_dir


def run_signature_check(result_dir: str | Path, out: str | Path) -> dict[str, Path]:
    result_dir = Path(result_dir)
    out = ensure_dir(out)
    frame = pd.read_csv(result_dir / "timeseries.csv")
    metrics = _read_json(result_dir / "metrics.json")
    spectral = detect_spectral_stages(frame)

    summary = _summarize(frame, metrics, spectral.as_dict())
    summary_path = out / "paper_signature_summary.csv"
    pd.DataFrame([summary]).to_csv(summary_path, index=False)
    md_path = out / "paper_signature_summary.md"
    md_path.write_text(_summary_markdown(summary), encoding="utf-8")

    loss_path = out / "paper_signature_loss.png"
    spectra_path = out / "paper_signature_spectra.png"
    _plot_loss(frame, loss_path)
    _plot_spectra(frame, spectra_path, summary)
    return {"summary_csv": summary_path, "summary_md": md_path, "loss_png": loss_path, "spectra_png": spectra_path}


def _summarize(frame: pd.DataFrame, metrics: dict[str, Any], spectral: dict[str, Any]) -> dict[str, Any]:
    closed = pd.to_numeric(frame["closed_test_loss"], errors="coerce").to_numpy(dtype=float)
    open_loop = pd.to_numeric(frame["open_test_loss"], errors="coerce").to_numpy(dtype=float)
    gain = pd.to_numeric(frame.get("closed_gain_0", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float)
    radius = pd.to_numeric(frame.get("closed_coupled_radius", pd.Series(np.nan, index=frame.index)), errors="coerce").to_numpy(dtype=float)
    unstable = pd.to_numeric(
        frame.get("closed_coupled_has_unstable_complex", pd.Series(np.nan, index=frame.index)), errors="coerce"
    ).to_numpy(dtype=float)

    peak_epoch = int(np.nanargmax(open_loop)) if np.any(np.isfinite(open_loop)) else -1
    first_stable = _first_persistent(np.isfinite(radius) & (radius < 1.0), persistence=5)
    first_unstable_complex = _first_persistent(np.isfinite(unstable) & (unstable > 0.5), persistence=5)
    first_negative_gain = _first_persistent(np.isfinite(gain) & (gain < -0.01), persistence=5)
    closed_min_epoch = int(np.nanargmin(closed)) if np.any(np.isfinite(closed)) else -1

    return {
        "result_dir": str(metrics.get("result_dir", "")),
        "experiment": metrics.get("experiment", result_dir_name(frame)),
        "seed": metrics.get("seed", np.nan),
        "protocol_target": metrics.get("protocol_target", "unspecified"),
        "initial_state_range": metrics.get("initial_state_range", ""),
        "loss_timing": metrics.get("loss_timing", ""),
        "normalize_loss_by_steps": metrics.get("normalize_loss_by_steps", ""),
        "open_loop_input": metrics.get("open_loop_input", ""),
        "control_penalty": metrics.get("control_penalty", np.nan),
        "closed_initial_loss": float(closed[0]),
        "closed_final_loss": float(closed[-1]),
        "closed_min_loss": float(np.nanmin(closed)),
        "closed_min_epoch": closed_min_epoch,
        "open_initial_loss": float(open_loop[0]),
        "open_final_loss": float(open_loop[-1]),
        "open_peak_loss": float(np.nanmax(open_loop)),
        "open_peak_epoch": peak_epoch,
        "open_peak_ratio_to_initial": float(np.nanmax(open_loop) / max(float(open_loop[0]), 1e-12)),
        "open_peak_ratio_to_final": float(np.nanmax(open_loop) / max(float(open_loop[-1]), 1e-12)),
        "open_peak_after_epoch0": bool(peak_epoch > 0),
        "open_peak_signature": bool(peak_epoch > 0 and np.nanmax(open_loop) > 1.5 * max(float(open_loop[0]), 1e-12)),
        "sigma_zm_min": float(np.nanmin(gain)) if np.any(np.isfinite(gain)) else float("nan"),
        "sigma_zm_final": float(gain[-1]) if np.isfinite(gain[-1]) else float("nan"),
        "first_negative_sigma_zm_epoch": first_negative_gain,
        "unstable_complex_present": bool(first_unstable_complex is not None),
        "first_unstable_complex_epoch": first_unstable_complex if first_unstable_complex is not None else np.nan,
        "first_stable_coupled_epoch": first_stable if first_stable is not None else np.nan,
        "spectral_stage1_end": spectral.get("spectral_stage1_end", np.nan),
        "spectral_stage2_end": spectral.get("spectral_stage2_end", np.nan),
        "spectral_lambda3_growth": spectral.get("spectral_lambda3_growth", np.nan),
        "claim_C2_spectral_three_stage": spectral.get("claim_C2_spectral_three_stage", False),
    }


def result_dir_name(frame: pd.DataFrame) -> str:
    del frame
    return "unknown"


def _summary_markdown(summary: dict[str, Any]) -> str:
    rows = ["# Paper Signature Sanity Check", ""]
    for key, value in summary.items():
        rows.append(f"- `{key}`: {value}")
    rows.append("")
    rows.append(
        "Interpretation: compare the loss and spectra figures against the original paper before launching large sweeps. "
        "The strongest match should show an open-loop deployed-loss peak, negative early position gain, unstable complex "
        "coupled eigenvalues, and a later crossing of the coupled radius into the unit disk."
    )
    return "\n".join(rows)


def _plot_loss(frame: pd.DataFrame, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    epoch = frame["epoch"].to_numpy(dtype=float)
    ax.plot(epoch, frame["closed_test_loss"], label="closed deployed", color="#1f77b4", linewidth=2.0)
    ax.plot(epoch, frame["open_test_loss"], label="open deployed", color="#d62728", linewidth=2.0)
    ax.set_yscale("log")
    ax.set_xlabel("epoch")
    ax.set_ylabel("deployed loss")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_spectra(frame: pd.DataFrame, path: Path, summary: dict[str, Any]) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    epoch = frame["epoch"].to_numpy(dtype=float)

    axes[0].plot(epoch, frame.get("closed_gain_0", np.nan), color="#2ca02c", linewidth=2.0)
    axes[0].axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("sigma_zm / gain on position")
    axes[0].set_title("effective policy")

    axes[1].plot(epoch, frame.get("closed_coupled_radius", np.nan), label="rho(P)", color="#1f77b4", linewidth=2.0)
    if "closed_coupled_unstable_complex_abs" in frame:
        axes[1].plot(
            epoch,
            frame["closed_coupled_unstable_complex_abs"],
            label="unstable complex |lambda|",
            color="#9467bd",
            linewidth=1.5,
            alpha=0.85,
        )
    axes[1].axhline(1.0, color="black", linewidth=1.0, linestyle="--", alpha=0.6)
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("spectral radius")
    axes[1].set_title("coupled stability")
    axes[1].legend(frameon=False)

    _plot_eig_plane(frame, axes[2], summary)
    fig.tight_layout()
    fig.savefig(path, dpi=200)
    plt.close(fig)


def _plot_eig_plane(frame: pd.DataFrame, ax: plt.Axes, summary: dict[str, Any]) -> None:
    if "closed_coupled_eigvals" not in frame:
        ax.text(0.5, 0.5, "set save_eigenvalues: true", ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()
        return
    selected = [0, _as_int(summary.get("spectral_stage1_end")), _as_int(summary.get("spectral_stage2_end")), int(frame["epoch"].iloc[-1])]
    selected = [idx for idx in dict.fromkeys(selected) if idx is not None and 0 <= idx < len(frame)]
    colors = ["#7f7f7f", "#ff7f0e", "#bcbd22", "#1f77b4"]
    theta = np.linspace(0, 2 * np.pi, 256)
    ax.plot(np.cos(theta), np.sin(theta), color="black", linewidth=1.0, alpha=0.35)
    for color, idx in zip(colors, selected):
        values = _parse_eigs(frame["closed_coupled_eigvals"].iloc[idx])
        if values.size:
            ax.scatter(values.real, values.imag, s=10, alpha=0.55, label=f"epoch {int(frame['epoch'].iloc[idx])}", color=color)
    ax.axhline(0, color="black", linewidth=0.8, alpha=0.25)
    ax.axvline(0, color="black", linewidth=0.8, alpha=0.25)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel("real")
    ax.set_ylabel("imaginary")
    ax.set_title("eig(P)")
    ax.legend(frameon=False, fontsize=8)


def _parse_eigs(payload: str) -> np.ndarray:
    try:
        raw = json.loads(payload)
    except Exception:
        return np.asarray([], dtype=np.complex128)
    return np.asarray([complex(float(real), float(imag)) for real, imag in raw], dtype=np.complex128)


def _first_persistent(mask: np.ndarray, persistence: int) -> int | None:
    for idx in range(0, len(mask)):
        if bool(mask[idx]) and bool(np.all(mask[idx : min(len(mask), idx + persistence)])):
            return idx
    return None


def _as_int(value: Any) -> int | None:
    try:
        if pd.isna(value):
            return None
        return int(value)
    except Exception:
        return None


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check one run for Ger-Barak paper-level signatures.")
    parser.add_argument("--result-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    print(run_signature_check(args.result_dir, args.out))


if __name__ == "__main__":
    main()
