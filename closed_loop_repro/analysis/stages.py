from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class StageResult:
    stage1_end: int
    plateau_end: int
    labels: np.ndarray
    plateau_detected: bool
    stability_crossing: int | None

    def as_dict(self) -> dict[str, object]:
        return {
            "stage1_end": int(self.stage1_end),
            "plateau_end": int(self.plateau_end),
            "plateau_detected": bool(self.plateau_detected),
            "plateau_length": int(max(0, self.plateau_end - self.stage1_end)),
            "stability_crossing": None if self.stability_crossing is None else int(self.stability_crossing),
        }


def detect_stages(
    loss: np.ndarray | list[float],
    coupled_radius: np.ndarray | list[float] | None = None,
    plateau_slope: float = 0.015,
    improvement_slope: float = 0.025,
    min_plateau: int = 8,
    stable_threshold: float = 1.0,
) -> StageResult:
    loss = np.asarray(loss, dtype=float)
    if loss.ndim != 1 or len(loss) < 3:
        labels = np.ones(len(loss), dtype=int)
        return StageResult(0, len(loss), labels, False, None)
    if not np.all(np.isfinite(loss)):
        labels = np.ones(len(loss), dtype=int)
        return StageResult(0, len(loss), labels, False, None)

    y = np.log(np.maximum(loss, 1e-12))
    y = _smooth(y, window=max(3, min(11, len(y) // 8 * 2 + 1)))
    slope = np.gradient(y)
    stage1_end = _first_plateau_like(slope, plateau_slope, min_plateau)
    if stage1_end is None:
        stage1_end = max(1, len(loss) // 5)

    stability_crossing = None
    if coupled_radius is not None:
        radius = np.asarray(coupled_radius, dtype=float)
        finite = np.isfinite(radius)
        for idx in range(stage1_end, len(radius)):
            if finite[idx] and radius[idx] < stable_threshold:
                tail = radius[idx : min(len(radius), idx + min_plateau)]
                if len(tail) >= min(3, min_plateau) and np.nanmax(tail) < stable_threshold:
                    stability_crossing = idx
                    break

    plateau_end = None
    start = min(len(slope) - 1, stage1_end + min_plateau)
    for idx in range(start, len(slope)):
        if slope[idx] < -improvement_slope:
            plateau_end = idx
            break
    if plateau_end is None and stability_crossing is not None:
        plateau_end = stability_crossing
    if plateau_end is None:
        plateau_end = min(len(loss) - 1, max(stage1_end + min_plateau, int(0.7 * len(loss))))

    labels = np.ones(len(loss), dtype=int)
    labels[stage1_end:plateau_end] = 2
    labels[plateau_end:] = 3
    plateau_detected = plateau_end - stage1_end >= min_plateau
    return StageResult(int(stage1_end), int(plateau_end), labels, bool(plateau_detected), stability_crossing)


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(values) < window:
        return values
    pad = window // 2
    padded = np.pad(values, (pad, pad), mode="edge")
    kernel = np.ones(window) / window
    return np.convolve(padded, kernel, mode="valid")


def _first_plateau_like(slope: np.ndarray, threshold: float, min_plateau: int) -> int | None:
    for idx in range(1, max(2, len(slope) - min_plateau)):
        segment = slope[idx : idx + min_plateau]
        if np.all(np.abs(segment) <= threshold):
            return idx
    return None
