from __future__ import annotations

import numpy as np


def bootstrap_ci(values, statistic=np.mean, confidence: float = 0.95, n_boot: int = 2000, seed: int = 0) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=arr.size, replace=True)
        stats.append(float(statistic(sample)))
    alpha = (1.0 - confidence) / 2.0
    return float(statistic(arr)), float(np.quantile(stats, alpha)), float(np.quantile(stats, 1 - alpha))


def fraction(values) -> float:
    arr = np.asarray(values, dtype=bool)
    if arr.size == 0:
        return float("nan")
    return float(np.mean(arr))


def pearson(x, y) -> float:
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)
    if np.sum(mask) < 2:
        return float("nan")
    return float(np.corrcoef(x[mask], y[mask])[0, 1])
