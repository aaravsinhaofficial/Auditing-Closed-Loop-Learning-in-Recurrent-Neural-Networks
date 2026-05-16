from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class SpectralStageResult:
    stage1_end: int | None
    stage2_end: int | None
    lambda3_growth: float
    stage1_supported: bool
    stage2_supported: bool
    stage3_supported: bool

    @property
    def three_stage_supported(self) -> bool:
        return bool(self.stage1_supported and self.stage2_supported and self.stage3_supported)

    def as_dict(self) -> dict[str, Any]:
        return {
            "spectral_stage1_end": np.nan if self.stage1_end is None else int(self.stage1_end),
            "spectral_stage2_end": np.nan if self.stage2_end is None else int(self.stage2_end),
            "spectral_lambda3_growth": float(self.lambda3_growth),
            "claim_C2_spectral_stage1": bool(self.stage1_supported),
            "claim_C2_spectral_stage2": bool(self.stage2_supported),
            "claim_C2_spectral_stage3": bool(self.stage3_supported),
            "claim_C2_spectral_three_stage": bool(self.three_stage_supported),
        }


def detect_spectral_stages(
    frame: pd.DataFrame,
    gain_threshold: float = -0.01,
    persistence: int = 5,
    lambda3_growth_threshold: float = 0.01,
) -> SpectralStageResult:
    """Operationalize Ger-Barak-style spectral stages from logged summaries.

    Stage 1 support: an effective negative-position policy together with an
    unstable complex coupled mode. Stage 2 support: the dominant coupled radius
    crosses inside the unit disk after Stage 1. Stage 3 support: the largest
    real eigenvalue grows after the stability transition.
    """

    if frame.empty:
        return _empty()
    required = {"closed_gain_0", "closed_coupled_radius", "closed_coupled_has_unstable_complex", "closed_coupled_third_real_abs"}
    if not required.issubset(frame.columns):
        return _empty()

    epochs = frame["epoch"].to_numpy(dtype=int) if "epoch" in frame else np.arange(len(frame))
    gain0 = frame["closed_gain_0"].to_numpy(dtype=float)
    radius = frame["closed_coupled_radius"].to_numpy(dtype=float)
    has_unstable_complex = frame["closed_coupled_has_unstable_complex"].to_numpy(dtype=float) > 0.5
    lambda3 = frame["closed_coupled_third_real_abs"].to_numpy(dtype=float)

    stage1_mask = np.isfinite(gain0) & (gain0 < gain_threshold) & has_unstable_complex & np.isfinite(radius) & (radius > 1.0)
    stage1_idx = _first_persistent(stage1_mask, persistence, start=0)
    stage1_end = None if stage1_idx is None else int(epochs[stage1_idx])

    stage2_mask = np.isfinite(radius) & (radius < 1.0)
    stage2_idx = _first_persistent(stage2_mask, persistence, start=0 if stage1_idx is None else stage1_idx)
    stage2_end = None if stage2_idx is None else int(epochs[stage2_idx])

    growth = float("nan")
    stage3_supported = False
    if stage2_idx is not None and np.isfinite(lambda3[stage2_idx]):
        tail = lambda3[stage2_idx:]
        finite_tail = tail[np.isfinite(tail)]
        if finite_tail.size:
            growth = float(np.nanmax(finite_tail) - lambda3[stage2_idx])
            stage3_supported = bool(growth > lambda3_growth_threshold)

    return SpectralStageResult(
        stage1_end=stage1_end,
        stage2_end=stage2_end,
        lambda3_growth=growth,
        stage1_supported=stage1_idx is not None,
        stage2_supported=stage2_idx is not None,
        stage3_supported=stage3_supported,
    )


def _first_persistent(mask: np.ndarray, persistence: int, start: int = 0) -> int | None:
    if mask.size == 0:
        return None
    persistence = max(1, int(persistence))
    for idx in range(max(0, start), len(mask)):
        if bool(mask[idx]) and bool(np.all(mask[idx : min(len(mask), idx + persistence)])):
            return idx
    return None


def _empty() -> SpectralStageResult:
    return SpectralStageResult(
        stage1_end=None,
        stage2_end=None,
        lambda3_growth=float("nan"),
        stage1_supported=False,
        stage2_supported=False,
        stage3_supported=False,
    )
