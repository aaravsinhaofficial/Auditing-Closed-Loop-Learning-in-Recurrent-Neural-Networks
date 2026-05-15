from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class RolloutBatch:
    states: torch.Tensor
    controls: torch.Tensor
    observations: torch.Tensor
    losses: torch.Tensor


class ControlTask:
    state_dim: int
    obs_dim: int
    action_dim: int

    def reset(self, batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
        raise NotImplementedError

    def observe(self, state: torch.Tensor, step: int) -> torch.Tensor:
        raise NotImplementedError

    def step(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        raise NotImplementedError

    def state_cost(self, state: torch.Tensor, step: int) -> torch.Tensor:
        raise NotImplementedError

    def numpy_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        return None
