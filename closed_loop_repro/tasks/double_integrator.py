from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from closed_loop_repro.tasks.base import ControlTask


@dataclass
class DoubleIntegratorTask(ControlTask):
    dt: float = 1.0
    feedback_strength: float = 1.0
    observation_noise: float = 0.0
    init_low: float = -1.0
    init_high: float = 1.0
    clamp: float | None = None

    state_dim: int = 2
    obs_dim: int = 1
    action_dim: int = 1

    def __post_init__(self) -> None:
        self.A = torch.tensor([[1.0, self.dt], [0.0, 1.0]], dtype=torch.float32)
        self.B = torch.tensor([[0.0], [self.dt * self.feedback_strength]], dtype=torch.float32)
        self.C = torch.tensor([[1.0, 0.0]], dtype=torch.float32)

    def reset(self, batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
        state = rng.uniform(self.init_low, self.init_high, size=(batch_size, self.state_dim))
        return torch.tensor(state, dtype=torch.float32, device=device)

    def observe(self, state: torch.Tensor, step: int) -> torch.Tensor:
        del step
        obs = state @ self.C.to(state.device).T
        if self.observation_noise > 0:
            obs = obs + torch.randn_like(obs) * self.observation_noise
        return obs

    def step(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        del step
        next_state = state @ self.A.to(state.device).T + action @ self.B.to(state.device).T
        if self.clamp is not None:
            next_state = torch.clamp(next_state, -self.clamp, self.clamp)
        return next_state

    def state_cost(self, state: torch.Tensor, step: int) -> torch.Tensor:
        del step
        return torch.sum(state**2, dim=-1)

    def numpy_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return self.A.numpy(), self.B.numpy(), self.C.numpy()
