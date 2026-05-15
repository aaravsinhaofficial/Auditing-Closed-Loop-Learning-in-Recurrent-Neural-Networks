from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch

from closed_loop_repro.tasks.base import ControlTask


@dataclass
class TrackingTask(ControlTask):
    dt: float = 0.1
    total_time: float = 10.0
    feedback_strength: float = 1.0
    observation_noise: float = 0.0
    clamp: float | None = 10.0
    amplitudes_x: tuple[float, ...] = (1.5, 1.0)
    frequencies_x: tuple[float, ...] = (0.10, 0.30)
    amplitudes_y: tuple[float, ...] = (1.2, 0.8)
    frequencies_y: tuple[float, ...] = (0.20, 0.40)
    phases_x: tuple[float, ...] = (0.0, 1.3)
    phases_y: tuple[float, ...] = (0.7, 2.1)
    state_dim: int = 4
    obs_dim: int = 4
    action_dim: int = 2
    _A: torch.Tensor = field(init=False, repr=False)
    _B: torch.Tensor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        self._A = torch.tensor(
            [[1.0, self.dt, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, self.dt], [0.0, 0.0, 0.0, 1.0]],
            dtype=torch.float32,
        )
        self._B = torch.tensor(
            [[0.0, 0.0], [self.dt * self.feedback_strength, 0.0], [0.0, 0.0], [0.0, self.dt * self.feedback_strength]],
            dtype=torch.float32,
        )

    def reset(self, batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
        del rng
        return torch.zeros((batch_size, self.state_dim), dtype=torch.float32, device=device)

    def observe(self, state: torch.Tensor, step: int) -> torch.Tensor:
        ref = self.reference(step, state.device).repeat(state.shape[0], 1)
        obs = torch.stack([state[:, 0], state[:, 2], ref[:, 0], ref[:, 1]], dim=1)
        if self.observation_noise > 0:
            obs = obs + torch.randn_like(obs) * self.observation_noise
        return obs

    def step(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        del step
        next_state = state @ self._A.to(state.device).T + action @ self._B.to(state.device).T
        if self.clamp is not None:
            next_state = torch.clamp(next_state, -self.clamp, self.clamp)
        return next_state

    def state_cost(self, state: torch.Tensor, step: int) -> torch.Tensor:
        ref = self.reference(step, state.device).repeat(state.shape[0], 1)
        return (state[:, 0] - ref[:, 0]) ** 2 + (state[:, 2] - ref[:, 1]) ** 2

    def reference(self, step: int, device: torch.device) -> torch.Tensor:
        t = torch.tensor(step * self.dt, dtype=torch.float32, device=device)
        rx = torch.zeros((), dtype=torch.float32, device=device)
        ry = torch.zeros((), dtype=torch.float32, device=device)
        for amp, freq, phase in zip(self.amplitudes_x, self.frequencies_x, self.phases_x):
            rx = rx + float(amp) * torch.cos(2 * torch.pi * float(freq) * t + float(phase))
        for amp, freq, phase in zip(self.amplitudes_y, self.frequencies_y, self.phases_y):
            ry = ry + float(amp) * torch.cos(2 * torch.pi * float(freq) * t + float(phase))
        ramp = torch.clamp(t / 1.0, 0.0, 1.0)
        return torch.stack([rx * ramp, ry * ramp])

    def numpy_matrices(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        C = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]])
        return self._A.numpy(), self._B.numpy(), C
