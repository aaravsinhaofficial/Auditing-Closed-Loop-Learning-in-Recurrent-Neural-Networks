from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from closed_loop_repro.tasks.base import ControlTask


@dataclass
class RingIntegrationTask(ControlTask):
    dt: float = 0.1
    feedback_strength: float = 1.0
    observation_noise: float = 0.0
    target_angle: float = 0.0
    velocity_penalty: float = 0.1

    state_dim: int = 2
    obs_dim: int = 3
    action_dim: int = 1

    def reset(self, batch_size: int, rng: np.random.Generator, device: torch.device) -> torch.Tensor:
        angle = rng.uniform(-np.pi, np.pi, size=(batch_size, 1))
        velocity = rng.uniform(-0.5, 0.5, size=(batch_size, 1))
        return torch.tensor(np.concatenate([angle, velocity], axis=1), dtype=torch.float32, device=device)

    def observe(self, state: torch.Tensor, step: int) -> torch.Tensor:
        del step
        err = _wrap_angle(state[:, :1] - self.target_angle)
        obs = torch.cat([torch.sin(err), torch.cos(err), state[:, 1:2]], dim=1)
        if self.observation_noise > 0:
            obs = obs + torch.randn_like(obs) * self.observation_noise
        return obs

    def step(self, state: torch.Tensor, action: torch.Tensor, step: int) -> torch.Tensor:
        del step
        velocity = state[:, 1:2] + self.dt * self.feedback_strength * action
        angle = _wrap_angle(state[:, :1] + self.dt * velocity)
        return torch.cat([angle, velocity], dim=1)

    def state_cost(self, state: torch.Tensor, step: int) -> torch.Tensor:
        del step
        err = _wrap_angle(state[:, 0] - self.target_angle)
        return err**2 + self.velocity_penalty * state[:, 1] ** 2


def _wrap_angle(angle: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(angle), torch.cos(angle))
