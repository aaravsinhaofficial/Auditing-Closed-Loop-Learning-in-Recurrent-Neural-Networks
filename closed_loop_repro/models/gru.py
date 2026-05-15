from __future__ import annotations

import torch
from torch import nn


class GRUController(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, output_scale: float = 1.0, **_: object) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.cell = nn.GRUCell(input_size, hidden_size)
        self.readout = nn.Linear(hidden_size, output_size)
        with torch.no_grad():
            nn.init.normal_(self.readout.weight, mean=0.0, std=output_scale / max(1, hidden_size) ** 0.5)
            nn.init.zeros_(self.readout.bias)

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=device)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.cell(observation, hidden)
        return self.readout(hidden), hidden
