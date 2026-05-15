from __future__ import annotations

import math

import torch
from torch import nn


class RecurrentController(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        activation: str = "linear",
        recurrent_scale: float = 0.1,
        input_scale: float = 1.0,
        output_scale: float = 1.0,
        leak: float = 1.0,
        train_input: bool = True,
        train_recurrent: bool = True,
        train_output: bool = True,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation_name = activation
        self.leak = float(leak)
        self.Wih = nn.Parameter(torch.empty(hidden_size, input_size))
        self.Whh = nn.Parameter(torch.empty(hidden_size, hidden_size))
        self.Who = nn.Parameter(torch.empty(output_size, hidden_size))
        self.bias_h = nn.Parameter(torch.zeros(hidden_size), requires_grad=bias)
        self.bias_o = nn.Parameter(torch.zeros(output_size), requires_grad=bias)
        self.reset_parameters(recurrent_scale, input_scale, output_scale)
        self.Wih.requires_grad = train_input
        self.Whh.requires_grad = train_recurrent
        self.Who.requires_grad = train_output

    def reset_parameters(self, recurrent_scale: float, input_scale: float, output_scale: float) -> None:
        with torch.no_grad():
            nn.init.normal_(self.Wih, mean=0.0, std=input_scale / math.sqrt(max(1, self.hidden_size)))
            nn.init.normal_(self.Whh, mean=0.0, std=recurrent_scale / math.sqrt(max(1, self.hidden_size)))
            nn.init.normal_(self.Who, mean=0.0, std=output_scale / math.sqrt(max(1, self.hidden_size)))

    def initial_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.zeros((batch_size, self.hidden_size), dtype=torch.float32, device=device)

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = observation @ self.Wih.T + hidden @ self.Whh.T + self.bias_h
        updated = self._activation(pre)
        hidden = (1.0 - self.leak) * hidden + self.leak * updated
        action = hidden @ self.Who.T + self.bias_o
        return action, hidden

    def _activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation_name == "linear":
            return x
        if self.activation_name == "tanh":
            return torch.tanh(x)
        if self.activation_name == "relu":
            return torch.relu(x)
        raise ValueError(f"Unknown activation {self.activation_name!r}")
