from __future__ import annotations

import math

import torch
from torch import nn

from closed_loop_repro.models.linear_rnn import RecurrentController


class LowRankController(RecurrentController):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        rank: int = 1,
        activation: str = "linear",
        recurrent_scale: float = 0.0,
        **kwargs,
    ) -> None:
        self.rank = rank
        super().__init__(input_size, hidden_size, output_size, activation=activation, recurrent_scale=0.0, **kwargs)
        self.U = nn.Parameter(torch.empty(hidden_size, rank))
        self.V = nn.Parameter(torch.empty(hidden_size, rank))
        self.register_buffer("W_random", torch.empty(hidden_size, hidden_size))
        with torch.no_grad():
            nn.init.normal_(self.U, mean=0.0, std=1.0 / math.sqrt(max(1, hidden_size)))
            nn.init.normal_(self.V, mean=0.0, std=1.0 / math.sqrt(max(1, hidden_size)))
            nn.init.normal_(self.W_random, mean=0.0, std=recurrent_scale / math.sqrt(max(1, hidden_size)))
            self.Whh.requires_grad = False

    @property
    def recurrent_matrix(self) -> torch.Tensor:
        return self.U @ self.V.T + self.W_random

    def forward(self, observation: torch.Tensor, hidden: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        pre = observation @ self.Wih.T + hidden @ self.recurrent_matrix.T + self.bias_h
        updated = self._activation(pre)
        hidden = (1.0 - self.leak) * hidden + self.leak * updated
        action = hidden @ self.Who.T + self.bias_o
        return action, hidden
