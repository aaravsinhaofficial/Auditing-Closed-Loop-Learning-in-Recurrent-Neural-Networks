from __future__ import annotations

from closed_loop_repro.models.linear_rnn import RecurrentController


class TanhRNNController(RecurrentController):
    def __init__(self, input_size: int, hidden_size: int, output_size: int, **kwargs) -> None:
        super().__init__(input_size, hidden_size, output_size, activation="tanh", **kwargs)
