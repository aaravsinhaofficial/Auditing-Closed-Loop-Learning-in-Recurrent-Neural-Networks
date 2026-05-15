from closed_loop_repro.models.gru import GRUController
from closed_loop_repro.models.linear_rnn import RecurrentController
from closed_loop_repro.models.low_rank_rnn import LowRankController
from closed_loop_repro.models.tanh_rnn import TanhRNNController


def make_controller(config, input_size: int, hidden_size: int, output_size: int):
    name = config.get("name", "tanh_rnn")
    kwargs = {k: v for k, v in config.items() if k != "name"}
    if name in {"linear_rnn", "linear"}:
        return RecurrentController(input_size, hidden_size, output_size, activation="linear", **kwargs)
    if name in {"tanh_rnn", "tanh"}:
        return TanhRNNController(input_size, hidden_size, output_size, **kwargs)
    if name == "gru":
        return GRUController(input_size, hidden_size, output_size, **kwargs)
    if name in {"low_rank", "low_rank_rnn"}:
        return LowRankController(input_size, hidden_size, output_size, **kwargs)
    raise ValueError(f"Unknown model {name!r}")
