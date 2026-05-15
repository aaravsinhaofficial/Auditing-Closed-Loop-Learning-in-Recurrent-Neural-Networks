from pathlib import Path

from closed_loop_repro.config import load_config
from closed_loop_repro.training.experiment import run_pair_experiment


def test_run_pair_writes_outputs(tmp_path):
    config = load_config("configs/smoke/double_integrator.yaml")
    config["training"]["epochs"] = 3
    config["training"]["steps"] = 4
    result = run_pair_experiment(config, output_dir=tmp_path)
    result_dir = Path(result["result_dir"])
    assert (result_dir / "metrics.json").exists()
    assert (result_dir / "timeseries.csv").exists()
