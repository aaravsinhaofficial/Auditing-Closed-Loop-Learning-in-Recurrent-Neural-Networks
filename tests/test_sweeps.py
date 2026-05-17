from closed_loop_repro.io import write_json
from closed_loop_repro.sweeps.generalization_sweep import run_generalization_sweep


def test_generalization_sweep_skips_existing_runs(tmp_path):
    existing = tmp_path / "generalization_toy" / "seed_0000" / "metrics.json"
    write_json({"experiment": "generalization_toy", "seed": 0, "final_closed_test_loss": 1.0, "final_open_test_loss": 2.0}, existing)
    rows = run_generalization_sweep(
        {
            "output_dir": str(tmp_path),
            "seeds": [0],
            "base": {
                "task": {"name": "tracking", "init_low": -2.0},
                "model": {"name": "linear_rnn", "hidden_size": 4},
                "training": {"epochs": 1, "steps": 1},
            },
            "variants": [{"name": "toy", "patch": {}}],
        }
    )
    assert len(rows) == 1
    assert rows[0]["variant"] == "toy"
    assert rows[0]["final_open_test_loss"] == 2.0
