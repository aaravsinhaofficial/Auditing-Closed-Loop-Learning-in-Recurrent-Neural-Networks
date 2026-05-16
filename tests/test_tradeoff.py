import numpy as np

from closed_loop_repro.sweeps.tradeoff_sweep import _summarize_tradeoff


def test_tradeoff_summary_detects_myopic_improvement_with_long_worsening():
    records = []
    for epoch in range(12):
        records.append(
            {
                "epoch": epoch,
                "closed_test_loss_T3": float(np.exp(-0.1 * epoch)),
                "closed_test_loss_T20": float(1.0 + 0.08 * epoch),
                "closed_coupled_radius": 0.9 + 0.01 * epoch,
            }
        )
    summary = _summarize_tradeoff(
        records,
        {
            "training": {"evaluation_horizons": [3, 20]},
            "analysis": {"tradeoff_min_log_improvement": 0.005, "tradeoff_min_log_worsening": 0.005},
        },
    )
    assert summary["tradeoff_evaluable"]
    assert summary["claim_C4_tradeoff_quantified"]
    assert summary["tradeoff_step_count"] >= 3
