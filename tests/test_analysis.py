import numpy as np
import pandas as pd

from closed_loop_repro.analysis.gains import effective_gain
from closed_loop_repro.analysis.make_claim_tables import make_claim_tables
from closed_loop_repro.analysis.spectra import spectral_radius
from closed_loop_repro.analysis.spectral_stages import detect_spectral_stages
from closed_loop_repro.analysis.stage_changepoints import _analyze_one
from closed_loop_repro.analysis.stages import detect_stages
from closed_loop_repro.analysis.statistics import bootstrap_ci


def test_spectral_radius():
    matrix = np.array([[0.5, 0.0], [0.0, -2.0]])
    assert spectral_radius(matrix) == 2.0


def test_effective_gain_recovers_linear_map():
    states = np.random.default_rng(0).normal(size=(10, 20, 2))
    controls = states @ np.array([[2.0], [-1.0]])
    gain = effective_gain(states, controls)
    np.testing.assert_allclose(gain, [[2.0, -1.0]], atol=1e-6)


def test_stage_detection_synthetic_plateau():
    loss = np.r_[np.linspace(100, 10, 10), np.linspace(10, 9, 20), np.linspace(9, 1, 20)]
    result = detect_stages(loss, min_plateau=5)
    assert result.plateau_detected
    assert result.plateau_exit_detected
    assert result.plateau_exit_reason == "slope"
    assert result.stage1_end < result.plateau_end


def test_stage_detection_rejects_nonfinite_loss():
    result = detect_stages([1.0, np.nan, np.nan, np.nan], min_plateau=2)
    assert not result.plateau_detected
    assert result.stability_crossing is None
    assert result.plateau_exit_reason == "nonfinite_loss"


def test_stage_detection_allows_plateau_when_stability_crosses_at_start():
    loss = np.r_[np.linspace(100, 10, 8), np.linspace(10, 9, 25), np.linspace(9, 8, 10)]
    radius = np.r_[np.ones(8) * 1.1, np.ones(len(loss) - 8) * 0.9]
    result = detect_stages(loss, radius, min_plateau=5)
    assert result.plateau_detected
    assert result.plateau_end - result.stage1_end >= 5
    assert result.plateau_exit_reason == "fallback"
    assert not result.plateau_exit_detected


def test_spectral_stage_detector_uses_ger_barak_signatures():
    frame = pd.DataFrame(
        {
            "epoch": np.arange(12),
            "closed_gain_0": [-0.02] * 12,
            "closed_coupled_radius": [1.3, 1.2, 1.1, 1.05, 0.95, 0.9, 0.88, 0.86, 0.84, 0.82, 0.8, 0.79],
            "closed_coupled_has_unstable_complex": [1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
            "closed_coupled_third_real_abs": [0.1, 0.11, 0.12, 0.13, 0.14, 0.15, 0.18, 0.21, 0.25, 0.27, 0.29, 0.3],
        }
    )
    result = detect_spectral_stages(frame, persistence=2)
    assert result.stage1_supported
    assert result.stage2_supported
    assert result.stage3_supported
    assert result.three_stage_supported
    assert result.stage1_end == 0
    assert result.stage2_end == 4


def test_bootstrap_ci_smoke():
    mean, lo, hi = bootstrap_ci([1, 2, 3], n_boot=100, seed=0)
    assert lo <= mean <= hi


def test_stage_changepoint_synthetic_three_stage():
    x = np.arange(180)
    y = np.r_[
        8.0 - 0.08 * x[:50],
        4.0 - 0.002 * np.arange(70),
        3.86 - 0.03 * np.arange(60),
    ]
    result = _analyze_one(np.exp(y), min_segment=15, stride=1)
    assert result["valid"]
    assert result["claim_C2_changepoint_three_stage"]
    assert 35 <= result["boundary1"] <= 65
    assert 105 <= result["boundary2"] <= 135


def test_claim_tables_prefer_summary_csvs(tmp_path):
    raw = tmp_path / "raw"
    full = raw / "generalization_maximal"
    smoke = raw / "smoke_double_integrator"
    full.mkdir(parents=True)
    smoke.mkdir(parents=True)
    columns = {
        "experiment": ["full"],
        "seed": [0],
        "final_closed_test_loss": [1.0],
        "final_open_test_loss": [1.2],
        "trajectory_gain_distance": [0.0],
        "claim_C2_stages": [True],
        "claim_C3_stability_transition": [False],
        "open_loop_test_loss_spike": [True],
        "closed_recovered": [True],
        "final_closed_coupled_radius": [np.nan],
        "variant": ["toy"],
    }
    pd.DataFrame(columns).to_csv(full / "generalization_summary.csv", index=False)
    pd.DataFrame({**columns, "experiment": ["smoke"], "final_open_test_loss": [1.0]}).to_csv(
        smoke / "sweep_summary.csv", index=False
    )

    paths = make_claim_tables(raw, tmp_path / "processed")
    claim_df = pd.read_csv(paths["claim_csv"])
    c1 = claim_df.loc[claim_df["claim"] == "C1"].iloc[0]
    a2 = claim_df.loc[claim_df["claim"] == "A2"].iloc[0]
    assert c1["n"] == 1
    assert c1["support_fraction"] == 1.0
    assert a2["n"] == 1


def test_claim_tables_use_targeted_tradeoff_summary_for_a1(tmp_path):
    raw = tmp_path / "raw" / "tradeoff_control_penalty_horizon"
    raw.mkdir(parents=True)
    pd.DataFrame(
        {
            "experiment": ["tradeoff_control_penalty_0", "tradeoff_control_penalty_0p05"],
            "seed": [0, 0],
            "final_closed_test_loss": [1.0, 1.0],
            "final_open_test_loss": [1.2, 1.2],
            "trajectory_gain_distance": [0.0, 0.0],
            "claim_C4_tradeoff_quantified": [True, False],
        }
    ).to_csv(raw / "tradeoff_summary.csv", index=False)
    paths = make_claim_tables(tmp_path / "raw", tmp_path / "processed")
    claim_df = pd.read_csv(paths["claim_csv"])
    a1 = claim_df.loc[claim_df["claim"] == "A1"].iloc[0]
    assert a1["n"] == 2
    assert a1["support_fraction"] == 0.5
