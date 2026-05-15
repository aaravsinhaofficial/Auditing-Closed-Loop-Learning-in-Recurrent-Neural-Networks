import numpy as np

from closed_loop_repro.analysis.gains import effective_gain
from closed_loop_repro.analysis.spectra import spectral_radius
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
    assert result.stage1_end < result.plateau_end


def test_bootstrap_ci_smoke():
    mean, lo, hi = bootstrap_ci([1, 2, 3], n_boot=100, seed=0)
    assert lo <= mean <= hi
