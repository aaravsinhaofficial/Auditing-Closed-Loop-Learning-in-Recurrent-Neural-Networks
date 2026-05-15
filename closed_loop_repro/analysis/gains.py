from __future__ import annotations

import numpy as np


def effective_gain(states: np.ndarray, controls: np.ndarray, ridge: float = 1e-8) -> np.ndarray:
    x = states.reshape(-1, states.shape[-1])
    u = controls.reshape(-1, controls.shape[-1])
    xtx = x.T @ x + ridge * np.eye(x.shape[1])
    return (np.linalg.solve(xtx, x.T @ u)).T


def gain_distance(gain_a: np.ndarray, gain_b: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(gain_a) - np.asarray(gain_b)))
