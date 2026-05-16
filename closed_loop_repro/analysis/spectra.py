from __future__ import annotations

import numpy as np
import torch


def eigvals(matrix: np.ndarray) -> np.ndarray:
    if matrix.size == 0 or np.any(~np.isfinite(matrix)):
        return np.asarray([], dtype=np.complex128)
    return np.linalg.eigvals(matrix)


def spectral_radius(matrix: np.ndarray) -> float:
    values = eigvals(matrix)
    if values.size == 0:
        return float("nan")
    return float(np.max(np.abs(values)))


def spectral_summary(matrix: np.ndarray | None, complex_tol: float = 1e-6) -> dict[str, float]:
    if matrix is None:
        return _empty_summary()
    values = eigvals(matrix)
    if values.size == 0:
        return _empty_summary()
    order = np.argsort(-np.abs(values))
    ordered = values[order]
    dominant = ordered[0]
    real_values = [value for value in ordered if abs(value.imag) <= complex_tol]
    third = real_values[0] if real_values else np.nan + 0j
    unstable_complex = [value for value in ordered if abs(value.imag) > complex_tol and abs(value) > 1.0]
    return {
        "radius": float(abs(dominant)),
        "dom_real": float(dominant.real),
        "dom_imag": float(dominant.imag),
        "dom_abs": float(abs(dominant)),
        "dom_is_complex": float(abs(dominant.imag) > complex_tol),
        "unstable_complex_abs": float(abs(unstable_complex[0])) if unstable_complex else float("nan"),
        "has_unstable_complex": float(bool(unstable_complex)),
        "third_real": float(third.real) if np.isfinite(third.real) else float("nan"),
        "third_real_abs": float(abs(third)) if np.isfinite(third.real) else float("nan"),
    }


def _empty_summary() -> dict[str, float]:
    return {
        "radius": float("nan"),
        "dom_real": float("nan"),
        "dom_imag": float("nan"),
        "dom_abs": float("nan"),
        "dom_is_complex": float("nan"),
        "unstable_complex_abs": float("nan"),
        "has_unstable_complex": float("nan"),
        "third_real": float("nan"),
        "third_real_abs": float("nan"),
    }


def rnn_matrix(controller: torch.nn.Module) -> np.ndarray | None:
    if hasattr(controller, "recurrent_matrix"):
        return controller.recurrent_matrix.detach().cpu().numpy()
    if hasattr(controller, "Whh"):
        return controller.Whh.detach().cpu().numpy()
    return None


def rnn_spectral_radius(controller: torch.nn.Module) -> float:
    matrix = rnn_matrix(controller)
    if matrix is None:
        return float("nan")
    return spectral_radius(matrix)


def coupled_matrix(task, controller: torch.nn.Module) -> np.ndarray | None:
    matrices = task.numpy_matrices()
    if matrices is None or not hasattr(controller, "Wih") or not hasattr(controller, "Who"):
        return None
    A, B, C = matrices
    Wih = controller.Wih.detach().cpu().numpy()
    Who = controller.Who.detach().cpu().numpy()
    Wrec = rnn_matrix(controller)
    if Wrec is None:
        return None
    leak = float(getattr(controller, "leak", 1.0))
    hidden = Wrec.shape[0]
    p11 = A
    p12 = B @ Who
    p21 = leak * Wih @ C @ A
    p22 = (1.0 - leak) * np.eye(hidden) + leak * Wrec + leak * Wih @ C @ B @ Who
    top = np.concatenate([p11, p12], axis=1)
    bottom = np.concatenate([p21, p22], axis=1)
    return np.concatenate([top, bottom], axis=0)


def coupled_spectral_radius(task, controller: torch.nn.Module) -> float:
    matrix = coupled_matrix(task, controller)
    if matrix is None:
        return float("nan")
    return spectral_radius(matrix)


def coupled_spectral_summary(task, controller: torch.nn.Module) -> dict[str, float]:
    return spectral_summary(coupled_matrix(task, controller))


def rnn_spectral_summary(controller: torch.nn.Module) -> dict[str, float]:
    return spectral_summary(rnn_matrix(controller))
