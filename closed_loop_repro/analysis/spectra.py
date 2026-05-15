from __future__ import annotations

import numpy as np
import torch


def spectral_radius(matrix: np.ndarray) -> float:
    if matrix.size == 0 or np.any(~np.isfinite(matrix)):
        return float("nan")
    return float(np.max(np.abs(np.linalg.eigvals(matrix))))


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
