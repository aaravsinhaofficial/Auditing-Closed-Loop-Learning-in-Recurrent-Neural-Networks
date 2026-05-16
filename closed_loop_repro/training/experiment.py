from __future__ import annotations

import json
import time
from copy import deepcopy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch import nn

from closed_loop_repro.analysis.gains import effective_gain, gain_distance
from closed_loop_repro.analysis.spectra import (
    coupled_matrix,
    coupled_spectral_summary,
    eigvals,
    rnn_matrix,
    rnn_spectral_summary,
)
from closed_loop_repro.analysis.spectral_stages import detect_spectral_stages
from closed_loop_repro.analysis.stages import detect_stages
from closed_loop_repro.config import save_config
from closed_loop_repro.io import ensure_dir, write_json
from closed_loop_repro.models import make_controller
from closed_loop_repro.progress import Heartbeat, log
from closed_loop_repro.random import clone_state_dict, seed_all
from closed_loop_repro.tasks import make_task


def run_pair_experiment(config: dict[str, Any], output_dir: str | Path | None = None) -> dict[str, Any]:
    start = time.time()
    seed = int(config.get("seed", 0))
    rng = seed_all(seed)
    requested_device = str(config.get("device", "cuda"))
    device = _resolve_device(requested_device)
    task = make_task(config.get("task", {"name": "double_integrator"}))
    train_cfg = config.get("training", {})
    progress_interval = float(config.get("progress_interval_seconds", train_cfg.get("progress_interval_seconds", 30.0)))
    model_cfg = config.get("model", {"name": "tanh_rnn"})
    hidden_size = int(model_cfg.get("hidden_size", train_cfg.get("hidden_size", 100)))
    model_kwargs = {k: v for k, v in model_cfg.items() if k != "hidden_size"}
    log(
        "run_pair start "
        f"experiment={config.get('experiment_name', 'unnamed')} seed={seed} "
        f"task={config.get('task', {}).get('name', 'double_integrator')} "
        f"model={model_cfg.get('name', 'tanh_rnn')} requested_device={requested_device} "
        f"resolved_device={device} epochs={train_cfg.get('epochs', 1000)} steps={train_cfg.get('steps', 50)}"
    )

    base_controller = make_controller(model_kwargs, task.obs_dim, hidden_size, task.action_dim).to(device)
    initial_state = clone_state_dict(base_controller)
    closed = make_controller(model_kwargs, task.obs_dim, hidden_size, task.action_dim).to(device)
    open_student = make_controller(model_kwargs, task.obs_dim, hidden_size, task.action_dim).to(device)
    closed.load_state_dict(initial_state)
    open_student.load_state_dict(initial_state)

    run_label = f"{config.get('experiment_name', 'unnamed')} seed={seed}"
    closed_history = _train_closed_loop(closed, task, train_cfg, rng, device, f"{run_label} closed-loop", progress_interval)
    teacher = deepcopy(closed).to(device)
    teacher.eval()
    open_history = _train_open_loop(open_student, teacher, task, train_cfg, rng, device, f"{run_label} open-loop", progress_interval)

    records = _merge_histories(closed_history, open_history)
    stages = detect_stages(
        np.asarray([r["closed_test_loss"] for r in records]),
        np.asarray([r["closed_coupled_radius"] for r in records]),
        min_plateau=int(config.get("analysis", {}).get("min_plateau", 8)),
    )
    for idx, label in enumerate(stages.labels):
        if idx < len(records):
            records[idx]["closed_stage"] = int(label)

    metrics = _compute_metrics(records, stages, seed, config)
    metrics["requested_device"] = requested_device
    metrics["resolved_device"] = str(device)
    metrics["runtime_seconds"] = time.time() - start

    result_dir = _result_dir(config, output_dir)
    ensure_dir(result_dir)
    pd.DataFrame(records).to_csv(result_dir / "timeseries.csv", index=False)
    write_json(metrics, result_dir / "metrics.json")
    save_config(config, result_dir / "config.yaml")
    log(f"run_pair done experiment={config.get('experiment_name', 'unnamed')} seed={seed} result_dir={result_dir}")
    return {"result_dir": str(result_dir), "metrics": metrics, "records": records}


def _train_closed_loop(
    controller: nn.Module,
    task,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
    progress_label: str,
    progress_interval: float,
) -> list[dict[str, float]]:
    optimizer = _optimizer(controller, cfg)
    epochs = int(cfg.get("epochs", 1000))
    history = []
    heartbeat = Heartbeat(progress_label, epochs, progress_interval)
    for epoch in range(epochs):
        loss = _closed_loop_loss(controller, task, cfg, rng, device)
        if epoch > 0:
            optimizer.zero_grad()
            loss.backward()
            _clip(controller, cfg)
            optimizer.step()
        record = _snapshot(
            epoch,
            "closed",
            float(loss.detach().cpu()),
            controller,
            task,
            cfg,
            device,
            _should_eval_extra_horizons(epoch, epochs, cfg),
        )
        history.append(record)
        heartbeat.maybe(
            epoch + 1,
            f"train_loss={record['closed_train_loss']:.6g} test_loss={record['closed_test_loss']:.6g} "
            f"coupled_radius={record['closed_coupled_radius']:.6g}",
        )
    heartbeat.done()
    return history


def _train_open_loop(
    student: nn.Module,
    teacher: nn.Module,
    task,
    cfg: dict[str, Any],
    rng: np.random.Generator,
    device: torch.device,
    progress_label: str,
    progress_interval: float,
) -> list[dict[str, float]]:
    optimizer = _optimizer(student, cfg)
    epochs = int(cfg.get("epochs", 1000))
    history = []
    heartbeat = Heartbeat(progress_label, epochs, progress_interval)
    for epoch in range(epochs):
        loss = _teacher_forcing_loss(student, teacher, task, cfg, rng, device)
        if epoch > 0:
            optimizer.zero_grad()
            loss.backward()
            _clip(student, cfg)
            optimizer.step()
        record = _snapshot(
            epoch,
            "open",
            float(loss.detach().cpu()),
            student,
            task,
            cfg,
            device,
            _should_eval_extra_horizons(epoch, epochs, cfg),
        )
        history.append(record)
        heartbeat.maybe(
            epoch + 1,
            f"train_loss={record['open_train_loss']:.6g} test_loss={record['open_test_loss']:.6g} "
            f"coupled_radius={record['open_coupled_radius']:.6g}",
        )
    heartbeat.done()
    return history


def _closed_loop_loss(controller: nn.Module, task, cfg: dict[str, Any], rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    batch_size = int(cfg.get("batch_size", 64))
    steps = int(cfg.get("steps", 50))
    control_penalty = float(cfg.get("control_penalty", 0.005))
    state = task.reset(batch_size, rng, device)
    hidden = controller.initial_hidden(batch_size, device)
    total = torch.zeros((), dtype=torch.float32, device=device)
    for step in range(steps):
        obs = task.observe(state, step)
        if step == 0 and bool(cfg.get("zero_initial_action", False)):
            action = torch.zeros((batch_size, task.action_dim), dtype=state.dtype, device=device)
        else:
            action, hidden = controller(obs, hidden)
        if cfg.get("loss_timing", "after_step") == "before_step":
            total = total + _state_loss(task, state, step, cfg) + control_penalty * _control_loss(action, cfg)
            state = task.step(state, action, step)
        else:
            state = task.step(state, action, step)
            total = total + _state_loss(task, state, step, cfg) + control_penalty * _control_loss(action, cfg)
    if bool(cfg.get("normalize_loss_by_steps", True)):
        return total / steps
    return total


def _teacher_forcing_loss(student: nn.Module, teacher: nn.Module, task, cfg: dict[str, Any], rng: np.random.Generator, device: torch.device) -> torch.Tensor:
    batch_size = int(cfg.get("batch_size", 64))
    steps = int(cfg.get("steps", 50))
    state = task.reset(batch_size, rng, device)
    h_student = student.initial_hidden(batch_size, device)
    h_teacher = teacher.initial_hidden(batch_size, device)
    total = torch.zeros((), dtype=torch.float32, device=device)
    if cfg.get("open_loop_input", "teacher_rollout") == "white_noise":
        noise = torch.randn((steps, batch_size, task.obs_dim), dtype=torch.float32, device=device)
        for step in range(steps):
            obs = noise[step]
            with torch.no_grad():
                teacher_action, h_teacher = teacher(obs, h_teacher)
            student_action, h_student = student(obs, h_student)
            total = total + torch.mean((student_action - teacher_action) ** 2)
        if bool(cfg.get("normalize_loss_by_steps", True)):
            return total / steps
        return total
    for step in range(steps):
        obs = task.observe(state, step)
        with torch.no_grad():
            teacher_action, h_teacher = teacher(obs, h_teacher)
        student_action, h_student = student(obs, h_student)
        total = total + torch.mean((student_action - teacher_action) ** 2)
        with torch.no_grad():
            state = task.step(state, teacher_action, step)
    if bool(cfg.get("normalize_loss_by_steps", True)):
        return total / steps
    return total


@torch.no_grad()
def _evaluate(controller: nn.Module, task, cfg: dict[str, Any], device: torch.device, steps_override: int | None = None) -> dict[str, Any]:
    eval_seed = int(cfg.get("eval_seed", 12345))
    rng = np.random.default_rng(eval_seed)
    batch_size = int(cfg.get("eval_batch_size", min(32, int(cfg.get("batch_size", 64)))))
    steps = int(steps_override if steps_override is not None else cfg.get("steps", 50))
    state = task.reset(batch_size, rng, device)
    hidden = controller.initial_hidden(batch_size, device)
    states, controls, losses = [], [], []
    for step in range(steps):
        obs = task.observe(state, step)
        if step == 0 and bool(cfg.get("eval_zero_initial_action", cfg.get("zero_initial_action", False))):
            action = torch.zeros((batch_size, task.action_dim), dtype=state.dtype, device=device)
        else:
            action, hidden = controller(obs, hidden)
        states.append(state.detach().cpu().numpy())
        controls.append(action.detach().cpu().numpy())
        if cfg.get("eval_loss_timing", cfg.get("loss_timing", "after_step")) == "before_step":
            losses.append(float(_state_loss(task, state, step, cfg).detach().cpu()))
            state = task.step(state, action, step)
        else:
            state = task.step(state, action, step)
            losses.append(float(_state_loss(task, state, step, cfg).detach().cpu()))
    states_np = np.stack(states)
    controls_np = np.stack(controls)
    gain = effective_gain(states_np[..., : min(states_np.shape[-1], max(1, controls_np.shape[-1] + 1))], controls_np)
    return {
        "test_loss": float(np.mean(losses)),
        "peak_loss": float(np.max(losses)),
        "gain": gain,
    }


def _snapshot(
    epoch: int,
    mode: str,
    train_loss: float,
    controller: nn.Module,
    task,
    cfg: dict[str, Any],
    device: torch.device,
    eval_extra_horizons: bool = True,
) -> dict[str, float]:
    eval_out = _evaluate(controller, task, cfg, device)
    gain = eval_out["gain"].reshape(-1)
    rnn_summary = rnn_spectral_summary(controller)
    coupled_summary = coupled_spectral_summary(task, controller)
    record = {
        "epoch": epoch,
        f"{mode}_train_loss": train_loss,
        f"{mode}_test_loss": eval_out["test_loss"],
        f"{mode}_peak_loss": eval_out["peak_loss"],
        f"{mode}_rnn_radius": rnn_summary["radius"],
        f"{mode}_coupled_radius": coupled_summary["radius"],
    }
    for key, value in rnn_summary.items():
        record[f"{mode}_rnn_{key}"] = float(value)
    for key, value in coupled_summary.items():
        record[f"{mode}_coupled_{key}"] = float(value)
    if bool(cfg.get("save_eigenvalues", False)):
        record[f"{mode}_rnn_eigvals"] = _eigvals_json(rnn_matrix(controller))
        record[f"{mode}_coupled_eigvals"] = _eigvals_json(coupled_matrix(task, controller))
    for idx, value in enumerate(gain[: min(8, len(gain))]):
        record[f"{mode}_gain_{idx}"] = float(value)
    if eval_extra_horizons:
        default_steps = int(cfg.get("steps", 50))
        for horizon in _evaluation_horizons(cfg):
            if horizon == default_steps:
                horizon_out = eval_out
            else:
                horizon_out = _evaluate(controller, task, cfg, device, steps_override=horizon)
            record[f"{mode}_test_loss_T{horizon}"] = horizon_out["test_loss"]
            record[f"{mode}_peak_loss_T{horizon}"] = horizon_out["peak_loss"]
    return record


def _evaluation_horizons(cfg: dict[str, Any]) -> list[int]:
    horizons = cfg.get("evaluation_horizons", [])
    if horizons is None:
        return []
    return sorted({int(horizon) for horizon in horizons if int(horizon) > 0})


def _should_eval_extra_horizons(epoch: int, epochs: int, cfg: dict[str, Any]) -> bool:
    if not _evaluation_horizons(cfg):
        return False
    interval = int(cfg.get("horizon_eval_interval", 1))
    return epoch == epochs - 1 or interval <= 1 or epoch % interval == 0


def _merge_histories(closed: list[dict[str, float]], open_loop: list[dict[str, float]]) -> list[dict[str, float]]:
    out = []
    for c, o in zip(closed, open_loop):
        merged = {"epoch": c["epoch"]}
        merged.update(c)
        merged.update({k: v for k, v in o.items() if k != "epoch"})
        out.append(merged)
    return out


def _compute_metrics(records: list[dict[str, float]], stages, seed: int, config: dict[str, Any]) -> dict[str, Any]:
    spectral_stages = detect_spectral_stages(pd.DataFrame(records))
    closed_test = np.asarray([r["closed_test_loss"] for r in records], dtype=float)
    open_test = np.asarray([r["open_test_loss"] for r in records], dtype=float)
    closed_radius = np.asarray([r["closed_coupled_radius"] for r in records], dtype=float)
    closed_gain = np.asarray([[r.get(f"closed_gain_{i}", np.nan) for i in range(8)] for r in records], dtype=float)
    open_gain = np.asarray([[r.get(f"open_gain_{i}", np.nan) for i in range(8)] for r in records], dtype=float)
    final_gain_distance = gain_distance(np.nan_to_num(closed_gain[-1]), np.nan_to_num(open_gain[-1]))
    initial_open = float(open_test[0]) if np.isfinite(open_test[0]) else 1e-12
    final_closed = float(closed_test[-1])
    final_open = float(open_test[-1])
    finite_final_losses = bool(np.isfinite(final_closed) and np.isfinite(final_open))
    loss_gap = final_open - final_closed
    relative_loss_gap = loss_gap / max(abs(final_closed), 1e-12) if finite_final_losses else float("nan")
    c1_loss_divergence = bool(finite_final_losses and relative_loss_gap > 0.05)
    c1_gain_divergence = bool(np.isfinite(final_gain_distance) and final_gain_distance > 0.05)
    open_spike = bool(np.any(np.isfinite(open_test)) and np.nanmax(open_test) > 2.0 * max(initial_open, final_open, 1e-12))
    open_post_initial_peak = bool(
        np.any(np.isfinite(open_test))
        and int(np.nanargmax(open_test)) > 0
        and np.nanmax(open_test) > 1.5 * max(initial_open, 1e-12)
    )
    stability_crossing = stages.stability_crossing
    plateau_exit = stages.plateau_end
    timing_gap = None if stability_crossing is None else int(plateau_exit - stability_crossing)
    recovered = bool(finite_final_losses and np.isfinite(closed_test[0]) and final_closed < closed_test[0])
    metrics = {
        "experiment": config.get("experiment_name", "unnamed"),
        "seed": seed,
        "task": config.get("task", {}).get("name", "double_integrator"),
        "model": config.get("model", {}).get("name", "tanh_rnn"),
        "epochs": len(records),
        "protocol_target": config.get("protocol", {}).get("target", "unspecified"),
        "initial_state_range": [
            config.get("task", {}).get("init_low", "task_default"),
            config.get("task", {}).get("init_high", "task_default"),
        ],
        "loss_timing": config.get("training", {}).get("loss_timing", "after_step"),
        "normalize_loss_by_steps": bool(config.get("training", {}).get("normalize_loss_by_steps", True)),
        "open_loop_input": config.get("training", {}).get("open_loop_input", "teacher_rollout"),
        "control_penalty": float(config.get("training", {}).get("control_penalty", 0.0)),
        "final_closed_test_loss": final_closed,
        "final_open_test_loss": final_open,
        "peak_open_test_loss": _safe_nanmax(open_test),
        "peak_closed_test_loss": _safe_nanmax(closed_test),
        "deployed_loss_gap": float(loss_gap),
        "deployed_loss_gap_relative_to_closed": float(relative_loss_gap),
        "trajectory_gain_distance": final_gain_distance,
        "open_loop_test_loss_spike": open_spike,
        "open_loop_post_initial_peak": open_post_initial_peak,
        "open_loop_peak_epoch": int(np.nanargmax(open_test)) if np.any(np.isfinite(open_test)) else None,
        "open_loop_peak_ratio_to_initial": float(_safe_nanmax(open_test) / max(initial_open, 1e-12)),
        "open_loop_peak_ratio_to_final": float(_safe_nanmax(open_test) / max(final_open, 1e-12)) if np.isfinite(final_open) else float("nan"),
        "closed_loop_plateau": bool(stages.plateau_detected),
        "plateau_length": stages.as_dict()["plateau_length"],
        "plateau_exit_detected": bool(stages.plateau_exit_detected),
        "plateau_exit_reason": stages.plateau_exit_reason,
        "stage1_end": stages.stage1_end,
        "plateau_end": stages.plateau_end,
        "stability_crossing": stability_crossing,
        "stability_to_plateau_gap": timing_gap,
        "final_closed_coupled_radius": float(closed_radius[-1]),
        "closed_recovered": recovered,
        "finite_final_losses": finite_final_losses,
        "claim_C1_loss_divergence": c1_loss_divergence,
        "claim_C1_gain_divergence": c1_gain_divergence,
        "claim_C1_divergence": bool(c1_loss_divergence or c1_gain_divergence),
        "claim_C2_stages": bool(finite_final_losses and stages.plateau_detected),
        "claim_C2_three_stage": bool(finite_final_losses and stages.plateau_detected and stages.plateau_exit_detected),
        "claim_C2_spectral_three_stage": bool(finite_final_losses and spectral_stages.three_stage_supported),
        "claim_C3_stability_transition": bool(np.isfinite(closed_radius[-1]) and stability_crossing is not None),
    }
    metrics.update(spectral_stages.as_dict())
    return metrics


def _state_loss(task, state: torch.Tensor, step: int, cfg: dict[str, Any]) -> torch.Tensor:
    if cfg.get("state_loss_reduction", "sum_state_mean_batch") in {"mean", "mean_elements"}:
        return torch.mean(state**2)
    return task.state_cost(state, step).mean()


def _control_loss(action: torch.Tensor, cfg: dict[str, Any]) -> torch.Tensor:
    if cfg.get("control_loss_reduction", "mean") == "sum":
        return torch.sum(action**2)
    return torch.mean(action**2)


def _safe_nanmax(values: np.ndarray) -> float:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.max(finite))


def _eigvals_json(matrix: np.ndarray | None) -> str:
    if matrix is None:
        return "[]"
    values = eigvals(matrix)
    if values.size == 0:
        return "[]"
    values = values[np.argsort(-np.abs(values))]
    return json.dumps([[float(value.real), float(value.imag)] for value in values])


def _optimizer(controller: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    params = [p for p in controller.parameters() if p.requires_grad]
    lr = float(cfg.get("learning_rate", 1e-2))
    if cfg.get("optimizer", "SGD").lower() == "adam":
        return torch.optim.Adam(params, lr=lr)
    return torch.optim.SGD(params, lr=lr)


def _resolve_device(requested: str) -> torch.device:
    if requested.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(requested)


def _clip(controller: nn.Module, cfg: dict[str, Any]) -> None:
    max_norm = cfg.get("gradient_clip", 1.0)
    if max_norm is not None and max_norm is not False:
        torch.nn.utils.clip_grad_norm_(controller.parameters(), max_norm=float(max_norm))


def _result_dir(config: dict[str, Any], output_dir: str | Path | None) -> Path:
    root = Path(output_dir or config.get("output_dir", "results/raw"))
    experiment = config.get("experiment_name", "run")
    seed = int(config.get("seed", 0))
    return root / experiment / f"seed_{seed:04d}"
