import numpy as np
import torch

from closed_loop_repro.models import make_controller
from closed_loop_repro.tasks import make_task


def test_double_integrator_rollout_shapes():
    task = make_task({"name": "double_integrator"})
    rng = np.random.default_rng(0)
    state = task.reset(5, rng, torch.device("cpu"))
    obs = task.observe(state, 0)
    action = torch.zeros(5, task.action_dim)
    next_state = task.step(state, action, 0)
    assert state.shape == (5, 2)
    assert obs.shape == (5, 1)
    assert next_state.shape == (5, 2)


def test_controller_output_shapes():
    task = make_task({"name": "double_integrator"})
    model = make_controller({"name": "tanh_rnn"}, task.obs_dim, 8, task.action_dim)
    hidden = model.initial_hidden(4, torch.device("cpu"))
    action, hidden = model(torch.zeros(4, task.obs_dim), hidden)
    assert action.shape == (4, task.action_dim)
    assert hidden.shape == (4, 8)


def test_ring_partial_observation_shape_and_moving_target():
    task = make_task({"name": "ring", "observe_velocity": False, "moving_target": True})
    rng = np.random.default_rng(0)
    state = task.reset(3, rng, torch.device("cpu"))
    obs0 = task.observe(state, 0)
    obs1 = task.observe(state, 10)
    assert task.obs_dim == 2
    assert obs0.shape == (3, 2)
    assert not torch.allclose(obs0, obs1)


def test_task_factory_filters_inherited_kwargs():
    tracking = make_task({"name": "tracking", "dt": 0.1, "init_low": -2.0, "init_high": 2.0})
    ring = make_task({"name": "ring", "dt": 0.1, "init_low": -2.0, "init_high": 2.0})
    assert tracking.state_dim == 4
    assert ring.state_dim == 2
