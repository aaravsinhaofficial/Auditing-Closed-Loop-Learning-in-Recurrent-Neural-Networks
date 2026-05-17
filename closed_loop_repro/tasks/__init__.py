from dataclasses import fields

from closed_loop_repro.tasks.angular_integration import RingIntegrationTask
from closed_loop_repro.tasks.double_integrator import DoubleIntegratorTask
from closed_loop_repro.tasks.tracking_task import TrackingTask


def make_task(config):
    name = config.get("name", "double_integrator")
    if name == "double_integrator":
        return DoubleIntegratorTask(**_task_kwargs(config, DoubleIntegratorTask))
    if name == "tracking":
        return TrackingTask(**_task_kwargs(config, TrackingTask))
    if name in {"ring", "angular_integration", "path_integration"}:
        return RingIntegrationTask(**_task_kwargs(config, RingIntegrationTask))
    raise ValueError(f"Unknown task {name!r}")


def _task_kwargs(config, task_cls):
    valid = {field.name for field in fields(task_cls) if field.init}
    return {key: value for key, value in config.items() if key in valid}
