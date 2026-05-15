from closed_loop_repro.tasks.angular_integration import RingIntegrationTask
from closed_loop_repro.tasks.double_integrator import DoubleIntegratorTask
from closed_loop_repro.tasks.tracking_task import TrackingTask


def make_task(config):
    name = config.get("name", "double_integrator")
    if name == "double_integrator":
        return DoubleIntegratorTask(**{k: v for k, v in config.items() if k != "name"})
    if name == "tracking":
        return TrackingTask(**{k: v for k, v in config.items() if k != "name"})
    if name in {"ring", "angular_integration", "path_integration"}:
        return RingIntegrationTask(**{k: v for k, v in config.items() if k != "name"})
    raise ValueError(f"Unknown task {name!r}")
