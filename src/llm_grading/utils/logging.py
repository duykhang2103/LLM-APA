"""Structured run logging for experiments and hardware metadata.

Minimum useful log fields are experiment ID, task, model/revision, seed,
device/GPU, data and split versions, config path, start/end time, and metrics.
"""


def configure_logging(experiment_id: str) -> None:
    """Configure one run's file/console logging context."""
    # TODO: Record task, model revision, seed, device, data version, and metrics.
    raise NotImplementedError("Logging is not implemented in the scaffold.")
