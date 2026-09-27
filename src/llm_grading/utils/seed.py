"""Central random-seed setup for reproducible experiments.

Call this once near the start of train/evaluate/predict. Record the same seed
in the run metadata and avoid silently overriding it in task modules.
"""


def set_seed(seed: int) -> None:
    """Seed every supported random source and deterministic backend setting."""
    # TODO: Seed Python, NumPy, PyTorch, and deterministic backend settings.
    raise NotImplementedError("Seed setup is not implemented in the scaffold.")
