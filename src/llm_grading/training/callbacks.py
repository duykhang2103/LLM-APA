"""Training callbacks for evaluation hooks and experiment metadata.

Callbacks should connect a checkpoint back to experiment ID, Git commit,
seed, model revision, and config.
"""


def build_callbacks() -> list[object]:
    """Return the shared training callback collection."""
    # TODO: Add metric logging, checkpoint metadata, and reproducibility records.
    raise NotImplementedError("Training callbacks are not implemented in the scaffold.")
