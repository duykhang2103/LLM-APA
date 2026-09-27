"""LoRA/QLoRA adapter configuration boundary.

Record rank, alpha, dropout, target modules, quantization, and trainable
parameter count. Begin with separate task adapters because they are easier for
new contributors to debug than an immediate multi-task adapter.
"""

from typing import Any, Mapping


def build_lora_config(config: Mapping[str, Any]) -> object:
    """Create the adapter settings used by the shared trainer."""
    # TODO: Choose target modules, rank, alpha, dropout, and quantization settings.
    raise NotImplementedError("LoRA configuration is not implemented in the scaffold.")
