"""LoRA/QLoRA adapter configuration boundary."""

from typing import Any, Mapping


def build_lora_config(config: Mapping[str, Any]) -> object:
    """Describe the future adapter configuration entry point."""
    # TODO: Choose target modules, rank, alpha, dropout, and quantization settings.
    raise NotImplementedError("LoRA configuration is not implemented in the scaffold.")
