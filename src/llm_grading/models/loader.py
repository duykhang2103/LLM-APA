"""Load tokenizers, base models, adapters, revisions, and devices centrally.

Task files should never call a model library directly. The shared loader is
where the team records the model name, revision, tokenizer, dtype, device,
quantization, and optional adapter so prompting and fine-tuning experiments
remain comparable.

Example config fragment::

    model:
      name: <open-weight-model>
      revision: <revision-or-commit>
      dtype: bfloat16
      device: auto
      adapter_path: null
"""

from typing import Any, Mapping


def load_model(config: Mapping[str, Any]) -> object:
    """Return a model/tokenizer bundle selected by configuration.

    The eventual return object should make tokenizer and model access explicit
    instead of hiding model-specific setup in each task.
    """
    # TODO: Resolve model revision, dtype, device, quantization, and adapters.
    raise NotImplementedError("Model loading is not implemented in the scaffold.")
