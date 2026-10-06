"""Save resume identity and adapter metadata alongside Trainer checkpoints."""

from pathlib import Path


def make_callbacks(directory, metadata, contract, tokenizer):
    import math

    from transformers import TrainerCallback

    from llm_grading.runtime import write_json

    class ResearchCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "eval_loss"):
                if key in (logs or {}) and not math.isfinite(logs[key]):
                    raise RuntimeError(
                        f"Non-finite {key}; stop before wasting GPU hours"
                    )

        def on_save(self, args, state, control, **kwargs):
            path = Path(args.output_dir) / f"checkpoint-{state.global_step}"
            write_json(path / "adapter_metadata.json", metadata)
            write_json(path / "training_contract.json", contract)
            tokenizer.save_pretrained(path)

    return [ResearchCallback()]
