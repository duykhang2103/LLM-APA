"""Future CLI for shared Task 1/2/3 training.

Target command:
`python scripts/train.py --config configs/task1/qwen_lora.yaml`

Implementation checklist:
1. Load config, seed, normalized split, and task pipeline.
2. Build the task-specific training dataset.
3. Load the configured model and optional LoRA/QLoRA adapter.
4. Train/evaluate according to config.
5. Save checkpoint reference, metrics, hardware, and metadata under one ID.
"""


def main() -> None:
    # TODO: Resolve config, seed, model, dataset, trainer, checkpoint, and metadata.
    raise NotImplementedError("Training is not implemented in the scaffold.")


if __name__ == "__main__":
    main()
