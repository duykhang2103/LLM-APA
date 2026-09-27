"""Future CLI for task-specific evaluation on a selected split.

Target command:
`python scripts/evaluate.py --config configs/task1/qwen_lora.yaml --split val`

Implementation checklist:
1. Load the exact config, model revision, prompt version, and split.
2. Generate or read predictions without using reference-only inputs.
3. Run the task evaluator and slice metrics by useful error groups.
4. Write metrics and notes to the experiment registry.
"""


def main() -> None:
    # TODO: Load predictions/references, call the selected evaluator, and register metrics.
    raise NotImplementedError("Evaluation is not implemented in the scaffold.")


if __name__ == "__main__":
    main()
