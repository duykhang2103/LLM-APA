"""Future CLI for task-specific evaluation on a selected split.

Target command:
`python scripts/evaluate.py --config configs/task1/qwen_lora.yaml --split val`
"""


def main() -> None:
    # TODO: Load predictions/references, call the selected evaluator, and register metrics.
    raise NotImplementedError("Evaluation is not implemented in the scaffold.")


if __name__ == "__main__":
    main()
