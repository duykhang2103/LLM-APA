"""Future CLI for automated prediction generation.

Target command:
`python scripts/predict.py --config configs/task1/qwen_lora.yaml --split test --output outputs/task1_predictions.json`

Implementation checklist:
1. Load one frozen config and model/prompt revision.
2. Read only fields allowed for the selected task.
3. Run every sample automatically.
4. Post-process and validate every output before writing JSON.
5. Never edit individual predictions manually.
"""


def main() -> None:
    # TODO: Run the selected pipeline without human intervention per sample.
    raise NotImplementedError("Prediction is not implemented in the scaffold.")


if __name__ == "__main__":
    main()
