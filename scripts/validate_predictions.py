"""Future CLI for validating leaderboard prediction JSON files.

Target command:
`python scripts/validate_predictions.py --task task1 --input outputs/task1_predictions.json`

Implementation checklist:
1. Load the selected task schema and canonical labels.
2. Check JSON shape and required ``sample_id`` values.
3. Check score ranges, totals, labels, empty-label behavior, and feedback fields.
4. Print actionable errors and return a non-zero status for invalid output.
"""


def main() -> None:
    # TODO: Validate sample IDs and task-specific output before submission.
    raise NotImplementedError("Prediction validation is not implemented in the scaffold.")


if __name__ == "__main__":
    main()
