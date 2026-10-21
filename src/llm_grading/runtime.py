"""Small runtime helpers shared by the five beginner-friendly CLIs."""

import json
from pathlib import Path
from typing import Any, Mapping

from .data.loader import _error_labels, load_samples
from .utils.paths import resolve_path


def load_samples_for_config(config: Mapping[str, Any], split: str = "test") -> list[dict[str, Any]]:
    """Load one task's configured split; val and validation are aliases."""
    data = config.get("data", {})
    split_key = "validation" if split == "val" else split
    source = data.get(f"{split_key}_path")
    if split_key == "validation":
        source = source or data.get("val_path")
    if not source and config.get("method") != "heuristic":
        raise ValueError(f"Set explicit data.{split_key}_path; learned methods never fall back to all input data")
    source = source or data.get("input_path")
    if not source:
        raise ValueError(f"No input path configured for split '{split}'")
    return load_samples(resolve_path(source), data.get("max_samples"), task=config.get("task"))


def attach_task3_error_labels(samples: list[dict[str, Any]], path: str | Path | None) -> list[dict[str, Any]]:
    """Fill Task 3 labels missing from the input with Task 2 predictions.

    The public test omits ``input.taxonomy_error`` while the challenge output
    still requires ``taxonomy_error``, so Task 3 reuses the Task 2 prediction.
    Accepts internal (``error_labels``) or challenge (``taxonomy_error``) files.
    """
    missing = [s for s in samples if s.get("error_labels_missing")]
    if not missing:
        return samples
    if not path:
        raise ValueError(
            f"{len(missing)} Task 3 samples lack input taxonomy_error; "
            "pass --task3-labels with a Task 2 prediction file"
        )
    predicted = {}
    for row in read_json(path):
        output = row.get("output", {}) if isinstance(row, dict) else {}
        predicted[row.get("sample_id")] = output.get("error_labels", output.get("taxonomy_error"))
    for sample in missing:
        sample_id = sample["sample_id"]
        if sample_id not in predicted:
            raise ValueError(f"Task 2 predictions in {path} have no labels for {sample_id}")
        labels = _error_labels(predicted[sample_id], sample_id)
        if len(labels) != len(set(labels)):
            raise ValueError(f"Duplicate Task 2 labels for {sample_id} in {path}")
        sample["error_labels"] = labels
        sample["error_labels_source"] = str(path)
        del sample["error_labels_missing"]
    return samples


def write_json(path: str | Path, payload: Any) -> Path:
    """Write UTF-8 JSON, creating only the requested parent directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(output_path.name + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temporary.replace(output_path)
    return output_path


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON with a clear path in errors from the CLIs."""
    input_path = Path(path)
    try:
        return json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"JSON file does not exist: {input_path}") from error


def attach_saved_evidence(
    samples: list[dict[str, Any]],
    compile_path: str | Path | None = None,
    tests_path: str | Path | None = None,
) -> list[dict[str, Any]]:
    """Fill compile_log/test_report from scripts/compile_evidence.py and
    scripts/run_tests.py outputs. Both are computed from code only."""
    from .data.evidence import attach_evidence
    from .data.testing import attach_test_reports

    if compile_path:
        attach_evidence(samples, read_json(compile_path)["evidence"])
    if tests_path:
        attach_test_reports(samples, read_json(tests_path)["reports"])
    return samples
