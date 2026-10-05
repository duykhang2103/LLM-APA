"""Task 2 multi-label metrics: macro-F1, micro-F1, and per-label scores.

Always report support and the per-label table. Macro-F1 can hide a broken
label if the team does not inspect the individual rows, especially when a
label is rare or nearly absent.
"""

from typing import Any, Iterable, Mapping

from llm_grading.data.taxonomy import ERROR_LABELS


def _labels(record: Mapping[str, Any], key: str = "output") -> set[str]:
    value: Any = record.get(key, record) if key == "output" else record.get(key, [])
    if isinstance(value, Mapping):
        value = value.get("error_labels", [])
    if not isinstance(value, list) or any(not isinstance(x, str) for x in value):
        raise ValueError("Task 2 error_labels must be a list")
    labels = set(value)
    unknown = labels.difference(ERROR_LABELS)
    if unknown:
        raise ValueError(f"Unknown Task 2 labels: {sorted(unknown)}")
    return labels


def evaluate_task2(predictions: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return macro/micro and per-label metrics from aligned records."""
    prediction_list = list(predictions)
    reference_list = list(references)
    if len(prediction_list) != len(reference_list):
        raise ValueError("Task 2 predictions and references must have the same length")
    predicted = [_labels(item) for item in prediction_list]
    expected = [_labels(item, "error_labels") for item in reference_list]
    per_label: dict[str, dict[str, float | int]] = {}
    true_positive = false_positive = false_negative = 0
    f1_values = []
    for label in ERROR_LABELS:
        tp = sum(label in p and label in r for p, r in zip(predicted, expected))
        fp = sum(label in p and label not in r for p, r in zip(predicted, expected))
        fn = sum(label not in p and label in r for p, r in zip(predicted, expected))
        support = sum(label in r for r in expected)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": support}
        f1_values.append(f1)
        true_positive += tp
        false_positive += fp
        false_negative += fn
    micro_precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
    micro_recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if micro_precision + micro_recall else 0.0
    return {"count": len(predicted), "macro_f1": sum(f1_values) / len(f1_values), "micro_f1": micro_f1, "per_label": per_label}
