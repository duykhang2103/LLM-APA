"""Task 1 rubric metrics: QWK, MAE, and component exact-match.

Evaluation order should be: validate components, derive predicted/reference
totals, compute QWK and MAE on totals, then compute exact match by component.
Report slices for single/multi-problem, score range, and problem type.
"""

from typing import Any, Iterable, Mapping

from tasks.task1_grading.postprocess import RUBRIC_RANGES, postprocess_rubric


def _output(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("output", record)
    if not isinstance(value, Mapping):
        raise ValueError("Task 1 record output must be an object")
    return value


def _reference(record: Mapping[str, Any]) -> Mapping[str, Any]:
    value = record.get("rubric", record.get("output", record))
    if not isinstance(value, Mapping):
        raise ValueError("Task 1 reference rubric must be an object")
    return value


def _quadratic_weighted_kappa(actual: list[int], predicted: list[int], maximum: int) -> float:
    if not actual:
        return 0.0
    categories = maximum + 1
    observed = [[0.0] * categories for _ in range(categories)]
    for reference, prediction in zip(actual, predicted):
        observed[reference][prediction] += 1
    total = float(len(actual))
    reference_counts = [sum(row) for row in observed]
    prediction_counts = [sum(observed[row][column] for row in range(categories)) for column in range(categories)]
    expected = [[reference_counts[i] * prediction_counts[j] / total for j in range(categories)] for i in range(categories)]
    denominator = float((maximum or 1) ** 2)
    observed_loss = sum(((i - j) ** 2 / denominator) * observed[i][j] for i in range(categories) for j in range(categories))
    expected_loss = sum(((i - j) ** 2 / denominator) * expected[i][j] for i in range(categories) for j in range(categories))
    if expected_loss == 0:
        return 1.0 if observed_loss == 0 else 0.0
    return 1.0 - observed_loss / expected_loss


def evaluate_task1(predictions: Iterable[Mapping[str, Any]], references: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Return Task 1 metrics from aligned prediction/reference records."""
    prediction_list = list(predictions)
    reference_list = list(references)
    if len(prediction_list) != len(reference_list):
        raise ValueError("Task 1 predictions and references must have the same length")
    predicted = [postprocess_rubric(_output(item)) for item in prediction_list]
    expected = [postprocess_rubric(_reference(item)) for item in reference_list]
    predicted_totals = [item["total"] for item in predicted]
    expected_totals = [item["total"] for item in expected]
    result: dict[str, Any] = {
        "count": len(predicted),
        "qwk_total": _quadratic_weighted_kappa(expected_totals, predicted_totals, 10),
        "mae_total": (sum(abs(a - b) for a, b in zip(expected_totals, predicted_totals)) / len(predicted)) if predicted else 0.0,
    }
    for name in RUBRIC_RANGES:
        result[f"mae_{name}"] = sum(abs(a[name] - b[name]) for a, b in zip(expected, predicted)) / len(predicted) if predicted else 0.0
        result[f"exact_match_{name}"] = sum(a[name] == b[name] for a, b in zip(expected, predicted)) / len(predicted) if predicted else 0.0
    return result
