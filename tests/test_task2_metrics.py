"""Fixed-taxonomy multi-label metrics, including absent labels."""

import unittest

from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.evaluation.task2 import evaluate_task2

try:
    from sklearn.metrics import f1_score, precision_recall_fscore_support
except ImportError:
    f1_score = precision_recall_fscore_support = None


def records(label_sets, predictions=False):
    return [
        {"output": {"error_labels": labels}}
        if predictions
        else {"error_labels": labels}
        for labels in label_sets
    ]


class Task2MetricTests(unittest.TestCase):
    def test_absent_labels_contribute_zero_to_fixed_taxonomy_macro(self):
        label = ERROR_LABELS[0]
        result = evaluate_task2(records([[label]], True), records([[label]]))
        self.assertAlmostEqual(result["macro_f1"], 1 / len(ERROR_LABELS))
        self.assertEqual(result["micro_f1"], 1.0)
        self.assertEqual(
            result["per_label"]["Lỗi edge case"],
            {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0},
        )

    def test_all_empty_labels_have_zero_f1(self):
        result = evaluate_task2(records([[], []], True), records([[], []]))
        self.assertEqual(result["macro_f1"], 0.0)
        self.assertEqual(result["micro_f1"], 0.0)

    @unittest.skipIf(f1_score is None, "scikit-learn is not installed")
    def test_sklearn_parity_across_empty_rare_and_false_positive_labels(self):
        cases = [
            ([[], []], [[], []]),
            ([[ERROR_LABELS[0]]], [[ERROR_LABELS[0]]]),
            (
                [[ERROR_LABELS[0]], [], [ERROR_LABELS[6]], [ERROR_LABELS[9]]],
                [[ERROR_LABELS[0]], [ERROR_LABELS[9]], [], [ERROR_LABELS[9]]],
            ),
        ]
        for targets, predictions in cases:
            with self.subTest(targets=targets, predictions=predictions):
                y_true = [
                    [int(label in row) for label in ERROR_LABELS] for row in targets
                ]
                y_pred = [
                    [int(label in row) for label in ERROR_LABELS] for row in predictions
                ]
                indices = list(range(len(ERROR_LABELS)))
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_true,
                    y_pred,
                    labels=indices,
                    average=None,
                    zero_division=0,
                )
                result = evaluate_task2(records(predictions, True), records(targets))
                for index, label in enumerate(ERROR_LABELS):
                    row = result["per_label"][label]
                    self.assertAlmostEqual(row["precision"], precision[index])
                    self.assertAlmostEqual(row["recall"], recall[index])
                    self.assertAlmostEqual(row["f1"], f1[index])
                    self.assertEqual(row["support"], support[index])
                for average in ("macro", "micro"):
                    self.assertAlmostEqual(
                        result[f"{average}_f1"],
                        f1_score(
                            y_true,
                            y_pred,
                            labels=indices,
                            average=average,
                            zero_division=0,
                        ),
                    )

    def test_mismatched_record_counts_fail(self):
        with self.assertRaises(ValueError):
            evaluate_task2(records([[]], True), [])


if __name__ == "__main__":
    unittest.main()
