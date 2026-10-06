"""Prediction validation rejects malformed records without crashing."""

import unittest

from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.validation.predictions import validate_prediction_records


class PredictionSchemaTests(unittest.TestCase):
    def test_unknown_task_rejected_even_for_empty_records(self):
        self.assertTrue(validate_prediction_records([], "unknown"))

    def test_valid_empty_taxonomy(self):
        self.assertEqual(
            validate_prediction_records(
                [
                    {"sample_id": "S1", "output": {"error_labels": []}},
                ],
                "task2",
            ),
            [],
        )

    def test_invalid_taxonomy_values_report_errors(self):
        for labels in ([[]], [{}], [1], [True], [None], ["not official"]):
            with self.subTest(labels=labels):
                messages = validate_prediction_records(
                    [
                        {"sample_id": "S1", "output": {"error_labels": labels}},
                    ],
                    "task2",
                )
                self.assertTrue(messages)

    def test_duplicate_taxonomy_labels_rejected(self):
        self.assertTrue(
            validate_prediction_records(
                [
                    {
                        "sample_id": "S1",
                        "output": {"error_labels": [ERROR_LABELS[0]] * 2},
                    },
                ],
                "task2",
            )
        )

    def test_duplicate_sample_ids_rejected(self):
        row = {"sample_id": "S1", "output": {"error_labels": []}}
        self.assertTrue(validate_prediction_records([row, row], "task2"))

    def test_boolean_feedback_level_rejected(self):
        self.assertTrue(
            validate_prediction_records(
                [
                    {
                        "sample_id": "S1",
                        "output": {
                            "feedback": "Hãy kiểm tra trường hợp biên.",
                            "feedback_level": True,
                            "compliance": {"pass": True, "violations": []},
                        },
                    },
                ],
                "task3",
            )
        )

    def test_non_object_records_and_output_rejected(self):
        for payload in ({}, [None], [{"sample_id": "S1", "output": []}]):
            with self.subTest(payload=payload):
                self.assertTrue(validate_prediction_records(payload, "task2"))


if __name__ == "__main__":
    unittest.main()
