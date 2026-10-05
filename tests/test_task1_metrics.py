import unittest

from test_experiment_contracts import RUBRIC

from llm_grading.data.schema import validate_rubric
from llm_grading.evaluation.task1 import evaluate_task1


class Task1Tests(unittest.TestCase):
    def test_total_and_range(self):
        self.assertEqual(validate_rubric(RUBRIC)["total"], 10)
        for value in [True, 1.5, 5]:
            with self.assertRaises(ValueError):
                validate_rubric({**RUBRIC, "logic": value})
        with self.assertRaises(ValueError):
            validate_rubric({**RUBRIC, "total": 9})

    def test_metrics(self):
        metrics = evaluate_task1([{"output": RUBRIC}], [{"rubric": RUBRIC}])
        self.assertEqual(metrics["mae_total"], 0)
        self.assertEqual(metrics["exact_match_logic"], 1)
