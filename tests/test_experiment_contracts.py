"""Lightweight data/prompt/training tests; no model downloads."""

import json
import tempfile
import unittest
from pathlib import Path

from llm_grading.data.loader import load_samples
from llm_grading.data.split import check_split_overlap, code_hash, create_splits
from llm_grading.prompting.formatter import parse_response, render_task_prompt
from llm_grading.training.dataset import build_training_dataset

RUBRIC = dict(
    compilable=1, io_format=1, logic=4, edge_case=2, complexity=1, code_quality=1
)


def sample(i="s1", code="int main() {}"):
    return dict(
        sample_id=i,
        problem_id="EX01",
        problem_type="multi_problem",
        problem_statement="problem",
        code=code,
        rubric=RUBRIC,
        error_labels=[],
        feedback="SECRET TEACHER TARGET",
        feedback_level=2,
    )


class Contracts(unittest.TestCase):
    def test_no_feedback_or_rubric_in_inputs(self):
        for task in ["task1", "task2", "task3"]:
            prompt = render_task_prompt(sample(), task, {})
            self.assertNotIn("SECRET TEACHER TARGET", prompt)
            self.assertIn("problem", prompt)
        records = build_training_dataset([sample()], "task1", {})
        self.assertNotIn("total", json.loads(records[0]["target"]))
        self.assertNotIn("SECRET", records[0]["prompt"])
        self.assertEqual(
            build_training_dataset([sample()], "task3", {})[0]["target"],
            "SECRET TEACHER TARGET",
        )

    def test_strict_json(self):
        self.assertEqual(
            parse_response(json.dumps(RUBRIC), "task1", sample())["total"], 10
        )
        for text in [
            json.dumps({**RUBRIC, "total": 10}),
            json.dumps({**RUBRIC, "logic": True}),
            "text " + json.dumps(RUBRIC),
        ]:
            with self.assertRaises(ValueError):
                parse_response(text, "task1", sample())
        with self.assertRaises(ValueError):
            parse_response('{"error_labels":["unknown"]}', "task2", sample())
        self.assertEqual(
            parse_response('{"error_labels":[]}', "task2", sample()),
            {"error_labels": []},
        )

    def test_grouped_split(self):
        rows = [
            sample(str(i), f"int f{i // 2}() {{ return {i // 2}; }}") for i in range(12)
        ]
        rows[1]["code"] = rows[0]["code"].replace(" ", "\r\n ")
        splits = create_splits(rows, 42, validation_fraction=0.3, test_fraction=0)
        self.assertEqual(
            splits, create_splits(rows, 42, validation_fraction=0.3, test_fraction=0)
        )
        positions = {sid: split for split, ids in splits.items() for sid in ids}
        self.assertEqual(positions["0"], positions["1"])
        self.assertEqual(set(positions), {str(i) for i in range(12)})
        self.assertEqual(code_hash("a  b\r\n c"), code_hash("a b c"))
        with self.assertRaises(ValueError):
            check_split_overlap([sample()], [sample("other")])

    def test_teacher_paths_and_targets_preserved(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d)
            (p / "s.cpp").write_text("int f() {}")
            (p / "exams.json").write_text(
                json.dumps(
                    {
                        "exams": [
                            {
                                "exam_id": "EX01",
                                "exam_type": "multi_problem",
                                "statement": "Q",
                            }
                        ]
                    }
                )
            )
            payload = {
                "task": "task1_grading",
                "samples": [
                    {
                        "sample_id": "s1",
                        "input": {
                            "exam_id": "EX01",
                            "exam_type": "multi_problem",
                            "code_file": "s.cpp",
                            "compile_log": "error",
                        },
                        "output": {"rubric": RUBRIC, "total_score": 10},
                    }
                ],
            }
            file = p / "task1_grading.json"
            file.write_text(json.dumps(payload))
            row = load_samples(p, task="task1")[0]
            self.assertEqual(row["compile_log"], "error")
            self.assertEqual(row["rubric"], RUBRIC)
            payload["samples"][0]["input"]["code_file"] = "../escape.cpp"
            file.write_text(json.dumps(payload))
            with self.assertRaises(ValueError):
                load_samples(p, task="task1")

    def test_training_rejects_missing_target(self):
        row = sample()
        row.pop("rubric")
        with self.assertRaises(ValueError):
            build_training_dataset([row], "task1", {})


if __name__ == "__main__":
    unittest.main()
