"""Deterministic Task 1 post-processing learned only from training labels.

Teacher rubrics are strongly templated (fully correct 1,1,4,2,1,1; compiles
with a logic bug 1,1,2,1,1,1; does not compile 0,0,0,0,0,1 on single-problem
exams), while models emit vectors the teacher never uses, such as 1,1,0,0,0,0.
``RubricSnapper`` moves a prediction to the nearest training template of the
same exam type that keeps the predicted ``compilable``. ``is_empty_submission``
flags code with no statement at all. Both are hypotheses: accept them only if
they improve leave-one-out metrics on the training samples.
"""

import re
from collections import Counter

from llm_grading.data.schema import RUBRIC_RANGES, validate_rubric

FIELDS = tuple(RUBRIC_RANGES)
COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)
BOILERPLATE = re.compile(r"^\s*#.*$|^\s*using\s+namespace\s+\w+\s*;|\breturn\s+0\s*;", re.M)


def is_empty_submission(code):
    """True when neither the code nor its comments contain a single statement.

    Comments count on purpose: a fully commented-out solution was graded 10/10
    in the training data (EX02-S214), so it must not be zeroed by this rule.
    """
    return ";" not in BOILERPLATE.sub("", code or "")


class RubricSnapper:
    def __init__(self, train_samples, min_count=1, group_by="problem_type"):
        if group_by not in {"problem_type", None}:
            raise ValueError("group_by must be problem_type or None")
        self.min_count, self.group_by = min_count, group_by
        self.counts, self.members = {}, {}
        for sample in train_samples:
            template = tuple(validate_rubric(sample["rubric"])[f] for f in FIELDS)
            group = self.group(sample)
            self.counts.setdefault(group, Counter())[template] += 1
            self.members.setdefault((group, template), set()).add(sample["sample_id"])

    def group(self, sample):
        return sample.get(self.group_by) if self.group_by else "all"

    def templates(self, group, exclude_id=None):
        """Template counts; exclude_id removes the query's own label (LOO)."""
        result = {}
        for template, count in self.counts.get(group, {}).items():
            count -= exclude_id in self.members[(group, template)]
            if count >= self.min_count:
                result[template] = count
        return result

    def snap(self, rubric, sample, exclude_id=None):
        """Return (six-field rubric, chosen template or None if unchanged)."""
        predicted = validate_rubric(rubric)
        vector = tuple(predicted[f] for f in FIELDS)
        candidates = {
            t: n
            for t, n in self.templates(self.group(sample), exclude_id).items()
            if t[0] == vector[0]
        }
        if not candidates:
            return {f: predicted[f] for f in FIELDS}, None
        best = min(
            candidates,
            key=lambda t: (
                sum(abs(a - b) for a, b in zip(t, vector)),
                abs(sum(t) - sum(vector)),
                -candidates[t],
                t,
            ),
        )
        return dict(zip(FIELDS, best)), best
