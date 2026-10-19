"""Training-only task examples, self exclusion, duplicate audit."""

import json
import re
from difflib import SequenceMatcher

from llm_grading.data.quality import select_training_samples
from llm_grading.data.split import code_hash, normalized_code
from llm_grading.prompting.formatter import BUILDERS
from llm_grading.retrieval.index import build_index
from llm_grading.training.dataset import build_training_dataset

COMMENT = re.compile(r"//[^\n]*|/\*.*?\*/", re.S)


def is_near_duplicate(a, b, cutoff):
    """Same result as SequenceMatcher.ratio() >= cutoff, without its quadratic
    cost when the cheap upper bounds already fall below the cutoff."""
    matcher = SequenceMatcher(None, a, b, autojunk=False)
    return (
        matcher.real_quick_ratio() >= cutoff
        and matcher.quick_ratio() >= cutoff
        and matcher.ratio() >= cutoff
    )


def retrieval_text(sample, task, key="input_json"):
    # input_json embeds the whole whitelisted view; with a short encoder window
    # it mostly sees the code's shared header. "code" keys on the code alone.
    if key == "code":
        return normalized_code(COMMENT.sub(" ", sample.get("code", "")))
    view = BUILDERS[task](sample)
    return json.dumps(view, ensure_ascii=False, sort_keys=True)


class Retriever:
    def __init__(self, samples, task, config, encoder=None):
        samples, self.quality_audit = select_training_samples(samples, task, config)
        if not samples:
            raise ValueError("Empty training retrieval corpus")
        self.samples, self.task, self.settings = (
            samples,
            task,
            config.get("retrieval", {}),
        )
        if self.settings.get("k", 3) not in {1, 3, 5}:
            raise ValueError("retrieval.k must be 1, 3, or 5")
        self.key = self.settings.get("key", "input_json")
        if self.key not in {"input_json", "code"}:
            raise ValueError("retrieval.key must be input_json or code")
        records = build_training_dataset(samples, task, config, training=False)
        self.targets = [row["target"] for row in records]
        self.normalized = [normalized_code(s["code"]) for s in samples]
        self.hashes = [code_hash(s["code"]) for s in samples]
        self.index = build_index(
            [retrieval_text(s, task, self.key) for s in samples],
            self.settings,
            encoder,
        )

    def retrieve(self, query):
        scores = self.index.scores(retrieval_text(query, self.task, self.key))
        candidates = []
        audit = {
            "quality_exclusions": self.quality_audit["excluded"],
            "embedding": self.index.metadata,
            "self_excluded": [],
            "exact_matches": [],
            "near_matches": [],
            "retrieved": [],
        }
        restrict = self.settings.get("exclude_duplicates", True)
        cutoff = self.settings.get("near_duplicate_threshold")
        query_hash = code_hash(query["code"])
        query_code = normalized_code(query["code"])
        for i, sample in enumerate(self.samples):
            if sample["sample_id"] == query["sample_id"]:
                audit["self_excluded"].append(sample["sample_id"])
                continue
            if (
                self.task == "task3"
                and sample["feedback_level"] != query["feedback_level"]
            ):
                continue
            exact = self.hashes[i] == query_hash
            near = bool(
                cutoff and is_near_duplicate(self.normalized[i], query_code, cutoff)
            )
            if exact:
                audit["exact_matches"].append(sample["sample_id"])
            if near:
                audit["near_matches"].append(sample["sample_id"])
            if restrict and (exact or near):
                continue
            candidates.append((float(scores[i]), i))
        same_exam_first = self.settings.get("same_exam_first", False)
        candidates.sort(
            key=lambda pair: (
                same_exam_first
                and self.samples[pair[1]].get("problem_id") != query.get("problem_id"),
                -pair[0],
                self.samples[pair[1]]["sample_id"],
            )
        )
        examples = []
        for score, i in candidates[: self.settings.get("k", 3)]:
            sample = self.samples[i]
            target = (
                self.targets[i] if self.task == "task3" else json.loads(self.targets[i])
            )
            examples.append({"input": BUILDERS[self.task](sample), "target": target})
            audit["retrieved"].append(
                {
                    "sample_id": sample["sample_id"],
                    "score": score,
                    "same_exam": sample.get("problem_id") == query.get("problem_id"),
                }
            )
        audit["shortfall"] = self.settings.get("k", 3) - len(examples)
        return examples, audit
