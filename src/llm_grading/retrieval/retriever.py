"""Training-only task examples, self exclusion, duplicate audit."""

import json
from difflib import SequenceMatcher

from llm_grading.data.quality import select_training_samples
from llm_grading.data.split import code_hash, normalized_code
from llm_grading.prompting.formatter import BUILDERS
from llm_grading.retrieval.index import DenseIndex
from llm_grading.training.dataset import build_training_dataset


def retrieval_text(sample, task):
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
        records = build_training_dataset(samples, task, config, training=False)
        self.targets = [row["target"] for row in records]
        self.index = DenseIndex(
            [retrieval_text(s, task) for s in samples], self.settings, encoder
        )

    def retrieve(self, query):
        scores = self.index.scores(retrieval_text(query, self.task))
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
        for i, sample in enumerate(self.samples):
            if sample["sample_id"] == query["sample_id"]:
                audit["self_excluded"].append(sample["sample_id"])
                continue
            if (
                self.task == "task3"
                and sample["feedback_level"] != query["feedback_level"]
            ):
                continue
            exact = code_hash(sample["code"]) == code_hash(query["code"])
            near = bool(
                cutoff
                and SequenceMatcher(
                    None,
                    normalized_code(sample["code"]),
                    normalized_code(query["code"]),
                    autojunk=False,
                ).ratio()
                >= cutoff
            )
            if exact:
                audit["exact_matches"].append(sample["sample_id"])
            if near:
                audit["near_matches"].append(sample["sample_id"])
            if restrict and (exact or near):
                continue
            candidates.append((float(scores[i]), i))
        candidates.sort(key=lambda pair: (-pair[0], self.samples[pair[1]]["sample_id"]))
        examples = []
        for score, i in candidates[: self.settings.get("k", 3)]:
            sample = self.samples[i]
            target = (
                self.targets[i] if self.task == "task3" else json.loads(self.targets[i])
            )
            examples.append({"input": BUILDERS[self.task](sample), "target": target})
            audit["retrieved"].append(
                {"sample_id": sample["sample_id"], "score": score}
            )
        audit["shortfall"] = self.settings.get("k", 3) - len(examples)
        return examples, audit
