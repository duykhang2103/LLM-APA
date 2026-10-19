"""Retrieval-only kNN evaluation on labeled training data; no LLM is called.

LOO: every training sample queries all others (the Retriever excludes self and
near duplicates). LOEO: queries from one exam retrieve only from the other
exams, mimicking public-test exams that have no labeled sample. The kNN
prediction copies neighbour labels, so it is the floor an LLM+RAG run must beat.
"""

import argparse
import json
import statistics
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.data.schema import RUBRIC_RANGES
from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.evaluation.task1 import evaluate_task1
from llm_grading.evaluation.task2 import evaluate_task2
from llm_grading.retrieval.retriever import Retriever
from llm_grading.runtime import write_json
from llm_grading.utils.logging import append_jsonl
from llm_grading.utils.seed import set_seed

EVALUATE = {"task1": evaluate_task1, "task2": evaluate_task2}


class CachingEncoder:
    """Encode each distinct text once across tasks, protocols, and variants."""

    def __init__(self, model):
        self.model, self.cache = model, {}

    def encode(self, texts, normalize_embeddings=True):
        missing = [t for t in dict.fromkeys(texts) if t not in self.cache]
        if missing:
            vectors = self.model.encode(missing, normalize_embeddings=normalize_embeddings)
            self.cache.update(zip(missing, vectors))
        return np.asarray([self.cache[t] for t in texts])

    def __getitem__(self, index):
        return self.model[index]

    def __getattr__(self, name):
        return getattr(self.model, name)


def load_encoder(settings):
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(
        settings["embedding_model"],
        revision=settings.get("embedding_revision"),
        device=settings.get("device", "cpu"),
    )
    if settings.get("max_seq_length"):
        model.max_seq_length = settings["max_seq_length"]
    return CachingEncoder(model)


def predict_task1(targets):
    return {k: int(statistics.median_low([t[k] for t in targets])) for k in RUBRIC_RANGES}


def predict_task2(targets):
    votes = Counter(label for t in targets for label in t["error_labels"])
    needed = len(targets) // 2 + 1
    return {"error_labels": [x for x in ERROR_LABELS if votes[x] >= needed]}


PREDICT = {"task1": predict_task1, "task2": predict_task2}


def majority(pool, task):
    if task == "task1":
        rubric = Counter(tuple(s["rubric"][k] for k in RUBRIC_RANGES) for s in pool)
        return dict(zip(RUBRIC_RANGES, rubric.most_common(1)[0][0]))
    sets = Counter(tuple(s["error_labels"]) for s in pool)
    return {"error_labels": list(sets.most_common(1)[0][0])}


def neighbour_gap(query, target, task):
    if task == "task1":
        return abs(sum(query["rubric"].values()) - sum(target.values()))
    a, b = set(query["error_labels"]), set(target["error_labels"])
    return len(a & b) / len(a | b) if a | b else 1.0


def run_protocol(samples, task, settings, protocol, encoders, details_path, name):
    """Return {k: predictions} plus per-query top-1 diagnostics in query order."""
    config = {"retrieval": settings}
    if protocol == "loo":
        groups = [(samples, samples)]
    else:
        exams = sorted({s["problem_id"] for s in samples})
        groups = [
            (
                [s for s in samples if s["problem_id"] == exam],
                [s for s in samples if s["problem_id"] != exam],
            )
            for exam in exams
        ]
    predictions = {"top1": {}, "top3": {}, "majority": {}}
    gaps, same_exam = {}, {}
    for queries, pool in groups:
        encoder = None
        if settings.get("index", "dense") != "bm25":
            key = (settings["embedding_model"], settings.get("max_seq_length"))
            if key not in encoders:
                encoders[key] = load_encoder(settings)
            encoder = encoders[key]
        retriever = Retriever(pool, task, config, encoder)
        fallback = majority(pool, task)
        for query in queries:
            examples, audit = retriever.retrieve(query)
            targets = [e["target"] for e in examples]
            sid = query["sample_id"]
            predictions["majority"][sid] = fallback
            predictions["top1"][sid] = PREDICT[task](targets[:1]) if targets else fallback
            predictions["top3"][sid] = PREDICT[task](targets) if targets else fallback
            if targets:
                gaps[sid] = neighbour_gap(query, targets[0], task)
                same_exam[sid] = audit["retrieved"][0]["same_exam"]
            append_jsonl(
                details_path,
                {"variant": name, "protocol": protocol, "task": task, "sample_id": sid, "retrieval": audit},
            )
    return predictions, gaps, same_exam


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--data", help="Training dataset directory (overrides data.train_path)")
    parser.add_argument("--variants", nargs="*", help="Subset of variant names")
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    set_seed(config.get("seed", 42))
    data = args.data or config["data"]["train_path"]
    out = Path(config["saving"]["output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    details = out / "retrieval_details.jsonl"
    details.unlink(missing_ok=True)
    samples = {task: load_samples(data, task=task) for task in EVALUATE}
    rows, encoders = [], {}
    for variant in config["variants"]:
        if args.variants and variant["name"] not in args.variants:
            continue
        settings = {**config.get("retrieval_defaults", {}), **variant}
        name = settings.pop("name")
        for protocol in ("loo", "loeo"):
            row = {"variant": name, "protocol": protocol}
            for task, task_samples in samples.items():
                predictions, gaps, same = run_protocol(
                    task_samples, task, settings, protocol, encoders, details, name
                )
                for mode, by_id in predictions.items():
                    ordered = [{"output": by_id[s["sample_id"]]} for s in task_samples]
                    metrics = EVALUATE[task](ordered, task_samples)
                    if task == "task1":
                        row[f"t1_{mode}_qwk"] = round(metrics["qwk_total"], 4)
                        row[f"t1_{mode}_mae"] = round(metrics["mae_total"], 4)
                    else:
                        row[f"t2_{mode}_macro_f1"] = round(metrics["macro_f1"], 4)
                        row[f"t2_{mode}_micro_f1"] = round(metrics["micro_f1"], 4)
                key = "t1_top1_abs_total_gap" if task == "task1" else "t2_top1_label_jaccard"
                row[key] = round(sum(gaps.values()) / len(gaps), 4) if gaps else None
                row[f"{task}_top1_same_exam_rate"] = round(sum(same.values()) / len(same), 4) if same else None
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    write_json(out / "metrics.json", {"config": config, "data": str(data), "rows": rows})
    print(f"Wrote {out / 'metrics.json'}")


if __name__ == "__main__":
    raise SystemExit(main())
