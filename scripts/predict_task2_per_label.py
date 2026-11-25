"""Task 2 by one Yes/No question per label (like the team's d3-mini run), on an
OpenAI-compatible API, with the compile and test evidence in the prompt.

For each submission and each of the ten labels the model answers Yes or No.
When the API returns token log-probabilities, P(Yes) = p(Yes) / (p(Yes) + p(No))
and a label is kept when P(Yes) > --threshold (0.5, not tuned); otherwise the
one-word answer decides. The submission comes first in every prompt so the ten
questions share a prefix (prompt caching). Output: internal Task 2 predictions
(error_labels) for calibrate_task2.py / package_predictions.py, plus
<output>.probs.json with every P(Yes).
"""

import argparse
import hashlib
import json
import math
import os
import sys
import threading
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]
from llm_grading.data.loader import load_samples
from llm_grading.data.preprocess import build_task2_input
from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.evaluation.task2 import evaluate_task2
from llm_grading.models.inference import RETRY_STATUS, retry_wait
from llm_grading.runtime import attach_saved_evidence, write_json
from llm_grading.validation.predictions import validate_prediction_records

DEFINITIONS = ROOT / "tasks" / "task2_errors" / "prompts" / "per_label_v001.json"


def answer_probability(choice):
    """(P(Yes) or None, answer text) from one chat completion choice."""
    text = (choice.get("message") or {}).get("content") or ""
    content = ((choice.get("logprobs") or {}).get("content") or [])
    if content:
        yes = no = 0.0
        for item in content[0].get("top_logprobs") or [content[0]]:
            token = str(item.get("token", "")).strip().lower()
            if token.startswith("yes"):
                yes += math.exp(item["logprob"])
            elif token.startswith("no"):
                no += math.exp(item["logprob"])
        if yes + no > 0:
            return yes / (yes + no), text
    return None, text


def label_examples(train, label, sample, positives, max_chars=1200):
    """Teacher-labelled training code for one error type, from OTHER exams only.

    Up to `positives` submissions the teacher gave this label, plus one without it,
    so the model sees how the teacher uses the label. Only code and the Yes/No
    label are shown (never the teacher's feedback). Empty when nothing qualifies.
    """
    if positives <= 0:
        return ""
    others = [t for t in train if t["problem_id"] != sample.get("problem_id")
              and t["sample_id"] != sample["sample_id"]]
    pos = [t for t in others if label in (t.get("error_labels") or [])][:positives]
    neg = [t for t in others if label not in (t.get("error_labels") or []) and t.get("error_labels")][:1]
    if not pos:
        return ""
    parts = [f'Teacher-labelled training submissions from other exams (code only) for "{label}":']
    for k, (item, answer) in enumerate([*((t, "Yes") for t in pos), *((t, "No") for t in neg)], 1):
        code = item["code"] if len(item["code"]) <= max_chars else item["code"][:max_chars] + "\n// ... (cut)"
        parts.append(f"Example {k} (teacher's answer: {answer}):\n{code}")
    return "\n\n".join(parts)


def decide(probability, text, threshold):
    if probability is not None:
        return probability > threshold
    return text.strip().lower().startswith("yes")


class Client:
    """Thread-safe API calls with a shared JSONL cache and a cost cap."""

    def __init__(self, args):
        self.args, self.lock = args, threading.Lock()
        self.key = os.environ.get(args.api_key_env)
        if not self.key:
            raise SystemExit(f"Set {args.api_key_env} first")
        self.extra = json.loads(args.extra_body) if args.extra_body else {}
        self.cache, self.cost, self.calls = {}, 0.0, 0
        self.cache_path = Path(args.cache)
        if self.cache_path.exists():
            for line in self.cache_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row

    def ask(self, prompt):
        body = {"model": self.args.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": 1, "temperature": 0, "logprobs": True, "top_logprobs": 10, **self.extra}
        key = hashlib.sha256(json.dumps([self.args.base_url, body], sort_keys=True, ensure_ascii=False)
                             .encode("utf-8")).hexdigest()
        with self.lock:
            if key in self.cache:
                return self.cache[key]
            if self.cost >= self.args.max_cost:
                raise RuntimeError(f"Stopped: estimated cost ${self.cost:.3f} reached --max-cost {self.args.max_cost}")
        request = urllib.request.Request(
            self.args.base_url.rstrip("/") + "/chat/completions", data=json.dumps(body).encode("utf-8"),
            headers={"Authorization": f"Bearer {self.key}", "Content-Type": "application/json",
                     "User-Agent": "Mozilla/5.0 (compatible; llm-apa/1.0)"})
        for attempt in range(self.args.retries):
            try:
                with urllib.request.urlopen(request, timeout=self.args.timeout) as response:
                    result = json.load(response)
                break
            except urllib.error.HTTPError as error:
                detail = error.read().decode(errors="replace")[:300]
                if error.code not in RETRY_STATUS or attempt == self.args.retries - 1:
                    raise RuntimeError(f"API request failed with {error.code}: {detail}") from error
                time.sleep(retry_wait(attempt, error.headers.get("Retry-After")))
            except (urllib.error.URLError, TimeoutError, ConnectionError):
                if attempt == self.args.retries - 1:
                    raise
                time.sleep(retry_wait(attempt))
        probability, text = answer_probability(result["choices"][0])
        usage = result.get("usage") or {}
        row = {"key": key, "p_yes": probability, "text": text, "usage": usage}
        with self.lock:
            self.cost += (usage.get("prompt_tokens", 0) * self.args.price[0]
                          + usage.get("completion_tokens", 0) * self.args.price[1]) / 1e6
            self.calls += 1
            self.cache[key] = row
            with self.cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        return row


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="Dataset folder")
    parser.add_argument("--evidence", help="compile_evidence.py output")
    parser.add_argument("--tests", help="run_tests.py output")
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--api-key-env", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--extra-body", help='JSON, e.g. {"reasoning_effort": "none"}')
    parser.add_argument("--price", nargs=2, type=float, default=(0.0, 0.0), metavar=("IN", "OUT"))
    parser.add_argument("--max-cost", type=float, default=2.0)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--timeout", type=int, default=120)
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--cache", required=True)
    parser.add_argument("--definitions", default=str(DEFINITIONS))
    parser.add_argument("--train", help="Labeled training dataset for --examples")
    parser.add_argument("--examples", type=int, default=0,
                        help="Training submissions WITH the label shown per question (plus one without); 0 = none")
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--labeled", action="store_true", help="Input carries teacher labels: print macro-F1")
    args = parser.parse_args()

    spec = json.loads(Path(args.definitions).read_text(encoding="utf-8"))
    assert set(spec["labels"]) == set(ERROR_LABELS), "definitions must cover the ten official labels"
    samples = load_samples(args.input, task="task2")[: args.max_samples]
    attach_saved_evidence(samples, args.evidence, args.tests)
    if args.examples and not args.train:
        raise SystemExit("--examples needs --train")
    train = load_samples(args.train, task="task2") if args.examples else []
    jobs = []
    for sample in samples:
        view = build_task2_input(sample)
        view.pop("grading_policy", None)  # the lecturer said to ignore the P1 prerequisite rule
        text = json.dumps(view, ensure_ascii=False)
        for label in ERROR_LABELS:
            prompt = spec["template"].replace("{input_json}", text).replace("{label}", label) \
                .replace("{definition}", spec["labels"][label]) \
                .replace("{examples}", label_examples(train, label, sample, args.examples))
            jobs.append((sample["sample_id"], label, prompt))
    client = Client(args)
    probs, start = {}, time.time()
    print(f"{len(samples)} samples x {len(ERROR_LABELS)} labels = {len(jobs)} questions, {args.workers} at a time",
          flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(client.ask, prompt): (sid, label) for sid, label, prompt in jobs}
        for done, future in enumerate(as_completed(futures), 1):
            sid, label = futures[future]
            row = future.result()
            probs.setdefault(sid, {})[label] = {"p_yes": row["p_yes"], "answer": row["text"]}
            if done % 50 == 0 or done == len(jobs):
                print(f"  {done}/{len(jobs)} answers | new calls {client.calls} | ~${client.cost:.3f} | "
                      f"{(time.time() - start) / 60:.1f} min", flush=True)
    result = []
    for sample in samples:
        sid = sample["sample_id"]
        labels = [lab for lab in ERROR_LABELS
                  if decide(probs[sid][lab]["p_yes"], probs[sid][lab]["answer"], args.threshold)]
        result.append({"sample_id": sid, "output": {"error_labels": labels}})
    errors = validate_prediction_records(result, "task2")
    if errors:
        raise ValueError(str(errors[:5]))
    write_json(args.output, result)
    write_json(Path(args.output).with_suffix(".probs.json"), probs)
    with_probs = sum(v["p_yes"] is not None for p in probs.values() for v in p.values())
    report = {"questions": len(jobs), "with_logprobs": with_probs, "new_api_calls": client.calls,
              "estimated_cost": round(client.cost, 4),
              "labels_per_sample": round(sum(len(r["output"]["error_labels"]) for r in result) / len(result), 2)}
    if args.labeled:
        report["macro_f1"] = round(evaluate_task2(result, samples)["macro_f1"], 4)
    print(json.dumps(report, ensure_ascii=False))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    raise SystemExit(main())
