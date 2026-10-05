"""Task-specific targets and completion-only causal language-model loss."""

import json

from llm_grading.data.schema import RUBRIC_RANGES, validate_rubric
from llm_grading.data.taxonomy import ERROR_LABELS
from llm_grading.prompting.formatter import render_task_prompt


def build_training_dataset(samples, task, config=None):
    records = []
    for s in samples:
        if s.get("_has_reference") is False:
            raise ValueError(f"Missing reference target for {s['sample_id']}")
        if task == "task1":
            validated = validate_rubric(s.get("rubric"))
            target = json.dumps({k: validated[k] for k in RUBRIC_RANGES})
        elif task == "task2":
            labels = s.get("error_labels")
            if not isinstance(labels, list) or any(
                not isinstance(x, str) or x not in ERROR_LABELS for x in labels
            ):
                raise ValueError(f"Missing/invalid labels for {s['sample_id']}")
            target = json.dumps({"error_labels": labels}, ensure_ascii=False)
        elif task == "task3":
            target = s.get("feedback")
            if not isinstance(target, str) or not target.strip():
                raise ValueError(f"Missing feedback target for {s['sample_id']}")
        else:
            raise ValueError(f"Unknown task: {task}")
        records.append(
            {
                "sample_id": s["sample_id"],
                "prompt": render_task_prompt(s, task, config or {}),
                "target": target,
            }
        )
    return records


def chat_text(tokenizer, prompt):
    if tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    return prompt + "\nAnswer:\n"


def tokenize_record(record, tokenizer, max_length):
    prefix = chat_text(tokenizer, record["prompt"])
    prompt_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    answer_ids = tokenizer(record["target"], add_special_tokens=False)["input_ids"]
    if tokenizer.eos_token_id is not None:
        answer_ids += [tokenizer.eos_token_id]
    # Do not silently remove grading evidence or supervised answer tokens.
    if len(prompt_ids) + len(answer_ids) > max_length:
        raise ValueError(
            f"{record['sample_id']}: {len(prompt_ids) + len(answer_ids)} tokens exceed max_sequence_length={max_length}; increase budget or explicitly filter the sample"
        )
    return {
        "input_ids": prompt_ids + answer_ids,
        "attention_mask": [1] * (len(prompt_ids) + len(answer_ids)),
        "labels": [-100] * len(prompt_ids) + answer_ids,
    }


class CompletionCollator:
    def __init__(self, tokenizer):
        self.pad_id = tokenizer.pad_token_id

    def __call__(self, rows):
        import torch

        n = max(len(row["input_ids"]) for row in rows)
        return {
            key: torch.tensor(
                [
                    row[key]
                    + [
                        self.pad_id
                        if key == "input_ids"
                        else (-100 if key == "labels" else 0)
                    ]
                    * (n - len(row[key]))
                    for row in rows
                ]
            )
            for key in ("input_ids", "attention_mask", "labels")
        }
