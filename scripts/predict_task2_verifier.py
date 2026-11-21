"""Apply one verifier checkpoint to frozen Task 2 taxonomy predictions."""

import argparse
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from llm_grading.data.loader import load_samples  # noqa: E402
from llm_grading.models.loader import load_model  # noqa: E402
from llm_grading.prompting.formatter import task_template  # noqa: E402
from llm_grading.runtime import read_json, write_json  # noqa: E402
from llm_grading.task2_verifier import (  # noqa: E402
    PROMPT_VERSION,
    has_source,
    merge_labels,
    render_verifier_prompt,
    routed_labels,
    verifier_choices,
)
from llm_grading.utils.config import load_config  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapter", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise ValueError("Use a fresh verifier output path")
    config = load_config(args.config)
    if config["task"] != "task2" or config["prompt"]["version"] != PROMPT_VERSION:
        raise ValueError("Expected the versioned Task 2 verifier config")
    if args.adapter:
        config["model"]["adapter_path"] = str(args.adapter)

    samples = load_samples(args.input, task="task2", preserve_source_bytes=True)
    rows = read_json(args.baseline)
    first_pass = {row["sample_id"]: row["output"]["taxonomy_error"] for row in rows}
    sample_ids = {s["sample_id"] for s in samples}
    if (
        len(first_pass) != len(rows)
        or len(sample_ids) != len(samples)
        or set(first_pass) != sample_ids
    ):
        raise ValueError("Exact unique baseline and source coverage required")
    routes = {
        sid: routed_labels(labels, recover=True) for sid, labels in first_pass.items()
    }

    import torch

    bundle = load_model(config)
    model, tokenizer = bundle["model"], bundle["tokenizer"]
    model.eval()
    template = task_template("task2", config)
    cap = config["generation"]["max_input_tokens"]
    results = []
    with torch.inference_mode():
        for sample in samples:
            sid = sample["sample_id"]
            baseline = first_pass[sid]
            decisions, skipped = {}, {}
            for label in routes[sid]:
                if not has_source(sample):
                    skipped[label] = "missing_source_or_problem"
                    continue
                prompt = render_verifier_prompt(template, sample, label)
                ids, no, yes = verifier_choices(tokenizer, prompt)
                if len(ids) > cap:
                    skipped[label] = "input_budget"
                    continue
                tokens = torch.tensor([ids], device=model.device)
                logits = model(
                    input_ids=tokens,
                    attention_mask=torch.ones_like(tokens),
                    use_cache=False,
                    logits_to_keep=1,
                ).logits
                margin = float(logits[0, -1, yes].float() - logits[0, -1, no].float())
                if not math.isfinite(margin):
                    raise RuntimeError("Non-finite verifier logit difference")
                decisions[label] = margin
            answers = {label: value > 0 for label, value in decisions.items()}
            broad = routed_labels(baseline, recover=False)
            results.append(
                {
                    "sample_id": sid,
                    "broad": merge_labels(
                        baseline,
                        {label: answers[label] for label in broad if label in answers},
                        recover=False,
                    ),
                    "full": merge_labels(baseline, answers, recover=True),
                    "margins": decisions,
                    "skipped": skipped,
                }
            )
    write_json(args.output, results)
    print(f"Wrote {len(results)} verifier decisions on {model.device}")


if __name__ == "__main__":
    main()
