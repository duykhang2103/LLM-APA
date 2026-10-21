"""P0/P1/F0 prediction, strict parsing, one bounded format-repair retry."""

from pathlib import Path

from llm_grading.data.loader import load_samples
from llm_grading.models.inference import ModelRunner
from llm_grading.prompting.formatter import parse_response, render_task_prompt
from llm_grading.retrieval.retriever import Retriever
from llm_grading.runtime import attach_saved_evidence
from llm_grading.utils.logging import append_jsonl, sample_metadata
from tasks.task3_feedback.compliance import check_compliance


class ModelPipeline:
    def __init__(self, config, runner=None, retriever=None):
        self.config, self.task = config, config["task"]
        if config["method"] in {"lora", "qlora"} and not config["model"].get(
            "adapter_path"
        ):
            raise ValueError(
                "Fine-tuned prediction requires model.adapter_path; cannot label a base-model run F0"
            )
        if (
            config["method"] in {"lora", "qlora"}
            and config["model"].get("provider", "local") != "local"
        ):
            raise ValueError("Fine-tuned inference must use the local adapter runner")
        self.retriever = retriever
        if config["method"] == "rag" and self.retriever is None:
            path = config["data"].get("train_path")
            if not path:
                raise ValueError(
                    "RAG requires explicit data.train_path; no validation/test fallback"
                )
            if any(
                config["data"].get(key)
                and Path(path).resolve() == Path(config["data"][key]).resolve()
                for key in ["validation_path", "test_path"]
            ):
                raise ValueError("RAG train_path cannot be a validation/test file")
            corpus = load_samples(path, task=self.task)
            # Examples must carry the same evidence fields as the query.
            evidence = config.get("evidence", {})
            attach_saved_evidence(corpus, evidence.get("train_compile"), evidence.get("train_tests"))
            self.retriever = Retriever(corpus, self.task, config)
        self.runner = runner or ModelRunner(config)
        self.log_path = Path(config["saving"]["output_dir"]) / "responses.jsonl"

    def predict(self, sample):
        examples, audit = (
            self.retriever.retrieve(sample) if self.retriever else ([], {})
        )
        prompt = render_task_prompt(sample, self.task, self.config, examples)
        error = None
        retries = self.config.get("task3", {}).get("max_compliance_retries", 1)
        if (
            isinstance(retries, bool)
            or not isinstance(retries, int)
            or not 0 <= retries <= 3
        ):
            raise ValueError("task3.max_compliance_retries must be between 0 and 3")
        attempts = retries + 1 if self.task == "task3" else 2
        for retry in range(attempts):
            retry_prompt = prompt
            if retry:
                instruction = (
                    "Rewrite the feedback at the requested level; do not reveal a fix or diagnosis beyond that level."
                    if self.task == "task3"
                    else "Return only the requested JSON schema."
                )
                retry_prompt += (
                    "\nPrevious response rejected. "
                    + instruction
                    + " Error: "
                    + str(error)
                )
            try:
                raw = self.runner.generate(retry_prompt)
            except Exception as exc:
                append_jsonl(
                    self.log_path,
                    {
                        **sample_metadata(sample),
                        "retry_count": retry,
                        "generation_error": str(exc),
                        "retrieval": audit,
                    },
                )
                # Oversized prompts raise ValueError; network/API failures do not
                # fall back, so a quota outage cannot silently fill a whole run.
                if isinstance(exc, ValueError) and self._fallback_enabled():
                    return self._fallback(sample, f"generation_error: {exc}")
                raise RuntimeError(
                    f"Generation failed for {sample['sample_id']}; see {self.log_path}"
                ) from exc
            try:
                parsed = parse_response(raw, self.task, sample)
                if self.task == "task3" and not parsed["compliance"]["pass"]:
                    raise ValueError(
                        "Feedback policy violation: "
                        + ", ".join(parsed["compliance"]["violations"])
                    )
                error = None
            except ValueError as exc:
                parsed = None
                error = str(exc)
            append_jsonl(
                self.log_path,
                {
                    **sample_metadata(sample),
                    "raw_response": raw,
                    "parsed_output": parsed,
                    "parse_error": error,
                    "retry_count": retry,
                    "retrieval": audit,
                    "inference": self.runner.metadata,
                },
            )
            if parsed is not None:
                return parsed
        if self._fallback_enabled():
            return self._fallback(sample, error)
        retry_description = "one retry" if attempts == 2 else f"{attempts - 1} retries"
        raise ValueError(
            f"Invalid response for {sample['sample_id']} after {retry_description}: {error}; see {self.log_path}"
        )

    def _fallback_enabled(self):
        return self.task == "task3" and bool(
            self.config.get("task3", {}).get("fallback_on_failure", False)
        )

    def _fallback(self, sample, reason):
        """Level-safe generic feedback so one bad sample cannot stop a full run."""
        feedback = level_safe_feedback(sample.get("error_labels", []), sample["feedback_level"])
        result = {
            "error_labels": list(sample.get("error_labels", [])),
            "feedback": feedback,
            "feedback_level": sample["feedback_level"],
            "compliance": check_compliance(feedback, sample["feedback_level"]),
            "fallback": True,
        }
        append_jsonl(
            self.log_path,
            {**sample_metadata(sample), "fallback": True, "fallback_reason": reason, "parsed_output": result},
        )
        return result

    def postprocess(self, output):
        return output


def level_safe_feedback(labels, level):
    if not labels:
        return "Bài làm của em đã đáp ứng yêu cầu của đề, em tiếp tục phát huy nhé."
    if level == 1:
        return "Bài làm của em vẫn chưa đáp ứng hết yêu cầu của đề, em hãy đọc lại đề và tự kiểm tra với vài bộ dữ liệu nhé."
    kinds = ", ".join(label[0].lower() + label[1:] for label in labels)
    return f"Bài làm của em còn {kinds}; em hãy xem lại phần xử lý liên quan trong bài và đối chiếu với yêu cầu của đề."
