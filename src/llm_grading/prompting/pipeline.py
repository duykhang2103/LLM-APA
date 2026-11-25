"""P0/P1/F0 prediction, strict parsing, one bounded format-repair retry."""

from pathlib import Path

from llm_grading.data.loader import load_samples
from llm_grading.models.inference import ModelRunner
from llm_grading.prompting.formatter import (
    parse_response,
    render_task_prompt,
    salvage_task2_labels,
)
from llm_grading.retrieval.retriever import Retriever
from llm_grading.runtime import attach_saved_evidence
from llm_grading.utils.logging import append_jsonl, sample_metadata
from llm_grading.validation.task3 import task3_feedback_policy, uses_joint_task3_output
from tasks.task1_grading.evidence_rules import FIELDS as RUBRIC_FIELDS
from tasks.task1_grading.evidence_rules import rubric_from_evidence
from tasks.task3_feedback.compliance import check_compliance


class ModelPipeline:
    def __init__(self, config, runner=None, retriever=None):
        self.config, self.task = config, config["task"]
        if (
            self.task == "task3"
            and uses_joint_task3_output(config)
            and config.get("task3", {}).get("fallback_on_failure", False)
        ):
            raise ValueError("Task 3 joint output does not support fallback_on_failure")
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
        if self.task == "task3" and self.config.get("task3", {}).get("two_stage", False):
            try:
                sample = self._diagnose(sample)
            except ValueError as exc:
                # Diagnosis JSON invalid twice or prompt too long for the GPU:
                # still write feedback, with evidence-only labels.
                if not self._fallback_enabled():
                    raise
                sample = {**sample, "error_labels": evidence_labels(sample)}
                sample.pop("error_labels_missing", None)
                append_jsonl(self.log_path, {**sample_metadata(sample), "stage": "diagnosis",
                                             "diagnosis_fallback": str(exc)})
        examples, audit = (
            self.retriever.retrieve(sample) if self.retriever else ([], {})
        )
        prompt = render_task_prompt(sample, self.task, self.config, examples)
        error = None
        raw = ""
        retries = self.config.get("task3", {}).get("max_compliance_retries", 1)
        if (
            isinstance(retries, bool)
            or not isinstance(retries, int)
            or not 0 <= retries <= 3
        ):
            raise ValueError("task3.max_compliance_retries must be between 0 and 3")
        attempts = retries + 1 if self.task == "task3" else 2
        policy = task3_feedback_policy(self.config) if self.task == "task3" else "strict"
        for retry in range(attempts):
            policy_warning = []
            retry_prompt = prompt
            if retry:
                instruction = (
                    "Return the requested response format and rewrite the feedback at the requested level; do not reveal a fix or diagnosis beyond that level."
                    if self.task == "task3"
                    else "Return only the requested JSON schema."
                )
                retry_prompt += (
                    "\nPrevious response rejected. "
                    + instruction
                    + " Error: "
                    + str(error)
                )
                if self.task == "task3":
                    retry_prompt += "\nRejected feedback to rewrite:\n" + raw
                    if uses_joint_task3_output(self.config):
                        retry_prompt += (
                            "\nFINAL INSTRUCTION: Return only valid JSON with exactly "
                            "taxonomy_error (a list of official labels) and feedback "
                            "(a Vietnamese string). Keep feedback within the requested level."
                        )
                    else:
                        retry_prompt += (
                            "\nFINAL INSTRUCTION: Rewrite in Vietnamese prose only. "
                            "Do not quote function signatures, declarations, assignments, "
                            "return statements, code fences or corrected expressions. "
                            "For Level 1 remove all locations and causes. For Level 2 "
                            "retain location, cause and consequence but remove all fixes. "
                            "For Level 3 give a correction direction in prose, not a complete solution."
                        )
            try:
                try:
                    raw = self.runner.generate(retry_prompt)
                except ValueError as first:
                    if not examples:
                        raise
                    # Too long for the GPU: drop the RAG example, keep the student's code whole.
                    suffix = retry_prompt[len(prompt):]
                    examples = []
                    audit = {**(audit or {}), "dropped_examples": str(first)}
                    prompt = render_task_prompt(sample, self.task, self.config, examples)
                    retry_prompt = prompt + suffix
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
                parsed = parse_response(raw, self.task, sample, self.config)
                if self.task == "task3" and not parsed["compliance"]["pass"]:
                    policy_warning = parsed["compliance"]["violations"]
                    if policy == "strict":
                        raise ValueError(
                            "Feedback policy violation: " + ", ".join(policy_warning)
                        )
                error = None
            except ValueError as exc:
                parsed = None
                error = str(exc)
                settings = self.config.get("task2", {})
                if self.task == "task2" and settings.get("salvage_labels", False):
                    # A repeating label list cut off by max_new_tokens: a retry
                    # under greedy decoding repeats it, so keep the labels.
                    labels = salvage_task2_labels(raw, settings.get("max_labels", 3))
                    if labels:
                        parsed, error = {"error_labels": labels, "salvaged": True}, None
            append_jsonl(
                self.log_path,
                {
                    **sample_metadata(sample),
                    "raw_response": raw,
                    "parsed_output": parsed,
                    "parse_error": error,
                    "policy_warning": policy_warning,
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
        return bool(
            self.config.get(self.task, {}).get("fallback_on_failure", False)
        )

    def _fallback(self, sample, reason):
        """Evidence-based rubric (Task 1) or labels (Task 2), or level-safe
        generic feedback (Task 3), so one bad sample cannot stop a full run.
        The Task 1 schema allows no marker field; the log records it."""
        if self.task == "task1":
            rubric = rubric_from_evidence(
                sample, sample.get("compile_evidence"), sample.get("test_results")
            ) or dict(zip(RUBRIC_FIELDS, (1, 1, 2, 1, 1, 1)))
            result = {**rubric, "total": sum(rubric.values())}
        elif self.task == "task2":
            result = {"error_labels": evidence_labels(sample), "fallback": True}
        else:
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

    def _diagnose(self, sample):
        """Infer labels and a grounded diagnosis without using test targets."""
        import json

        from llm_grading.data.taxonomy import ERROR_LABELS

        # Task 2 prompt used to diagnose labels (team notebook: v005).
        version = self.config.get("task3", {}).get("diagnosis_prompt", "v005")
        cfg = {**self.config, "prompt": {"version": version}}
        prompt = render_task_prompt(sample, "task2", cfg)
        prompt += (
            "\nOverride the output format: return ONLY JSON with error_labels and diagnosis. "
            "diagnosis is a Vietnamese string identifying the function/problem, cause and "
            "consequence of each confirmed defect, with evidence from code. For multi_problem, "
            "inspect every required function separately. Do not invent test results. "
            "Do not include corrected code. If no defect is confirmed, explain briefly. "
            "The diagnosis is internal evidence, not student feedback."
        )
        # Optional budget: a long diagnosis is cut at max_new_tokens and its JSON breaks.
        budget = self.config.get("task3", {}).get("diagnosis_max_new_tokens")
        gen_cfg = None
        if budget:
            prompt += (" Keep diagnosis short: at most 3 sentences and about 60 words in total, "
                       "no code, no quoted declarations.")
            gen_cfg = {**self.config, "generation": {**self.config["generation"], "max_new_tokens": int(budget)}}
        error = None
        for attempt in range(2):
            text = prompt + ("\nReturn valid JSON: " + str(error) if attempt else "")
            raw = self.runner.generate(text, gen_cfg) if gen_cfg else self.runner.generate(text)
            try:
                from llm_grading.prompting.formatter import (
                    _clean_json_text,
                    _unique_object,
                )
                obj = json.loads(_clean_json_text(raw), object_pairs_hook=_unique_object)
                if not isinstance(obj, dict) or set(obj) != {"error_labels", "diagnosis"}:
                    raise ValueError("Expected error_labels and diagnosis")
                labels = obj["error_labels"]
                if (not isinstance(labels, list)
                        or any(not isinstance(x, str) or x not in ERROR_LABELS for x in labels)
                        or len(labels) != len(set(labels))):
                    raise ValueError("Invalid taxonomy labels")
                if not isinstance(obj["diagnosis"], str) or not obj["diagnosis"].strip():
                    raise ValueError("Missing grounded diagnosis")
                error = None
            except (ValueError, TypeError) as exc:
                error = str(exc)
            salvaged = False
            if error is not None and self.config.get("task3", {}).get("salvage_diagnosis"):
                # Small models often put unescaped quotes inside the diagnosis text.
                rescue = salvage_diagnosis(raw)
                if rescue is not None:
                    labels, diagnosis = rescue
                    obj, error, salvaged = {"error_labels": labels, "diagnosis": diagnosis}, None, True
            append_jsonl(self.log_path, {
                **sample_metadata(sample), "stage": "diagnosis", "retry_count": attempt,
                "raw_response": raw, "parse_error": error, "salvaged": salvaged,
                "inference": self.runner.metadata,
            })
            if error is None:
                result = dict(sample)
                result.pop("error_labels_missing", None)
                result.update(error_labels=labels, diagnosis=obj["diagnosis"])
                return result
        raise ValueError(f"Invalid diagnosis for {sample['sample_id']}: {error}")


def salvage_diagnosis(raw):
    """(labels, diagnosis) from a diagnosis answer whose JSON is broken, or None.
    Labels are official ones named before "diagnosis"; the diagnosis is the rest of the text."""
    from llm_grading.prompting.formatter import salvage_task2_labels

    text = str(raw)
    start = text.find('"diagnosis"')
    if start < 0 or '"error_labels"' not in text[:start]:
        return None
    labels = salvage_task2_labels(text[:start], max_labels=10)
    rest = text[start + len('"diagnosis"'):].lstrip(" :")
    rest = rest[1:] if rest.startswith('"') else rest
    rest = rest.rstrip().rstrip("`").rstrip().rstrip("}").rstrip().rstrip('"').strip()
    return (labels, rest) if rest else None


def evidence_labels(sample):
    """Task 2 labels from compile/test evidence alone (fallback only)."""
    evidence = sample.get("compile_evidence")
    if evidence is not None and not evidence["relaxed"]:
        return ["Lỗi biên dịch"]
    report = sample.get("test_results")
    if report and report["total"] and report["pass_rate"] == 1:
        return []
    return ["Lỗi logic"]  # the most frequent label in the training data


def level_safe_feedback(labels, level):
    if not labels:
        return "Bài làm của em đã đáp ứng yêu cầu của đề, em tiếp tục phát huy nhé."
    if level == 1:
        return "Bài làm của em vẫn chưa đáp ứng hết yêu cầu của đề, em hãy đọc lại đề và tự kiểm tra với vài bộ dữ liệu nhé."
    kinds = ", ".join(label[0].lower() + label[1:] for label in labels)
    return f"Bài làm của em còn {kinds}; em hãy xem lại phần xử lý liên quan trong bài và đối chiếu với yêu cầu của đề."
