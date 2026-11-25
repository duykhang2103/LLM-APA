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
from tasks.task1_grading.evidence_rules import FIELDS as RUBRIC_FIELDS
from tasks.task1_grading.evidence_rules import rubric_from_evidence
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
        if self.task == "task3" and self.config.get("task3", {}).get("two_stage", False):
            sample = self._diagnose(sample)
            if sample.get("diagnosis_unavailable"):
                return self._fallback(sample, sample["diagnosis_failure"])
        if (self.task == "task3" and sample["feedback_level"] == 1
                and self.config.get("task3", {}).get("level1_summary", False)):
            # Diagnose first so taxonomy remains meaningful, then project to the
            # allowed level without asking the model to conceal its findings.
            text = (
                "Bài làm của em còn vấn đề và chưa đáp ứng đầy đủ yêu cầu của đề."
                if sample.get("error_labels") else
                "Qua xem xét bài làm, chưa phát hiện vấn đề rõ ràng; em hãy tiếp tục tự kiểm tra với các dữ liệu hợp lệ của đề."
            )
            result = parse_response(text, "task3", sample)
            append_jsonl(self.log_path, {
                **sample_metadata(sample), "stage": "level1_summary", "parsed_output": result,
            })
            return result
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
                if self.task == "task3":
                    retry_prompt += "\nRejected feedback to rewrite:\n" + raw
                    retry_prompt += (
                        "\nFINAL INSTRUCTION: Rewrite in Vietnamese prose only. "
                        "Do not quote function signatures, declarations, assignments, "
                        "return statements, code fences or corrected expressions. "
                        "For Level 1 remove all locations and causes. For Level 2 "
                        "retain location, cause and consequence but remove all fixes. "
                        "For Level 3 give a correction direction in prose, not a complete solution."
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
                if (self.task == "task3"
                        and self.config.get("task3", {}).get("require_correction_direction", False)
                        and sample["feedback_level"] >= 3 and sample.get("error_labels")):
                    from tasks.task3_feedback.compliance import has_correction_direction

                    if not has_correction_direction(parsed["feedback"]):
                        raise ValueError(
                            "Level 3/4 needs an explicit correction direction in Vietnamese, "
                            "grounded in the diagnosis, not just a description of defects"
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
            if sample.get("diagnosis_unavailable"):
                feedback = "Em hãy đối chiếu bài làm với yêu cầu của đề và tự kiểm tra bằng các bộ dữ liệu hợp lệ; hiện chưa đủ cơ sở để đưa ra nhận xét cụ thể."
            result = {
                "error_labels": list(sample.get("error_labels", [])),
                "feedback": feedback,
                "feedback_level": sample["feedback_level"],
                "compliance": check_compliance(feedback, sample["feedback_level"]),
                "fallback": True,
            }
            if sample.get("diagnosis_unavailable"):
                result["diagnosis_unavailable"] = True
        append_jsonl(
            self.log_path,
            {**sample_metadata(sample), "fallback": True, "fallback_reason": reason, "parsed_output": result},
        )
        return result

    def postprocess(self, output):
        return output

    def _diagnose(self, sample, *, candidate=None):
        """Infer labels and a grounded diagnosis without using test targets."""
        import json

        from llm_grading.data.taxonomy import ERROR_LABELS

        cfg = {**self.config, "prompt": {"version": "v005"}}
        compact_diagnosis = self.config.get("task3", {}).get("compact_diagnosis_prompt", False)
        if compact_diagnosis:
            from llm_grading.prompting.formatter import ROOT

            version = self.config.get("task3", {}).get("diagnosis_prompt_version", "v001")
            if version not in {"v001", "v002", "v003"}:
                raise ValueError("Unknown diagnosis_prompt_version")
            cfg["prompt"] = {"path": str(ROOT / "tasks/task3_feedback/prompts" / f"diagnosis_{version}.txt")}
        prompt = render_task_prompt(sample, "task2", cfg)
        if candidate is not None:
            from llm_grading.prompting.formatter import ROOT, format_prompt

            template = (ROOT / "tasks/task3_feedback/prompts/diagnosis_review_v001.txt").read_text(encoding="utf-8")
            prompt += "\n" + format_prompt(template, {
                "candidate_json": json.dumps(candidate, ensure_ascii=False),
            })
        prompt += "" if compact_diagnosis else (
            "\nOverride the output format: return ONLY JSON with error_labels and diagnosis. "
            "diagnosis is a Vietnamese string identifying the function/problem, cause and "
            "consequence of each confirmed defect, with evidence from code. For multi_problem, "
            "inspect every required function separately. Do not invent test results. "
            "Do not include corrected code. If no defect is confirmed, explain briefly. "
            "The diagnosis is internal evidence, not student feedback."
        )
        error = None
        raw = ""
        for attempt in range(2):
            repair = ""
            if attempt:
                repair = (
                    "\nPrevious output rejected: " + str(error)
                    + "\nPrevious output excerpt (may be truncated): " + raw[:2000]
                    + "\nRecheck the source and return ONLY a complete JSON object with error_labels "
                    "and diagnosis. Use exact official labels. diagnosis must be under 120 words; "
                    "state only confirmed findings, no self-debate or long numerical traces. "
                    "If a trace matches the expected result it does not demonstrate a bug. "
                    "Close every string and brace. Do not repeat the excerpt."
                )
            raw = self.runner.generate(prompt + repair)
            normalized = None
            try:
                from llm_grading.prompting.formatter import (
                    _clean_json_text,
                    _unique_object,
                )
                obj = json.loads(_clean_json_text(raw), object_pairs_hook=_unique_object)
                if not isinstance(obj, dict) or set(obj) != {"error_labels", "diagnosis"}:
                    raise ValueError("Expected error_labels and diagnosis")
                labels = obj["error_labels"]
                if isinstance(labels, list) and all(isinstance(x, str) for x in labels):
                    original_labels = list(labels)
                    aliases = {"Lỗi biên": "Lỗi edge case"}
                    labels = list(dict.fromkeys(aliases.get(x.strip(), x.strip()) for x in labels))
                    if labels != original_labels:
                        normalized = {"original": original_labels, "normalized": labels}
                if (not isinstance(labels, list)
                        or any(not isinstance(x, str) or x not in ERROR_LABELS for x in labels)
                        or len(labels) != len(set(labels))):
                    raise ValueError("Invalid taxonomy labels")
                if not isinstance(obj["diagnosis"], str) or not obj["diagnosis"].strip():
                    raise ValueError("Missing grounded diagnosis")
                if len(obj["diagnosis"].split()) > 200:
                    raise ValueError("Diagnosis exceeds 200 words; return concise confirmed findings")
                error = None
            except (ValueError, TypeError) as exc:
                error = str(exc)
            append_jsonl(self.log_path, {
                **sample_metadata(sample), "stage": "diagnosis_review" if candidate is not None else "diagnosis", "retry_count": attempt,
                "raw_response": raw, "parse_error": error, "inference": self.runner.metadata,
                "taxonomy_normalization": normalized,
            })
            if error is None:
                if (candidate is None
                        and self.config.get("task3", {}).get("verify_diagnosis", False)):
                    # Re-read the full original source, not only the compressed
                    # diagnosis. The reviewed JSON replaces the draft, including
                    # labels; unioning the lists would retain false positives.
                    return self._diagnose(sample, candidate={
                        "error_labels": labels, "diagnosis": obj["diagnosis"],
                    })
                result = dict(sample)
                result.pop("error_labels_missing", None)
                result.update(error_labels=labels, diagnosis=obj["diagnosis"])
                return result
        if self._fallback_enabled():
            result = dict(sample)
            result.pop("error_labels_missing", None)
            result.update(diagnosis_unavailable=True,
                          diagnosis_failure=f"Invalid diagnosis: {error}")
            return result
        raise ValueError(f"Invalid diagnosis for {sample['sample_id']}: {error}")


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
