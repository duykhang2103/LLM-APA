"""P0/P1/F0 prediction, strict parsing, one bounded format-repair retry."""

from pathlib import Path

from llm_grading.data.loader import load_samples
from llm_grading.models.inference import ModelRunner
from llm_grading.prompting.formatter import parse_response, render_task_prompt
from llm_grading.retrieval.retriever import Retriever
from llm_grading.utils.logging import append_jsonl, sample_metadata


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
            self.retriever = Retriever(
                load_samples(path, task=self.task), self.task, config
            )
        self.runner = runner or ModelRunner(config)
        self.log_path = Path(config["saving"]["output_dir"]) / "responses.jsonl"

    def predict(self, sample):
        examples, audit = (
            self.retriever.retrieve(sample) if self.retriever else ([], {})
        )
        prompt = render_task_prompt(sample, self.task, self.config, examples)
        error = None
        attempts = 1 if self.task == "task3" else 2
        for retry in range(attempts):
            retry_prompt = (
                prompt
                if retry == 0
                else prompt
                + "\nYour previous response was invalid. Return only the requested JSON schema. Error: "
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
                raise RuntimeError(
                    f"Generation failed for {sample['sample_id']}; see {self.log_path}"
                ) from exc
            try:
                parsed = parse_response(raw, self.task, sample)
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
        raise ValueError(
            f"Invalid response for {sample['sample_id']} after one retry: {error}; see {self.log_path}"
        )

    def postprocess(self, output):
        return output
