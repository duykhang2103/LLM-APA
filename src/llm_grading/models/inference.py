"""Local HF or one OpenAI-compatible chat endpoint; no provider framework."""

import json
import os
import time
import urllib.request

from llm_grading.models.generation import generate_text
from llm_grading.models.loader import load_model


class ModelRunner:
    def __init__(self, config):
        self.config = config
        self.provider = config["model"].get("provider", "local")
        if self.provider not in {"local", "api"}:
            raise ValueError("model.provider must be local or api")
        self.bundle = load_model(config) if self.provider == "local" else None
        if self.provider == "api" and not config["model"].get(
            "allow_private_data", False
        ):
            raise ValueError(
                "API transmission requires model.allow_private_data=true after checking course data permission"
            )
        self.metadata = {}

    def generate(self, prompt, config=None):
        cfg = config or self.config
        start = time.perf_counter()
        if self.provider == "local":
            text = generate_text(self.bundle, prompt, cfg["generation"])
            usage = self.bundle["last_usage"]
            revision = self.bundle["revision"]
        else:
            settings = cfg["model"]
            generation = cfg["generation"]
            key = os.environ.get(settings.get("api_key_env", "LLM_API_KEY"))
            if not key:
                raise ValueError("Missing configured API key environment variable")
            body = {
                "model": settings["name_or_path"],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": generation.get("temperature", 0.0),
                "max_tokens": generation.get("max_new_tokens", 256),
            }
            request = urllib.request.Request(
                settings["base_url"].rstrip("/") + "/chat/completions",
                data=json.dumps(body).encode(),
                headers={
                    "Authorization": f"Bearer {key}",
                    "Content-Type": "application/json",
                    "User-Agent": "Mozilla/5.0 (compatible; llm-apa/1.0)",
                },
            )
            max_api_retries = 5
            for attempt in range(max_api_retries):
                try:
                    with urllib.request.urlopen(
                        request, timeout=settings.get("timeout_seconds", 120)
                    ) as response:
                        result = json.load(response)
                    break
                except urllib.error.HTTPError as e:
                    err_body = e.read().decode("utf-8", errors="replace")
                    if e.code == 429 and attempt < max_api_retries - 1:
                        time.sleep(15 * (attempt + 1))
                        continue
                    raise RuntimeError(f"API request failed with {e.code} {e.reason}: {err_body}") from e
            text = result["choices"][0]["message"]["content"]
            usage = result.get("usage", {})
            revision = result.get("system_fingerprint") or settings.get("revision")
        rates = cfg["model"].get("cost_per_million_tokens", {})
        cost = (
            sum(
                usage.get(f"{kind}_tokens", 0) * rates.get(kind, 0) / 1e6
                for kind in ("prompt", "completion")
            )
            if rates
            else None
        )
        self.metadata = {
            "model": cfg["model"]["name_or_path"],
            "revision": revision,
            "decoding": cfg["generation"],
            "usage": usage,
            "latency_seconds": time.perf_counter() - start,
            "estimated_cost": cost,
        }
        return text


def generate(prompt, config):
    """Convenience for one call; pipelines reuse ModelRunner to avoid reloads."""
    return ModelRunner(config).generate(prompt)


def run_inference(model, model_input):
    return model.generate(model_input["prompt"])
