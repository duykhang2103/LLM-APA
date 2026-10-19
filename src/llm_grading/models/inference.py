"""Local HF or one OpenAI-compatible chat endpoint; no provider framework."""

import hashlib
import json
import os
import time
import urllib.request
from pathlib import Path

from llm_grading.models.generation import generate_text
from llm_grading.models.loader import load_model


def cache_key(cfg, prompt):
    """Identify one deterministic request: same model, decoding, and prompt."""
    model = cfg["model"]
    identity = {
        "provider": model.get("provider", "local"),
        "model": model.get("name_or_path"),
        "revision": model.get("revision"),
        "adapter": model.get("adapter_path"),
        "base_url": model.get("base_url"),
        "generation": cfg.get("generation", {}),
        "prompt": prompt,
    }
    return hashlib.sha256(
        json.dumps(identity, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


class ModelRunner:
    def __init__(self, config):
        self.config = config
        self.provider = config["model"].get("provider", "local")
        if self.provider not in {"local", "api"}:
            raise ValueError("model.provider must be local or api")
        if self.provider == "api" and not config["model"].get(
            "allow_private_data", False
        ):
            raise ValueError(
                "API transmission requires model.allow_private_data=true after checking course data permission"
            )
        # Optional response cache: a rerun after a failure reuses paid answers.
        self.cache_path = config["model"].get("cache_path")
        self.cache = {}
        if self.cache_path and Path(self.cache_path).exists():
            for line in Path(self.cache_path).read_text(encoding="utf-8").splitlines():
                if line.strip():
                    row = json.loads(line)
                    self.cache[row["key"]] = row
        self.max_cost = config["model"].get("max_cost_usd")
        self.total_cost = 0.0
        self.bundle = load_model(config) if self.provider == "local" else None
        self.metadata = {}

    def generate(self, prompt, config=None):
        cfg = config or self.config
        key = cache_key(cfg, prompt) if self.cache_path else None
        if key in self.cache:
            row = self.cache[key]
            self.metadata = {
                "model": cfg["model"]["name_or_path"],
                "revision": row.get("revision"),
                "decoding": cfg["generation"],
                "usage": row.get("usage", {}),
                "latency_seconds": 0.0,
                "estimated_cost": 0.0,
                "cached": True,
            }
            return row["text"]
        if self.max_cost is not None and self.total_cost >= self.max_cost:
            raise RuntimeError(
                f"Stopped: run cost ${self.total_cost:.4f} reached model.max_cost_usd={self.max_cost}; "
                "cached answers are kept, rerun to continue"
            )
        text = self._generate(prompt, cfg)
        self.total_cost += self.metadata.get("estimated_cost") or 0.0
        if self.cache_path:
            row = {
                "key": key,
                "text": text,
                "usage": self.metadata.get("usage", {}),
                "revision": self.metadata.get("revision"),
            }
            Path(self.cache_path).parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            self.cache[key] = row
        self.metadata["cached"] = False
        self.metadata["run_total_cost"] = self.total_cost
        return text

    def _generate(self, prompt, cfg):
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
