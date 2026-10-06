"""Load the standalone experiment YAML files used by every CLI."""

import json
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None


REQUIRED_SECTIONS = {
    "experiment", "method", "model", "data", "training", "logging",
    "saving", "evaluation", "generation",
}


def _strip_comment(value: str) -> str:
    quoted = False
    quote = ""
    for index, character in enumerate(value):
        if character in "\"'":
            if not quoted:
                quoted, quote = True, character
            elif quote == character:
                quoted = False
        elif character == "#" and not quoted and (index == 0 or value[index - 1].isspace()):
            return value[:index].rstrip()
    return value.rstrip()


def _parse_scalar(value: str) -> Any:
    value = value.strip()
    if value in {"null", "Null", "NULL", "~"}:
        return None
    if value.lower() in {"true", "false"}:
        return value.lower() == "true"
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
        return value[1:-1]
    if value.startswith("[") and value.endswith("]"):
        inner = value[1:-1].strip()
        return [] if not inner else [_parse_scalar(part) for part in inner.split(",")]
    if value == "{}":
        return {}
    if re.fullmatch(r"[-+]?\d+", value):
        return int(value)
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+|\d+)(?:[eE][-+]?\d+)?", value):
        return float(value)
    return value


def _minimal_yaml_load(text: str) -> dict[str, Any]:
    """Parse the scalar/list YAML subset used in this repository."""
    lines = []
    for raw in text.splitlines():
        content = _strip_comment(raw)
        if content:
            lines.append((len(content) - len(content.lstrip()), content.strip()))

    def parse_block(index: int, indent: int) -> tuple[Any, int]:
        is_list = lines[index][1].startswith("-")
        result: Any = [] if is_list else {}
        while index < len(lines):
            current_indent, content = lines[index]
            if current_indent < indent:
                break
            if current_indent > indent:
                raise ValueError(f"Unexpected indentation near: {content}")
            if is_list:
                if not content.startswith("-"):
                    break
                result.append(_parse_scalar(content[1:].strip()))
                index += 1
                continue
            if content.startswith("-"):
                break
            key, separator, raw_value = content.partition(":")
            if not separator:
                raise ValueError(f"Expected a YAML key near: {content}")
            index += 1
            if raw_value.strip():
                result[key.strip()] = _parse_scalar(raw_value)
            elif index < len(lines) and lines[index][0] > indent:
                result[key.strip()], index = parse_block(index, lines[index][0])
            else:
                result[key.strip()] = None
        return result, index

    if not lines:
        return {}
    parsed, _ = parse_block(0, lines[0][0])
    if not isinstance(parsed, dict):
        raise ValueError("Config root must be a mapping")
    return parsed


def load_config(path: str | Path) -> dict[str, Any]:
    """Load one YAML config and return a validated dictionary."""
    config_path = Path(path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file does not exist: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    config = json.loads(text) if config_path.suffix == ".json" else (yaml.safe_load(text) if yaml is not None else _minimal_yaml_load(text))
    if not isinstance(config, dict):
        raise ValueError(f"Config must contain a mapping: {config_path}")
    missing = REQUIRED_SECTIONS.difference(config)
    if missing:
        raise ValueError(f"Missing config sections: {sorted(missing)}")
    task = config.get("task")
    if not task:
        match = re.search(r"task[123]", config_path.parent.name.lower())
        task = match.group(0) if match else None
    if not task:
        raise ValueError("Config must define a top-level 'task' such as 'task1'")
    if task not in {"task1", "task2", "task3"}:
        raise ValueError(f"Unknown task: {task}")
    if config["method"] not in {"heuristic", "zero_shot", "rag", "lora", "qlora"}:
        raise ValueError("Unknown experiment method")
    if any(k in config["model"] for k in ("api_key", "token")):
        raise ValueError("API credentials belong in environment variables")
    if config["method"] in {"lora", "qlora"}:
        if not config.get("lora") or config["training"].get("max_sequence_length", 0) < 1:
            raise ValueError("LoRA settings and positive training.max_sequence_length are required")
    config["task"] = task
    config["_config_path"] = str(config_path)
    return config
