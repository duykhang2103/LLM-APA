"""Small runtime helpers shared by the five beginner-friendly CLIs."""

from pathlib import Path
import json
from typing import Any, Mapping

from .data.loader import load_samples
from .utils.paths import resolve_path


def load_samples_for_config(config: Mapping[str, Any], split: str = "test") -> list[dict[str, Any]]:
    """Load one task's configured split; val and validation are aliases."""
    data = config.get("data", {})
    split_key = "validation" if split == "val" else split
    source = data.get(f"{split_key}_path")
    if split_key == "validation":
        source = source or data.get("val_path")
    source = source or data.get("input_path")
    if not source:
        raise ValueError(f"No input path configured for split '{split}'")
    return load_samples(resolve_path(source), data.get("max_samples"), task=config.get("task"))


def write_json(path: str | Path, payload: Any) -> Path:
    """Write UTF-8 JSON, creating only the requested parent directory."""
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return output_path


def read_json(path: str | Path) -> Any:
    """Read UTF-8 JSON with a clear path in errors from the CLIs."""
    input_path = Path(path)
    try:
        return json.loads(input_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise FileNotFoundError(f"JSON file does not exist: {input_path}") from error
