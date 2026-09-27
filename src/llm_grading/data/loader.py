"""Load raw challenge records and resolve source-code file references.

Expected future flow:

1. Read challenge JSON from a private path.
2. Validate required metadata and the ``code_file`` field.
3. Resolve the source path below the configured private data root.
4. Read C++ text into ``code`` without copying it into tracked files.
5. Return records matching :mod:`llm_grading.data.schema`.

Use ``examples/normalized_sample.json`` as a safe shape example. It is not
an instruction to commit or distribute real submissions.
"""

from pathlib import Path
import json

from .schema import NormalizedSample


def load_samples(source: str | Path, max_samples: int | None = None) -> list[NormalizedSample]:
    """Return normalized samples from a private JSON/data directory.

    The implementation should also report missing files, duplicate IDs, and
    malformed records in a data-audit report instead of silently dropping them.
    """
    source_path = Path(source).resolve()
    if not source_path.exists():
        raise FileNotFoundError(f"Sample source does not exist: {source_path}")
    paths = sorted(source_path.glob("*.json")) if source_path.is_dir() else [source_path]
    if not paths:
        raise FileNotFoundError(f"No JSON files found under: {source_path}")
    records: list[NormalizedSample] = []
    for path in paths:
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload = [payload] if isinstance(payload, dict) else payload
        if not isinstance(payload, list):
            raise ValueError(f"Expected an object or list in {path}")
        for record in payload:
            if not isinstance(record, dict) or not record.get("sample_id"):
                raise ValueError(f"Every sample must be an object with sample_id: {path}")
            sample = dict(record)
            if not sample.get("code") and sample.get("code_file"):
                code_path = Path(str(sample["code_file"]))
                candidates = [path.parent / code_path, source_path.parent / code_path]
                for candidate in candidates:
                    if candidate.exists():
                        sample["code"] = candidate.read_text(encoding="utf-8")
                        break
                if not sample.get("code"):
                    raise FileNotFoundError(
                        f"Could not resolve code_file for {sample['sample_id']}: {sample['code_file']}"
                    )
            sample.setdefault("code", "")
            records.append(sample)
    if max_samples is not None:
        records = records[:max_samples]
    ids = [str(record["sample_id"]) for record in records]
    if len(ids) != len(set(ids)):
        duplicates = sorted({item for item in ids if ids.count(item) > 1})
        raise ValueError(f"Duplicate sample_id values: {duplicates}")
    return records
