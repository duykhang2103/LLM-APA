"""Paste this entire file into one cell in the existing Kaggle P1 session.

Uses the existing runtime API and first stored attempts; never loads a model.
"""
import json
import re
import time
import unicodedata
from pathlib import Path

RUN = Path("/kaggle/working/task3_P1_dev")
DESTINATION = Path("/kaggle/working/predictions_P1.json")


def recover_p1(runtime, run=RUN, destination=DESTINATION):
    run, destination = Path(run), Path(destination)
    rows = runtime.read_json(run / "targets.private.json")
    by_id = runtime.unique_rows(rows)
    if len(by_id) != 328:
        raise ValueError("Recovery requires the complete 328-target P1 run")
    manifest = runtime.read_json(run / "manifest.json")
    cfg = dict(manifest["config"])
    if cfg["method"] != "P1" or cfg["mode"] != "dev":
        raise ValueError("Expected P1 dev manifest")
    official = manifest["contract"]["policy"]["taxonomy_error"]

    def key(text):
        return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()

    lookup = {key(label): label for label in official}
    lookup.update({key(alias): "Lỗi biên dịch" for alias in ("Lỗi syntax", "Lỗi cú pháp")})
    failure_files = sorted((run / "failures").glob("*.json"))
    if len(failure_files) != 6:
        raise ValueError(f"Expected the six confirmed failure records, found {len(failure_files)}")
    plans, audit, seen = {}, [], set()
    for path in failure_files:
        failure = runtime.read_json(path)
        sid = failure["sample_id"]
        if sid not in by_id or sid in seen or failure["stage"] != "feedback":
            raise ValueError("Unexpected/duplicate failure ID or stage")
        seen.add(sid)
        # Deliberately use only index zero, never select a later retry.
        first = failure["attempts"][0]
        if first.get("ended_with_eos") is not True or first.get("error"):
            raise ValueError(f"First response incomplete for {sid}; no inference started")
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", first["text"].strip()).strip()
        raw = json.loads(text)
        if not isinstance(raw, dict) or set(raw) != {"taxonomy_error", "feedback"}:
            raise ValueError(f"Invalid first-response schema for {sid}")
        labels, events = [], []
        if not isinstance(raw["taxonomy_error"], list):
            raise ValueError(f"Invalid label list for {sid}")
        for label in raw["taxonomy_error"]:
            if not isinstance(label, str) or key(label) not in lookup:
                raise ValueError(f"Unknown taxonomy label for {sid}: {label!r}")
            canonical = lookup[key(label)]
            duplicate = canonical in labels
            if label != canonical or duplicate:
                events.append({"original": label, "canonical": canonical, "duplicate_removed": duplicate})
            if not duplicate:
                labels.append(canonical)
        prediction = {"taxonomy_error": labels, "feedback": raw["feedback"]}
        # Existing runtime performs the structural and internal-content checks.
        checked = runtime.parse_response(json.dumps(prediction, ensure_ascii=False), by_id[sid])
        if checked != prediction:
            raise ValueError(f"Existing validation altered prediction for {sid}")
        for name in ("batch_seconds", "batch_size", "input_tokens", "output_tokens"):
            if name not in first:
                raise ValueError(f"Missing generation metadata {name} for {sid}")
        plans[sid] = {"sample_id": sid, **prediction, "flags": [],
                      "warnings": runtime.feedback_flags(prediction["feedback"], by_id[sid]["level"]),
                      "generation": {**first, "attempts": 1}, "retrieved_ids": [], "worker": None}
        audit.append({"sample_id": sid, "failure_file": str(path), "attempt_index": 0, "events": events})

    merged, to_write = [], []
    for row in rows:
        path = runtime.sample_cache_path(run, row)
        if path.exists():
            cached = runtime.read_json(path)  # Never replace an existing checkpoint.
        else:
            if row["sample_id"] not in plans:
                raise ValueError(f"No recoverable first response for {row['sample_id']}")
            cached = plans[row["sample_id"]]
            to_write.append((path, cached))
        if cached["sample_id"] != row["sample_id"] or cached["flags"]:
            raise ValueError("Invalid existing checkpoint; left unchanged")
        merged.append({"sample_id": cached["sample_id"], "output": {
            "taxonomy_error": cached["taxonomy_error"], "feedback": cached["feedback"]}})
    runtime.validate_submission(merged, rows, official, leaderboard=True)
    # Preflight above validates all 328 before creating any recovered checkpoints.
    for path, cached in to_write:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as handle:
            json.dump(cached, handle, ensure_ascii=False, indent=2)
    audit_path = run / f"recovery_normalization_{time.time_ns()}.private.json"
    with audit_path.open("x", encoding="utf-8") as handle:
        json.dump({"created": len(to_write), "preserved": 328 - len(to_write),
                   "selection": "first_stored_attempt_only", "events": audit}, handle, ensure_ascii=False, indent=2)
    runtime.atomic_json(destination, merged)
    runtime.validate_submission(runtime.read_json(destination), rows, official, leaderboard=True)
    print(f"Recovered {len(to_write)}; preserved {328 - len(to_write)}; validated 328: {destination}")
    print("No model loading/inference. Failure records and existing checkpoints unchanged.")
    return destination


if "runtime" in globals():
    recovered_submission = recover_p1(globals()["runtime"])
elif __name__ == "__main__":
    raise RuntimeError("Paste into the existing notebook session where runtime is already loaded")
