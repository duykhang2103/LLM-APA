"""Deterministic alias handling and no-inference recovery of 322+6 checkpoints."""
import importlib.util
import json
import unicodedata
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_grading import task3_controlled as t3

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p1_recovery_cell", ROOT / "scripts/recover_task3_p1_kaggle_cell.py")
CELL = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CELL)


@pytest.mark.parametrize("alias", ["Lỗi syntax", "Lỗi cú pháp", "  LỖI  SYNTAX\t", unicodedata.normalize("NFD", "Lỗi cú pháp")])
def test_confirmed_aliases_and_unicode(alias):
    events = []
    raw = {"taxonomy_error": [alias, "Lỗi biên dịch", "Lỗi logic", "lỗi LOGIC"], "feedback": "Kiểm tra lại bài làm."}
    out = t3.parse_response(json.dumps(raw), {"level": 2}, normalization_events=events)
    assert out["taxonomy_error"] == ["Lỗi biên dịch", "Lỗi logic"]
    assert out["feedback"] == raw["feedback"] and events


@pytest.mark.parametrize("labels", [["Lỗi synta"], ["Loi bien dich"], ["Lỗi cú pháp", "unknown"], [42], None])
def test_unknown_labels_are_rejected(labels):
    with pytest.raises(ValueError):
        t3.normalize_taxonomy_labels(labels)


def test_p3_diagnostic_aliases_are_normalized_consistently():
    row = {"level": 2, "code": "int f(){return x;}"}
    diagnostic = {"taxonomy_error": ["Lỗi syntax", "Lỗi cú pháp"], "issues": [{
        "label": "LỖI CÚ PHÁP", "evidence": "return x", "cause": "Biến chưa khai báo."}]}
    parsed = t3.parse_response(json.dumps(diagnostic), row, stage="diagnosis")
    assert parsed["taxonomy_error"] == ["Lỗi biên dịch"]
    assert parsed["issues"][0]["label"] == "Lỗi biên dịch"
    prediction = {"taxonomy_error": ["Lỗi syntax"], "feedback": "Biến x chưa được khai báo."}
    assert t3.parse_response(json.dumps(prediction), row, diagnosis=parsed)["taxonomy_error"] == ["Lỗi biên dịch"]


@pytest.fixture
def recovery_run(tmp_path):
    run = tmp_path / "task3_P1_dev"
    rows = [{"sample_id": f"s{i}", "level": 2, "examples": []} for i in range(328)]
    t3.atomic_json(run / "targets.private.json", rows)
    t3.atomic_json(run / "manifest.json", {"config": {"method": "P1", "mode": "dev"},
                                          "contract": {"policy": {"taxonomy_error": list(t3.TAXONOMY)}}})
    first_feedback = "  Biến n chưa được khai báo.\nGiữ nguyên câu này.  "
    for index, row in enumerate(rows):
        if index < 322:
            t3.atomic_json(t3.sample_cache_path(run, row), {"sample_id": row["sample_id"],
                "taxonomy_error": [], "feedback": "Bài làm đã được xem xét.", "flags": [], "generation": {},
                "retrieved_ids": [], "worker": index % 2})
        else:
            first = {"text": json.dumps({"taxonomy_error": ["Lỗi syntax", "Lỗi cú pháp"], "feedback": first_feedback}),
                     "ended_with_eos": True, "batch_seconds": 1.5, "batch_size": 1, "input_tokens": 30, "output_tokens": 20}
            second = {**first, "text": json.dumps({"taxonomy_error": [], "feedback": "LATER RETRY MUST NEVER BE SELECTED"})}
            t3.atomic_json(run / "failures" / f"{index}.json", {"sample_id": row["sample_id"],
                          "stage": "feedback", "attempts": [first, second]})
    return run, rows, first_feedback


def old_runtime_api():
    # Emulate the currently loaded runtime: its parser rejects aliases until the
    # standalone recovery cell canonicalizes them first. No inference API exists.
    def strict_old_parser(text, row):
        t3.validate_labels(json.loads(text)["taxonomy_error"], list(t3.TAXONOMY))
        return t3.parse_response(text, row)

    return SimpleNamespace(**{name: getattr(t3, name) for name in (
        "read_json", "unique_rows", "sample_cache_path", "feedback_flags", "validate_submission", "atomic_json")},
        parse_response=strict_old_parser)


@pytest.mark.parametrize("fenced", [False, True])
def test_recover_322_plus_six_first_attempts_without_overwriting(recovery_run, tmp_path, fenced):
    run, rows, feedback = recovery_run
    if fenced:
        for path in (run / "failures").glob("*.json"):
            failure = t3.read_json(path)
            failure["attempts"][0]["text"] = "```json\n" + failure["attempts"][0]["text"] + "\n```"
            t3.atomic_json(path, failure)
    preserved = {p: p.read_bytes() for folder in ("samples", "failures") for p in (run / folder).glob("*.json")}
    destination = tmp_path / "predictions_P1.json"
    api = old_runtime_api()
    CELL.recover_p1(api, run, destination)
    result = t3.read_json(destination)
    assert len(result) == len({r["sample_id"] for r in result}) == 328
    assert all(p.read_bytes() == content for p, content in preserved.items())
    assert all(r["output"] == {"taxonomy_error": ["Lỗi biên dịch"], "feedback": feedback} for r in result[322:])
    assert all(t3.read_json(t3.sample_cache_path(run, row))["generation"]["attempts"] == 1 for row in rows[322:])
    all_checkpoints = {p: p.read_bytes() for p in (run / "samples").glob("*.json")}
    CELL.recover_p1(api, run, destination)  # idempotent, no checkpoints rewritten
    assert all(p.read_bytes() == content for p, content in all_checkpoints.items())
    assert len(list((run / "failures").glob("*.json"))) == 6


@pytest.mark.parametrize("invalid", ["unknown", "truncated", "internal"])
def test_unrecoverable_first_attempt_never_selects_retry_or_writes(recovery_run, tmp_path, invalid):
    run, rows, _ = recovery_run
    path = sorted((run / "failures").glob("*.json"))[0]
    failure = t3.read_json(path)
    if invalid == "truncated":
        failure["attempts"][0]["ended_with_eos"] = False
    else:
        raw = json.loads(failure["attempts"][0]["text"])
        if invalid == "unknown":
            raw["taxonomy_error"] = ["unconfirmed alias"]
        else:
            raw["feedback"] = "<think>internal</think>"
        failure["attempts"][0]["text"] = json.dumps(raw)
    t3.atomic_json(path, failure)
    destination = tmp_path / "predictions_P1.json"
    with pytest.raises(ValueError):
        CELL.recover_p1(old_runtime_api(), run, destination)
    assert len(list((run / "samples").glob("*.json"))) == 322
    assert not destination.exists()


def test_all_notebooks_have_same_confirmed_configuration():
    for path in (ROOT / "notebooks").glob("Task3_P*.ipynb"):
        config = {}
        for cell in t3.read_json(path)["cells"]:
            source = "".join(cell["source"])
            if source.startswith("CONFIG = "):
                exec(source, {}, config)
        assert config["CONFIG"]["batch_size"] == 1
        assert config["CONFIG"]["seed"] == 42 and config["CONFIG"]["max_new_tokens"] == 512
