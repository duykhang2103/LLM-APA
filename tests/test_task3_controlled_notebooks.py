"""Synthetic checks for the standalone inference notebooks; no model download."""

import ast
import json
import sys
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from llm_grading import task3_controlled as t3

ROOT = Path(__file__).resolve().parents[1]
POLICY = {"taxonomy_error": list(t3.TAXONOMY),
          "feedback_level": [{"level": f"Level {i}", "definition": str(i)} for i in range(1, 5)]}
FEEDBACK = "Điều kiện dừng trong vòng lặp khiến kết quả không đúng."


def test_oom_fallback_preserves_order_and_full_prompts():
    class OOM(RuntimeError):
        pass

    calls, cleared = [], []
    prompts = ["full source A", "full source B"]

    def generate(batch):
        calls.append(batch)
        if len(batch) > 1:
            raise OOM("test allocation failure")
        return [{"text": batch[0]}]

    results = t3.generate_with_oom_fallback(generate, prompts, lambda: cleared.append(True), OOM)
    assert calls == [prompts, prompts[:1], prompts[1:]]
    assert [r["text"] for r in results] == prompts
    assert sum(r["oom_batch_splits"] for r in results) == 1
    assert cleared == [True]
    with pytest.raises(OOM):
        t3.generate_with_oom_fallback(lambda _: (_ for _ in ()).throw(OOM()), ["one"], lambda: None, OOM)
    with pytest.raises(ValueError):
        t3.generate_with_oom_fallback(lambda _: (_ for _ in ()).throw(ValueError()), prompts, lambda: None, OOM)


def write_dataset(root, n, prefix="dev", references=False):
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for i in range(n):
        relative = f"submissions/{prefix}{i}.cpp"
        (root / relative).parent.mkdir(exist_ok=True)
        (root / relative).write_text(f"int {prefix}{i}() {{ return {i}; }}", encoding="utf-8")
        row = {"sample_id": f"{prefix}{i}", "input": {
            "exam_id": "E", "code_file": relative,
            "target_feedback_level": "Level 2 - Chẩn đoán", "taxonomy_error": ["Lỗi logic"],
        }}
        if references:
            row["output"] = {"feedback": FEEDBACK}
        else:
            # Poisoned target must never make it into inference prompts/checkpoints.
            row["output"] = {"feedback": "TARGET_REFERENCE_MUST_NOT_LEAK"}
        rows.append(row)
    t3.atomic_json(root / "task3_feedback.json", {"n_samples": n, "samples": rows})
    t3.atomic_json(root / "label_space.json", POLICY)
    t3.atomic_json(root / "exams.json", {"exams": [{
        "exam_id": "E", "statement": "Tính tổng theo đề.", "problems": [], "grading_policy": "Giữ nguyên đề.",
    }]})
    return root


@pytest.fixture
def config(tmp_path, monkeypatch):
    monkeypatch.setattr(t3.importlib.metadata, "version", lambda _: "test-version")
    dev = write_dataset(tmp_path / "dev", 328)
    train = write_dataset(tmp_path / "train", 32, "train", True)
    return {"method": "P1", "mode": "smoke", "seed": 42, "batch_size": 2,
            "max_new_tokens": 512, "context_limit": 8192, "model": "Qwen/Qwen3-4B",
            "model_revision": "1cfa9a7", "dev_path": str(dev), "train_path": str(train),
            "confirm_training_partition": True, "validation_fraction": 0.2,
            "output_root": str(tmp_path / "working"), "shared_dir": str(tmp_path / "shared"),
            "labels_path": None, "levels_path": None, "input_root": str(tmp_path)}


def test_target_reference_is_never_loaded(config):
    rows, _ = t3.load_dataset(config["dev_path"])
    assert "TARGET_REFERENCE" not in json.dumps(rows)
    prepared = t3.prepare(config)
    assert "TARGET_REFERENCE" not in (Path(prepared["run_dir"]) / "targets.private.json").read_text(encoding="utf-8")
    row = {**rows[0], "reference": "POISON", "extra": "POISON"}
    assert "POISON" not in json.dumps(t3.messages_for(row))
    assert "taxonomy_error" not in t3.input_view({**row, "labels": ["POISON"]})
    assert all("labels" not in r for r in t3.read_json(Path(prepared["run_dir"]) / "targets.private.json"))


def test_target_labels_discarded_and_level_sidecar_works(config, tmp_path):
    rows, _ = t3.load_dataset(config["dev_path"])
    for row in rows:
        assert "labels" not in row
    t3.apply_sidecars(rows, None, None, POLICY["taxonomy_error"])
    labels = tmp_path / "labels.json"
    levels = tmp_path / "levels.json"
    t3.atomic_json(labels, [{"sample_id": r["sample_id"], "output": {"taxonomy_error": []}} for r in rows])
    t3.atomic_json(levels, [{"sample_id": r["sample_id"], "requested_level": 1} for r in rows])
    with pytest.raises(ValueError, match="forbidden"):
        t3.apply_sidecars(rows, str(labels), str(levels), POLICY["taxonomy_error"])
    t3.apply_sidecars(rows, None, str(levels), POLICY["taxonomy_error"])
    assert all("labels" not in r and r["level"] == 1 for r in rows)
    t3.atomic_json(levels, t3.read_json(levels)[:-1])
    with pytest.raises(ValueError, match="match dataset IDs"):
        t3.apply_sidecars(rows, None, str(levels), POLICY["taxonomy_error"])


def test_source_path_cannot_escape(config):
    source = t3.DatasetSource(config["dev_path"])
    for path in ["../secret.cpp", "/absolute.cpp", "C:\\secret.cpp", "..\\secret.cpp"]:
        with pytest.raises(ValueError, match="inside dataset root"):
            source.text(path)


def test_optional_validation_for_p1_p3_but_no_p2_corpus_split(config):
    cfg = {**config, "mode": "validation"}
    first = t3.prepare(cfg)
    targets = t3.read_json(Path(first["run_dir"]) / "targets.private.json")
    with pytest.raises(ValueError, match="without a validation split"):
        t3.prepare({**cfg, "method": "P2"})
    third = t3.prepare({**cfg, "method": "P3"})
    val_ids = {r["sample_id"] for r in targets}
    assert val_ids == {r["sample_id"] for r in t3.read_json(Path(third["run_dir"]) / "targets.private.json")}
    assert all("reference" not in r and "output" not in r and "labels" not in r for r in targets)


def test_duplicate_code_grouping_and_retrieval(config):
    rows, _ = t3.load_dataset(config["train_path"], references=True)
    copy = {**rows[0], "sample_id": "copy", "code": rows[0]["code"].replace(" ", "\n")}
    ids = t3.validation_groups(rows + [copy], 42, .2)
    assert (rows[0]["sample_id"] in ids) == ("copy" in ids)
    retriever = t3.Retriever(rows + [copy])
    assert len(retriever.pool) == len(rows)
    examples = retriever.retrieve(rows[0])
    assert all(t3.code_hash(e["code"]) != t3.code_hash(rows[0]["code"]) for e in examples)
    assert retriever.retrieve({**rows[0], "level": 4}) == []
    poisoned = {**rows[0], "labels": ["NEVER_USE_TARGET_LABELS"]}
    assert retriever.retrieve(poisoned) == examples


def test_context_overflow_never_truncates():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return messages[0]["content"]

        def __call__(self, text, **kwargs):
            return {"input_ids": list(range(len(text)))}

    with pytest.raises(ValueError, match="No code"):
        t3.render_prompt(Tokenizer(), [{"content": "x" * 100}], 110, 20)
    assert t3.render_prompt(Tokenizer(), [{"content": "x" * 100}], 120, 20)[1] == 100


def test_resume_fingerprint_rejects_changed_settings(config):
    t3.prepare(config)
    t3.prepare(config)
    with pytest.raises(ValueError, match="comparison contract changed"):
        t3.prepare({**config, "max_new_tokens": 100})
    cfg = {**config, "shared_dir": config["shared_dir"] + "new", "max_new_tokens": 100}
    with pytest.raises(ValueError, match="Resume fingerprint mismatch"):
        t3.prepare(cfg)


@pytest.mark.parametrize("text,level", [
    ("```cpp\nint a=1;\n```", 2), ("Hãy sửa x thành 2.", 2),
    ("Lỗi logic ở dòng 4.", 1), ("<think>secret</think>Xin chào", 4),
    ("Lỗi logic | Bằng chứng | Nguyên nhân", 2), ("", 3),
])
def test_unsafe_feedback_is_flagged(text, level):
    assert t3.feedback_flags(text, level)


def fake_model_runtime(monkeypatch, calls, responses=None):
    import torch

    class Batch(dict):
        def to(self, _):
            return self

    class Tokenizer:
        pad_token_id = 0
        eos_token = "EOS"
        eos_token_id = 1

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return cls()

        def apply_chat_template(self, messages, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "diagnosis" if messages[0]["content"] == t3.DIAGNOSIS_SYSTEM else "feedback"

        def __call__(self, text, **kwargs):
            if isinstance(text, str):
                return {"input_ids": [2] * len(text)}
            return Batch(input_ids=torch.tensor([[2 if t == "diagnosis" else 3] for t in text]))

        def decode(self, tokens, **kwargs):
            return "".join(chr(t) for t in tokens if t > 3)

    class Model:
        device = "cuda:0"
        config = SimpleNamespace(max_position_embeddings=32768)
        generation_config = SimpleNamespace(eos_token_id=1, do_sample=True, temperature=.6, top_p=.95, top_k=20)

        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            assert kwargs["torch_dtype"] == torch.float16
            return cls()

        def to(self, _):
            return self

        def eval(self):
            return self

        def _prepare_generation_config(self, generation_config, **kwargs):
            assert kwargs == {"use_model_defaults": False, "do_sample": False}
            return generation_config, {}

        def generate(self, input_ids, generation_config, **kwargs):
            assert kwargs == {"do_sample": False, "use_model_defaults": False}
            assert not generation_config.do_sample
            assert generation_config.temperature is None
            assert generation_config.top_p is None and generation_config.top_k is None
            diagnosis = input_ids[0, 0].item() == 2
            calls.append("diagnosis" if diagnosis else "feedback")
            # Real model outputs are structured; the predicted labels intentionally
            # differ from the poisoned input labels to test self-diagnosis.
            value = {"taxonomy_error": ["Lỗi hàm"], "issues": [{
                "label": "Lỗi hàm", "evidence": "return", "cause": "Sai giá trị trả về."}]} if diagnosis else {
                "taxonomy_error": ["Lỗi hàm"], "feedback": FEEDBACK}
            text = json.dumps(value, ensure_ascii=False)
            if responses:
                text = responses.pop(0)
            return torch.tensor([[r[0].item()] + [ord(c) for c in text] + [1] for r in input_ids])

    for name in ("synchronize", "reset_peak_memory_stats"):
        monkeypatch.setattr(torch.cuda, name, lambda: None)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "CPU test double")
    monkeypatch.setattr(torch.cuda, "max_memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: 1)
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda _: 0)
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=16 * 1024 ** 3))

    class FakeGenerationConfig(SimpleNamespace):
        def to_dict(self):
            return vars(self)

    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoModelForCausalLM=Model, AutoTokenizer=Tokenizer, GenerationConfig=FakeGenerationConfig,
        set_seed=lambda _: None,
    ))


@pytest.mark.parametrize("method", ["P1", "P2", "P3"])
def test_worker_batching_resume_and_safe_export(config, monkeypatch, method):
    prepared = t3.prepare({**config, "method": method})
    prepared.update(snapshot="fake", resolved_revision="1cfa9a7" + "0" * 33)
    run = Path(prepared["run_dir"])
    config_path = run / "worker_config.private.json"
    t3.atomic_json(config_path, prepared)
    calls = []
    fake_model_runtime(monkeypatch, calls)
    t3.worker(str(config_path), 0, 2)
    t3.worker(str(config_path), 1, 2)
    assert calls.count("feedback") == 6  # Each GPU gets 5 items, batched as 2+2+1.
    assert calls.count("diagnosis") == (6 if method == "P3" else 0)
    t3.worker(str(config_path), 0, 1)  # Reassign to one worker: no regeneration.
    assert len(calls) == (12 if method == "P3" else 6)
    output = t3.export_results(prepared)
    assert output.name == f"smoke_{method}.json"
    assert len(t3.read_json(output)) == 10
    assert all(r["output"]["taxonomy_error"] == ["Lỗi hàm"] for r in t3.read_json(output))
    assert "Bằng chứng" not in output.read_text(encoding="utf-8")
    stats = t3.read_json(run / "runtime_stats.json")
    assert stats["generation_calls"] == (20 if method == "P3" else 10)
    # Simulate lost final checkpoint: P3 must reuse the cached diagnostic record.
    if method == "P3":
        row = t3.read_json(run / "targets.private.json")[0]
        t3.sample_cache_path(run, row).unlink()
        t3.worker(str(config_path), 0, 1)
        assert calls.count("diagnosis") == 6
        assert calls.count("feedback") == 7


def test_official_export_exactly_328_and_no_metadata(config):
    prepared = t3.prepare({**config, "mode": "dev"})
    run = Path(prepared["run_dir"])
    rows = t3.read_json(run / "targets.private.json")
    records = [{"sample_id": r["sample_id"], "output": {
        "taxonomy_error": [], "feedback": FEEDBACK}} for r in rows]
    t3.validate_submission(records, rows, POLICY["taxonomy_error"], leaderboard=True)
    for bad in (records[:-1], records + [records[0]], [{**records[0], "metadata": 1}] + records[1:]):
        with pytest.raises(ValueError):
            t3.validate_submission(bad, rows, POLICY["taxonomy_error"], leaderboard=True)


def test_notebook_syntax_and_embedded_runtime_match():
    runtime_source = (ROOT / "src/llm_grading/task3_controlled.py").read_text(encoding="utf-8")
    paths = sorted((ROOT / "notebooks").glob("Task3_P*.ipynb"))
    assert len(paths) == 3
    for path in paths:
        notebook = t3.read_json(path)
        assert notebook["nbformat"] == 4
        scope = {}
        for i, cell in enumerate(notebook["cells"]):
            if cell["cell_type"] == "code":
                source = "".join(cell["source"])
                ast.parse(source, filename=f"{path.name}:{i}")
                assert cell["outputs"] == [] and cell["execution_count"] is None
                tree = ast.parse(source)
                for node in tree.body:
                    if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "RUNTIME_SOURCE" for t in node.targets):
                        scope["runtime"] = ast.literal_eval(node.value)
        assert scope["runtime"] == runtime_source


def test_prepare_never_selects_fixture_as_training(config):
    with pytest.raises(ValueError, match="confirm_training_partition"):
        t3.prepare({**config, "confirm_training_partition": False})
    with pytest.raises(ValueError, match="explicit official train_path"):
        t3.prepare({**config, "method": "P2", "train_path": None})


def test_public_methods_share_contract_without_p1_p3_training(config):
    t3.prepare({**config, "method": "P1", "train_path": None})
    t3.prepare({**config, "method": "P2"})
    t3.prepare({**config, "method": "P3", "train_path": None})


@pytest.mark.parametrize("wrapper", [
    lambda s: s, lambda s: "```json\n" + s + "\n```",
    lambda s: "Result:\n" + s, lambda s: s[:-1] + ",}",
])
def test_json_repair_preserves_prediction(wrapper):
    value = {"taxonomy_error": [], "feedback": "Bài làm đáp ứng yêu cầu đã kiểm tra."}
    assert t3.parse_response(wrapper(json.dumps(value, ensure_ascii=False)), {"level": 1}) == value
    assert t3.parse_response(repr(value), {"level": 1}) == value


@pytest.mark.parametrize("value", [
    {"taxonomy_error": ["Unknown"], "feedback": FEEDBACK},
    {"taxonomy_error": [], "feedback": ""},
    {"taxonomy_error": [], "feedback": FEEDBACK, "internal": "secret"},
])
def test_invalid_model_schema_rejected(value):
    with pytest.raises(ValueError):
        t3.parse_response(json.dumps(value), {"level": 2})


def test_p3_requires_source_evidence_and_label_consistency():
    row = {"level": 2, "code": "int f() {return 1;}"}
    diagnosis = {"taxonomy_error": ["Lỗi hàm"], "issues": [{
        "label": "Lỗi hàm", "evidence": "return 1", "cause": "Sai giá trị trả về."}]}
    assert t3.parse_response(json.dumps(diagnosis), row, stage="diagnosis") == diagnosis
    invalid = {**diagnosis, "issues": [{**diagnosis["issues"][0], "evidence": "invented code"}]}
    with pytest.raises(ValueError, match="source quote"):
        t3.parse_response(json.dumps(invalid), row, stage="diagnosis")
    with pytest.raises(ValueError, match="differ"):
        t3.parse_response(json.dumps({"taxonomy_error": [], "feedback": FEEDBACK}), row, diagnosis=diagnosis)
    assert t3.parse_response('{"taxonomy_error": [], "issues": []}', row, stage="diagnosis")


def test_real_notebook_schema_if_nbformat_available():
    nbformat = pytest.importorskip("nbformat")
    for path in (ROOT / "notebooks").glob("Task3_P*.ipynb"):
        nbformat.validate(nbformat.read(path, as_version=4))


@pytest.mark.parametrize("recover", [True, False])
def test_worker_retries_invalid_json_without_fabricated_fallback(config, monkeypatch, recover):
    prepared = t3.prepare({**config, "train_path": None})
    prepared.update(snapshot="fake", resolved_revision="1cfa9a7" + "0" * 33)
    prepared["failure_retry_passes"] = 0
    run = Path(prepared["run_dir"])
    rows = t3.read_json(run / "targets.private.json")[:1]
    t3.atomic_json(run / "targets.private.json", rows)
    config_path = run / "worker_config.private.json"
    t3.atomic_json(config_path, prepared)
    calls = []
    valid = json.dumps({"taxonomy_error": [], "feedback": FEEDBACK})
    fake_model_runtime(monkeypatch, calls, ["broken JSON", valid if recover else "still broken"])
    if recover:
        t3.worker(str(config_path), 0, 1)
        cached = t3.read_json(t3.sample_cache_path(run, rows[0]))
        assert cached["generation"]["attempts"] == 2
        assert cached["taxonomy_error"] == []
    else:
        t3.worker(str(config_path), 0, 1)
        assert not t3.sample_cache_path(run, rows[0]).exists()
        assert len(list((run / "failures").glob("*.json"))) == 1
    assert calls == ["feedback", "feedback"]


def test_empty_submission_is_preserved(config):
    source = Path(config["dev_path"]) / "submissions/dev0.cpp"
    source.write_text("", encoding="utf-8")
    rows, _ = t3.load_dataset(config["dev_path"])
    assert rows[0]["code"] == ""
    diagnosis = {"taxonomy_error": ["Lỗi thuật toán"], "issues": [{
        "label": "Lỗi thuật toán", "evidence": "", "cause": "Bài nộp rỗng."}]}
    assert t3.parse_response(json.dumps(diagnosis), rows[0], stage="diagnosis") == diagnosis
    with pytest.raises(ValueError, match="missing"):
        t3.parse_response(json.dumps(diagnosis), {**rows[0], "code": "return 0;"}, stage="diagnosis")


def test_taxonomy_and_authorized_training_counts_are_validated(config):
    root = Path(config["dev_path"])
    policy = t3.read_json(root / "label_space.json")
    t3.atomic_json(root / "label_space.json", {**policy, "taxonomy_error": ["other"]})
    with pytest.raises(ValueError, match="ten labels"):
        t3.prepare(config)
    t3.atomic_json(root / "label_space.json", policy)
    path = Path(config["train_path"]) / "task3_feedback.json"
    payload = t3.read_json(path)
    t3.atomic_json(path, {**payload, "n_samples": 31, "samples": payload["samples"][:-1]})
    with pytest.raises(ValueError, match="32 authorized"):
        t3.prepare(config)


def test_authorized_examples_auto_discovery(config):
    discovered = t3.discover_examples(config["input_root"])
    assert Path(discovered).parent == Path(config["train_path"])
    prepared = t3.prepare({**config, "method": "P2", "train_path": "auto_sample_dataset"})
    assert prepared["train_path"] == discovered
    targets = t3.read_json(Path(prepared["run_dir"]) / "targets.private.json")
    assert all(len(row["examples"]) == 2 for row in targets)
    write_dataset(Path(config["input_root"]) / "another-copy", 32, "other", True)
    with pytest.raises(ValueError, match="set train_path explicitly"):
        t3.discover_examples(config["input_root"])


def test_demonstrations_normalize_old_schema_without_changing_source(config):
    path = Path(config["train_path"]) / "task3_feedback.json"
    original = path.read_bytes()
    rows, _ = t3.load_dataset(config["train_path"], references=True)
    demo = t3.normalized_demonstration(rows[0])
    assert set(demo) == {"sample_id", "input", "output"}
    assert "taxonomy_error" not in demo["input"]
    assert demo["output"] == {"taxonomy_error": ["Lỗi logic"], "feedback": FEEDBACK}
    assert demo["input"]["source_code"] == rows[0]["code"]
    assert demo["input"]["exam_id"] == "E"
    assert demo["input"]["problem"]["statement"] == "Tính tổng theo đề."
    assert json.loads(t3.messages_for(rows[1], [rows[0]])[2]["content"]) == demo["output"]
    assert path.read_bytes() == original


@pytest.mark.parametrize("feedback", [
    'Chỉ cần lấy abs(năm mất - năm sinh).',
    'P3 lặp vô tận vì không có cur = cur->next',
    'P3 lặp vô tận (không có head = head->next)',
])
def test_quality_filter_rejects_inline_corrections_at_level2(config, feedback):
    rows, _ = t3.load_dataset(config["train_path"], references=True)
    unsafe = {**rows[0], "reference": feedback}
    assert t3.demonstration_flags(unsafe)
    assert t3.Retriever([unsafe]).retrieve(rows[1]) == []


def test_retrieval_handles_zero_one_and_two_suitable_examples(config):
    rows, _ = t3.load_dataset(config["train_path"], references=True)
    target = rows[0]
    assert t3.Retriever([]).retrieve(target) == []
    assert t3.Retriever(rows[:1]).retrieve(target) == []  # self-match excluded
    assert t3.Retriever(rows[1:2]).retrieve(target) == rows[1:2]
    assert len(t3.Retriever(rows[1:]).retrieve(target)) == 2
    assert t3.Retriever(rows).retrieve({**target, "level": 4}) == []
    assert t3.messages_for(target, []) == t3.messages_for(target)
    # Same exam outranks another exam even if its code is more similar.
    same_exam = {**rows[1], "code": "void f() { while(true) {} }"}
    other_exam = {**rows[2], "problem_id": "OTHER", "code": target["code"] + " // similar"}
    assert t3.Retriever([same_exam, other_exam]).retrieve(target)[0] == same_exam


def test_p2_initial_corpus_keeps_all_32_and_audits_fallback(config):
    sidecar = Path(config["input_root"]) / "levels.json"
    rows, _ = t3.load_dataset(config["dev_path"])
    t3.atomic_json(sidecar, [{"sample_id": r["sample_id"], "requested_level": 4} for r in rows])
    prepared = t3.prepare({**config, "method": "P2", "levels_path": str(sidecar)})
    run = Path(prepared["run_dir"])
    assert len(t3.read_json(run / "normalized_corpus.private.json")) == 32
    audit = t3.read_json(run / "retrieval_audit.private.json")
    assert audit["initial_corpus_count"] == audit["eligible_corpus_count"] == 32
    assert audit["validation_split"] is False and audit["top_k"] == 2
    assert len(audit["zero_shot_fallback_ids"]) == 10
    assert all(r["examples"] == [] for r in t3.read_json(run / "targets.private.json"))
    assert not (Path(config["shared_dir"]) / "validation_split.json").exists()


def test_zip_corpus_normalizes_like_directory_and_auto_discovers(config, tmp_path):
    root = Path(config["train_path"])
    upload = tmp_path / "uploaded"
    upload.mkdir()
    archive = upload / "sample_dataset.zip"
    with zipfile.ZipFile(archive, "w") as z:
        for path in root.rglob("*"):
            if path.is_file():
                z.write(path, "sample_dataset/" + path.relative_to(root).as_posix())
    assert t3.discover_examples(upload) == str(archive)
    zipped, _ = t3.load_dataset(archive, references=True)
    directory, _ = t3.load_dataset(root, references=True)
    assert [t3.normalized_demonstration(r) for r in zipped] == [t3.normalized_demonstration(r) for r in directory]


def test_current_schema_references_and_conflicting_labels(config):
    path = Path(config["train_path"]) / "task3_feedback.json"
    payload = t3.read_json(path)
    for row in payload["samples"]:
        row["output"]["taxonomy_error"] = row["input"].pop("taxonomy_error")
    t3.atomic_json(path, payload)
    rows, _ = t3.load_dataset(config["train_path"], references=True)
    assert len(rows) == 32 and rows[0]["output"]["taxonomy_error"] == ["Lỗi logic"]
    assert t3.discover_examples(config["input_root"]) == str(path)
    payload["samples"][0]["input"]["taxonomy_error"] = []
    t3.atomic_json(path, payload)
    with pytest.raises(ValueError, match="Conflicting"):
        t3.load_dataset(config["train_path"], references=True)


def test_s21_lexical_regression_and_strict_internal_validation():
    # The log contains the sample ID and exception, but not the generated text.
    # Reproduce the overbroad check with an illustrative diagnostic sentence.
    row = {"sample_id": "Class03-Final-1-S21", "level": 2, "code": "int f(){return 0;}"}
    feedback = "Trong hàm f, biến n ảnh hưởng đến giá trị return và kết quả cout."
    assert t3.feedback_flags(feedback, 2)
    prediction = {"taxonomy_error": ["Lỗi logic"], "feedback": feedback}
    assert t3.parse_response(json.dumps(prediction), row) == prediction
    assert t3.parse_response(json.dumps({**prediction, "feedback": "Biến taxonomy_error và requested_level ảnh hưởng hàm return_value."}), row)
    t3.validate_submission([{"sample_id": row["sample_id"], "output": prediction}], [row], list(t3.TAXONOMY), leaderboard=False)
    for bad in ["", "<think>secret</think>", "NHÃN | BẰNG CHỨNG | NGUYÊN NHÂN"]:
        with pytest.raises(ValueError):
            t3.parse_response(json.dumps({**prediction, "feedback": bad}), row)


def test_s21_actual_input_when_public_release_available():
    archive = ROOT / "public_test.zip"
    if not archive.exists():
        pytest.skip("Private public release not attached")
    rows, _ = t3.load_dataset(archive)
    row = next(r for r in rows if r["sample_id"] == "Class03-Final-1-S21")
    assert "labels" not in row
    assert t3.input_view(row)["source_code"] == row["code"]
    text = "Cần kiểm tra biểu thức return trong hàm và biến điều kiện của vòng lặp."
    assert t3.parse_response(json.dumps({"taxonomy_error": ["Lỗi logic"], "feedback": text}), row)


def test_worker_lexical_warning_does_not_trigger_retry_or_block_export(config, monkeypatch):
    prepared = t3.prepare({**config, "train_path": None})
    prepared.update(snapshot="fake", resolved_revision="1cfa9a7" + "0" * 33)
    run = Path(prepared["run_dir"])
    rows = t3.read_json(run / "targets.private.json")[:1]
    rows[0]["sample_id"] = "Class03-Final-1-S21"
    t3.atomic_json(run / "targets.private.json", rows)
    path = run / "worker_config.private.json"
    t3.atomic_json(path, prepared)
    feedback = "Trong hàm f, giá trị return phụ thuộc biến n."
    calls = []
    fake_model_runtime(monkeypatch, calls, [json.dumps({"taxonomy_error": [], "feedback": feedback})])
    t3.worker(str(path), 0, 1)
    assert calls == ["feedback"]
    cached = t3.read_json(t3.sample_cache_path(run, rows[0]))
    assert cached["flags"] == [] and cached["warnings"]
    warning = t3.read_json(next((run / "warnings").glob("*.json")))
    assert warning["sample_id"] == "Class03-Final-1-S21"
    assert warning["requested_level"] == 2 and warning["generated_feedback"] == feedback
    assert t3.read_json(t3.export_results(prepared))[0]["output"]["feedback"] == feedback


@pytest.mark.parametrize("method", ["P1", "P2", "P3"])
def test_worker_alias_normalization_is_logged_without_retry(config, monkeypatch, method):
    prepared = t3.prepare({**config, "method": method})
    prepared.update(snapshot="fake", resolved_revision="1cfa9a7" + "0" * 33)
    run = Path(prepared["run_dir"])
    rows = t3.read_json(run / "targets.private.json")[:1]
    t3.atomic_json(run / "targets.private.json", rows)
    path = run / "worker_config.private.json"
    t3.atomic_json(path, prepared)
    responses = []
    if method == "P3":
        responses.append(json.dumps({"taxonomy_error": ["Lỗi cú pháp"], "issues": [{
            "label": "Lỗi syntax", "evidence": "return", "cause": "Sai cú pháp."}]}))
    responses.append(json.dumps({"taxonomy_error": ["Lỗi syntax", "Lỗi cú pháp"], "feedback": FEEDBACK}))
    calls = []
    fake_model_runtime(monkeypatch, calls, responses)
    t3.worker(str(path), 0, 1)
    assert len(calls) == (2 if method == "P3" else 1)
    cached = t3.read_json(t3.sample_cache_path(run, rows[0]))
    assert cached["taxonomy_error"] == ["Lỗi biên dịch"]
    logs = list((run / "normalization_logs").glob("*.json"))
    assert len(logs) == (2 if method == "P3" else 1)
    assert all(t3.read_json(p)[0]["sample_id"] == rows[0]["sample_id"] for p in logs)


@pytest.mark.parametrize("method", ["P1", "P2", "P3"])
def test_worker_failure_isolation_recovery_pass_and_resume(config, monkeypatch, method):
    prepared = t3.prepare({**config, "method": method})
    prepared.update(snapshot="fake", resolved_revision="1cfa9a7" + "0" * 33)
    run = Path(prepared["run_dir"])
    rows = t3.read_json(run / "targets.private.json")[:4]
    t3.atomic_json(run / "targets.private.json", rows)
    path = run / "worker_config.private.json"
    t3.atomic_json(path, prepared)
    # Permanently invalid first batch; the second batch must still be checkpointed.
    # Repeat malformed output for every call except those processing the good batch.
    calls = []
    if method == "P3":
        diagnosis = json.dumps({"taxonomy_error": ["Lỗi hàm"], "issues": [{
            "label": "Lỗi hàm", "evidence": "return", "cause": "Sai giá trị trả về."}]})
        valid = json.dumps({"taxonomy_error": ["Lỗi hàm"], "feedback": FEEDBACK})
        responses = ["bad", "bad", diagnosis, valid, "bad", "bad"]
    else:
        valid = json.dumps({"taxonomy_error": ["Lỗi hàm"], "feedback": FEEDBACK})
        responses = ["bad", "bad", valid, "bad", "bad"]
    fake_model_runtime(monkeypatch, calls, responses)
    t3.worker(str(path), 0, 1)
    assert not t3.sample_cache_path(run, rows[0]).exists()
    assert all(t3.sample_cache_path(run, r).exists() for r in rows[2:])
    with pytest.raises(ValueError, match="unresolved"):
        t3.export_results(prepared)
    assert not (Path(prepared["output_root"]) / f"smoke_{method}.json").exists()
    assert len(t3.read_json(run / "unresolved_failures.private.json")["missing_ids"]) == 2
    saved = [t3.sample_cache_path(run, r).read_bytes() for r in rows[2:]]
    fake_model_runtime(monkeypatch, calls)
    t3.worker(str(path), 0, 1)
    assert saved == [t3.sample_cache_path(run, r).read_bytes() for r in rows[2:]]
    assert not list((run / "failures").glob("*.json"))
    assert len(t3.read_json(t3.export_results(prepared))) == 4
