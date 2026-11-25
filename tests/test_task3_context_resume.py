"""Context-only recovery regression tests using synthetic data and CPU doubles."""
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from test_task3_controlled_notebooks import config, fake_model_runtime  # noqa: F401

from llm_grading import task3_controlled as t3

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("p2_context", ROOT / "scripts/resume_task3_p2_context.py")
recovery = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(recovery)


class FullTokenizer:
    model_max_length = 131072

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == {"tokenize": False, "add_generation_prompt": True, "enable_thinking": False}
        return json.dumps(messages, ensure_ascii=False)

    def __call__(self, text, **kwargs):
        assert kwargs == {"add_special_tokens": False}
        return {"input_ids": list(range(len(text)))}


@pytest.mark.parametrize("count", [8175, 7995, 8147, 8116, 7793, 9246, 8070, 8919])
def test_reported_long_prompts_preserved(count):
    class ExactTokenizer(FullTokenizer):
        def apply_chat_template(self, messages, **kwargs):
            return messages[0]["content"]

    messages = [{"role": "user", "content": "x" * count}]
    with pytest.raises(ValueError, match="Context overflow"):
        t3.render_prompt(ExactTokenizer(), messages, 8192, 512)
    assert t3.render_prompt(ExactTokenizer(), messages, 12288, 512) == ("x" * count, count)
    with pytest.raises(ValueError, match="Context overflow"):
        t3.render_prompt(ExactTokenizer(), [{"content": "x" * 11777}], 12288, 512)


@pytest.fixture
def old_run(config, monkeypatch):  # noqa: F811 - imported pytest fixture
    prepared = t3.prepare({**config, "method": "P2", "mode": "dev", "batch_size": 1})
    run = Path(prepared["run_dir"])
    rows = t3.read_json(run / "targets.private.json")
    cfg = {**prepared, "resolved_revision": "1cfa9a7" + "0" * 33}
    t3.atomic_json(run / "worker_config.private.json", cfg)
    startup = {"pytorch_version": "test", "cuda_runtime": "test", "gpu": "test",
               "dtype": "float16", "enable_thinking": False,
               "generation_config": {"do_sample": False, "num_beams": 1, "max_new_tokens": 512}}
    t3.atomic_json(run / "worker_0_startup.json", startup)
    for row in rows[:320]:
        prediction = {"taxonomy_error": [], "feedback": "Bài làm đã được xem xét."}
        record = {"sample_id": row["sample_id"], **prediction, "flags": [],
                  "retrieved_ids": [r["sample_id"] for r in row["examples"]], "worker": 0,
                  "generation": {"text": json.dumps(prediction), "ended_with_eos": True,
                                 "input_tokens": t3.render_prompt(FullTokenizer(), t3.messages_for(row, row["examples"]), 8192, 512)[1],
                                 "output_tokens": 20, "attempts": 1,
                                 "batch_seconds": 1, "batch_size": 1}}
        t3.atomic_json(t3.sample_cache_path(run, row), record)
    for row in rows[320:]:
        t3.atomic_json(run / "failures" / (t3.digest(row["sample_id"] + "feedback") + ".json"),
                       {"sample_id": row["sample_id"], "stage": "feedback", "error": "Context overflow: test"})
    monkeypatch.setattr(recovery, "snapshot_preflight", lambda *_: (FullTokenizer(), startup))
    return run, rows


@pytest.mark.parametrize("batch_size", [1, 2])
def test_migrate_320_resume_only_eight_and_strict_export(old_run, tmp_path, monkeypatch, batch_size):
    source, rows = old_run
    if batch_size == 2:
        manifest = t3.read_json(source / "manifest.json")
        manifest["config"]["batch_size"] = 2
        manifest["contract"]["batch_size"] = 2
        manifest["fingerprint"] = t3.digest({"contract": manifest["contract"], "method": "P2",
            "runtime": Path(t3.__file__).read_text(encoding="utf-8"), "targets": rows})
        t3.atomic_json(source / "manifest.json", manifest)
        worker = t3.read_json(source / "worker_config.private.json")
        worker["batch_size"] = 2
        t3.atomic_json(source / "worker_config.private.json", worker)
    before = {p: p.read_bytes() for p in source.rglob("*") if p.is_file()}
    prepared = recovery.migrate_p2(t3, source, tmp_path / "amended", tmp_path / "comparison")
    run = Path(prepared["run_dir"])
    assert prepared["batch_size"] == batch_size
    assert all(p.read_bytes() == data for p, data in before.items())
    assert len(list((run / "samples").glob("*.json"))) == 320
    provenance = t3.read_json(run / "context_migration.private.json")
    assert provenance["pending_ids"] == [r["sample_id"] for r in rows[320:]]
    for name, sha in provenance["source_file_sha256"].items():
        assert recovery.hashlib.sha256((Path(provenance["backup"]) / name).read_bytes()).hexdigest() == sha
    with pytest.raises(ValueError, match="8 unresolved"):
        t3.export_results(prepared)
    assert not (tmp_path / "amended" / "predictions_P2.json").exists()
    calls = []
    fake_model_runtime(monkeypatch, calls)
    prepared.update(snapshot="fake", resolved_revision="1cfa9a7" + "0" * 33)
    path = run / "worker_config.private.json"
    t3.atomic_json(path, prepared)
    t3.worker(str(path), 0, 2)
    t3.worker(str(path), 1, 2)
    expected_calls = 8 // batch_size
    assert calls == ["feedback"] * expected_calls  # Eight samples, no diagnosis.
    t3.worker(str(path), 0, 1)
    assert len(calls) == expected_calls
    output = t3.export_results(prepared)
    assert len(t3.read_json(output)) == 328
    for row in rows[:320]:
        assert t3.sample_cache_path(run, row).read_bytes() == before[t3.sample_cache_path(source, row)]
    assert all(p.read_bytes() == data for p, data in before.items())
    # Normal fingerprint validation remains enabled on the migrated run.
    t3.prepare(t3.read_json(run / "manifest.json")["config"])
    with pytest.raises(ValueError, match="must be new"):
        recovery.migrate_p2(t3, source, tmp_path / "amended", tmp_path / "comparison")


@pytest.mark.parametrize("corruption", ["fingerprint", "runtime", "raw", "retrieval", "packages", "targets"])
def test_migration_rejects_incompatible_checkpoint(old_run, tmp_path, corruption):
    source, rows = old_run
    if corruption in {"fingerprint", "runtime", "packages", "targets"}:
        path = source / "manifest.json"
        manifest = t3.read_json(path)
        if corruption == "runtime":
            manifest["runtime_sha256"] = "changed"
        elif corruption == "packages":
            manifest["contract"]["packages"]["torch"] = "different"
        elif corruption == "targets":
            targets = t3.read_json(source / "targets.private.json")
            targets[0]["code"] += " changed"
            t3.atomic_json(source / "targets.private.json", targets)
        else:
            manifest["fingerprint"] = "changed"
        t3.atomic_json(path, manifest)
    else:
        path = t3.sample_cache_path(source, rows[0])
        record = t3.read_json(path)
        record["feedback" if corruption == "raw" else "retrieved_ids"] = "changed"
        t3.atomic_json(path, record)
    with pytest.raises(ValueError):
        recovery.migrate_p2(t3, source, tmp_path / "amended", tmp_path / "comparison")
    assert not (tmp_path / "amended").exists()


def test_preflight_uses_native_context_and_exact_snapshot(tmp_path, monkeypatch):
    import torch

    snapshot = tmp_path / ("1cfa9a7" + "0" * 33)
    model = {"model_type": "qwen3", "max_position_embeddings": 40960,
             "num_hidden_layers": 36, "num_key_value_heads": 8, "head_dim": 128}
    t3.atomic_json(snapshot / "config.json", model)
    tokenizer = SimpleNamespace(pad_token_id=0, model_max_length=131072)
    monkeypatch.setitem(sys.modules, "huggingface_hub", SimpleNamespace(snapshot_download=lambda *a, **k: str(snapshot)))
    monkeypatch.setitem(sys.modules, "transformers", SimpleNamespace(
        AutoTokenizer=SimpleNamespace(from_pretrained=lambda *a, **k: tokenizer)))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda _: "T4")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda _: SimpleNamespace(total_memory=16 * 1024 ** 3))
    cfg = {"model": "Qwen/Qwen3-4B", "model_revision": "1cfa9a7", "context_limit": 12288}
    previous = {"resolved_revision": snapshot.name}
    _, evidence = recovery.snapshot_preflight(t3, cfg, previous)
    assert evidence["kv_cache_bytes_at_limit_batch1"] == 1811939328
    for change in ({"context_limit": 40961},):
        with pytest.raises(ValueError, match="native"):
            recovery.snapshot_preflight(t3, {**cfg, **change}, previous)
    with pytest.raises(ValueError, match="snapshot changed"):
        recovery.snapshot_preflight(t3, cfg, {"resolved_revision": "another"})
    t3.atomic_json(snapshot / "config.json", {**model, "use_sliding_window": True})
    with pytest.raises(ValueError, match="full-attention"):
        recovery.snapshot_preflight(t3, cfg, previous)


def test_only_p2_default_context_amended():
    for method in ("P1", "P2", "P3"):
        notebook = next((ROOT / "notebooks").glob(f"Task3_{method}_*.ipynb"))
        cfg = {}
        exec("".join(t3.read_json(notebook)["cells"][4]["source"]), cfg)
        assert cfg["CONFIG"]["context_limit"] == (12288 if method == "P2" else 8192)
        if method == "P2":
            assert cfg["CONFIG"]["mode"] == "dev"
            assert cfg["CONFIG"]["batch_size"] == 1
            assert cfg["CONFIG"]["max_new_tokens"] == 512
            assert cfg["CONFIG"]["output_root"] == "/kaggle/working/p2_context12288_fresh"
            sources = "\n".join("".join(c["source"]) for c in t3.read_json(notebook)["cells"] if c["cell_type"] == "code")
            assert "RESUME_P2_CONTEXT_FROM" not in sources
            assert "migrate_p2" not in sources
            assert "prepared = runtime.prepare(CONFIG)" in sources
