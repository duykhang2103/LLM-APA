"""CPU orchestration check: 328 synthetic inputs, six stress calls, four repeats."""

import hashlib
import json
import sys
from types import SimpleNamespace

import pytest

from llm_grading.defined_decisions import defined_questions
from llm_grading.release_ledger import atomic_json, fingerprint
from scripts import run_defined_decisions_release as runner


def test_release_orchestration_has_complete_coverage(tmp_path, monkeypatch):
    class Tensor:
        def __mul__(self, other):
            return self

        def all(self):
            return True

    cuda = SimpleNamespace(
        is_available=lambda: True,
        set_device=lambda _: None,
        get_device_properties=lambda _: SimpleNamespace(
            name="fake", major=7, minor=5, total_memory=16
        ),
        is_bf16_supported=lambda **_: False,
        synchronize=lambda: None,
        empty_cache=lambda: None,
        reset_peak_memory_stats=lambda: None,
        max_memory_allocated=lambda: 10,
        max_memory_reserved=lambda: 12,
    )
    torch = SimpleNamespace(
        cuda=cuda,
        bfloat16="bf16",
        float32="fp32",
        manual_seed=lambda _: None,
        randn=lambda *a, **k: Tensor(),
        isfinite=lambda _: Tensor(),
        nn=SimpleNamespace(
            functional=SimpleNamespace(
                linear=lambda *a: Tensor(),
                scaled_dot_product_attention=lambda *a, **k: Tensor(),
            )
        ),
    )
    runtime = SimpleNamespace(
        readout=SimpleNamespace(device="cuda:0", dtype="fp32"),
        runtime_info=lambda: {},
        provenance=lambda: {},
        text=lambda *a: "text",
        tokenizer=SimpleNamespace(encode=lambda *a, **k: [1, 2]),
    )
    calls = []

    class Model:
        backbone = SimpleNamespace(
            parameters=lambda: iter([SimpleNamespace(device="cuda:0", dtype="bf16")])
        )

        def __init__(self):
            self.runtime = runtime

        def system_one(self, state, questions):
            calls.append(state)
            return {"answers": {key: {"noul": 0.2} for key in questions}}

    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(
        sys.modules,
        "transformers",
        SimpleNamespace(
            AutoModel=SimpleNamespace(from_pretrained=lambda *a, **k: Model())
        ),
    )
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    inputs = [
        {
            "sample_id": str(i),
            "state": str(i),
            "state_sha256": hashlib.sha256(str(i).encode()).hexdigest(),
            "source_sha256": "synthetic",
        }
        for i in range(328)
    ]
    atomic_json(bundle / "decision-inputs.json", inputs)
    jobs = [
        {"job_id": "c" + str(i), "sample_id": str(i), "repeat": False}
        for i in range(328)
    ]
    jobs += [
        {"job_id": "r" + str(i), "sample_id": str(i), "repeat": True} for i in range(4)
    ]
    freeze = {
        "inputs_sha256": hashlib.sha256(
            (bundle / "decision-inputs.json").read_bytes()
        ).hexdigest(),
        "jobs": jobs,
        "model_id": runner.MODEL_ID,
        "model_revision": runner.MODEL_REVISION,
        "wall_seconds": 23400,
        "expected_runtime": {},
        "expected_provenance": {},
        "question_hash": fingerprint(defined_questions()),
        "max_input_tokens": 6400,
        "stress_ids": [str(i) for i in range(6)],
        "token_lengths": {str(i): [2] * 10 for i in range(328)},
        "token_ids_hashes": {
            str(i): {key: fingerprint([1, 2]) for key in defined_questions()}
            for i in range(328)
        },
    }
    atomic_json(bundle / "execution-freeze.json", freeze)
    output = tmp_path / "output"
    monkeypatch.setattr(
        sys, "argv", ["release", "--bundle", str(bundle), "--output", str(output)]
    )
    runner.main()
    assert len(calls) == 333
    assert json.loads((output / "receipt.json").read_text())["canonical_count"] == 328
    assert (
        json.loads((output / "stress-gate.json").read_text())["finite_decisions"] == 60
    )
    assert len(json.loads((output / "stability.json").read_text())) == 4


def test_second_process_cannot_use_same_release_directory(tmp_path, monkeypatch):
    fcntl = pytest.importorskip("fcntl")
    output = tmp_path / "output"
    output.mkdir()
    monkeypatch.setattr(
        sys,
        "argv",
        ["release", "--bundle", str(tmp_path / "bundle"), "--output", str(output)],
    )
    with (output / ".run.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        with pytest.raises(RuntimeError, match="Another release run"):
            runner.main()
    assert list(output.iterdir()) == [output / ".run.lock"]
