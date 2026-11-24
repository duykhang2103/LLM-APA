"""Probe fixed BF16 CUDA operations before loading d3; no casts or CPU offload."""

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from llm_grading.decision_diagnostic import labels_from_probabilities  # noqa: E402
from llm_grading.defined_decisions import defined_questions  # noqa: E402
from llm_grading.release_ledger import (  # noqa: E402
    ReleaseLedger,
    atomic_json,
    fingerprint,
)

MODEL_ID = "vllm-sr/d3-mini"
MODEL_REVISION = "97dd054df4b3e0b3beb6af23ab8c82b7c0052e0e"
save = atomic_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import fcntl

    args.output.mkdir(parents=True, exist_ok=True)
    # Keep the lock file stable across reruns. The OS releases the lock when the process exits.
    with (args.output / ".run.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(
                "Another release run is using this output directory"
            ) from error
        run_release(args.bundle, args.output)


def run_release(bundle: Path, output: Path) -> None:
    import torch
    from transformers import AutoModel

    freeze = json.loads((bundle / "execution-freeze.json").read_text())
    if freeze["model_id"] != MODEL_ID or freeze["model_revision"] != MODEL_REVISION:
        raise ValueError("The frozen model must match the pinned release model")
    data = bundle / "decision-inputs.json"
    if hashlib.sha256(data.read_bytes()).hexdigest() != freeze["inputs_sha256"]:
        raise ValueError("D3 frozen input integrity failed")
    inputs = json.loads(data.read_text())
    if len(inputs) != 328:
        raise ValueError("Only frozen release328 is authorized")
    by_id = {sample["sample_id"]: sample for sample in inputs}
    if len(by_id) != 328:
        raise ValueError("Duplicate release IDs")
    jobs = freeze["jobs"]
    if (
        len(jobs) != 332
        or len({j["job_id"] for j in jobs}) != 332
        or {j["sample_id"] for j in jobs if not j["repeat"]} != set(by_id)
        or len([j for j in jobs if j["repeat"]]) != 4
    ):
        raise ValueError("Invalid frozen job coverage")
    ledger = ReleaseLedger(output / "ledger", freeze, [j["job_id"] for j in jobs])
    rows = ledger.rows
    budget_path = output / "model-budget.json"
    if not budget_path.exists():
        save(
            budget_path,
            {"started_unix": time.time(), "seconds": freeze["wall_seconds"]},
        )
    budget = json.loads(budget_path.read_text())
    if budget["seconds"] != freeze["wall_seconds"]:
        raise ValueError("Recovery budget differs")
    deadline_unix = budget["started_unix"] + budget["seconds"]
    if time.time() >= deadline_unix:
        raise TimeoutError("Original model execution budget exhausted")
    started = time.monotonic()
    receipt = {
        "status": "PREFLIGHT",
        "freeze": freeze,
        "official_submissions": 0,
        "dtype_casts": False,
        "CPU_offload": False,
        "training": False,
    }
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is required; CPU fallback is disabled")
        torch.cuda.set_device(0)
        prop = torch.cuda.get_device_properties(0)
        probe = {
            "device": prop.name,
            "capability": [prop.major, prop.minor],
            "total_bytes": prop.total_memory,
            "native_bf16": torch.cuda.is_bf16_supported(including_emulation=False),
        }
        receipt["bf16_probe"] = probe
        torch.manual_seed(42)
        x = torch.randn(1, 16, 128, device="cuda:0", dtype=torch.bfloat16) * 0.01
        weight = torch.randn(256, 128, device="cuda:0", dtype=torch.bfloat16) * 0.01
        y = torch.nn.functional.linear(x, weight)
        q = torch.randn(1, 2, 16, 32, device="cuda:0", dtype=torch.bfloat16)
        attention = torch.nn.functional.scaled_dot_product_attention(
            q, q, q, is_causal=False
        )
        torch.cuda.synchronize()
        if not torch.isfinite(y).all() or not torch.isfinite(attention).all():
            raise RuntimeError("BF16 CUDA probe produced nonfinite values")
        probe["fixed_bf16_linear_and_sdpa"] = "PASS"
        save(output / "bf16-probe.json", probe)
        del x, weight, y, q, attention
        torch.cuda.empty_cache()
        model = AutoModel.from_pretrained(
            freeze["model_id"],
            revision=freeze["model_revision"],
            trust_remote_code=True,
            dtype=torch.bfloat16,
            device_map="cuda:0",
            batch_size=1,
            verify="full",
        )
        runtime = model.runtime
        if (
            {str(p.device) for p in model.backbone.parameters()} != {"cuda:0"}
            or {p.dtype for p in model.backbone.parameters()} != {torch.bfloat16}
            or str(runtime.readout.device) != "cuda:0"
            or runtime.readout.dtype != torch.float32
        ):
            raise RuntimeError(
                "Decision model/readout placement or dtype differs from contract"
            )
        receipt.update(
            model_class=type(model).__name__,
            model_load_seconds=time.monotonic() - started,
            bf16_probe=probe,
            runtime=runtime.runtime_info(),
            provenance=runtime.provenance(),
        )
        smoke = model.system_one(
            state="int main() { return 0; }",
            questions={
                "probe": {
                    "type": "noul",
                    "instructions": "Is the state C++ source code?",
                }
            },
        )
        if not 0 <= smoke["answers"]["probe"]["noul"] <= 1:
            raise RuntimeError("Custom decision readout smoke failed")
        receipt["runtime_smoke"] = "PASS"
        save(output / "runtime-smoke.json", receipt)
        observed_runtime = {
            k: v for k, v in runtime.runtime_info().items() if k != "loaded_seconds"
        }
        if (
            observed_runtime != freeze["expected_runtime"]
            or runtime.provenance() != freeze["expected_provenance"]
        ):
            raise ValueError("Frozen decision runtime/provenance differs")
        request_questions = defined_questions()
        if fingerprint(request_questions) != freeze["question_hash"]:
            raise ValueError("Frozen question wording differs")
        if len(rows) >= 6:
            gate_path = output / "stress-gate.json"
            if not gate_path.exists():
                raise ValueError(
                    "Completed stress calls lack gate receipt; stop before any new call"
                )
            gate = json.loads(gate_path.read_text())
            if (
                gate["status"] != "PASS"
                or gate["contract_sha256"] != fingerprint(freeze)
                or gate["completed_rows_sha256"] != fingerprint(rows[:6])
            ):
                raise ValueError("Recovered stress gate integrity differs")
        torch.cuda.reset_peak_memory_stats()
        for index, job in enumerate(jobs, 1):
            if index <= len(rows):
                continue
            if index > 6 and not (output / "stress-gate.json").exists():
                raise ValueError("Missing initial stress gate before next model call")
            sample = by_id[job["sample_id"]]
            if time.time() >= deadline_unix:
                raise TimeoutError("D3 bounded process window exceeded")
            if (
                hashlib.sha256(sample["state"].encode()).hexdigest()
                != sample["state_sha256"]
            ):
                raise ValueError("State integrity differs")
            lengths = [
                len(
                    runtime.tokenizer.encode(
                        runtime.text(sample["state"], item), add_special_tokens=False
                    )
                )
                for item in request_questions.values()
            ]
            for key, item in request_questions.items():
                token_ids = runtime.tokenizer.encode(
                    runtime.text(sample["state"], item), add_special_tokens=False
                )
                encoded_ids = json.dumps(
                    token_ids, ensure_ascii=False, sort_keys=True, separators=(",", ":")
                )
                if (
                    hashlib.sha256(encoded_ids.encode()).hexdigest()
                    != freeze["token_ids_hashes"][sample["sample_id"]][key]
                ):
                    raise ValueError("Frozen input token identities differ")
            if lengths != freeze["token_lengths"][sample["sample_id"]]:
                raise ValueError("CPU/GPU token coverage differs")
            if max(lengths) > freeze["max_input_tokens"]:
                raise ValueError("No decision input truncation is permitted")
            ledger.start(job["job_id"])
            torch.cuda.synchronize()
            begin = time.monotonic()
            result = model.system_one(
                state=sample["state"], questions=request_questions
            )
            torch.cuda.synchronize()
            elapsed = time.monotonic() - begin
            answers = result["answers"]
            probabilities = [answers[f"label_{i}"]["noul"] for i in range(10)]
            labels = labels_from_probabilities(probabilities)
            row = {
                "sample_id": sample["sample_id"],
                "repeat": job["repeat"],
                "state_sha256": sample["state_sha256"],
                "source_sha256": sample["source_sha256"],
                "question_sha256": freeze["question_hash"],
                "input_tokens": lengths,
                "probabilities_true": probabilities,
                "labels": labels,
                "seconds": elapsed,
                "answers": answers,
                "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
            }
            ledger.complete(job["job_id"], row)
            if index == 6:
                if (
                    any(r["repeat"] for r in rows)
                    or [r["sample_id"] for r in rows] != freeze["stress_ids"]
                ):
                    raise ValueError("Stress input order differs")
                save(
                    output / "stress-gate.json",
                    {
                        "status": "PASS",
                        "count": 6,
                        "contract_sha256": fingerprint(freeze),
                        "completed_rows_sha256": fingerprint(rows),
                        "seconds": sum(r["seconds"] for r in rows),
                        "gpu_peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                        "gpu_peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                        "all_token_id_checks": "PASS",
                        "finite_decisions": 60,
                        "runtime_and_provenance": "PASS",
                    },
                )
            if index > 6 and not (output / "stress-gate.json").exists():
                raise ValueError("Missing initial stress gate")
            print(
                json.dumps(
                    {
                        "done": index,
                        "count": len(jobs),
                        "repeat": job["repeat"],
                        "seconds": round(elapsed, 3),
                    }
                ),
                flush=True,
            )
        canonical = {r["sample_id"]: r for r in rows if not r["repeat"]}
        stability = []
        for row in rows:
            if row["repeat"]:
                original = canonical[row["sample_id"]]
                stability.append(
                    {
                        "sample_id": row["sample_id"],
                        "labels_identical": row["labels"] == original["labels"],
                        "probabilities_identical": row["probabilities_true"]
                        == original["probabilities_true"],
                        "max_probability_delta": max(
                            abs(a - b)
                            for a, b in zip(
                                row["probabilities_true"],
                                original["probabilities_true"],
                                strict=True,
                            )
                        ),
                    }
                )
        save(output / "stability.json", stability)
        save(output / "decision-rows.json", rows)
        if len(canonical) != 328 or len(stability) != 4:
            raise ValueError("Incomplete final coverage")
        if not all(r["labels_identical"] for r in stability):
            raise ValueError(
                "Repeat label instability; do not select alternative predictions"
            )
        receipt.update(
            status="RELEASE_INFERENCE_COMPLETE",
            request_count=len(rows),
            canonical_count=328,
            repeat_count=4,
            smoke_binary_count=1,
            binary_decision_count=len(rows) * 10,
            inference_seconds=sum(r["seconds"] for r in rows),
            gpu_peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            gpu_peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            rows_sha256=hashlib.sha256(
                (output / "decision-rows.json").read_bytes()
            ).hexdigest(),
        )
    except Exception as error:
        receipt.update(
            status="BLOCKED_RUNTIME"
            if receipt.get("runtime_smoke") != "PASS"
            else "FAILED_RELEASE",
            error_type=type(error).__name__,
            error=str(error)[:1800],
        )
        # Preserve the failure; never cast, offload, change the runtime or rerun automatically.
    receipt["total_seconds"] = time.monotonic() - started
    save(output / "receipt.json", receipt)
    print(
        json.dumps(
            {
                "status": receipt["status"],
                "error_type": receipt.get("error_type"),
                "requests": receipt.get("request_count", 0),
            }
        ),
        flush=True,
    )

    if receipt["status"] != "RELEASE_INFERENCE_COMPLETE":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
