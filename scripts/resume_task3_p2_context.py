"""Safe, copy-only P2 context migration; embedded in the P2 Kaggle notebook.

No project imports and no model inference. The original run is never written.
"""
import hashlib
import re
import shutil
from pathlib import Path


def require(condition, message):
    if not condition:
        raise ValueError(message)


def snapshot_preflight(runtime, cfg, previous_worker):
    """Inspect the actual pinned snapshot, never tokenizer metadata alone."""
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer

    snapshot = Path(snapshot_download(cfg["model"], revision=cfg["model_revision"],
                                     allow_patterns=["*.json", "*.model", "merges.txt", "vocab.json"]))
    revision = snapshot.name
    require(re.fullmatch(r"[0-9a-f]{40}", revision) and revision.startswith(cfg["model_revision"]),
            "Snapshot is not the pinned immutable revision")
    require(previous_worker["resolved_revision"] == revision, "Model snapshot changed")
    model = runtime.read_json(snapshot / "config.json")
    require(model["model_type"] == "qwen3" and not model.get("rope_scaling")
            and not model.get("use_sliding_window"), "Expected native full-attention Qwen3")
    require(cfg["context_limit"] <= model["max_position_embeddings"],
            "Requested context exceeds native model context")
    require(torch.cuda.is_available(), "Enable Kaggle GPU before migration")
    tokenizer = AutoTokenizer.from_pretrained(snapshot, padding_side="left", local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    evidence = {"resolved_revision": revision, "model_config": model,
                "model_config_sha256": hashlib.sha256((snapshot / "config.json").read_bytes()).hexdigest(),
                "tokenizer_model_max_length": tokenizer.model_max_length,
                "backend": "transformers_sdpa_fp16", "context_limit": cfg["context_limit"],
                "pytorch_version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                "gpu": torch.cuda.get_device_name(0),
                "total_gpu_bytes": torch.cuda.get_device_properties(0).total_memory,
                "kv_cache_bytes_at_limit_batch1": 2 * model["num_hidden_layers"]
                * model["num_key_value_heads"] * model["head_dim"] * cfg["context_limit"] * 2,
                "memory_note": "KV only; weights, activations, logits, SDPA workspace and allocator extra. "
                "T4 SDPA may use math attention; a single long input can still OOM."}
    return tokenizer, evidence


def migrate_p2(runtime, source_run="/kaggle/working/task3_P2_dev",
               output_root="/kaggle/working/p2_context12288",
               shared_dir="/kaggle/working/p2_context12288/comparison",
               context_limit=12288, expected_completed=320):
    """Validate old fingerprint/prompts/results, back up, then copy compatible caches.

    Requires the exact original runtime source. Refuses arbitrary runtime migrations,
    package/model/hardware changes, changed retrieval or incomplete provenance.
    """
    source, destination = Path(source_run).resolve(), Path(output_root).resolve()
    shared = Path(shared_dir).resolve()
    require(not destination.exists(), "Migration output_root must be new; resume an existing migration normally")
    require(source != destination and source not in destination.parents
            and destination not in source.parents, "Source and destination must be separate trees")
    require(shared != source and source not in shared.parents and shared not in source.parents,
            "shared_dir must not overlap the original checkpoint")
    require(not shared.exists(), "Use a fresh shared_dir for the amended context contract")
    old = runtime.read_json(source / "manifest.json")
    targets = runtime.read_json(source / "targets.private.json")
    previous_worker = runtime.read_json(source / "worker_config.private.json")
    old_cfg = old["config"]
    require(old_cfg["method"] == "P2" and old_cfg["mode"] == "dev", "Only P2 dev migration supported")
    require(type(old_cfg["batch_size"]) is int and old_cfg["batch_size"] > 0
            and old_cfg["max_new_tokens"] == 512 and old_cfg["context_limit"] == 8192,
            "Expected positive original batch size, output=512, context=8192")
    require(type(context_limit) is int and context_limit > 8192, "Context must increase explicitly")
    runtime_path = Path(runtime.__file__)
    runtime_source = runtime_path.read_text(encoding="utf-8")
    require(old["runtime_sha256"] == hashlib.sha256(runtime_path.read_bytes()).hexdigest(),
            "Runtime source changed; restore the exact source used by the old run")
    require(old["fingerprint"] == runtime.digest({"contract": old["contract"], "method": "P2",
                                                "runtime": runtime_source, "targets": targets}),
            "Original fingerprint does not match runtime/targets/contract")
    require(len(runtime.unique_rows(targets)) == 328, "Expected 328 unique targets")
    for key in ("model", "model_revision", "seed", "batch_size", "context_limit", "max_new_tokens",
                "packages", "json_retries", "failure_retry_passes"):
        require(previous_worker.get(key, 1) == old_cfg.get(key, 1), f"Worker config changed: {key}")
    cfg = {**old_cfg, "context_limit": context_limit, "output_root": str(destination),
           "shared_dir": str(shared)}
    tokenizer, evidence = snapshot_preflight(runtime, cfg, previous_worker)
    startups = list(source.glob("worker_*_startup.json"))
    require(bool(startups), "Missing original worker startup provenance")
    for path in startups:
        startup = runtime.read_json(path)
        for key in ("pytorch_version", "cuda_runtime", "gpu"):
            require(startup[key] == evidence[key], f"Inference environment changed: {key}")
        generation = startup["generation_config"]
        require(startup["dtype"] == "float16" and startup["enable_thinking"] is False
                and generation["do_sample"] is False and generation["num_beams"] == 1
                and generation["max_new_tokens"] == 512, "Original decoding configuration differs")
    completed, missing, prompt_audit = [], [], []
    for row in targets:
        messages = runtime.messages_for(row, row["examples"])
        prompt, count = runtime.render_prompt(tokenizer, messages, context_limit, 512)
        prompt_audit.append({"sample_id": row["sample_id"], "input_tokens": count,
                             "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest()})
        path = runtime.sample_cache_path(source, row)
        if not path.exists():
            missing.append(row["sample_id"])
            failure = runtime.read_json(source / "failures" / (runtime.digest(row["sample_id"] + "feedback") + ".json"))
            require(failure["sample_id"] == row["sample_id"] and failure["stage"] == "feedback"
                    and "Context overflow:" in failure["error"], "Unresolved sample is not a feedback overflow")
            continue
        require(count + 512 <= 8192, "Completed sample did not fit the original context")
        result = runtime.read_json(path)
        generation = result["generation"]
        parsed = runtime.parse_response(generation["text"], row)
        require(result["sample_id"] == row["sample_id"] and not result["flags"]
                and generation["ended_with_eos"] is True, "Invalid/truncated cached result")
        attempts = generation.get("attempts", 1)
        require(type(attempts) is int and 1 <= attempts <= old_cfg.get("json_retries", 1) + 1
                and generation["input_tokens"] >= count
                and 0 < generation["output_tokens"] <= 512 * attempts,
                "Cached token accounting exceeds the original generation budget")
        require(parsed == {"taxonomy_error": result["taxonomy_error"], "feedback": result["feedback"]},
                "Cached parsed result differs from raw generation")
        require(result["retrieved_ids"] == [e["sample_id"] for e in row["examples"]],
                "Cached retrieval differs from target examples")
        completed.append(row["sample_id"])
    require(len(completed) == expected_completed, f"Expected {expected_completed} completed; found {len(completed)}")
    # Validate package versions/data/retrieval again through unchanged prepare().
    # New directories only; source remains read-only even if a later check fails.
    prepared = runtime.prepare(cfg)
    run = Path(prepared["run_dir"])
    new = runtime.read_json(run / "manifest.json")
    expected_contract = {**old["contract"], "context_limit": context_limit}
    require(new["contract"] == expected_contract, "Contract changed beyond context_limit")
    require(runtime.read_json(run / "targets.private.json") == targets,
            "Inputs, retrieval examples or their order changed")
    for name in ("normalized_corpus.private.json", "retrieval_audit.private.json"):
        require(runtime.read_json(run / name) == runtime.read_json(source / name), f"Corpus audit changed: {name}")
    backup = destination / "original_checkpoint_backup"
    shutil.copytree(source, backup)
    hashes = {str(p.relative_to(source)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in source.rglob("*") if p.is_file()}
    require(all(hashlib.sha256((backup / name).read_bytes()).hexdigest() == sha for name, sha in hashes.items()),
            "Backup hash mismatch; stop concurrent workers before migration")
    # Preserve failure logs/history in backup only. Active run gets exact valid caches.
    for row in targets:
        if row["sample_id"] in completed:
            path = runtime.sample_cache_path(run, row)
            path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(runtime.sample_cache_path(source, row), path)
    for folder in ("warnings", "normalization_logs", "sessions"):
        if (source / folder).exists():
            shutil.copytree(source / folder, run / folder)
    runtime.atomic_json(run / "context_migration.private.json", {
        "version": "p2_context_only_v001", "source_run": str(source), "backup": str(backup),
        "old_fingerprint": old["fingerprint"], "new_fingerprint": new["fingerprint"],
        "source_file_sha256": hashes, "reused_ids": completed, "pending_ids": missing,
        "snapshot_preflight": evidence, "prompts": prompt_audit,
        "migration_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "compatibility": "Exact runtime, targets, retrieval, packages, snapshot, decoding and GPU type; "
        "only admission context increased. Existing prompts/generations unchanged. "
        "No bitwise claim for fresh generation across different GPU sessions."})
    print(f"Migrated {len(completed)}/328 unchanged caches; pending {len(missing)}: {missing}")
    return prepared
