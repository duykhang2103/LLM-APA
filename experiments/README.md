# Experiment registry

Keep existing `T{task}-{number}` IDs. New runs may use task-prefixed method versions: `task1-P0-v1`, `task1-P1-v1`, `task1-F0-v0-smoke`, `task1-F0-v1`, `task1-F0-v2`. Each version answers one hypothesis and uses its own output directory.

Scripts write resolved config, Git revision (or unavailable), dataset/split versions, model/prompt revision, seed, losses/metrics, checkpoint paths, runtime/hardware and notes under `saving.output_dir`. Learned responses and retrieval audits stay private there. Trainer checkpoints support optimizer-state resume; final adapters support inference or deliberate warm starts. Never put keys in config.

Use the existing `results.csv` for aggregate comparison rows; owners can append reviewed results after failure analysis. Do not commit raw student code, response logs, predictions, or checkpoints. See [RUN_EXPERIMENTS.md](../docs/project/RUN_EXPERIMENTS.md) for exact CLI/notebook steps and GPU acceptance limits. SemIf/F1 is optional later.
