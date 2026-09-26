# Experiment registry

Use the naming convention `T{task}-{number}`, for example `T1-001`, `T2-003`, or `T3-005`.

Every official run should eventually record:

- owner and date;
- task, method, model, and exact model revision;
- seed, dataset version, and split version;
- config path and prompt version;
- hardware and inference/training parameters;
- primary and secondary metrics;
- checkpoint location, Git commit, and notes.

Store per-run details under `experiments/runs/<experiment-id>/` and add one summary row to `results.csv`. Large checkpoints stay outside Git and are referenced by path.
