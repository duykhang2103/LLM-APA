# Data placement

The challenge data contains anonymized student submissions and remains private to the course group.

Keep the supplied `sample_dataset/` unchanged. That directory is ignored by Git and can be read directly with `configs/task*/sample_heuristic.yaml`. The 32 records demonstrate the format; they are not the official train/dev/test split.

Place the full distributed dataset under `data/raw/` using the same layout:

```text
data/raw/
  exams.json
  label_space.json
  task1_grading.json
  task2_error_taxonomy.json
  task3_feedback.json
  submissions/<exam_id>/<sample_id>.cpp
```

Task documents wrap records in a `samples` list with nested `input` and optional reference `output`. `input.exam_id` joins with `exams.json`; `input.code_file` is relative to the dataset root. The loader resolves source text in memory and rejects paths outside that root. It never rewrites raw files or copies student code into tracked fixtures.

The shared normalized shape retains `problem_id`, `problem_type`, `problem_statement`, `code`, optional evidence, and task targets. Exam `problems` and `grading_policy` remain unchanged. Task 1 maps `output.rubric` and verifies `output.total_score`; Task 2 maps `output.taxonomy_error` to `error_labels`; Task 3 maps its input labels and descriptive `target_feedback_level` to known `error_labels` and an integer level, while keeping reference feedback out of model input. Unlabeled records may omit `output` for inference.

Use `load_samples(root, task="task1")` for a teacher-format directory or pass a task JSON directly. The three task documents share sample IDs and are selected separately, not concatenated. Existing flat normalized files remain supported. Configure each task's `data.input_path` or explicit split paths to the new dataset location; keep `exams.json` and source files alongside any raw task documents.

Use `validation_path` for a fixed validation split; CLI names `val` and `validation` are aliases. Document the grouping strategy and persist IDs before comparing model methods. The sample smoke configs intentionally evaluate all 32 examples without creating a train/validation split.

EX01 has conflicting question weights (statement 1/4/2/3 versus metadata 2.5 each). Preserve both and resolve the grading interpretation with the teacher; the loader does not repair annotations or enforce prerequisite scoring. Treat compile logs as auxiliary evidence even when they disagree with rubric labels.

Storage destinations:

- `data/raw/`: original challenge files; ignored by Git.
- `data/processed/`: normalized records and manifests; ignored by Git.
- `data/splits/`: safe split IDs if the team is permitted to share them.
- `data/cache/`: local preprocessing/retrieval cache; ignored by Git.

Do not commit raw submissions, private test data, manually labeled test examples, or student-identifying information.
