# Synthetic examples

These files are safe teaching fixtures. They are not copied from the course dataset and must not be treated as real training or validation data.

## Files

- `normalized_sample.json`: the common record shape after loading one submission.
- `predictions/task1.json`: structured rubric output; the total is derived from six components.
- `predictions/task2.json`: multi-label output; an empty label list is valid.
- `predictions/task3.json`: Vietnamese feedback output with a requested level.

## How to use these examples

1. Read `normalized_sample.json` before implementing a loader.
2. Identify which fields each task is allowed to read.
3. Compare each prediction file with the output contract in the task README.
4. Use the examples to design future unit tests and schema validation.

Important leakage rule:

```text
Task 1 input: problem + code + permitted evidence; no feedback
Task 2 input: problem + code + permitted evidence; no feedback
Task 3 input: problem + code + known labels + requested level
Task 3 target: reference feedback
```
