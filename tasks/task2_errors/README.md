# Task 2 — Multi-label Error Classification

Owner boundary: M4, coordinated with shared taxonomy and evaluation utilities.

## Contract

Input is the same problem/code/auxiliary-signal view as Task 1. Output is a set of labels from the single official ten-label taxonomy. An empty set is valid. The input whitelist excludes `feedback`, rubric scores, and reference labels.

## Evaluation target

Primary: macro-F1. Secondary: micro-F1, per-label precision/recall/F1, support, and rare-label analysis.

## Files

- `dataset.py`: leakage-safe task view.
- `pipeline.py`: task interface and orchestration boundary.
- `thresholds.py`: optional probability-to-label helper; generated label lists do not require threshold tuning.
- `prompts/`: immutable prompt versions.

P0/P1/F0 use `prompts/v001.txt` and the shared execution/training paths. Macro-F1 averages all ten taxonomy labels with zero_division=0; confirm lecturer semantics and inspect support. Follow [RUN_EXPERIMENTS.md](../../docs/project/RUN_EXPERIMENTS.md).
