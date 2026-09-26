# Task 2 — Multi-label Error Classification

Owner boundary: M4, coordinated with shared taxonomy and evaluation utilities.

## Contract

Input is the same problem/code/auxiliary-signal view as Task 1. Output is a set of labels from the single official ten-label taxonomy. An empty set is valid. The future model input must not contain `feedback`.

## Evaluation target

Primary: macro-F1. Secondary: micro-F1, per-label precision/recall/F1, support, and rare-label analysis.

## Files

- `dataset.py`: leakage-safe task view.
- `pipeline.py`: task interface and orchestration boundary.
- `thresholds.py`: config-driven global/per-label threshold handling.
- `prompts/`: immutable prompt versions.
