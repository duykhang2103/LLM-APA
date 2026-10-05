# Preliminary Sample Findings

These are **sample-only observations**, supplied by the team and earlier sample notes, not full-dataset conclusions or a fresh audit in this checkout. Preserve originals; investigate private flagged sample IDs rather than silently correcting annotations.

| Finding | Sample evidence | Implementation/analysis implication | Verify on full data? |
|---|---|---|---|
| Repetitive rubric | Some dimensions are constant or strongly correlated within an exam. | Retain six fields; compare against B0 and inspect per-exam behavior. | Yes |
| Taxonomy imbalance | Rare labels; `Lỗi edge case` has zero sample support. | Fixed ten-label macro-F1, zero_division=0; report support and audit lecturer semantics. | Yes |
| Rubric differs from taxonomy | Scores and error labels encode different information. | Task 2 is not a mandatory bottleneck for Task 1. | Yes |
| Evidence conflicts | Compile/test evidence sometimes disagrees with teacher scores; preprocessing may obscure whitespace errors. | Logs are auxiliary; preserve original reports and flag conflicts. | Yes |
| Possible source mismatch | Some code is hard to reconcile with assigned labels/scores. | Preserve and flag for later investigation, not relabeling. | Yes |
| EX01 ambiguity | Statement weights 1/4/2/3 vs metadata 2.5 each; prerequisite policy lacks rubric mapping. | Carry policy through input; never invent per-question score conversion. | Yes |
| Feedback supervision noise | Levels are imbalanced/confounded with exam identity; some feedback seems too explicit. | Condition on requested level, filter RAG by level, review correctness/compliance separately. | Yes |
| Memorization risk | Few exam contexts and possible near-duplicate templates. | Group exact normalized code for splits; log/restrict retrieval duplicates and compare gains. | Yes |

Preparation writes conflict-review flags and persisted split IDs. Response artifacts retain exam/type/level/hash and retrieval audit fields; metrics include slices and label support. These hooks expose issues without changing teacher targets. More complete test-conflict/source-version review remains manual research work. Detailed evidence stays private.
