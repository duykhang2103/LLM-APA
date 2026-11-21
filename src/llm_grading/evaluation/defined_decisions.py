"""Paired, masked taxonomy metrics for the frozen definition experiment."""

from llm_grading.data.taxonomy import ERROR_LABELS

RARE_INDICES = (0, 3, 4, 5, 6, 8, 9)


def evaluate_defined(cohort: list[dict], predictions: dict, judgments: dict) -> dict:
    samples = {r["sample_id"]: r for r in cohort}
    if len(samples) != len(cohort) or set(judgments) != set(samples):
        raise ValueError("Exact unique source/judgment coverage required")
    if set(predictions) != {"A", "B"}:
        raise ValueError("Two frozen arms required")
    for sid, values in judgments.items():
        if len(values) != 10 or set(values) - {"present", "absent", "uncertain"}:
            raise ValueError("Ten present/absent/uncertain judgments required")
    for arm in predictions.values():
        if set(arm) != set(samples):
            raise ValueError("Exact paired prediction coverage required")
        for labels in arm.values():
            if len(labels) != len(set(labels)) or set(labels) - set(ERROR_LABELS):
                raise ValueError("Invalid taxonomy prediction")
    defects = [
        sid for sid, r in samples.items() if r["functional_defect_expected"] is True
    ]
    controls = [
        sid
        for sid, r in samples.items()
        if r["functional_defect_expected"] is False and not r["policy_only"]
    ]
    policy = [sid for sid, r in samples.items() if r["policy_only"]]
    unknown = [
        sid for sid, r in samples.items() if r["functional_defect_expected"] is None
    ]
    alarms = {
        arm: {sid: bool(set(labels) - {"Lỗi style"}) for sid, labels in values.items()}
        for arm, values in predictions.items()
    }
    metrics = {}
    for arm, values in predictions.items():
        per_class = {}
        for i, label in enumerate(ERROR_LABELS):
            known = [sid for sid in samples if judgments[sid][i] != "uncertain"]
            tp = sum(
                label in values[sid] and judgments[sid][i] == "present" for sid in known
            )
            fp = sum(
                label in values[sid] and judgments[sid][i] == "absent" for sid in known
            )
            fn = sum(
                label not in values[sid] and judgments[sid][i] == "present"
                for sid in known
            )
            per_class[label] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "known": len(known),
                "support": tp + fn,
                "precision": tp / (tp + fp) if tp + fp else 0.0,
                "recall": tp / (tp + fn) if tp + fn else 0.0,
                "f1": (2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else 0.0)
                if known
                else None,
            }
        missed = [sid for sid in defects if not alarms[arm][sid]]
        metrics[arm] = {
            "macro_f1": (
                sum(v["f1"] for v in per_class.values()) / 10
                if all(v["known"] for v in per_class.values())
                else None
            ),
            "per_class": per_class,
            "behavior": {
                "tp": len(defects) - len(missed),
                "fn": len(missed),
                "fp": sum(alarms[arm][sid] for sid in controls),
                "tn": sum(not alarms[arm][sid] for sid in controls),
                "missed": missed,
                "severity3_missed": [
                    sid for sid in missed if samples[sid]["severity"] == 3
                ],
                "weighted_fn": sum(samples[sid]["severity"] for sid in missed),
            },
            "unsupported_rare_assertions": sum(
                per_class[ERROR_LABELS[i]]["fp"] for i in RARE_INDICES
            ),
            "unsupported_all_assertions": sum(v["fp"] for v in per_class.values()),
            "logic_algorithm_loop_multilabel": sum(
                sum(
                    label in labels
                    for label in (ERROR_LABELS[2], ERROR_LABELS[3], ERROR_LABELS[7])
                )
                >= 2
                for labels in values.values()
            ),
        }
    a, b = metrics["A"], metrics["B"]
    criteria = {
        "macro_f1_gain_at_least_003": (
            b["macro_f1"] >= a["macro_f1"] + 0.03
            if a["macro_f1"] is not None and b["macro_f1"] is not None
            else None
        ),
        "at_most_one_additional_behavioral_fn": b["behavior"]["fn"]
        <= a["behavior"]["fn"] + 1,
        "no_additional_severity3_miss": not (
            set(b["behavior"]["severity3_missed"])
            - set(a["behavior"]["severity3_missed"])
        ),
        "unsupported_rare_not_increased": (
            b["unsupported_rare_assertions"] <= a["unsupported_rare_assertions"]
            if all(
                a["per_class"][ERROR_LABELS[i]]["known"]
                > a["per_class"][ERROR_LABELS[i]]["support"]
                for i in RARE_INDICES
            )
            else None
        ),
        "supported_category_recall_loss_at_most_020": all(
            b["per_class"][label]["recall"] >= v["recall"] - 0.20
            for label, v in a["per_class"].items()
            if v["support"] >= 3
        ),
    }
    pairs = []
    for sid in samples:
        aa, bb = set(predictions["A"][sid]), set(predictions["B"][sid])
        pairs.append(
            {
                "sample_id": sid,
                "A": list(predictions["A"][sid]),
                "B": list(predictions["B"][sid]),
                "added": sorted(bb - aa),
                "removed": sorted(aa - bb),
                "A_alarm": alarms["A"][sid],
                "B_alarm": alarms["B"][sid],
            }
        )

    def contingency(ids):
        return {
            f"A{int(x)}_B{int(y)}": sum(
                alarms["A"][sid] == x and alarms["B"][sid] == y for sid in ids
            )
            for x in (True, False)
            for y in (True, False)
        }

    return {
        "metrics": metrics,
        "criteria": criteria,
        "decision": "FAIL"
        if any(value is False for value in criteria.values())
        else "INCONCLUSIVE",
        "criterion_evaluability": {
            key: value is not None for key, value in criteria.items()
        },
        "promotion_limitation": "AI-reviewed operational labels are not lecturer ground truth",
        "contingencies": {
            name: contingency(ids)
            for name, ids in [
                ("defects", defects),
                ("controls", controls),
                ("policy_only", policy),
                ("unknown", unknown),
            ]
        },
        "paired": pairs,
        "category_presence_pairs": {
            label: {
                "both": sum(
                    label in predictions["A"][s] and label in predictions["B"][s]
                    for s in samples
                ),
                "A_only": sum(
                    label in predictions["A"][s] and label not in predictions["B"][s]
                    for s in samples
                ),
                "B_only": sum(
                    label not in predictions["A"][s] and label in predictions["B"][s]
                    for s in samples
                ),
                "neither": sum(
                    label not in predictions["A"][s]
                    and label not in predictions["B"][s]
                    for s in samples
                ),
            }
            for label in ERROR_LABELS
        },
    }
