"""Source-bound decision states and ten independent binary taxonomy questions."""

import hashlib

from llm_grading.data.taxonomy import ERROR_LABELS


def find_build_evidence(sample: dict, receipts: list[dict]) -> dict:
    """Receipts use execution cache keys; admit by BOTH content bindings instead."""
    admitted = [trusted_build_evidence(sample, receipt) for receipt in receipts]
    available = [r for r in admitted if r["status"] != "unavailable"]
    # A link observation covers more of a full-program build than syntax alone.
    available.sort(key=lambda r: r.get("phase") == "link", reverse=True)
    return available[0] if available else trusted_build_evidence(sample, None)


def trusted_build_evidence(sample: dict, receipt: dict | None) -> dict:
    unavailable = {"status": "unavailable", "runtime_observations": "none"}
    if not receipt:
        return unavailable
    if (
        receipt.get("source_sha256")
        != hashlib.sha256(sample["code"].encode()).hexdigest()
        or receipt.get("statement_sha256")
        != hashlib.sha256(sample["problem_statement"].encode()).hexdigest()
        or receipt.get("reproducible") is not True
        or receipt.get("standard") != "c++11"
        or receipt.get("phase") not in {"syntax_check", "link"}
        or receipt.get("status") not in {"compile_success", "compile_failure"}
        or receipt.get("diagnostics_truncated") is True
    ):
        return unavailable
    diagnostics = [
        {k: d.get(k) for k in ("kind", "message", "line", "column")}
        for d in receipt.get("diagnostics", [])
        if d.get("kind") == "error"
        and isinstance(d.get("message"), str)
        and type(d.get("line")) is int
        and 1 <= d["line"] <= len(sample["code"].splitlines())
    ][:1]
    if receipt["status"] == "compile_failure" and not diagnostics:
        return unavailable
    return {
        "status": receipt["status"],
        "phase": receipt["phase"],
        "standard": "c++11",
        "scope": "local build only; official environment equivalence unverified; no execution claim",
        "source_sha256": receipt["source_sha256"],
        "diagnostics": diagnostics,
        "runtime_observations": "none",
    }


def state_view(sample: dict, receipt: dict | None) -> dict:
    return {
        "problem_statement": sample["problem_statement"],
        "code": sample["code"],
        "compiler_evidence": trusted_build_evidence(sample, receipt),
    }


def questions(order: str = "noul") -> dict:
    result = {}
    for index, label in enumerate(ERROR_LABELS):
        item = {
            "type": "noul",
            "instructions": (
                f"Does this C++ submission contain the error category '{label}'? "
                "Judge this category independently; several categories or none may apply. "
                "Use only the source, problem and admitted build observations. "
                "Build success does not prove semantic correctness or official environment equivalence. "
                "Do not infer unavailable executions or teacher policies."
            ),
        }
        if order in {"normal", "reversed"}:
            item["type"] = "choice"
            pairs = [("false", "No / false"), ("true", "Yes / true")]
            item["criteria"] = dict(pairs if order == "normal" else reversed(pairs))
        elif order != "noul":
            raise ValueError("Unknown option order")
        result[f"label_{index}"] = item
    return result


def labels_from_probabilities(probabilities: list[float]) -> list[str]:
    if len(probabilities) != len(ERROR_LABELS) or any(
        type(p) not in (int, float) or not 0 <= p <= 1 for p in probabilities
    ):
        raise ValueError("One finite true-option probability per category is required")
    return [
        label for label, p in zip(ERROR_LABELS, probabilities, strict=True) if p > 0.5
    ]
