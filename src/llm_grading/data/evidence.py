"""Compile evidence produced by the pipeline itself (allowed by the lecturer).

The released compile_log/test_report fields are empty, so the pipeline compiles
each submission. "strict" is plain C++11; "relaxed" also accepts MSVC habits
the teacher tolerated (void main, missing standard includes). Evidence is an
input signal and a post-processing rule, never a training label.
"""

import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

PRELUDE = "".join(
    f"#include <{h}>\n"
    for h in (
        "cmath", "cstring", "cstdlib", "climits", "string", "iostream",
        "fstream", "sstream", "vector", "algorithm", "iomanip",
    )
) + "using namespace std;\n"
VOID_MAIN = re.compile(r"\bvoid\s+main\s*\(")


def default_compiler():
    """g++ when installed (Kaggle/Linux); otherwise the pip `ziglang` clang."""
    if shutil.which("g++"):
        return ["g++"]
    try:
        import ziglang  # noqa: F401
    except ImportError as error:
        raise RuntimeError("No C++ compiler: install g++ or `pip install ziglang`") from error
    return [sys.executable, "-m", "ziglang", "c++", "-target", "x86_64-linux-gnu"]


def _compile(code, compiler, timeout):
    with tempfile.TemporaryDirectory() as directory:
        source, obj = Path(directory, "submission.cpp"), Path(directory, "submission.o")
        source.write_text(code, encoding="utf-8")
        try:
            done = subprocess.run(
                [*compiler, "-std=c++11", "-w", "-c", "-o", str(obj), str(source)],
                capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            return False, ["compiler timeout"]
    errors = [
        line.replace(str(source), "submission.cpp")
        for line in done.stderr.splitlines()
        if "error" in line.lower() and "In file included" not in line
    ]
    return done.returncode == 0, errors[:5]


def compile_evidence(code, compiler=None, timeout=60):
    compiler = compiler or default_compiler()
    strict, errors = _compile(code, compiler, timeout)
    relaxed = strict
    if not strict:
        relaxed, _ = _compile(PRELUDE + VOID_MAIN.sub("int main(", code), compiler, timeout)
    return {"strict": strict, "relaxed": relaxed, "errors": errors}


def evidence_log(evidence):
    """Short compile_log text placed in the model input."""
    if evidence["strict"]:
        return "Pipeline compile check (C++11): compiled without errors."
    if evidence["relaxed"]:
        return (
            "Pipeline compile check (C++11): only MSVC-style issues (void main / missing "
            "standard include); the teacher treats this as compilable. " + " | ".join(evidence["errors"][:2])
        )
    return "Pipeline compile check (C++11): compile errors: " + " | ".join(evidence["errors"][:3])


def attach_evidence(samples, evidence_by_id):
    """Fill empty compile_log fields; never overwrite a supplied log."""
    for sample in samples:
        evidence = evidence_by_id.get(sample["sample_id"])
        if evidence is None:
            raise ValueError(f"No compile evidence for {sample['sample_id']}")
        sample["compile_evidence"] = evidence
        if not sample.get("compile_log"):
            sample["compile_log"] = evidence_log(evidence)
    return samples
