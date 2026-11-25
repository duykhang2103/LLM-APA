"""Generate and verify plain-array Task 2 diagnostic cases."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from itertools import product
from pathlib import Path


@dataclass(frozen=True)
class Exercise:
    name: str
    requirement: str
    correct: str
    array_fault: str
    other_fault: str
    oracle: Callable[[tuple[int, ...]], tuple[int, ...]]


EXERCISES = (
    Exercise(
        "pair-swap-snapshot",
        "Swap the two entries in each consecutive pair and print the four values in order.",
        "for (int j=0;j<4;j+=2) std::swap(a[j],a[j+1]); show(a,4);",
        "for (int j=0;j<4;j+=2) { a[j]=a[j+1]; a[j+1]=a[j]; } show(a,4);",
        'for (int j=0;j<4;j+=2) std::swap(a[j],a[j+1]); std::cout<<"result="; show(a,4);',
        lambda a: (a[1], a[0], a[3], a[2]),
    ),
    Exercise(
        "frequency-buckets",
        "Print the counts of -1, 0, and 1, in that order.",
        "int count[3]={}; for (int value:a) ++count[value+1]; show(count,3);",
        "int count[3]={}; for (int value:a) ++count[value==0?0:value+1]; show(count,3);",
        'int count[3]={}; for (int value:a) ++count[value+1]; std::cout<<"counts: "; show(count,3);',
        lambda a: tuple(a.count(value) for value in (-1, 0, 1)),
    ),
    Exercise(
        "crossing-lane-copy",
        "Swap the first half with the second half, keeping order within each half. Print the result.",
        "int out[4]; for (int j=0;j<4;++j) out[j]=a[(j+2)%4]; show(out,4);",
        "int out[4]; for (int j=0;j<4;++j) out[j]=a[(j+1)%4]; show(out,4);",
        'int out[4]; for (int j=0;j<4;++j) out[j]=a[(j+2)%4]; show(out,4); std::cout<<"done\\n";',
        lambda a: a[2:] + a[:2],
    ),
    Exercise(
        "neighbor-snapshot-sums",
        "Print the sums of the three adjacent pairs from left to right, using the original values.",
        "int out[3]; for (int j=0;j<3;++j) out[j]=a[j]+a[j+1]; show(out,3);",
        "for (int j=2;j>=0;--j) a[j]=a[j]+a[j+1]; show(a,3);",
        'int out[3]; for (int j=0;j<3;++j) out[j]=a[j]+a[j+1]; std::cout<<"sums "; show(out,3);',
        lambda a: tuple(x + y for x, y in zip(a, a[1:])),
    ),
    Exercise(
        "matrix-transpose-storage",
        "Treat the four values as a row-major 2 by 2 matrix. Print its transpose in row-major order.",
        "int m[2][2]={{a[0],a[1]},{a[2],a[3]}}; int out[4]; for(int r=0;r<2;++r) for(int c=0;c<2;++c) out[2*r+c]=m[c][r]; show(out,4);",
        "int m[2][2]={{a[0],a[1]},{a[2],a[3]}}; int out[4]; for(int r=0;r<2;++r) for(int c=0;c<2;++c) out[2*r+c]=m[r][c]; show(out,4);",
        "int m[2][2]={{a[0],a[1]},{a[2],a[3]}}; int out[4]; for(int r=0;r<2;++r) for(int c=0;c<2;++c) out[2*r+c]=m[c][r]; for(int value:out) std::cout<<value<<','; std::cout<<'\\n';",
        lambda a: (a[0], a[2], a[1], a[3]),
    ),
    Exercise(
        "remove-last-slot",
        "Print the first three input values, preserving order.",
        "show(a,3);",
        "show(a+1,3);",
        'std::cout<<"first three: "; show(a,3);',
        lambda a: a[:3],
    ),
    Exercise(
        "odd-position-dot-product",
        "Multiply the values in zero-based positions 1 and 3. Print their product.",
        "int out[]={a[1]*a[3]}; show(out,1);",
        "int out[]={a[1]*a[2]}; show(out,1);",
        'int out[]={a[1]*a[3]}; std::cout<<"product="; show(out,1);',
        lambda a: (a[1] * a[3],),
    ),
    Exercise(
        "paired-signs",
        "Return 1 if the endpoint signs agree, including zero. Otherwise return 0. Print only the result.",
        "int out[]={(a[0]==a[3])}; show(out,1);",
        "int out[]={(a[0]==a[2])}; show(out,1);",
        'int out[]={(a[0]==a[3])}; std::cout<<(out[0]? "true" : "false")<<\'\\n\';',
        lambda a: (int(a[0] == a[3]),),
    ),
)


def source(body: str) -> str:
    return (
        "#include <iostream>\n#include <utility>\n"
        "void show(const int* x,int n){ for(int j=0;j<n;++j){if(j) std::cout<<' ';"
        " std::cout<<x[j];} std::cout<<'\\n'; }\n"
        "int main(){ int a[4]; for(int& x:a) if(!(std::cin>>x)) return 0;\n"
        f"    {body}\n"
        "}\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    observations, cases, answers = {}, [], {}
    domain = list(product((-1, 0, 1), repeat=4))
    with tempfile.TemporaryDirectory() as temporary:
        folder = Path(temporary)
        cpp, exe = folder / "case.cpp", folder / "case"
        for exercise in EXERCISES:
            requirement = (
                "Read exactly four integers, each in {-1, 0, 1}. "
                f"{exercise.requirement} Use spaces between numbers and end with a newline."
            )
            # Select fixed opaque case IDs without embedding variant names.
            for kind, body in (
                ("control", exercise.correct),
                ("array", exercise.array_fault),
                ("other", exercise.other_fault),
            ):
                code = source(body)
                case = "arr-v005-" + hashlib.sha256(code.encode()).hexdigest()[:12]
                cpp.write_text(code)
                subprocess.run(
                    [
                        "clang++",
                        "-std=c++11",
                        "-pedantic-errors",
                        "-O0",
                        "-fsanitize=undefined",
                        "-fno-sanitize-recover=all",
                        str(cpp),
                        "-o",
                        str(exe),
                    ],
                    check=True,
                    capture_output=True,
                    timeout=20,
                )
                failures = []
                for values in domain:
                    stdin = " ".join(map(str, values)) + "\n"
                    expected = " ".join(map(str, exercise.oracle(values))) + "\n"
                    got = subprocess.run(
                        [str(exe)],
                        input=stdin,
                        capture_output=True,
                        text=True,
                        check=True,
                        timeout=2,
                    )
                    if got.stderr:
                        raise ValueError(case)
                    if got.stdout != expected:
                        failures.append(
                            {"stdin": stdin, "expected": expected, "actual": got.stdout}
                        )
                if bool(failures) != (kind != "control"):
                    raise ValueError(f"Unexpected behavior for {case}")
                cases.append(
                    dict(
                        sample_id=case,
                        family=exercise.name,
                        problem_statement=requirement,
                        code=code,
                    )
                )
                observations[case] = dict(
                    cases=len(domain),
                    mismatches=len(failures),
                    witness=failures[0] if failures else None,
                    source_sha256=hashlib.sha256(code.encode()).hexdigest(),
                )
                answers[case] = dict(array_string=kind == "array", fixture_role=kind)
    if len(answers) != 3 * len(EXERCISES):
        raise ValueError("Duplicate source cases")
    cases.sort(key=lambda row: row["sample_id"])
    packet = [{**row, "execution": observations[row["sample_id"]]} for row in cases]
    for name, payload in (
        ("inputs.json", cases),
        ("blind-review.json", packet),
        ("author-labels.json", answers),
    ):
        (args.output / name).write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
    print(f"{len(cases)} programs, {len(domain) * len(cases)} executed inputs")


if __name__ == "__main__":
    main()
