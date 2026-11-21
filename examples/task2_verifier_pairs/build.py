"""Build and verify original Task 2 C++ contrast pairs with g++-compatible clang."""

import argparse
import hashlib
import json
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path


# Two integers or one string per case. Each example includes a passing input
# and a failing witness for its mutant. The same driver is used on both versions.
@dataclass(frozen=True)
class Exercise:
    name: str
    requirement: str
    body: str
    edit: tuple[str, str]
    tests: tuple[tuple[str, int], ...]


ARRAY = [
    Exercise(
        "string-hyphens",
        "Count the hyphens in the input word.",
        "int n=0; for (auto c : s) if (c=='-') ++n; return n;",
        ("for (auto c : s)", "for (auto c : s.substr(1))"),
        (("ab-", 1), ("-ab-", 2)),
    ),
    Exercise(
        "prefix-vowels",
        "Count lowercase a characters in the first two positions. The word has length at least two.",
        "return (s[0]=='a')+(s[1]=='a');",
        ("s[1]", "s.back()"),
        (("aba", 1), ("baa", 1), ("aab", 2)),
    ),
    Exercise(
        "middle-marker",
        "Return 1 if the middle character of the odd-length word is x, otherwise 0. Length is at least three.",
        "return s[s.size()/2]=='x';",
        ("s.size()/2", "s.size()/2+1"),
        (("aaa", 0), ("axa", 1)),
    ),
    Exercise(
        "second-field-width",
        "A word consists of two nonempty fields separated by one slash. Return the width of the second field.",
        "return s.size()-s.find('/')-1;",
        ("-1;", ";"),
        (("a/b", 1), ("first/abc", 3)),
    ),
    Exercise(
        "common-endpoints",
        "Return 1 if the first and last characters are equal, otherwise 0. Length is at least two.",
        "return s.front()==s.back();",
        ("s.back()", "s[1]"),
        (("zz", 1), ("aba", 1), ("aab", 0)),
    ),
    Exercise(
        "truncate-command",
        "Return the length after removing up to four characters from a nonempty command word.",
        "return s.substr(std::min<size_t>(4, s.size())).size();",
        ("<size_t>(4,", "<size_t>(3,"),
        (("a", 0), ("firewall", 4)),
    ),
    Exercise(
        "last-colon",
        "The word contains a colon. Return the zero-based position of its last colon.",
        "return s.rfind(':');",
        ("rfind", "find"),
        (("a:b", 1), ("x:y:z", 3), ("z::z", 2)),
    ),
    Exercise(
        "exclude-terminal-flag",
        "Count the uppercase X characters in every position before the final character of the word.",
        "return std::count(s.begin(),s.end()-1,'X');",
        ("s.end()-1", "s.end()"),
        (("AXa", 1), ("XAX", 1), ("X", 0)),
    ),
]

LOGIC = [
    Exercise(
        "night-credit",
        "Return 1 when both readings reach 4, otherwise 0.",
        "return a>=4 && b>=4;",
        ("&&", "||"),
        (("4 4", 1), ("6 2", 0), ("1 1", 0)),
    ),
    Exercise(
        "humidity-limit",
        "Return 1 only when the first reading is strictly higher than the second.",
        "return a>b;",
        ("a>b", "a>=b"),
        (("6 5", 1), ("3 3", 0), ("1 2", 0)),
    ),
    Exercise(
        "ticket-override",
        "Return 1 if a day pass is valid. A pass is valid at age 65 or older unless suspended (b=1).",
        "return a>=65 && b==0;",
        ("b==0", "b==1"),
        (("30 0", 0), ("72 0", 1), ("70 1", 0)),
    ),
    Exercise(
        "battery-delta",
        "Return the net charge change if a units are added and b are consumed.",
        "return a-b;",
        ("a-b", "a+b"),
        (("20 0", 20), ("7 9", -2), ("11 3", 8)),
    ),
    Exercise(
        "outside-interval",
        "Return 1 when reading a is outside the closed interval from 0 to b. The upper bound b is nonnegative.",
        "return a<0 || a>b;",
        ("a>b", "a>=b"),
        (("-2 4", 1), ("5 5", 0), ("8 5", 1)),
    ),
    Exercise(
        "permit-exclusion",
        "Return 1 when exactly one of the two permit flags is set. The flags are integers 0 or 1.",
        "return (a==1)!=(b==1);",
        ("!=", "=="),
        (("1 0", 1), ("1 1", 0), ("0 0", 0)),
    ),
    Exercise(
        "service-charge",
        "Return the bill after applying a fee of b units only when the order subtotal a is below 100.",
        "if (a<100) return a+b; return a;",
        ("a<100", "a>100"),
        (("100 10", 100), ("90 6", 96), ("125 7", 125)),
    ),
    Exercise(
        "wet-weather",
        "The inputs indicate rain and snow, each 0 or 1. Return 1 if either kind of precipitation occurs.",
        "return a==1 || b==1;",
        ("||", "&&"),
        (("0 0", 0), ("0 1", 1), ("1 1", 1)),
    ),
]

FUNCTION = [
    Exercise(
        "crate-change",
        "Use the helper add_stock to increase the caller's stock count by b, then print the updated count.",
        "auto add_stock=[](int& stock,int n){stock+=n;}; add_stock(a,b); return a;",
        ("int& stock", "int stock"),
        (("12 0", 12), ("12 4", 16), ("2 1", 3)),
    ),
    Exercise(
        "mounting-clearance",
        "Return the required clearance. The helper takes the outer diameter first and inner diameter second. Outer is at least inner.",
        "auto clearance=[](int outer,int inner){return outer-inner;}; return clearance(a,b);",
        ("return clearance(a,b)", "return clearance(b,a)"),
        (("4 4", 0), ("11 3", 8)),
    ),
    Exercise(
        "initial-emission",
        "Call the helper to compute twice the input a, then print its returned value. The second integer is ignored.",
        "auto emission=[](int amount){return 2*amount;}; a=emission(a); return a;",
        ("a=emission(a)", "emission(a)"),
        (("0 5", 0), ("8 8", 16)),
    ),
    Exercise(
        "max-through-output",
        "The helper writes the larger input into its output parameter. Print that output.",
        "int result=0; auto best=[](int a,int b,int& out){out=std::max(a,b);}; best(a,b,result); return result;",
        ("int& out", "int out"),
        (("0 0", 0), ("9 3", 9), ("2 6", 6)),
    ),
    Exercise(
        "reserve-floor",
        "Call reserve with an extra capacity of b and print the result. The helper defaults to zero extra capacity.",
        "auto reserve=[](int used,int spare=0){return used+spare;}; return reserve(a,b);",
        ("return reserve(a,b)", "return reserve(a)"),
        (("10 0", 10), ("3 4", 7)),
    ),
    Exercise(
        "new-temperature",
        "Use a helper to subtract offset b from the caller's temperature a and report the updated temperature.",
        "auto adjust=[](int* current,int d){*current-=d;}; adjust(&a,b); return a;",
        ("adjust(&a,b)", "int copy=a; adjust(&copy,b)"),
        (("5 0", 5), ("6 2", 4)),
    ),
    Exercise(
        "indexed-color",
        "For a in 0..9 and b in 1..10, return the code of item a in row b. The helper takes row first, item second. The code is 10 times the row plus the item.",
        "auto encode=[](int row,int item){return 10*row+item;}; return encode(b,a);",
        ("return encode(b,a)", "return encode(a,b)"),
        (("4 4", 44), ("2 8", 82)),
    ),
    Exercise(
        "points-redemption",
        "The helper returns the updated points balance after spending b points. Print that balance.",
        "auto spend=[](int balance,int cost){return balance-cost;}; return spend(a,b);",
        ("return spend(a,b);", "spend(a,b); return a;"),
        (("25 0", 25), ("50 7", 43)),
    ),
]

ALGORITHM = [
    Exercise(
        "paving-square",
        "Find the side of the largest square tile that exactly covers a rectangle of integer sides a and b. Both sides are positive.",
        "while (b) { int r=a%b; a=b; b=r; } return a;",
        ("while (b) { int r=a%b; a=b; b=r; } return a;", "return std::min(a,b);"),
        (("8 4", 4), ("15 6", 3), ("11 7", 1)),
    ),
    Exercise(
        "shared-cycle",
        "Two timers repeat every a and b minutes. Return the first positive minute when both ring. Periods are 1..100.",
        "for(int t=1;;++t) if(t%a==0 && t%b==0) return t;",
        ("for(int t=1;;++t) if(t%a==0 && t%b==0) return t;", "return a*b;"),
        (("3 5", 15), ("6 8", 24), ("7 7", 7)),
    ),
    Exercise(
        "digit-checksum",
        "Return the sum of all decimal digits of a nonnegative integer a. Ignore b.",
        "int sum=0; for(;a>0;a/=10) sum+=a%10; return sum;",
        ("int sum=0; for(;a>0;a/=10) sum+=a%10; return sum;", "return a%10+a/10;"),
        (("27 1", 9), ("423 0", 9), ("9 6", 9)),
    ),
    Exercise(
        "any-factor",
        "Return 1 when integer a is prime, otherwise 0. a is 2..1000. Ignore b.",
        "for(int d=2;d*d<=a;++d) if(a%d==0) return 0; return 1;",
        (
            "for(int d=2;d*d<=a;++d) if(a%d==0) return 0; return 1;",
            "return a==2 || a%2!=0;",
        ),
        (("13 0", 1), ("49 0", 0), ("10 0", 0)),
    ),
    Exercise(
        "overlapping-resources",
        "Two closed intervals have endpoints [0,a] and [b,b+3]. Return the number of integer points covered by their union. Both inputs are nonnegative.",
        "int overlap=std::max(0,std::min(a,b+3)-b+1); return a+1+4-overlap;",
        (
            "int overlap=std::max(0,std::min(a,b+3)-b+1); return a+1+4-overlap;",
            "return a+1+4;",
        ),
        (("2 5", 7), ("5 3", 7), ("0 0", 4)),
    ),
    Exercise(
        "slide-alignment",
        "Return the number of one bits in the binary representation of the nonnegative integer a. Ignore b.",
        "int total=0; for(;a;a>>=1) total+=a&1; return total;",
        ("for(;a;a>>=1) total+=a&1;", "for(;a;a>>=1) ++total;"),
        (("7 0", 3), ("8 0", 1), ("10 2", 2)),
    ),
    Exercise(
        "exchange-machine",
        "Find the fewest coins to make a units using denominations 1, 3, and 4. Amount a is 1..40. Ignore b.",
        "std::vector<int> best(a+1,100); best[0]=0; for(int n=1;n<=a;++n) for(int coin:{1,3,4}) if(coin<=n) best[n]=std::min(best[n],1+best[n-coin]); return best[a];",
        (
            "std::vector<int> best(a+1,100); best[0]=0; for(int n=1;n<=a;++n) for(int coin:{1,3,4}) if(coin<=n) best[n]=std::min(best[n],1+best[n-coin]); return best[a];",
            "int total=0; for(int coin:{4,3,1}) { total+=a/coin; a%=coin; } return total;",
        ),
        (("4 0", 1), ("6 0", 2), ("10 0", 3)),
    ),
    Exercise(
        "shortest-cover",
        "Return the smallest positive integer n such that 1+2+...+n is at least a. a is 1..1000. Ignore b.",
        "int n=0,total=0; while(total<a) total+=++n; return n;",
        ("int n=0,total=0; while(total<a) total+=++n; return n;", "return a;"),
        (("1 0", 1), ("10 0", 4), ("11 0", 5)),
    ),
]

GROUPS = {
    "Lỗi mảng/chuỗi": ARRAY,
    "Lỗi logic": LOGIC,
    "Lỗi hàm": FUNCTION,
    "Lỗi thuật toán": ALGORITHM,
}
# Reserve confirmation first; the two later slices may inform model selection.
SPLITS = {"confirmation": slice(6, 8), "train": slice(0, 4), "val": slice(4, 6)}


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def program(body: str, *, string_input: bool) -> str:
    arguments = "const std::string& s" if string_input else "int a, int b"
    declaration = "std::string s;" if string_input else "int a, b;"
    read = "s" if string_input else "a >> b"
    call = "s" if string_input else "a, b"
    return (
        "#include <algorithm>\n#include <iostream>\n#include <string>\n#include <vector>\n\n"
        f"int solve({arguments}) {{\n    {body}\n}}\n\n"
        f"int main() {{\n    {declaration}\n    if (std::cin >> {read}) {{\n"
        f"        std::cout << solve({call}) << '\\n';\n    }}\n}}\n"
    )


def verify(source: str, tests: tuple[tuple[str, int], ...], folder: Path) -> dict:
    path = folder / "exercise.cpp"
    binary = folder / "exercise"
    path.write_text(source)
    subprocess.run(
        [
            "clang++",
            "-std=c++11",
            "-O0",
            "-fsanitize=undefined",
            "-fno-sanitize-recover=all",
            str(path),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
        timeout=20,
    )
    failures = []
    for query, expected in tests:
        result = subprocess.run(
            [str(binary)],
            input=query + "\n",
            text=True,
            capture_output=True,
            check=True,
            timeout=2,
        )
        if result.stderr:
            raise ValueError(f"Unexpected fixture runtime diagnostic: {result.stderr}")
        if result.stdout != f"{expected}\n":
            failures.append(
                {"input": query, "expected": expected, "actual": result.stdout}
            )
    return {
        "cases": len(tests),
        "mismatches": failures,
        "source_sha256": digest(source.encode()),
    }


def build() -> tuple[dict, dict]:
    splits = {}
    receipts = {}
    with tempfile.TemporaryDirectory() as directory:
        folder = Path(directory)
        for split, selection in SPLITS.items():
            rows = []
            for label, examples in GROUPS.items():
                for item in examples[selection]:
                    old, new = item.edit
                    if item.body.count(old) != 1:
                        raise ValueError(
                            f"{item.name}: edit must affect exactly one location"
                        )
                    family = f"verifier-pairs-v002-{item.name}"
                    for defect, body in [
                        (False, item.body),
                        (True, item.body.replace(old, new)),
                    ]:
                        code = program(body, string_input=label == "Lỗi mảng/chuỗi")
                        sid = family + ("-b" if defect else "-a")
                        evidence = verify(code, item.tests, folder)
                        if bool(evidence["mismatches"]) != defect:
                            raise ValueError(
                                f"{sid}: observed behavior differs from authored role"
                            )
                        receipts[sid] = evidence
                        rows.append(
                            dict(
                                sample_id=sid,
                                family=family,
                                problem_id=family,
                                problem_statement=item.requirement
                                + (
                                    " Input is a nonempty word of at most 100 characters."
                                    if label == "Lỗi mảng/chuỗi"
                                    else " Inputs are two integers between -1000 and 1000."
                                ),
                                code=code,
                                error_labels=[label] if defect else [],
                                verifier_judgments={
                                    label: "present" if defect else "absent"
                                },
                            )
                        )
            splits[split] = rows
    all_rows = [row for rows in splits.values() for row in rows]
    if len(receipts) != len(all_rows) or len(
        {r["source_sha256"] for r in receipts.values()}
    ) != len(all_rows):
        raise ValueError("Duplicate fixture IDs or source")
    return splits, receipts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    splits, receipts = build()
    args.output.mkdir(parents=True)
    payloads = {**splits, "behavior-receipts": receipts}
    for name, payload in payloads.items():
        (args.output / f"{name}.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n"
        )
    print({name: len(rows) for name, rows in splits.items()})


if __name__ == "__main__":
    main()
