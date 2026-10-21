"""Code-keyed BM25/hybrid retrieval and same-exam preference."""

import numpy as np
import pytest
from test_experiment_contracts import sample

from llm_grading.retrieval.index import BM25Index, build_index, code_tokens
from llm_grading.retrieval.retriever import Retriever, retrieval_text


class Encoder:
    def encode(self, texts, normalize_embeddings=True):
        return np.array([[1.0, float(len(t) % 7)] for t in texts])


def rows():
    loop = sample("loop", "int f(int n){ for(int i=0;i<n;i++) s+=i; return s; }")
    loop2 = sample("loop2", "int g(int m){ for(int j=0;j<m;j++) t+=j; return t; }")
    other = sample("other", "string h(string a){ return a.substr(0,1); }")
    loop["problem_id"], loop2["problem_id"], other["problem_id"] = "E1", "E2", "E1"
    return [loop, loop2, other]


def test_code_key_ignores_comments_and_statement():
    s = sample("x", "int f(){ return 1; } // ghi chu\n/* block */")
    text = retrieval_text(s, "task1", "code")
    assert "ghi chu" not in text and "block" not in text and "return 1" in text
    assert s["problem_statement"] not in text


def test_bm25_prefers_shared_code_tokens():
    index = BM25Index(["for ( int i = 0 ; i < n ; i ++ )", "return a . substr ( 0 , 1 )"], {})
    scores = index.scores("for ( int j = 0 ; j < m ; j ++ )")
    assert scores[0] > scores[1]
    assert code_tokens("a->b<<c") == ["a", "->", "b", "<<", "c"]


def test_bm25_retriever_and_same_exam_preference(tmp_path):
    query = sample("q", "int k(int n){ for(int i=0;i<n;i++) s+=i; return s; }")
    query["problem_id"] = "E1"
    cfg = {"task": "task1", "retrieval": {"k": 1, "key": "code", "index": "bm25"}}
    _, audit = Retriever(rows(), "task1", cfg).retrieve(query)
    assert audit["retrieved"][0]["sample_id"] in {"loop", "loop2"}
    query["problem_id"] = "E2"
    cfg["retrieval"]["same_exam_first"] = True
    _, audit = Retriever(rows(), "task1", cfg).retrieve(query)
    assert audit["retrieved"][0] == {
        "sample_id": "loop2",
        "score": audit["retrieved"][0]["score"],
        "same_exam": True,
    }


def test_compact_examples_drop_statement_only():
    cfg = {"retrieval": {"k": 1, "key": "code", "index": "bm25"}}
    examples, _ = Retriever(rows(), "task1", cfg).retrieve(sample("q", "int f(){}"))
    assert "problem_statement" in examples[0]["input"]
    cfg["retrieval"]["compact_examples"] = True
    examples, _ = Retriever(rows(), "task1", cfg).retrieve(sample("q", "int f(){}"))
    assert "problem_statement" not in examples[0]["input"]
    assert "code" in examples[0]["input"] and "SECRET" not in str(examples)


def test_v003_prompts_render_with_literal_json_braces():
    from llm_grading.prompting.formatter import render_task_prompt

    for task in ("task1", "task2", "task3"):
        prompt = render_task_prompt(sample(), task, {"prompt": {"version": "v003"}})
        assert "{examples}" not in prompt and "{input_json}" not in prompt
    assert '{"compilable": 0-1' in render_task_prompt(
        sample(), "task1", {"prompt": {"version": "v003"}}
    )


def test_fast_near_duplicate_matches_full_ratio():
    from difflib import SequenceMatcher

    from llm_grading.retrieval.retriever import is_near_duplicate

    codes = [
        "int f(){return 1;}",
        "int f(){return 2;}",
        "int f(){ return 1; }",
        "string g(string a){return a;}",
        "",
    ]
    for a in codes:
        for b in codes:
            for cutoff in (0.5, 0.9, 0.95, 1.0):
                full = SequenceMatcher(None, a, b, autojunk=False).ratio() >= cutoff
                assert is_near_duplicate(a, b, cutoff) == full


def test_hybrid_and_invalid_settings():
    index = build_index(["a b", "c d"], {"index": "hybrid"}, Encoder())
    assert index.scores("a b").shape == (2,)
    assert index.metadata["index"] == "hybrid"
    with pytest.raises(ValueError):
        build_index(["a"], {"index": "faiss"})
    with pytest.raises(ValueError):
        Retriever(rows(), "task1", {"retrieval": {"key": "statement"}}, Encoder())


def test_exclude_same_exam_for_leave_one_exam_out():
    query = sample("q", "int k(int n){ for(int i=0;i<n;i++) s+=i; return s; }")
    query["problem_id"] = "E1"
    cfg = {"task": "task1", "retrieval": {"k": 3, "key": "code", "index": "bm25", "exclude_same_exam": True}}
    _, audit = Retriever(rows(), "task1", cfg).retrieve(query)
    assert [r["sample_id"] for r in audit["retrieved"]] == ["loop2"]


def test_v004_prompts_render_and_keep_schema():
    from llm_grading.prompting.formatter import render_task_prompt

    for task in ("task1", "task2"):
        prompt = render_task_prompt(sample(), task, {"prompt": {"version": "v004"}})
        assert "{examples}" not in prompt and "Pipeline-generated tests" in prompt
    assert '{"compilable": 0-1' in render_task_prompt(sample(), "task1", {"prompt": {"version": "v004"}})
    assert '{"error_labels": [ ... ]}' in render_task_prompt(sample(), "task2", {"prompt": {"version": "v004"}})
