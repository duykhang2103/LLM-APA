import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from predict_task2_per_label import answer_probability, decide  # noqa: E402


def test_probability_from_top_logprobs_and_text_fallback():
    choice = {"message": {"content": "No"}, "logprobs": {"content": [{"token": "No", "logprob": math.log(0.6),
              "top_logprobs": [{"token": "No", "logprob": math.log(0.6)}, {"token": " Yes", "logprob": math.log(0.3)}]}]}}
    p, text = answer_probability(choice)
    assert abs(p - 1 / 3) < 1e-9 and decide(p, text, 0.5) is False and decide(p, text, 0.3) is True
    p, text = answer_probability({"message": {"content": "Yes."}})
    assert p is None and decide(p, text, 0.5) is True
