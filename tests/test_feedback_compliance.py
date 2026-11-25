"""Pedagogical policy checks on synthetic feedback, without private samples."""

import pytest

from tasks.task3_feedback.compliance import check_compliance, strip_fix_advice


@pytest.mark.parametrize(
    "text, level", [("Tốt;", 2), ("P1 đúng; P2 đúng;", 2), ("Em hãy kiểm tra lại bài làm.", 3)]
)
def test_prose_punctuation_is_not_code(text, level):
    assert check_compliance(text, level)["pass"]


@pytest.mark.parametrize(
    "text",
    [
        "Hàm prob1 không trả về giá trị. Em cần thêm lệnh return.",
        "P1 sai do khởi tạo sai biến; em kiểm tra lại cách khởi tạo nhé.",
        "Lỗi truy cập mảng. Cần kiểm tra điều kiện biên.",
    ],
)
def test_level_two_rejects_fix_advice(text):
    result = check_compliance(text, 2)
    assert "fix_advice_at_level_2" in result["violations"]


def test_level_two_keeps_cause_wording():
    # "thay vì" describes the cause, not a fix.
    assert check_compliance("P1 sai do lấy ký tự đầu thay vì toàn bộ họ.", 2)["pass"]


def test_level_three_needs_a_fix_direction():
    assert "missing_fix_direction_at_level_3" in check_compliance("P1 sai do vòng lặp dừng sớm.", 3)["violations"]
    assert check_compliance("P1 sai do vòng lặp dừng sớm. Em nên xét lại điều kiện dừng.", 3)["pass"]


def test_strip_fix_advice_keeps_the_diagnosis():
    assert strip_fix_advice("Hàm prob1 không trả về giá trị. Em cần thêm lệnh return.") == (
        "Hàm prob1 không trả về giá trị."
    )
    assert strip_fix_advice("P1 sai do khởi tạo sai biến; em kiểm tra lại cách khởi tạo nhé.") == (
        "P1 sai do khởi tạo sai biến."
    )
    assert strip_fix_advice("P1 sai do vòng lặp dừng sớm.") == "P1 sai do vòng lặp dừng sớm."
    only_advice = "Em hãy xem lại bài."
    assert strip_fix_advice(only_advice) == only_advice


@pytest.mark.parametrize(
    "text",
    [
        "Vòng lặp thiếu bước cập nhật nên không kết thúc.",
        "Em đã comment hết bài nên không có hàm main.",
        "Lỗi ở dòng 9 trong hàm tính mũ.",
    ],
)
def test_level_one_rejects_specific_diagnosis(text):
    assert not check_compliance(text, 1)["pass"]


@pytest.mark.parametrize(
    "text",
    [
        "Chỉ cần thay điều kiện bằng i < n.",
        "Sửa rất dễ, bỏ chữ int ở dòng số 9 là được.",
        "Chỉ cần lấy abs(năm mất - năm sinh).",
        "cur = cur->next;",
    ],
)
def test_level_two_rejects_concrete_fixes(text):
    assert not check_compliance(text, 2)["pass"]


def test_diagnosis_allowed_at_two_and_fix_allowed_at_three():
    assert check_compliance("Vòng lặp thiếu cập nhật con trỏ nên không kết thúc.", 2)[
        "pass"
    ]
    assert check_compliance("Thêm cur = cur->next; vào vòng lặp.", 3)["pass"]
    assert not check_compliance("```cpp\nint main() { return 0; }\n```", 3)["pass"]
    assert check_compliance("```cpp\nint main() { return 0; }\n```", 4)["pass"]


def test_boolean_level_rejected():
    with pytest.raises(ValueError):
        check_compliance("Tốt.", True)
