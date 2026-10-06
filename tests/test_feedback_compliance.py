"""Pedagogical policy checks on synthetic feedback, without private samples."""

import pytest

from tasks.task3_feedback.compliance import check_compliance


@pytest.mark.parametrize(
    "text", ["Tốt;", "P1 đúng; P2 đúng;", "Em hãy kiểm tra lại bài làm."]
)
def test_prose_punctuation_is_not_code(text):
    assert check_compliance(text, 2)["pass"]


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
