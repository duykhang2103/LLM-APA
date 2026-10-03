"""Canonical location for the official ten-label error taxonomy.

Names and order match the supplied ``label_space.json``. Task pipelines,
metrics, and validators import this module; private data is not needed at
import time. The former LABEL_01 through LABEL_10 placeholders are retired.
"""

ERROR_LABELS = [
    "Lỗi biên dịch",
    "Lỗi nhập/xuất",
    "Lỗi logic",
    "Lỗi vòng lặp",
    "Lỗi mảng/chuỗi",
    "Lỗi hàm",
    "Lỗi edge case",
    "Lỗi thuật toán",
    "Lỗi hard-code",
    "Lỗi style",
]
