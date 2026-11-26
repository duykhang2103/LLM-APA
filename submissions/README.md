# File đã nộp leaderboard (kết quả tốt nhất)

| File | Task | Metric | Điểm public | Tạo bởi |
|---|---|---|---|---|
| [`submission_task1.json`](submission_task1.json) | Task 1: chấm rubric | QWK | 0.8817 | hướng Prompting |
| [`submission_task2.json`](submission_task2.json) | Task 2: phân loại lỗi | macro-F1 | 0.3027 | hướng Prompting |
| [`submission_task3.json`](submission_task3.json) | Task 3: phản hồi | LLM judge | 0.4223 | hướng Prompting |

Cả 3 file đều là output trên **tập public test** (328 bài), theo đúng định dạng nộp của giảng viên.

Tạo lại các file này bằng một lệnh, kết quả ghi vào `outputs/runs/prompting/final/`:

```sh
python prompting/run_prompting.py --data <thư mục public_test_release>
```

Cài đặt và API key: xem [prompting/README.md](../prompting/README.md). Chạy lại trên máy khác có thể cho điểm lệch nhẹ, vì API LLM không bảo đảm trả lời giống từng chữ.
