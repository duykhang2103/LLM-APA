# Chấm điểm và phản hồi bài C++ tự động: hướng Prompting

Nhóm thử hai hướng là fine-tune và prompting. Bài nộp dùng hướng **prompting**, vì hướng này cho điểm cao hơn. Cách tiếp cận và lý do chọn được trình bày trong `approach.pdf`.

| Task | Metric | Điểm public | File |
|---|---|---|---|
| Task 1: chấm rubric | QWK | 0.8865 | `submission/task1.json` |
| Task 2: phân loại lỗi | macro-F1 | 0.3027 | `submission/task2.json` |
| Task 3: phản hồi | LLM judge | 0.4223 | `submission/task3.json` |

## Chạy bằng một lệnh

Trong thư mục `code/`, sau khi cài đặt và đặt `DEEPINFRA_API_KEY` (xem bên dưới):

```sh
python prompting/run_prompting.py --data <thư mục test> --train <thư mục sample_dataset>
```

Ví dụ với dữ liệu giảng viên phát:

```sh
python prompting/run_prompting.py --data D:/Data/public_test_release/public_test_release --train D:/Data/sample_dataset/sample_dataset
```

Lệnh này chạy cả 3 task và ghi 3 file nộp vào `outputs/runs/prompting/final/` (`submission_task1.json`, `submission_task2.json`, `submission_task3.json`).

## Không có weights

Hướng prompting **không huấn luyện mô hình** nên không có file weights. Mô hình là `Qwen/Qwen3.5-9B` (open-weight, giữ nguyên weight gốc), gọi qua API DeepInfra với temperature 0.

Phần được tối ưu là pipeline: prompt, RAG, bằng chứng biên dịch/test và hậu xử lý. Toàn bộ được chốt trong `code/prompting/model/prompting_model.yaml`. Đây là file đóng vai trò "model" của bài nộp.

## Thư mục

```text
code/
  prompting/run_prompting.py          chạy cả pipeline bằng một lệnh
  prompting/model/prompting_model.yaml cấu hình chốt (model, prompt, RAG, hậu xử lý)
  tasks/*/prompts/                    chỉ các prompt đang dùng
  scripts/, src/                      biên dịch, sinh/chạy test, RAG, gọi LLM, hậu xử lý
  outputs/testsuites/                 bộ test do LLM sinh từ đề (không chứa code sinh viên)
submission/                           3 file ứng với điểm cao nhất trên leaderboard
```

## Cài đặt

Python 3.10 hoặc 3.11. Mở terminal trong `code/`:

```sh
pip install -r requirements.txt
pip install -e .
pip install ziglang      # chỉ cần khi máy không có g++
```

API key đặt bằng biến môi trường:

| Biến | Dùng cho | Khi nào cần |
|---|---|---|
| `DEEPINFRA_API_KEY` | Qwen3.5-9B chấm bài | Luôn cần |
| `GROQ_API_KEY` | `openai/gpt-oss-120b` sinh test từ đề | Chỉ khi đề chưa có trong `outputs/testsuites/` (đã có sẵn 6 đề public và 2 đề train) |

## Tham số của lệnh chạy

- `--data`: thư mục có `exams.json` và `submissions/`, ví dụ `public_test_release/public_test_release` hoặc tập private test.
- `--train`: 32 bài có nhãn (`sample_dataset/sample_dataset`). Đây là kho RAG duy nhất.
- Kết quả: `outputs/runs/prompting/final/submission_task{1,2,3}.json`. Script tự kiểm tra định dạng nộp ở bước cuối.
- Xem trước các lệnh sẽ chạy: thêm `--dry-run`.
- Với tập private, thêm `--name private`.

Chạy mới hoàn toàn trên 328 bài mất khoảng 1–1,5 giờ, tốn khoảng $2.5 tiền API. Câu trả lời của LLM qua API ở temperature 0 không bảo đảm giống từng chữ, nên điểm chạy lại có thể lệch nhẹ so với điểm đã nộp.

## Pipeline

1. **Bằng chứng:** biên dịch C++11 từng bài. Chạy bộ test do LLM sinh từ đề bài, không có test viết tay. Bài bị comment toàn bộ được chấm theo code trong comment. Chương trình chạy test được biên dịch giống lệnh `g++ id.cpp --std=c++11` trong đề (không tối ưu).
2. **RAG:** BM25 lấy 1 bài train giống nhất theo code.
3. **Prompt:** Qwen3.5-9B nhận hướng dẫn, ví dụ RAG, đề, code, compile log và test report.
4. **Hậu xử lý bằng code:**
   - Task 1: tổng điểm do code cộng từ 6 chiều.
   - Task 2: ghép 3 cách hỏi (liệt kê nhãn, Yes/No từng nhãn có và không có ví dụ), áp luật theo kết quả biên dịch/test, thêm nhãn edge case.
   - Task 3: kiểm tra vi phạm level và sinh lại. Level 1 dùng câu chuẩn, Level 2 cắt phần gợi ý sửa. Nhãn lỗi đồng bộ với Task 2.

Feedback của giảng viên không được đưa vào input của Task 1/2. RAG chỉ lấy từ dữ liệu train. Lịch sử đầy đủ của các phiên bản prompt cũ nằm trên repo GitHub của nhóm.
