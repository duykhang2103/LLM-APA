# Task 3 v3 review1: Qwen3-4B-Instruct-2507 trên T4 x2

Import `notebooks/run_task3_kaggle_v3.ipynb`. Bật Internet, chọn GPU T4 x2. Không cần API key. Đây là prompting/RAG trên model gốc, không phải fine-tune.

1. Tạo ZIP mã nguồn bằng `python scripts/package_kaggle_source.py`. Upload `outputs/task3_kaggle_source.zip` mới vào dataset riêng tư trên Kaggle. Không giữ hai bộ source trong Input.
2. Add Input đủ sample_dataset (32 mẫu, code và exams.json), public test và requested levels.
3. Chạy setup; notebook nhận source dạng thư mục giải nén hoặc ZIP. Phải có cấu hình `configs/task3/kaggle_v3.yaml`, prompt v007, diagnosis_v003, diagnosis_review_v001 và pipeline mới.
4. Chạy GPU check, data check và stress test attention trên cả hai card trước khi nạp trọng số.
5. Model config tải Qwen3-4B-Instruct-2507 một lần và ghim revision. Mỗi GPU chạy một bản NF4/FP16, batch 1.
6. Smoke 3 bài, sau đó kiểm tra 8 mẫu train có nhãn, trải đều level. Lượt kiểm tra bỏ nhãn và feedback tham chiếu khỏi query; RAG loại chính mẫu, code trùng/gần trùng và toàn bộ mẫu cùng đề. Đọc phản hồi dự đoán cạnh nhận xét giảng viên. `validation_report.json` ghi precision/recall/F1 của taxonomy và exact match; đây là tập chẩn đoán nhỏ, không phải thước đo độ chính xác ngữ nghĩa của feedback hoặc ước lượng điểm leaderboard.
7. Chạy hai shard rồi ghép, kiểm tra đủ sample_id/level/compliance và không có fallback trước khi xuất JSON nộp bài.

Đỉnh bộ nhớ attention được giảm bằng chia query thành khối 128 và dùng SDPA MATH; không cần kernel Flash/Efficient hỗ trợ GQA. Causal mask được tính theo vị trí gốc, không reset theo mỗi khối. Không chia hoặc cắt code sinh viên.

Bước chẩn đoán dùng diagnosis_v003, đọc đầy đủ đề/code trong ngân sách 8192 token. Bước kiểm tra lại đọc cùng đề/code và bản nháp, loại nhận định không có bằng chứng hoặc sử dụng đầu vào ngoài ràng buộc. Nhãn và chẩn đoán sau kiểm tra thay thế hoàn toàn bản nháp. Đây vẫn là model tự kiểm tra, không chứng minh chẩn đoán đúng; chi phí inference tăng do thêm một lượt đọc code.

Bước feedback dùng prompt v007, chỉ nhận chẩn đoán, nhãn, level và nhận xét train cùng level. Chẩn đoán có cả hướng sửa bằng lời để Level 3 không chỉ mô tả lỗi. Level 1 vẫn chẩn đoán taxonomy, sau đó dùng câu đánh giá chung cố định để tránh lộ vị trí/nguyên nhân. Khi không xác nhận lỗi, không khẳng định bài đã qua mọi test. Level 3 có lỗi phải vượt kiểm tra từ khóa hướng sửa; kiểm tra này không đánh giá được tính đúng của hướng sửa.

Output: `/kaggle/working/task3-qwen3-4b-t4-v3-review1`. File nộp trực tiếp: `/kaggle/working/submission_task3_qwen3_4b_review1.json`, đúng dạng JSON được giao diện yêu cầu. Source upload Kaggle vẫn là ZIP. Không tái sử dụng checkpoint dự đoán v3 cũ cho cấu hình mới. Revision model được ghi vào `model_revision.json` và tái sử dụng khi resume.

`--resume` chỉ bỏ qua bài đã hoàn tất nếu config/dữ liệu/prompt khớp. Lưu riêng tư output và cache trước khi tắt session. Cấu hình mới tắt fallback chung chung; lỗi dai dẳng dừng worker và giữ partial checkpoint. Xem `responses.jsonl`, gồm stage `diagnosis` và `diagnosis_review`, trước khi sửa nguyên nhân và chạy lại. Không còn gọi `recover_task3_diagnosis.py` bên ngoài hoặc sửa source bằng thay chuỗi trong notebook.

Kiểm tra lại file submission cũ bằng heuristic cập nhật phát hiện 59/115 phản hồi Level 1 có nội dung cụ thể vượt mức. Đây là dấu hiệu vận hành, không phải kết luận của bộ chấm chính thức và không giải thích toàn bộ điểm 0.1654. Bản sửa có test tổng hợp cho thay thế nhãn sau review, giữ code/ràng buộc trong review, không rò rỉ reference vào query, Level 1 không lộ chẩn đoán, Level 3 viết lại khi thiếu hướng sửa, và dừng khi review không hợp lệ.

Máy sửa mã không có CUDA: chưa đo điểm leaderboard mới hoặc chạy đủ 328 mẫu bằng model. Stress test CUDA vẫn nằm trong notebook để chạy trên Kaggle. Nếu gặp OOM, giữ checkpoint/log; không giảm max_input_tokens để cắt code âm thầm. Không tự thực thi code sinh viên trong notebook.
