# Chạy Task 3 trên Kaggle

1. Import notebooks/run_task3_kaggle.ipynb vào Kaggle.
2. Add Input một dataset riêng tư chứa task3_kaggle_source.zip, thư mục sample_dataset đầy đủ (32 mẫu, code và exams.json), public_test.zip và task3_requested_levels*.json.
3. Bật Internet, chọn Accelerator: None. Thêm Secret OPENROUTER_API_KEY và cấp quyền cho notebook.
4. Chạy cell theo thứ tự. Nếu có nhiều bản dữ liệu, điền TRAIN_JSON, PUBLIC_JSON, LEVELS_JSON thủ công.
5. Kiểm tra MODEL và BASE_URL theo tài khoản API. Model cấu hình mặc định là google/gemma-4-31b-it:free qua OpenRouter; không đảm bảo nhà cung cấp còn phục vụ ID này.
6. Chạy thử 3 mẫu, xem feedback và responses.jsonl; sau đó chạy toàn bộ.
7. Tải /kaggle/working/submission_task3.zip để nộp.

ZIP mã nguồn chứa scripts, src, tasks và configs; không chứa trọng số model hoặc dữ liệu sinh viên. Notebook giải nén ZIP để dùng đúng phiên bản pipeline đã chỉnh, không phụ thuộc mã trên GitHub.

Pipeline gọi cùng model hai lần: chẩn đoán nhãn lỗi và nguyên nhân, sau đó viết feedback theo level với ví dụ RAG từ 32 mẫu train. Public test chỉ dùng suy luận. Notebook không fine-tune; phần fine-tune open-weight của đồ án vẫn cần thực hiện riêng.

Cache trong task3-v4/api_cache.jsonl tái sử dụng request khi chạy lại cùng cấu hình. Qua session mới cần tự lưu riêng tư và phục hồi cache. Tối thiểu 656 request cho 328 mẫu, chưa tính retry. Chi phí/quota tùy model và nhà cung cấp. Chưa đo điểm leaderboard mới.

Bằng chứng compile/test là tùy chọn qua --evidence và --tests. Khi không có, model phân tích tĩnh đề/code. Bộ kiểm tra level là heuristic, không bảo đảm đúng ngữ nghĩa. Không publish logs/cache/predictions chứa dữ liệu sinh viên.
